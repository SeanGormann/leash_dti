import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch  # Assuming you are using PyTorch Geometric
from torch.nn import Linear, ReLU, Sequential, Dropout, BatchNorm1d
from torch_geometric.nn import GatedGraphConv, global_mean_pool
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm import tqdm
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

class GraphDataset(Dataset):
    def __init__(self, flat_bbs, graph_dict, ys):
        self.flat_bbs = flat_bbs
        self.graph_dict = graph_dict
        self.ys = ys

    def __len__(self):
        return len(self.flat_bbs) // 3

    def __getitem__(self, idx):
        start_idx = idx * 3
        bb1 = self.graph_dict[self.flat_bbs[start_idx]]
        bb2 = self.graph_dict[self.flat_bbs[start_idx + 1]]
        bb3 = self.graph_dict[self.flat_bbs[start_idx + 2]]
        ys = self.ys[idx]
        
        return bb1, bb2, bb3, ys

def custom_collate_fn(batch):
    graphs_array_1 = [item[0] for item in batch]
    graphs_array_2 = [item[1] for item in batch]
    graphs_array_3 = [item[2] for item in batch]
    ys_array = np.array([item[3] for item in batch])  # Convert list of numpy arrays to a single numpy array
    return Batch.from_data_list(graphs_array_1), Batch.from_data_list(graphs_array_2), Batch.from_data_list(graphs_array_3), torch.tensor(ys_array)

# Load your dataset
loaded_data = np.load('data/dataset.npz')
loaded_buildingblock_ids = loaded_data['buildingblock_ids']
loaded_targets = loaded_data['targets']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Loaded buildingblock_ids shape: {loaded_buildingblock_ids.shape}')

len_dataset = len(loaded_buildingblock_ids)
batch_size = 2048

graphs = pd.read_pickle('data/all_buildingblock.pkl')
graph_dict = {row['id']: row['mol_graph'] for _, row in graphs.iterrows()}

# Flatten the loaded_buildingblock_ids for batching
flat_bbs = loaded_buildingblock_ids.flatten()

# Create dataset and dataloader
dataset = GraphDataset(flat_bbs, graph_dict, loaded_targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6, collate_fn=custom_collate_fn, pin_memory=True)


class MolGNN2(torch.nn.Module):
    def __init__(self, num_node_features, num_layers=6, hidden_dim=96, bb_dims=(180, 180, 180), device = 'cuda'):
        super(MolGNN2, self).__init__()

        # Parameters
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Define GatedGraphConv for each graph component
        self.gated_conv1 = GatedGraphConv(out_channels=bb_dims[0], num_layers=num_layers)
        self.gated_conv2 = GatedGraphConv(out_channels=bb_dims[1], num_layers=num_layers)
        self.gated_conv3 = GatedGraphConv(out_channels=bb_dims[2], num_layers=num_layers)

        # Dropout and batch norm after pooling
        self.dropout = Dropout(0.1)
        self.graph_dropout = Dropout(0.1)


        fc_dim = bb_dims[0] + bb_dims[1]  + bb_dims[2] 
        self.batch_norm = BatchNorm1d(fc_dim) 
        self.tanh = nn.Tanh()

        # Fully connected layers
        self.fc1 = Linear(fc_dim, fc_dim * 3)
        self.fc2 = Linear(fc_dim * 3, fc_dim*3)
        self.fc25 = Linear(fc_dim * 3, fc_dim)
        self.fc3 = Linear(fc_dim, 3)  # Output layer

        self.device = device

    def forward(self, batch_data):
        #print("batch_data", batch_data)
        #batch_data = batch_data['mol_graphs']
        # Process each graph component through its GatedGraphConv, pooling, and batch norm
        x1, edge_index1, batch1 = batch_data[0].x.to(self.device), batch_data[0].edge_index.to(self.device), batch_data[0].batch.to(self.device)
        x2, edge_index2, batch2 = batch_data[1].x.to(self.device), batch_data[1].edge_index.to(self.device), batch_data[1].batch.to(self.device)
        x3, edge_index3, batch3 = batch_data[2].x.to(self.device), batch_data[2].edge_index.to(self.device), batch_data[2].batch.to(self.device)

        x1 = self.process_graph_component(x1, edge_index1, batch1, self.gated_conv1)
        x2 = self.process_graph_component(x2, edge_index2, batch2, self.gated_conv2)
        x3 = self.process_graph_component(x3, edge_index3, batch3, self.gated_conv3)

        # Concatenate the outputs from the three graph components
        #xx = x1 * x2 * x3
        x = torch.cat((x1, x2, x3), dim=1) # xx
        #x = self.tanh(torch.einsum('b d, b d , b d -> b d', x1, x2, x3))
        x = self.batch_norm(x)
        #xx = x1 * x2 * x3
        #x = torch.einsum('ij,ij,ij->ij', x1, x2, x3)

        # Apply dropout and fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc25(x)))
        x = self.fc3(x)

        return x

    def process_graph_component(self, x, edge_index, batch, conv_layer):
        x = F.relu(conv_layer(x, edge_index))
        x = self.graph_dropout(x) # self.graph_dropout(x) self.dropout(x)
        x = global_mean_pool(x, batch)
        return x

criterion = nn.BCEWithLogitsLoss()  #pos_weight=weights[1]
scaler = GradScaler()

def process_batch(batch, train=True, protein_index=None):
    with autocast():
        outputs = model(batch)

        # Adjust reshaping based on your batch data structure and need
        if protein_index is not None:
            y_vals = batch[3].view(-1, 3)[:, protein_index].view(-1, 1)  # Select the correct targets and reshape
        else:
            y_vals = batch[3].view(-1, 3)  #

        y_vals = y_vals.float().to(device)

        #loss = criterion(outputs.squeeze(), y_vals)
        loss = criterion(outputs, y_vals)

    if train:
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return outputs.detach(), y_vals.detach(), loss.item()


def train_model(model, dataloader, num_epochs, targets):
    #model = model.cuda()  # Move the model to GPU

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        total_loss = 0
        train_outputs, train_targets = [], []
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}") as pbar:
            for data in dataloader:
                #print(data)
                outs, labs, loss = process_batch(data, train=True, protein_index=None)
                total_loss += loss
                epoch_loss += 1
                if epoch_loss % 100 == 0:
                    print(loss)
        
                train_outputs.append(outs.cpu().numpy())  # Collect outputs for MAP calculation
                train_targets.append(labs.cpu().numpy())
                
                pbar.set_postfix(loss=epoch_loss / (pbar.n + 1))
                pbar.update()
                
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader)}")

if __name__ == '__main__':
    mp.set_start_method('spawn')

    num_epochs = 10
    initial_lr = 0.0001
    do = 0.1
    
    mol_node_features = 8 # 8 11
    mol_edge_features = 4

    model = MolGNN2(mol_node_features, 4, 96, bb_dims=(320, 320, 320), device=device)
    model.to(device)
    print(model)
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr) #lr=0.00

    # Train the model
    train_model(model, dataloader, num_epochs, loaded_targets)
