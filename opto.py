import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import KFold
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch  # Assuming you are using PyTorch Geometric
from torch.nn import Linear, ReLU, Sequential, Dropout, BatchNorm1d
from torch_geometric.nn import GatedGraphConv, global_mean_pool
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from utils import *

class GraphDataset(Dataset):
    def __init__(self, flat_bbs, graph_dict, ys, fold=0, nfolds=5, train=True, seed=2023, blind=False):
        self.flat_bbs = flat_bbs
        self.graph_dict = graph_dict
        self.ys = ys
        
        # K-Fold Splitting
        if blind:
            self.flat_bbs = flat_bbs
            self.ys = ys
        else:
            kf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
            indices = np.arange(len(self.flat_bbs) // 3)
            folds = list(kf.split(indices))
            train_idx, val_idx = folds[fold]
            
            if train:
                selected_idx = train_idx
            else:
                selected_idx = val_idx
    
            selected_flat_bbs = []
            selected_ys = []
            for idx in selected_idx:
                start_idx = idx * 3
                selected_flat_bbs.extend(self.flat_bbs[start_idx:start_idx + 3])
                selected_ys.append(self.ys[idx])
    
            self.flat_bbs = np.array(selected_flat_bbs)
            self.ys = np.array(selected_ys)

    def __len__(self):
        return len(self.flat_bbs) // 3

    def __getitem__(self, idx):
        start_idx = idx * 3
        bb1 = self.graph_dict[self.flat_bbs[start_idx]]
        bb2 = self.graph_dict[self.flat_bbs[start_idx + 1]]
        bb3 = self.graph_dict[self.flat_bbs[start_idx + 2]]
        ys = self.ys[idx]
        
        return bb1, bb2, bb3, ys

class MolGNN2(torch.nn.Module):
    def __init__(self, num_node_features, num_layers=6, hidden_dim=96, bb_dims=(180, 180, 180)):
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

        fc_dim = bb_dims[0] + bb_dims[1] + bb_dims[2]
        self.batch_norm = BatchNorm1d(fc_dim)
        self.tanh = nn.Tanh()

        # Fully connected layers
        self.fc1 = Linear(fc_dim, fc_dim * 3)
        self.fc2 = Linear(fc_dim * 3, fc_dim * 3)
        self.fc25 = Linear(fc_dim * 3, fc_dim)
        self.fc3 = Linear(fc_dim, 3)  # Output layer

    def forward(self, batch_data):
        # Debugging: Print the device of the input batch_data
        #print(f"batch_data device: {batch_data[0].x.device}")

        x1, edge_index1, batch1 = batch_data[0].x, batch_data[0].edge_index, batch_data[0].batch
        x2, edge_index2, batch2 = batch_data[1].x, batch_data[1].edge_index, batch_data[1].batch
        x3, edge_index3, batch3 = batch_data[2].x, batch_data[2].edge_index, batch_data[2].batch

        x1 = self.process_graph_component(x1, edge_index1, batch1, self.gated_conv1)
        x2 = self.process_graph_component(x2, edge_index2, batch2, self.gated_conv2)
        x3 = self.process_graph_component(x3, edge_index3, batch3, self.gated_conv3)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.batch_norm(x)

        # Apply dropout and fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc25(x)))
        x = self.fc3(x)

        return x

    def process_graph_component(self, x, edge_index, batch, conv_layer):
        x = F.relu(conv_layer(x, edge_index))
        x = self.graph_dropout(x)
        x = global_mean_pool(x, batch)
        return x

def process_batch(batch, model, scaler, optimizer, criterion, train=True, protein_index=None):
    # Move batch to the appropriate device
    device = next(model.parameters()).device
    batch = [item.to(device) for item in batch]

    with autocast():
        outputs = model(batch)

        # Adjust reshaping based on your batch data structure and need
        if protein_index is not None:
            y_vals = batch[3].view(-1, 3)[:, protein_index].view(-1, 1)  # Select the correct targets and reshape
        else:
            y_vals = batch[3].view(-1, 3)

        y_vals = y_vals.float().to(outputs.device)  # Match the device of the model outputs

        # Ensure the target size matches the input size
        if outputs.size() != y_vals.size():
            raise ValueError(f"Target size ({y_vals.size()}) must be the same as input size ({outputs.size()})")

        loss = criterion(outputs, y_vals)

    if train:
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return outputs.detach(), y_vals.detach(), loss.item()

def train_model(model, dataloader, test_dataloader, num_epochs, targets, scaler, optimizer, criterion):
    # Print the device IDs used by DataParallel
    print(f"DataParallel is using devices: {model.device_ids}")
    best_map = 0.0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        total_loss = 0
        train_outputs, train_targets = [], []
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}") as pbar:
            for data in dataloader:
                outs, labs, loss = process_batch(data, model, scaler, optimizer, criterion, train=True, protein_index=None)
                total_loss += loss

                train_outputs.append(outs.cpu().numpy())  # Collect outputs for MAP calculation
                train_targets.append(labs.cpu().numpy())

                pbar.set_postfix(loss=epoch_loss / (pbar.n + 1))
                pbar.update()

                if pbar.n % 2000 == 0 and pbar.n != 0:
                    map_brd4, map_hsa, map_seh, average_map, true_positives_brd4, predicted_positives_brd4, \
                    true_positives_hsa, predicted_positives_hsa, true_positives_seh, predicted_positives_seh = calculate_individual_map(train_outputs, train_targets)
                    print(f"Epoch {epoch} - Partial Training Average MAP: {average_map:.4f}")
                    print(f"BRD4 - True Positives: {true_positives_brd4} - Predicted Positives: {predicted_positives_brd4} - MAP: {map_brd4:.4f}")
                    print(f"HSA - True Positives: {true_positives_hsa} - Predicted Positives: {predicted_positives_hsa} - MAP: {map_hsa:.4f}")
                    print(f"SEH - True Positives: {true_positives_seh} - Predicted Positives: {predicted_positives_seh} - MAP: {map_seh:.4f}")

        # Evaluate on test data after each epoch
        with torch.no_grad():
            model.eval()
            val_loss = 0
            val_outputs, val_targets = [], []
            for data in test_dataloader:
                outs, labs, loss = process_batch(data, model, scaler, optimizer, criterion, train=False, protein_index=None)
                val_loss += loss

                val_outputs.append(outs.cpu().numpy())  # Collect outputs for MAP calculation
                val_targets.append(labs.cpu().numpy())

        map_brd4, map_hsa, map_seh, average_map, true_positives_brd4, predicted_positives_brd4, \
        true_positives_hsa, predicted_positives_hsa, true_positives_seh, predicted_positives_seh = calculate_individual_map(val_outputs, val_targets, targets)
        print(f"Epoch {epoch} - Validation Average MAP: {average_map:.4f}")
        print(f"BRD4 - True Positives: {true_positives_brd4} - Predicted Positives: {predicted_positives_brd4} - MAP: {map_brd4:.4f}")
        print(f"HSA - True Positives: {true_positives_hsa} - Predicted Positives: {predicted_positives_hsa} - MAP: {map_hsa:.4f}")
        print(f"SEH - True Positives: {true_positives_seh} - Predicted Positives: {predicted_positives_seh} - MAP: {map_seh:.4f}")
        print(f"Train loss: {total_loss:.4f} - Validation loss: {val_loss:.4f}")
        
        if map_micro > average_map:
            best_map = map_micro
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist
            model_path = os.path.join(model_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.module.state_dict(), model_path)

    cleanup()


def custom_collate_fn(batch):
    graphs_array_1 = [item[0] for item in batch]
    graphs_array_2 = [item[1] for item in batch]
    graphs_array_3 = [item[2] for item in batch]
    ys_array = np.array([item[3] for item in batch])  # Convert list of numpy arrays to a single numpy array
    return Batch.from_data_list(graphs_array_1), Batch.from_data_list(graphs_array_2), Batch.from_data_list(graphs_array_3), torch.tensor(ys_array)


"""
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12353'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def init_process_group():
    dist.init_process_group(backend='nccl')
    
def cleanup():
    dist.destroy_process_group()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()

args = get_args()
torch.cuda.set_device(args.local_rank)
device = torch.device(f'cuda:{args.local_rank}')
"""

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main_worker(rank, world_size, num_epochs, initial_lr, batch_size, loaded_targets, flat_bbs, graph_dict):
    setup(rank, world_size)
    
    device = torch.device(f'cuda:{rank}')
    model = MolGNN2(8, 4, 96, bb_dims=(320, 320, 320)).to(device)
    model = DDP(model, device_ids=[rank])
    
    criterion = nn.BCEWithLogitsLoss().to(device)
    scaler = GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    
    dataset = GraphDataset(flat_bbs, graph_dict, loaded_targets, fold=0, nfolds=5, train=True, seed=2023, blind=True)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=8, pin_memory=True, collate_fn=custom_collate_fn)
    
    print(f"Starting training on rank {rank}")
    train_model(model, dataloader, dataloader, num_epochs, loaded_targets, scaler, optimizer, criterion) #second dataloader should be  validation
    
    cleanup()

def spawn_workers(world_size, num_epochs, initial_lr, batch_size, loaded_targets, flat_bbs, graph_dict):
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, num_epochs, initial_lr, batch_size, loaded_targets, flat_bbs, graph_dict))

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs")
    num_epochs = 10
    initial_lr = 0.0001
    batch_size = 2048
    
    # Load your dataset
    loaded_data = np.load('data/dataset.npz')
    loaded_buildingblock_ids, loaded_targets = loaded_data['buildingblock_ids'], loaded_data['targets']
    print(f'Loaded buildingblock_ids shape: {loaded_buildingblock_ids.shape}')
    
    graphs = pd.read_pickle('data/all_buildingblock.pkl')
    graph_dict = {row['id']: row['mol_graph'] for _, row in graphs.iterrows()}
    
    # Flatten the loaded_buildingblock_ids for batching
    flat_bbs = loaded_buildingblock_ids.flatten()
    
    spawn_workers(world_size, num_epochs, initial_lr, batch_size, loaded_targets, flat_bbs, graph_dict)


