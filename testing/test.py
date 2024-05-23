import pandas as pd
import numpy as np
import time

import numpy as np
import pandas as pd
import time
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader as GeoDataLoader

from torch_geometric.nn import GatedGraphConv, NNConv, global_mean_pool, GCNConv, GATConv, GATv2Conv
import torch
from torch.nn import Linear, ReLU, Sequential, Dropout, BatchNorm1d, Dropout
from torch_geometric.nn import GatedGraphConv, global_mean_pool, MessagePassing
import torch.nn.functional as F
import torch
from tqdm import tqdm
import torch.nn as nn
from models import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

all_dtis = '../data/test_bb.csv' #test_prot test_bb
building_blocks = '../data/all_buildingblock.pkl' #'/content/drive/MyDrive/all_buildingblock.pkl' '/content/drive/MyDrive/all_bbs_pca_8mixfeats.pkl' 

bbs = pd.read_pickle(building_blocks)
all_dtis = pd.read_csv(all_dtis)

class MolGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_layers=6, hidden_dim=96):
        super(MolGNN, self).__init__()
        # Parameters
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Define GatedGraphConv for each graph component
        self.gated_conv1 = GatedGraphConv(out_channels=hidden_dim, num_layers=num_layers)
        self.gated_conv2 = GatedGraphConv(out_channels=hidden_dim, num_layers=num_layers)
        self.gated_conv3 = GatedGraphConv(out_channels=hidden_dim, num_layers=num_layers)

        # Dropout and batch norm after pooling
        self.dropout = Dropout(0.1)
        self.batch_norm = BatchNorm1d(hidden_dim * 4)  # Size multiplied by 3 because of concatenation


        fc_dim = hidden_dim * 4

        # Fully connected layers
        self.fc1 = Linear(fc_dim, fc_dim * 3)
        self.fc2 = Linear(fc_dim * 3, fc_dim*3)
        self.fc25 = Linear(fc_dim * 3, fc_dim)
        self.fc3 = Linear(fc_dim, 3)  # Output layer

    def forward(self, batch_data):
        #print("batch_data", batch_data)
        batch_data = batch_data['mol_graphs']
        # Process each graph component through its GatedGraphConv, pooling, and batch norm
        x1, edge_index1, batch1 = batch_data[0].x, batch_data[0].edge_index, batch_data[0].batch
        x2, edge_index2, batch2 = batch_data[1].x, batch_data[1].edge_index, batch_data[1].batch
        x3, edge_index3, batch3 = batch_data[2].x, batch_data[2].edge_index, batch_data[2].batch

        x1 = self.process_graph_component(x1, edge_index1, batch1, self.gated_conv1)
        x2 = self.process_graph_component(x2, edge_index2, batch2, self.gated_conv2)
        x3 = self.process_graph_component(x3, edge_index3, batch3, self.gated_conv3)

        # Concatenate the outputs from the three graph components
        xx = x1 * x2 * x3
        x = torch.cat((x1, x2, x3, xx), dim=1) # xx
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
        x = global_mean_pool(x, batch)
        return x



from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset

class TestDataset(Dataset):
    def __init__(self, reaction_df, block_df, device='cpu', fold=0, nfolds=5, train=True, test=False, blind=False, seed=2023, max_len=584):
        self.device = device
        self.reaction_df = reaction_df
        self.block_df = block_df
        self.max_len = max_len

        # K-Fold Splitting
        # Preload all graphs into a list
        self.graphs = [self.prepare_graph(row['mol_graph']) for index, row in self.block_df.iterrows()]

        self.ids = self.reaction_df[['buildingblock1_id', 'buildingblock2_id', 'buildingblock3_id']].to_numpy().astype(int)
        self.reaction_df = self.reaction_df.fillna(-1)
        self.y = self.reaction_df[['BRD4', 'HSA', 'sEH']].to_numpy().astype(int)
        #self.y = self.reaction_df[['id']].to_numpy().astype(int)
        self.y = torch.tensor(self.y, dtype=torch.long, device=device)
        #self.protein_ids = self.reaction_df['protein'].to_numpy()
        self.mode = 'train' if train else 'eval'

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        graph1 = self.graphs[self.ids[idx, 0]]
        graph2 = self.graphs[self.ids[idx, 1]]
        graph3 = self.graphs[self.ids[idx, 2]]
        y = self.y[idx]
        #protein_seq = self.protein_dict[self.protein_ids[idx]].to(self.device)
        protein_seq = torch.tensor([1])
        #prot = self.protein_ids[idx]

        return {'mol_graphs': (graph1, graph2, graph3), 'y': y}

    def prepare_graph(self, graph):
        graph.to(self.device)
        return graph



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ds_train = TestDataset(all_dtis, bbs, device = device, fold=0, nfolds=20, train=False, test=True)
dl_test = GeoDataLoader(ds_train, batch_size=2048, shuffle=True) #, collate_fn=custom_collate_fn). 192 2048

print(f"Dataset: {len(ds_train)}, Dataloader: {len(dl_test)}")
print(f"Data: {next(iter(dl_test))}")

#model = Cerberus(num_node_features=8, num_layers=4, hidden_dim=180, bb_dims=(96, 240, 240), prot_dim=240)
model = MolGNN(num_node_features=8, num_layers=4, hidden_dim=180)
multi_model_path = 'models/multi_mod_1.pth'

model.load_state_dict(torch.load(multi_model_path))
model.to(device)
model.eval()

# Function to process and predict for all proteins
def predict_batch(batch):
    with torch.no_grad():
        outputs = model(batch)
        predictions = torch.sigmoid(outputs)  # Convert logits to probabilities for all proteins
    return predictions.cpu()  # Detach from CUDA and send to CPU

# Collect all predictions
all_predictions = []
all_ids = []  # Assuming we're storing IDs for filtering
start_time = time.time()

for batch in tqdm(dl_test, desc="Inference"):
    #graphs, ids = batch  # Assume the DataLoader provides both graphs and corresponding ids
    preds = predict_batch(batch)
    all_predictions.extend(preds.tolist())
    all_ids.extend(batch['y'].tolist())

print(f"Completed inference in {time.time() - start_time:.2f} seconds.")



final_predictions = []
final_ids = []

# Iterate over each set of predictions and corresponding IDs
for preds, ids in zip(all_predictions, all_ids):
    for pred, id in zip(preds, ids):
        if id != -1:  # Check if the ID is valid (not -1)
            final_predictions.append(pred)  # Add the valid prediction to the flat list
            final_ids.append(id)  # Add the valid ID to the flat list

# Now final_predictions and final_ids are flat lists containing only valid entries
print(len(final_predictions), len(final_ids))


threshold = 0.5
num_positive_hsa = np.sum(np.array(final_predictions) > threshold)
print(f"Number of positive predictions for final_preds after sigmoid: {num_positive_hsa}")


results = pd.DataFrame({
    'id': final_ids,
    'prediction': final_predictions

})


sample_submission = pd.read_csv('data/sample_submission.csv')

# Check the updated DataFrame
# Merge or update the sample_submission DataFrame
final_submission = sample_submission.merge(results, on='id', how='left')


# Drop the 'binds' column
final_submission = final_submission.drop(columns=['binds'])

# Rename 'prediction' to 'binds'
final_submission = final_submission.rename(columns={'prediction': 'binds'})

# Display the updated DataFrame
print(final_submission.shape)

final_submission.to_csv('submissions/final_predictions_v49.csv', index=False)
