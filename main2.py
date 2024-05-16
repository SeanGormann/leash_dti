#Imports 
import pandas as pd
import polars as pl
import numpy as np
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import gc

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader as GeoDataLoader

from scipy.spatial.distance import pdist, squareform
print(f"GPUS available: {torch.cuda.device_count()}")


print("Imports Loaded In")



# Fix fastai bug to enable fp16 training with dictionaries
import torch
from fastai.vision.all import *

def flatten(o):
    "Concatenate all collections and items as a generator"
    for item in o:
        if isinstance(o, dict): yield o[item]; continue
        elif isinstance(item, str): yield item; continue
        try: yield from flatten(item)
        except TypeError: yield item

from torch.cuda.amp import GradScaler, autocast
@delegates(GradScaler)
class MixedPrecision(Callback):
    "Mixed precision training using Pytorch's `autocast` and `GradScaler`"
    order = 10
    def __init__(self, **kwargs): self.kwargs = kwargs
    def before_fit(self):
        self.autocast,self.learn.scaler,self.scales = autocast(),GradScaler(**self.kwargs),L()
    def before_batch(self): self.autocast.__enter__()
    def after_pred(self):
        if next(flatten(self.pred)).dtype==torch.float16: self.learn.pred = to_float(self.pred)
    def after_loss(self): self.autocast.__exit__(None, None, None)
    def before_backward(self): self.learn.loss_grad = self.scaler.scale(self.loss_grad)
    def before_step(self):
        "Use `self` as a fake optimizer. `self.skipped` will be set to True `after_step` if gradients overflow. "
        self.skipped=True
        self.scaler.step(self)
        if self.skipped: raise CancelStepException()
        self.scales.append(self.scaler.get_scale())
    def after_step(self): self.learn.scaler.update()

    @property
    def param_groups(self):
        "Pretend to be an optimizer for `GradScaler`"
        return self.opt.param_groups
    def step(self, *args, **kwargs):
        "Fake optimizer step to detect whether this batch was skipped from `GradScaler`"
        self.skipped=False
    def after_fit(self): self.autocast,self.learn.scaler,self.scales = None,None,None

import fastai
fastai.callback.fp16.MixedPrecision = MixedPrecision



building_blocks = 'data/all_buildingblock.pkl'
all_dtis = 'data/5mNegs_allp.csv' # 30mNegs_allp

bbs = pd.read_pickle(building_blocks)
all_dtis = pd.read_csv(all_dtis)

print("Data Loaded")


import numpy as np
import torch
from torch.utils.data import Dataset

class MultiGraphDataset(Dataset):
    def __init__(self, reaction_df, block_df, device='cpu', fold=0, nfolds=5, train=True, test=False, seed=2023):
        self.device = device
        self.reaction_df = reaction_df
        self.block_df = block_df

        # K-Fold Splitting
        kf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
        folds = list(kf.split(self.reaction_df))
        train_idx, eval_idx = folds[fold]
        self.reaction_df = self.reaction_df.iloc[train_idx if train else eval_idx]

        # Preload all graphs into a numpy array
        self.graphs = [self.prepare_graph(row['mol_graph']) for index, row in self.block_df.iterrows()]
        
        self.ids = self.reaction_df[['buildingblock1_id', 'buildingblock2_id', 'buildingblock3_id']].to_numpy().astype(int)
        y_data = self.reaction_df[['binds_BRD4', 'binds_HSA', 'binds_sEH']].to_numpy().astype(int)
        self.y = torch.tensor(y_data, dtype=torch.float, device=device)
        self.mode = 'train' if train else 'eval'

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Assuming each block_id indexes directly into self.graphs
        graph1 = self.graphs[self.ids[idx, 0]]
        graph2 = self.graphs[self.ids[idx, 1]]
        graph3 = self.graphs[self.ids[idx, 2]]
        y = self.y[idx]

        # Prepare a single batched graph if your downstream process expects that,
        # or handle multiple graphs properly according to your model's requirements.
        return {'mol_graphs': (graph1, graph2, graph3), 'y': y}

    def prepare_graph(self, graph):
        graph.to(self.device)
        return graph



import torch
from torch.nn import Linear, ReLU, Sequential, Dropout, BatchNorm1d
from torch_geometric.nn import GatedGraphConv, global_mean_pool
import torch.nn.functional as F

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
        self.batch_norm = BatchNorm1d(hidden_dim * 3)  # Size multiplied by 3 because of concatenation


        fc_dim = hidden_dim * 4

        # Fully connected layers
        self.fc1 = Linear(fc_dim, fc_dim * 4)
        self.fc2 = Linear(fc_dim * 4, fc_dim)
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
        x = self.fc3(x)

        return x

    def process_graph_component(self, x, edge_index, batch, conv_layer):
        x = F.relu(conv_layer(x, edge_index))
        x = global_mean_pool(x, batch)
        return x

print(f"GPUS available: {torch.cuda.device_count()}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


target = 'HSA'
#for fold in [0]: # running multiple folds at kaggle may cause OOM
ds_train = MultiGraphDataset(all_dtis, bbs, device = device, fold=0, nfolds=20, train=True, test=False)
dl_train = GeoDataLoader(ds_train, batch_size=512, shuffle=True)#, collate_fn=custom_collate_fn). 192 2048
#dl_train = DeviceDataLoader(dl_train, device)


ds_val = MultiGraphDataset(all_dtis, bbs, device = device, fold=0, nfolds=20, train=False, test=False)
dl_val= GeoDataLoader(ds_val, batch_size=512,  shuffle=True)#, collate_fn=custom_collate_fn)
#dl_val = DeviceDataLoader(dl_val, device)

gc.collect()
print("Data Prepped")


import torch
from torch import nn
from torch_geometric.data import DataLoader
from sklearn.metrics import average_precision_score
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


# Total number of epochs and warmup steps
n_epochs = 10
warmup_steps = 196
total_steps = len(ds_train) * n_epochs
#current_lr = 0.0001
initial_lr = 0.0001
do = 0.1


#model = MolGNN(num_node_features=8, num_edge_features=4, num_layers=4, hidden_dim=258, dropout=do, ff_multi=4, t_heads=6)
mol_node_features = 8
mol_edge_features = 4
nl = 3
hd = 320

model = MolGNN(mol_node_features, nl, hd)
model.to(device)
print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')


optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr) #lr=0.0001)
criterion = nn.BCEWithLogitsLoss()

weights = torch.tensor([1.0, 2.0])
# Use Binary Cross Entropy Loss for binary classification with class weights
criterion = nn.BCEWithLogitsLoss(pos_weight=weights[1])  #pos_weight=weights[1]
#criterion = DiceLoss()
#criterion = FocalLoss(alpha=0.5, gamma=2.0)

# Scheduler and scaler for AMP
scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs - 1, eta_min=0)
#scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
scaler = GradScaler()

protein_index = {'BRD4': 0, 'HSA': 1, 'sEH': 2}  # Default to 0th index if protein not found


def update_learning_rate(optimizer, epoch, batch_idx, total_steps, warmup_steps, initial_lr):
    """Update learning rate based on warmup and decay."""
    if epoch * len(ds_train) + batch_idx < warmup_steps:
        # Warm-up phase: linear increase
        lr = initial_lr * (epoch * len(ds_train) + batch_idx + 1) / warmup_steps
    else:
        # Exponential decay
        decay_rate = 0.95  # Decay rate per epoch after warmup
        lr = initial_lr * (decay_rate ** (epoch - warmup_steps / len(ds_train)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def process_batch(batch, train=True, protein_index=None):
    with autocast():
        outputs = model(batch)

        # Adjust reshaping based on your batch data structure and need
        if protein_index is not None:
            y_vals = batch['y'].view(-1, 3)[:, protein_index].view(-1, 1)  # Select the correct targets and reshape
        else:
            y_vals = batch['y'].view(-1, 3)  #

        y_vals = y_vals.float()

        #loss = criterion(outputs.squeeze(), y_vals)
        loss = criterion(outputs, y_vals)

    if train:
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return outputs.detach(), y_vals.detach(), loss.item()


model_save_path = 'models/model_1.pth'
best_map = 0.0
for epoch in range(n_epochs):
    start_time = time.time()
    model.train()
    total_loss = 0
    pbar = tqdm(enumerate(dl_train), total=len(dl_train), desc=f"Epoch {epoch + 1}/{n_epochs}")
    for i, batch in pbar:
        # Warm-up phase
        if epoch == 0 and i < warmup_steps:
            lr = initial_lr * (i + 1) / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = optimizer.param_groups[0]['lr']

        #_, _, loss = process_batch(batch.to(device), train=True)
        _, _, loss = process_batch(batch, train=True, protein_index=None)  #None protein_index['HSA']
        total_loss += loss
        pbar.set_postfix(loss=f"{total_loss / (i + 1):.4f}", lr=f"{lr:.6f}") #, do=f"{model.dropout:.4f}")

    # Step the scheduler only after warm-up is complete
    #if epoch >= 1:
    scheduler.step()
    #model.dropout = Dropout((do * (lr*10000)))

    # Validation and Metric Calculation
    model.eval()
    model_outputs = []
    true_labels = []
    vloss = 0
    with torch.no_grad():

        for batch in dl_val:
            #outputs, labels, val_loss = process_batch(batch.to(device), train=False)
            outputs, labels, val_loss = process_batch(batch, train=False, protein_index=None)
            model_outputs.append(outputs.squeeze().cpu())
            true_labels.append(labels.cpu())
            vloss += val_loss


    model_outputs = torch.cat(model_outputs)
    true_labels = torch.cat(true_labels)
    model_outputs = torch.sigmoid(model_outputs).numpy()  # Apply sigmoid to convert logits to probabilities

    #train_dataset.shuffle_files()
    # Calculate Mean Average Precision (micro)
    map_micro = average_precision_score(true_labels.numpy(), model_outputs, average='micro')
    if map_micro > best_map:
        best_map = map_micro
        torch.save(model.state_dict(), model_save_path)

    #b_map_micro = average_precision_score(b_true_labels.numpy(), b_model_outputs, average='micro')
    #scheduler.step(b_map_micro) #On plateau
    print(f"Epoch {epoch+1}, Train Loss: {(total_loss / len(dl_train)):.4f}, Val Loss: {(vloss/len(dl_val)):.4f},Val MAP (micro): {map_micro:.5f},Time: {(time.time() - start_time)/60:.2f}m\n----")
    #print(f"Epoch {epoch+1}, Train Loss: {(total_loss / len(dl_train)):.4f}, Blind MAP (micro): {b_map_micro:.5f} ,Time: {(time.time() - start_time)/60:.2f}m\n----")