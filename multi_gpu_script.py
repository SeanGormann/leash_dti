from fastai.vision.all import *
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from torch import nn

from torch.cuda.amp import GradScaler, autocast
from torch.distributed import init_process_group, destroy_process_group
from torch.nn import Linear, ReLU, Sequential, Dropout, BatchNorm1d
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader

from torch_geometric.nn import GatedGraphConv, global_mean_pool
from tqdm import tqdm
import argparse
import gc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import polars as pl
import time
import torch

import torch.multiprocessing as mp

import torch.nn.functional as F


from scipy.spatial.distance import pdist, squareform


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"GPUS available: {torch.cuda.device_count()}")

print("Imports Loaded")

BUILDING_BLOCKS_PATH ='data/all_buildingblock.pkl'
DATA_PATH = 'data/5mNegs_allp.csv' # 30mNegs_allp

#BUILDING_BLOCKS_PATH = '/kaggle/input/dna-protien/all_buildingblock.pkl'
#DATA_PATH = '/kaggle/input/dna-protien/5mNegs_allp.csv' 

bbs = pd.read_pickle(BUILDING_BLOCKS_PATH)
all_dtis = pd.read_csv(DATA_PATH)

print("Data Loaded")

def flatten(o):
    "Concatenate all collections and items as a generator"
    for item in o:
        if isinstance(o, dict): yield o[item]; continue
        elif isinstance(item, str): yield item; continue
        try: yield from flatten(item)
        except TypeError: yield item

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



target = 'HSA'
#for fold in [0]: # running multiple folds at kaggle may cause OOM
ds_train = MultiGraphDataset(all_dtis, bbs, device = device, fold=0, nfolds=20, train=True, test=False)
dl_train = GeoDataLoader(ds_train, batch_size=512, shuffle=True)

ds_val = MultiGraphDataset(all_dtis, bbs, device = device, fold=0, nfolds=20, train=False, test=False)
dl_val= GeoDataLoader(ds_val, batch_size=512,  shuffle=True)

gc.collect()
print("Data Prepped")


# Total number of epochs and warmup steps
n_epochs = 10
warmup_steps = 196
total_steps = len(ds_train) * n_epochs
initial_lr = 0.0001
do = 0.1

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
criterion = nn.BCEWithLogitsLoss(pos_weight=weights[1]) 

# Scheduler and scaler for AMP
scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs - 1, eta_min=0)
scaler = GradScaler()

protein_index = {'BRD4': 0, 'HSA': 1, 'sEH': 2}  # Default to 0th index if protein not found


model_save_path = 'models/model_1.pth'
best_map = 0.0


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int, 
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _run_batch(self, batch, train=True, protein_index=None):
        with autocast():
            outputs = model(batch)

            # Adjust reshaping based on your batch data structure and need
            if protein_index is not None:
                y_vals = batch['y'].view(-1, 3)[:, protein_index].view(-1, 1)  # Select the correct targets and reshape
            else:
                y_vals = batch['y'].view(-1, 3)  #

            y_vals = y_vals.float()

            loss = criterion(outputs, y_vals)

        if train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        return outputs.detach(), y_vals.detach(), loss.item()

    def _run_epoch(self, epoch):
        start_time = time.time()
        self.model.train()
        total_loss = 0
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Steps: {len(self.train_data)}")
        pbar = tqdm(enumerate(dl_train), total=len(dl_train), desc=f"Epoch {epoch + 1}/{n_epochs}")
        for i, batch in pbar:
            # Warm-up phase
            if epoch == 0 and i < warmup_steps:
                lr = initial_lr * (i + 1) / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr = optimizer.param_groups[0]['lr']
            _, _, loss = self._run_batch(batch, train=True, protein_index=None)  
            total_loss += loss
            pbar.set_postfix(loss=f"{total_loss / (i + 1):.4f}", lr=f"{lr:.6f}") 

        # Step the scheduler only after warm-up is complete
        #if epoch >= 1:
        scheduler.step()

        # Validation and Metric Calculation
        self.model.eval()
        model_outputs = []
        true_labels = []
        vloss = 0
        with torch.no_grad():
            for batch in dl_val:
                outputs, labels, val_loss = self._run_batch(batch, train=False, protein_index=None)
                model_outputs.append(outputs.squeeze().cpu())
                true_labels.append(labels.cpu())
                vloss += val_loss

        model_outputs = torch.cat(model_outputs)
        true_labels = torch.cat(true_labels)
        model_outputs = torch.sigmoid(model_outputs).numpy()  # Apply sigmoid to convert logits to probabilities

        # Calculate Mean Average Precision (micro)
        map_micro = average_precision_score(true_labels.numpy(), model_outputs, average='micro')
        if map_micro > best_map:
            best_map = map_micro
            torch.save(model.state_dict(), model_save_path)
        print(f"Epoch {epoch+1}, Train Loss: {(total_loss / len(dl_train)):.4f}, Val Loss: {(vloss/len(dl_val)):.4f},Val MAP (micro): {map_micro:.5f},Time: {(time.time() - start_time)/60:.2f}m\n----")
        

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

def prepare_dataloader(dataset, batch_size):
    return GeoDataLoader(dataset, batch_size, sampler=DistributedSampler(dataset))


def get_dataset():
    ds_train = MultiGraphDataset(all_dtis, bbs, device = device, fold=0, nfolds=20, train=True, test=False)
    dl_train = GeoDataLoader(ds_train, batch_size=512, shuffle=True)
    return dl_train

def get_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    mol_node_features = 8
    nl = 3
    hd = 320
    model = MolGNN(mol_node_features, nl, hd)
    model.to(device)
    return model 

def get_optimizer(model_params):
    initial_lr = 0.0001
    torch.optim.Adam(model_params, lr=initial_lr)
    return optimizer

def load_train_objs():
    train_set = get_dataset()  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def main_function(rank, world_size, save_every, total_epochs, batch_size):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()
    

if __name__ == "__main__":
    batch_size = 512 
    total_epochs = 10
    save_every = 10
    world_size = torch.cuda.device_count()  # shorthand for cuda:0
    mp.spawn(main_function, args=(world_size, save_every, total_epochs, batch_size), nprocs=world_size)

