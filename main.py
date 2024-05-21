
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
import __main__

import torch.multiprocessing as mp

import torch.nn.functional as F

from scipy.spatial.distance import pdist, squareform

#BUILDING_BLOCKS_PATH ='data/all_buildingblock.pkl'
#DATA_PATH = 'data/5mNegs_allp.csv' # 30mNegs_allp

BUILDING_BLOCKS_PATH = 'data/all_buildingblock.pkl'
DATA_PATH = 'data/5mNegs_allp.csv' 

BUILDING_BLOCKS_PATH = 'data/all_buildingblock.pkl'
DATA_PATH = 'data/5mNegs_allp.csv' 


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
    def __init__(self, reaction_df, block_df, device, fold=0, nfolds=5, train=True, test=False, seed=2023):
        self.reaction_df = reaction_df.dropna()
        self.block_df = block_df
        self.device = device  # Device is now explicitly passed to the constructor
        print(f"Dataset Device: {self.device}, Len of df: {len(self.reaction_df)}")

        kf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
        folds = list(kf.split(self.reaction_df))
        train_idx, eval_idx = folds[fold]
        self.reaction_df = self.reaction_df.iloc[train_idx if train else eval_idx]

        self.graphs = [self.prepare_graph(row['mol_graph']) for index, row in self.block_df.iterrows()]
        self.ids = self.reaction_df[['buildingblock1_id', 'buildingblock2_id', 'buildingblock3_id']].to_numpy()
        y_data = self.reaction_df[['binds_BRD4', 'binds_HSA', 'binds_sEH']].to_numpy().astype(int)
        self.y = torch.tensor(y_data, dtype=torch.float, device=device)
        self.mode = 'train' if train else 'eval'

    def prepare_graph(self, graph):
        return graph.to(self.device) 

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        #Assuming each block_id indexes directly into self.graphs
        idx1 = self.ids[idx, 0]
        graph1 = self.graphs[idx1]
        idx2 = self.ids[idx, 0]
        graph2 = self.graphs[idx2]
        idx3 = self.ids[idx, 0]
        graph3 = self.graphs[idx3]
        y = self.y[idx]

        # Prepare a single batched graph if your downstream process expects that,
        # or handle multiple graphs properly according to your model's requirements.
        return {'mol_graphs': (graph1, graph2, graph3), 'y': y}


class MolGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_layers=6, hidden_dim=96, device='cpu'):
        super(MolGNN, self).__init__()

        # Parameters
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.device = device

        # Define GatedGraphConv for each graph component
        self.gated_conv1 = GatedGraphConv(out_channels=hidden_dim, num_layers=num_layers)
        self.gated_conv2 = GatedGraphConv(out_channels=hidden_dim, num_layers=num_layers)
        self.gated_conv3 = GatedGraphConv(out_channels=hidden_dim, num_layers=num_layers)

        # Dropout and batch norm after pooling
        self.dropout = Dropout(0.1)
        self.batch_norm = BatchNorm1d(hidden_dim * 3)  # Size multiplied by 3 because of concatenation


        fc_dim = hidden_dim * 4

        # Fully connected layers
        self.fc1 = Linear(fc_dim, fc_dim * 3)
        self.fc2 = Linear(fc_dim * 3, fc_dim*3)
        self.fc25 = Linear(fc_dim * 3, fc_dim)
        self.fc3 = Linear(fc_dim, 3) # 1 2 3   # Output layer

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


def ddp_setup(rank, world_size, port="12356"):
    """
      Args:
          rank: Unique identifier of each process
          world_size: Total number of processes
    """
    print("Setting Up DDP")
    try: 
      os.environ["MASTER_ADDR"] = "localhost"
      os.environ['MASTER_PORT'] = port 
      init_process_group(backend="nccl", rank=rank, world_size=world_size)
      print("Process Group Initialised, Setting Device Rank")
      torch.cuda.set_device(rank)
      print("DDP Initialised")
    except Exception as e:
      print(f"Error initalizing DDP: {e}")

class Trainer:
    def __init__(self, model, train_data, val_data, optimizer, scheduler, criterion, scaler, gpu_id, save_every, device):
        self.gpu_id = gpu_id
        print(f"GID: {gpu_id}, device: {device}")
        self.model = DDP(model.to(gpu_id), device_ids=[gpu_id], find_unused_parameters=True)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.scaler = scaler
        self.save_every = save_every
        self.best_map = 0.0
        self.average_map = 0.0
        self.model_save_path = 'model_1.pth'

    def _run_batch(self, batch, train=True):
        protein_index = {'BRD4': 0, 'HSA': 1, 'sEH': 2}
        with torch.cuda.amp.autocast():
            outputs = self.model(batch)
            #y_vals = batch['y'].view(-1, 3)[:, protein_index['HSA']].view(-1, 1)
            y_vals = batch['y'].view(-1, 3).float()
            loss = self.criterion(outputs, y_vals)

        if train:
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return outputs.detach(), y_vals.detach(), loss.item()

    def calculate_individual_map(self, y_true, y_scores):
        """
        Calculate the mean average precision (MAP) for each protein individually and return their average.
    
        Args:
        y_true (np.array): True labels reshaped to (-1, 3) where each column is a protein.
        y_scores (np.array): Predicted scores reshaped to (-1, 3) where each column is a protein.
    
        Returns:
        tuple: A tuple containing individual MAPs for BRD4, HSA, sEH, and the average MAP.
        """
        # Ensure y_true and y_scores are reshaped correctly
        y_true = y_true.reshape(-1, 3)
        y_scores = y_scores.reshape(-1, 3)
    
        # Calculate MAP for each column (protein)
        map_brd4 = average_precision_score(y_true[:, 0], y_scores[:, 0])
        map_hsa = average_precision_score(y_true[:, 1], y_scores[:, 1])
        map_seh = average_precision_score(y_true[:, 2], y_scores[:, 2])
    
        # Calculate the average MAP across all proteins
        average_map = np.mean([map_brd4, map_hsa, map_seh])
    
        return map_brd4, map_hsa, map_seh, average_map

    def _run_epoch(self, epoch, n_epochs):
        start_time = time.time()
        self.model.train()
        self.train_data.sampler.set_epoch(epoch)
        total_loss = 0
        train_outputs, train_targets = [], []
        pbar = tqdm(enumerate(self.train_data), total=len(self.train_data), desc=f"Epoch {epoch + 1}/{n_epochs}")
        for i, batch in pbar:
            # Warm-up phase
            if epoch == 0 and i < 192:
                lr = 0.001 * (i + 1) / 192
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr = self.optimizer.param_groups[0]['lr'] 

            # Process the batch
            outs, labs, loss = self._run_batch(batch, train=True)
            total_loss += loss
    
            train_outputs.append(outs.cpu().numpy())  # Collect outputs for MAP calculation
            train_targets.append(labs.cpu().numpy())
    
            # Optionally calculate MAP every 2500 steps to reduce computation
            if i % 3400 == 0 and i != 0:
                # Flatten the lists and then calculate the MAP for each protein and the average
                flat_outputs = np.vstack(train_outputs)
                flat_targets = np.vstack(train_targets)
                map_brd4, map_hsa, map_seh, average_map = self.calculate_individual_map(flat_targets, flat_outputs)
                self.average_map = average_map
                pbar.set_postfix(loss=f"{total_loss / (i + 1):.4f}", lr=f"{lr:.6f}", 
                                 train_map=f"{self.average_map:.4f}", map_brd4=f"{map_brd4:.4f}", 
                                 map_hsa=f"{map_hsa:.4f}", map_seh=f"{map_seh:.4f}")
                print(f"Train MAP: Average: {average_map:.4f}, BRD4: {map_brd4:.4f}, HSA: {map_hsa:.4f}, sEH: {map_seh:.4f}")
            else:
                pbar.set_postfix(loss=f"{total_loss / (i + 1):.4f}", lr=f"{lr:.6f}", train_map=f"{self.average_map:.4f}")

        self.scheduler.step()

        self.model.eval()
        model_outputs = []
        true_labels = []
        vloss = 0
        with torch.no_grad():
            for batch in self.val_data:
                outputs, labels, val_loss = self._run_batch(batch, train=False)
                model_outputs.append(outputs.squeeze().cpu())
                true_labels.append(labels.cpu())
                vloss += val_loss

        model_outputs = torch.cat(model_outputs)
        true_labels = torch.cat(true_labels)
        map_micro = average_precision_score(true_labels.numpy(), model_outputs.numpy(), average='micro')

        if map_micro > self.best_map:
            self.best_map = map_micro
            torch.save(self.model.module.state_dict(), self.model_save_path)
        
        print(f"Epoch {epoch+1}, Train Loss: {(total_loss / len(self.train_data)):.4f}, Val Loss: {(vloss / len(self.val_data)):.4f}, Val MAP (micro): {map_micro:.5f}, Time: {(time.time() - start_time)/60:.2f}m\n----")

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = f"checkpoint_{epoch}.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs):
        for epoch in range(max_epochs):
            self._run_epoch(epoch, max_epochs)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

    def destroy_process_group(self):
        torch.distributed.destroy_process_group()
    

"""
def prepare_dataloader(dataset, batch_size):
    return GeoDataLoader(dataset, batch_size, shuffle=False, sampler=DistributedSampler(dataset))
"""

def prepare_dataloader(dataset, batch_size, rank, world_size):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    return GeoDataLoader(dataset, batch_size=batch_size, sampler=sampler)

def get_dataset(device, fold=0, nfolds=20, train=True, test=False):
    bbs = pd.read_pickle(BUILDING_BLOCKS_PATH)
    all_dtis = pd.read_csv(DATA_PATH)
    
    #Oversampling
    y_data = all_dtis[['binds_BRD4', 'binds_HSA', 'binds_sEH']].to_numpy().astype(int)
    brd4_positive_indices = np.where(y_data[:, 0] == 1)[0]
    hsa_positive_indices = np.where(y_data[:, 1] == 1)[0]
    brd4_oversampled_rows = all_dtis.iloc[brd4_positive_indices].copy()
    hsa_oversampled_rows = all_dtis.iloc[hsa_positive_indices].copy()
    
    all_dtis = pd.concat([all_dtis, brd4_oversampled_rows, hsa_oversampled_rows])
    all_dtis = all_dtis.sample(frac=1).reset_index(drop=True) # Shuffle the dataframe
    
    print("Data Loaded")
    ds_train = MultiGraphDataset(all_dtis, bbs, device=device, fold=fold, nfolds=nfolds, train=train, test=test)
    return ds_train

def get_model(device):
    mol_node_features = 8
    nl = 6
    hd = 180
    model = MolGNN(mol_node_features, nl, hd, device)
    model.to(device)
    print(f"Model on device: {device}")
    return model

def get_optimizer(model_params):
    initial_lr = 0.001
    optimizer = torch.optim.Adam(model_params, lr=initial_lr)
    return optimizer

def load_train_objs(device, total_epochs):
    train_set = get_dataset(device)
    model = get_model(device)
    optimizer = get_optimizer(model.parameters())
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - 1, eta_min=0)
    scaler = GradScaler()
    criterion = nn.BCEWithLogitsLoss()
    return train_set, model, optimizer, scheduler, scaler, criterion

def main_function(rank, world_size, save_every, total_epochs, batch_size):
    ddp_setup(rank, world_size, "12351")
    device = torch.device(f'cuda:{rank}')
    dataset, model, optimizer, scheduler, scaler, criterion = load_train_objs(device, total_epochs)
    val_data = get_dataset(device, fold=0, nfolds=20, train=False, test=False)
    val_data = prepare_dataloader(val_data, batch_size, rank, world_size)
    train_data = prepare_dataloader(dataset, batch_size, rank, world_size)
    trainer = Trainer(model, train_data, val_data, optimizer, scheduler, criterion, scaler, rank, save_every, device)
    trainer.train(total_epochs)
    destroy_process_group()
    


if __name__ == "__main__":
    print("Running")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"GPUS available: {torch.cuda.device_count()}")
    batch_size = 512
    total_epochs = 10
    save_every = 10
    world_size = torch.cuda.device_count()
    mp.spawn(main_function, args=(world_size, save_every, total_epochs, batch_size), nprocs=world_size)
