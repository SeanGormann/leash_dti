import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import KFold
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch  
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
from models import *
from test import *
from test import run_test

class GraphDataset(Dataset):
    def __init__(self, flat_bbs, graph_dict, ys):
        self.flat_bbs = flat_bbs
        self.graph_dict = graph_dict
        self.ys = ys

        # Count positive y values for brd4, hsa, and seh
        brd4_positive_count = np.sum(self.ys[:, 0] > 0)
        hsa_positive_count = np.sum(self.ys[:, 1] > 0)
        seh_positive_count = np.sum(self.ys[:, 2] > 0)
        
        print(f"Positive counts - brd4: {brd4_positive_count}, hsa: {hsa_positive_count}, seh: {seh_positive_count}")

    def __len__(self):
        return len(self.flat_bbs) // 3

    def __getitem__(self, idx):
        start_idx = idx * 3
        bb1 = self.graph_dict[self.flat_bbs[start_idx]]
        bb2 = self.graph_dict[self.flat_bbs[start_idx + 1]]
        bb3 = self.graph_dict[self.flat_bbs[start_idx + 2]]
        ys = self.ys[idx]
        
        return bb1, bb2, bb3, ys


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

        if outputs.size() != y_vals.size():
            raise ValueError(f"Target size ({y_vals.size()}) must be the same as input size ({outputs.size()})")

        loss = criterion(outputs, y_vals)

    return outputs.detach(), y_vals.detach(), loss


def train_model(model, dataloader, test_dataloader, num_epochs, scaler, optimizer, criterion):
    # Print the device IDs used by DataParallel
    model_filename = "attentive_fp.pth"
    print(f"DataParallel is using devices: {model.device_ids}")
    scheduler = CosineAnnealingLR(optimizer, T_max=10 - 1, eta_min=0)
    best_map, average_map = 0.0, 0.0
    accumulation_steps = 4
    for epoch in range(num_epochs):
        dataloader.sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0
        total_loss = 0
        train_outputs, train_targets = [], []
        lr = optimizer.param_groups[0]['lr']
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}") as pbar:
            for data in dataloader:
                outs, labs, loss = process_batch(data, model, scaler, optimizer, criterion, train=True, protein_index=None)
                
                # Normalize loss to account for gradient accumulation
                loss = loss / accumulation_steps
                scaler.scale(loss).backward()
        
                if (pbar.n + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
        
                total_loss += loss.item() * accumulation_steps  

                train_outputs.append(outs.cpu().numpy())  # Collect outputs for MAP calculation
                train_targets.append(labs.cpu().numpy())

                if pbar.n % 4000 == 0 and pbar.n != 0:
                    map_brd4, map_hsa, map_seh, average_map, true_positives_brd4, predicted_positives_brd4, \
                    true_positives_hsa, predicted_positives_hsa, true_positives_seh, predicted_positives_seh = calculate_individual_map(train_outputs, train_targets)
                    print(f"Epoch {epoch} - Partial Training Average MAP: {average_map:.4f}")
                    print(f"BRD4 - True Positives: {true_positives_brd4} - Predicted Positives: {predicted_positives_brd4} - MAP: {map_brd4:.4f}")
                    print(f"HSA - True Positives: {true_positives_hsa} - Predicted Positives: {predicted_positives_hsa} - MAP: {map_hsa:.4f}")
                    print(f"SEH - True Positives: {true_positives_seh} - Predicted Positives: {predicted_positives_seh} - MAP: {map_seh:.4f}")
                
                pbar.set_postfix(loss=f"{total_loss / (pbar.n  + 1):.4f}", lr=f"{lr:.6f}", train_map=f"{average_map:.4f}")
                pbar.update()

        if len(dataloader) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    
        scheduler.step()

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
            
            val_outputs = np.vstack(val_outputs)
            val_targets = np.vstack(val_targets)
        map_brd4, map_hsa, map_seh, average_map, true_positives_brd4, predicted_positives_brd4, \
        true_positives_hsa, predicted_positives_hsa, true_positives_seh, predicted_positives_seh = calculate_individual_map(val_outputs, val_targets)
        print(f"Epoch {epoch} - Validation Average MAP: {average_map:.4f}")
        print(f"BRD4 - True Positives: {true_positives_brd4} - Predicted Positives: {predicted_positives_brd4} - MAP: {map_brd4:.4f}")
        print(f"HSA - True Positives: {true_positives_hsa} - Predicted Positives: {predicted_positives_hsa} - MAP: {map_hsa:.4f}")
        print(f"SEH - True Positives: {true_positives_seh} - Predicted Positives: {predicted_positives_seh} - MAP: {map_seh:.4f}")
        print(f"Train loss: {total_loss:.4f} - Validation loss: {val_loss:.4f}")

        if average_map > best_map:
            best_map = average_map
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist
            model_path = os.path.join(model_dir, model_filename)
            torch.save(model.module.state_dict(), model_path)
            print(f"Model saved to {model_path} with MAP: {best_map:.4f}")
    run_test(model_path)

    cleanup()


def custom_collate_fn(batch):
    graphs_array_1 = [item[0] for item in batch]
    graphs_array_2 = [item[1] for item in batch]
    graphs_array_3 = [item[2] for item in batch]
    ys_array = np.array([item[3] for item in batch])  # Convert list of numpy arrays to a single numpy array
    return Batch.from_data_list(graphs_array_1), Batch.from_data_list(graphs_array_2), Batch.from_data_list(graphs_array_3), torch.tensor(ys_array)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def spawn_workers(world_size, num_epochs, initial_lr, batch_size, train_data, val_data):
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, num_epochs, initial_lr, batch_size, train_data, val_data))


def main_worker(rank, world_size, num_epochs, initial_lr, batch_size, train_data, val_data):
    setup(rank, world_size)
    
    device = torch.device(f'cuda:{rank}')
    model = Net().to(device)
    model = DDP(model, device_ids=[rank])
    
    criterion = nn.BCEWithLogitsLoss().to(device)
    scaler = GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, betas=(0.9, 0.95), eps=1e-8)

    train_dataset = GraphDataset(train_data['flat_bbs'], train_data['graph_dict'], train_data['ys'])
    val_dataset = GraphDataset(val_data['flat_bbs'], val_data['graph_dict'], val_data['ys'])
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8, pin_memory=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=8, pin_memory=True, collate_fn=custom_collate_fn)
    
    print(f"Starting training on rank {rank}")
    train_model(model, train_loader, val_loader, num_epochs, scaler, optimizer, criterion)
    
    cleanup()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs")
    num_epochs = 100
    initial_lr = 0.001
    batch_size = 2048
    
    # Load your dataset
    loaded_data = np.load('data/10m_data.npz')
    loaded_buildingblock_ids, loaded_targets = loaded_data['buildingblocks'], loaded_data['targets']
    print(f'Loaded buildingblock_ids shape: {loaded_buildingblock_ids.shape}')
    
    graphs = pd.read_pickle('data/bbs_edge_and_eigens.pkl')
    graph_dict = {row['id']: row['mol_graph'] for _, row in graphs.iterrows()}
    
    # Flatten the loaded_buildingblock_ids for batching
    flat_bbs = loaded_buildingblock_ids.flatten()

    print("Getting data slits...")
    train_data, val_data = pre_split_data(flat_bbs, graph_dict, loaded_targets, fold=0, nfolds=5, testing=False)
    
    spawn_workers(world_size, num_epochs, initial_lr, batch_size, train_data, val_data)
