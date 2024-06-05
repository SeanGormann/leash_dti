
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import KFold
import argparse

import numpy as np
import os
import gc
from sklearn.model_selection import KFold

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import Linear, ReLU, Sequential, Dropout, BatchNorm1d
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Subset

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler


from utils import *
from models import *


class MoleculeDataset(Dataset):
    def __init__(self, inputs, labels, train=True, device='cpu'):
        """
        Initializes the dataset by loading all `.npz` files from the specified directory,
        and storing the arrays as large concatenated numpy arrays.

        Args:
        data_dir (str): The directory containing `.npz` files with token and label data.
        filenames (list of str): List of filenames within the data_dir to be used.
        """
        self.inputs = inputs
        self.labels = labels

        self.inputs = np.memmap(inputs, dtype=np.uint8, mode='r', shape=(98415610, 130))
        self.labels = np.memmap(labels, dtype=np.uint8, mode='r', shape=(98415610, 3))

        print(f"Total samples loaded: {len(self.inputs)}")

        # Count positive y values for brd4, hsa, and seh
        brd4_positive_count = np.sum(self.labels[:, 0] > 0)
        hsa_positive_count = np.sum(self.labels[:, 1] > 0)
        seh_positive_count = np.sum(self.labels[:, 2] > 0)
        
        print(f"Positive counts - brd4: {brd4_positive_count}, hsa: {hsa_positive_count}, seh: {seh_positive_count}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Fetches the item at the provided index.

        Args:
        idx (int): The index of the item.

        Returns:
        tuple: A tuple containing input IDs and labels, converted to tensors.
        """
        # Convert specific index arrays to tensors on demand
        #tokens = torch.tensor(self.inputs[idx], dtype=torch.long)
        #labels = torch.tensor(self.labels[idx], dtype=torch.float)
        return self.inputs[idx], self.labels[idx]
        #return tokens, labels




def prepare_kfold_splits(num_files, k=30):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    indices = np.arange(num_files)
    splits = list(kf.split(indices))
    return splits




def custom_collate(batch):
    # Unzip the batch
    inputs, labels = zip(*batch)
    
    # Convert inputs to a stacked numpy array before converting to tensor
    inputs_np = np.array(inputs)
    labels_np = np.array(labels)

    # Convert numpy arrays to tensors
    inputs_tensor = torch.from_numpy(inputs_np).to(torch.long)
    labels_tensor = torch.from_numpy(labels_np).to(torch.float)

    return inputs_tensor, labels_tensor


"""
def custom_collate(batch):
    inputs = batch[0]
    labels = batch[1]

    inputs_np = np.stack(inputs, axis=0)
    labels_np = np.stack(labels, axis=0)

    inputs_tensor = torch.from_numpy(inputs_np).to(torch.long)
    labels_tensor = torch.from_numpy(labels_np).to(torch.float)

    print(f"Input Tensor Type in collate: {inputs_tensor.dtype}")  # This should print torch.int64 or torch.long

    return inputs_tensor, labels_tensor
"""



def pre_split_data(tokens, ys, fold=0, nfolds=5, seed=2023, testing=False):
    """
    Splits the dataset into training and validation sets.

    Args:
    tokens (np.ndarray): The input tokens.
    ys (np.ndarray): Corresponding target labels.
    fold (int): The current fold index for the split.
    nfolds (int): Total number of folds for the K-Fold splitting.
    seed (int): Random seed for reproducibility.
    testing (bool): Flag to indicate if this is for testing (splits a smaller subset).

    Returns:
    tuple: Two tuples containing the training and validation data respectively.
    """
    if testing:
        subset_size = len(tokens) // 4 # // 150  # Adjust as necessary for smaller test subset
        indices = np.arange(subset_size)
        np.random.seed(seed)
        np.random.shuffle(indices)

        return (tokens[indices], ys[indices]), (tokens[indices], ys[indices])

    else:
        # Perform K-Fold Splitting
        kf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
        indices = np.arange(len(tokens))
        for idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
            if idx == fold:
                return (tokens[train_idx], ys[train_idx]), (tokens[val_idx], ys[val_idx])




def process_batch(batch, model, scaler, optimizer, criterion, train=True, protein_index=None):
    # Move batch to the appropriate device
    device = next(model.parameters()).device
    #batch = [item.to(device) for item in batch]
    inputs, labels = batch
    inputs = inputs.to(model.device)
    labels = labels.to(model.device)

    with autocast():
        outputs = model(inputs)

        # Adjust reshaping based on your batch data structure and need
        if protein_index is not None:
            y_vals = labels.view(-1, 3)[:, protein_index].view(-1, 1)  # Select the correct targets and reshape
        else:
            y_vals = labels.view(-1, 3)

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



from models import *


class HyenaOperator(nn.Module):
    def __init__(
            self,
            d_model,
            l_max,
            d_out=None,
            order=2, 
            filter_order=64,
            dropout=0.0,  
            filter_dropout=0.0, 
            **filter_args,
        ):

        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.order = order
        inner_width = d_model * (order + 1)
        self.dropout = nn.Dropout(dropout)
        self.in_proj = nn.Linear(d_model, inner_width)
        self.d_out = d_out if d_out else d_model
        print(f"Output dimension: {self.d_out}")
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.short_filter = nn.Conv1d(
            inner_width, 
            inner_width, 
            3,
            padding=2,
            groups=inner_width
        )
        self.filter_fn = HyenaFilter(
            d_model * (order - 1), 
            order=filter_order, 
            seq_len=l_max,
            channels=1, 
            dropout=filter_dropout, 
            **filter_args
        ) 

    def forward(self, u):
        #print(f"Input shape: {u.shape}")
        #u = u.permute(0, 2, 1)
        u = self.in_proj(u)
        #print(f"After in_proj shape: {u.shape}")
        u = rearrange(u, 'b l d -> b d l')
        
        uc = self.short_filter(u)[..., :self.l_max] 
        *x, v = uc.split(self.d_model, dim=1)
        
        k = self.filter_fn.filter(self.l_max)[0]
        k = rearrange(k, 'l (o d) -> o d l', o=self.order - 1)
        
        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            v = self.filter_fn(v, self.l_max, k=k[o], bias=self.filter_fn.bias)

        y = rearrange(v * x[0], 'b d l -> b l d')
        y = self.out_proj(y)
        #print(f"Output shape: {y.shape}")
        return y


class HyenaNet(nn.Module):
    def __init__(self, vocab_size, emb_dim, seq_len, d_model, num_classes, order=2, filter_order=32):
        super(HyenaNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.hyena_operator = HyenaOperator(d_model=emb_dim, l_max=seq_len, order=order, filter_order=filter_order)
        
        # Classifier head
        self.fc1 = nn.Linear(emb_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # Convert token indices to embeddings
        #x = x.permute(0, 2, 1)  # Adjust dimension ordering for convolution
        x = self.hyena_operator(x)
        #x = x.permute(0, 2, 1)  # Adjust back after convolution
        
        x = x.mean(dim=1)  # Global average pooling over the sequence dimension
        #print(x.shape)
        # Classifier
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.1)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.1)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, 0.1)
        x = self.output(x)
        return x




def train_model(model, dataloader, test_dataloader, num_epochs, scaler, optimizer, criterion):
    # Print the device IDs used by DataParallel
    print(f"DataParallel is using devices: {model.device_ids}")
    scheduler = CosineAnnealingLR(optimizer, T_max=10 - 1, eta_min=0)
    best_map, average_map = 0.0, 0.0
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
                total_loss += loss

                train_outputs.append(outs.cpu().numpy())  # Collect outputs for MAP calculation
                train_targets.append(labs.cpu().numpy())

                if pbar.n % 4000 == 0 and pbar.n != 0:
                    map_brd4, map_hsa, map_seh, average_map, true_positives_brd4, predicted_positives_brd4, \
                    true_positives_hsa, predicted_positives_hsa, true_positives_seh, predicted_positives_seh = calculate_individual_map(train_outputs, train_targets)
                    
                    print(f"Epoch {epoch} - Avg MAP: {average_map:.4f} | "
                        f"BRD4: TP {true_positives_brd4}, PP {predicted_positives_brd4}, MAP {map_brd4:.4f} | "
                        f"HSA: TP {true_positives_hsa}, PP {predicted_positives_hsa}, MAP {map_hsa:.4f} | "
                        f"SEH: TP {true_positives_seh}, PP {predicted_positives_seh}, MAP {map_seh:.4f}")

                pbar.set_postfix(loss=f"{total_loss / (pbar.n + 1):.4f}", lr=f"{lr:.6f}", train_map=f"{average_map:.4f}")
                pbar.update()


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

        print(f"Epoch {epoch} - Validation Avg MAP: {average_map:.4f} | "
            f"BRD4: TP {true_positives_brd4}, PP {predicted_positives_brd4}, MAP {map_brd4:.4f} | "
            f"HSA: TP {true_positives_hsa}, PP {predicted_positives_hsa}, MAP {map_hsa:.4f} | "
            f"SEH: TP {true_positives_seh}, PP {predicted_positives_seh}, MAP {map_seh:.4f} | "
            f"Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")


        if average_map > best_map:
            best_map = average_map
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist
            model_path = os.path.join(model_dir, f"multi_model_hyena_1.pth")
            torch.save(model.module.state_dict(), model_path)

    cleanup()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

#def main_worker(rank, world_size, num_epochs, initial_lr, batch_size, loaded_targets, flat_bbs, graph_dict):
def main_worker(rank, world_size, num_epochs, initial_lr, batch_size, tokens_path, targets_path):
    setup(rank, world_size)
    
    device = torch.device(f'cuda:{rank}')

    vocab_size = 43  # Total unique tokens
    emb_dim = 128  # Embedding dimensionality
    seq_len = 130  # Sequence length
    num_classes = 3  # Output classes
    model = HyenaNet(vocab_size, emb_dim, seq_len, emb_dim, num_classes).to(device)

    model = DDP(model, device_ids=[rank])
    
    criterion = nn.BCEWithLogitsLoss().to(device)
    scaler = GradScaler()
    #optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, betas=(0.9, 0.95), eps=1e-8)
    

    # Load the entire dataset via memmap (only file paths are passed, not the data itself)
    full_dataset = MoleculeDataset(tokens_path, targets_path)

    nfolds, fold = 10, 0

    # Create indices for KFold
    num_samples = len(full_dataset)
    indices = np.arange(num_samples)
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=42)
    train_idx, val_idx = list(kf.split(indices))[rank % nfolds]  # Use rank to select fold

    # Subset the full dataset based on indices
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    # Create samplers for distributing data across the GPUs
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=8, pin_memory=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=8, pin_memory=True, collate_fn=custom_collate)
    
    print(f"Starting training on rank {rank}")
    train_model(model, train_loader, val_loader, num_epochs, scaler, optimizer, criterion)
    
    cleanup()



def spawn_workers(world_size, num_epochs, initial_lr, batch_size, tokens_path, targets_path):
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, num_epochs, initial_lr, batch_size, tokens_path, targets_path))


if __name__ == '__main__':
    world_size = torch.cuda.device_count()

    # Load your dataset
    #loaded_data = np.load('../leash_dti/data/dataset.npz')
    loaded_data = np.load('../data/selfies_data.npz')

    tokens_path, targets_path = 'tokens_memmap.dat', 'targets_memmap.dat'
    print(f'Loaded buildingblock_ids shape: 98milly')

    num_epochs = 10
    initial_lr = 1e-3
    batch_size = 1024

    spawn_workers(world_size, num_epochs, initial_lr, batch_size, tokens_path, targets_path)