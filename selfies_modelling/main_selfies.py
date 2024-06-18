
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
import os
import psutil

#from memory_profiler import profile


import psutil
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#os.environ['CUDNN_LOGINFO_DBG'] = "1"
#os.environ['CUDNN_LOGDEST_DBG'] = "stdout"

process = psutil.Process(os.getpid())

def print_memory_usage():
    mem = process.memory_info()
    print(f"VIRT: {mem.vms / (1024**3):.2f} GB; RES: {mem.rss / (1024**3):.2f} GB")


class MoleculeDataset(Dataset):
    def __init__(self, inputs, labels, train=True, device='cpu'):
        self.inputs = np.memmap(inputs, dtype=np.uint8, mode='r', shape=(98415610, 142))
        self.labels = np.memmap(labels, dtype=np.uint8, mode='r', shape=(98415610, 3))
        #print(f'Input shape: {self.inputs.shape}, Labels shape: {self.labels.shape}')
        #print(f"Max_Vocab = {np.max(self.inputs)}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        assert idx < len(self.inputs), "Index out of range"
        tokens = torch.tensor(self.inputs[idx], dtype=torch.long)
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        return tokens, labels
        



def custom_collate(batch):
    inputs, labels = zip(*batch)  # Unpack batch of tuples

    # Stack inputs and labels
    inputs_tensor = torch.stack(inputs).long()  # Ensure input is converted to torch.long
    labels_tensor = torch.stack(labels).float()  # Ensure labels are torch.float

    return inputs_tensor, labels_tensor



def process_batch(batch, model, scaler, optimizer, criterion, train=True, protein_index=None):
    # Move batch to the appropriate device
    device = next(model.parameters()).device
    #batch = [item.to(device) for item in batch]
    inputs, labels = batch

    inputs = inputs.to(next(model.parameters()).device)
    labels = labels.to(next(model.parameters()).device)

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

    """if train:
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()"""

    return outputs.detach(), y_vals.detach(), loss #.item()



def process_batch(batch, model, scaler, optimizer, criterion, train=True, protein_index=None):
    # Move batch to the appropriate device
    device = next(model.parameters()).device
    #batch = [item.to(device) for item in batch]
    inputs, labels = batch

    inputs = inputs.to(next(model.parameters()).device)
    labels = labels.to(next(model.parameters()).device)

    #with autocast():
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

    return outputs.detach(), y_vals.detach(), loss #.item()

import torch
import torch.nn.functional as F

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
        #print(f"Output dimension: {self.d_out}")
        self.out_proj = nn.Linear(d_model, d_out)
        
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
        self.hyena_operator = HyenaOperator(d_model=emb_dim, l_max=seq_len, d_out = emb_dim*2, order=order, filter_order=filter_order)
        self.hyena_operator2 = HyenaOperator(d_model=emb_dim*2, l_max=seq_len,d_out = emb_dim*2, order=order, filter_order=filter_order)
        #self.hyena_operator3 = HyenaOperator(d_model=emb_dim*3, l_max=seq_len, d_out = emb_dim*3,order=order, filter_order=filter_order)
        
        # Classifier head
        self.fc1 = nn.Linear(emb_dim*2, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # Convert token indices to embeddings
        #x = x.permute(0, 2, 1)  # Adjust dimension ordering for convolution
        x = self.hyena_operator(x)
        x = self.hyena_operator2(x)
        #x = self.hyena_operator3(x)
        #x = x.permute(0, 2, 1)  # Adjust back after convolution
        
        #x = x.mean(dim=1)  # Global average pooling over the sequence dimension
        x, _ = x.max(dim=1)  # Global max pooling over the sequence dimension

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

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNetWithMaxPooling(nn.Module):
    def __init__(self, vocab_size, emb_dim, seq_len, num_classes):
        super(ConvNetWithMaxPooling, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        # Convolutional layers
        self.conv1 = nn.Conv1d(emb_dim, 32, kernel_size=3, padding='valid', stride=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding='valid', stride=1)
        self.conv3 = nn.Conv1d(64, 96, kernel_size=3, padding='valid', stride=1)

        # Max pooling layer
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Classifier head
        self.fc1 = nn.Linear(96, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.output = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # Convert token indices to embeddings
        x = x.permute(0, 2, 1)  # Adjust dimension ordering for convolution

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.global_max_pool(x)
        x = torch.flatten(x, 1)  # Flatten the output for the dense layers

        # Classifier
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.1)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.1)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, 0.1)
        x = torch.sigmoid(self.output(x))

        return x




def train_model(model, dataloader, test_dataloader, num_epochs, scaler, optimizer, criterion):
    # Print the device IDs used by DataParallel
    print(f"DataParallel is using devices: {model.device_ids}")
    scheduler = CosineAnnealingLR(optimizer, T_max=10 - 1, eta_min=0)
    best_map, average_map = 0.0, 0.0
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}")
        dataloader.sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0
        total_loss = 0
        train_outputs, train_targets = [], []
        lr = optimizer.param_groups[0]['lr']
        accumulation_steps = 2  # Number of steps to accumulate gradients

        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}") as pbar:
            for batch_idx, data in enumerate(dataloader):
                #torch.cuda.empty_cache() 
                outs, labs, loss = process_batch(data, model, scaler, optimizer, criterion, train=True)
                loss = loss / accumulation_steps  # Normalize the loss to account for accumulation
                scaler.scale(loss).backward()  # Backpropagate the loss
    
                if (batch_idx + 1) % accumulation_steps == 0 or batch_idx + 1 == len(dataloader):
                    scaler.step(optimizer)  # Only step every `accumulation_steps`
                    scaler.update()
                    optimizer.zero_grad()  # Zero out the gradients after updating

                total_loss += loss.item() * accumulation_steps 
                train_outputs.append(outs.cpu().numpy())  # Collect outputs for MAP calculation
                train_targets.append(labs.cpu().numpy())

                if pbar.n % 500 == 0 and pbar.n != 0:
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
    os.environ['MASTER_PORT'] = '12359'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    print("Cleaned up distributed process group.")

#@profile
def main_worker(rank, world_size, num_epochs, initial_lr, batch_size, tokens_path, targets_path, indicies):
    setup(rank, world_size)
    
    print(f"Rank {rank}: Setting up device and model.")
    device = torch.device(f'cuda:{rank}')
    #model = HyenaNet(37, 128, 142, 128, 3).to(device) #128 -> 32 
    model = ConvNetWithMaxPooling(vocab_size=37, emb_dim=128, seq_len=142, num_classes=3).to(device)
    model = DDP(model, device_ids=[rank])

    #criterion = nn.BCEWithLogitsLoss().to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.005)
    #optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    scaler = GradScaler()

    print(f"Rank {rank}: Loading dataset.")
    full_dataset = MoleculeDataset(tokens_path, targets_path)
    print(f"Rank {rank}: Dataset loaded with {len(full_dataset)} samples.")

    # Assuming KFold split is done outside and indices are passed or managed via a shared file or setting
    """num_samples = len(full_dataset)
    indices = np.arange(num_samples)
    kf = KFold(n_splits=15, shuffle=True, random_state=42)
    train_idx, val_idx = next(iter(kf.split(indices)))"""

    train_idx, val_idx = indicies[0], indicies[1]

    print(len(train_idx), len(val_idx))

    subset_indices = np.random.choice(len(full_dataset), size=1000000, replace=False) 

    train_dataset = Subset(full_dataset, train_idx) # train_idx) # subset_indices) #
    val_dataset = Subset(full_dataset, val_idx)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=1, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=1, pin_memory=False)

    print(f"Rank {rank}: Starting training.")
    train_model(model, train_loader, val_loader, num_epochs, scaler, optimizer, criterion)
    
    cleanup()




def spawn_workers(world_size, num_epochs, initial_lr, batch_size, tokens_path, targets_path, indicies):
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, num_epochs, initial_lr, batch_size, tokens_path, targets_path, indicies))

"""
from memory_profiler import profile

@profile
def test_data_loading():
    dataset = MoleculeDataset('tokens_memmap.dat', 'targets_memmap.dat')
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    counter = 0
    for data in loader:
        print(f"Loaded batch with data shape: {data[0].shape}")
        counter += 1
        if counter >= 20:
            break

if __name__ == "__main__":
    test_data_loading()

"""

from sklearn.model_selection import StratifiedKFold


if __name__ == '__main__':
    world_size = torch.cuda.device_count()

    # Load your dataset
    #loaded_data = np.load('../leash_dti/data/dataset.npz')
    #loaded_data = np.load('../data/selfies_data.npz')

    tokens_path, targets_path = 'tokens_memmap_smiles3.dat', 'targets_memmap_smiles3.dat'
    print(f'Loaded buildingblock_ids shape: 98milly')


    # Assuming targets are uint8 and there are 3 targets per sample
    targets = np.memmap(targets_path, dtype=np.uint8, mode='r', shape=(98415610, 3))
    target_sums = targets.sum(axis=1)
    skf = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)

    # Retrieve indices from stratified k-fold
    for fold, (train_idx, valid_idx) in enumerate(skf.split(np.arange(98415610), target_sums)):
        if fold not in [0]:
            continue
        break

    del targets
    gc.collect()
    print(f'Fold: {fold}, Train: {len(train_idx)}, Valid: {len(valid_idx)}')
    num_epochs = 10
    initial_lr = 1e-3
    batch_size = 2048
    print_memory_usage()
    spawn_workers(world_size, num_epochs, initial_lr, batch_size, tokens_path, targets_path, (train_idx, valid_idx))
