print("Loading libraries...")
import pandas as pd
#import polars as pl
import numpy as np
import time
import gc
import os
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

import torch
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
#from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader

#import zipfile

from models import *
from data import *
from utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
print("...Libraries loaded.")



root_path_ = "data/"
filenames = os.listdir(os.path.join(root_path_, 'graph_data_v11'))
filenames = [f"graph_data_v11/{filename}" for filename in filenames if filename.endswith(".pt")]

"""
pos_filenames = os.listdir(os.path.join(root_path_, 'positives'))
pos_filenames = [f"positives/{filename}" for filename in pos_filenames if filename.endswith(".pt")]
neg_filenames = os.listdir(os.path.join(root_path_, 'negatives_2ndhalf'))
neg_filenames = [f"negatives_2ndhalf/{filename}" for filename in neg_filenames if filename.endswith(".pt")]

path_to_blind = "data/all_blind_mols_half"
blind_filenames = os.listdir(path_to_blind)
blind_filenames = [filename for filename in blind_filenames if filename.endswith(".pt")]
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

TESTING = True

if TESTING:
    train_dataset = DynamicGraphDataset(root_path_, filenames, fold=0, nfolds=2, train=True, device=device, active_batches=1)
    val_dataset = DynamicGraphDataset(root_path_, filenames, fold=0, nfolds=2, train=False, device=device, active_batches=1)
else:
    train_dataset = DynamicGraphDataset(root_path_, filenames, fold=0, nfolds=15, train=True, device=device, active_batches=28)
    val_dataset = DynamicGraphDataset(root_path_, filenames, fold=0, nfolds=15, train=False, device=device, active_batches=2)
"""
if TESTING:
    train_dataset = GraphDataset(root_path_, pos_filenames[:1], neg_filenames[:1], eval=False, device = device) #, oversample_factor=2) #use_half_pos=True)
    blind_dataset = GraphDataset(path_to_blind, blind_filenames[:1], None, eval=True, device = device)
else:
    train_dataset = GraphDataset(root_path_, pos_filenames, neg_filenames, eval=False, device = device) #, oversample_factor=2) # use_half_pos=True)
    blind_dataset = GraphDataset(path_to_blind, blind_filenames, None, eval=True, device = device)
"""


dl_train = GeoDataLoader(train_dataset, batch_size=256, shuffle=True) #, persistent_workers=True, num_workers=4)
dl_val = GeoDataLoader(val_dataset, batch_size=256, shuffle=True) #, persistent_workers=False, num_workers=2)

print(f"Number of training batches: {len(dl_train)}, Number of validation batches: {len(dl_val)}")


model = MultiMolGNN(num_node_features=8, num_edge_features=4, num_layers=8, hidden_dim=320, dropout=0.1)
#model = MolGraphTransformer(n_layers=4, node_dim=8, edge_dim=4, hidden_dim=320, n_heads=8, in_feat_dropout=0.1, dropout=0.1, pos_enc_dim=4)
model.to(device)
print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')


# Total number of epochs and warmup steps
n_epochs = 15
warmup_steps = 196 #196
total_steps = len(train_dataset) * n_epochs
initial_lr = 0.0001 #0.0001

optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr) #lr=0.0001)
#criterion = nn.BCEWithLogitsLoss(pos_weight=2)
weights = torch.tensor([1.0, 2.0])
criterion = nn.BCEWithLogitsLoss(pos_weight=weights[1]) 
criterion = FocalLoss(alpha=0.5, gamma=2.0)

# Scheduler and scaler for AMP
scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs - 1, eta_min=0)
#scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
#scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=2)
scaler = GradScaler()

protein_index = {'BRD4': 0, 'HSA': 1, 'sEH': 2}  # Default to 0th index if protein not found

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

        _, _, loss = process_batch(batch.to(device), model, scaler, optimizer, criterion, train=True, protein_index=None) #protein_index['HSA']
        total_loss += loss
        pbar.set_postfix(loss=f"{total_loss / (i + 1):.4f}", lr=f"{lr:.6f}")

    # Step the scheduler only after warm-up is complete
    scheduler.step()

    # Validation and Metric Calculation
    model.eval()
    model_outputs = []
    true_labels = []
    vloss = 0
    with torch.no_grad():
        for batch in dl_val:
            outputs, labels, val_loss = process_batch(batch.to(device),  model, scaler, optimizer, criterion, train=False, protein_index=None)
            model_outputs.append(outputs.squeeze().cpu())
            true_labels.append(labels.cpu())
            vloss += val_loss

    model_outputs = torch.cat(model_outputs)
    true_labels = torch.cat(true_labels)
    model_outputs = torch.sigmoid(model_outputs).numpy()  # Apply sigmoid to convert logits to probabilities

    #train_dataset.shuffle_files()
    #train_dataset.update_active_set()
    #dl_train = GeoDataLoader(train_dataset, batch_size=196, shuffle=True)
    # Calculate Mean Average Precision (micro)
    map_micro = average_precision_score(true_labels.numpy(), model_outputs, average='micro')
    #print(map_micro)
    #scheduler.step(map_micro)
    print(f"Epoch {epoch+1}, Train Loss: {total_loss / len(dl_train)}, Val Loss: {vloss/len(dl_val)},Validation MAP (micro): {map_micro}, Time: {time.time() - start_time}\n----\n")


# Specify the path and filename
model_path = 'models/multi_model_v17.pth'

# Save the model's state dictionary
torch.save(model.state_dict(), model_path)
