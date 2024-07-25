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
from collections import OrderedDict


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

all_dtis = '../data/test_bb.csv' #test_prot test_bb
building_blocks = '../data/bbs_with_edgeattr.pkl' #'/content/drive/MyDrive/all_buildingblock.pkl' '/content/drive/MyDrive/all_bbs_pca_8mixfeats.pkl' 

bbs = pd.read_pickle(building_blocks)
all_dtis = pd.read_csv(all_dtis)


class TestDataset(Dataset):
    def __init__(self, reaction_df, block_df, protein_dict, device='cpu', fold=0, nfolds=5, train=True, test=False, blind=False, seed=2023, max_len=584):
        self.device = device
        self.reaction_df = reaction_df
        self.block_df = block_df
        self.protein_dict = protein_dict  # Precomputed encoded protein sequences
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
        protein_seq = torch.tensor([1])

        return graph1, graph2, graph3, y

    def prepare_graph(self, graph):
        graph.to(self.device)
        return graph
    
    
def custom_collate_fn(batch):
    graphs_array_1 = [item[0] for item in batch]
    graphs_array_2 = [item[1] for item in batch]
    graphs_array_3 = [item[2] for item in batch]
    ys_array = np.array([item[3] for item in batch])  # Convert list of numpy arrays to a single numpy array
    return Batch.from_data_list(graphs_array_1), Batch.from_data_list(graphs_array_2), Batch.from_data_list(graphs_array_3), torch.tensor(ys_array)


def run_test(model_path):
    all_dtis = '../data/test_bb.csv' #test_prot test_bb
    building_blocks = '../data/bbs_with_edgeattr.pkl' #'/content/drive/MyDrive/all_buildingblock.pkl' '/content/drive/MyDrive/all_bbs_pca_8mixfeats.pkl' 

    bbs = pd.read_pickle(building_blocks)
    all_dtis = pd.read_csv(all_dtis)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds_train = TestDataset(all_dtis, bbs, device = device, fold=0, nfolds=20, train=False, test=True)
    dl_test = GeoDataLoader(ds_train, batch_size=2048, shuffle=True) #, collate_fn=custom_collate_fn). 192 2048

    ### Load model
    from collections import OrderedDict
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # Remove 'module.' prefix
        if 'lin.weight' in k:
            # Duplicate weight for both lin_src and lin_dst
            new_key_src = k.replace('lin.weight', 'lin_src.weight')
            new_key_dst = k.replace('lin.weight', 'lin_dst.weight')
            new_state_dict[new_key_src] = v
            new_state_dict[new_key_dst] = v
        else:
            new_state_dict[k] = v

    model = Net().to(device)  # Make sure to initialize your model before loading the state_dict
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # Function to process and predict for all proteins
    def predict_batch(batch):
        with torch.no_grad():
            outputs = model(batch)
            predictions = torch.sigmoid(outputs)  # Convert logits to probabilities for all proteins
        return predictions.cpu()  # Detach from CUDA and send to CPU

    start_time = time.time()
    all_predictions = []
    all_ids = []
    for batch in tqdm(dl_test, desc="Inference"):
        preds = predict_batch(batch)
        all_predictions.extend(preds.tolist())
        all_ids.extend(batch[3].tolist())

    print(f"Completed inference in {time.time() - start_time:.2f} seconds.")
    final_predictions = []
    final_ids = []

    for preds, ids in zip(all_predictions, all_ids):
        for pred, id in zip(preds, ids):
            if id != -1:
                final_predictions.append(pred)
                final_ids.append(id)

    print(len(final_predictions), len(final_ids))

    threshold = 0.5
    num_positive_hsa = np.sum(np.array(final_predictions) > threshold)
    print(f"Number of positive predictions for final_preds after sigmoid: {num_positive_hsa}")

    results = pd.DataFrame({
        'id': final_ids,
        'prediction': final_predictions
    })

    sample_submission = pd.read_csv('../data/sample_submission.csv')
    final_submission = sample_submission.merge(results, on='id', how='left')
    final_submission = final_submission.drop(columns=['binds'])
    final_submission = final_submission.rename(columns={'prediction': 'binds'})

    print(final_submission.shape)
    final_submission.to_csv('submissions/final_predictions_v84.csv', index=False)