from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.utils.data import Dataset
import os
import torch
import numpy as np
from sklearn.model_selection import KFold


class DynamicGraphDataset(Dataset):
    def __init__(self, directory, filenames, fold, nfolds, train=True, seed=2023, device='cpu', active_batches=8):
        """
        Args:
            directory (str): Path to the directory containing .pt graph files.
            filenames (list of str): A list of filenames for all samples.
            fold (int): Current fold index.
            nfolds (int): Total number of folds.
            train (bool): Whether this dataset is for training or validation.
            seed (int): Random seed for reproducibility.
            device (str): Device to load data onto ('cpu', 'cuda', etc.).
            active_batches (int): Number of batches to keep loaded in memory at any time.
        """
        self.directory = directory
        self.device = device
        self.active_batches = active_batches

        # Load filenames and perform K-Fold splitting
        self.all_filenames = np.array(filenames)
        kf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
        folds = list(kf.split(self.all_filenames))
        train_idx, eval_idx = folds[fold]

        self.selected_indices = train_idx if train else eval_idx
        self.selected_filenames = self.all_filenames[self.selected_indices]

        print(f"Active batches: {self.active_batches}, Total batches: {len(self.selected_filenames)}")

        if active_batches > len(self.selected_filenames):
            self.active_batches = len(self.selected_filenames)

        print(f"Active batches: {self.active_batches}, Total batches: {len(self.selected_filenames)}")

        # Load initial set of batches
        self.loaded_graphs = [torch.load(os.path.join(self.directory, self.selected_filenames[i])) for i in range(self.active_batches)]
        self.graphs_per_batch = len(self.loaded_graphs[0])  # Assume all batches have the same size

        # Pre-compute the index mapping
        self.index_mapping = []
        for i in range(self.active_batches):
            for j in range(self.graphs_per_batch):
                self.index_mapping.append((i, j))

    def __len__(self):
        return self.active_batches * self.graphs_per_batch


    def __getitem__(self, idx):
        batch_index = idx // len(self.loaded_graphs[0])
        index_in_batch = idx % len(self.loaded_graphs[0])
        graph = self.loaded_graphs[batch_index][index_in_batch]

        return graph

    def __getitem__(self, idx):
        batch_index, index_in_batch = self.index_mapping[idx]
        graph = self.loaded_graphs[batch_index][index_in_batch]
        return graph

    def replace_batch(self, old_batch_index, new_batch_index):
        # Load new batch
        new_graphs = torch.load(os.path.join(self.directory, self.selected_filenames[new_batch_index]))
        # Replace the old batch with the new one
        self.loaded_graphs[old_batch_index] = new_graphs

    def shuffle_files(self):
        np.random.shuffle(self.selected_filenames)





import os
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import KFold

class GraphDataset(Dataset):
    def __init__(self, directory, pos_filenames, neg_filenames, eval=False, fold=0, nfolds=0, seed=2023, device='cpu'):
        """
        Args:
            directory (str): Path to the directory containing graph files.
            pos_filenames (list of str): List of filenames for positive samples.
            neg_filenames (list of str): List of filenames for negative samples.
            eval (bool): Whether this dataset is for evaluation.
            fold (int): Current fold index.
            nfolds (int): Total number of folds for cross-validation.
            seed (int): Random seed for reproducibility.
            device (str): Device to load data onto ('cpu', 'cuda', etc.).
        """
        self.directory = directory
        self.device = device

        if eval:
            # Load all data for evaluation
            if neg_filenames is None:
                all_filenames = pos_filenames
            else:
                all_filenames = pos_filenames + neg_filenames
            self.data = []
            for fname in all_filenames:
                graphs = torch.load(os.path.join(directory, fname))
                self.data.extend([graph.to(device) for graph in graphs])
        elif nfolds == 0:
            # Load all positives to GPU and negatives to CPU without K-Fold
            self.pos_data = [graph for fname in pos_filenames for graph in torch.load(os.path.join(directory, fname))]
            self.neg_data = [graph for sublist in neg_filenames for graph in torch.load(os.path.join(directory, sublist))]
        else:
            # K-Fold splitting for both positives and negatives
            self.all_filenames = np.array(pos_filenames + neg_filenames)
            kf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
            folds = list(kf.split(self.all_filenames))
            train_idx, eval_idx = folds[fold]
            self.selected_filenames = self.all_filenames[train_idx if fold % 2 == 0 else eval_idx]
            self.data = []
            for fname in self.selected_filenames:
                graphs = torch.load(os.path.join(directory, fname))
                self.data.extend([graph.to(device) for graph in graphs])

    def __len__(self):
        if hasattr(self, 'data'):
            return len(self.data)
        else:
            return len(self.pos_data) + len(self.neg_data)

    def __getitem__(self, idx):
        if hasattr(self, 'data'):
            return self.data[idx]
        elif idx < len(self.pos_data):
            return self.pos_data[idx]
        else:
            return self.neg_data[idx - len(self.pos_data)]

    def shuffle_files(self):
        if hasattr(self, 'data'):
            np.random.shuffle(self.data)
        else:
            np.random.shuffle(self.neg_data)  # Only shuffle negatives, positives always loaded in full


"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import KFold

class GraphDataset(Dataset):
    def __init__(self, directory, pos_filenames, neg_filenames, eval=False, fold=0, nfolds=0, seed=2023, device='cpu', use_half_pos=True):
        
        Args:
            directory (str): Path to the directory containing graph files.
            pos_filenames (list of str): List of filenames for positive samples.
            neg_filenames (list of str): List of filenames for negative samples.
            eval (bool): Whether this dataset is for evaluation.
            fold (int): Current fold index.
            nfolds (int): Total number of folds for cross-validation.
            seed (int): Random seed for reproducibility.
            device (str): Device to load data onto ('cpu', 'cuda', etc.).
            use_half_pos (bool): Whether to use only half of the positive data at a time.
        
        self.directory = directory
        self.device = device
        self.use_half_pos = use_half_pos
        self.current_half = 0  # Start with the first half

        if eval:
            all_filenames = pos_filenames + (neg_filenames if neg_filenames else [])
            self.data = [graph.to(device) for fname in all_filenames for graph in torch.load(os.path.join(directory, fname))]
        else:
            self.pos_data = [graph for fname in pos_filenames for graph in torch.load(os.path.join(directory, fname))]
            self.neg_data = [graph for sublist in neg_filenames for graph in torch.load(os.path.join(directory, sublist))]

            if use_half_pos:
                self.half_size = len(self.pos_data) // 2
                self.active_pos_indices = np.arange(self.half_size)  # Indices for the first half
                self.update_active_set()

    def __len__(self):
        if hasattr(self, 'data'):
            return len(self.data)
        else:
            return len(self.active_pos_indices) + len(self.neg_data)

    def __getitem__(self, idx):
        if hasattr(self, 'data'):
            return self.data[idx]
        else:
            if idx < len(self.active_pos_indices):
                return self.pos_data[self.active_pos_indices[idx]]
            return self.neg_data[idx - len(self.active_pos_indices)]

    def update_active_set(self):
        if self.use_half_pos:
            if self.current_half == 0:
                self.active_pos_indices = np.arange(self.half_size)
            else:
                self.active_pos_indices = np.arange(self.half_size, 2 * self.half_size)
            self.current_half = 1 - self.current_half  # Toggle between 0 and 1

    def shuffle_files(self):
        if hasattr(self, 'neg_data'):
            np.random.shuffle(self.neg_data)





class GraphDataset(Dataset):
    def __init__(self, directory, pos_filenames, neg_filenames, eval=False, fold=0, nfolds=0, seed=2023, device='cpu', oversample_factor=2):
        
        Initialize the GraphDataset with given parameters.

        Parameters:
        - directory (str): Directory containing graph files.
        - pos_filenames (list of str): Positive sample filenames.
        - neg_filenames (list of str): Negative sample filenames.
        - eval (bool): Flag to load dataset for evaluation.
        - fold (int): Fold index for K-Fold validation.
        - nfolds (int): Number of folds for validation.
        - seed (int): Seed for random operations.
        - device (str): Computation device.
        - oversample_factor (int): Factor by which to oversample the positive samples.
        
        self.directory = directory
        self.device = device
        self.oversample_factor = oversample_factor

        if eval:
            all_filenames = pos_filenames + (neg_filenames if neg_filenames else [])
            self.data = [graph for fname in all_filenames for graph in torch.load(os.path.join(directory, fname))]
        else:
            self.pos_data = [graph for fname in pos_filenames for graph in torch.load(os.path.join(directory, fname))]
            self.neg_data = [graph for sublist in neg_filenames for graph in torch.load(os.path.join(directory, sublist))]

            # Oversample the positive data by replicating indices
            self.pos_indices = np.arange(len(self.pos_data) * oversample_factor) % len(self.pos_data)
            self.neg_indices = np.arange(len(self.neg_data))

    def __len__(self):
        return len(self.data) if hasattr(self, 'data') else len(self.pos_indices) + len(self.neg_indices)

    def __getitem__(self, idx):
        if hasattr(self, 'data'):
            return self.data[idx]
        elif idx < len(self.pos_indices):
            # Fetch the graph using the oversampled index
            return self.pos_data[self.pos_indices[idx]]
        else:
            # Adjust index for negatives and fetch
            adjusted_idx = idx - len(self.pos_indices)
            return self.neg_data[self.neg_indices[adjusted_idx]]

    def shuffle_files(self):
        np.random.shuffle(self.pos_indices)  # Shuffle oversampled indices
        np.random.shuffle(self.neg_indices)
"""
