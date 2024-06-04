import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import gc

import numpy as np
import os
import torch
from torch.utils.data import Dataset

class MoleculeDataset(Dataset):
    def __init__(self, data_dir, filenames, train=True, device='cpu'):
        """
        Initializes the dataset by loading all `.npz` files from the specified directory,
        and storing the arrays as large concatenated numpy arrays.

        Args:
        data_dir (str): The directory containing `.npz` files with token and label data.
        filenames (list of str): List of filenames within the data_dir to be used.
        """
        self.inputs = []
        self.labels = []

        # Load all .npz files from the directory
        for filename in filenames:
            data_path = os.path.join(data_dir, filename)
            with np.load(data_path) as data:
                # Append NumPy arrays
                self.inputs.append(data['tokens'])
                self.labels.append(data['labels'])
            print(f"Loaded {filename}")

        # Vertically stack all arrays to form a large array
        self.inputs = np.vstack(self.inputs)
        self.labels = np.vstack(self.labels)

        print(f"Total samples loaded: {len(self.inputs)}")

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
        tokens = torch.tensor(self.inputs[idx], dtype=torch.long)
        labels = torch.tensor(self.labels[idx], dtype=torch.float)
        return tokens, labels




from sklearn.model_selection import KFold

def prepare_kfold_splits(num_files, k=30):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    indices = np.arange(num_files)
    splits = list(kf.split(indices))
    return splits



import numpy as np
import os
from torch.utils.data import Dataset

import torch
import numpy as np

def custom_collate(batch):
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    inputs_np = np.stack(inputs, axis=0)
    labels_np = np.stack(labels, axis=0)

    inputs_tensor = torch.from_numpy(inputs_np).to(torch.long)
    labels_tensor = torch.from_numpy(labels_np).to(torch.float)

    print(f"Input Tensor Type in collate: {inputs_tensor.dtype}")  # This should print torch.int64 or torch.long

    return inputs_tensor, labels_tensor



class DynamicMoleculeDataset(Dataset):
    def __init__(self, data_dir, train_filenames, train=True, device='cpu'):
        self.data_dir = data_dir
        self.train_filenames = train_filenames  # List of all training file names
        self.current_files = []
        self.used_files = set()  # Track files that have been used
        #self.load_new_files()
        self.device = device
        self.train = train
        if not train:
            self.load_new_files()

    def load_new_files(self):
        # Reset used files if all have been used
        if len(self.used_files) == len(self.train_filenames):
            self.used_files = set()

        # Select files that have not been used
        available_files = list(set(self.train_filenames) - self.used_files)
        # Handle cases where available files are fewer than 5
        num_files_to_select = min(5, len(available_files))
        if num_files_to_select > 0:
            selected_indices = np.random.choice(len(available_files), num_files_to_select, replace=False)
            self.current_files = [available_files[idx] for idx in selected_indices]
            self.used_files.update(self.current_files)
        else:
            self.current_files = []

        # Load and stack data from these files
        inputs = []
        labels = []
        for filename in self.current_files:
            data_path = os.path.join(self.data_dir, filename)
            with np.load(data_path) as data:
                inputs.append(data['tokens'])
                labels.append(data['labels'])
        print(f"Loaded {self.current_files}")
        
        self.inputs = np.vstack(inputs)
        self.labels = np.vstack(labels)
        del inputs, labels
        gc.collect()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        # Return data from currently loaded chunks
        return self.inputs[index], self.labels[index]






class DynamicTorchMoleculeDataset(Dataset):
    def __init__(self, data_dir, train_filenames, train=True, device='cpu', files_per_subbatch=5):
        self.data_dir = data_dir
        self.train_filenames = train_filenames  # List of all training file names
        self.current_files = []
        self.used_files = set()  # Track files that have been used
        self.device = device
        self.train = train
        self.files_per_subbatch= files_per_subbatch
        if not train:
            self.load_new_files()
        
    def load_new_files(self):
        # Reset used files if all have been used
        if len(self.used_files) == len(self.train_filenames):
            self.used_files = set()

        # Select files that have not been used
        available_files = list(set(self.train_filenames) - self.used_files)
        # Handle cases where available files are fewer than 5
        num_files_to_select = min(self.files_per_subbatch, len(available_files))
        if num_files_to_select > 0:
            selected_indices = np.random.choice(len(available_files), num_files_to_select, replace=False)
            self.current_files = [available_files[idx] for idx in selected_indices]
            self.used_files.update(self.current_files)
        else:
            self.current_files = []

        inputs = []
        labels = []
        # Load all .npz files from the directory
        for filename in self.current_files:
            data_path = os.path.join(self.data_dir, filename)
            with np.load(data_path) as data:
                # Convert NumPy arrays to tensors
                inputs.append(torch.tensor(data['tokens'], dtype=torch.long))
                labels.append(torch.tensor(data['labels'], dtype=torch.float))
        print(f"Loaded {self.current_files}")
        
        # Concatenate all chunks into two big tensors
        self.inputs = torch.cat(inputs, dim=0)
        self.labels = torch.cat(labels, dim=0)
        del inputs, labels
        gc.collect()

        """if not self.train:
            self.inputs.to(self.device)
            self.labels.to(self.device)"""


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        # Return data from currently loaded chunks
        return self.inputs[index], self.labels[index]




