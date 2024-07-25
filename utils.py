import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import average_precision_score
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score


def update_learning_rate(optimizer, epoch, batch_idx, total_steps, warmup_steps, initial_lr, dataset_len):
    """Update learning rate based on warmup and decay."""
    if epoch * dataset_len + batch_idx < warmup_steps:
        # Warm-up phase: linear increase
        lr = initial_lr * (epoch * dataset_len + batch_idx + 1) / warmup_steps
    else:
        # Exponential decay
        decay_rate = 0.95  # Decay rate per epoch after warmup
        lr = initial_lr * (decay_rate ** (epoch - warmup_steps / dataset_len))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


def calculate_individual_map(y_scores, y_true):
    """
    Calculate the mean average precision (MAP) for each protein individually and return their average.

    Args:
    y_true (np.array): True labels reshaped to (-1, 3) where each column is a protein.
    y_scores (np.array): Predicted scores reshaped to (-1, 3) where each column is a protein.

    Returns:
    tuple: A tuple containing individual MAPs for BRD4, HSA, sEH, and the average MAP.
    """
    # Ensure y_true and y_scores are reshaped correctly
    y_true = np.array(y_true, dtype=np.float32).reshape(-1, 3)
    y_scores = np.array(y_scores, dtype=np.float32).reshape(-1, 3)

    # Calculate MAP for each column (protein)
    map_brd4 = average_precision_score(y_true[:, 0], y_scores[:, 0])
    map_hsa = average_precision_score(y_true[:, 1], y_scores[:, 1])
    map_seh = average_precision_score(y_true[:, 2], y_scores[:, 2])

    # Calculate the average MAP across all proteins
    average_map = np.mean([map_brd4, map_hsa, map_seh])

    # Calculate true positives and predicted positives
    threshold = 0.5
    true_positives_brd4 = np.sum((y_true[:, 0] == 1) & (y_scores[:, 0] > threshold))
    predicted_positives_brd4 = np.sum(y_scores[:, 0] > threshold)
    
    true_positives_hsa = np.sum((y_true[:, 1] == 1) & (y_scores[:, 1] > threshold))
    predicted_positives_hsa = np.sum(y_scores[:, 1] > threshold)
    
    true_positives_seh = np.sum((y_true[:, 2] == 1) & (y_scores[:, 2] > threshold))
    predicted_positives_seh = np.sum(y_scores[:, 2] > threshold)

    return map_brd4, map_hsa, map_seh, average_map, true_positives_brd4, predicted_positives_brd4, true_positives_hsa, predicted_positives_hsa, true_positives_seh, predicted_positives_seh




def pre_split_data(flat_bbs, graph_dict, ys, fold=0, nfolds=5, seed=2023, testing=False):
    if testing:
        # If test, return only a 50th of the data
        subset_size = len(flat_bbs) // 150  # Adjusting for 3 items per group
        indices = np.arange(subset_size)
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        def select_data(idx):
            selected_flat_bbs = []
            selected_ys = []
            for i in idx:
                start_idx = i * 3
                selected_flat_bbs.extend(flat_bbs[start_idx:start_idx + 3])
                selected_ys.append(ys[i])
            return {'flat_bbs': np.array(selected_flat_bbs), 'graph_dict': graph_dict, 'ys': np.array(selected_ys)}
        
        subset_indices = indices[:subset_size]
        test_data = select_data(subset_indices)
        return test_data, test_data  # Returning the same subset for train and val for simplicity

    else:
        # Perform K-Fold Splitting
        kf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
        indices = np.arange(len(flat_bbs) // 3)
        folds = list(kf.split(indices))
        train_idx, val_idx = folds[fold]
        
        def select_data(idx):
            selected_flat_bbs = []
            selected_ys = []
            for i in idx:
                start_idx = i * 3
                selected_flat_bbs.extend(flat_bbs[start_idx:start_idx + 3])
                selected_ys.append(ys[i])
            return {'flat_bbs': np.array(selected_flat_bbs), 'graph_dict': graph_dict, 'ys': np.array(selected_ys)}
        
        train_data = select_data(train_idx)
        val_data = select_data(val_idx)
        
        return train_data, val_data