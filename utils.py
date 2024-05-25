import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import average_precision_score
import numpy as np



def update_learning_rate(optimizer, epoch, batch_idx, total_steps, warmup_steps, initial_lr):
    """Update learning rate based on warmup and decay."""
    if epoch * len(train_dataset) + batch_idx < warmup_steps:
        # Warm-up phase: linear increase
        lr = initial_lr * (epoch * len(train_dataset) + batch_idx + 1) / warmup_steps
    else:
        # Exponential decay
        decay_rate = 0.95  # Decay rate per epoch after warmup
        lr = initial_lr * (decay_rate ** (epoch - warmup_steps / len(train_dataset)))

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


from sklearn.metrics import average_precision_score
import numpy as np

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

    # Print statements for debugging
    #print(f"y_true shape: {y_true.shape}, dtype: {y_true.dtype}")
    #print(f"y_scores shape: {y_scores.shape}, dtype: {y_scores.dtype}")


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

