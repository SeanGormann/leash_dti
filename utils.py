

# Fix fastai bug to enable fp16 training with dictionaries
import torch
from fastai.vision.all import *
def flatten(o):
    "Concatenate all collections and items as a generator"
    for item in o:
        if isinstance(o, dict): yield o[item]; continue
        elif isinstance(item, str): yield item; continue
        try: yield from flatten(item)
        except TypeError: yield item

from torch.cuda.amp import GradScaler, autocast
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

import fastai
fastai.callback.fp16.MixedPrecision = MixedPrecision



def dict_to_device(data, device='cuda'):
    """
    Recursively move all tensor values in the dictionary to the specified device.
    """
    if isinstance(data, dict):
        return {k: dict_to_device(v, device=device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        #return [dict_to_device(x, device=device) for x in data]
        return [dict_to_device(x, device=device) for x in data]
    elif hasattr(data, 'to'):  # For PyTorch Geometric Batch or Data objects
        return data.to(device)
    else:
        return data


class DeviceDataLoader:
    """
    Wrap a dataloader to move data to a device.
    """
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """
        Iterate over batches and move each batch to the specified device.
        """
        for batch in self.dl:
            yield dict_to_device(batch, device=self.device)

    def __len__(self):
        return len(self.dl)



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

# Batch processing function
def process_batch(batch, train=True):
    #batch = batch.to(device)
    outputs = model(batch)
    #y_vals = batch.y... #change this
    y_vals = batch.y.view(-1, 3)  #batch.y[1::3]

    loss = criterion(outputs.squeeze(),y_vals)
    if train:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return outputs.detach(), y_vals.detach(), loss.item()

def process_batch(batch, model, scaler, optimizer, train=True):
    # Assuming the model, criterion, and optimizer are passed as arguments or are available globally
    with autocast():
        outputs = model(batch)
        y_vals = batch.y.view(-1, 3)  # Adjust this line based on your batch data structure
        loss = criterion(outputs.squeeze(), y_vals)

    if train:
        optimizer.zero_grad()
        # Use the scaler to scale the loss and call backward
        #if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #else:
        #    loss.backward()
        #    optimizer.step()

    return outputs.detach(), y_vals.detach(), loss.item()


def process_batch(batch, model, scaler, optimizer, criterion, train=True, protein_index=None):
    with autocast():
        outputs = model(batch)
        
        # Adjust reshaping based on your batch data structure and need
        if protein_index is not None:
            y_vals = batch.y.view(-1, 3)[:, protein_index].view(-1, 1)  # Select the correct targets and reshape
        else:
            y_vals = batch.y.view(-1, 3)  # 
        
        #loss = criterion(outputs.squeeze(), y_vals)
        loss = criterion(outputs, y_vals)

    if train:
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return outputs.detach(), y_vals.detach(), loss.item()



import torch
import torch.nn as nn
import torch.nn.functional as F

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
