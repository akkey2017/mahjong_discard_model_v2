"""
Utility functions for training and evaluation.
"""

import sys
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm


class TopKAccuracy:
    """Calculate top-k accuracy metric."""
    
    def __init__(self, k=3):
        self.k = k
        self.correct = 0
        self.total = 0
    
    def update(self, preds, labels):
        """Update accuracy with new predictions."""
        _, top_k_preds = preds.topk(self.k, dim=1)
        self.correct += torch.any(top_k_preds == labels.view(-1, 1), dim=1).sum().item()
        self.total += labels.size(0)
    
    def compute(self):
        """Compute current accuracy."""
        return self.correct / self.total if self.total > 0 else 0.0
    
    def reset(self):
        """Reset counters."""
        self.correct = 0
        self.total = 0


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=5, min_delta=0.0, mode='max'):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        """Check if training should stop."""
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class ModelCheckpoint:
    """Save model checkpoints during training."""
    
    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=True):
        """
        Args:
            filepath: Path to save the model
            monitor: Metric to monitor
            mode: 'min' or 'max' for the monitored metric
            save_best_only: Whether to save only when monitored metric improves
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None
    
    def __call__(self, model, metrics):
        """Save model if conditions are met."""
        score = metrics.get(self.monitor)
        if score is None:
            return
        
        should_save = False
        
        if not self.save_best_only:
            should_save = True
        elif self.best_score is None:
            should_save = True
            self.best_score = score
        elif self.mode == 'min' and score < self.best_score:
            should_save = True
            self.best_score = score
        elif self.mode == 'max' and score > self.best_score:
            should_save = True
            self.best_score = score
        
        if should_save:
            torch.save(model.state_dict(), self.filepath)
            tqdm.write(f"Model checkpoint saved to {self.filepath} ({self.monitor}={score:.4f})",
                       file=sys.stderr)


def get_optimizer(model, optimizer_name='adamw', lr=1e-4, weight_decay=1e-2):
    """
    Get optimizer for the model.
    
    Args:
        model: The neural network model
        optimizer_name: Name of the optimizer ('adam', 'adamw', 'sgd')
        lr: Learning rate
        weight_decay: Weight decay factor
    
    Returns:
        PyTorch optimizer
    """
    if optimizer_name.lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(optimizer, scheduler_name='cosine', **kwargs):
    """
    Get learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_name: Name of the scheduler ('cosine', 'plateau', 'none')
        **kwargs: Additional arguments for the scheduler
    
    Returns:
        PyTorch learning rate scheduler or None
    """
    if scheduler_name.lower() == 'cosine':
        T_max = kwargs.get('T_max', 10)
        eta_min = kwargs.get('eta_min', 1e-6)
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_name.lower() == 'plateau':
        mode = kwargs.get('mode', 'max')
        factor = kwargs.get('factor', 0.5)
        patience = kwargs.get('patience', 3)
        return ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
    elif scheduler_name.lower() == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def train_one_epoch(model, train_loader, loss_fn, optimizer, device, max_grad_norm=None):
    """
    Train the model for one epoch.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to train on
        max_grad_norm: Maximum gradient norm for clipping (None to disable)
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False, 
                file=sys.stderr, dynamic_ncols=True, mininterval=0.1)
    for xb, yb, _ in pbar:
        xb, yb = xb.to(device), yb.to(device)
        
        optimizer.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        
        # Gradient clipping
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar with current loss
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            pbar.set_postfix(loss=f"{avg_loss:.4f}")
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def evaluate(model, val_loader, loss_fn, device, metrics=None):
    """
    Evaluate the model.
    
    Args:
        model: Neural network model
        val_loader: DataLoader for validation data
        loss_fn: Loss function
        device: Device to evaluate on
        metrics: Dictionary of metric objects to update (e.g., TopKAccuracy)
    
    Returns:
        Dictionary containing evaluation results
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Reset metrics
    if metrics:
        for metric in metrics.values():
            if hasattr(metric, 'reset'):
                metric.reset()
    
    pbar = tqdm(val_loader, desc="Evaluating", leave=False,
                file=sys.stderr, dynamic_ncols=True, mininterval=0.1)
    with torch.no_grad():
        for xb, yb, _ in pbar:
            xb, yb = xb.to(device), yb.to(device)
            
            out = model(xb)
            loss = loss_fn(out, yb)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update metrics
            if metrics:
                for metric in metrics.values():
                    if hasattr(metric, 'update'):
                        metric.update(out, yb)
            
            # Update progress bar with current loss
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                pbar.set_postfix(loss=f"{avg_loss:.4f}")
    
    results = {'loss': total_loss / num_batches if num_batches > 0 else 0.0}
    
    # Compute metrics
    if metrics:
        for name, metric in metrics.items():
            if hasattr(metric, 'compute'):
                results[name] = metric.compute()
    
    return results


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model, input_shape=(1, 380, 4, 9)):
    """
    Print a summary of the model architecture.
    
    Args:
        model: Neural network model
        input_shape: Shape of the input tensor
    """
    print(f"\n{'='*60}")
    print(f"Model Summary")
    print(f"{'='*60}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Input shape: {input_shape}")
    
    # Try to get output shape
    try:
        device = next(model.parameters()).device
        dummy_input = torch.randn(*input_shape).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Output shape: {tuple(output.shape)}")
    except Exception as e:
        print(f"Could not determine output shape: {e}")
    
    print(f"{'='*60}\n")
