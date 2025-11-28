"""
Training script for sequential mahjong discard prediction models.

This script provides a training pipeline for sequence-based models that learn
the flow of entire mahjong games, rather than making single-step predictions.

Supports:
- LSTM-based sequential models
- Transformer-based sequential models
- Learning rate scheduling
- Early stopping
- Model checkpointing
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from tqdm import tqdm

from sequence_dataset import MahjongSequenceDataset, create_sequence_dataloaders
from sequence_models import create_lstm_model, create_transformer_model

# Import utilities from parent module
from utils import (
    TopKAccuracy, EarlyStopping, ModelCheckpoint,
    get_optimizer, get_scheduler, count_parameters
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train sequential mahjong discard prediction model'
    )
    
    # Data parameters
    parser.add_argument('--data', type=str, default='data2023.zip',
                       help='Path to training data ZIP file')
    parser.add_argument('--max-files', type=int, default=2000,
                       help='Maximum number of game files to load')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                       help='Ratio of data to use for training')
    parser.add_argument('--max-seq-len', type=int, default=30,
                       help='Maximum sequence length')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='lstm',
                       choices=['lstm', 'transformer'],
                       help='Model architecture to use')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension size')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of layers (LSTM layers or Transformer layers)')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads (Transformer only)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--bidirectional', action='store_true',
                       help='Use bidirectional LSTM (LSTM only)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-2,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'plateau', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping')
    
    # Regularization
    parser.add_argument('--early-stopping', type=int, default=5,
                       help='Early stopping patience (0 to disable)')
    
    # System parameters
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Output parameters
    parser.add_argument('--output', type=str, default=None,
                       help='Output model path')
    parser.add_argument('--save-best', action='store_true',
                       help='Save only the best model')
    
    return parser.parse_args()


def train_one_epoch(model, train_loader, loss_fn, optimizer, device, max_grad_norm=None):
    """
    Train the model for one epoch.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to train on
        max_grad_norm: Maximum gradient norm for clipping
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=True, 
                file=sys.stderr, dynamic_ncols=True, mininterval=0.1, unit="batch")
    
    for states, labels, lengths in pbar:
        states = states.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(states, lengths)  # (batch, seq_len, num_classes)
        
        # Reshape for loss calculation
        batch_size, seq_len, num_classes = logits.shape
        logits_flat = logits.view(-1, num_classes)  # (batch * seq_len, num_classes)
        labels_flat = labels.view(-1)  # (batch * seq_len,)
        
        # Calculate loss (ignore_index=-100 for padding)
        loss = loss_fn(logits_flat, labels_flat)
        loss.backward()
        
        # Gradient clipping
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
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
        metrics: Dictionary of metric objects
    
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
    
    pbar = tqdm(val_loader, desc="Evaluating", leave=True,
                file=sys.stderr, dynamic_ncols=True, mininterval=0.1, unit="batch")
    
    with torch.no_grad():
        for states, labels, lengths in pbar:
            states = states.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            
            # Forward pass
            logits = model(states, lengths)
            
            # Reshape for loss calculation
            batch_size, seq_len, num_classes = logits.shape
            logits_flat = logits.view(-1, num_classes)
            labels_flat = labels.view(-1)
            
            # Calculate loss
            loss = loss_fn(logits_flat, labels_flat)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update metrics (only for non-padding positions)
            if metrics:
                valid_mask = labels_flat != -100
                if valid_mask.sum() > 0:
                    valid_logits = logits_flat[valid_mask]
                    valid_labels = labels_flat[valid_mask]
                    for metric in metrics.values():
                        if hasattr(metric, 'update'):
                            metric.update(valid_logits, valid_labels)
            
            # Update progress bar
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


def print_model_summary(model, input_shape=(1, 10, 380, 4, 9)):
    """Print a summary of the model."""
    print(f"\n{'='*60}")
    print(f"Model Summary")
    print(f"{'='*60}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Input shape: {input_shape} (batch, seq_len, channels, H, W)")
    
    try:
        device = next(model.parameters()).device
        dummy_input = torch.randn(*input_shape).to(device)
        lengths = torch.tensor([input_shape[1]]).to(device)
        with torch.no_grad():
            output = model(dummy_input, lengths)
        print(f"Output shape: {tuple(output.shape)} (batch, seq_len, num_classes)")
    except Exception as e:
        print(f"Could not determine output shape: {e}")
    
    print(f"{'='*60}\n")


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Set output path
    if args.output is None:
        args.output = f"sequential_discard_model_{args.model}.pth"
    
    print("\n" + "="*60)
    print("Sequential Model Training Configuration")
    print("="*60)
    for arg, value in vars(args).items():
        print(f"{arg:20s}: {value}")
    print("="*60 + "\n")
    
    # Load dataset
    print("Loading sequence dataset...")
    dataset = MahjongSequenceDataset(
        args.data, 
        max_files=args.max_files,
        max_seq_len=args.max_seq_len
    )
    print(f"Total sequences loaded: {len(dataset)}")
    
    stats = dataset.get_statistics()
    print(f"Dataset statistics: {stats}")
    
    if len(dataset) == 0:
        print("Error: No sequences found in dataset!")
        return
    
    # Create data loaders
    train_loader, val_loader = create_sequence_dataloaders(
        dataset,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print(f"\nCreating {args.model.upper()} sequential model...")
    if args.model == 'lstm':
        model = create_lstm_model(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional
        )
    elif args.model == 'transformer':
        model = create_transformer_model(
            d_model=args.hidden_dim,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
            max_seq_len=args.max_seq_len
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = model.to(device)
    print_model_summary(model)
    
    # Setup training
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = get_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    
    scheduler = None
    if args.scheduler != 'none':
        scheduler_kwargs = {'T_max': args.epochs} if args.scheduler == 'cosine' else {}
        scheduler = get_scheduler(optimizer, args.scheduler, **scheduler_kwargs)
    
    # Setup callbacks
    early_stopping = None
    if args.early_stopping > 0:
        early_stopping = EarlyStopping(patience=args.early_stopping, mode='max')
        print(f"Early stopping enabled with patience={args.early_stopping}")
    
    checkpoint = ModelCheckpoint(
        args.output,
        monitor='top3_acc',
        mode='max',
        save_best_only=args.save_best
    )
    
    # Metrics
    top1_acc = TopKAccuracy(k=1)
    top3_acc = TopKAccuracy(k=3)
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Sequential Model Training")
    print("="*60 + "\n")
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        sys.stdout.flush()
        sys.stderr.flush()
        tqdm.write(f"Epoch {epoch+1}/{args.epochs}")
        tqdm.write("-" * 60)
        
        # Training
        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, args.max_grad_norm
        )
        
        # Validation
        val_metrics = evaluate(
            model, val_loader, loss_fn, device,
            metrics={'top1_acc': top1_acc, 'top3_acc': top3_acc}
        )
        
        val_loss = val_metrics['loss']
        val_top1 = val_metrics['top1_acc']
        val_top3 = val_metrics['top3_acc']
        
        # Print results
        tqdm.write(f"Train Loss: {train_loss:.4f}")
        tqdm.write(f"Val Loss:   {val_loss:.4f}")
        tqdm.write(f"Val Top-1:  {val_top1:.4f}")
        tqdm.write(f"Val Top-3:  {val_top3:.4f}")
        
        # Learning rate scheduling
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_top3)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            tqdm.write(f"Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint
        checkpoint(model, {'top3_acc': val_top3, 'top1_acc': val_top1})
        
        # Track best accuracy
        if val_top3 > best_val_acc:
            best_val_acc = val_top3
        
        # Early stopping
        if early_stopping is not None:
            if early_stopping(val_top3):
                tqdm.write(f"\nEarly stopping triggered after epoch {epoch+1}")
                break
        
        tqdm.write("")
    
    # Final save if not using save_best
    if not args.save_best:
        torch.save(model.state_dict(), args.output)
        tqdm.write(f"\nFinal model saved to {args.output}")
    
    tqdm.write("\n" + "="*60)
    tqdm.write("Training Complete")
    tqdm.write("="*60)
    tqdm.write(f"Best validation Top-3 accuracy: {best_val_acc:.4f}")
    tqdm.write(f"Model saved to: {args.output}")
    tqdm.write("="*60 + "\n")


if __name__ == '__main__':
    main()
