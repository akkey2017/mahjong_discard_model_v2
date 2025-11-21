"""
Training script for mahjong discard prediction models.

This script provides a clean, configurable training pipeline with support for:
- Multiple model architectures (CoAtNet, ResNet, Vision Transformer)
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Gradient clipping
"""

import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import MahjongDataset, create_dataloaders
from models import create_coatnet_model, create_resnet_model, create_vit_model
from utils import (
    TopKAccuracy, EarlyStopping, ModelCheckpoint,
    get_optimizer, get_scheduler, train_one_epoch, evaluate,
    print_model_summary
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train mahjong discard prediction model')
    
    # Data parameters
    parser.add_argument('--data', type=str, default='data2023.zip',
                       help='Path to training data ZIP file')
    parser.add_argument('--max-files', type=int, default=2000,
                       help='Maximum number of game files to load')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                       help='Ratio of data to use for training')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='coatnet',
                       choices=['coatnet', 'resnet', 'vit'],
                       help='Model architecture to use')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
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
    parser.add_argument('--max-grad-norm', type=float, default=None,
                       help='Maximum gradient norm for clipping (None to disable)')
    
    # Regularization
    parser.add_argument('--early-stopping', type=int, default=0,
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
                       help='Output model path (default: discard_model_{model}.pth)')
    parser.add_argument('--save-best', action='store_true',
                       help='Save only the best model based on validation accuracy')
    
    return parser.parse_args()


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
        args.output = f"discard_model_{args.model}.pth"
    
    print("\n" + "="*60)
    print("Training Configuration")
    print("="*60)
    for arg, value in vars(args).items():
        print(f"{arg:20s}: {value}")
    print("="*60 + "\n")
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = MahjongDataset(args.data, max_files=args.max_files)
    print(f"Total samples loaded: {len(full_dataset)}")
    
    stats = full_dataset.get_statistics()
    print(f"Dataset statistics: {stats}")
    
    # Filter for discard actions only
    discard_dataset = full_dataset.filter_by_action('discard')
    print(f"Discard samples: {len(discard_dataset)}")
    
    if len(discard_dataset) == 0:
        print("Error: No discard samples found in dataset!")
        return
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        discard_dataset,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print(f"\nCreating {args.model.upper()} model...")
    if args.model == 'coatnet':
        model = create_coatnet_model(dropout=args.dropout)
    elif args.model == 'resnet':
        model = create_resnet_model(dropout=args.dropout)
    elif args.model == 'vit':
        model = create_vit_model(dropout=args.dropout)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = model.to(device)
    print_model_summary(model)
    
    # Setup training
    loss_fn = nn.CrossEntropyLoss()
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
    print("Starting Training")
    print("="*60 + "\n")
    
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        print("-" * 60)
        
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
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val Top-1:  {val_top1:.4f}")
        print(f"Val Top-3:  {val_top3:.4f}")
        
        # Learning rate scheduling
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_top3)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint
        checkpoint(model, {'top3_acc': val_top3, 'top1_acc': val_top1})
        
        # Track best accuracy
        if val_top3 > best_val_acc:
            best_val_acc = val_top3
        
        # Early stopping
        if early_stopping is not None:
            if early_stopping(val_top3):
                print(f"\nEarly stopping triggered after epoch {epoch+1}")
                break
        
        print()
    
    # Final save if not using save_best
    if not args.save_best:
        torch.save(model.state_dict(), args.output)
        print(f"\nFinal model saved to {args.output}")
    
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"Best validation Top-3 accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {args.output}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
