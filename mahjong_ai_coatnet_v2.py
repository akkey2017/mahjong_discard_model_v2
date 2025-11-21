"""
Original CoAtNet training script (refactored).

This is the refactored version of the original training script.
For more features and flexibility, use train.py instead.
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import MahjongDataset, create_dataloaders
from models import create_coatnet_model
from utils import TopKAccuracy, train_one_epoch, evaluate


def main():
    """Main training function."""
    # Configuration
    zip_path = "data2023.zip"
    max_files = 2000
    batch_size = 64
    num_workers = 2
    epochs = 10
    lr = 1e-4
    weight_decay = 1e-2
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = MahjongDataset(zip_path, max_files=max_files)
    print(f"Total samples found: {len(full_dataset)}")
    
    discard_dataset = full_dataset.filter_by_action('discard')
    print(f"Discard samples: {len(discard_dataset)}")
    
    if len(discard_dataset) == 0:
        print("No discard samples found!")
        return
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        discard_dataset,
        train_ratio=0.9,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Create model
    model = create_coatnet_model(dropout=0.0)
    model = model.to(device)
    
    # Training setup
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Metrics
    top1_acc = TopKAccuracy(k=1)
    top3_acc = TopKAccuracy(k=3)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        
        # Validate
        val_metrics = evaluate(
            model, val_loader, loss_fn, device,
            metrics={'top1_acc': top1_acc, 'top3_acc': top3_acc}
        )
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Top-1 Acc: {val_metrics['top1_acc']:.4f} | "
              f"Val Top-3 Acc: {val_metrics['top3_acc']:.4f}")
    
    # Save model
    output_path = "discard_model_coatnet_v2_2000.pth"
    torch.save(model.state_dict(), output_path)
    print(f"\nSaved trained model to {output_path}")


if __name__ == '__main__':
    main()

