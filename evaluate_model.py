"""
Model evaluation script for mahjong discard prediction.

This script evaluates a trained model on a validation dataset and provides
inference examples.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataset import MahjongDataset, create_dataloaders
from models import create_coatnet_model, create_resnet_model, create_vit_model
from utils import TopKAccuracy, evaluate



# Tile ID to tile string mapping (34-dimensional)
ID_TO_TILE_34 = {
    **{i-1: f"m{i}" for i in range(1, 10)},
    **{i-1+9: f"p{i}" for i in range(1, 10)},
    **{i-1+18: f"s{i}" for i in range(1, 10)},
    **{i-1+27: f"z{i}" for i in range(1, 8)}
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate mahjong discard prediction model')
    
    parser.add_argument('--model-path', type=str, default='discard_model_coatnet.pth',
                       help='Path to trained model weights')
    parser.add_argument('--model-type', type=str, default='coatnet',
                       choices=['coatnet', 'resnet', 'vit'],
                       help='Model architecture type')
    parser.add_argument('--data', type=str, default='data2022.zip',
                       help='Path to evaluation data ZIP file')
    parser.add_argument('--max-files', type=int, default=200,
                       help='Maximum number of game files to load')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cuda/cpu)')
    parser.add_argument('--show-demo', action='store_true',
                       help='Show inference demo on sample data')
    parser.add_argument('--num-demo-samples', type=int, default=5,
                       help='Number of samples to show in demo')
    
    return parser.parse_args()


def run_inference_demo(model, val_set, device, num_samples=5):
    """
    Run inference demo on sample data.
    
    Args:
        model: Trained model
        val_set: Validation dataset
        device: Device to run on
        num_samples: Number of samples to demonstrate
    """
    print("\n" + "="*60)
    print("Inference Demo")
    print("="*60 + "\n")
    
    model.eval()
    
    for i in range(min(num_samples, len(val_set))):
        sample_idx = torch.randint(0, len(val_set), (1,)).item()
        xb_sample, yb_sample, _ = val_set[sample_idx]
        
        with torch.no_grad():
            # Add batch dimension and predict
            out_sample = model(xb_sample.unsqueeze(0).to(device))
            
            # Convert to probabilities
            probabilities = F.softmax(out_sample, dim=1)
            top5_probs, top5_indices = torch.topk(probabilities, 5)
        
        actual_discard = ID_TO_TILE_34.get(yb_sample.item(), "Unknown")
        print(f"Sample {i+1} (Index: {sample_idx})")
        print(f"Actual discard: {actual_discard}")
        print("Model predictions (Top 5):")
        
        for j in range(5):
            pred_tile = ID_TO_TILE_34.get(top5_indices[0, j].item(), "Unknown")
            prob = top5_probs[0, j].item()
            marker = "✓" if top5_indices[0, j].item() == yb_sample.item() else " "
            print(f"  {marker} {j+1}. {pred_tile:<4} ({prob:.2%})")
        
        print()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")
    
    print("\n" + "="*60)
    print("Evaluation Configuration")
    print("="*60)
    for arg, value in vars(args).items():
        print(f"{arg:20s}: {value}")
    print("="*60 + "\n")
    
    # Create model
    print(f"Creating {args.model_type.upper()} model...")
    if args.model_type == 'coatnet':
        model = create_coatnet_model(dropout=0.0)  # No dropout during evaluation
    elif args.model_type == 'resnet':
        model = create_resnet_model(dropout=0.0)
    elif args.model_type == 'vit':
        model = create_vit_model(dropout=0.0)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Load trained weights
    try:
        state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded model weights from '{args.model_path}'")
    except FileNotFoundError:
        print(f"Error: Model file '{args.model_path}' not found")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.to(device)
    model.eval()
    
    # Load evaluation dataset
    print(f"\nLoading evaluation dataset from '{args.data}'...")
    full_dataset = MahjongDataset(args.data, max_files=args.max_files)
    discard_dataset = full_dataset.filter_by_action('discard')
    
    if len(discard_dataset) == 0:
        print("Error: No discard samples found in evaluation dataset!")
        return
    
    print(f"Loaded {len(discard_dataset)} discard samples")
    
    # Create data loaders
    _, val_loader = create_dataloaders(
        discard_dataset,
        train_ratio=0.9,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=42
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    loss_fn = nn.CrossEntropyLoss()
    top1_acc = TopKAccuracy(k=1)
    top3_acc = TopKAccuracy(k=3)
    top5_acc = TopKAccuracy(k=5)
    
    val_metrics = evaluate(
        model, val_loader, loss_fn, device,
        metrics={'top1_acc': top1_acc, 'top3_acc': top3_acc, 'top5_acc': top5_acc}
    )
    
    # Display results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Average Loss:     {val_metrics['loss']:.4f}")
    print(f"Top-1 Accuracy:   {val_metrics['top1_acc']:.4f} ({top1_acc.correct}/{top1_acc.total})")
    print(f"Top-3 Accuracy:   {val_metrics['top3_acc']:.4f} ({top3_acc.correct}/{top3_acc.total})")
    print(f"Top-5 Accuracy:   {val_metrics['top5_acc']:.4f} ({top5_acc.correct}/{top5_acc.total})")
    print("="*60 + "\n")
    
    # Show inference demo
    if args.show_demo:
        # Get validation set for demo
        from torch.utils.data import random_split
        generator = torch.Generator().manual_seed(42)
        train_size = int(len(discard_dataset) * 0.9)
        val_size = len(discard_dataset) - train_size
        _, val_set = random_split(discard_dataset, [train_size, val_size], generator=generator)
        
        run_inference_demo(model, val_set, device, args.num_demo_samples)


if __name__ == '__main__':
    main()

