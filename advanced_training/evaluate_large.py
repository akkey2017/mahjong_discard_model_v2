"""
Evaluation script for large mahjong discard prediction models.

This script evaluates trained large models (CoAtNet Large, ResNet Large, ViT Large)
on a validation dataset using multi-ZIP support and provides inference examples.
"""

import argparse
import os
import re
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from tqdm import tqdm

# Ensure repository root is importable when running as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import create_dataloaders  # noqa: E402
from utils import TopKAccuracy, evaluate  # noqa: E402
from advanced_training.large_models import MODEL_FACTORIES  # noqa: E402
from advanced_training.multizip_dataset import MultiZipMahjongDataset  # noqa: E402


# Tile ID to tile string mapping (34-dimensional)
ID_TO_TILE_34 = {
    **{i - 1: f"m{i}" for i in range(1, 10)},
    **{i - 1 + 9: f"p{i}" for i in range(1, 10)},
    **{i - 1 + 18: f"s{i}" for i in range(1, 10)},
    **{i - 1 + 27: f"z{i}" for i in range(1, 8)},
}


def infer_model_type_from_path(model_path):
    """
    Infer model type from the model file path.

    Looks for architecture names (coatnet_large, resnet_large, vit_large) in the
    filename, treating common separators (_, -, .) as word boundaries.

    Args:
        model_path: Path to the model file

    Returns:
        Inferred model type ('coatnet_large', 'resnet_large', 'vit_large') or None
    """
    filename = os.path.basename(model_path).lower()

    # Split on common separators (underscore, hyphen, dot) to get words
    words = re.split(r"[_.-]", filename)

    # Check for combinations indicating large models
    has_large = "large" in words

    if has_large:
        if "vit" in words:
            return "vit_large"
        elif "resnet" in words:
            return "resnet_large"
        elif "coatnet" in words:
            return "coatnet_large"

    # Fallback: check for architecture names even without "large"
    if "vit" in words:
        return "vit_large"
    elif "resnet" in words:
        return "resnet_large"
    elif "coatnet" in words:
        return "coatnet_large"

    return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate large mahjong discard prediction models"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="large_discard_model_coatnet_large.pth",
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=sorted(MODEL_FACTORIES.keys()),
        help="Model architecture type (auto-detected from filename if not specified)",
    )
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="Path(s) to evaluation data ZIP file(s)",
    )
    parser.add_argument(
        "--max-files-per-zip",
        type=int,
        default=200,
        help="Maximum number of game files to load from each ZIP",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto/cuda/cpu)",
    )
    parser.add_argument(
        "--show-demo",
        action="store_true",
        help="Show inference demo on sample data",
    )
    parser.add_argument(
        "--num-demo-samples",
        type=int,
        default=5,
        help="Number of samples to show in demo",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits",
    )

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
    print("\n" + "=" * 60)
    print("Inference Demo")
    print("=" * 60 + "\n")

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
        print(f"Sample {i + 1} (Index: {sample_idx})")
        print(f"Actual discard: {actual_discard}")
        print("Model predictions (Top 5):")

        for j in range(5):
            pred_tile = ID_TO_TILE_34.get(top5_indices[0, j].item(), "Unknown")
            prob = top5_probs[0, j].item()
            marker = "✓" if top5_indices[0, j].item() == yb_sample.item() else " "
            print(f"  {marker} {j + 1}. {pred_tile:<4} ({prob:.2%})")

        print()


def main():
    """Main evaluation function."""
    args = parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Determine model type
    if args.model_type is not None:
        model_type = args.model_type
    else:
        model_type = infer_model_type_from_path(args.model_path)
        if model_type is None:
            model_type = "coatnet_large"  # Default fallback
            print(f"⚠️  Could not infer model type from filename, defaulting to '{model_type}'")
        else:
            print(f"🔍 Auto-detected model type: {model_type}")

    print("\n" + "=" * 60)
    print("Evaluation Configuration")
    print("=" * 60)
    print(f"{'model_path':20s}: {args.model_path}")
    print(f"{'model_type':20s}: {model_type}")
    print(f"{'data':20s}: {args.data}")
    print(f"{'max_files_per_zip':20s}: {args.max_files_per_zip}")
    print(f"{'batch_size':20s}: {args.batch_size}")
    print(f"{'num_workers':20s}: {args.num_workers}")
    print(f"{'device':20s}: {device}")
    print(f"{'show_demo':20s}: {args.show_demo}")
    print(f"{'num_demo_samples':20s}: {args.num_demo_samples}")
    print(f"{'seed':20s}: {args.seed}")
    print("=" * 60 + "\n")

    # Create model
    print(f"Creating {model_type.upper().replace('_', ' ')} model...")
    if model_type not in MODEL_FACTORIES:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(MODEL_FACTORIES.keys())}"
        )
    model = MODEL_FACTORIES[model_type](dropout=0.0)  # No dropout during evaluation

    # Load trained weights
    try:
        state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"✅ Successfully loaded model weights from '{args.model_path}'")
    except FileNotFoundError:
        print(f"❌ Error: Model file '{args.model_path}' not found")
        return
    except RuntimeError as e:
        print(f"❌ Error loading model weights: {e}")
        print("   This may be due to a model architecture mismatch.")
        print(f"   Please verify the --model-type ({model_type}) matches the saved weights.")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    model.to(device)
    model.eval()

    # Load evaluation dataset
    print(f"\n📂 Loading evaluation dataset from:")
    for path in args.data:
        print(f"   - {path}")

    try:
        full_dataset = MultiZipMahjongDataset(
            zip_paths=args.data,
            max_files_per_zip=args.max_files_per_zip,
            verbose=True,
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    stats = full_dataset.get_statistics()
    print(f"Combined samples: {len(full_dataset)}")
    print(f"Per-archive counts: {stats.get('source_counts', {})}")

    discard_dataset = full_dataset.filter_by_action("discard")

    if len(discard_dataset) == 0:
        print("❌ Error: No discard samples found in evaluation dataset!")
        return

    print(f"📊 Loaded {len(discard_dataset)} discard samples")

    # Create data loaders
    _, val_loader = create_dataloaders(
        discard_dataset,
        train_ratio=0.9,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Evaluate model
    print("\n🔍 Evaluating model...")
    loss_fn = nn.CrossEntropyLoss()
    top1_acc = TopKAccuracy(k=1)
    top3_acc = TopKAccuracy(k=3)
    top5_acc = TopKAccuracy(k=5)

    val_metrics = evaluate(
        model,
        val_loader,
        loss_fn,
        device,
        metrics={"top1_acc": top1_acc, "top3_acc": top3_acc, "top5_acc": top5_acc},
    )

    # Display results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Average Loss:     {val_metrics['loss']:.4f}")
    print(
        f"Top-1 Accuracy:   {val_metrics['top1_acc']:.4f} "
        f"({top1_acc.correct}/{top1_acc.total})"
    )
    print(
        f"Top-3 Accuracy:   {val_metrics['top3_acc']:.4f} "
        f"({top3_acc.correct}/{top3_acc.total})"
    )
    print(
        f"Top-5 Accuracy:   {val_metrics['top5_acc']:.4f} "
        f"({top5_acc.correct}/{top5_acc.total})"
    )
    print("=" * 60 + "\n")

    # Show inference demo
    if args.show_demo:
        # Get validation set for demo
        generator = torch.Generator().manual_seed(args.seed)
        train_size = int(len(discard_dataset) * 0.9)
        val_size = len(discard_dataset) - train_size
        _, val_set = random_split(
            discard_dataset, [train_size, val_size], generator=generator
        )

        run_inference_demo(model, val_set, device, args.num_demo_samples)


if __name__ == "__main__":
    main()
