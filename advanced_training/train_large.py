"""
Training entrypoint for enlarged CNN/Transformer/CoAtNet models with multi-ZIP support.
"""

import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import create_dataloaders  # noqa: E402
from utils import (  # noqa: E402
    EarlyStopping,
    ModelCheckpoint,
    TopKAccuracy,
    evaluate,
    get_optimizer,
    get_scheduler,
    print_model_summary,
    train_one_epoch,
)
from advanced_training.large_models import MODEL_FACTORIES  # noqa: E402
from advanced_training.multizip_dataset import MultiZipMahjongDataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train enlarged Mahjong discard models with multiple ZIP archives."
    )

    # Data parameters
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="One or more ZIP files containing game logs.",
    )
    parser.add_argument(
        "--max-files-per-zip",
        type=int,
        default=4000,
        help="Maximum files to read from each ZIP archive.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Ratio of data used for training.",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_FACTORIES.keys()),
        default="coatnet_large",
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate.",
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, default=12, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=96, help="Batch size.")
    parser.add_argument("--lr", type=float, default=8e-5, help="Learning rate.")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-2, help="Weight decay strength."
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw", "sgd"],
        default="adamw",
        help="Optimizer type.",
    )
    parser.add_argument(
        "--scheduler",
        choices=["cosine", "plateau", "none"],
        default="cosine",
        help="Learning rate scheduler.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping threshold (None to disable).",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=5,
        help="Early stopping patience (0 to disable).",
    )

    # System parameters
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto/cuda/cpu).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="Only save the best checkpoint based on top-3 accuracy.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if args.output is None:
        args.output = f"large_discard_model_{args.model}.pth"

    tqdm.write(f"Using device: {device}")
    tqdm.write("Loading datasets from:")
    for path in args.data:
        tqdm.write(f"  - {path}")

    dataset = MultiZipMahjongDataset(
        zip_paths=args.data,
        max_files_per_zip=args.max_files_per_zip,
        verbose=True,
    )
    stats = dataset.get_statistics()
    tqdm.write(f"Combined samples: {len(dataset)}")
    tqdm.write(f"Per-archive counts: {stats.get('source_counts', {})}")

    discard_dataset = dataset.filter_by_action("discard")
    if len(discard_dataset) == 0:
        raise RuntimeError("No discard samples found across provided archives.")

    train_loader, val_loader = create_dataloaders(
        discard_dataset,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        seed=args.seed,
    )

    model = MODEL_FACTORIES[args.model](dropout=args.dropout).to(device)
    print_model_summary(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args.optimizer, args.lr, args.weight_decay)

    scheduler = None
    if args.scheduler != "none":
        scheduler_kwargs = {"T_max": args.epochs} if args.scheduler == "cosine" else {}
        scheduler = get_scheduler(optimizer, args.scheduler, **scheduler_kwargs)

    early_stopping = None
    if args.early_stopping > 0:
        early_stopping = EarlyStopping(patience=args.early_stopping, mode="max")
        tqdm.write(f"Early stopping enabled with patience={args.early_stopping}")

    checkpoint = ModelCheckpoint(
        args.output,
        monitor="top3_acc",
        mode="max",
        save_best_only=args.save_best,
    )

    top1_acc = TopKAccuracy(k=1)
    top3_acc = TopKAccuracy(k=3)

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        tqdm.write(f"\nEpoch {epoch + 1}/{args.epochs}")
        tqdm.write("-" * 60)

        train_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, args.max_grad_norm
        )

        val_metrics = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            metrics={"top1_acc": top1_acc, "top3_acc": top3_acc},
        )

        val_loss = val_metrics["loss"]
        val_top1 = val_metrics["top1_acc"]
        val_top3 = val_metrics["top3_acc"]

        tqdm.write(f"Train Loss: {train_loss:.4f}")
        tqdm.write(f"Val Loss:   {val_loss:.4f}")
        tqdm.write(f"Val Top-1:  {val_top1:.4f}")
        tqdm.write(f"Val Top-3:  {val_top3:.4f}")

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_top3)
            else:
                scheduler.step()
            tqdm.write(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        checkpoint(model, {"top3_acc": val_top3, "top1_acc": val_top1})
        best_val_acc = max(best_val_acc, val_top3)

        if early_stopping is not None and early_stopping(val_top3):
            tqdm.write(f"\nEarly stopping triggered after epoch {epoch + 1}")
            break

    if not args.save_best:
        torch.save(model.state_dict(), args.output)
        tqdm.write(f"\nFinal model saved to {args.output}")


if __name__ == "__main__":
    main()
