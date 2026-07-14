"""
Evaluation script for large mahjong discard prediction models.

Loads a checkpoint saved by :mod:`advanced_training.train_large` (rich dict
with architecture info), reports per-tile accuracy + a confusion summary, and
optionally runs an inference demo.
"""

import argparse
import csv
import os
import re
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure repository root is importable when running as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import (  # noqa: E402
    multitask_collate,
)
from utils import (  # noqa: E402
    TopKAccuracy,
    evaluate,
    evaluate_multitask,
    load_checkpoint,
)
from advanced_training.large_models import MODEL_FACTORIES, MULTITASK_MODELS  # noqa: E402
from advanced_training.multizip_dataset import MultiZipMahjongDataset  # noqa: E402
from mahjong_ai_features import FEATURE_SCHEMA_VERSION  # noqa: E402


ID_TO_TILE_34 = {
    **{i - 1: f"m{i}" for i in range(1, 10)},
    **{i - 1 + 9: f"p{i}" for i in range(1, 10)},
    **{i - 1 + 18: f"s{i}" for i in range(1, 10)},
    **{i - 1 + 27: f"z{i}" for i in range(1, 8)},
}


def infer_model_type_from_path(model_path):
    """Fallback heuristic: guess model type from filename words."""
    filename = os.path.basename(model_path).lower()
    words = re.split(r"[_.-]", filename)
    has_large = "large" in words
    has_multitask = "multitask" in words

    def _base():
        if "vit" in words:
            return "vit"
        if "resnet" in words:
            return "resnet"
        if "coatnet" in words:
            return "coatnet"
        return None

    base = _base()
    if base is None:
        return None
    if has_multitask:
        return f"{base}_multitask_large"
    if has_large:
        return f"{base}_large"
    return f"{base}_large"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate large mahjong models.")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to a checkpoint (run_dir/best_model.pth or legacy .pth).")
    parser.add_argument("--model-type", type=str, default=None,
                        choices=sorted(MODEL_FACTORIES.keys()),
                        help="Override architecture (otherwise auto-detected from checkpoint).")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--max-files-per-zip", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--show-demo", action="store_true")
    parser.add_argument("--num-demo-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Write per-tile accuracy table to this CSV file.")
    parser.add_argument(
        "--fulou-negatives",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include fulou pass negatives. Defaults to the checkpoint config.",
    )
    return parser.parse_args()


def _resolve_model_type(args, payload):
    """Pick the model type from (--model-type) > checkpoint > filename heuristic."""
    if args.model_type:
        return args.model_type
    saved = payload.get("model_type")
    if saved and saved in MODEL_FACTORIES:
        print(f"Using model_type from checkpoint: {saved}")
        return saved
    guess = infer_model_type_from_path(args.model_path)
    if guess in MODEL_FACTORIES:
        print(f"Guessed model_type from filename: {guess}")
        return guess
    raise ValueError(
        "Could not determine model type. Pass --model-type explicitly."
    )


def _run_demo(model, val_set, device, num_samples):
    print("\n" + "=" * 60)
    print("Inference Demo")
    print("=" * 60 + "\n")
    model.eval()
    for i in range(min(num_samples, len(val_set))):
        sample_idx = torch.randint(0, len(val_set), (1,)).item()
        xb_sample, yb_sample, action = val_set[sample_idx]
        if action != "dapai":
            continue
        with torch.no_grad():
            out = model(xb_sample.unsqueeze(0).to(device))
            if isinstance(out, dict):
                out = out["dapai"]
            probs = F.softmax(out, dim=1)
            top5_probs, top5_idx = torch.topk(probs, 5)
        actual = ID_TO_TILE_34.get(yb_sample.item() if torch.is_tensor(yb_sample) else yb_sample, "?")
        print(f"Sample {i + 1} actual: {actual}")
        for j in range(5):
            t = ID_TO_TILE_34.get(top5_idx[0, j].item(), "?")
            p = top5_probs[0, j].item()
            marker = "*" if top5_idx[0, j].item() == (yb_sample.item() if torch.is_tensor(yb_sample) else yb_sample) else " "
            print(f"  {marker} {j + 1}. {t:<4} ({p:.2%})")
        print()


@torch.no_grad()
def _compute_per_tile_stats(model, val_loader, device, num_classes=34, is_multitask=False):
    """Return (per_tile_correct, per_tile_total, confusion[NxN])."""
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    model.eval()
    pbar = tqdm(val_loader, desc="Per-tile", leave=False, file=sys.stderr,
                dynamic_ncols=True, mininterval=0.1)
    for batch in pbar:
        if is_multitask:
            xb, yb, actions = batch
            if isinstance(actions, torch.Tensor):
                actions = [str(a) for a in actions.tolist()]
            # restrict to dapai (discard) samples
            idx = [i for i, a in enumerate(actions) if a == "dapai"]
            if not idx:
                continue
            xb = xb[idx].to(device)
            yb = yb[idx].to(device)
        else:
            xb, yb, _ = batch
            xb = xb.to(device)
            yb = yb.to(device)

        logits = model(xb)
        if isinstance(logits, dict):
            logits = logits["dapai"]
        preds = logits.argmax(dim=-1)
        for gt, pr in zip(yb.tolist(), preds.tolist()):
            if 0 <= gt < num_classes and 0 <= pr < num_classes:
                confusion[gt, pr] += 1

    per_tile_total = confusion.sum(dim=1)
    per_tile_correct = confusion.diag()
    return per_tile_correct, per_tile_total, confusion


def _print_per_tile_table(per_correct, per_total, output_csv=None):
    print("\n" + "=" * 60)
    print("Per-tile accuracy (discard head)")
    print("=" * 60)
    print(f"{'tile':<6} {'acc':>8} {'correct':>10} {'total':>8}")
    rows = []
    for i in range(len(per_correct)):
        total = int(per_total[i])
        correct = int(per_correct[i])
        acc = correct / total if total else 0.0
        tile = ID_TO_TILE_34[i]
        rows.append({"tile": tile, "accuracy": f"{acc:.4f}",
                     "correct": correct, "total": total})
        print(f"{tile:<6} {acc:>8.4f} {correct:>10d} {total:>8d}")
    # Category rollup
    def _range_acc(ids):
        c = sum(int(per_correct[i]) for i in ids)
        t = sum(int(per_total[i]) for i in ids)
        return c / t if t else 0.0, c, t
    m_acc, m_c, m_t = _range_acc(range(0, 9))
    p_acc, p_c, p_t = _range_acc(range(9, 18))
    s_acc, s_c, s_t = _range_acc(range(18, 27))
    z_acc, z_c, z_t = _range_acc(range(27, 34))
    print("-" * 40)
    print(f"{'manzu':<6} {m_acc:>8.4f} {m_c:>10d} {m_t:>8d}")
    print(f"{'pinzu':<6} {p_acc:>8.4f} {p_c:>10d} {p_t:>8d}")
    print(f"{'souzu':<6} {s_acc:>8.4f} {s_c:>10d} {s_t:>8d}")
    print(f"{'honor':<6} {z_acc:>8.4f} {z_c:>10d} {z_t:>8d}")

    if output_csv:
        with open(output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["tile", "accuracy", "correct", "total"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nPer-tile CSV written to {output_csv}")


def _print_confusion_summary(confusion, top_k_errors=15):
    """Print the top confused (gt, pred) pairs."""
    print("\n" + "=" * 60)
    print(f"Top-{top_k_errors} confused (ground truth -> predicted) pairs")
    print("=" * 60)
    n = confusion.size(0)
    pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            c = int(confusion[i, j])
            if c > 0:
                pairs.append((c, i, j))
    pairs.sort(reverse=True)
    for count, gt, pr in pairs[:top_k_errors]:
        print(f"  {ID_TO_TILE_34[gt]:<4} -> {ID_TO_TILE_34[pr]:<4} : {count}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"Using device: {device}")

    # ---- Load checkpoint ----
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    payload = load_checkpoint(args.model_path, map_location=device)
    saved_config = payload.get("config", {})
    saved_schema = saved_config.get("feature_schema_version")
    if saved_schema != FEATURE_SCHEMA_VERSION:
        raise RuntimeError(
            "Checkpoint feature schema is incompatible with the corrected encoder "
            f"(saved={saved_schema!r}, required={FEATURE_SCHEMA_VERSION!r}). "
            "Retrain the model before evaluation."
        )
    model_type = _resolve_model_type(args, payload)
    is_multitask = model_type in MULTITASK_MODELS

    print(f"Model type: {model_type} (multitask={is_multitask})")
    model = MODEL_FACTORIES[model_type](dropout=0.0)
    try:
        model.load_state_dict(payload["model_state"])
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load weights into {model_type}. "
            f"Verify that the checkpoint matches this architecture.\n{e}"
        )
    model.to(device).eval()

    # ---- Dataset ----
    fulou_negatives = args.fulou_negatives
    if fulou_negatives is None:
        fulou_negatives = bool(saved_config.get("fulou_negatives", False))
    full_dataset = MultiZipMahjongDataset(
        zip_paths=args.data,
        max_files_per_zip=args.max_files_per_zip,
        verbose=True,
        collect_all_actions=is_multitask,
        include_fulou_negatives=is_multitask and fulou_negatives,
    )
    stats = full_dataset.get_statistics()
    print(f"Combined samples: {len(full_dataset)}")
    print(f"Per-archive counts: {stats.get('source_counts', {})}")
    print(f"Action counts: {stats.get('action_counts', {})}")

    evaluation_dataset = full_dataset if is_multitask else full_dataset.filter_by_action("dapai")
    if len(evaluation_dataset) == 0:
        raise RuntimeError("No evaluation samples found in the supplied data.")
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": str(device).split(":", 1)[0] == "cuda",
    }
    if is_multitask:
        loader_kwargs["collate_fn"] = multitask_collate
    val_loader = DataLoader(evaluation_dataset, **loader_kwargs)
    print(f"Evaluating all {len(evaluation_dataset)} samples")

    # ---- Aggregate metrics ----
    loss_fn = nn.CrossEntropyLoss()
    top1 = TopKAccuracy(k=1)
    top3 = TopKAccuracy(k=3)
    top5 = TopKAccuracy(k=5)

    if is_multitask:
        task_weights = {
            "dapai": float(saved_config.get("dapai_weight", 1.0)),
            "riichi": float(saved_config.get("riichi_weight", 0.5)),
            "fulou": float(saved_config.get("fulou_weight", 0.4)),
            "gang": float(saved_config.get("gang_weight", 0.3)),
            "hule": float(saved_config.get("hule_weight", 0.0)),
        }
        loss_fns = {
            k: nn.CrossEntropyLoss()
            for k in ["dapai", "riichi", "fulou", "gang", "hule"]
        }
        loss_fns["_default"] = nn.CrossEntropyLoss()
        results = evaluate_multitask(
            model, val_loader, loss_fns, device, task_weights=task_weights
        )
        print("\n" + "=" * 60)
        print("Multi-task results")
        print("=" * 60)
        for k, v in sorted(results.items()):
            if isinstance(v, float):
                print(f"{k:<20s}: {v:.4f}")
            else:
                print(f"{k:<20s}: {v}")
    else:
        results = evaluate(
            model, val_loader, loss_fn, device,
            metrics={"top1_acc": top1, "top3_acc": top3, "top5_acc": top5},
        )
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        print(f"Loss:           {results['loss']:.4f}")
        print(f"Top-1 Accuracy: {results['top1_acc']:.4f} "
              f"({top1.correct}/{top1.total})")
        print(f"Top-3 Accuracy: {results['top3_acc']:.4f} "
              f"({top3.correct}/{top3.total})")
        print(f"Top-5 Accuracy: {results['top5_acc']:.4f} "
              f"({top5.correct}/{top5.total})")

    # ---- Per-tile + confusion (always computed for discard head) ----
    per_correct, per_total, confusion = _compute_per_tile_stats(
        model, val_loader, device, is_multitask=is_multitask,
    )
    _print_per_tile_table(per_correct, per_total, output_csv=args.output_csv)
    _print_confusion_summary(confusion)

    # ---- Optional demo ----
    if args.show_demo:
        demo_set = full_dataset.filter_by_action("dapai")
        _run_demo(model, demo_set, device, args.num_demo_samples)


if __name__ == "__main__":
    main()
