#!/usr/bin/env python3
"""Evaluate Snapshot ViT on fixed test and temporal holdout cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mahjong_ai.data import (  # noqa: E402
    TARGET_SCHEMA_VERSION,
    StreamingMultiTaskDataset,
    build_multitask_dataloader,
)
from mahjong_ai.evaluation import EvaluationAccumulator, summarize_task_metrics  # noqa: E402
from mahjong_ai.models import ViTConfig, create_vit  # noqa: E402
from mahjong_ai.state import FEATURE_SCHEMA_VERSION  # noqa: E402
from mahjong_ai.training.checkpoint import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
    MODEL_FAMILY,
    CheckpointCompatibilityError,
    load_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="test")
    parser.add_argument("--years", nargs="+", type=int)
    parser.add_argument("--training-years", nargs="+", type=int, default=list(range(2014, 2023)))
    parser.add_argument("--holdout-year", type=int, default=2023)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--calibration-bins", type=int, default=15)
    parser.add_argument("--max-errors", type=int, default=100)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_model(path: Path, device: torch.device, weights: str):
    payload = load_payload(path, map_location="cpu")
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_family": MODEL_FAMILY,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "target_schema_version": TARGET_SCHEMA_VERSION,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise CheckpointCompatibilityError(
                f"{key} mismatch: checkpoint={payload.get(key)!r}, expected={value!r}"
            )
    config = ViTConfig.from_dict(payload["model_config"])
    model = create_vit(config)
    state = payload.get("ema_state") if weights == "ema" else payload.get("model_state")
    if state is None:
        raise CheckpointCompatibilityError(f"checkpoint has no {weights}_state")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), payload, config


@torch.inference_mode()
def evaluate_cohort(args, model, device, *, name: str, split: str, years: list[int]) -> dict:
    dataset = StreamingMultiTaskDataset(
        args.data_manifest,
        split=split,
        years=years,
        seed=42,
        shuffle=False,
        shuffle_buffer_rounds=0,
        include_fulou_negatives=True,
        encode_features=True,
    )
    loader = build_multitask_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        drop_last=False,
        generator=torch.Generator().manual_seed(42),
    )
    overall = EvaluationAccumulator(
        calibration_bins=args.calibration_bins, max_errors=args.max_errors
    )
    per_year = {
        year: EvaluationAccumulator(
            calibration_bins=args.calibration_bins, max_errors=args.max_errors
        )
        for year in years
    }
    started = time.perf_counter()
    samples = 0
    batches = 0
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if device.type == "cuda" else None,
            enabled=device.type == "cuda",
        ):
            logits = model(features)
        overall.update(logits, labels, masks, batch["sample_ids"])
        batch_years = batch["years"]
        for year in years:
            active = batch_years == year
            if not active.any():
                continue
            positions = active.nonzero(as_tuple=False).flatten().tolist()
            per_year[year].update(
                {task: values[active.to(device)] for task, values in logits.items()},
                labels[active.to(device)],
                masks[active.to(device)],
                [batch["sample_ids"][position] for position in positions],
            )
        samples += int(features.shape[0])
        batches += 1
        if args.max_batches is not None and batches >= args.max_batches:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    tasks = overall.compute()
    by_year = {}
    for year, metrics in per_year.items():
        year_tasks = metrics.compute()
        by_year[str(year)] = {
            "summary": summarize_task_metrics(year_tasks),
            "tasks": year_tasks,
        }
    return {
        "name": name,
        "split": split,
        "years": years,
        "batches": batches,
        "samples": samples,
        "elapsed_seconds": elapsed,
        "samples_per_second": samples / elapsed if elapsed else 0.0,
        "summary": summarize_task_metrics(tasks),
        "tasks": tasks,
        "by_year": by_year,
    }


def main() -> int:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model, payload, config = load_model(args.checkpoint.resolve(), device, args.weights)
    if args.suite:
        cohorts = [
            ("fixed_test", "test", sorted(args.training_years)),
            ("temporal_holdout", "all", [args.holdout_year]),
        ]
    else:
        if not args.years:
            raise SystemExit("--years is required unless --suite is used")
        cohorts = [("custom", args.split, sorted(args.years))]
    results = [
        evaluate_cohort(args, model, device, name=name, split=split, years=years)
        for name, split, years in cohorts
    ]
    report = {
        "schema_version": "vit-evaluation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "manifest": str(args.data_manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.data_manifest.read_bytes()).hexdigest(),
        "weights": args.weights,
        "model_config": config.to_dict(),
        "task_weights": payload.get("task_weights", {}),
        "device": str(device),
        "cohorts": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "cohorts": [
            {"name": result["name"], **result["summary"]} for result in results
        ],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
