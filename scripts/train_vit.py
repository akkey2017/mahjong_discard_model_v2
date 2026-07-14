#!/usr/bin/env python3
"""Train the ViT-only unified multi-task stack on prepared shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mahjong_ai.data import (  # noqa: E402
    TARGET_SCHEMA_VERSION,
    NegativeSamplingConfig,
    StreamingMultiTaskDataset,
    TaskSamplingPolicy,
    build_multitask_dataloader,
)
from mahjong_ai.models import VIT_PRESETS, create_vit, vit_config  # noqa: E402
from mahjong_ai.state import FEATURE_SCHEMA_VERSION  # noqa: E402
from mahjong_ai.training import (  # noqa: E402
    StepTrainer,
    TrainingConfig,
    load_legacy_vit_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--model", choices=tuple(VIT_PRESETS), default="vit_large")
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--drop-path", type=float)
    parser.add_argument("--max-steps", type=int, default=300_000)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--validate-every", type=int, default=5_000)
    parser.add_argument("--validation-steps", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--samples-per-virtual-epoch", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validation-workers", type=int, default=2)
    parser.add_argument("--train-years", nargs="+", type=int)
    parser.add_argument("--validation-years", nargs="+", type=int)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--shuffle-buffer-rounds", type=int, default=8192)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--profile-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument(
        "--include-fulou-negatives", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--riichi-negative-keep", type=float, default=0.25)
    parser.add_argument("--fulou-negative-ratio", type=float, default=4.0)
    parser.add_argument("--gang-negative-keep", type=float, default=1.0)
    parser.add_argument("--dapai-weight", type=float, default=1.0)
    parser.add_argument("--riichi-weight", type=float, default=0.5)
    parser.add_argument("--fulou-weight", type=float, default=0.4)
    parser.add_argument("--gang-weight", type=float, default=0.3)
    parser.add_argument("--hule-weight", type=float, default=0.0)
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_metadata(device: torch.device) -> dict:
    gpu_name = None
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
    return {
        "git_commit": git_commit(),
        "pytorch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": gpu_name,
    }


def main() -> int:
    args = parse_args()
    if args.resume and args.init_checkpoint:
        raise SystemExit("--resume and --init-checkpoint are mutually exclusive")
    manifest = args.data_manifest.resolve()
    manifest_bytes = manifest.read_bytes()
    raw_manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    dataset_selection = {
        "train_split": "train",
        "train_years": sorted(args.train_years) if args.train_years else None,
        "validation_split": "validation",
        "validation_years": (
            sorted(args.validation_years) if args.validation_years else None
        ),
    }
    if args.train_years or args.validation_years:
        selection_bytes = json.dumps(
            dataset_selection, sort_keys=True, separators=(",", ":")
        ).encode()
        manifest_hash = hashlib.sha256(manifest_bytes + selection_bytes).hexdigest()
    else:
        manifest_hash = raw_manifest_hash
    device = choose_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    overrides = {}
    if args.dropout is not None:
        overrides["dropout"] = args.dropout
    if args.drop_path is not None:
        overrides["drop_path"] = args.drop_path
    model_config = vit_config(args.model, **overrides)
    model = create_vit(model_config).to(device)
    legacy_report = None
    if args.init_checkpoint:
        legacy_report = load_legacy_vit_weights(args.init_checkpoint, model, map_location=device)
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is unavailable")
        mode = None if args.compile_mode == "default" else args.compile_mode
        model = torch.compile(model, mode=mode)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    sampling = NegativeSamplingConfig(
        seed=args.seed,
        policies={
            "riichi": TaskSamplingPolicy(keep_probability=args.riichi_negative_keep),
            "fulou": TaskSamplingPolicy(
                max_negative_per_positive=args.fulou_negative_ratio
            ),
            "gang": TaskSamplingPolicy(keep_probability=args.gang_negative_keep),
        },
    )
    train_dataset = StreamingMultiTaskDataset(
        manifest,
        split="train",
        seed=args.seed,
        shuffle=True,
        shuffle_buffer_rounds=args.shuffle_buffer_rounds,
        negative_sampling=sampling,
        include_fulou_negatives=args.include_fulou_negatives,
        encode_features=True,
        years=args.train_years,
    )
    validation_dataset = StreamingMultiTaskDataset(
        manifest,
        split="validation",
        seed=args.seed,
        shuffle=False,
        shuffle_buffer_rounds=0,
        include_fulou_negatives=args.include_fulou_negatives,
        encode_features=True,
        years=args.validation_years,
    )
    train_loader = build_multitask_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = build_multitask_dataloader(
        validation_dataset,
        batch_size=args.batch_size,
        num_workers=args.validation_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        drop_last=False,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    training_config = TrainingConfig(
        max_steps=args.max_steps,
        warmup_steps=min(args.warmup_steps, args.max_steps),
        min_lr_ratio=args.min_lr_ratio,
        validate_every=args.validate_every,
        validation_steps=args.validation_steps,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        samples_per_virtual_epoch=args.samples_per_virtual_epoch,
        accumulation_steps=args.accumulation_steps,
        grad_clip_norm=args.grad_clip,
        amp_dtype=args.amp_dtype,
        tf32=args.tf32,
        cudnn_benchmark=args.cudnn_benchmark,
        ema_decay=args.ema_decay,
        compile_model=args.compile,
        compile_mode=args.compile_mode,
        profile_steps=args.profile_steps,
    )
    if args.resume:
        run_dir = args.run_dir or args.resume.resolve().parent
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = args.run_dir or Path("runs") / f"{args.model}_{stamp}"
    run_dir = run_dir.resolve()
    task_weights = {
        "dapai": args.dapai_weight,
        "riichi": args.riichi_weight,
        "fulou": args.fulou_weight,
        "gang": args.gang_weight,
        "hule": args.hule_weight,
    }
    metadata = {
        **environment_metadata(device),
        "dataset_manifest": str(manifest),
        "dataset_manifest_sha256": manifest_hash,
        "dataset_manifest_file_sha256": raw_manifest_hash,
        "dataset_selection": dataset_selection,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "target_schema_version": TARGET_SCHEMA_VERSION,
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "optimizer": {
            "name": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "task_weights": task_weights,
        "target_generation": train_dataset.target_manifest(),
        "seed": args.seed,
        "legacy_initialization": legacy_report,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    trainer = StepTrainer(
        model=model,
        model_config=model_config,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        config=training_config,
        run_dir=run_dir,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        target_schema_version=TARGET_SCHEMA_VERSION,
        dataset_manifest_sha256=manifest_hash,
        task_weights=task_weights,
        run_metadata=metadata,
    )
    if args.resume:
        trainer.resume(args.resume)
    summary = trainer.train()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
