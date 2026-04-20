"""
Training entrypoint for enlarged CNN/Transformer/CoAtNet models with multi-ZIP support.

Outputs go into a self-contained per-run directory::

    runs/<model>_<timestamp>/
        config.json
        metrics.csv
        training.log
        best_model.pth
        last_model.pth

Multi-task variants ('*_multitask_large') train the ``dapai``, ``riichi``,
``fulou``, ``gang``, and ``hule`` heads jointly, using task-masked losses.
``fulou`` covers chi/pon/daiminkan calls; ``gang`` covers ankan/kakan.
"""

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import create_dataloaders, create_multitask_dataloaders  # noqa: E402
from utils import (  # noqa: E402
    EarlyStopping,
    ModelCheckpoint,
    ModelEMA,
    TopKAccuracy,
    _amp_device_type,
    evaluate,
    evaluate_multitask,
    get_optimizer,
    get_scheduler,
    load_checkpoint,
    print_model_summary,
    save_checkpoint,
    train_one_epoch,
    train_one_epoch_multitask,
)
from advanced_training.experiment import ExperimentLogger  # noqa: E402
from advanced_training.large_models import MODEL_FACTORIES, MULTITASK_MODELS  # noqa: E402
from advanced_training.multizip_dataset import MultiZipMahjongDataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train enlarged Mahjong discard models with multiple ZIP archives."
    )

    # Data
    parser.add_argument("--data", nargs="+", required=True,
                        help="One or more ZIP files containing game logs.")
    parser.add_argument("--max-files-per-zip", type=int, default=4000,
                        help="Maximum files to read from each ZIP archive.")
    parser.add_argument("--train-ratio", type=float, default=0.9,
                        help="Ratio of data used for training.")
    parser.add_argument("--split-by-game", action="store_true",
                        help="Split train/val by game file rather than sample.")
    parser.add_argument("--fulou-negatives", action="store_true",
                        help="Synthesize fulou-head pass (label 0) negatives for "
                             "players who could have called chi/pon/daiminkan on "
                             "a discard but chose not to.")

    # Model
    parser.add_argument("--model", choices=sorted(MODEL_FACTORIES.keys()),
                        default="coatnet_large", help="Model architecture to train.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")

    # Training
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adamw")
    parser.add_argument("--scheduler", choices=["cosine", "warmup_cosine", "plateau", "none"],
                        default="warmup_cosine")
    parser.add_argument("--warmup-epochs", type=int, default=3,
                        help="Warmup epochs when scheduler=warmup_cosine.")
    parser.add_argument(
        "--max-grad-norm",
        type=lambda value: None if value.lower() == "none" else float(value),
        default=1.0,
        help="Gradient clipping threshold (pass 'none' to disable).",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.05,
                        help="Label smoothing for CrossEntropyLoss (discard head).")
    parser.add_argument("--accumulation-steps", type=int, default=1,
                        help="Gradient accumulation steps for larger effective batch.")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed precision (torch.amp.autocast).")
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="If > 0, maintain an EMA copy of the model with this decay.")
    parser.add_argument("--early-stopping", type=int, default=5,
                        help="Early stopping patience (0 to disable).")

    # System
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    # Output / experiment management
    parser.add_argument("--run-dir", type=str, default="runs",
                        help="Base directory that will contain per-run subdirectories.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Override the auto-generated run directory name.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to an existing run directory or last_model.pth to resume from.")

    return parser.parse_args()


def _resolve_run_dir(args):
    """Return an ExperimentLogger, either fresh or attached to a resume dir."""
    if args.resume:
        resume_path = Path(args.resume)
        # Accept either a run dir or a *.pth inside it
        if resume_path.is_file():
            resume_dir = resume_path.parent
        else:
            resume_dir = resume_path
        return ExperimentLogger.from_existing(resume_dir), True

    return (
        ExperimentLogger.create(
            base_dir=args.run_dir,
            model_type=args.model,
            config=vars(args),
            run_name=args.run_name,
        ),
        False,
    )


def _build_loss(args, is_multitask):
    if is_multitask:
        # Per-head losses using 牌譜形式 head names; dapai uses label smoothing.
        loss_fns = {
            "dapai":  nn.CrossEntropyLoss(label_smoothing=args.label_smoothing),
            "riichi": nn.CrossEntropyLoss(),
            "fulou":  nn.CrossEntropyLoss(),
            "gang":   nn.CrossEntropyLoss(),
            "hule":   nn.CrossEntropyLoss(),
            "_default": nn.CrossEntropyLoss(),
        }
        # ``hule`` is weighted 0 because the dataset only supplies positive
        # (label=1) samples — see ``_extract_samples_from_kyoku`` in
        # ``dataset.py``.  Training a 2-class head on positives alone would
        # collapse the decision to "always win"; negatives require
        # ron/tenpai detection that is out of scope for this change.  Once
        # negatives are added the weight should be restored (e.g. 0.5).
        task_weights = {"dapai": 1.0, "riichi": 0.5, "fulou": 0.4,
                        "gang": 0.3, "hule": 0.0}
        return loss_fns, task_weights
    return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing), None


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    is_multitask = args.model in MULTITASK_MODELS

    exp, resuming = _resolve_run_dir(args)
    if resuming:
        # Override args.model with the resumed config if available
        resumed_model = exp.config.get("model")
        if resumed_model and resumed_model != args.model:
            exp.log(f"Overriding --model with resumed value: {resumed_model}")
            args.model = resumed_model
            is_multitask = args.model in MULTITASK_MODELS

    exp.log(f"Run directory: {exp.run_dir}")
    exp.log(f"Using device: {device}")
    exp.log(f"Model: {args.model} (multitask={is_multitask})")
    exp.log("Loading datasets:")
    for path in args.data:
        exp.log(f"  - {path}")

    # ---- Data ----
    dataset = MultiZipMahjongDataset(
        zip_paths=args.data,
        max_files_per_zip=args.max_files_per_zip,
        verbose=True,
        collect_all_actions=is_multitask,
        include_fulou_negatives=is_multitask and args.fulou_negatives,
    )
    stats = dataset.get_statistics()
    exp.log(f"Combined samples: {len(dataset)}")
    exp.log(f"Per-archive counts: {stats.get('source_counts', {})}")
    exp.log(f"Action counts: {stats.get('action_counts', {})}")

    if is_multitask:
        train_loader, val_loader = create_multitask_dataloaders(
            dataset,
            train_ratio=args.train_ratio,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            seed=args.seed,
            split_by_game=args.split_by_game,
        )
    else:
        discard_dataset = dataset.filter_by_action("dapai")
        if len(discard_dataset) == 0:
            raise RuntimeError("No dapai samples found across provided archives.")
        train_loader, val_loader = create_dataloaders(
            discard_dataset,
            train_ratio=args.train_ratio,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            seed=args.seed,
            split_by_game=args.split_by_game,
        )

    # ---- Model / optimizer / scheduler ----
    model = MODEL_FACTORIES[args.model](dropout=args.dropout).to(device)
    print_model_summary(model)

    loss_obj, task_weights = _build_loss(args, is_multitask)
    optimizer = get_optimizer(model, args.optimizer, args.lr, args.weight_decay)

    scheduler = None
    if args.scheduler != "none":
        sched_kwargs = {"T_max": args.epochs, "warmup_epochs": args.warmup_epochs}
        scheduler = get_scheduler(optimizer, args.scheduler, **sched_kwargs)

    early_stopping = None
    if args.early_stopping > 0:
        early_stopping = EarlyStopping(patience=args.early_stopping, mode="max")
        exp.log(f"Early stopping enabled with patience={args.early_stopping}")

    monitor_metric = "top1_acc" if is_multitask else "top3_acc"
    checkpoint = ModelCheckpoint(
        str(exp.best_checkpoint),
        monitor=monitor_metric,
        mode="max",
        save_best_only=True,
        model_type=args.model,
        config=vars(args),
    )

    # GradScaler requires the base device type ("cuda"), not "cuda:N".
    scaler = torch.amp.GradScaler(device=_amp_device_type(device)) if args.amp else None
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    # ---- Resume ----
    start_epoch = 0
    if resuming and exp.last_checkpoint.exists():
        payload = load_checkpoint(exp.last_checkpoint, map_location=device)
        model.load_state_dict(payload["model_state"])
        start_epoch = int(payload.get("extra", {}).get("epoch", 0))

        # Restore the rest of the training state so the resumed run actually
        # continues from the same optimization trajectory (LR schedule,
        # optimizer moments, AMP scaler, EMA shadow weights).
        extra = payload.get("extra", {})
        restored_parts = ["model"]
        missing_parts = []

        optimizer_state = extra.get("optimizer_state")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            restored_parts.append("optimizer")
        else:
            missing_parts.append("optimizer")

        scheduler_state = extra.get("scheduler_state")
        if scheduler is not None:
            if scheduler_state is not None:
                scheduler.load_state_dict(scheduler_state)
                restored_parts.append("scheduler")
            else:
                missing_parts.append("scheduler")

        scaler_state = extra.get("scaler_state")
        if scaler is not None:
            if scaler_state is not None:
                scaler.load_state_dict(scaler_state)
                restored_parts.append("scaler")
            else:
                missing_parts.append("scaler")

        ema_state = extra.get("ema_state")
        if ema is not None:
            if ema_state is not None:
                ema.ema.load_state_dict(ema_state)
                restored_parts.append("ema")
            else:
                missing_parts.append("ema")

        exp.log(
            f"Resumed from {exp.last_checkpoint} (epoch {start_epoch}); "
            f"restored state: {', '.join(restored_parts)}"
        )
        if missing_parts:
            exp.log(
                "WARNING: Resume checkpoint is missing training-state "
                f"components ({', '.join(missing_parts)}). Those components "
                "will restart from scratch, so LR schedule, optimizer "
                "moments, AMP scaler, or EMA may not continue exactly from "
                "the previous run."
            )

        # Restore ModelCheckpoint.best_score so the first post-resume epoch
        # doesn't unconditionally overwrite best_model.pth.
        if exp.best_checkpoint.exists():
            best_payload = load_checkpoint(exp.best_checkpoint, map_location=device)
            best_metrics = best_payload.get("extra", {}).get("metrics", {})
            if monitor_metric in best_metrics:
                checkpoint.best_score = best_metrics[monitor_metric]
                exp.log(
                    f"Restored best {monitor_metric}={checkpoint.best_score:.4f} "
                    f"from {exp.best_checkpoint}"
                )
            else:
                exp.log(
                    f"Best checkpoint {exp.best_checkpoint} does not contain "
                    f"metric '{monitor_metric}'; best_score not restored."
                )

    # ---- Metrics objects (single-task) ----
    top1_acc = TopKAccuracy(k=1)
    top3_acc = TopKAccuracy(k=3)

    # ---- Training loop ----
    for epoch in range(start_epoch, args.epochs):
        exp.log(f"\nEpoch {epoch + 1}/{args.epochs}")
        exp.log("-" * 60)

        if is_multitask:
            train_loss = train_one_epoch_multitask(
                model, train_loader, loss_obj, optimizer, device,
                task_weights=task_weights,
                max_grad_norm=args.max_grad_norm,
                scaler=scaler, use_amp=args.amp,
                accumulation_steps=args.accumulation_steps,
                ema=ema,
            )
            val_metrics = evaluate_multitask(
                ema.ema if ema else model, val_loader, loss_obj, device,
                task_weights=task_weights, use_amp=args.amp,
            )
        else:
            train_loss = train_one_epoch(
                model, train_loader, loss_obj, optimizer, device,
                max_grad_norm=args.max_grad_norm,
                scaler=scaler, use_amp=args.amp,
                accumulation_steps=args.accumulation_steps,
                ema=ema,
            )
            val_metrics = evaluate(
                ema.ema if ema else model, val_loader, loss_obj, device,
                metrics={"top1_acc": top1_acc, "top3_acc": top3_acc},
                use_amp=args.amp,
            )

        row = {"epoch": epoch + 1, "train_loss": f"{train_loss:.6f}"}
        for k, v in val_metrics.items():
            row[f"val_{k}"] = f"{v:.6f}" if isinstance(v, float) else v
        row["lr"] = f"{optimizer.param_groups[0]['lr']:.8f}"
        exp.log_metrics(row)

        exp.log(f"Train Loss: {train_loss:.4f}")
        for k, v in val_metrics.items():
            if isinstance(v, float):
                exp.log(f"Val {k}: {v:.4f}")

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_metrics.get(monitor_metric, train_loss))
            else:
                scheduler.step()
            exp.log(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best (weights + metrics only; don't leak optimizer state into
        # best_model.pth since "best" is an evaluation artifact, not a resume
        # target). When EMA is active, "best" reflects the EMA-smoothed
        # weights that were actually evaluated.
        checkpoint(
            ema.ema if ema else model,
            val_metrics,
            extra={"epoch": epoch + 1, "metrics": val_metrics},
        )
        # Always save last — include the full *training* state so --resume
        # can continue from the same optimization trajectory. Critically,
        # the main model_state must be the training weights (not the EMA
        # shadow), otherwise restored optimizer/scheduler/scaler state would
        # no longer correspond to the model parameters they were tracking.
        last_extra = {
            "epoch": epoch + 1,
            "metrics": val_metrics,
            "optimizer_state": optimizer.state_dict(),
        }
        if scheduler is not None:
            last_extra["scheduler_state"] = scheduler.state_dict()
        if scaler is not None:
            last_extra["scaler_state"] = scaler.state_dict()
        if ema is not None:
            # EMA shadow weights are stored separately so the main payload
            # stays the training model.
            last_extra["ema_state"] = ema.ema.state_dict()
        save_checkpoint(
            str(exp.last_checkpoint),
            model,
            model_type=args.model,
            config=vars(args),
            extra=last_extra,
        )

        monitored = val_metrics.get(monitor_metric)
        if early_stopping is not None and monitored is not None and early_stopping(monitored):
            exp.log(f"\nEarly stopping triggered after epoch {epoch + 1}")
            break

    exp.log(f"\nFinal best model: {exp.best_checkpoint}")
    exp.log(f"Last model:       {exp.last_checkpoint}")

    # Dump a summary.json for convenience
    summary_path = exp.run_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "model_type": args.model,
        "best_checkpoint": str(exp.best_checkpoint),
        "last_checkpoint": str(exp.last_checkpoint),
        "monitor_metric": monitor_metric,
        "config": vars(args),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
