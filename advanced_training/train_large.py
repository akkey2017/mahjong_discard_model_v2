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
import time

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
    resolve_amp_dtype,
    save_checkpoint,
    train_one_epoch,
    train_one_epoch_multitask,
)
from advanced_training.experiment import ExperimentLogger  # noqa: E402
from advanced_training.large_models import MODEL_FACTORIES, MULTITASK_MODELS  # noqa: E402
from advanced_training.multizip_dataset import MultiZipMahjongDataset  # noqa: E402
from mahjong_ai_features import FEATURE_SCHEMA_VERSION  # noqa: E402


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
    parser.add_argument("--amp-dtype", choices=["auto", "fp16", "bf16"], default="bf16",
                        help="Autocast dtype when --amp is enabled. Blackwell/Ampere+ GPUs "
                             "usually run ViT training most stably with bf16.")
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="If > 0, maintain an EMA copy of the model with this decay.")
    parser.add_argument("--early-stopping", type=int, default=5,
                        help="Early stopping patience (0 to disable).")
    parser.add_argument("--monitor-metric", type=str, default=None,
                        help="Validation metric used for best checkpoint/early stopping. "
                             "Defaults to top1_acc for multitask and top3_acc otherwise.")

    # Multi-task loss balancing
    parser.add_argument("--dapai-weight", type=float, default=1.0)
    parser.add_argument("--riichi-weight", type=float, default=0.5)
    parser.add_argument("--fulou-weight", type=float, default=0.4)
    parser.add_argument("--gang-weight", type=float, default=0.3)
    parser.add_argument("--hule-weight", type=float, default=0.0,
                        help="Keep at 0.0 until hule negatives are available.")

    # System
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-num-workers", type=int, default=2,
                        help="Validation DataLoader workers; kept separate to limit RAM use.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefetch-factor", type=int, default=4,
                        help="DataLoader prefetch factor when num_workers > 0.")
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Keep DataLoader workers alive between epochs.")
    parser.add_argument("--drop-last", action="store_true",
                        help="Drop the final incomplete training batch.")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="Allow TF32 matmul/cuDNN kernels on NVIDIA GPUs.")
    parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Enable cuDNN benchmark for fixed-size inputs.")
    parser.add_argument("--compile", action="store_true",
                        help="Wrap the model with torch.compile after moving it to device.")
    parser.add_argument("--compile-mode",
                        choices=["default", "reduce-overhead", "max-autotune"],
                        default="reduce-overhead")
    parser.add_argument("--profile-batches", type=int, default=0,
                        help="Run only N train batches, log throughput/peak VRAM, then exit.")

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


def _explicit_cli_dests(argv=None):
    """Return argparse destination names explicitly present on the CLI."""
    argv = sys.argv[1:] if argv is None else argv
    explicit = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        name = token.split("=", 1)[0][2:]
        if name.startswith("no-"):
            name = name[3:]
        explicit.add(name.replace("-", "_"))
    return explicit


def _restore_resume_config(args, saved_config, argv=None):
    """Restore saved training options unless the current CLI overrides them.

    Paths and the execution device belong to the current invocation. All other
    training/data options are restored so scheduler construction, dataset
    sampling and AMP behavior match the original run. ``model`` is always
    restored because checkpoint weights cannot be loaded into another model.
    """
    explicit = _explicit_cli_dests(argv)
    invocation_only = {"data", "resume", "run_dir", "run_name", "device"}
    restored = []
    for key, value in saved_config.items():
        if key in invocation_only or not hasattr(args, key):
            continue
        if key != "model" and key in explicit:
            continue
        if getattr(args, key) != value:
            setattr(args, key, value)
            restored.append(key)
    return restored


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
        task_weights = {
            "dapai": args.dapai_weight,
            "riichi": args.riichi_weight,
            "fulou": args.fulou_weight,
            "gang": args.gang_weight,
            "hule": args.hule_weight,
        }
        return loss_fns, task_weights
    return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing), None


def _configure_torch_runtime(args, device):
    """Apply workstation-oriented runtime flags before model construction."""
    if _amp_device_type(device) == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.backends.cudnn.benchmark = args.cudnn_benchmark


def _run_device_preflight(model, device, args, exp):
    """Run a real forward/backward before workers and long training begin."""
    device_type = _amp_device_type(device)
    if device_type == "cuda":
        exp.log(
            f"CUDA preflight: gpu={torch.cuda.get_device_name(device)} "
            f"capability={torch.cuda.get_device_capability(device)} "
            f"torch_arches={torch.cuda.get_arch_list()}"
        )
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    try:
        dummy = torch.randn(1, 380, 4, 9, device=device)
        with torch.amp.autocast(
            device_type=device_type,
            enabled=args.amp,
            dtype=resolve_amp_dtype(device, args.amp_dtype),
        ):
            output = model(dummy)
            if isinstance(output, dict):
                probe = sum(value.float().mean() for value in output.values())
            else:
                probe = output.float().mean()
        probe.backward()
        if device_type == "cuda":
            torch.cuda.synchronize(device)
    except Exception as exc:
        raise RuntimeError(
            f"Device preflight failed on {device}. Check the PyTorch/CUDA build, "
            f"GPU architecture support and AMP dtype. Original error: {exc}"
        ) from exc
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    exp.log("Device preflight: forward/backward OK")


def _compile_model_if_requested(model, args, exp):
    if not args.compile:
        return model
    if not hasattr(torch, "compile"):
        exp.log("WARNING: torch.compile is unavailable in this PyTorch; continuing without it.")
        return model
    mode = None if args.compile_mode == "default" else args.compile_mode
    exp.log(f"Compiling model with torch.compile(mode={args.compile_mode})")
    return torch.compile(model, mode=mode)


def _checkpoint_module(model):
    """Return the original module when torch.compile wraps it."""
    return getattr(model, "_orig_mod", model)


def _profile_training_batches(
    model, train_loader, loss_obj, optimizer, device, args, is_multitask, task_weights, scaler, ema, exp
):
    """Run a bounded streaming benchmark and exit without checkpoints."""
    if args.profile_batches <= 0:
        return

    class _ProfileLoader:
        def __init__(self, loader, limit):
            self.loader = loader
            self.limit = min(limit, len(loader))
            self.batch_count = 0
            self.sample_count = 0
            self.data_seconds = 0.0
            self.first_batch_seconds = None

        def __len__(self):
            return self.limit

        def __iter__(self):
            iterator_start = time.perf_counter()
            iterator = iter(self.loader)
            self.data_seconds += time.perf_counter() - iterator_start
            for _ in range(self.limit):
                wait_start = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    return
                wait_seconds = time.perf_counter() - wait_start
                self.data_seconds += wait_seconds
                if self.first_batch_seconds is None:
                    self.first_batch_seconds = time.perf_counter() - iterator_start
                self.batch_count += 1
                self.sample_count += int(batch[1].numel())
                yield batch

    if _amp_device_type(device) == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    exp.log(
        f"Profile: streaming {args.profile_batches} batches; progress includes DataLoader wait"
    )
    start = time.perf_counter()
    limited_loader = _ProfileLoader(train_loader, args.profile_batches)

    if is_multitask:
        train_loss = train_one_epoch_multitask(
            model, limited_loader, loss_obj, optimizer, device,
            task_weights=task_weights,
            max_grad_norm=args.max_grad_norm,
            scaler=scaler, use_amp=args.amp, amp_dtype=args.amp_dtype,
            accumulation_steps=args.accumulation_steps,
            ema=ema,
        )
    else:
        train_loss = train_one_epoch(
            model, limited_loader, loss_obj, optimizer, device,
            max_grad_norm=args.max_grad_norm,
            scaler=scaler, use_amp=args.amp, amp_dtype=args.amp_dtype,
            accumulation_steps=args.accumulation_steps,
            ema=ema,
        )
    if _amp_device_type(device) == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(time.perf_counter() - start, 1e-9)
    compute_seconds = max(0.0, elapsed - limited_loader.data_seconds)
    exp.log(
        f"Profile: batches={limited_loader.batch_count}, samples={limited_loader.sample_count}, "
        f"samples/sec={limited_loader.sample_count / elapsed:.2f}, loss={train_loss:.4f}"
    )
    exp.log(
        f"Profile timing: first_batch={limited_loader.first_batch_seconds or 0.0:.2f}s, "
        f"data_wait={limited_loader.data_seconds:.2f}s, "
        f"compute_and_transfer_estimate={compute_seconds:.2f}s, total={elapsed:.2f}s"
    )
    if _amp_device_type(device) == "cuda":
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        exp.log(f"Profile: peak allocated VRAM={peak_gib:.2f} GiB")
    raise SystemExit(0)


def main():
    args = parse_args()
    args.feature_schema_version = FEATURE_SCHEMA_VERSION
    exp, resuming = _resolve_run_dir(args)
    if resuming:
        saved_schema = exp.config.get("feature_schema_version")
        if saved_schema != FEATURE_SCHEMA_VERSION:
            raise RuntimeError(
                "This run predates the private-hand/relative-seat feature fix "
                f"(saved={saved_schema!r}, required={FEATURE_SCHEMA_VERSION!r}). "
                "Its weights are incompatible with the corrected encoder; start a new run."
            )
        restored = _restore_resume_config(args, exp.config)
        if restored:
            exp.log(f"Restored saved config: {', '.join(sorted(restored))}")
        if not exp.last_checkpoint.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {exp.last_checkpoint}. "
                "Pass a run directory containing last_model.pth."
            )

    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    _configure_torch_runtime(args, device)
    is_multitask = args.model in MULTITASK_MODELS

    exp.log(f"Run directory: {exp.run_dir}")
    exp.log(f"Using device: {device}")
    exp.log(f"Model: {args.model} (multitask={is_multitask})")
    exp.log(
        f"Runtime: amp={args.amp} amp_dtype={args.amp_dtype} "
        f"tf32={args.tf32} cudnn_benchmark={args.cudnn_benchmark} "
        f"compile={args.compile}"
    )
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
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            drop_last=args.drop_last,
            val_num_workers=args.val_num_workers,
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
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            drop_last=args.drop_last,
            val_num_workers=args.val_num_workers,
        )

    # ---- Model / optimizer / scheduler ----
    model = MODEL_FACTORIES[args.model](dropout=args.dropout).to(device)
    _run_device_preflight(model, device, args, exp)
    model = _compile_model_if_requested(model, args, exp)
    print_model_summary(model, raise_on_error=True)

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

    monitor_metric = args.monitor_metric or ("top1_acc" if is_multitask else "top3_acc")
    checkpoint = ModelCheckpoint(
        str(exp.best_checkpoint),
        monitor=monitor_metric,
        mode="max",
        save_best_only=True,
        model_type=args.model,
        config=vars(args),
    )

    # GradScaler requires the base device type ("cuda"), not "cuda:N".
    amp_dtype = resolve_amp_dtype(device, args.amp_dtype)
    use_scaler = args.amp and _amp_device_type(device) == "cuda" and amp_dtype != torch.bfloat16
    scaler = torch.amp.GradScaler(device=_amp_device_type(device)) if use_scaler else None
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    exp.log(
        f"DataLoader: batch_size={args.batch_size}, accumulation_steps={args.accumulation_steps}, "
        f"effective_batch_size={args.batch_size * args.accumulation_steps}, "
        f"num_workers={args.num_workers}, val_num_workers={args.val_num_workers}, "
        f"prefetch_factor={args.prefetch_factor}, "
        f"persistent_workers={args.persistent_workers}, drop_last={args.drop_last}"
    )
    prefetched_batches = (
        args.num_workers * args.prefetch_factor if args.num_workers > 0 else 0
    )
    exp.log(
        f"DataLoader prefetch capacity: batches={prefetched_batches}, "
        f"samples={prefetched_batches * args.batch_size}"
    )
    if is_multitask:
        exp.log(f"Task weights: {task_weights}")

    # ---- Resume ----
    start_epoch = 0
    if resuming and exp.last_checkpoint.exists():
        payload = load_checkpoint(exp.last_checkpoint, map_location=device)
        _checkpoint_module(model).load_state_dict(payload["model_state"])
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
                _checkpoint_module(ema.ema).load_state_dict(ema_state)
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

    _profile_training_batches(
        model, train_loader, loss_obj, optimizer, device, args,
        is_multitask, task_weights, scaler, ema, exp,
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
                scaler=scaler, use_amp=args.amp, amp_dtype=args.amp_dtype,
                accumulation_steps=args.accumulation_steps,
                ema=ema,
            )
            val_metrics = evaluate_multitask(
                ema.ema if ema else model, val_loader, loss_obj, device,
                task_weights=task_weights, use_amp=args.amp, amp_dtype=args.amp_dtype,
            )
        else:
            train_loss = train_one_epoch(
                model, train_loader, loss_obj, optimizer, device,
                max_grad_norm=args.max_grad_norm,
                scaler=scaler, use_amp=args.amp, amp_dtype=args.amp_dtype,
                accumulation_steps=args.accumulation_steps,
                ema=ema,
            )
            val_metrics = evaluate(
                ema.ema if ema else model, val_loader, loss_obj, device,
                metrics={"top1_acc": top1_acc, "top3_acc": top3_acc},
                use_amp=args.amp, amp_dtype=args.amp_dtype,
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
            _checkpoint_module(ema.ema if ema else model),
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
            last_extra["ema_state"] = _checkpoint_module(ema.ema).state_dict()
        save_checkpoint(
            str(exp.last_checkpoint),
            _checkpoint_module(model),
            model_type=args.model,
            config=vars(args),
            extra=last_extra,
        )

        monitored = val_metrics.get(monitor_metric)
        if monitored is None:
            available = ", ".join(sorted(val_metrics))
            exp.log(
                f"WARNING: monitor metric '{monitor_metric}' was not produced; "
                f"available metrics: {available}. Best checkpoint and early "
                "stopping will not update for this epoch."
            )
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
