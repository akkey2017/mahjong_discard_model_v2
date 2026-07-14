"""Step-based trainer for unified multi-task Snapshot ViT."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Mapping

import torch

from mahjong_ai.data.multitask import MultiTaskMetrics, masked_multitask_loss
from mahjong_ai.models import ViTConfig

from .checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    MODEL_FAMILY,
    atomic_torch_save,
    checkpoint_module,
    load_payload,
    validate_training_checkpoint,
)
from .config import TrainingConfig
from .ema import ModelEMA
from .profiler import TrainingProfiler
from .scheduler import StepWarmupCosineScheduler


def _checkpoint_primitive(value):
    """Normalize metadata without weakening weights_only checkpoint loading."""

    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _checkpoint_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_checkpoint_primitive(item) for item in value]
    raise TypeError(f"unsupported checkpoint metadata type: {type(value).__name__}")


def configure_accelerator(tf32: bool, cudnn_benchmark: bool) -> None:
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    if tf32:
        torch.set_float32_matmul_precision("high")


class StepTrainer:
    def __init__(
        self,
        *,
        model,
        model_config: ViTConfig,
        optimizer,
        train_loader,
        validation_loader,
        device: torch.device,
        config: TrainingConfig,
        run_dir: Path,
        feature_schema_version: str,
        target_schema_version: str,
        dataset_manifest_sha256: str,
        task_weights: Mapping[str, float] | None = None,
        run_metadata: Mapping[str, object] | None = None,
    ):
        self.model = model
        self.model_config = model_config
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.device = torch.device(device)
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.feature_schema_version = feature_schema_version
        self.target_schema_version = target_schema_version
        self.dataset_manifest_sha256 = dataset_manifest_sha256
        self.task_weights = dict(task_weights or {})
        self.run_metadata = _checkpoint_primitive(dict(run_metadata or {}))
        self.scheduler = StepWarmupCosineScheduler(
            optimizer,
            max_steps=config.max_steps,
            warmup_steps=config.warmup_steps,
            min_lr_ratio=config.min_lr_ratio,
        )
        self.ema = ModelEMA(model, config.ema_decay) if config.ema_decay else None
        scaler_enabled = config.amp_dtype == "fp16" and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        self.profiler = TrainingProfiler(config.profile_steps)
        self.global_step = 0
        self.samples_seen = 0
        self.data_epoch = 0
        self.resume_sample_offsets: dict[int, tuple[int, int]] = {}
        self.last_metrics: dict[str, float | int] = {}
        self.metrics_path = self.run_dir / "metrics.jsonl"
        configure_accelerator(config.tf32, config.cudnn_benchmark)

    @property
    def amp_enabled(self) -> bool:
        return self.config.amp_dtype != "fp32" and self.device.type in ("cuda", "cpu")

    @property
    def amp_dtype(self):
        return {"bf16": torch.bfloat16, "fp16": torch.float16}[self.config.amp_dtype]

    def _sync_profile(self) -> None:
        if self.profiler.active and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _move_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            batch["features"].to(self.device, non_blocking=True),
            batch["labels"].to(self.device, non_blocking=True),
            batch["masks"].to(self.device, non_blocking=True),
        )

    def _record_consumed(self, batch: dict) -> None:
        for worker, sequence, sample_index in zip(
            batch["worker_ids"].tolist(),
            batch["worker_sequences"].tolist(),
            batch["sample_indices"].tolist(),
        ):
            position = (int(sequence), int(sample_index) + 1)
            worker = int(worker)
            if position > self.resume_sample_offsets.get(worker, (-1, -1)):
                self.resume_sample_offsets[worker] = position

    def _reset_stream_for_next_epoch(self):
        self.data_epoch += 1
        self.resume_sample_offsets = {}
        dataset = self.train_loader.dataset
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(self.data_epoch)
        if hasattr(dataset, "set_resume_sample_offsets"):
            dataset.set_resume_sample_offsets({})

    def _log(self, values: dict) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(values, ensure_ascii=False) + "\n")
        print(json.dumps(values, ensure_ascii=False), flush=True)

    @torch.no_grad()
    def evaluate(self) -> dict[str, float | int]:
        if self.validation_loader is None or self.config.validation_steps == 0:
            return {}
        model = self.ema.model if self.ema is not None else checkpoint_module(self.model)
        was_training = model.training
        model.eval()
        metrics = MultiTaskMetrics(self.task_weights)
        loss_sum = 0.0
        batches = 0
        for batch in self.validation_loader:
            features, labels, masks = self._move_batch(batch)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype if self.config.amp_dtype != "fp32" else None,
                enabled=self.amp_enabled,
            ):
                logits = model(features)
                loss, per_task = masked_multitask_loss(
                    logits, labels, masks, task_weights=self.task_weights
                )
            metrics.update(logits, labels, masks, per_task)
            loss_sum += float(loss)
            batches += 1
            if batches >= self.config.validation_steps:
                break
        if was_training:
            model.train()
        result = metrics.compute()
        result["loss"] = loss_sum / batches if batches else 0.0
        result["batches"] = batches
        return result

    def _rng_state(self) -> dict:
        result = {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
        }
        if torch.cuda.is_available():
            result["cuda"] = torch.cuda.get_rng_state_all()
        return result

    def _restore_rng(self, state: dict) -> None:
        if "torch" in state:
            torch.set_rng_state(state["torch"].cpu())
        if "python" in state:
            random.setstate(state["python"])
        if "cuda" in state and torch.cuda.is_available():
            # A checkpoint loaded with ``map_location=cuda`` also maps these
            # ByteTensors to CUDA, while set_rng_state_all requires CPU state.
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])

    def checkpoint_payload(self) -> dict:
        train_state = {
            "global_step": self.global_step,
            "samples_seen": self.samples_seen,
            "data_epoch": self.data_epoch,
            "resume_sample_offsets": {
                worker: list(position)
                for worker, position in self.resume_sample_offsets.items()
            },
            "last_metrics": self.last_metrics,
        }
        payload = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_family": MODEL_FAMILY,
            "model_config": self.model_config.to_dict(),
            "feature_schema_version": self.feature_schema_version,
            "target_schema_version": self.target_schema_version,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "training_config": self.config.to_dict(),
            "task_weights": self.task_weights,
            "run_metadata": self.run_metadata,
            "model_state": checkpoint_module(self.model).state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "ema_state": self.ema.model.state_dict() if self.ema is not None else None,
            "train_state": train_state,
            "rng_state": self._rng_state(),
        }
        return payload

    def save_checkpoint(self, *, numbered: bool = True) -> Path:
        payload = self.checkpoint_payload()
        last = self.run_dir / "last.pt"
        atomic_torch_save(payload, last)
        if numbered:
            atomic_torch_save(payload, self.run_dir / f"step_{self.global_step:08d}.pt")
        return last

    def resume(self, path: Path) -> None:
        payload = load_payload(path, map_location=self.device)
        validate_training_checkpoint(
            payload,
            model_config=self.model_config,
            feature_schema_version=self.feature_schema_version,
            target_schema_version=self.target_schema_version,
            dataset_manifest_sha256=self.dataset_manifest_sha256,
        )
        saved_training_config = payload.get("training_config", {})
        for key in ("max_steps", "warmup_steps", "min_lr_ratio", "accumulation_steps"):
            if saved_training_config.get(key) != getattr(self.config, key):
                raise ValueError(f"training config {key} differs from resume checkpoint")
        checkpoint_module(self.model).load_state_dict(payload["model_state"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.scheduler.load_state_dict(payload["scheduler_state"])
        if payload.get("scaler_state"):
            self.scaler.load_state_dict(payload["scaler_state"])
        if self.ema is not None:
            if payload.get("ema_state") is None:
                raise ValueError("EMA is enabled but checkpoint has no EMA state")
            self.ema.model.load_state_dict(payload["ema_state"], strict=True)
        state = payload["train_state"]
        self.global_step = int(state["global_step"])
        self.samples_seen = int(state["samples_seen"])
        self.data_epoch = int(state["data_epoch"])
        self.last_metrics = dict(state.get("last_metrics", {}))
        self.resume_sample_offsets = {
            int(worker): (int(position[0]), int(position[1]))
            for worker, position in state.get("resume_sample_offsets", {}).items()
        }
        dataset = self.train_loader.dataset
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(self.data_epoch)
        if hasattr(dataset, "set_resume_sample_offsets"):
            dataset.set_resume_sample_offsets(self.resume_sample_offsets)
        self._restore_rng(payload.get("rng_state", {}))

    def train(self) -> dict:
        summary_path = self.run_dir / "summary.json"
        if self.global_step >= self.config.max_steps and summary_path.exists():
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                existing.get("global_step") == self.global_step
                and existing.get("samples_seen") == self.samples_seen
            ):
                return existing
        self.model.train()
        iterator = iter(self.train_loader)
        interval_loss = 0.0
        interval_started = time.perf_counter()
        while self.global_step < self.config.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            step_samples = 0
            data_wait = transfer = forward = backward = 0.0
            for _ in range(self.config.accumulation_steps):
                wait_started = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    self._reset_stream_for_next_epoch()
                    iterator = iter(self.train_loader)
                    batch = next(iterator)
                data_wait += time.perf_counter() - wait_started
                self._record_consumed(batch)

                stage = time.perf_counter()
                features, labels, masks = self._move_batch(batch)
                self._sync_profile()
                transfer += time.perf_counter() - stage

                stage = time.perf_counter()
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype if self.config.amp_dtype != "fp32" else None,
                    enabled=self.amp_enabled,
                ):
                    logits = self.model(features)
                    loss, _ = masked_multitask_loss(
                        logits, labels, masks, task_weights=self.task_weights
                    )
                    scaled_loss = loss / self.config.accumulation_steps
                self._sync_profile()
                forward += time.perf_counter() - stage

                stage = time.perf_counter()
                self.scaler.scale(scaled_loss).backward()
                self._sync_profile()
                backward += time.perf_counter() - stage
                step_loss += float(loss.detach())
                step_samples += int(features.shape[0])

            stage = time.perf_counter()
            if self.config.grad_clip_norm:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    checkpoint_module(self.model).parameters(), self.config.grad_clip_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.ema is not None:
                self.ema.update(self.model)
            self.global_step += 1
            self.samples_seen += step_samples
            self.scheduler.step(self.global_step)
            self._sync_profile()
            optimizer_seconds = time.perf_counter() - stage
            self.profiler.add(
                data_wait=data_wait,
                transfer=transfer,
                forward=forward,
                backward=backward,
                optimizer=optimizer_seconds,
            )

            step_loss /= self.config.accumulation_steps
            interval_loss += step_loss
            if self.config.log_every and self.global_step % self.config.log_every == 0:
                elapsed = time.perf_counter() - interval_started
                values = {
                    "type": "train",
                    "step": self.global_step,
                    "virtual_epoch": self.samples_seen / self.config.samples_per_virtual_epoch,
                    "samples_seen": self.samples_seen,
                    "loss": interval_loss / self.config.log_every,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "samples_per_second": (
                        step_samples * self.config.log_every / elapsed if elapsed else 0.0
                    ),
                }
                self._log(values)
                interval_loss = 0.0
                interval_started = time.perf_counter()

            if self.config.validate_every and self.global_step % self.config.validate_every == 0:
                self.last_metrics = self.evaluate()
                self._log({"type": "validation", "step": self.global_step, **self.last_metrics})
                self.model.train()

            if self.config.checkpoint_every and self.global_step % self.config.checkpoint_every == 0:
                self.save_checkpoint(numbered=True)

        self.save_checkpoint(numbered=False)
        summary = {
            "global_step": self.global_step,
            "samples_seen": self.samples_seen,
            "virtual_epoch": self.samples_seen / self.config.samples_per_virtual_epoch,
            "last_metrics": self.last_metrics,
            "profile": self.profiler.summary(),
            "last_checkpoint": str(self.run_dir / "last.pt"),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return summary
