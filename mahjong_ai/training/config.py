"""Step-based ViT training configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrainingConfig:
    max_steps: int
    warmup_steps: int = 10_000
    min_lr_ratio: float = 0.01
    validate_every: int = 5_000
    validation_steps: int = 100
    checkpoint_every: int = 5_000
    log_every: int = 100
    samples_per_virtual_epoch: int = 1_000_000
    accumulation_steps: int = 1
    grad_clip_norm: float = 1.0
    amp_dtype: str = "bf16"
    tf32: bool = True
    cudnn_benchmark: bool = True
    ema_decay: float = 0.9999
    compile_model: bool = False
    compile_mode: str = "default"
    profile_steps: int = 20

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not 0 <= self.warmup_steps <= self.max_steps:
            raise ValueError("warmup_steps must be in [0, max_steps]")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        for name in ("validate_every", "validation_steps", "checkpoint_every", "log_every"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.samples_per_virtual_epoch < 1 or self.accumulation_steps < 1:
            raise ValueError("sample and accumulation counts must be positive")
        if self.grad_clip_norm < 0:
            raise ValueError("grad_clip_norm cannot be negative")
        if self.amp_dtype not in ("bf16", "fp16", "fp32"):
            raise ValueError("amp_dtype must be bf16, fp16, or fp32")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")

    def to_dict(self) -> dict:
        return asdict(self)
