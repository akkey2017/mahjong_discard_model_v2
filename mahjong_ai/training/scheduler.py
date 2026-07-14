"""Optimizer-step warmup + cosine learning-rate schedule."""

from __future__ import annotations

import math


class StepWarmupCosineScheduler:
    def __init__(
        self,
        optimizer,
        *,
        max_steps: int,
        warmup_steps: int,
        min_lr_ratio: float = 0.01,
    ):
        if max_steps < 1 or not 0 <= warmup_steps <= max_steps:
            raise ValueError("invalid max_steps/warmup_steps")
        self.optimizer = optimizer
        self.max_steps = int(max_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.last_step = 0
        self._apply(self.multiplier(0))

    def multiplier(self, completed_steps: int) -> float:
        if self.warmup_steps and completed_steps < self.warmup_steps:
            return (completed_steps + 1) / self.warmup_steps
        decay_steps = max(1, self.max_steps - self.warmup_steps)
        progress = (completed_steps - self.warmup_steps) / decay_steps
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def _apply(self, multiplier: float) -> None:
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * multiplier

    def step(self, completed_steps: int | None = None) -> None:
        self.last_step = self.last_step + 1 if completed_steps is None else int(completed_steps)
        self._apply(self.multiplier(self.last_step))

    def state_dict(self) -> dict:
        return {
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": list(self.base_lrs),
            "last_step": self.last_step,
        }

    def load_state_dict(self, state: dict) -> None:
        for key in ("max_steps", "warmup_steps", "min_lr_ratio"):
            if state[key] != getattr(self, key):
                raise ValueError(f"scheduler {key} differs from checkpoint")
        if len(state["base_lrs"]) != len(self.optimizer.param_groups):
            raise ValueError("scheduler optimizer group count differs")
        self.base_lrs = [float(value) for value in state["base_lrs"]]
        self.last_step = int(state["last_step"])
        self._apply(self.multiplier(self.last_step))
