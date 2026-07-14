"""Lightweight stage timings for initial training steps."""

from __future__ import annotations

from collections import Counter


class TrainingProfiler:
    def __init__(self, max_steps: int):
        self.max_steps = max_steps
        self.steps = 0
        self.seconds = Counter()

    @property
    def active(self) -> bool:
        return self.steps < self.max_steps

    def add(self, **seconds: float) -> None:
        if not self.active:
            return
        self.seconds.update(seconds)
        self.steps += 1

    def summary(self) -> dict:
        total = sum(self.seconds.values())
        result = {"profiled_steps": self.steps, "total_seconds": total}
        for name, value in sorted(self.seconds.items()):
            result[f"{name}_seconds"] = value
            result[f"{name}_fraction"] = value / total if total else 0.0
        return result
