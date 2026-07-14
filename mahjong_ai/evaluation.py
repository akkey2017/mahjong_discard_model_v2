"""Streaming task metrics, calibration, and error analysis for Snapshot ViT."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from mahjong_ai.data import TASK_CLASS_COUNTS, TASK_INDEX


@dataclass
class TaskEvaluation:
    task: str
    classes: int
    calibration_bins: int = 15
    max_errors: int = 100
    confusion: torch.Tensor = field(init=False)
    bin_count: torch.Tensor = field(init=False)
    bin_confidence: torch.Tensor = field(init=False)
    bin_correct: torch.Tensor = field(init=False)
    nll_sum: float = 0.0
    brier_sum: float = 0.0
    total: int = 0
    _errors: list = field(default_factory=list)
    _error_sequence: int = 0

    def __post_init__(self) -> None:
        self.confusion = torch.zeros((self.classes, self.classes), dtype=torch.int64)
        self.bin_count = torch.zeros(self.calibration_bins, dtype=torch.int64)
        self.bin_confidence = torch.zeros(self.calibration_bins, dtype=torch.float64)
        self.bin_correct = torch.zeros(self.calibration_bins, dtype=torch.float64)

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        sample_ids: Sequence[str],
    ) -> None:
        if labels.numel() == 0:
            return
        logits = logits.detach().float().cpu()
        labels = labels.detach().long().cpu()
        probabilities = logits.softmax(dim=-1)
        confidence, predictions = probabilities.max(dim=-1)
        correct = predictions.eq(labels)
        count = int(labels.numel())
        self.total += count
        self.nll_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
        targets = F.one_hot(labels, num_classes=self.classes).float()
        self.brier_sum += float(((probabilities - targets) ** 2).sum())
        indices = labels * self.classes + predictions
        self.confusion += torch.bincount(
            indices, minlength=self.classes * self.classes
        ).reshape(self.classes, self.classes)

        bins = torch.clamp((confidence * self.calibration_bins).long(), max=self.calibration_bins - 1)
        for index in range(self.calibration_bins):
            active = bins == index
            if not active.any():
                continue
            self.bin_count[index] += int(active.sum())
            self.bin_confidence[index] += float(confidence[active].sum())
            self.bin_correct[index] += float(correct[active].sum())

        incorrect = (~correct).nonzero(as_tuple=False).flatten()
        if incorrect.numel() and self.max_errors:
            top = incorrect[torch.argsort(confidence[incorrect], descending=True)[:self.max_errors]]
            for index in top.tolist():
                error = {
                    "sample_id": str(sample_ids[index]),
                    "true": int(labels[index]),
                    "predicted": int(predictions[index]),
                    "confidence": float(confidence[index]),
                }
                item = (error["confidence"], self._error_sequence, error)
                self._error_sequence += 1
                if len(self._errors) < self.max_errors:
                    heapq.heappush(self._errors, item)
                elif item[0] > self._errors[0][0]:
                    heapq.heapreplace(self._errors, item)

    def compute(self) -> dict:
        support = self.confusion.sum(dim=1)
        predicted = self.confusion.sum(dim=0)
        true_positive = self.confusion.diag()
        precision = true_positive.float() / predicted.clamp_min(1)
        recall = true_positive.float() / support.clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
        accuracy = float(true_positive.sum() / max(self.total, 1))
        ece = 0.0
        calibration = []
        for index in range(self.calibration_bins):
            count = int(self.bin_count[index])
            average_confidence = float(self.bin_confidence[index] / max(count, 1))
            average_accuracy = float(self.bin_correct[index] / max(count, 1))
            ece += count / max(self.total, 1) * abs(average_accuracy - average_confidence)
            calibration.append({
                "lower": index / self.calibration_bins,
                "upper": (index + 1) / self.calibration_bins,
                "count": count,
                "confidence": average_confidence,
                "accuracy": average_accuracy,
            })
        errors = [item[2] for item in sorted(self._errors, reverse=True)]
        return {
            "total": self.total,
            "accuracy": accuracy,
            "nll": self.nll_sum / max(self.total, 1),
            "brier": self.brier_sum / max(self.total, 1),
            "ece": ece,
            "macro_f1": float(f1.mean()),
            "per_class": [
                {
                    "class": index,
                    "support": int(support[index]),
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                }
                for index in range(self.classes)
            ],
            "confusion_matrix": self.confusion.tolist(),
            "calibration": calibration,
            "high_confidence_errors": errors,
        }


class EvaluationAccumulator:
    def __init__(self, *, calibration_bins: int = 15, max_errors: int = 100):
        self.tasks = {
            task: TaskEvaluation(
                task,
                classes,
                calibration_bins=calibration_bins,
                max_errors=max_errors,
            )
            for task, classes in TASK_CLASS_COUNTS.items()
        }

    def update(
        self,
        logits: Mapping[str, torch.Tensor],
        labels: torch.Tensor,
        masks: torch.Tensor,
        sample_ids: Sequence[str],
    ) -> None:
        for task, index in TASK_INDEX.items():
            active = masks[:, index].bool()
            if not active.any():
                continue
            positions = active.nonzero(as_tuple=False).flatten().tolist()
            self.tasks[task].update(
                logits[task][active],
                labels[:, index][active],
                [sample_ids[position] for position in positions],
            )

    def compute(self) -> dict:
        return {task: metrics.compute() for task, metrics in self.tasks.items()}


def summarize_task_metrics(tasks: Mapping[str, dict]) -> dict:
    totals = sum(metrics["total"] for metrics in tasks.values())
    correct = sum(metrics["accuracy"] * metrics["total"] for metrics in tasks.values())
    finite_nll = [metrics["nll"] for metrics in tasks.values() if math.isfinite(metrics["nll"])]
    return {
        "active_targets": totals,
        "micro_accuracy": correct / max(totals, 1),
        "mean_task_nll": sum(finite_nll) / max(len(finite_nll), 1),
    }
