"""Unified multi-task samples built from one incremental state pass per round."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Iterator, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from mahjong_ai.state import IncrementalStateEncoder, RoundState
from mahjong_ai_features import FEATURE_ID_TO_TILE, _process_single_number
from mahjong_rules import (
    can_ankan,
    can_chi,
    can_daiminkan,
    can_pon,
    normalize_red_five,
    tile_count,
)

from .schema import (
    DAPAI_RIICHI,
    DAPAI_TSUMOGIRI,
    EVENT_TYPE_NAMES,
    FULOU_DAIMINKAN,
    FULOU_FOUR_TILES,
    FULOU_SOURCE_SHIFT,
    GANG_KAKAN,
    NO_MELD,
)
from .streaming_dataset import CompactRoundRecord, StreamingRoundDataset


TASK_NAMES = ("dapai", "riichi", "fulou", "gang", "hule")
TASK_INDEX = {name: index for index, name in enumerate(TASK_NAMES)}
NEGATIVE_SAMPLING_TASKS = ("riichi", "fulou", "gang", "hule")
TASK_CLASS_COUNTS = {
    "dapai": 34,
    "riichi": 2,
    "fulou": 4,
    "gang": 3,
    "hule": 2,
}
TARGET_SCHEMA_VERSION = "unified-multitask-target-v1"


@dataclass(frozen=True)
class MultiTaskTarget:
    """Fixed-order labels and validity masks for every model head."""

    labels: np.ndarray
    masks: np.ndarray

    def __post_init__(self) -> None:
        if self.labels.shape != (len(TASK_NAMES),):
            raise ValueError(f"labels must have shape ({len(TASK_NAMES)},)")
        if self.masks.shape != (len(TASK_NAMES),):
            raise ValueError(f"masks must have shape ({len(TASK_NAMES)},)")
        if self.labels.dtype != np.int64:
            raise ValueError("labels must use int64")
        if self.masks.dtype != np.bool_:
            raise ValueError("masks must use bool")
        for task, index in TASK_INDEX.items():
            label = int(self.labels[index])
            if self.masks[index] and not 0 <= label < TASK_CLASS_COUNTS[task]:
                raise ValueError(f"{task} label {label} is out of range")

    @classmethod
    def from_mapping(cls, values: Mapping[str, int]) -> "MultiTaskTarget":
        labels = np.zeros(len(TASK_NAMES), dtype=np.int64)
        masks = np.zeros(len(TASK_NAMES), dtype=np.bool_)
        for task, label in values.items():
            if task not in TASK_INDEX:
                raise ValueError(f"unknown task: {task}")
            index = TASK_INDEX[task]
            labels[index] = int(label)
            masks[index] = True
        return cls(labels=labels, masks=masks)

    def label(self, task: str) -> int:
        return int(self.labels[TASK_INDEX[task]])

    def has(self, task: str) -> bool:
        return bool(self.masks[TASK_INDEX[task]])

    def with_mask(self, task: str, enabled: bool) -> "MultiTaskTarget":
        masks = self.masks.copy()
        masks[TASK_INDEX[task]] = enabled
        return MultiTaskTarget(labels=self.labels.copy(), masks=masks)


@dataclass(frozen=True)
class MultiTaskSample:
    """One target-relative state encoded once with all applicable labels."""

    sample_id: str
    round_sample_id: str
    year: int
    event_index: int
    player_id: int
    features: torch.Tensor | None
    target: MultiTaskTarget
    worker_id: int
    worker_sequence: int
    sample_index: int


@dataclass(frozen=True)
class TaskSamplingPolicy:
    """Training-only deterministic negative sampling policy for one head."""

    keep_probability: float = 1.0
    max_negative_per_positive: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.keep_probability <= 1.0:
            raise ValueError("keep_probability must be in [0, 1]")
        if self.max_negative_per_positive is not None and self.max_negative_per_positive < 0:
            raise ValueError("max_negative_per_positive cannot be negative")


@dataclass(frozen=True)
class NegativeSamplingConfig:
    """Policies are applied only to negative labels in the train split.

    Ratio caps are exact within a round. A round with no positive for that task
    keeps no ratio-capped negatives; use ``keep_probability`` when negatives
    must also be retained in negative-only rounds.
    """

    seed: int = 0
    policies: Mapping[str, TaskSamplingPolicy] | None = None

    def __post_init__(self) -> None:
        for task in (self.policies or {}):
            if task not in NEGATIVE_SAMPLING_TASKS:
                raise ValueError(f"unknown sampling task: {task}")

    def policy(self, task: str) -> TaskSamplingPolicy:
        return (self.policies or {}).get(task, TaskSamplingPolicy())

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "policies": {
                task: {
                    "keep_probability": policy.keep_probability,
                    "max_negative_per_positive": policy.max_negative_per_positive,
                }
                for task, policy in sorted((self.policies or {}).items())
            },
        }


def _stable_hash(seed: int, *parts: object) -> int:
    payload = ":".join((str(seed), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _counter_to_hand(counter: Sequence[int]) -> str:
    groups = []
    for suit, indices in (
        ("m", range(0, 10)),
        ("p", range(10, 20)),
        ("s", range(20, 30)),
        ("z", range(30, 37)),
    ):
        digits = []
        for tile in indices:
            raw = FEATURE_ID_TO_TILE[tile]
            digits.extend(raw[1] for _ in range(int(counter[tile])))
        if digits:
            groups.append(suit + "".join(digits))
    return "".join(groups)


def _initial_state(record: CompactRoundRecord) -> RoundState:
    return RoundState.from_qipai(
        {
            "shoupai": [_counter_to_hand(hand) for hand in record.hands],
            "baopai": FEATURE_ID_TO_TILE[record.initial_dora],
            "defen": [int(value) for value in record.scores],
            "zhuangfeng": record.round_wind,
            "jushu": record.dealer,
            "changbang": record.honba,
            "lizhibang": record.kyotaku,
        }
    )


def _meld_string(tiles: Sequence[int], *, player: int, source: int | None) -> str:
    if not tiles:
        raise ValueError("meld cannot be empty")
    raw_tiles = [FEATURE_ID_TO_TILE[int(tile)] for tile in tiles]
    suit = raw_tiles[0][0]
    if any(raw[0] != suit for raw in raw_tiles):
        raise ValueError("meld tiles must share a suit")
    marker = ""
    if source is not None:
        offset = (source - player) % 4
        marker = {1: "+", 2: "=", 3: "-"}.get(offset, "")
    return suit + "".join(raw[1] for raw in raw_tiles) + marker


def _event_meld_tiles(record: CompactRoundRecord, event: np.void) -> list[int]:
    offset = int(event["meld_offset"])
    if offset == int(NO_MELD):
        return []
    kind = EVENT_TYPE_NAMES[int(event["type"])]
    if kind == "fulou":
        count = 4 if int(event["flags"]) & FULOU_FOUR_TILES else 3
    elif kind == "gang":
        count = 4
    else:
        return []
    return [int(tile) for tile in record.meld_tiles[offset:offset + count]]


def _event_as_raw(record: CompactRoundRecord, event: np.void) -> dict:
    kind = EVENT_TYPE_NAMES[int(event["type"])]
    player = int(event["player"])
    tile = int(event["tile"])
    flags = int(event["flags"])
    if kind in ("zimo", "gangzimo"):
        return {kind: {"l": player, "p": FEATURE_ID_TO_TILE[tile]}}
    if kind == "dapai":
        suffix = ("*" if flags & DAPAI_RIICHI else "") + (
            "_" if flags & DAPAI_TSUMOGIRI else ""
        )
        return {kind: {"l": player, "p": FEATURE_ID_TO_TILE[tile] + suffix}}
    if kind == "fulou":
        source = (flags >> FULOU_SOURCE_SHIFT) & 0b11
        meld = _meld_string(_event_meld_tiles(record, event), player=player, source=source)
        return {kind: {"l": player, "m": meld}}
    if kind == "gang":
        # RoundState only needs a direction marker to distinguish kakan.
        source = (player + 1) % 4 if flags == GANG_KAKAN else None
        meld = _meld_string(_event_meld_tiles(record, event), player=player, source=source)
        return {kind: {"l": player, "m": meld}}
    if kind == "kaigang":
        return {kind: {"baopai": FEATURE_ID_TO_TILE[tile]}}
    if kind == "lizhi":
        return {kind: {"l": player}}
    if kind == "hule":
        return {kind: {"l": player}}
    return {kind: {}}


def _has_kakan_option(state: RoundState, player: int) -> bool:
    hand = state.hands[player]
    for meld in state.melds[player]:
        if len(meld) != 3:
            continue
        normalized = [normalize_red_five(tile) for tile in meld]
        if len(set(normalized)) == 1 and tile_count(hand, normalized[0]) >= 1:
            return True
    return False


def _can_call(state: RoundState, player: int, discarder: int, tile: int) -> bool:
    from_wrong_seat = player != (discarder + 1) % 4
    return (
        can_chi(state.hands[player], tile, from_shimocha=from_wrong_seat)
        or can_pon(state.hands[player], tile)
        or can_daiminkan(state.hands[player], tile)
    )


class MultiTaskSampleBuilder:
    """Replay a compact round once and emit unique target-relative states."""

    def __init__(
        self,
        *,
        split: str,
        negative_sampling: NegativeSamplingConfig | None = None,
        include_fulou_negatives: bool = True,
        encode_features: bool = True,
    ):
        self.split = split
        self.negative_sampling = negative_sampling or NegativeSamplingConfig()
        self.include_fulou_negatives = include_fulou_negatives
        self.encode_features = encode_features

    def _sample(
        self,
        record: CompactRoundRecord,
        state: RoundState,
        event_index: int,
        player: int,
        labels: Mapping[str, int],
    ) -> MultiTaskSample:
        sample_id = f"{record.sample_id}:{event_index}:{player}"
        features = IncrementalStateEncoder(state, player).encode() if self.encode_features else None
        return MultiTaskSample(
            sample_id=sample_id,
            round_sample_id=record.sample_id,
            year=record.year,
            event_index=event_index,
            player_id=player,
            features=features,
            target=MultiTaskTarget.from_mapping(labels),
            worker_id=record.worker_id,
            worker_sequence=record.worker_sequence,
            sample_index=0,
        )

    def _sample_negatives(self, samples: list[MultiTaskSample]) -> list[MultiTaskSample]:
        if self.split != "train" or not samples:
            return samples
        keep_masks = [sample.target.masks.copy() for sample in samples]
        for task in NEGATIVE_SAMPLING_TASKS:
            task_index = TASK_INDEX[task]
            policy = self.negative_sampling.policy(task)
            negative_indices = [
                index for index, sample in enumerate(samples)
                if sample.target.masks[task_index] and sample.target.labels[task_index] == 0
            ]
            positive_count = sum(
                sample.target.masks[task_index] and sample.target.labels[task_index] > 0
                for sample in samples
            )
            kept = []
            threshold = int(policy.keep_probability * (1 << 64))
            for index in negative_indices:
                value = _stable_hash(
                    self.negative_sampling.seed, samples[index].sample_id, task, "keep"
                )
                if value < threshold:
                    kept.append(index)
                else:
                    keep_masks[index][task_index] = False
            if policy.max_negative_per_positive is not None:
                cap = math.floor(policy.max_negative_per_positive * positive_count)
                ranked = sorted(
                    kept,
                    key=lambda index: _stable_hash(
                        self.negative_sampling.seed, samples[index].sample_id, task, "cap"
                    ),
                )
                for index in ranked[cap:]:
                    keep_masks[index][task_index] = False

        result = []
        for sample, masks in zip(samples, keep_masks):
            if not masks.any():
                continue
            result.append(
                MultiTaskSample(
                    sample_id=sample.sample_id,
                    round_sample_id=sample.round_sample_id,
                    year=sample.year,
                    event_index=sample.event_index,
                    player_id=sample.player_id,
                    features=sample.features,
                    target=MultiTaskTarget(labels=sample.target.labels.copy(), masks=masks),
                    worker_id=sample.worker_id,
                    worker_sequence=sample.worker_sequence,
                    sample_index=sample.sample_index,
                )
            )
        return result

    def build_round(self, record: CompactRoundRecord) -> list[MultiTaskSample]:
        state = _initial_state(record)
        samples: list[MultiTaskSample] = []
        seen_ids: set[str] = set()
        events = record.events

        for event_index, event in enumerate(events):
            kind = EVENT_TYPE_NAMES[int(event["type"])]
            player = int(event["player"])
            flags = int(event["flags"])
            if kind == "dapai":
                labels = {
                    "dapai": _process_single_number(int(event["tile"])),
                    "riichi": int(bool(flags & DAPAI_RIICHI)),
                }
                if can_ankan(state.hands[player]) or _has_kakan_option(state, player):
                    labels["gang"] = 0
                sample = self._sample(record, state, event_index, player, labels)
                samples.append(sample)
                seen_ids.add(sample.sample_id)
            elif kind == "fulou":
                sample = self._sample(
                    record, state, event_index, player, {"fulou": flags & 0b11}
                )
                if sample.sample_id in seen_ids:
                    raise RuntimeError(f"duplicate unified state: {sample.sample_id}")
                samples.append(sample)
                seen_ids.add(sample.sample_id)
            elif kind == "gang":
                label = 2 if flags == GANG_KAKAN else 1
                sample = self._sample(record, state, event_index, player, {"gang": label})
                samples.append(sample)
                seen_ids.add(sample.sample_id)
            elif kind == "hule":
                sample = self._sample(record, state, event_index, player, {"hule": 1})
                samples.append(sample)
                seen_ids.add(sample.sample_id)

            state.apply_event(_event_as_raw(record, event))

            if kind == "dapai" and self.include_fulou_negatives:
                next_kind = (
                    EVENT_TYPE_NAMES[int(events[event_index + 1]["type"])]
                    if event_index + 1 < len(events) else None
                )
                if next_kind == "hule":
                    continue
                called_by = (
                    int(events[event_index + 1]["player"])
                    if next_kind == "fulou" else None
                )
                tile = int(event["tile"])
                for candidate in range(4):
                    if candidate in (player, called_by):
                        continue
                    if not _can_call(state, candidate, player, tile):
                        continue
                    sample = self._sample(
                        record, state, event_index + 1, candidate, {"fulou": 0}
                    )
                    if sample.sample_id in seen_ids:
                        raise RuntimeError(f"duplicate unified state: {sample.sample_id}")
                    samples.append(sample)
                    seen_ids.add(sample.sample_id)

        sampled = self._sample_negatives(samples)
        return [replace(sample, sample_index=index) for index, sample in enumerate(sampled)]


class StreamingMultiTaskDataset(IterableDataset):
    """Phase-4 round streaming followed by Phase-5 unified target generation."""

    def __init__(
        self,
        manifest_path,
        *,
        split: str = "train",
        seed: int = 0,
        shuffle: bool = True,
        shuffle_buffer_rounds: int = 8192,
        rank: int = 0,
        world_size: int = 1,
        resume_offsets: Mapping[int, int] | None = None,
        resume_sample_offsets: Mapping[int, Sequence[int]] | None = None,
        years: Sequence[int] | None = None,
        negative_sampling: NegativeSamplingConfig | None = None,
        include_fulou_negatives: bool = True,
        encode_features: bool = True,
    ):
        super().__init__()
        self.resume_sample_offsets = {
            int(worker): (int(position[0]), int(position[1]))
            for worker, position in (resume_sample_offsets or {}).items()
        }
        round_offsets = dict(resume_offsets or {})
        round_offsets.update({
            worker: position[0] for worker, position in self.resume_sample_offsets.items()
        })
        self.rounds = StreamingRoundDataset(
            manifest_path,
            split=split,
            seed=seed,
            shuffle=shuffle,
            shuffle_buffer_rounds=shuffle_buffer_rounds,
            rank=rank,
            world_size=world_size,
            resume_offsets=round_offsets,
            years=years,
            tensorize=False,
        )
        self.builder = MultiTaskSampleBuilder(
            split=split,
            negative_sampling=negative_sampling,
            include_fulou_negatives=include_fulou_negatives,
            encode_features=encode_features,
        )

    @property
    def epoch(self) -> int:
        return self.rounds.epoch

    def set_epoch(self, epoch: int) -> None:
        self.rounds.set_epoch(epoch)

    def state_dict(self) -> dict:
        state = self.rounds.state_dict()
        state["target_schema_version"] = TARGET_SCHEMA_VERSION
        state["negative_sampling"] = self.builder.negative_sampling.to_dict()
        state["include_fulou_negatives"] = self.builder.include_fulou_negatives
        state["resume_sample_offsets"] = {
            worker: list(position)
            for worker, position in self.resume_sample_offsets.items()
        }
        return state

    def set_resume_sample_offsets(self, offsets: Mapping[int, Sequence[int]]) -> None:
        self.resume_sample_offsets = {
            int(worker): (int(position[0]), int(position[1]))
            for worker, position in offsets.items()
        }
        self.rounds.resume_offsets = {
            worker: position[0] for worker, position in self.resume_sample_offsets.items()
        }

    @staticmethod
    def update_resume_sample_offsets(
        offsets: dict[int, tuple[int, int]], sample: MultiTaskSample
    ) -> None:
        position = (sample.worker_sequence, sample.sample_index + 1)
        if position > offsets.get(sample.worker_id, (-1, -1)):
            offsets[sample.worker_id] = position

    def target_manifest(self) -> dict:
        """Serializable target-generation metadata for checkpoints/runs."""

        return {
            "target_schema_version": TARGET_SCHEMA_VERSION,
            "source_manifest": str(self.rounds.manifest_path),
            "split": self.rounds.split,
            "years": sorted(self.rounds.years) if self.rounds.years is not None else None,
            "negative_sampling": self.builder.negative_sampling.to_dict(),
            "include_fulou_negatives": self.builder.include_fulou_negatives,
            "feature_encoding": self.builder.encode_features,
        }

    def __iter__(self) -> Iterator[MultiTaskSample]:
        for record in self.rounds:
            skip = 0
            position = self.resume_sample_offsets.get(record.worker_id)
            if position is not None and record.worker_sequence == position[0]:
                skip = position[1]
            yield from self.builder.build_round(record)[skip:]


def unified_multitask_collate(batch: Sequence[MultiTaskSample]) -> dict[str, object]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    if any(sample.features is None for sample in batch):
        raise ValueError("feature encoding is disabled for at least one sample")
    return {
        "features": torch.stack([sample.features for sample in batch]),
        "labels": torch.from_numpy(np.stack([sample.target.labels for sample in batch])),
        "masks": torch.from_numpy(np.stack([sample.target.masks for sample in batch])),
        "sample_ids": tuple(sample.sample_id for sample in batch),
        "years": torch.tensor([sample.year for sample in batch], dtype=torch.int16),
        "player_ids": torch.tensor([sample.player_id for sample in batch], dtype=torch.uint8),
        "worker_ids": torch.tensor([sample.worker_id for sample in batch], dtype=torch.int64),
        "worker_sequences": torch.tensor(
            [sample.worker_sequence for sample in batch], dtype=torch.int64
        ),
        "sample_indices": torch.tensor(
            [sample.sample_index for sample in batch], dtype=torch.int64
        ),
    }


def build_multitask_dataloader(
    dataset: StreamingMultiTaskDataset,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    drop_last: bool = False,
    generator: torch.Generator | None = None,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "collate_fn": unified_multitask_collate,
        "generator": generator,
    }
    if num_workers:
        kwargs.update(
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
    return DataLoader(**kwargs)


def masked_multitask_loss(
    logits: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    masks: torch.Tensor,
    *,
    task_weights: Mapping[str, float] | None = None,
    loss_fns: Mapping[str, object] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute one masked loss per head after a single all-head forward pass."""

    if labels.ndim != 2 or labels.shape[1] != len(TASK_NAMES):
        raise ValueError(f"labels must be [batch, {len(TASK_NAMES)}]")
    if masks.shape != labels.shape or masks.dtype != torch.bool:
        raise ValueError("masks must be bool and have the same shape as labels")
    missing = [task for task in TASK_NAMES if task not in logits]
    if missing:
        raise ValueError(f"all heads must be forwarded; missing {missing}")
    task_weights = task_weights or {}
    loss_fns = loss_fns or {}
    per_task: dict[str, torch.Tensor] = {}
    total = None
    for task, index in TASK_INDEX.items():
        active = masks[:, index]
        if not torch.any(active):
            continue
        task_logits = logits[task][active]
        targets = labels[:, index][active]
        loss_fn = loss_fns.get(task)
        loss = loss_fn(task_logits, targets) if loss_fn is not None else F.cross_entropy(
            task_logits, targets
        )
        per_task[task] = loss
        weighted = loss * float(task_weights.get(task, 1.0))
        total = weighted if total is None else total + weighted
    if total is None:
        # Keep a differentiable zero on the same device as model output.
        total = next(iter(logits.values())).sum() * 0.0
    return total, per_task


class MultiTaskMetrics:
    """Sample-weighted loss and accuracy counters for unified masked batches."""

    def __init__(self, task_weights: Mapping[str, float] | None = None):
        self.task_weights = dict(task_weights or {})
        self.correct = {task: 0 for task in TASK_NAMES}
        self.total = {task: 0 for task in TASK_NAMES}
        self.loss_sum = {task: 0.0 for task in TASK_NAMES}

    def update(
        self,
        logits: Mapping[str, torch.Tensor],
        labels: torch.Tensor,
        masks: torch.Tensor,
        per_task_losses: Mapping[str, torch.Tensor],
    ) -> None:
        for task, index in TASK_INDEX.items():
            active = masks[:, index]
            count = int(active.sum().item())
            if not count:
                continue
            predictions = logits[task][active].argmax(dim=-1)
            targets = labels[:, index][active]
            self.correct[task] += int((predictions == targets).sum().item())
            self.total[task] += count
            if task in per_task_losses:
                self.loss_sum[task] += float(per_task_losses[task].detach().item()) * count

    def compute(self) -> dict[str, float | int]:
        result: dict[str, float | int] = {}
        enabled = [
            task for task in TASK_NAMES
            if self.total[task] and float(self.task_weights.get(task, 1.0)) > 0.0
        ]
        enabled_total = sum(self.total[task] for task in enabled)
        enabled_correct = sum(self.correct[task] for task in enabled)
        result["top1_acc"] = enabled_correct / enabled_total if enabled_total else 0.0
        for task in TASK_NAMES:
            total = self.total[task]
            result[f"{task}_total"] = total
            result[f"{task}_acc"] = self.correct[task] / total if total else 0.0
            result[f"{task}_loss"] = self.loss_sum[task] / total if total else 0.0
        return result
