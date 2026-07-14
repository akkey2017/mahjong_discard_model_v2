"""Memory-bounded streaming of compact round shards."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .schema import (
    DATASET_SCHEMA_VERSION,
    EVENT_DTYPE,
    EVENT_TYPES,
    FULOU_FOUR_TILES,
    NO_MELD,
)


@dataclass(frozen=True)
class ShardDescriptor:
    path: str
    shard_id: int
    rounds: int
    events: int


@dataclass(frozen=True)
class CompactRoundRecord:
    """One detached, self-contained round read from a memory-mapped shard."""

    sample_id: str
    shard_id: int
    local_round_index: int
    game_id: str
    year: int
    round_index: int
    round_wind: int
    dealer: int
    honba: int
    kyotaku: int
    scores: np.ndarray
    hands: np.ndarray
    initial_dora: int
    events: np.ndarray
    meld_tiles: np.ndarray
    worker_id: int = 0
    worker_sequence: int = 0

    def tensorize(self) -> "TensorRoundRecord":
        event_bytes = self.events.view(np.uint8).reshape(len(self.events), EVENT_DTYPE.itemsize)
        return TensorRoundRecord(
            sample_id=self.sample_id,
            shard_id=self.shard_id,
            local_round_index=self.local_round_index,
            game_id=self.game_id,
            year=self.year,
            round_index=self.round_index,
            round_wind=self.round_wind,
            dealer=self.dealer,
            honba=self.honba,
            kyotaku=self.kyotaku,
            scores=torch.from_numpy(self.scores),
            hands=torch.from_numpy(self.hands),
            initial_dora=self.initial_dora,
            events=torch.from_numpy(event_bytes.copy()),
            meld_tiles=torch.from_numpy(self.meld_tiles),
            worker_id=self.worker_id,
            worker_sequence=self.worker_sequence,
        )


@dataclass(frozen=True)
class TensorRoundRecord:
    """Tensor transfer form; DataLoader can pin its variable-length buffers."""

    sample_id: str
    shard_id: int
    local_round_index: int
    game_id: str
    year: int
    round_index: int
    round_wind: int
    dealer: int
    honba: int
    kyotaku: int
    scores: torch.Tensor
    hands: torch.Tensor
    initial_dora: int
    events: torch.Tensor
    meld_tiles: torch.Tensor
    worker_id: int
    worker_sequence: int

    def pin_memory(self) -> "TensorRoundRecord":
        return replace(
            self,
            scores=self.scores.pin_memory(),
            hands=self.hands.pin_memory(),
            events=self.events.pin_memory(),
            meld_tiles=self.meld_tiles.pin_memory(),
        )


class CompactShardReader:
    """Open one shard with NumPy mmap and yield detached round records."""

    def __init__(self, root: Path, descriptor: ShardDescriptor):
        self.root = Path(root)
        self.descriptor = descriptor
        shard = self.root / descriptor.path
        self.rounds = np.load(shard / "rounds.npy", mmap_mode="r", allow_pickle=False)
        self.events = np.load(shard / "events.npy", mmap_mode="r", allow_pickle=False)
        self.melds = np.load(shard / "melds.npy", mmap_mode="r", allow_pickle=False)
        self.offsets = np.load(shard / "offsets.npy", mmap_mode="r", allow_pickle=False)
        self.metadata = np.load(shard / "metadata.npy", mmap_mode="r", allow_pickle=False)

    @staticmethod
    def _meld_count(event: np.void) -> int:
        kind = int(event["type"])
        if kind == EVENT_TYPES["fulou"]:
            return 4 if int(event["flags"]) & FULOU_FOUR_TILES else 3
        if kind == EVENT_TYPES["gang"]:
            return 4
        return 0

    def read_round(self, local_index: int) -> CompactRoundRecord:
        row = self.rounds[local_index]
        event_start = int(self.offsets[local_index])
        event_end = int(self.offsets[local_index + 1])
        events = np.array(self.events[event_start:event_end], copy=True)
        used_offsets = events["meld_offset"] != NO_MELD
        meld_start = 0
        meld_end = 0
        if np.any(used_offsets):
            absolute_offsets = events["meld_offset"][used_offsets]
            meld_start = int(absolute_offsets.min())
            ends = []
            for event in events[used_offsets]:
                ends.append(int(event["meld_offset"]) + self._meld_count(event))
            meld_end = max(ends)
            events["meld_offset"][used_offsets] -= meld_start
        meld_tiles = np.array(self.melds[meld_start:meld_end], copy=True)

        game = self.metadata[int(row["game_index"])]
        game_id = bytes(game["game_id"]).rstrip(b"\x00").decode("ascii")
        round_index = int(row["round_index"])
        return CompactRoundRecord(
            sample_id=f"{game_id}:{round_index}",
            shard_id=self.descriptor.shard_id,
            local_round_index=local_index,
            game_id=game_id,
            year=int(game["year"]),
            round_index=round_index,
            round_wind=int(row["round_wind"]),
            dealer=int(row["dealer"]),
            honba=int(row["honba"]),
            kyotaku=int(row["kyotaku"]),
            scores=np.array(row["scores"], copy=True),
            hands=np.array(row["hands"], copy=True),
            initial_dora=int(row["initial_dora"]),
            events=events,
            meld_tiles=meld_tiles,
        )

    def __iter__(self) -> Iterator[CompactRoundRecord]:
        for local_index in range(len(self.rounds)):
            yield self.read_round(local_index)

    def close(self) -> None:
        for array in (self.rounds, self.events, self.melds, self.offsets, self.metadata):
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def __enter__(self) -> "CompactShardReader":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def _seed(base_seed: int, epoch: int, split: str, worker: int, purpose: str) -> int:
    value = f"{base_seed}:{epoch}:{split}:{worker}:{purpose}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def _shuffle_buffer(
    records: Iterator[CompactRoundRecord],
    size: int,
    rng: random.Random,
) -> Iterator[CompactRoundRecord]:
    if size <= 1:
        yield from records
        return
    buffer = []
    for record in records:
        if len(buffer) < size:
            buffer.append(record)
            continue
        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = record
    rng.shuffle(buffer)
    yield from buffer


class StreamingRoundDataset(IterableDataset):
    """Stream rounds with deterministic distributed/DataLoader partitioning."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        split: str = "train",
        seed: int = 0,
        shuffle: bool = True,
        shuffle_buffer_rounds: int = 8192,
        rank: int = 0,
        world_size: int = 1,
        resume_offsets: Mapping[int, int] | None = None,
        years: Iterable[int] | None = None,
        tensorize: bool = True,
    ):
        super().__init__()
        manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
            raise ValueError(f"unsupported dataset schema: {manifest.get('schema_version')!r}")
        if not manifest.get("complete"):
            raise ValueError("streaming requires a complete prepared dataset")
        if split not in ("train", "validation", "test", "all"):
            raise ValueError(f"unknown split: {split}")
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("rank must be in 0..world_size-1")
        if shuffle_buffer_rounds < 0:
            raise ValueError("shuffle_buffer_rounds cannot be negative")
        self.manifest_path = manifest_path
        self.root = manifest_path.parent
        self.split = split
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.shuffle_buffer_rounds = int(shuffle_buffer_rounds)
        self.rank = rank
        self.world_size = world_size
        self.resume_offsets = {int(key): int(value) for key, value in (resume_offsets or {}).items()}
        self.years = None if years is None else frozenset(int(year) for year in years)
        if self.years is not None and not self.years:
            raise ValueError("years cannot be empty")
        self.tensorize = tensorize
        self.shards = tuple(
            ShardDescriptor(
                path=item["path"],
                shard_id=int(item["id"]),
                rounds=int(item["rounds"]),
                events=int(item["events"]),
            )
            for item in manifest["shards"]
            if split == "all" or item["split"] == split
        )
        self.total_rounds = sum(item.rounds for item in self.shards)
        # RawValue has no semaphore and is visible to persistent forked workers.
        self._epoch = mp.Value("q", 0, lock=False)

    def __len__(self) -> int:
        return self.total_rounds

    @property
    def epoch(self) -> int:
        return int(self._epoch.value)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self._epoch.value = epoch

    def state_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "resume_offsets": dict(self.resume_offsets),
            "seed": self.seed,
            "split": self.split,
            "rank": self.rank,
            "world_size": self.world_size,
            "years": sorted(self.years) if self.years is not None else None,
        }

    @staticmethod
    def update_resume_offsets(offsets: dict[int, int], item: CompactRoundRecord | TensorRoundRecord) -> None:
        offsets[item.worker_id] = max(offsets.get(item.worker_id, 0), item.worker_sequence + 1)

    def assigned_shards(self, worker_id: int, num_workers: int, epoch: int | None = None) -> tuple[ShardDescriptor, ...]:
        if num_workers < 1 or not 0 <= worker_id < num_workers:
            raise ValueError("worker_id must be in 0..num_workers-1")
        epoch = self.epoch if epoch is None else epoch
        shards = list(self.shards)
        if self.shuffle:
            random.Random(_seed(self.seed, epoch, self.split, 0, "shards")).shuffle(shards)
        global_worker = self.rank * num_workers + worker_id
        global_workers = self.world_size * num_workers
        return tuple(shards[global_worker::global_workers])

    def iter_for_worker(
        self,
        worker_id: int,
        num_workers: int,
        *,
        epoch: int | None = None,
    ) -> Iterator[CompactRoundRecord | TensorRoundRecord]:
        epoch = self.epoch if epoch is None else epoch
        global_worker = self.rank * num_workers + worker_id
        descriptors = self.assigned_shards(worker_id, num_workers, epoch)

        def records() -> Iterator[CompactRoundRecord]:
            for descriptor in descriptors:
                with CompactShardReader(self.root, descriptor) as reader:
                    for record in reader:
                        if self.years is None or record.year in self.years:
                            yield record

        stream: Iterator[CompactRoundRecord]
        if self.shuffle:
            rng = random.Random(_seed(self.seed, epoch, self.split, global_worker, "rounds"))
            stream = _shuffle_buffer(records(), self.shuffle_buffer_rounds, rng)
        else:
            stream = records()
        skip = self.resume_offsets.get(global_worker, 0)
        for sequence, record in enumerate(stream):
            if sequence < skip:
                continue
            record = replace(record, worker_id=global_worker, worker_sequence=sequence)
            yield record.tensorize() if self.tensorize else record

    def __iter__(self) -> Iterator[CompactRoundRecord | TensorRoundRecord]:
        info = get_worker_info()
        if info is None:
            return self.iter_for_worker(0, 1)
        return self.iter_for_worker(info.id, info.num_workers)


def build_streaming_dataloader(
    dataset: StreamingRoundDataset,
    *,
    num_workers: int,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> DataLoader:
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    kwargs = {
        "dataset": dataset,
        "batch_size": None,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers:
        kwargs.update(
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
    return DataLoader(**kwargs)
