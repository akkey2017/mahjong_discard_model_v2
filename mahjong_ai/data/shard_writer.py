"""Bounded-memory compact shard writer and checksum verifier."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .normalize import NormalizedGame
from .schema import (
    DATASET_SCHEMA_VERSION,
    EVENT_DTYPE,
    EVENT_TYPES,
    METADATA_DTYPE,
    NO_MELD,
    ROUND_DTYPE,
    SPLIT_IDS,
)


ARRAY_FILES = ("rounds.npy", "events.npy", "melds.npy", "offsets.npy", "metadata.npy")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


@dataclass
class ShardBuffer:
    split: str
    max_rounds: int
    staging_root: Path
    next_shard_id: int
    games: list[NormalizedGame] = field(default_factory=list)
    round_count: int = 0

    def add(self, game: NormalizedGame) -> list[dict]:
        if game.split != self.split:
            raise ValueError(f"game split {game.split!r} does not match writer {self.split!r}")
        completed = []
        game_rounds = len(game.rounds)
        if self.games and self.round_count + game_rounds > self.max_rounds:
            completed.append(self.flush())
        self.games.append(game)
        self.round_count += game_rounds
        if self.round_count >= self.max_rounds:
            completed.append(self.flush())
        return completed

    def flush(self) -> dict:
        if not self.games:
            raise ValueError("cannot flush an empty shard")
        shard_name = f"shard_{self.next_shard_id:06d}"
        relative_path = f"{self.split}/{shard_name}"
        shard_dir = self.staging_root / relative_path
        shard_dir.mkdir(parents=True, exist_ok=False)

        metadata = np.zeros(len(self.games), dtype=METADATA_DTYPE)
        total_rounds = sum(len(game.rounds) for game in self.games)
        rounds = np.zeros(total_rounds, dtype=ROUND_DTYPE)
        event_parts: list[np.ndarray] = []
        meld_parts: list[np.ndarray] = []
        offsets = np.zeros(total_rounds + 1, dtype=np.uint64)

        round_cursor = 0
        event_cursor = 0
        meld_cursor = 0
        for game_index, game in enumerate(self.games):
            metadata[game_index] = (
                game.game_id.encode("ascii"),
                game.archive_index,
                game.year,
                SPLIT_IDS[game.split],
                len(game.rounds),
                game.source_crc32,
                game.source_size,
            )
            for normalized_round in game.rounds:
                rounds[round_cursor] = (
                    game_index,
                    normalized_round.round_index,
                    normalized_round.round_wind,
                    normalized_round.dealer,
                    normalized_round.honba,
                    normalized_round.kyotaku,
                    normalized_round.scores,
                    normalized_round.hands,
                    normalized_round.initial_dora,
                )
                events = normalized_round.events.copy()
                has_meld = events["meld_offset"] != NO_MELD
                events["meld_offset"][has_meld] += meld_cursor
                event_parts.append(events)
                meld_parts.append(normalized_round.meld_tiles)
                event_cursor += len(events)
                meld_cursor += len(normalized_round.meld_tiles)
                round_cursor += 1
                offsets[round_cursor] = event_cursor

        events = np.concatenate(event_parts) if event_parts else np.empty(0, dtype=EVENT_DTYPE)
        melds = np.concatenate(meld_parts) if meld_parts else np.empty(0, dtype=np.uint8)
        arrays = {
            "rounds.npy": rounds,
            "events.npy": events,
            "melds.npy": melds,
            "offsets.npy": offsets,
            "metadata.npy": metadata,
        }
        for filename, array in arrays.items():
            np.save(shard_dir / filename, array, allow_pickle=False)

        file_checksums = {
            filename: {
                "bytes": (shard_dir / filename).stat().st_size,
                "sha256": _sha256(shard_dir / filename),
            }
            for filename in ARRAY_FILES
        }
        checksum = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "split": self.split,
            "shard_id": self.next_shard_id,
            "counts": {
                "games": len(metadata),
                "rounds": len(rounds),
                "events": len(events),
                "meld_tiles": len(melds),
            },
            "files": file_checksums,
        }
        checksum_bytes = _canonical_json(checksum)
        (shard_dir / "checksum.json").write_bytes(checksum_bytes)
        total_bytes = sum(item["bytes"] for item in file_checksums.values()) + len(checksum_bytes)
        descriptor = {
            "path": relative_path,
            "split": self.split,
            "id": self.next_shard_id,
            "games": len(metadata),
            "rounds": len(rounds),
            "events": len(events),
            "meld_tiles": len(melds),
            "bytes": total_bytes,
            "checksum_sha256": hashlib.sha256(checksum_bytes).hexdigest(),
        }
        self.next_shard_id += 1
        self.games = []
        self.round_count = 0
        return descriptor


def flush_all(buffers: Iterable[ShardBuffer]) -> list[dict]:
    descriptors = []
    for buffer in buffers:
        if buffer.games:
            descriptors.append(buffer.flush())
    return descriptors


def verify_shard(shard_dir: Path) -> dict:
    checksum_path = shard_dir / "checksum.json"
    checksum_bytes = checksum_path.read_bytes()
    checksum = json.loads(checksum_bytes)
    if checksum.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError(f"unsupported shard schema: {checksum.get('schema_version')!r}")
    for filename in ARRAY_FILES:
        expected = checksum["files"][filename]
        path = shard_dir / filename
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"size mismatch: {path}")
        if _sha256(path) != expected["sha256"]:
            raise ValueError(f"checksum mismatch: {path}")

    rounds = np.load(shard_dir / "rounds.npy", mmap_mode="r", allow_pickle=False)
    events = np.load(shard_dir / "events.npy", mmap_mode="r", allow_pickle=False)
    melds = np.load(shard_dir / "melds.npy", mmap_mode="r", allow_pickle=False)
    offsets = np.load(shard_dir / "offsets.npy", mmap_mode="r", allow_pickle=False)
    metadata = np.load(shard_dir / "metadata.npy", mmap_mode="r", allow_pickle=False)
    if rounds.dtype != ROUND_DTYPE or events.dtype != EVENT_DTYPE or metadata.dtype != METADATA_DTYPE:
        raise ValueError(f"dtype mismatch in {shard_dir}")
    if melds.dtype != np.dtype("u1") or offsets.dtype != np.dtype("<u8"):
        raise ValueError(f"auxiliary dtype mismatch in {shard_dir}")
    if len(offsets) != len(rounds) + 1 or offsets[0] != 0 or offsets[-1] != len(events):
        raise ValueError(f"invalid offsets in {shard_dir}")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(f"non-monotonic offsets in {shard_dir}")
    if len(rounds) and (len(metadata) == 0 or int(rounds["game_index"].max()) >= len(metadata)):
        raise ValueError(f"invalid game index in {shard_dir}")
    if len(events):
        if not np.all(np.isin(events["type"], list(EVENT_TYPES.values()))):
            raise ValueError(f"unknown event type in {shard_dir}")
        meld_offsets = events["meld_offset"]
        used = meld_offsets != NO_MELD
        if np.any(meld_offsets[used] >= len(melds)):
            raise ValueError(f"invalid meld offset in {shard_dir}")
    counts = checksum["counts"]
    actual = {
        "games": len(metadata),
        "rounds": len(rounds),
        "events": len(events),
        "meld_tiles": len(melds),
    }
    if counts != actual:
        raise ValueError(f"count mismatch in {shard_dir}: {counts} != {actual}")
    return {
        **actual,
        "bytes": sum((shard_dir / name).stat().st_size for name in ARRAY_FILES)
        + checksum_path.stat().st_size,
        "checksum_sha256": hashlib.sha256(checksum_bytes).hexdigest(),
    }


def verify_dataset(manifest_path: Path) -> dict:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema: {manifest.get('schema_version')!r}")
    totals = {"games": 0, "rounds": 0, "events": 0, "meld_tiles": 0, "bytes": 0}
    for descriptor in manifest.get("shards", []):
        result = verify_shard(manifest_path.parent / descriptor["path"])
        if result["checksum_sha256"] != descriptor["checksum_sha256"]:
            raise ValueError(f"manifest checksum mismatch: {descriptor['path']}")
        for key in totals:
            totals[key] += result[key]
    return {"shards": len(manifest.get("shards", [])), **totals}
