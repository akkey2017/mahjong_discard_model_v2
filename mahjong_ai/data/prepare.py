"""Multiprocess, resumable preparation of compact mahjong round shards."""

from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import shutil
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from mahjong_ai.state import FEATURE_SCHEMA_VERSION

from .normalize import NormalizedGame, normalize_game
from .schema import DATASET_SCHEMA_VERSION, SPLIT_NAMES
from .shard_writer import ShardBuffer, flush_all, verify_dataset


YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


@dataclass(frozen=True)
class Member:
    name: str
    crc32: int
    size: int


@dataclass(frozen=True)
class Archive:
    path: Path
    name: str
    year: int
    members: tuple[Member, ...]
    size: int
    index_sha256: str


@dataclass(frozen=True)
class NormalizeTask:
    archive_path: str
    archive_name: str
    archive_index: int
    year: int
    seed: int
    members: tuple[Member, ...]


@dataclass(frozen=True)
class NormalizeResult:
    games: tuple[NormalizedGame, ...]
    errors: tuple[dict, ...]


def _atomic_json(path: Path, value: object) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _archive_year(path: Path) -> int:
    match = YEAR_RE.search(path.name)
    if not match:
        raise ValueError(f"archive filename must contain a year: {path.name}")
    return int(match.group(1))


def _index_archive(path: Path) -> Archive:
    digest = hashlib.sha256()
    members = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.lower().endswith((".txt", ".json")):
                continue
            member = Member(info.filename, info.CRC, info.file_size)
            members.append(member)
            digest.update(info.filename.encode("utf-8"))
            digest.update(info.CRC.to_bytes(4, "little"))
            digest.update(info.file_size.to_bytes(8, "little"))
            digest.update(info.compress_size.to_bytes(8, "little"))
    if not members:
        raise ValueError(f"archive contains no JSON/TXT members: {path}")
    return Archive(
        path=path.resolve(),
        name=path.name,
        year=_archive_year(path),
        members=tuple(members),
        size=path.stat().st_size,
        index_sha256=digest.hexdigest(),
    )


def index_archives(inputs: Iterable[Path], max_files: int | None = None) -> list[Archive]:
    paths = sorted({Path(path).expanduser().resolve() for path in inputs})
    archives = [_index_archive(path) for path in paths]
    if len({archive.name for archive in archives}) != len(archives):
        raise ValueError("archive basenames must be unique")
    if max_files is None:
        return archives
    remaining = max_files
    limited = []
    for archive in archives:
        selected = archive.members[:remaining]
        limited.append(Archive(
            path=archive.path,
            name=archive.name,
            year=archive.year,
            members=selected,
            size=archive.size,
            index_sha256=archive.index_sha256,
        ))
        remaining -= len(selected)
        if remaining == 0:
            break
    return limited


def _normalize_task(task: NormalizeTask) -> NormalizeResult:
    games = []
    errors = []
    try:
        archive = zipfile.ZipFile(task.archive_path)
    except (OSError, zipfile.BadZipFile) as exc:
        return NormalizeResult((), tuple(
            {
                "archive": task.archive_name,
                "member": member.name,
                "error": f"{type(exc).__name__}: {exc}",
            }
            for member in task.members
        ))
    with archive:
        for member in task.members:
            try:
                raw = archive.read(member.name)
                game = json.loads(raw.decode("utf-8"))
                games.append(normalize_game(
                    game,
                    archive_name=task.archive_name,
                    archive_index=task.archive_index,
                    member=member.name,
                    year=task.year,
                    seed=task.seed,
                    source_crc32=member.crc32,
                    source_size=member.size,
                ))
            except Exception as exc:  # isolated per raw member by design
                errors.append({
                    "archive": task.archive_name,
                    "member": member.name,
                    "error": f"{type(exc).__name__}: {exc}"[:1000],
                })
    return NormalizeResult(tuple(games), tuple(errors))


def _chunks(values: tuple[Member, ...], size: int) -> Iterator[tuple[Member, ...]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _run_tasks(tasks: list[NormalizeTask], workers: int) -> Iterator[NormalizeResult]:
    if workers == 1:
        yield from map(_normalize_task, tasks)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(
            _normalize_task,
            tasks,
            chunksize=1,
            buffersize=max(2, workers * 2),
        )


def _input_descriptors(archives: list[Archive]) -> list[dict]:
    return [{
        "name": archive.name,
        "year": archive.year,
        "bytes": archive.size,
        "members": len(archive.members),
        "zip_index_sha256": archive.index_sha256,
    } for archive in archives]


def _base_manifest(archives: list[Archive], config: dict) -> dict:
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "format": {
            "rounds": "rounds.npy",
            "events": "events.npy",
            "melds": "melds.npy",
            "offsets": "offsets.npy",
            "metadata": "metadata.npy",
            "checksum": "checksum.json",
        },
        "inputs": _input_descriptors(archives),
        "config": config,
        "split": {
            "algorithm": "sha256(seed:archive:member) % 10000",
            "train": [0, 9799],
            "validation": [9800, 9899],
            "test": [9900, 9999],
        },
        "rule": "unknown",
        "shards": [],
        "corrupted_files": [],
        "totals": {},
        "complete": False,
    }


def _base_progress(archives: list[Archive]) -> dict:
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "archive_positions": [0] * len(archives),
        "next_shard_ids": {split: 0 for split in SPLIT_NAMES},
        "complete": False,
    }


def _totals(manifest: dict, progress: dict) -> dict:
    totals = {
        "source_files_processed": sum(progress["archive_positions"]),
        "corrupted_files": len(manifest["corrupted_files"]),
        "games": 0,
        "rounds": 0,
        "events": 0,
        "meld_tiles": 0,
        "bytes": 0,
        "by_split": {
            split: {"shards": 0, "games": 0, "rounds": 0, "events": 0, "bytes": 0}
            for split in SPLIT_NAMES
        },
    }
    for shard in manifest["shards"]:
        split = shard["split"]
        for key in ("games", "rounds", "events", "meld_tiles", "bytes"):
            totals[key] += shard[key]
        totals["by_split"][split]["shards"] += 1
        for key in ("games", "rounds", "events", "bytes"):
            totals["by_split"][split][key] += shard[key]
    return totals


def _write_corrupted(root: Path, records: list[dict]) -> None:
    path = root / "corrupted.jsonl"
    temp = root / ".corrupted.jsonl.tmp"
    with temp.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temp, path)


def _complete_pending(root: Path) -> None:
    pending_path = root / ".pending_commit.json"
    if not pending_path.exists():
        return
    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    staging = root / ".staging_batch"
    for shard in pending["new_shards"]:
        source = staging / shard["path"]
        destination = root / shard["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            continue
        if not source.exists():
            raise RuntimeError(f"pending shard is missing: {source}")
        os.replace(source, destination)
    _atomic_json(root / "manifest.json", pending["manifest"])
    _atomic_json(root / "progress.json", pending["progress"])
    _write_corrupted(root, pending["manifest"]["corrupted_files"])
    pending_path.unlink()
    if staging.exists():
        shutil.rmtree(staging)


def _commit_batch(
    root: Path,
    manifest: dict,
    progress: dict,
    descriptors: list[dict],
    errors: list[dict],
) -> None:
    updated_manifest = json.loads(json.dumps(manifest))
    updated_progress = json.loads(json.dumps(progress))
    updated_manifest["shards"].extend(descriptors)
    updated_manifest["shards"].sort(key=lambda item: (SPLIT_NAMES.index(item["split"]), item["id"]))
    updated_manifest["corrupted_files"].extend(errors)
    updated_manifest["totals"] = _totals(updated_manifest, updated_progress)
    updated_manifest["complete"] = updated_progress["complete"]
    pending = {
        "manifest": updated_manifest,
        "progress": updated_progress,
        "new_shards": descriptors,
    }
    _atomic_json(root / ".pending_commit.json", pending)
    _complete_pending(root)


def prepare_dataset(
    inputs: Iterable[Path],
    output: Path,
    *,
    workers: int = 1,
    rounds_per_shard: int = 4096,
    checkpoint_members: int = 4096,
    chunk_size: int = 8,
    seed: int = 0,
    max_files: int | None = None,
    resume: bool = False,
    stop_after_checkpoints: int | None = None,
) -> dict:
    if workers < 1 or rounds_per_shard < 1 or checkpoint_members < 1 or chunk_size < 1:
        raise ValueError("workers, shard/checkpoint sizes, and chunk size must be positive")
    if max_files is not None and max_files < 1:
        raise ValueError("max_files must be positive")
    if stop_after_checkpoints is not None and stop_after_checkpoints < 1:
        raise ValueError("stop_after_checkpoints must be positive")
    archives = index_archives(inputs, max_files=max_files)
    if not archives:
        raise ValueError("no input archives")
    output = Path(output)
    manifest_path = output / "manifest.json"
    progress_path = output / "progress.json"
    config = {
        "seed": seed,
        "rounds_per_shard": rounds_per_shard,
        "checkpoint_members": checkpoint_members,
        "chunk_size": chunk_size,
        "max_files": max_files,
    }

    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"output exists; pass --resume: {output}")
        _complete_pending(output)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        expected = _base_manifest(archives, config)
        for key in ("schema_version", "feature_schema_version", "inputs", "config", "split"):
            if manifest.get(key) != expected.get(key):
                raise ValueError(f"resume configuration mismatch: {key}")
        if progress.get("complete"):
            verify_dataset(manifest_path)
            return manifest
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        manifest = _base_manifest(archives, config)
        progress = _base_progress(archives)
        _atomic_json(manifest_path, manifest)
        _atomic_json(progress_path, progress)
        _write_corrupted(output, [])

    _complete_pending(output)
    started = time.perf_counter()
    checkpoints_completed = 0
    for archive_index, archive in enumerate(archives):
        while progress["archive_positions"][archive_index] < len(archive.members):
            start = progress["archive_positions"][archive_index]
            end = min(len(archive.members), start + checkpoint_members)
            selected = archive.members[start:end]
            staging = output / ".staging_batch"
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)
            buffers = {
                split: ShardBuffer(
                    split=split,
                    max_rounds=rounds_per_shard,
                    staging_root=staging,
                    next_shard_id=progress["next_shard_ids"][split],
                )
                for split in SPLIT_NAMES
            }
            tasks = [NormalizeTask(
                archive_path=str(archive.path),
                archive_name=archive.name,
                archive_index=archive_index,
                year=archive.year,
                seed=seed,
                members=chunk,
            ) for chunk in _chunks(selected, chunk_size)]
            descriptors: list[dict] = []
            errors: list[dict] = []
            for result in _run_tasks(tasks, workers):
                errors.extend(result.errors)
                for game in result.games:
                    descriptors.extend(buffers[game.split].add(game))
            descriptors.extend(flush_all(buffers.values()))

            next_progress = json.loads(json.dumps(progress))
            next_progress["archive_positions"][archive_index] = end
            next_progress["next_shard_ids"] = {
                split: buffers[split].next_shard_id for split in SPLIT_NAMES
            }
            next_progress["complete"] = all(
                position == len(indexed.members)
                for position, indexed in zip(next_progress["archive_positions"], archives)
            )
            progress = next_progress
            _commit_batch(output, manifest, progress, descriptors, errors)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            elapsed = time.perf_counter() - started
            rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
            print(
                f"Prepared {manifest['totals']['source_files_processed']:,}/"
                f"{sum(len(item.members) for item in archives):,} files; "
                f"{manifest['totals']['rounds']:,} rounds, "
                f"{manifest['totals']['events']:,} events, "
                f"{len(manifest['shards']):,} shards; "
                f"elapsed={elapsed:.1f}s parent_peak_rss={rss_gib:.2f}GiB",
                flush=True,
            )
            checkpoints_completed += 1
            if (
                stop_after_checkpoints is not None
                and checkpoints_completed >= stop_after_checkpoints
                and not progress["complete"]
            ):
                verify_dataset(manifest_path)
                return manifest
    verify_dataset(manifest_path)
    return manifest
