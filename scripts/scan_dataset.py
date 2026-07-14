#!/usr/bin/env python3
"""Scan raw mahjong game archives without materializing the dataset in RAM.

The scanner intentionally uses only the Python standard library.  It reads one
archive member at a time, counts structural and task-related statistics, and
can repeat the scan with different process counts to measure worker scaling.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


KNOWN_EVENTS = (
    "qipai",
    "zimo",
    "dapai",
    "fulou",
    "gang",
    "gangzimo",
    "kaigang",
    "lizhi",
    "hule",
    "pingju",
)
YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


@dataclass(frozen=True)
class EstimateConfig:
    """Byte model for the future compact prepared representation."""

    event_bytes: int = 8
    round_bytes: int = 256
    offset_bytes: int = 8
    game_metadata_bytes: int = 64


def _empty_counts() -> dict:
    return {
        "archive_members": 0,
        "files_scanned": 0,
        "files_valid_json": 0,
        "games": 0,
        "rounds": 0,
        "events": 0,
        "raw_uncompressed_bytes": 0,
        "malformed_json": 0,
        "decode_errors": 0,
        "read_errors": 0,
        "invalid_game_structure": 0,
        "invalid_round_structure": 0,
        "invalid_event_structure": 0,
        "event_counts": Counter(),
        "task_candidates": Counter(),
        "task_positives": Counter(),
        "years": Counter(),
        "rules": Counter(),
        "archives": Counter(),
        "malformed_examples": [],
    }


def _rule_label(game: dict) -> str:
    rule = game.get("rule", game.get("rules"))
    if rule is None:
        return "unknown"
    if isinstance(rule, str):
        return rule or "unknown"
    try:
        return json.dumps(rule, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(rule)


def _year_label(archive: str, member: str, game: dict) -> str:
    # The archive name is authoritative for the raw yearly dumps.  Some older
    # archives contain member names with unrelated 20xx fragments, which would
    # otherwise inflate the number of observed years and corrupt projections.
    for candidate in (
        game.get("year"),
        Path(archive).name,
        member,
        game.get("date"),
    ):
        if candidate is None:
            continue
        match = YEAR_RE.search(str(candidate))
        if match:
            return match.group(1)
    return "unknown"


def _record_example(counts: dict, limit: int, archive: str, member: str, error: str) -> None:
    if len(counts["malformed_examples"]) < limit:
        counts["malformed_examples"].append(
            {"archive": archive, "member": member, "error": error[:500]}
        )


def _scan_game(counts: dict, game: object, archive: str, member: str) -> None:
    if not isinstance(game, dict) or not isinstance(game.get("log"), list):
        counts["invalid_game_structure"] += 1
        return

    counts["games"] += 1
    counts["years"][_year_label(archive, member, game)] += 1
    counts["rules"][_rule_label(game)] += 1

    for round_log in game["log"]:
        if not isinstance(round_log, list):
            counts["invalid_round_structure"] += 1
            continue
        counts["rounds"] += 1
        for event in round_log:
            if not isinstance(event, dict):
                counts["invalid_event_structure"] += 1
                continue
            counts["events"] += 1
            event_types = [key for key in KNOWN_EVENTS if key in event]
            if not event_types:
                counts["event_counts"]["unknown"] += 1
                continue
            for event_type in event_types:
                counts["event_counts"][event_type] += 1

            if "dapai" in event:
                counts["task_candidates"]["dapai"] += 1
                counts["task_candidates"]["riichi"] += 1
                counts["task_positives"]["dapai"] += 1
                payload = event.get("dapai")
                if isinstance(payload, dict) and "*" in str(payload.get("p", "")):
                    counts["task_positives"]["riichi"] += 1
            if "fulou" in event:
                counts["task_candidates"]["fulou"] += 1
                counts["task_positives"]["fulou"] += 1
            if "gang" in event:
                counts["task_candidates"]["gang"] += 1
                counts["task_positives"]["gang"] += 1
            if "hule" in event:
                counts["task_candidates"]["hule"] += 1
                counts["task_positives"]["hule"] += 1


def _scan_chunk(task: tuple[str, list[str], int]) -> dict:
    archive, members, malformed_limit = task
    counts = _empty_counts()
    counts["archive_members"] = len(members)
    counts["archives"][Path(archive).name] += len(members)
    try:
        zf = zipfile.ZipFile(archive, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        counts["read_errors"] += len(members) or 1
        _record_example(counts, malformed_limit, archive, "<archive>", repr(exc))
        return counts

    with zf:
        for member in members:
            counts["files_scanned"] += 1
            try:
                raw = zf.read(member)
                counts["raw_uncompressed_bytes"] += len(raw)
            except (OSError, KeyError, RuntimeError, zipfile.BadZipFile) as exc:
                counts["read_errors"] += 1
                _record_example(counts, malformed_limit, archive, member, repr(exc))
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                counts["decode_errors"] += 1
                _record_example(counts, malformed_limit, archive, member, repr(exc))
                continue
            try:
                game = json.loads(text)
            except json.JSONDecodeError as exc:
                counts["malformed_json"] += 1
                _record_example(counts, malformed_limit, archive, member, str(exc))
                continue
            counts["files_valid_json"] += 1
            _scan_game(counts, game, archive, member)
    return counts


def _merge_counts(parts: Iterable[dict], malformed_limit: int) -> dict:
    total = _empty_counts()
    scalar_keys = (
        "archive_members", "files_scanned", "files_valid_json", "games", "rounds", "events",
        "raw_uncompressed_bytes", "malformed_json", "decode_errors", "read_errors",
        "invalid_game_structure", "invalid_round_structure", "invalid_event_structure",
    )
    counter_keys = (
        "event_counts", "task_candidates", "task_positives", "years", "rules", "archives",
    )
    for part in parts:
        for key in scalar_keys:
            total[key] += part[key]
        for key in counter_keys:
            total[key].update(part[key])
        remaining = malformed_limit - len(total["malformed_examples"])
        if remaining > 0:
            total["malformed_examples"].extend(part["malformed_examples"][:remaining])
    return total


def discover_archives(inputs: Sequence[str]) -> list[Path]:
    """Resolve ZIP paths and recursively discover ZIPs in supplied directories."""

    archives: set[Path] = set()
    for value in inputs:
        path = Path(value).expanduser()
        if path.is_dir():
            archives.update(p.resolve() for p in path.rglob("*.zip") if p.is_file())
        elif path.is_file() and path.suffix.lower() == ".zip":
            archives.add(path.resolve())
        else:
            raise FileNotFoundError(f"ZIP archive or directory not found: {path}")
    if not archives:
        raise FileNotFoundError("No ZIP archives were found")
    return sorted(archives)


def _member_index(archives: Sequence[Path], max_files: int | None) -> list[tuple[str, list[str]]]:
    indexed: list[tuple[str, list[str]]] = []
    remaining = max_files
    for archive in archives:
        try:
            with zipfile.ZipFile(archive, "r") as zf:
                members = [
                    info.filename
                    for info in zf.infolist()
                    if not info.is_dir() and info.filename.lower().endswith((".txt", ".json"))
                ]
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Invalid ZIP archive: {archive}") from exc
        if remaining is not None:
            members = members[:remaining]
            remaining -= len(members)
        indexed.append((str(archive), members))
        if remaining == 0:
            break
    return indexed


def _make_tasks(indexed: Sequence[tuple[str, list[str]]], workers: int,
                malformed_limit: int) -> list[tuple[str, list[str], int]]:
    tasks = []
    for archive, members in indexed:
        if not members:
            continue
        chunks = min(workers, len(members))
        chunk_size = math.ceil(len(members) / chunks)
        tasks.extend(
            (archive, members[start:start + chunk_size], malformed_limit)
            for start in range(0, len(members), chunk_size)
        )
    return tasks


def scan_index(indexed: Sequence[tuple[str, list[str]]], workers: int = 1,
               malformed_limit: int = 20) -> tuple[dict, float]:
    """Scan a pre-built member index and return counters plus elapsed seconds."""

    if workers < 1:
        raise ValueError("workers must be at least 1")
    tasks = _make_tasks(indexed, workers, malformed_limit)
    started = time.perf_counter()
    if workers == 1:
        parts = [_scan_chunk(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            parts = list(executor.map(_scan_chunk, tasks))
    return _merge_counts(parts, malformed_limit), time.perf_counter() - started


def _estimated_prepared_bytes(counts: dict, config: EstimateConfig) -> int:
    return (
        counts["events"] * config.event_bytes
        + counts["rounds"] * (config.round_bytes + config.offset_bytes)
        + counts["games"] * config.game_metadata_bytes
        + config.offset_bytes
    )


def _jsonable_counts(counts: dict) -> dict:
    output = dict(counts)
    for key in ("event_counts", "task_candidates", "task_positives", "years", "rules", "archives"):
        output[key] = dict(sorted(counts[key].items()))
    return output


def build_report(
    archives: Sequence[Path],
    counts: dict,
    elapsed: float,
    workers: int,
    benchmarks: list[dict],
    estimate: EstimateConfig,
    projection_years: int,
) -> dict:
    prepared_bytes = _estimated_prepared_bytes(counts, estimate)
    known_years = [year for year in counts["years"] if year != "unknown"]
    observed_years = len(known_years) or 1
    projection_factor = projection_years / observed_years
    malformed_total = (
        counts["malformed_json"] + counts["decode_errors"] + counts["read_errors"]
        + counts["invalid_game_structure"]
    )
    attempted = counts["files_scanned"]
    return {
        "schema_version": "dataset-scan-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": [str(path) for path in archives],
        "scan": {
            "workers": workers,
            "elapsed_seconds": elapsed,
            "files_per_second": counts["files_scanned"] / elapsed if elapsed else 0.0,
            "games_per_second": counts["games"] / elapsed if elapsed else 0.0,
            "rounds_per_second": counts["rounds"] / elapsed if elapsed else 0.0,
            "events_per_second": counts["events"] / elapsed if elapsed else 0.0,
        },
        "counts": _jsonable_counts(counts),
        "quality": {
            "malformed_or_unreadable": malformed_total,
            "malformed_rate": malformed_total / attempted if attempted else 0.0,
        },
        "prepared_size_estimate": {
            "bytes": prepared_bytes,
            "gib": prepared_bytes / (1024 ** 3),
            "model": {
                "event_bytes": estimate.event_bytes,
                "round_bytes": estimate.round_bytes,
                "offset_bytes": estimate.offset_bytes,
                "game_metadata_bytes": estimate.game_metadata_bytes,
                "compression": "not_applied",
            },
        },
        "task_count_semantics": {
            "dapai": "one candidate and one positive for each observed discard",
            "riichi": "one candidate per discard; positives are discards marked with '*'",
            "fulou": "observed positive calls only; legal pass opportunities require state replay",
            "gang": "observed positive declarations only; legal pass opportunities require state replay",
            "hule": "observed positive wins only; legal pass opportunities require hand evaluation",
        },
        "projection": {
            "target_years": projection_years,
            "observed_distinct_known_years": len(known_years),
            "assumed_observed_years": observed_years,
            "scale_factor": projection_factor,
            "estimated_files": round(counts["files_scanned"] * projection_factor),
            "estimated_games": round(counts["games"] * projection_factor),
            "estimated_rounds": round(counts["rounds"] * projection_factor),
            "estimated_events": round(counts["events"] * projection_factor),
            "estimated_prepared_bytes": round(prepared_bytes * projection_factor),
            "estimated_prepared_gib": prepared_bytes * projection_factor / (1024 ** 3),
            "estimated_scan_seconds_at_current_rate": elapsed * projection_factor,
        },
        "benchmarks": benchmarks,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan mahjong ZIP datasets and write a reproducible JSON report."
    )
    parser.add_argument("inputs", nargs="+", help="ZIP archives or directories containing ZIPs")
    parser.add_argument("--output", type=Path, default=Path("dataset_scan_report.json"))
    parser.add_argument(
        "--workers", type=int, nargs="+", default=[1], metavar="N",
        help="Worker counts to benchmark; the last run supplies report counts (default: 1)",
    )
    parser.add_argument("--max-files", type=int, help="Scan at most this many members in total")
    parser.add_argument("--malformed-examples", type=int, default=20)
    parser.add_argument("--projection-years", type=int, default=10)
    parser.add_argument("--event-bytes", type=int, default=8)
    parser.add_argument("--round-bytes", type=int, default=256)
    parser.add_argument("--offset-bytes", type=int, default=8)
    parser.add_argument("--game-metadata-bytes", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_files is not None and args.max_files < 1:
        raise SystemExit("--max-files must be at least 1")
    if any(worker < 1 for worker in args.workers):
        raise SystemExit("--workers values must be at least 1")
    if args.projection_years < 1:
        raise SystemExit("--projection-years must be at least 1")
    if args.malformed_examples < 0:
        raise SystemExit("--malformed-examples cannot be negative")
    estimate_values = (
        args.event_bytes, args.round_bytes, args.offset_bytes, args.game_metadata_bytes,
    )
    if any(value < 0 for value in estimate_values):
        raise SystemExit("prepared-size byte estimates cannot be negative")

    archives = discover_archives(args.inputs)
    indexed = _member_index(archives, args.max_files)
    member_count = sum(len(members) for _, members in indexed)
    if member_count == 0:
        raise SystemExit("No .txt or .json members found in the supplied archives")

    print(f"Discovered {len(archives)} archive(s), {member_count:,} member(s)", file=sys.stderr)
    benchmarks = []
    final_counts = None
    final_elapsed = 0.0
    for workers in args.workers:
        print(f"Scanning with {workers} worker(s)...", file=sys.stderr)
        counts, elapsed = scan_index(indexed, workers, args.malformed_examples)
        benchmarks.append({
            "workers": workers,
            "elapsed_seconds": elapsed,
            "files_per_second": counts["files_scanned"] / elapsed if elapsed else 0.0,
            "events_per_second": counts["events"] / elapsed if elapsed else 0.0,
        })
        final_counts, final_elapsed = counts, elapsed

    assert final_counts is not None
    estimate = EstimateConfig(
        event_bytes=args.event_bytes,
        round_bytes=args.round_bytes,
        offset_bytes=args.offset_bytes,
        game_metadata_bytes=args.game_metadata_bytes,
    )
    report = build_report(
        archives, final_counts, final_elapsed, args.workers[-1], benchmarks,
        estimate, args.projection_years,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"Scanned {final_counts['games']:,} games, {final_counts['rounds']:,} rounds, "
        f"{final_counts['events']:,} events; report: {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
