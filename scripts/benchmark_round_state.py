#!/usr/bin/env python3
"""Compare legacy replay encoding with the incremental round state engine."""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mahjong_ai.state import (  # noqa: E402
    FEATURE_SCHEMA_VERSION,
    IncrementalStateEncoder,
    RoundState,
)
from mahjong_ai_features import (  # noqa: E402
    FEATURE_SCHEMA_VERSION as LEGACY_SCHEMA_VERSION,
    StateEncoderV2,
)


def _load_rounds(archives: list[Path], limit: int) -> list[list[dict]]:
    rounds: list[list[dict]] = []
    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.is_dir() or not info.filename.lower().endswith((".txt", ".json")):
                    continue
                game = json.loads(zf.read(info).decode("utf-8"))
                for round_log in game.get("log", []):
                    if round_log and isinstance(round_log[0], dict) and "qipai" in round_log[0]:
                        rounds.append(round_log)
                        if len(rounds) >= limit:
                            return rounds
    return rounds


def _legacy(rounds: list[list[dict]]) -> tuple[int, float]:
    samples = 0
    checksum = 0.0
    for round_log in rounds:
        for index, event in enumerate(round_log[1:], 1):
            if "dapai" not in event:
                continue
            player = event["dapai"]["l"]
            checksum += float(StateEncoderV2(round_log, player).encode(index)[0].sum())
            samples += 1
    return samples, checksum


def _incremental(rounds: list[list[dict]]) -> tuple[int, float]:
    samples = 0
    checksum = 0.0
    for round_log in rounds:
        state = RoundState.from_round_log(round_log)
        for event in round_log[1:]:
            if "dapai" in event:
                player = event["dapai"]["l"]
                checksum += float(IncrementalStateEncoder(state, player).encode()[0].sum())
                samples += 1
            state.apply_event(event)
    return samples, checksum


def _measure(callback, rounds: list[list[dict]]) -> tuple[int, float, float]:
    started = time.perf_counter()
    samples, checksum = callback(rounds)
    return samples, checksum, time.perf_counter() - started


def _replay_state_for_every_discard(rounds: list[list[dict]]) -> tuple[int, float]:
    """Isolate the legacy O(decisions * events) replay pattern."""

    applied_events = 0
    checksum = 0.0
    for round_log in rounds:
        for index, event in enumerate(round_log[1:], 1):
            if "dapai" not in event:
                continue
            state = RoundState.from_round_log(round_log)
            for prior in round_log[1:index]:
                state.apply_event(prior)
                applied_events += 1
            checksum += state.event_index
    return applied_events, checksum


def _advance_state_once(rounds: list[list[dict]]) -> tuple[int, float]:
    """Isolate the new O(events) forward-only transition pattern."""

    applied_events = 0
    checksum = 0.0
    for round_log in rounds:
        state = RoundState.from_round_log(round_log)
        for event in round_log[1:]:
            if "dapai" in event:
                checksum += state.event_index
            state.apply_event(event)
            applied_events += 1
    return applied_events, checksum


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--max-rounds", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("round_state_benchmark.json"))
    args = parser.parse_args()
    if args.max_rounds < 1:
        raise SystemExit("--max-rounds must be at least 1")
    for archive in args.archives:
        if not archive.is_file():
            raise SystemExit(f"archive not found: {archive}")

    torch.set_num_threads(1)
    rounds = _load_rounds(args.archives, args.max_rounds)
    if not rounds:
        raise SystemExit("no valid rounds found")

    # Warm up imports and tensor allocators outside the measurements.
    StateEncoderV2(rounds[0], 0).encode(1)
    IncrementalStateEncoder(RoundState.from_round_log(rounds[0]), 0).encode()

    legacy_samples, legacy_checksum, legacy_seconds = _measure(_legacy, rounds)
    incremental_samples, incremental_checksum, incremental_seconds = _measure(
        _incremental, rounds
    )
    if legacy_samples != incremental_samples:
        raise RuntimeError("legacy and incremental sample counts differ")
    replay_events, replay_checksum, replay_seconds = _measure(
        _replay_state_for_every_discard, rounds
    )
    applied_events, forward_checksum, forward_seconds = _measure(
        _advance_state_once, rounds
    )
    if replay_checksum != forward_checksum:
        raise RuntimeError("replayed and forward-only state positions differ")

    report = {
        "schema_version": "round-state-benchmark-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "archives": [str(path.resolve()) for path in args.archives],
        "rounds": len(rounds),
        "discard_samples": legacy_samples,
        "legacy_feature_schema": LEGACY_SCHEMA_VERSION,
        "incremental_feature_schema": FEATURE_SCHEMA_VERSION,
        "legacy": {
            "elapsed_seconds": legacy_seconds,
            "samples_per_second": legacy_samples / legacy_seconds,
        },
        "incremental": {
            "elapsed_seconds": incremental_seconds,
            "samples_per_second": incremental_samples / incremental_seconds,
        },
        "speedup": legacy_seconds / incremental_seconds,
        "observable_hand_checksum_match": legacy_checksum == incremental_checksum,
        "state_transition_benchmark": {
            "replay_for_every_discard": {
                "elapsed_seconds": replay_seconds,
                "applied_events": replay_events,
            },
            "forward_only": {
                "elapsed_seconds": forward_seconds,
                "applied_events": applied_events,
            },
            "event_work_reduction": replay_events / applied_events,
            "speedup": replay_seconds / forward_seconds,
            "position_checksum_match": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
