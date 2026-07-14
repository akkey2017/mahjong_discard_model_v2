#!/usr/bin/env python3
"""Benchmark complete split iteration across DataLoader worker counts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mahjong_ai.data.streaming_dataset import (  # noqa: E402
    StreamingRoundDataset,
    build_streaming_dataloader,
)


MASK_128 = (1 << 128) - 1


def _id_value(sample_id: str) -> int:
    return int.from_bytes(hashlib.sha256(sample_id.encode()).digest()[:16], "little")


def _run(manifest: Path, split: str, workers: int, pin_memory: bool) -> dict:
    dataset = StreamingRoundDataset(
        manifest,
        split=split,
        seed=0,
        shuffle=False,
        shuffle_buffer_rounds=0,
        tensorize=True,
    )
    loader = build_streaming_dataloader(
        dataset,
        num_workers=workers,
        prefetch_factor=2,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )
    started = time.perf_counter()
    first_seconds = None
    count = 0
    xor_digest = 0
    sum_digest = 0
    events = 0
    worker_counts: dict[int, int] = {}
    for item in loader:
        if first_seconds is None:
            first_seconds = time.perf_counter() - started
        value = _id_value(item.sample_id)
        xor_digest ^= value
        sum_digest = (sum_digest + value) & MASK_128
        count += 1
        events += item.events.shape[0]
        worker_counts[item.worker_id] = worker_counts.get(item.worker_id, 0) + 1
    elapsed = time.perf_counter() - started
    del loader
    gc.collect()
    return {
        "workers": workers,
        "rounds": count,
        "events": events,
        "elapsed_seconds": elapsed,
        "first_round_seconds": first_seconds,
        "rounds_per_second": count / elapsed,
        "events_per_second": events / elapsed,
        "id_xor_128": f"{xor_digest:032x}",
        "id_sum_128": f"{sum_digest:032x}",
        "worker_round_counts": worker_counts,
        "peak_parent_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 12, 16])
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=Path("streaming_benchmark.json"))
    args = parser.parse_args()
    results = []
    for workers in args.workers:
        if workers < 0:
            raise SystemExit("worker counts cannot be negative")
        result = _run(args.manifest, args.split, workers, args.pin_memory)
        results.append(result)
        print(json.dumps(result), flush=True)
    expected = results[0]
    for result in results[1:]:
        for key in ("rounds", "events", "id_xor_128", "id_sum_128"):
            if result[key] != expected[key]:
                raise RuntimeError(f"worker result mismatch for {key}: {result} != {expected}")
    report = {
        "schema_version": "streaming-benchmark-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "split": args.split,
        "pin_memory": args.pin_memory,
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
