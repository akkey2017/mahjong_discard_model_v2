#!/usr/bin/env python3
"""Count unified Phase-5 targets over prepared shards without feature expansion."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mahjong_ai.data.multitask import (  # noqa: E402
    NegativeSamplingConfig,
    TASK_INDEX,
    TASK_NAMES,
    MultiTaskSampleBuilder,
    TaskSamplingPolicy,
)
from mahjong_ai.data.schema import (  # noqa: E402
    DAPAI_RIICHI,
    EVENT_TYPE_NAMES,
)
from mahjong_ai.data.streaming_dataset import StreamingRoundDataset  # noqa: E402


MASK = (1 << 128) - 1


def _digest(sample_id: str) -> int:
    return int.from_bytes(hashlib.sha256(sample_id.encode()).digest()[:16], "little")


def _count_partition(args: tuple[str, str, int, int, bool, bool, int, tuple[int, ...] | None]) -> dict:
    (
        manifest,
        split,
        rank,
        world_size,
        include_fulou_negatives,
        training_sampling,
        seed,
        years,
    ) = args
    rounds = StreamingRoundDataset(
        Path(manifest),
        split=split,
        seed=0,
        shuffle=False,
        shuffle_buffer_rounds=0,
        rank=rank,
        world_size=world_size,
        years=years,
        tensorize=False,
    )
    negative_sampling = None
    if training_sampling:
        negative_sampling = NegativeSamplingConfig(
            seed=seed,
            policies={
                "riichi": TaskSamplingPolicy(keep_probability=0.25),
                "fulou": TaskSamplingPolicy(max_negative_per_positive=4.0),
                "gang": TaskSamplingPolicy(keep_probability=1.0),
            },
        )
    builder = MultiTaskSampleBuilder(
        split=split,
        negative_sampling=negative_sampling,
        include_fulou_negatives=include_fulou_negatives,
        encode_features=False,
    )
    task_total = Counter()
    task_positive = Counter()
    task_negative = Counter()
    event_counts = Counter()
    record_count = 0
    multi_label_records = 0
    digest_xor = 0
    digest_sum = 0
    round_count = 0
    started = time.perf_counter()

    for record in rounds.iter_for_worker(0, 1):
        round_count += 1
        for event in record.events:
            kind = EVENT_TYPE_NAMES[int(event["type"])]
            event_counts[kind] += 1
            if kind == "dapai" and int(event["flags"]) & DAPAI_RIICHI:
                event_counts["riichi_positive"] += 1
        for sample in builder.build_round(record):
            record_count += 1
            active_count = int(sample.target.masks.sum())
            multi_label_records += int(active_count > 1)
            value = _digest(sample.sample_id)
            digest_xor ^= value
            digest_sum = (digest_sum + value) & MASK
            for task, index in TASK_INDEX.items():
                if not sample.target.masks[index]:
                    continue
                task_total[task] += 1
                if task == "dapai" or sample.target.labels[index] > 0:
                    task_positive[task] += 1
                else:
                    task_negative[task] += 1
    return {
        "rank": rank,
        "rounds": round_count,
        "records": record_count,
        "multi_label_records": multi_label_records,
        "task_total": dict(task_total),
        "task_positive": dict(task_positive),
        "task_negative": dict(task_negative),
        "event_counts": dict(event_counts),
        "id_xor_128": digest_xor,
        "id_sum_128": digest_sum,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _merge(parts: list[dict]) -> dict:
    result = {
        "rounds": 0,
        "records": 0,
        "multi_label_records": 0,
        "task_total": Counter(),
        "task_positive": Counter(),
        "task_negative": Counter(),
        "event_counts": Counter(),
        "id_xor_128": 0,
        "id_sum_128": 0,
    }
    for part in parts:
        for key in ("rounds", "records", "multi_label_records"):
            result[key] += part[key]
        for key in ("task_total", "task_positive", "task_negative", "event_counts"):
            result[key].update(part[key])
        result["id_xor_128"] ^= part["id_xor_128"]
        result["id_sum_128"] = (result["id_sum_128"] + part["id_sum_128"]) & MASK
    for key in ("task_total", "task_positive", "task_negative", "event_counts"):
        result[key] = {task: int(result[key].get(task, 0)) for task in sorted(result[key])}
    result["id_xor_128"] = f"{result['id_xor_128']:032x}"
    result["id_sum_128"] = f"{result['id_sum_128']:032x}"
    return result


def _validate_counts(
    result: dict, include_fulou_negatives: bool, training_sampling: bool
) -> dict:
    events = result["event_counts"]
    totals = result["task_total"]
    positives = result["task_positive"]
    checks = {
        "dapai_total_equals_events": totals.get("dapai", 0) == events.get("dapai", 0),
        "riichi_total_valid": (
            totals.get("riichi", 0) <= events.get("dapai", 0)
            if training_sampling
            else totals.get("riichi", 0) == events.get("dapai", 0)
        ),
        "riichi_positives_equal_flags": positives.get("riichi", 0)
        == events.get("riichi_positive", 0),
        "fulou_positives_equal_events": positives.get("fulou", 0) == events.get("fulou", 0),
        "gang_positives_equal_events": positives.get("gang", 0) == events.get("gang", 0),
        "hule_positives_equal_events": positives.get("hule", 0) == events.get("hule", 0),
    }
    if not include_fulou_negatives:
        checks["fulou_total_equals_events"] = totals.get("fulou", 0) == events.get("fulou", 0)
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"task count validation failed: {failed}")
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--include-fulou-negatives", action="store_true")
    parser.add_argument(
        "--training-sampling",
        action="store_true",
        help="Apply the default train_vit.py negative-sampling policies.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--years", nargs="+", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    manifest = args.manifest.resolve()
    started = time.perf_counter()
    jobs = [
        (
            str(manifest),
            args.split,
            rank,
            args.workers,
            args.include_fulou_negatives,
            args.training_sampling,
            args.seed,
            tuple(args.years) if args.years else None,
        )
        for rank in range(args.workers)
    ]
    if args.workers == 1:
        parts = [_count_partition(jobs[0])]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            parts = list(executor.map(_count_partition, jobs))
    merged = _merge(parts)
    checks = _validate_counts(
        merged, args.include_fulou_negatives, args.training_sampling
    )
    legacy_records = sum(merged["task_total"].get(task, 0) for task in TASK_NAMES)
    reduction = 1.0 - merged["records"] / legacy_records if legacy_records else 0.0
    report = {
        "target_schema_version": "unified-multitask-target-v1",
        "manifest": str(manifest),
        "split": args.split,
        "workers": args.workers,
        "include_fulou_negatives": args.include_fulou_negatives,
        "training_sampling": args.training_sampling,
        "seed": args.seed,
        "years": sorted(args.years) if args.years else None,
        "elapsed_seconds": time.perf_counter() - started,
        **merged,
        "legacy_task_records": legacy_records,
        "unified_backbone_records": merged["records"],
        "backbone_record_reduction": reduction,
        "checks": checks,
        "partitions": parts,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
