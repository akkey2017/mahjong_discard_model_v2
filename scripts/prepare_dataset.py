#!/usr/bin/env python3
"""Prepare deterministic, resumable compact shards from raw ZIP archives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mahjong_ai.data.prepare import prepare_dataset  # noqa: E402
from mahjong_ai.data.shard_writer import verify_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/prepared/schema_v1"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--rounds-per-shard", type=int, default=4096)
    parser.add_argument("--checkpoint-members", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--stop-after-checkpoints",
        type=int,
        help="Stop cleanly after N committed checkpoints (for resume testing/operations)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify_only:
        result = verify_dataset(args.output / "manifest.json")
        print(json.dumps(result, indent=2))
        return 0
    if not args.inputs:
        raise SystemExit("at least one input archive is required")
    manifest = prepare_dataset(
        args.inputs,
        args.output,
        workers=args.workers,
        rounds_per_shard=args.rounds_per_shard,
        checkpoint_members=args.checkpoint_members,
        chunk_size=args.chunk_size,
        seed=args.seed,
        max_files=args.max_files,
        resume=args.resume,
        stop_after_checkpoints=args.stop_after_checkpoints,
    )
    print(json.dumps({
        "manifest": str((args.output / "manifest.json").resolve()),
        "complete": manifest["complete"],
        "totals": manifest["totals"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
