#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT/ma/bin/python"
MANIFEST="$ROOT/data/prepared/schema_v1_2014_2023/manifest.json"
RUN_ROOT="${PHASE8_RUN_ROOT:-/mnt/model/akkey/mahjong_discard_model_v2/phase8_snapshot_vit_2014_2022}"
RUN_DIR="$RUN_ROOT/vit_large_seed42"
MODE="fresh"
RESUME_PATH="$RUN_DIR/last.pt"
DRY_RUN=0

case "${1:-}" in
  "") ;;
  --dry-run) DRY_RUN=1 ;;
  --resume)
    MODE="resume"
    if [[ -n "${2:-}" ]]; then
      RESUME_PATH="$2"
    fi
    ;;
  *)
    echo "usage: $0 [--dry-run | --resume [CHECKPOINT]]" >&2
    exit 2
    ;;
esac

COMMAND=(
  "$PYTHON_BIN" "$ROOT/scripts/train_vit.py"
  --data-manifest "$MANIFEST"
  --model vit_large
  --max-steps 214654
  --warmup-steps 10000
  --validate-every 10000
  --validation-steps 100
  --checkpoint-every 5000
  --log-every 100
  --samples-per-virtual-epoch 879225950
  --batch-size 4096
  --workers 4
  --validation-workers 2
  --prefetch-factor 4
  --train-years 2014 2015 2016 2017 2018 2019 2020 2021 2022
  --validation-years 2014 2015 2016 2017 2018 2019 2020 2021 2022
  --amp-dtype bf16
  --tf32
  --compile
  --compile-mode default
  --device cuda
  --run-dir "$RUN_DIR"
)

if [[ "$MODE" == "resume" ]]; then
  COMMAND+=(--resume "$RESUME_PATH")
fi

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'run_dir=%s\ncommand=' "$RUN_DIR"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment is missing: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "Prepared manifest is missing: $MANIFEST" >&2
  exit 1
fi
if [[ "$MODE" == "fresh" && -e "$RUN_DIR/last.pt" ]]; then
  echo "Refusing to overwrite an existing run; use --resume: $RUN_DIR" >&2
  exit 1
fi
if [[ "$MODE" == "resume" && ! -f "$RESUME_PATH" ]]; then
  echo "Resume checkpoint is missing: $RESUME_PATH" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
printf 'Starting Phase 8 %s run in %s\n' "$MODE" "$RUN_DIR"
"${COMMAND[@]}" 2>&1 | tee -a "$RUN_DIR/console.log"
