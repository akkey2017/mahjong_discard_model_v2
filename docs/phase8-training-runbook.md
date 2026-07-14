# Phase 8: 10-year Snapshot ViT runbook

This run uses the complete 2014–2023 prepared dataset while holding out all
of 2023 for temporal evaluation. Training and hash-validation use only
2014–2022. Do not initialize from the 2023 checkpoint because that would leak
the temporal holdout into the model.

## Fixed configuration

- train/validation years: 2014–2022
- temporal holdout: every 2023 split (`train`, `validation`, and `test`)
- fixed test: hash `test` split from 2014–2022
- train records after negative sampling: 879,225,950
- batch size: 4096
- max steps: 214,654 (3,166 records remain below one global batch)
- workers/prefetch: 4/4
- ViT Large, BF16, TF32, `torch.compile`
- validation: every 10,000 steps, 100 batches
- checkpoint: every 5,000 steps
- expected training time: approximately 10 hours including validation/checkpoints
- expected numbered-checkpoint storage: approximately 17 GiB

The run is stored in an `akkey`-specific hierarchy on the shared SSD. The SSD
root is not writable by regular users, so create the hierarchy once:

```bash
sudo install -d -o akkey -g akkey -m 0750 \
  /mnt/model/akkey/mahjong_discard_model_v2/phase8_snapshot_vit_2014_2022
```

Confirm the command without starting training:

```bash
cd /home/akkey/ma/mahjong_discard_model_v2
bash scripts/run_phase8_10y.sh --dry-run
```

## Start in tmux

The launcher uses `ma/bin/python` directly, so it has the same effect as
activating `ma/bin/activate` and is independent of the shell startup state.

```bash
tmux new -s mahjong-phase8
cd /home/akkey/ma/mahjong_discard_model_v2
bash scripts/run_phase8_10y.sh
```

Detach with `Ctrl-b d` and reattach with:

```bash
tmux attach -t mahjong-phase8
```

Training metrics are appended to `metrics.jsonl`; complete console output is
also appended to `console.log` in the run directory.

## Resume after interruption

Start or attach a tmux session, then run:

```bash
cd /home/akkey/ma/mahjong_discard_model_v2
bash scripts/run_phase8_10y.sh --resume
```

The launcher defaults to the atomic `last.pt`. A specific numbered checkpoint
can be supplied as the second argument to `--resume`.

## Evaluation after training

Do not run this until training has completed:

```bash
cd /home/akkey/ma/mahjong_discard_model_v2
ma/bin/python scripts/evaluate_vit.py \
  --checkpoint /mnt/model/akkey/mahjong_discard_model_v2/phase8_snapshot_vit_2014_2022/vit_large_seed42/last.pt \
  --data-manifest data/prepared/schema_v1_2014_2023/manifest.json \
  --output /mnt/model/akkey/mahjong_discard_model_v2/phase8_snapshot_vit_2014_2022/vit_large_seed42/evaluation.json \
  --suite \
  --batch-size 4096 \
  --workers 4 \
  --prefetch-factor 4 \
  --device cuda
```

The report contains fixed-test metrics, the full-year 2023 temporal holdout,
per-year metrics, per-task confusion matrices and class metrics, calibration
(ECE/Brier), and high-confidence error examples.
