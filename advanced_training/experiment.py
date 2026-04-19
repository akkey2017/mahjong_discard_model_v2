"""
Experiment directory and logging helpers for advanced_training.

An experiment produces a self-contained directory under ``--run-dir`` (default
``runs/``) named ``<model>_<timestamp>``. The directory contains:

- ``config.json``   : serialized CLI args
- ``metrics.csv``   : per-epoch train/val metrics
- ``training.log``  : plain-text log lines mirroring stdout
- ``best_model.pth``: best checkpoint (monitored metric)
- ``last_model.pth``: most recent checkpoint (for --resume)

Checkpoints are rich dicts (see :func:`utils.save_checkpoint`) containing the
model architecture name, so evaluation does not need to infer it from the
filename.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class ExperimentLogger:
    """Owns the per-run directory, config.json, metrics.csv and log file."""

    run_dir: Path
    model_type: str
    config: Dict[str, Any]
    metrics_path: Path
    log_path: Path
    best_checkpoint: Path
    last_checkpoint: Path
    _metrics_fieldnames: Optional[list] = None

    @classmethod
    def create(
        cls,
        base_dir: str | os.PathLike,
        model_type: str,
        config: Dict[str, Any],
        run_name: Optional[str] = None,
    ) -> "ExperimentLogger":
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)
        name = run_name or f"{model_type}_{_timestamp()}"
        run_dir = base / name
        run_dir.mkdir(parents=True, exist_ok=True)

        inst = cls(
            run_dir=run_dir,
            model_type=model_type,
            config=dict(config),
            metrics_path=run_dir / "metrics.csv",
            log_path=run_dir / "training.log",
            best_checkpoint=run_dir / "best_model.pth",
            last_checkpoint=run_dir / "last_model.pth",
        )
        inst._init_files()
        return inst

    @classmethod
    def from_existing(cls, run_dir: str | os.PathLike) -> "ExperimentLogger":
        """Attach to an existing run directory (for --resume)."""
        run_dir = Path(run_dir)
        config_path = run_dir / "config.json"
        config = {}
        model_type = "unknown"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            model_type = config.get("model", "unknown")
        inst = cls(
            run_dir=run_dir,
            model_type=model_type,
            config=config,
            metrics_path=run_dir / "metrics.csv",
            log_path=run_dir / "training.log",
            best_checkpoint=run_dir / "best_model.pth",
            last_checkpoint=run_dir / "last_model.pth",
        )
        # Re-attach file logger so resumed runs continue writing to training.log.
        logger = logging.getLogger(f"experiment.{run_dir.name}")
        logger.setLevel(logging.INFO)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == inst.log_path
                   for h in logger.handlers):
            fh = logging.FileHandler(inst.log_path, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            logger.addHandler(fh)
        inst._logger = logger
        # Restore the existing CSV header so resumed rows use the same column order.
        if inst.metrics_path.exists() and inst.metrics_path.stat().st_size > 0:
            with inst.metrics_path.open(newline="") as f:
                header = next(csv.reader(f), None)
                if header:
                    inst._metrics_fieldnames = header
        return inst

    def _init_files(self) -> None:
        (self.run_dir / "config.json").write_text(
            json.dumps(self.config, indent=2, ensure_ascii=False, default=str)
        )
        # File logger
        logger = logging.getLogger(f"experiment.{self.run_dir.name}")
        logger.setLevel(logging.INFO)
        # Avoid duplicate handlers if re-instantiated
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == self.log_path
                   for h in logger.handlers):
            fh = logging.FileHandler(self.log_path, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            logger.addHandler(fh)
        self._logger = logger

    def log(self, msg: str) -> None:
        self._logger.info(msg)
        print(msg, file=sys.stderr, flush=True)

    def log_metrics(self, row: Dict[str, Any]) -> None:
        """Append one epoch's metrics to metrics.csv."""
        if self._metrics_fieldnames is None:
            # Lock field order on first call
            self._metrics_fieldnames = list(row.keys())
            write_header = not self.metrics_path.exists() or self.metrics_path.stat().st_size == 0
            mode = "w" if write_header else "a"
            with self.metrics_path.open(mode, newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._metrics_fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        else:
            # Align with existing header; add missing columns as empty.
            aligned = {k: row.get(k, "") for k in self._metrics_fieldnames}
            with self.metrics_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._metrics_fieldnames)
                writer.writerow(aligned)

    def iter_metrics(self) -> Iterable[Dict[str, Any]]:
        if not self.metrics_path.exists():
            return []
        with self.metrics_path.open() as f:
            return list(csv.DictReader(f))
