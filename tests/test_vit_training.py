import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from advanced_training.large_models import create_large_multitask_vit
from mahjong_ai.models import ViTConfig, create_vit, vit_config
from mahjong_ai.training import (
    CheckpointCompatibilityError,
    StepTrainer,
    StepWarmupCosineScheduler,
    TrainingConfig,
)


class ViTModelTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(torch, "compile"), "torch.compile unavailable")
    def test_model_is_compile_compatible(self):
        config = ViTConfig(
            name="compile-test", in_channels=4, embed_dim=16, depth=1,
            heads=4, dropout=0.0, drop_path=0.0,
        )
        compiled = torch.compile(create_vit(config), backend="eager")
        outputs = compiled(torch.randn(2, 4, 4, 9))
        self.assertEqual(set(outputs), {"dapai", "riichi", "fulou", "gang", "hule"})

    def test_presets_forward_every_head(self):
        expected = {"dapai": 34, "riichi": 2, "fulou": 4, "gang": 3, "hule": 2}
        for name in ("vit_small", "vit_base", "vit_large"):
            config = vit_config(name)
            model = create_vit(config).eval()
            with torch.no_grad():
                outputs = model(torch.randn(1, 380, 4, 9))
            self.assertEqual({task: value.shape[1] for task, value in outputs.items()}, expected)

    def test_legacy_multitask_large_is_strict_and_numerically_compatible(self):
        torch.manual_seed(3)
        legacy = create_large_multitask_vit(dropout=0.1).eval()
        current = create_vit(vit_config("vit_large")).eval()
        current.load_state_dict(legacy.state_dict(), strict=True)
        inputs = torch.randn(1, 380, 4, 9)
        with torch.no_grad():
            old = legacy(inputs)
            new = current(inputs)
        for task in old:
            torch.testing.assert_close(new[task], old[task], rtol=0, atol=0)


class SchedulerTests(unittest.TestCase):
    def test_step_warmup_cosine_state_roundtrip(self):
        parameter = torch.nn.Parameter(torch.ones(()))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        scheduler = StepWarmupCosineScheduler(
            optimizer, max_steps=10, warmup_steps=2, min_lr_ratio=0.1
        )
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.5)
        scheduler.step(1)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1.0)
        scheduler.step(10)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1)

        optimizer2 = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=1.0)
        scheduler2 = StepWarmupCosineScheduler(
            optimizer2, max_steps=10, warmup_steps=2, min_lr_ratio=0.1
        )
        scheduler2.load_state_dict(scheduler.state_dict())
        self.assertEqual(scheduler2.last_step, 10)
        self.assertAlmostEqual(optimizer2.param_groups[0]["lr"], 0.1)


class _Dataset:
    def __init__(self):
        self.epoch = 0
        self.offsets = {}

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_resume_sample_offsets(self, offsets):
        self.offsets = dict(offsets)


class _Loader:
    def __init__(self):
        self.dataset = _Dataset()

    def __iter__(self):
        return iter(())


class CheckpointTests(unittest.TestCase):
    def _trainer(self, root, manifest_hash="a" * 64):
        model_config = ViTConfig(
            name="test", in_channels=4, embed_dim=16, depth=1, heads=4,
            dropout=0.0, drop_path=0.0,
        )
        model = create_vit(model_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return StepTrainer(
            model=model,
            model_config=model_config,
            optimizer=optimizer,
            train_loader=_Loader(),
            validation_loader=None,
            device=torch.device("cpu"),
            config=TrainingConfig(
                max_steps=10, warmup_steps=2, validate_every=0,
                checkpoint_every=0, log_every=0, amp_dtype="fp32",
                ema_decay=0.9, profile_steps=0,
            ),
            run_dir=root,
            feature_schema_version="features-test",
            target_schema_version="targets-test",
            dataset_manifest_sha256=manifest_hash,
            run_metadata={"torch_version": torch.__version__},
        )

    def test_checkpoint_restores_training_and_stream_state_and_rejects_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer = self._trainer(root)
            trainer.global_step = 4
            trainer.samples_seen = 128
            trainer.data_epoch = 2
            trainer.resume_sample_offsets = {0: (7, 3), 1: (8, 1)}
            trainer.scheduler.step(4)
            checkpoint = trainer.save_checkpoint(numbered=False)

            restored = self._trainer(root / "restored")
            restored.resume(checkpoint)
            self.assertEqual(restored.global_step, 4)
            self.assertEqual(restored.samples_seen, 128)
            self.assertEqual(restored.resume_sample_offsets, {0: (7, 3), 1: (8, 1)})
            self.assertEqual(restored.train_loader.dataset.epoch, 2)
            self.assertEqual(restored.train_loader.dataset.offsets, {0: (7, 3), 1: (8, 1)})
            for left, right in zip(
                trainer.model.parameters(), restored.model.parameters()
            ):
                torch.testing.assert_close(left, right)

            incompatible = self._trainer(root / "bad", manifest_hash="b" * 64)
            with self.assertRaises(CheckpointCompatibilityError):
                incompatible.resume(checkpoint)

    def test_cuda_rng_state_is_moved_to_cpu_before_restore(self):
        with tempfile.TemporaryDirectory() as temporary:
            trainer = self._trainer(Path(temporary))
            cuda_state = mock.Mock()
            cpu_state = object()
            cuda_state.cpu.return_value = cpu_state

            with (
                mock.patch("torch.cuda.is_available", return_value=True),
                mock.patch("torch.cuda.set_rng_state_all") as restore,
            ):
                trainer._restore_rng({"cuda": [cuda_state]})

        cuda_state.cpu.assert_called_once_with()
        restore.assert_called_once_with([cpu_state])

    def test_completed_resume_preserves_existing_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trainer = self._trainer(root)
            trainer.global_step = trainer.config.max_steps
            trainer.samples_seen = 320
            summary = {
                "global_step": trainer.global_step,
                "samples_seen": trainer.samples_seen,
                "profile": {"profiled_steps": 7},
            }
            (root / "summary.json").write_text(json.dumps(summary))

            self.assertEqual(trainer.train(), summary)
            self.assertEqual(json.loads((root / "summary.json").read_text()), summary)


if __name__ == "__main__":
    unittest.main()
