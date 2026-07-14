import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from advanced_training.train_large import _restore_resume_config
from mahjong_ai_features import StateEncoderV2
from utils import evaluate_multitask


def _qipai_log():
    return [{
        "qipai": {
            "zhuangfeng": 0,
            "jushu": 0,
            "changbang": 0,
            "lizhibang": 0,
            "defen": [10000, 20000, 30000, 40000],
            "baopai": "m1",
            "shoupai": [
                "m123p123s123z1234",
                "m111p222s333z1111",
                "m456p456s456z2345",
                "m789p789s789z4567",
            ],
        }
    }]


class StateEncoderPrivacyTests(unittest.TestCase):
    def test_target_player_is_first_and_other_hands_are_hidden(self):
        tensor = StateEncoderV2(_qipai_log(), player_id=1).encode(1)

        hand_block_sums = [
            float(tensor[offset:offset + 7].sum())
            for offset in range(0, 28, 7)
        ]
        self.assertGreater(hand_block_sums[0], 0.0)
        self.assertEqual(hand_block_sums[1:], [0.0, 0.0, 0.0])

        # Score channels start after A..G: 28+64+16+28+4+4+9 = 153.
        scores = [float(tensor[153 + i, 0, 0]) for i in range(4)]
        for actual, expected in zip(scores, [0.2, 0.3, 0.4, 0.1]):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_seat_wind_is_offset_from_dealer(self):
        log = _qipai_log()
        log[0]["qipai"]["jushu"] = 1
        tensor = StateEncoderV2(log, player_id=1).encode(1)

        # Seat-wind block starts after the four score channels at channel 157.
        own_wind = tensor[157:161, 0, 0].tolist()
        self.assertEqual(own_wind, [1.0, 0.0, 0.0, 0.0])


class ResumeConfigTests(unittest.TestCase):
    def test_saved_config_is_restored_except_explicit_and_invocation_values(self):
        args = SimpleNamespace(
            data=["current.zip"], resume="run", run_dir="runs",
            run_name=None, device="auto", model="coatnet_large",
            epochs=40, amp=False, split_by_game=False,
        )
        saved = {
            "data": ["old.zip"], "device": "cuda", "model": "vit_large",
            "epochs": 30, "amp": True, "split_by_game": True,
        }

        restored = _restore_resume_config(args, saved, argv=["--epochs", "40"])

        self.assertEqual(args.data, ["current.zip"])
        self.assertEqual(args.device, "auto")
        self.assertEqual(args.model, "vit_large")
        self.assertEqual(args.epochs, 40)
        self.assertTrue(args.amp)
        self.assertTrue(args.split_by_game)
        self.assertIn("model", restored)
        self.assertNotIn("epochs", restored)


class _FixedMultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleDict({
            "dapai": nn.Linear(1, 2),
            "hule": nn.Linear(1, 2),
        })

    def forward(self, x, head_names=None):
        batch = x.shape[0]
        outputs = {
            "dapai": torch.tensor([[4.0, 0.0]]).repeat(batch, 1),
            "hule": torch.tensor([[4.0, 0.0]]).repeat(batch, 1),
        }
        names = head_names or outputs.keys()
        return {name: outputs[name] for name in names}


class MultiTaskMetricTests(unittest.TestCase):
    def test_zero_weight_head_does_not_affect_overall_accuracy(self):
        model = _FixedMultiTaskModel()
        loader = [(
            torch.zeros(2, 1),
            torch.tensor([0, 1]),
            ["dapai", "hule"],
        )]
        losses = {
            "dapai": nn.CrossEntropyLoss(),
            "hule": nn.CrossEntropyLoss(),
            "_default": nn.CrossEntropyLoss(),
        }

        result = evaluate_multitask(
            model, loader, losses, "cpu",
            task_weights={"dapai": 1.0, "hule": 0.0},
        )

        self.assertEqual(result["top1_acc"], 1.0)
        self.assertEqual(result["hule_acc"], 0.0)


if __name__ == "__main__":
    unittest.main()
