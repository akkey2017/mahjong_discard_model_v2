import unittest

import torch

from mahjong_ai.evaluation import EvaluationAccumulator, TaskEvaluation


class EvaluationTests(unittest.TestCase):
    def test_task_metrics_include_calibration_confusion_and_errors(self):
        metrics = TaskEvaluation("test", 3, calibration_bins=5, max_errors=2)
        logits = torch.tensor([
            [6.0, 0.0, 0.0],
            [0.0, 6.0, 0.0],
            [0.0, 0.0, 6.0],
            [5.0, 0.0, 0.0],
        ])
        labels = torch.tensor([0, 1, 1, 2])
        metrics.update(logits, labels, ["a", "b", "c", "d"])
        result = metrics.compute()

        self.assertEqual(result["total"], 4)
        self.assertEqual(result["confusion_matrix"][0][0], 1)
        self.assertEqual(result["confusion_matrix"][1][2], 1)
        self.assertEqual(len(result["calibration"]), 5)
        self.assertEqual(len(result["high_confidence_errors"]), 2)
        self.assertGreater(result["ece"], 0.0)

    def test_accumulator_honors_task_masks(self):
        accumulator = EvaluationAccumulator(calibration_bins=3, max_errors=1)
        logits = {
            "dapai": torch.zeros(2, 34),
            "riichi": torch.tensor([[0.0, 2.0], [2.0, 0.0]]),
            "fulou": torch.zeros(2, 4),
            "gang": torch.zeros(2, 3),
            "hule": torch.zeros(2, 2),
        }
        labels = torch.zeros(2, 5, dtype=torch.long)
        masks = torch.zeros(2, 5, dtype=torch.bool)
        masks[:, 0] = True
        masks[0, 1] = True
        accumulator.update(logits, labels, masks, ["one", "two"])
        result = accumulator.compute()

        self.assertEqual(result["dapai"]["total"], 2)
        self.assertEqual(result["riichi"]["total"], 1)
        self.assertEqual(result["fulou"]["total"], 0)


if __name__ == "__main__":
    unittest.main()
