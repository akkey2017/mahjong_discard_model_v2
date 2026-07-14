import unittest
from collections import Counter

import numpy as np
import torch
from torch import nn

from dataset import _extract_samples_from_kyoku
from mahjong_ai.data.multitask import (
    TASK_CLASS_COUNTS,
    TASK_INDEX,
    TASK_NAMES,
    MultiTaskMetrics,
    MultiTaskSampleBuilder,
    NegativeSamplingConfig,
    TaskSamplingPolicy,
    masked_multitask_loss,
    unified_multitask_collate,
)
from mahjong_ai.data.normalize import normalize_game
from mahjong_ai.data.streaming_dataset import CompactRoundRecord
from models import MultiTaskDiscardModel


def _round_log():
    return [
        {"qipai": {
            "zhuangfeng": 0,
            "jushu": 1,
            "changbang": 2,
            "lizhibang": 1,
            "defen": [25000, 26000, 24000, 25000],
            "baopai": "s3",
            "shoupai": [
                "m0555p123s123z123",
                "m123p555s123z1234",
                "m234p555s234z2345",
                "m345p345s345z11112",
            ],
        }},
        {"zimo": {"l": 0, "p": "m5"}},
        {"dapai": {"l": 0, "p": "m5*"}},
        {"zimo": {"l": 1, "p": "s9"}},
        {"dapai": {"l": 1, "p": "p5_"}},
        {"fulou": {"l": 2, "m": "p55-5"}},
        {"dapai": {"l": 2, "p": "m2"}},
        {"zimo": {"l": 2, "p": "p5"}},
        {"gang": {"l": 2, "m": "p555-5"}},
        {"gangzimo": {"l": 2, "p": "m9"}},
        {"dapai": {"l": 2, "p": "m9_"}},
        {"zimo": {"l": 3, "p": "z1"}},
        {"gang": {"l": 3, "m": "z1111"}},
        {"gangzimo": {"l": 3, "p": "s9"}},
        {"kaigang": {"baopai": "p2"}},
        {"hule": {"l": 3, "baojia": None}},
    ]


def _compact_round():
    normalized = normalize_game(
        {"log": [_round_log()]},
        archive_name="data2023.zip",
        archive_index=0,
        member="fixture.json",
        year=2023,
        seed=7,
        source_crc32=1,
        source_size=1,
    ).rounds[0]
    return CompactRoundRecord(
        sample_id="fixture:0",
        shard_id=0,
        local_round_index=0,
        game_id="fixture",
        year=2023,
        round_index=0,
        round_wind=normalized.round_wind,
        dealer=normalized.dealer,
        honba=normalized.honba,
        kyotaku=normalized.kyotaku,
        scores=np.asarray(normalized.scores, dtype=np.int32),
        hands=normalized.hands.copy(),
        initial_dora=normalized.initial_dora,
        events=normalized.events.copy(),
        meld_tiles=normalized.meld_tiles.copy(),
    )


def _mask_counts(samples):
    return Counter({
        task: sum(sample.target.has(task) for sample in samples)
        for task in TASK_NAMES
    })


class MultiTaskTargetBuilderTests(unittest.TestCase):
    def test_discard_classes_cannot_be_configured_as_negative_sampling(self):
        with self.assertRaises(ValueError):
            NegativeSamplingConfig(policies={"dapai": TaskSamplingPolicy(0.5)})

    def test_unified_mask_counts_match_legacy_tasks_without_duplicate_states(self):
        legacy = list(_extract_samples_from_kyoku(
            _round_log(), collect_all_actions=True, include_fulou_negatives=True
        ))
        expected = Counter(sample[3] for sample in legacy)
        builder = MultiTaskSampleBuilder(
            split="validation", include_fulou_negatives=True, encode_features=True
        )
        samples = builder.build_round(_compact_round())

        self.assertEqual(_mask_counts(samples), expected)
        self.assertEqual(len({sample.sample_id for sample in samples}), len(samples))
        self.assertLess(len(samples), len(legacy))
        first_discard = next(
            sample for sample in samples
            if sample.target.has("dapai") and sample.player_id == 0
        )
        self.assertTrue(first_discard.target.has("riichi"))
        self.assertTrue(first_discard.target.has("gang"))
        self.assertEqual(first_discard.target.label("riichi"), 1)
        self.assertEqual(first_discard.target.label("gang"), 0)
        self.assertEqual(tuple(first_discard.features.shape), (380, 4, 9))

    def test_negative_sampling_is_deterministic_and_train_only(self):
        config = NegativeSamplingConfig(
            seed=41,
            policies={
                "riichi": TaskSamplingPolicy(keep_probability=0.0),
                "fulou": TaskSamplingPolicy(keep_probability=0.5),
                "gang": TaskSamplingPolicy(keep_probability=0.0),
            },
        )
        kwargs = dict(
            negative_sampling=config,
            include_fulou_negatives=True,
            encode_features=False,
        )
        train_a = MultiTaskSampleBuilder(split="train", **kwargs).build_round(_compact_round())
        train_b = MultiTaskSampleBuilder(split="train", **kwargs).build_round(_compact_round())
        validation = MultiTaskSampleBuilder(
            split="validation", **kwargs
        ).build_round(_compact_round())

        signature = lambda samples: [
            (sample.sample_id, sample.target.labels.tolist(), sample.target.masks.tolist())
            for sample in samples
        ]
        self.assertEqual(signature(train_a), signature(train_b))
        self.assertEqual(_mask_counts(validation)["riichi"], _mask_counts(validation)["dapai"])
        self.assertEqual(_mask_counts(train_a)["riichi"], 1)  # positive only
        self.assertFalse(any(
            sample.target.has("gang") and sample.target.label("gang") == 0
            for sample in train_a
        ))

    def test_ratio_cap_is_deterministic_within_round(self):
        config = NegativeSamplingConfig(
            seed=9,
            policies={"riichi": TaskSamplingPolicy(max_negative_per_positive=1)},
        )
        samples = MultiTaskSampleBuilder(
            split="train",
            negative_sampling=config,
            include_fulou_negatives=False,
            encode_features=False,
        ).build_round(_compact_round())
        riichi = [
            sample.target.label("riichi") for sample in samples
            if sample.target.has("riichi")
        ]
        self.assertLessEqual(riichi.count(0), riichi.count(1))


class _CountingBackbone(nn.Module):
    def __init__(self, in_features=16, out_features=12):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        return torch.tanh(self.linear(inputs))


class UnifiedLossMetricTests(unittest.TestCase):
    def test_all_heads_share_one_backbone_call_and_each_loss_decreases(self):
        torch.manual_seed(5)
        backbone = _CountingBackbone()
        model = MultiTaskDiscardModel(backbone, final_channels=12)
        features = torch.randn(96, 16)
        teacher = {
            task: torch.randn(16, classes)
            for task, classes in TASK_CLASS_COUNTS.items()
        }
        labels = torch.stack(
            [(features @ teacher[task]).argmax(dim=1) for task in TASK_NAMES], dim=1
        )
        masks = torch.ones_like(labels, dtype=torch.bool)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.04)

        with torch.no_grad():
            _, initial = masked_multitask_loss(model(features), labels, masks)
            initial = {task: float(loss) for task, loss in initial.items()}
        calls_before_training = backbone.calls
        for _ in range(45):
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss, _ = masked_multitask_loss(logits, labels, masks)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits = model(features)
            _, final = masked_multitask_loss(logits, labels, masks)
        for task in TASK_NAMES:
            self.assertLess(float(final[task]), initial[task], task)
        self.assertEqual(backbone.calls - calls_before_training, 46)

        metrics = MultiTaskMetrics(task_weights={"hule": 0.0})
        metrics.update(logits, labels, masks, final)
        result = metrics.compute()
        self.assertEqual(result["dapai_total"], len(features))
        self.assertGreaterEqual(result["top1_acc"], 0.0)

    def test_collate_produces_fixed_label_and_mask_tensors(self):
        samples = MultiTaskSampleBuilder(
            split="validation", encode_features=True
        ).build_round(_compact_round())[:4]
        batch = unified_multitask_collate(samples)
        self.assertEqual(tuple(batch["labels"].shape), (4, 5))
        self.assertEqual(tuple(batch["masks"].shape), (4, 5))
        self.assertEqual(batch["labels"].dtype, torch.int64)
        self.assertEqual(batch["masks"].dtype, torch.bool)
        self.assertEqual(batch["years"].tolist(), [2023] * 4)


if __name__ == "__main__":
    unittest.main()
