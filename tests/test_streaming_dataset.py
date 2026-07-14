import json
import tempfile
import unittest
import zipfile
from itertools import islice
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from mahjong_ai.data.multitask import StreamingMultiTaskDataset
from mahjong_ai.data.normalize import stable_split
from mahjong_ai.data.prepare import prepare_dataset
from mahjong_ai.data.schema import EVENT_TYPES, NO_MELD
from mahjong_ai.data.streaming_dataset import (
    StreamingRoundDataset,
    TensorRoundRecord,
    build_streaming_dataloader,
)


def _round(round_index):
    return [
        {"qipai": {
            "zhuangfeng": 0,
            "jushu": round_index,
            "changbang": 0,
            "lizhibang": 0,
            "defen": [25000] * 4,
            "baopai": "m1",
            "shoupai": [
                "m123p123s123z1234",
                "m456p456s456z1234",
                "m789p789s789z1234",
                "m123p456s789z1234",
            ],
        }},
        {"zimo": {"l": 0, "p": "m5"}},
        {"dapai": {"l": 0, "p": "m5"}},
        {"fulou": {"l": 1, "m": "m55-5"}},
        {"dapai": {"l": 1, "p": "p4"}},
        {"pingju": {"fenpei": [0] * 4}},
    ]


def _game():
    return {"log": [_round(0), _round(1)]}


def _names_for_split(split, count, seed=13):
    names = []
    candidate = 0
    while len(names) < count:
        name = f"record-{candidate}.json"
        if stable_split("data2023.zip", name, seed) == split:
            names.append(name)
        candidate += 1
    return names


class StreamingDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        archive_path = self.root / "data2023.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for name in _names_for_split("train", 12):
                archive.writestr(name, json.dumps(_game()))
            for name in _names_for_split("validation", 2):
                archive.writestr(name, json.dumps(_game()))
            for name in _names_for_split("test", 2):
                archive.writestr(name, json.dumps(_game()))
        output = self.root / "prepared"
        prepare_dataset(
            [archive_path],
            output,
            workers=1,
            rounds_per_shard=4,
            checkpoint_members=16,
            chunk_size=4,
            seed=13,
        )
        self.manifest = output / "manifest.json"

    def tearDown(self):
        self.tempdir.cleanup()

    def _dataset(self, **kwargs):
        defaults = dict(
            split="train",
            seed=99,
            shuffle=False,
            shuffle_buffer_rounds=0,
            tensorize=False,
        )
        defaults.update(kwargs)
        return StreamingRoundDataset(self.manifest, **defaults)

    def test_worker_and_distributed_partitions_have_no_duplicates_or_missing_rounds(self):
        dataset = self._dataset()
        expected = {item.sample_id for item in dataset.iter_for_worker(0, 1)}
        self.assertEqual(len(expected), len(dataset))

        worker_sets = [
            {item.sample_id for item in dataset.iter_for_worker(worker, 4)}
            for worker in range(4)
        ]
        self.assertEqual(set().union(*worker_sets), expected)
        self.assertEqual(sum(map(len, worker_sets)), len(expected))

        rank_sets = []
        for rank in range(2):
            ranked = self._dataset(rank=rank, world_size=2)
            rank_sets.append({item.sample_id for item in ranked.iter_for_worker(0, 1)})
        self.assertTrue(rank_sets[0].isdisjoint(rank_sets[1]))
        self.assertEqual(rank_sets[0] | rank_sets[1], expected)

    def test_shuffle_is_deterministic_per_epoch_and_resume_is_exact(self):
        dataset = self._dataset(shuffle=True, shuffle_buffer_rounds=5)
        epoch0 = [item.sample_id for item in dataset.iter_for_worker(0, 1, epoch=0)]
        repeated = [item.sample_id for item in dataset.iter_for_worker(0, 1, epoch=0)]
        epoch1 = [item.sample_id for item in dataset.iter_for_worker(0, 1, epoch=1)]
        self.assertEqual(epoch0, repeated)
        self.assertNotEqual(epoch0, epoch1)
        self.assertEqual(set(epoch0), set(epoch1))

        consumed = list(islice(dataset.iter_for_worker(0, 1, epoch=0), 7))
        offsets = {}
        for item in consumed:
            dataset.update_resume_offsets(offsets, item)
        resumed = self._dataset(
            shuffle=True,
            shuffle_buffer_rounds=5,
            resume_offsets=offsets,
        )
        remainder = [item.sample_id for item in resumed.iter_for_worker(0, 1, epoch=0)]
        self.assertEqual(remainder, epoch0[7:])

    def test_round_records_detach_mmaps_and_rebase_meld_offsets(self):
        dataset = self._dataset()
        item = next(iter(dataset))
        self.assertIs(type(item.events), np.ndarray)
        self.assertIsNone(item.events.base)
        self.assertIsNone(item.hands.base)
        used = item.events["meld_offset"] != NO_MELD
        self.assertTrue(np.any(used))
        self.assertTrue(np.all(item.events["meld_offset"][used] < len(item.meld_tiles)))
        self.assertIn(EVENT_TYPES["fulou"], item.events["type"])

        tensor_item = next(iter(self._dataset(tensorize=True)))
        self.assertIsInstance(tensor_item, TensorRoundRecord)
        self.assertEqual(tuple(tensor_item.hands.shape), (4, 37))
        self.assertEqual(tensor_item.events.shape[1], 8)

    def test_year_filter_and_all_split_are_deterministic(self):
        included = list(self._dataset(years=[2023]))
        excluded = list(self._dataset(years=[2022]))
        all_splits = list(self._dataset(split="all", years=[2023]))

        self.assertEqual(len(included), len(self._dataset()))
        self.assertEqual(excluded, [])
        self.assertGreater(len(all_splits), len(included))
        self.assertTrue(all(item.year == 2023 for item in all_splits))

    def test_persistent_workers_observe_epoch_changes(self):
        dataset = self._dataset(shuffle=True, shuffle_buffer_rounds=5, tensorize=True)
        try:
            loader = build_streaming_dataloader(
                dataset,
                num_workers=2,
                pin_memory=False,
                persistent_workers=True,
            )
            epoch0 = [item.sample_id for item in loader]
            dataset.set_epoch(1)
            epoch1 = [item.sample_id for item in loader]
        except PermissionError as exc:
            self.skipTest(f"multiprocessing unavailable: {exc}")
        self.assertEqual(set(epoch0), set(epoch1))
        self.assertNotEqual(epoch0, epoch1)
        self.assertEqual(len(epoch0), len(dataset))

    def test_unified_multitask_stream_has_unique_worker_partition(self):
        def make_dataset():
            return StreamingMultiTaskDataset(
                self.manifest,
                split="train",
                seed=99,
                shuffle=False,
                shuffle_buffer_rounds=0,
                include_fulou_negatives=False,
                encode_features=False,
            )

        expected = [sample.sample_id for sample in make_dataset()]
        try:
            loader = DataLoader(
                make_dataset(), batch_size=None, num_workers=2,
                persistent_workers=False,
            )
            parallel = [sample.sample_id for sample in loader]
        except PermissionError as exc:
            self.skipTest(f"multiprocessing unavailable: {exc}")
        self.assertEqual(set(parallel), set(expected))
        self.assertEqual(len(parallel), len(expected))
        self.assertEqual(len(parallel), len(set(parallel)))

    def test_unified_stream_resumes_inside_a_round_without_missing_samples(self):
        defaults = dict(
            split="train",
            seed=99,
            shuffle=False,
            shuffle_buffer_rounds=0,
            include_fulou_negatives=True,
            encode_features=False,
        )
        expected = list(StreamingMultiTaskDataset(self.manifest, **defaults))
        consumed = expected[:3]
        self.assertEqual(consumed[0].worker_sequence, consumed[-1].worker_sequence)
        offsets = {}
        for sample in consumed:
            StreamingMultiTaskDataset.update_resume_sample_offsets(offsets, sample)
        resumed = list(StreamingMultiTaskDataset(
            self.manifest, resume_sample_offsets=offsets, **defaults
        ))
        self.assertEqual(
            [sample.sample_id for sample in resumed],
            [sample.sample_id for sample in expected[3:]],
        )


if __name__ == "__main__":
    unittest.main()
