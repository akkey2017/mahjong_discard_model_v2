import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from mahjong_ai.data.normalize import normalize_game, stable_split
from mahjong_ai.data.prepare import prepare_dataset
from mahjong_ai.data.schema import (
    DAPAI_RIICHI,
    DAPAI_TSUMOGIRI,
    EVENT_TYPES,
    FULOU_PON,
    FULOU_SOURCE_SHIFT,
    NO_MELD,
)
from mahjong_ai.data.shard_writer import verify_dataset


def _round(round_index=0):
    return [
        {"qipai": {
            "zhuangfeng": 0,
            "jushu": round_index,
            "changbang": 0,
            "lizhibang": 0,
            "defen": [25000, 25000, 25000, 25000],
            "baopai": "m1",
            "shoupai": [
                "m123p123s123z1234",
                "m456p456s456z1234",
                "m789p789s789z1234",
                "m123p456s789z1234",
            ],
        }},
        {"zimo": {"l": 0, "p": "m5"}},
        {"dapai": {"l": 0, "p": "m5*_"}},
        {"fulou": {"l": 1, "m": "m55-5"}},
        {"dapai": {"l": 1, "p": "p4"}},
        {"pingju": {"fenpei": [0, 0, 0, 0]}},
    ]


def _game():
    return {"log": [_round(0), _round(1)]}


def _name_for_split(split, used):
    index = 0
    while True:
        name = f"game-{index}.json"
        if name not in used and stable_split("data2023.zip", name, 7) == split:
            return name
        index += 1


class NormalizerTests(unittest.TestCase):
    def test_compact_events_preserve_flags_and_meld_reference(self):
        normalized = normalize_game(
            _game(),
            archive_name="data2023.zip",
            archive_index=0,
            member="game.json",
            year=2023,
            seed=7,
            source_crc32=123,
            source_size=456,
        )
        first = normalized.rounds[0]
        self.assertEqual(first.events.dtype.itemsize, 8)
        self.assertEqual(first.events[0]["type"], EVENT_TYPES["zimo"])
        discard = first.events[1]
        self.assertEqual(discard["flags"], DAPAI_RIICHI | DAPAI_TSUMOGIRI)
        call = first.events[2]
        self.assertEqual(call["type"], EVENT_TYPES["fulou"])
        self.assertEqual(call["flags"] & 0b11, FULOU_PON)
        self.assertEqual(call["flags"] >> FULOU_SOURCE_SHIFT & 0b11, 0)
        self.assertNotEqual(call["meld_offset"], NO_MELD)
        self.assertEqual(first.meld_tiles.tolist(), [5, 5, 5])


class PreparedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.archive = self.root / "data2023.zip"
        used = set()
        self.members = []
        for split in ("train", "validation", "test"):
            name = _name_for_split(split, used)
            used.add(name)
            self.members.append((split, name))
        with zipfile.ZipFile(self.archive, "w") as archive:
            for _, name in self.members:
                archive.writestr(name, json.dumps(_game()))
            archive.writestr("broken.json", "{not json")

    def tearDown(self):
        self.tempdir.cleanup()

    def _prepare(self, output, **kwargs):
        workers = kwargs.pop("workers", 1)
        return prepare_dataset(
            [self.archive],
            output,
            workers=workers,
            rounds_per_shard=2,
            checkpoint_members=2,
            chunk_size=1,
            seed=7,
            **kwargs,
        )

    def test_clean_stop_resume_checksum_split_and_deterministic_manifest(self):
        resumed = self.root / "resumed"
        manifest = self._prepare(resumed, stop_after_checkpoints=1)
        self.assertFalse(manifest["complete"])
        self.assertEqual(manifest["totals"]["source_files_processed"], 2)

        manifest = self._prepare(resumed, resume=True)
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["totals"]["games"], 3)
        self.assertEqual(manifest["totals"]["rounds"], 6)
        self.assertEqual(manifest["totals"]["corrupted_files"], 1)
        self.assertEqual({item["split"] for item in manifest["shards"]}, {"train", "validation", "test"})
        verified = verify_dataset(resumed / "manifest.json")
        self.assertEqual(verified["games"], 3)
        self.assertFalse((resumed / ".pending_commit.json").exists())
        self.assertFalse((resumed / ".staging_batch").exists())

        fresh = self.root / "fresh"
        self._prepare(fresh)
        self.assertEqual(
            (resumed / "manifest.json").read_bytes(),
            (fresh / "manifest.json").read_bytes(),
        )
        self.assertEqual(
            (resumed / "progress.json").read_bytes(),
            (fresh / "progress.json").read_bytes(),
        )

        parallel = self.root / "parallel"
        self._prepare(parallel, workers=2)
        self.assertEqual(
            (resumed / "manifest.json").read_bytes(),
            (parallel / "manifest.json").read_bytes(),
        )

        for descriptor in manifest["shards"]:
            metadata = np.load(resumed / descriptor["path"] / "metadata.npy", allow_pickle=False)
            self.assertTrue(np.all(metadata["round_count"] == 2))

    def test_checksum_detects_modified_array(self):
        output = self.root / "corrupt-check"
        manifest = self._prepare(output)
        shard = output / manifest["shards"][0]["path"]
        events = shard / "events.npy"
        with events.open("r+b") as stream:
            stream.seek(-1, 2)
            byte = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([byte[0] ^ 0xFF]))
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verify_dataset(output / "manifest.json")


if __name__ == "__main__":
    unittest.main()
