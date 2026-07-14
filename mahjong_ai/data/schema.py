"""Stable NumPy schemas for compact round shards."""

from __future__ import annotations

import numpy as np


DATASET_SCHEMA_VERSION = "compact-round-shards-v1"
NO_PLAYER = np.uint8(255)
NO_TILE = np.uint8(255)
NO_MELD = np.uint32(0xFFFFFFFF)

EVENT_TYPES = {
    "zimo": 1,
    "dapai": 2,
    "fulou": 3,
    "gang": 4,
    "gangzimo": 5,
    "kaigang": 6,
    "lizhi": 7,
    "hule": 8,
    "pingju": 9,
}
EVENT_TYPE_NAMES = {value: key for key, value in EVENT_TYPES.items()}

# Exactly eight bytes per normalized event. For meld events, meld_offset points
# into shard-level melds.npy. Event-specific flags are documented below.
EVENT_DTYPE = np.dtype([
    ("type", "u1"),
    ("player", "u1"),
    ("tile", "u1"),
    ("flags", "u1"),
    ("meld_offset", "<u4"),
])

# dapai flags
DAPAI_RIICHI = 1 << 0
DAPAI_TSUMOGIRI = 1 << 1

# fulou flags: bits 0..1 subtype, bits 2..3 source player, bit 4 is four tiles.
FULOU_CHI = 1
FULOU_PON = 2
FULOU_DAIMINKAN = 3
FULOU_SOURCE_SHIFT = 2
FULOU_FOUR_TILES = 1 << 4

# gang flags
GANG_ANKAN = 1
GANG_KAKAN = 2

# hule flags: bit 0 is tsumo; bits 1..2 contain baojia for ron.
HULE_TSUMO = 1 << 0
HULE_BAOJIA_SHIFT = 1

ROUND_DTYPE = np.dtype([
    ("game_index", "<u4"),
    ("round_index", "<u2"),
    ("round_wind", "u1"),
    ("dealer", "u1"),
    ("honba", "<u2"),
    ("kyotaku", "<u2"),
    ("scores", "<i4", (4,)),
    ("hands", "u1", (4, 37)),
    ("initial_dora", "u1"),
])

METADATA_DTYPE = np.dtype([
    ("game_id", "S32"),
    ("archive_index", "<u2"),
    ("year", "<u2"),
    ("split", "u1"),
    ("round_count", "<u2"),
    ("source_crc32", "<u4"),
    ("source_size", "<u4"),
])

SPLIT_NAMES = ("train", "validation", "test")
SPLIT_IDS = {name: index for index, name in enumerate(SPLIT_NAMES)}
