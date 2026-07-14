"""Validate raw JSON games and normalize them into compact integer arrays."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from dataset import classify_fulou, classify_gang
from mahjong_ai.state import RoundState
from mahjong_ai_features import (
    DEFAULT_STARTING_SCORES,
    FEATURE_TILE_MAP,
    _fulou_to_pais,
    _make_pai_counter_list_from,
)
from mahjong_rules import which_player_discarded_from

from .schema import (
    DAPAI_RIICHI,
    DAPAI_TSUMOGIRI,
    EVENT_DTYPE,
    EVENT_TYPES,
    FULOU_CHI,
    FULOU_DAIMINKAN,
    FULOU_FOUR_TILES,
    FULOU_PON,
    FULOU_SOURCE_SHIFT,
    GANG_ANKAN,
    GANG_KAKAN,
    HULE_BAOJIA_SHIFT,
    HULE_TSUMO,
    NO_MELD,
    NO_PLAYER,
    NO_TILE,
    SPLIT_IDS,
)


class ValidationError(ValueError):
    """Raw game cannot be represented safely by the compact schema."""


@dataclass(frozen=True)
class NormalizedRound:
    round_index: int
    round_wind: int
    dealer: int
    honba: int
    kyotaku: int
    scores: tuple[int, int, int, int]
    hands: np.ndarray
    initial_dora: int
    events: np.ndarray
    meld_tiles: np.ndarray


@dataclass(frozen=True)
class NormalizedGame:
    game_id: str
    archive_index: int
    year: int
    split: str
    source_crc32: int
    source_size: int
    rounds: tuple[NormalizedRound, ...]


def stable_game_id(archive_name: str, member: str) -> str:
    return hashlib.sha256(f"{archive_name}:{member}".encode()).hexdigest()[:32]


def stable_split(archive_name: str, member: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{archive_name}:{member}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "little") % 10000
    if bucket < 9800:
        return "train"
    if bucket < 9900:
        return "validation"
    return "test"


def _tile(raw: object) -> int | None:
    if not isinstance(raw, str):
        return None
    return FEATURE_TILE_MAP.get(raw.replace("*", "").replace("_", ""))


def _player(payload: object, *, required: bool = True) -> int:
    if isinstance(payload, dict) and isinstance(payload.get("l"), int):
        player = payload["l"]
        if 0 <= player < 4:
            return player
    if required:
        raise ValidationError("event player must be in 0..3")
    return int(NO_PLAYER)


def _event_record(event: dict[str, Any], melds: list[int]) -> tuple:
    keys = [key for key in EVENT_TYPES if key in event]
    if len(keys) != 1:
        raise ValidationError(f"event must contain exactly one known type: {tuple(event)}")
    kind = keys[0]
    payload = event[kind]
    player = int(NO_PLAYER)
    tile = int(NO_TILE)
    flags = 0
    meld_offset = int(NO_MELD)

    if kind in ("zimo", "gangzimo", "dapai"):
        player = _player(payload)
        if not isinstance(payload, dict):
            raise ValidationError(f"{kind} payload must be a dict")
        parsed = _tile(payload.get("p"))
        if parsed is None:
            raise ValidationError(f"invalid {kind} tile: {payload.get('p')!r}")
        tile = parsed
        if kind == "dapai":
            raw = payload.get("p")
            flags |= DAPAI_RIICHI if "*" in raw else 0
            flags |= DAPAI_TSUMOGIRI if "_" in raw else 0
    elif kind == "fulou":
        player = _player(payload)
        if not isinstance(payload, dict) or not isinstance(payload.get("m"), str):
            raise ValidationError("fulou.m must be a string")
        meld_string = payload["m"]
        parsed_tiles = _fulou_to_pais(meld_string)
        if len(parsed_tiles) not in (3, 4):
            raise ValidationError(f"fulou must contain 3 or 4 tiles: {meld_string!r}")
        subtype = {
            "chi": FULOU_CHI,
            "pon": FULOU_PON,
            "daiminkan": FULOU_DAIMINKAN,
        }[classify_fulou(meld_string)]
        source = which_player_discarded_from(meld_string, player)
        flags = subtype | (source << FULOU_SOURCE_SHIFT)
        flags |= FULOU_FOUR_TILES if len(parsed_tiles) == 4 else 0
        meld_offset = len(melds)
        melds.extend(parsed_tiles)
    elif kind == "gang":
        player = _player(payload)
        if not isinstance(payload, dict) or not isinstance(payload.get("m"), str):
            raise ValidationError("gang.m must be a string")
        meld_string = payload["m"]
        parsed_tiles = _fulou_to_pais(meld_string)
        if len(parsed_tiles) != 4:
            raise ValidationError(f"gang must contain 4 tiles: {meld_string!r}")
        flags = GANG_KAKAN if classify_gang(meld_string) == "kakan" else GANG_ANKAN
        meld_offset = len(melds)
        melds.extend(parsed_tiles)
    elif kind == "kaigang":
        if not isinstance(payload, dict):
            raise ValidationError("kaigang payload must be a dict")
        parsed = _tile(payload.get("baopai"))
        if parsed is None:
            raise ValidationError("kaigang.baopai must be a tile")
        tile = parsed
    elif kind == "lizhi":
        player = _player(payload)
    elif kind == "hule":
        player = _player(payload)
        if not isinstance(payload, dict):
            raise ValidationError("hule payload must be a dict")
        baojia = payload.get("baojia")
        if baojia is None:
            flags = HULE_TSUMO
        elif isinstance(baojia, int) and 0 <= baojia < 4:
            flags = baojia << HULE_BAOJIA_SHIFT
        else:
            raise ValidationError("hule.baojia must be null or in 0..3")
    elif kind == "pingju":
        if not isinstance(payload, dict):
            raise ValidationError("pingju payload must be a dict")

    return (EVENT_TYPES[kind], player, tile, flags, meld_offset)


def _normalize_round(round_log: object, round_index: int) -> NormalizedRound:
    if not isinstance(round_log, list) or not round_log or not isinstance(round_log[0], dict):
        raise ValidationError("round must be a non-empty event list")
    qipai = round_log[0].get("qipai")
    if not isinstance(qipai, dict):
        raise ValidationError("round must start with qipai")
    shoupai = qipai.get("shoupai")
    if not isinstance(shoupai, Sequence) or isinstance(shoupai, (str, bytes)) or len(shoupai) != 4:
        raise ValidationError("qipai.shoupai must contain four hands")
    hands = np.asarray(
        [_make_pai_counter_list_from(str(hand)) for hand in shoupai], dtype=np.uint8
    )
    scores_value = qipai.get("defen", DEFAULT_STARTING_SCORES)
    if not isinstance(scores_value, Sequence) or len(scores_value) != 4:
        raise ValidationError("qipai.defen must contain four scores")
    scores = tuple(int(value) for value in scores_value)
    initial_dora = _tile(qipai.get("baopai"))
    if initial_dora is None:
        raise ValidationError("qipai.baopai must be a tile")

    state = RoundState.from_round_log(round_log)
    melds: list[int] = []
    records = []
    for event_index, event in enumerate(round_log[1:], 1):
        if not isinstance(event, dict):
            raise ValidationError(f"event {event_index} must be a dict")
        records.append(_event_record(event, melds))
        try:
            state.apply_event(event)
        except (TypeError, ValueError, IndexError) as exc:
            raise ValidationError(f"invalid transition at event {event_index}: {exc}") from exc
    events = np.asarray(records, dtype=EVENT_DTYPE)
    return NormalizedRound(
        round_index=round_index,
        round_wind=int(qipai.get("zhuangfeng", 0)),
        dealer=int(qipai.get("jushu", 0)) % 4,
        honba=int(qipai.get("changbang", 0)),
        kyotaku=int(qipai.get("lizhibang", 0)),
        scores=scores,
        hands=hands,
        initial_dora=initial_dora,
        events=events,
        meld_tiles=np.asarray(melds, dtype=np.uint8),
    )


def normalize_game(
    game: object,
    *,
    archive_name: str,
    archive_index: int,
    member: str,
    year: int,
    seed: int,
    source_crc32: int,
    source_size: int,
) -> NormalizedGame:
    if not isinstance(game, dict) or not isinstance(game.get("log"), list):
        raise ValidationError("game.log must be a list")
    rounds = tuple(
        _normalize_round(round_log, round_index)
        for round_index, round_log in enumerate(game["log"])
    )
    if not rounds:
        raise ValidationError("game contains no rounds")
    split = stable_split(archive_name, member, seed)
    if split not in SPLIT_IDS:
        raise AssertionError(split)
    return NormalizedGame(
        game_id=stable_game_id(archive_name, member),
        archive_index=archive_index,
        year=year,
        split=split,
        source_crc32=source_crc32,
        source_size=source_size,
        rounds=rounds,
    )
