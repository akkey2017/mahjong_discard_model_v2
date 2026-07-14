"""Incremental state transitions for one mahjong round.

The mutable :class:`RoundState` is worker-local preparation state.  Callers
must use :meth:`RoundState.snapshot` when producing model inputs: the immutable
snapshot deliberately contains only the target player's concealed hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from mahjong_ai_features import (
    DEFAULT_STARTING_SCORES,
    FEATURE_TILE_MAP,
    _fulou_to_pais,
    _make_pai_counter_list_from,
)
from mahjong_rules import normalize_red_five, which_player_discarded_from


TileCounts = tuple[int, ...]
Meld = tuple[int, ...]


def _tile_id(raw: object) -> int | None:
    if not isinstance(raw, str):
        return None
    return FEATURE_TILE_MAP.get(raw.replace("*", "").replace("_", ""))


def _same_tile(left: int, right: int) -> bool:
    return normalize_red_five(left) == normalize_red_five(right)


def _remove_one(hand: list[int], tile: int) -> bool:
    """Remove an exact tile, falling back to its red/non-red five variant."""

    if 0 <= tile < len(hand) and hand[tile] > 0:
        hand[tile] -= 1
        return True
    normalized = normalize_red_five(tile)
    alternatives = {
        5: (5, 0),
        15: (15, 10),
        25: (25, 20),
    }.get(normalized, (normalized,))
    for alternative in alternatives:
        if hand[alternative] > 0:
            hand[alternative] -= 1
            return True
    return False


@dataclass(frozen=True)
class RoundSnapshot:
    """Target-relative, inference-safe view of a :class:`RoundState`."""

    player_id: int
    own_hand: TileCounts
    rivers: tuple[TileCounts, ...]
    melds: tuple[tuple[Meld, ...], ...]
    ankans: tuple[tuple[Meld, ...], ...]
    scores: tuple[int, ...]
    riichi_status: tuple[bool, ...]
    riichi_turn: tuple[int | None, ...]
    ippatsu_status: tuple[bool, ...]
    furiten_status: tuple[bool, ...]
    dora_indicators: tuple[int, ...]
    visible_tiles: TileCounts
    last_discard: tuple[int, int] | None
    last_discard_by_player: tuple[int | None, ...]
    last_discard_tsumogiri_by_player: tuple[bool | None, ...]
    draw_count: int
    player_draw_counts: tuple[int, ...]
    turn_count: int
    player_turn_counts: tuple[int, ...]
    remaining_tiles: int
    honba: int
    kyotaku: int
    dealer: int
    round_wind: int
    no_calls_yet: bool
    terminal: str | None
    event_index: int


@dataclass
class RoundState:
    """Mutable state advanced once, in order, for every round event."""

    hands: list[list[int]]
    rivers: list[list[int]]
    melds: list[list[list[int]]]
    ankans: list[list[list[int]]]
    scores: list[int]
    riichi_status: list[bool]
    riichi_turn: list[int | None]
    ippatsu_status: list[bool]
    furiten_status: list[bool]
    dora_indicators: list[int]
    visible_tiles: list[int]
    last_discard: tuple[int, int] | None
    last_discard_by_player: list[int | None]
    last_discard_tsumogiri_by_player: list[bool | None]
    draw_count: int
    player_draw_counts: list[int]
    turn_count: int
    player_turn_counts: list[int]
    honba: int
    kyotaku: int
    dealer: int
    round_wind: int
    no_calls_yet: bool
    terminal: str | None
    event_index: int

    @classmethod
    def from_qipai(cls, qipai: dict[str, Any]) -> "RoundState":
        shoupai = qipai.get("shoupai")
        if not isinstance(shoupai, Sequence) or isinstance(shoupai, (str, bytes)) or len(shoupai) != 4:
            raise ValueError("qipai.shoupai must contain four concealed hands")
        hands = [_make_pai_counter_list_from(str(hand)) for hand in shoupai]
        baopai = _tile_id(qipai.get("baopai"))
        scores = list(qipai.get("defen", DEFAULT_STARTING_SCORES))
        if len(scores) != 4:
            scores = list(DEFAULT_STARTING_SCORES)
        return cls(
            hands=hands,
            rivers=[[] for _ in range(4)],
            melds=[[] for _ in range(4)],
            ankans=[[] for _ in range(4)],
            scores=[int(score) for score in scores],
            riichi_status=[False] * 4,
            riichi_turn=[None] * 4,
            ippatsu_status=[False] * 4,
            furiten_status=[False] * 4,
            dora_indicators=[] if baopai is None else [baopai],
            visible_tiles=[0] * 37,
            last_discard=None,
            last_discard_by_player=[None] * 4,
            last_discard_tsumogiri_by_player=[None] * 4,
            draw_count=0,
            player_draw_counts=[0] * 4,
            turn_count=0,
            player_turn_counts=[0] * 4,
            honba=int(qipai.get("changbang", 0)),
            kyotaku=int(qipai.get("lizhibang", 0)),
            dealer=int(qipai.get("jushu", 0)) % 4,
            round_wind=int(qipai.get("zhuangfeng", 0)),
            no_calls_yet=True,
            terminal=None,
            event_index=1,
        )

    @classmethod
    def from_round_log(cls, round_log: Sequence[dict[str, Any]]) -> "RoundState":
        if not round_log or "qipai" not in round_log[0]:
            raise ValueError("round log must start with qipai")
        return cls.from_qipai(round_log[0]["qipai"])

    @property
    def remaining_tiles(self) -> int:
        # A kan replacement draw also consumes one tile from the effective live
        # wall because the wall tail is moved into the dead wall.
        return max(0, 70 - self.draw_count)

    def _cancel_ippatsu(self) -> None:
        self.ippatsu_status = [False] * 4

    def _apply_draw(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        player = payload.get("l")
        tile = _tile_id(payload.get("p"))
        if not isinstance(player, int) or not 0 <= player < 4 or tile is None:
            return
        self.hands[player][tile] += 1
        self.draw_count += 1
        self.player_draw_counts[player] += 1

    def _apply_discard(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        player = payload.get("l")
        raw = payload.get("p")
        tile = _tile_id(raw)
        if not isinstance(player, int) or not 0 <= player < 4 or tile is None:
            return
        _remove_one(self.hands[player], tile)
        self.rivers[player].append(tile)
        self.visible_tiles[tile] += 1
        self.last_discard = (player, tile)
        self.last_discard_by_player[player] = tile
        self.last_discard_tsumogiri_by_player[player] = isinstance(raw, str) and "_" in raw
        self.turn_count += 1
        self.player_turn_counts[player] += 1

        declares_riichi = isinstance(raw, str) and "*" in raw
        if declares_riichi:
            self.riichi_status[player] = True
            self.riichi_turn[player] = self.player_turn_counts[player]
            self.ippatsu_status[player] = True
        elif self.riichi_status[player] and self.ippatsu_status[player]:
            # The declaring player's next discard closes the ippatsu window.
            self.ippatsu_status[player] = False

    def _pop_claimed_discard(self, source: int, meld_tiles: list[int]) -> int | None:
        if not self.rivers[source]:
            return None
        claimed = self.rivers[source].pop()
        self.visible_tiles[claimed] = max(0, self.visible_tiles[claimed] - 1)
        for index, tile in enumerate(meld_tiles):
            if tile == claimed:
                meld_tiles.pop(index)
                return claimed
        for index, tile in enumerate(meld_tiles):
            if _same_tile(tile, claimed):
                meld_tiles.pop(index)
                return claimed
        return claimed

    def _apply_fulou(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        player = payload.get("l")
        meld_string = payload.get("m")
        if not isinstance(player, int) or not 0 <= player < 4 or not isinstance(meld_string, str):
            return
        all_tiles = _fulou_to_pais(meld_string)
        if not all_tiles:
            return
        consumed = list(all_tiles)
        source = which_player_discarded_from(meld_string, player)
        self._pop_claimed_discard(source, consumed)
        for tile in consumed:
            _remove_one(self.hands[player], tile)
        self.melds[player].append(list(all_tiles))
        for tile in all_tiles:
            self.visible_tiles[tile] += 1
        self.no_calls_yet = False
        self._cancel_ippatsu()

    def _apply_gang(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        player = payload.get("l")
        meld_string = payload.get("m")
        if not isinstance(player, int) or not 0 <= player < 4 or not isinstance(meld_string, str):
            return
        tiles = _fulou_to_pais(meld_string)
        if not tiles:
            return
        is_kakan = any(marker in meld_string for marker in "+-=")
        if is_kakan:
            added = tiles[-1]
            _remove_one(self.hands[player], added)
            upgraded = False
            for meld in self.melds[player]:
                if len(meld) == 3 and all(_same_tile(tile, added) for tile in meld):
                    meld.append(added)
                    upgraded = True
                    break
            if not upgraded:
                # Preserve a structurally valid state for malformed/truncated
                # logs while avoiding duplicate visibility for a known pon.
                self.melds[player].append(list(tiles))
                for tile in tiles[:-1]:
                    self.visible_tiles[tile] += 1
            self.visible_tiles[added] += 1
        else:
            for tile in tiles:
                _remove_one(self.hands[player], tile)
                self.visible_tiles[tile] += 1
            self.ankans[player].append(list(tiles))
        self.no_calls_yet = False
        self._cancel_ippatsu()

    def apply_event(self, event: dict[str, Any]) -> None:
        """Apply one event. Unknown event keys are safely ignored."""

        if not isinstance(event, dict):
            raise TypeError("round event must be a dict")
        if "qipai" in event:
            raise ValueError("qipai initializes RoundState and cannot be reapplied")
        if "zimo" in event:
            self._apply_draw(event["zimo"])
        elif "gangzimo" in event:
            self._apply_draw(event["gangzimo"])
        elif "dapai" in event:
            self._apply_discard(event["dapai"])
        elif "fulou" in event:
            self._apply_fulou(event["fulou"])
        elif "gang" in event:
            self._apply_gang(event["gang"])
        elif "kaigang" in event:
            payload = event["kaigang"]
            if isinstance(payload, dict):
                indicator = _tile_id(payload.get("baopai"))
                if indicator is not None:
                    self.dora_indicators.append(indicator)
        elif "hule" in event:
            self.terminal = "hule"
        elif "pingju" in event:
            self.terminal = "pingju"
            # Score settlement belongs to the game-level transition.  Keeping
            # the qipai scores stable is also required for consecutive hule
            # records (double/triple ron); the next round supplies new scores.
        self.event_index += 1

    def snapshot(self, player_id: int) -> RoundSnapshot:
        """Return an immutable, target-relative snapshot without hidden hands."""

        if not 0 <= player_id < 4:
            raise ValueError("player_id must be in 0..3")
        order = [(player_id + offset) % 4 for offset in range(4)]

        def relative_player(absolute: int) -> int:
            return (absolute - player_id) % 4

        last_discard = None
        if self.last_discard is not None:
            absolute, tile = self.last_discard
            last_discard = (relative_player(absolute), tile)
        return RoundSnapshot(
            player_id=player_id,
            own_hand=tuple(self.hands[player_id]),
            rivers=tuple(tuple(self.rivers[p]) for p in order),
            melds=tuple(tuple(tuple(meld) for meld in self.melds[p]) for p in order),
            ankans=tuple(tuple(tuple(meld) for meld in self.ankans[p]) for p in order),
            scores=tuple(self.scores[p] for p in order),
            riichi_status=tuple(self.riichi_status[p] for p in order),
            riichi_turn=tuple(self.riichi_turn[p] for p in order),
            ippatsu_status=tuple(self.ippatsu_status[p] for p in order),
            furiten_status=tuple(self.furiten_status[p] for p in order),
            dora_indicators=tuple(self.dora_indicators),
            visible_tiles=tuple(self.visible_tiles),
            last_discard=last_discard,
            last_discard_by_player=tuple(self.last_discard_by_player[p] for p in order),
            last_discard_tsumogiri_by_player=tuple(
                self.last_discard_tsumogiri_by_player[p] for p in order
            ),
            draw_count=self.draw_count,
            player_draw_counts=tuple(self.player_draw_counts[p] for p in order),
            turn_count=self.turn_count,
            player_turn_counts=tuple(self.player_turn_counts[p] for p in order),
            remaining_tiles=self.remaining_tiles,
            honba=self.honba,
            kyotaku=self.kyotaku,
            dealer=relative_player(self.dealer),
            round_wind=self.round_wind,
            no_calls_yet=self.no_calls_yet,
            terminal=self.terminal,
            event_index=self.event_index,
        )
