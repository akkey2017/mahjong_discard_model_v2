"""
Lightweight mahjong rules helpers used for:

1. Generating negative samples for call decisions (chi/pon/kan) during training.
2. Online inference (MahjongAgent) to mask impossible actions.

This module deliberately keeps to *local* checks that can be evaluated in
O(1)/O(hand_size) without implementing a full yaku/tenpai engine.

Tile representation follows ``mahjong_ai_features.FEATURE_TILE_MAP`` (37-dim
with red fives m0/p0/s0). Hands are passed as a length-37 count list.
"""

from __future__ import annotations

from typing import List, Tuple

from mahjong_ai_features import FEATURE_TILE_MAP, _make_pai_counter_list_from


# ---- tile-id helpers -----------------------------------------------------


def is_suited(tile_37: int) -> bool:
    """True for m/p/s, False for honors (z)."""
    return tile_37 < 30


def suit_and_number(tile_37: int) -> Tuple[str, int]:
    """Return (suit_letter, 1..9). Red fives (0) normalize to 5."""
    if tile_37 < 10:
        suit = "m"
        n = tile_37 if tile_37 != 0 else 5
    elif tile_37 < 20:
        suit = "p"
        n = (tile_37 - 10) if tile_37 != 10 else 5
    elif tile_37 < 30:
        suit = "s"
        n = (tile_37 - 20) if tile_37 != 20 else 5
    else:
        suit = "z"
        n = tile_37 - 30 + 1
    return suit, n


def normalize_red_five(tile_37: int) -> int:
    """Map m0/p0/s0 -> m5/p5/s5 (37-dim). Other tiles unchanged."""
    if tile_37 == 0:
        return 5
    if tile_37 == 10:
        return 15
    if tile_37 == 20:
        return 25
    return tile_37


def tile_count(hand37: List[int], tile_37: int) -> int:
    """Effective count of a tile in hand, treating red-5 as 5 for matching."""
    norm = normalize_red_five(tile_37)
    if norm == tile_37:
        # Also include red-5 variant when normal 5 is requested
        if norm == 5:
            return hand37[5] + hand37[0]
        if norm == 15:
            return hand37[15] + hand37[10]
        if norm == 25:
            return hand37[25] + hand37[20]
        return hand37[norm]
    return hand37[norm] + hand37[tile_37]


# ---- call feasibility ----------------------------------------------------


def can_pon(hand37: List[int], discarded_tile_37: int) -> bool:
    """Two matching tiles in hand -> pon possible (red-5 counts as 5)."""
    return tile_count(hand37, discarded_tile_37) >= 2


def can_daiminkan(hand37: List[int], discarded_tile_37: int) -> bool:
    """Three matching tiles in hand -> daiminkan possible."""
    return tile_count(hand37, discarded_tile_37) >= 3


def can_chi(hand37: List[int], discarded_tile_37: int, from_shimocha: bool = False) -> bool:
    """Chi is only legal from the upper-seat (kamicha) discard.

    Args:
        hand37: current hand.
        discarded_tile_37: the discarded tile id.
        from_shimocha: if True, this is not from kamicha -> chi impossible.

    Returns True if any of the three chi configurations is possible.
    """
    if from_shimocha:
        return False
    if not is_suited(discarded_tile_37):
        return False

    suit, n = suit_and_number(discarded_tile_37)
    # Candidate pairs of adjacent numbers in the same suit that complete a run
    need_pairs = [
        (n - 2, n - 1),
        (n - 1, n + 1),
        (n + 1, n + 2),
    ]
    for a, b in need_pairs:
        if 1 <= a <= 9 and 1 <= b <= 9:
            tile_a = FEATURE_TILE_MAP.get(f"{suit}{a}")
            tile_b = FEATURE_TILE_MAP.get(f"{suit}{b}")
            if tile_a is None or tile_b is None:
                continue
            if tile_count(hand37, tile_a) >= 1 and tile_count(hand37, tile_b) >= 1:
                return True
    return False


def can_ankan(hand37: List[int]) -> bool:
    """Any quadruplet held in hand?"""
    for i, c in enumerate(hand37):
        if c >= 4:
            return True
        # Account for red-5 variants: {m5, m0} combined count
        if i == 5 and hand37[5] + hand37[0] >= 4:
            return True
        if i == 15 and hand37[15] + hand37[10] >= 4:
            return True
        if i == 25 and hand37[25] + hand37[20] >= 4:
            return True
    return False


# ---- from-string convenience --------------------------------------------


def hand_counter_from_str(shoupai_str: str) -> List[int]:
    """Convert a shoupai string (e.g. 'm178p36s15578z356') to a 37-dim count list."""
    return _make_pai_counter_list_from(shoupai_str)


def which_player_discarded_from(meld_str: str, caller: int) -> int:
    """Given a fulou meld string and the caller's seat, return the discarder's seat.

    The direction markers are '+': shimocha, '-': kamicha, '=': toimen, which
    are the same semantics as :func:`mahjong_ai_features._fulou_to_pais`.
    """
    if "+" in meld_str:
        return (caller + 1) % 4
    if "=" in meld_str:
        return (caller + 2) % 4
    if "-" in meld_str:
        return (caller + 3) % 4
    return caller  # shouldn't happen for fulou, but fall back to self
