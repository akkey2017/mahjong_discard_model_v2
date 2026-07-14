"""Encode an incremental :class:`RoundState` as a 380-channel snapshot."""

from __future__ import annotations

import numpy as np
import torch

from mahjong_ai_features import (
    FEATURE_ID_TO_TILE,
    StateEncoderV2,
    _make_dora_list_from,
    _process_single_number,
)

from .round_state import RoundSnapshot, RoundState


class IncrementalStateEncoder(StateEncoderV2):
    """Render features without replaying the round from its beginning.

    The inherited tile/meld rendering helpers keep compatible feature blocks
    byte-identical to ``StateEncoderV2``.  Channels that were documented as
    approximations in the legacy encoder use the exact incremental counters.
    """

    def __init__(self, state: RoundState, player_id: int):
        self.state = state
        self.player_id = player_id
        self.num_channels = 380

    @staticmethod
    def _tile_counter(tiles: tuple[int, ...] | list[int]) -> list[int]:
        counter = [0] * 37
        for tile in tiles:
            if 0 <= tile < 37:
                counter[tile] += 1
        return counter

    def encode(self) -> torch.Tensor:
        snapshot = self.state.snapshot(self.player_id)
        tensor = np.zeros((self.num_channels, 4, 9), dtype=np.float32)
        offset = 0

        # A. Only the target hand is observable.
        for relative_player in range(4):
            hand = list(snapshot.own_hand) if relative_player == 0 else [0] * 37
            self._encode_tiles(
                tensor,
                offset,
                self._convert_to_34_dim(hand),
                [hand[0], hand[10], hand[20]],
                is_red_channel=True,
            )
            offset += 7

        # B. Public melds.
        for melds in snapshot.melds:
            self._encode_melds(tensor, offset, melds)
            offset += 16

        # C. Declared concealed kans.
        for ankans in snapshot.ankans:
            self._encode_melds(tensor, offset, ankans, is_ankan=True)
            offset += 4

        # D. Rivers.
        for river in snapshot.rivers:
            counter = self._tile_counter(river)
            self._encode_tiles(
                tensor,
                offset,
                self._convert_to_34_dim(counter),
                [counter[0], counter[10], counter[20]],
                is_red_channel=True,
            )
            offset += 7

        # E. Riichi state.
        for active in snapshot.riichi_status:
            if active:
                tensor[offset, :, :] = 1.0
            offset += 1

        # F. Dora.
        dora_tiles = _make_dora_list_from(
            [FEATURE_ID_TO_TILE[tile] for tile in snapshot.dora_indicators]
        )
        dora_34 = [0] * 34
        for tile in dora_tiles:
            dora_34[_process_single_number(tile)] += 1
        self._encode_tiles(tensor, offset, dora_34, [0, 0, 0], is_red_channel=False)
        offset += 4

        # G. Round metadata. The absolute dealer equals qipai.jushu in this
        # record format; recover it from the relative snapshot.
        if 0 <= snapshot.round_wind < 3:
            tensor[offset + snapshot.round_wind, :, :] = 1.0
        offset += 3
        absolute_dealer = (snapshot.player_id + snapshot.dealer) % 4
        tensor[offset + absolute_dealer, :, :] = 1.0
        offset += 4
        tensor[offset, :, :] = snapshot.honba / 5.0
        offset += 1
        tensor[offset, :, :] = snapshot.kyotaku / 4.0
        offset += 1

        # H. Target-relative scores.
        for score in snapshot.scores:
            tensor[offset, :, :] = score / 100000.0
            offset += 1

        # I. Seat winds, target first.
        for relative_player in range(4):
            seat_wind = (relative_player - snapshot.dealer) % 4
            tensor[offset + seat_wind, :, :] = 1.0
            offset += 4

        # J. Exact live-wall count.
        tensor[offset, :, :] = snapshot.remaining_tiles / 70.0
        offset += 1

        # K. Publicly visible tiles (rivers and declared melds/kans).
        visible = list(snapshot.visible_tiles)
        self._encode_tiles(
            tensor,
            offset,
            self._convert_to_34_dim(visible),
            [visible[0], visible[10], visible[20]],
            is_red_channel=True,
        )
        offset += 7

        # L. Hidden ura-dora placeholders.
        self._encode_tiles(tensor, offset, [0] * 34, [0, 0, 0], is_red_channel=False)
        offset += 4

        # M. Furiten state (reserved until the hand evaluator supplies it).
        for furiten in snapshot.furiten_status:
            tensor[offset, :, :] = float(furiten)
            offset += 1

        # N. Most recent discard.
        last = [0] * 37
        if snapshot.last_discard is not None:
            last[snapshot.last_discard[1]] = 1
        self._encode_tiles(
            tensor,
            offset,
            self._convert_to_34_dim(last),
            [last[0], last[10], last[20]],
            is_red_channel=True,
        )
        offset += 7

        # O/P. Exact declaration turn and current ippatsu window.
        for turn in snapshot.riichi_turn:
            tensor[offset, :, :] = (turn or 0) / 18.0
            offset += 1
        for active in snapshot.ippatsu_status:
            tensor[offset, :, :] = float(active)
            offset += 1

        # Q/R. Target's first-turn opportunity and first-turn state.
        target_first_turn = snapshot.player_turn_counts[0] == 0 and snapshot.no_calls_yet
        tensor[offset, :, :] = float(target_first_turn)
        offset += 1
        tensor[offset, :, :] = float(target_first_turn)
        offset += 1

        # S. Haitei/Houtei proximity.
        tensor[offset, :, :] = float(snapshot.remaining_tiles < 5)
        offset += 1

        # T. Dora in observable hands and public melds.
        for relative_player, melds in enumerate(snapshot.melds):
            hand = snapshot.own_hand if relative_player == 0 else (0,) * 37
            count = sum(hand[tile] for tile in dora_tiles)
            count += sum(tile in dora_tiles for meld in melds for tile in meld)
            tensor[offset, :, :] = count / 10.0
            offset += 1

        # U. Exact tiles unknown to the target: subtract their hand, public
        # tiles, and dora indicators from a red-five inventory.
        unseen = [4] * 37
        for red, normal_five in ((0, 5), (10, 15), (20, 25)):
            unseen[red] = 1
            unseen[normal_five] = 3
        for tile, count in enumerate(snapshot.own_hand):
            unseen[tile] = max(0, unseen[tile] - count)
        for tile, count in enumerate(snapshot.visible_tiles):
            unseen[tile] = max(0, unseen[tile] - count)
        for tile in snapshot.dora_indicators:
            unseen[tile] = max(0, unseen[tile] - 1)
        self._encode_tiles(
            tensor,
            offset,
            self._convert_to_34_dim(unseen),
            [unseen[0], unseen[10], unseen[20]],
            is_red_channel=True,
        )
        offset += 7

        # V. Genbutsu against each riichi player.
        for river, riichi in zip(snapshot.rivers, snapshot.riichi_status):
            safe = [0] * 37
            if riichi:
                for tile in river:
                    safe[tile] = 1
            self._encode_tiles(
                tensor,
                offset,
                self._convert_to_34_dim(safe),
                [safe[0], safe[10], safe[20]],
                is_red_channel=True,
            )
            offset += 7

        # W. Exact target turn count.
        tensor[offset, :, :] = snapshot.player_turn_counts[0] / 20.0
        offset += 1

        # X. Last unclaimed discard in each river.
        for river in snapshot.rivers:
            last_by_player = [0] * 37
            if river:
                last_by_player[river[-1]] = 1
            self._encode_tiles(
                tensor,
                offset,
                self._convert_to_34_dim(last_by_player),
                [last_by_player[0], last_by_player[10], last_by_player[20]],
                is_red_channel=True,
            )
            offset += 7

        # Y/Z. Honba and public meld counts.
        tensor[offset, :, :] = snapshot.honba / 5.0
        offset += 1
        for melds in snapshot.melds:
            tensor[offset, :, :] = len(melds) / 4.0
            offset += 1

        assert offset == 280
        return torch.from_numpy(tensor)
