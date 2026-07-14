import unittest

import torch

from mahjong_ai.state import FEATURE_SCHEMA_VERSION, IncrementalStateEncoder, RoundState
from mahjong_ai_features import FEATURE_TILE_MAP, StateEncoderV2


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
        {"pingju": {"fenpei": [1000, -1000, 0, 0]}},
    ]


class RoundStateTransitionTests(unittest.TestCase):
    def test_snapshot_is_relative_and_does_not_expose_opponent_hands(self):
        state = RoundState.from_round_log(_round_log())
        snapshot = state.snapshot(2)

        self.assertEqual(snapshot.player_id, 2)
        self.assertEqual(snapshot.scores, (24000, 25000, 25000, 26000))
        self.assertEqual(snapshot.dealer, 3)
        self.assertFalse(hasattr(snapshot, "hands"))
        self.assertEqual(len(snapshot.own_hand), 37)
        self.assertEqual(FEATURE_SCHEMA_VERSION, "snapshot-v2-incremental")

    def test_draw_discard_riichi_tsumogiri_call_and_exact_counters(self):
        log = _round_log()
        state = RoundState.from_round_log(log)
        state.apply_event(log[1])
        state.apply_event(log[2])

        self.assertEqual(state.draw_count, 1)
        self.assertEqual(state.remaining_tiles, 69)
        self.assertEqual(state.player_turn_counts, [1, 0, 0, 0])
        self.assertTrue(state.riichi_status[0])
        self.assertEqual(state.riichi_turn[0], 1)
        self.assertTrue(state.ippatsu_status[0])

        state.apply_event(log[3])
        state.apply_event(log[4])
        self.assertTrue(state.last_discard_tsumogiri_by_player[1])
        discarded = FEATURE_TILE_MAP["p5"]
        self.assertEqual(state.visible_tiles[discarded], 1)

        state.apply_event(log[5])
        self.assertEqual(state.rivers[1], [])
        self.assertEqual(len(state.melds[2][0]), 3)
        self.assertEqual(state.visible_tiles[discarded], 3)
        self.assertFalse(any(state.ippatsu_status))
        self.assertFalse(state.no_calls_yet)

    def test_kakan_ankan_dora_and_terminal_updates(self):
        log = _round_log()
        state = RoundState.from_round_log(log)
        for event in log[1:]:
            state.apply_event(event)

        p5 = FEATURE_TILE_MAP["p5"]
        z1 = FEATURE_TILE_MAP["z1"]
        self.assertEqual(state.melds[2][0], [p5, p5, p5, p5])
        self.assertEqual(state.ankans[3][0], [z1, z1, z1, z1])
        self.assertEqual(state.visible_tiles[p5], 4)
        self.assertEqual(state.visible_tiles[z1], 4)
        self.assertEqual(state.dora_indicators, [FEATURE_TILE_MAP["s3"], FEATURE_TILE_MAP["p2"]])
        self.assertEqual(state.draw_count, 6)
        self.assertEqual(state.remaining_tiles, 64)
        self.assertEqual(state.terminal, "pingju")
        self.assertEqual(state.scores, [25000, 26000, 24000, 25000])

    def test_chi_daiminkan_red_five_and_hule(self):
        qipai = {
            "zhuangfeng": 0,
            "jushu": 0,
            "changbang": 0,
            "lizhibang": 0,
            "baopai": "m1",
            "shoupai": ["m034p123s123z1234", "m12p123s123z12345", "z5", "z555"],
        }
        state = RoundState.from_qipai(qipai)

        state.apply_event({"dapai": {"l": 0, "p": "m0"}})
        self.assertEqual(state.visible_tiles[FEATURE_TILE_MAP["m0"]], 1)
        state.apply_event({"dapai": {"l": 0, "p": "m3"}})
        state.apply_event({"fulou": {"l": 1, "m": "m12-3"}})
        self.assertEqual(len(state.melds[1][0]), 3)
        self.assertNotIn(FEATURE_TILE_MAP["m3"], state.rivers[0])

        state.apply_event({"dapai": {"l": 2, "p": "z5"}})
        state.apply_event({"fulou": {"l": 3, "m": "z5555-"}})
        self.assertEqual(len(state.melds[3][0]), 4)
        self.assertEqual(state.visible_tiles[FEATURE_TILE_MAP["z5"]], 4)

        state.apply_event({"hule": {"l": 1, "baojia": None}})
        self.assertEqual(state.terminal, "hule")


class IncrementalEncoderCompatibilityTests(unittest.TestCase):
    # Blocks whose semantics intentionally changed are excluded: remaining
    # tiles (J), riichi turn/ippatsu/first-turn/end flags (O-S), unseen tiles
    # (U), and turn count (W).
    COMPATIBLE_SLICES = (
        slice(0, 173),
        slice(174, 196),
        slice(207, 211),
        slice(218, 246),
        slice(247, 280),
        slice(280, 380),
    )

    def test_legacy_compatible_feature_blocks_match_before_every_event(self):
        log = _round_log()
        state = RoundState.from_round_log(log)
        for index in range(1, len(log)):
            for player in range(4):
                legacy = StateEncoderV2(log, player).encode(index)
                incremental = IncrementalStateEncoder(state, player).encode()
                for block in self.COMPATIBLE_SLICES:
                    self.assertTrue(
                        torch.equal(legacy[block], incremental[block]),
                        f"feature mismatch at event={index}, player={player}, block={block}",
                    )
            state.apply_event(log[index])

    def test_exact_channels_use_incremental_counters(self):
        log = _round_log()
        state = RoundState.from_round_log(log)
        for event in log[1:11]:
            state.apply_event(event)
        tensor = IncrementalStateEncoder(state, 2).encode()

        self.assertAlmostEqual(float(tensor[173, 0, 0]), state.remaining_tiles / 70.0)
        # Player 0 is relative seat 2 from target player 2.
        self.assertAlmostEqual(float(tensor[198, 0, 0]), 1 / 18.0)
        self.assertEqual(float(tensor[200, 0, 0]), 0.0)
        self.assertAlmostEqual(
            float(tensor[246, 0, 0]),
            state.player_turn_counts[2] / 20.0,
        )


if __name__ == "__main__":
    unittest.main()
