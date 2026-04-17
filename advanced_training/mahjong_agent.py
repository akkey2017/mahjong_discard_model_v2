"""
Inference wrapper that turns a trained checkpoint into a playable agent.

The agent accepts a running kyoku log (same shape used at training time) and
returns a decision dict in the *same* moupa record format (``{'dapai': ...}``,
``{'fulou': ...}`` etc.), so it can be plugged into existing game-replay or
online-play infrastructure.

Usage::

    from advanced_training.mahjong_agent import MahjongAgent

    agent = MahjongAgent("runs/vit_large_.../best_model.pth", device="cuda")

    # After my zimo, decide next action:
    act = agent.on_zimo(kyoku_log, log_index=len(kyoku_log) - 1, player_id=0)

    # After an opponent's dapai, decide whether to call:
    act = agent.on_opponent_dapai(kyoku_log, log_index=..., my_player_id=0)
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mahjong_ai_features import StateEncoderV2, FEATURE_TILE_MAP  # noqa: E402
from mahjong_rules import (  # noqa: E402
    can_ankan,
    can_chi,
    can_daiminkan,
    can_pon,
    hand_counter_from_str,
)
from utils import load_checkpoint  # noqa: E402
from advanced_training.large_models import MODEL_FACTORIES, MULTITASK_MODELS  # noqa: E402


_ID_TO_TILE_34 = {
    **{i - 1: f"m{i}" for i in range(1, 10)},
    **{i - 1 + 9: f"p{i}" for i in range(1, 10)},
    **{i - 1 + 18: f"s{i}" for i in range(1, 10)},
    **{i - 1 + 27: f"z{i}" for i in range(1, 8)},
}


class MahjongAgent:
    """Decision policy driven by a trained (multi-task) model."""

    def __init__(self, checkpoint_path: str, device: Optional[str] = None,
                 model_type_override: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        payload = load_checkpoint(checkpoint_path, map_location=self.device)
        self.model_type = model_type_override or payload.get("model_type")
        if self.model_type not in MODEL_FACTORIES:
            raise ValueError(
                f"Unknown or missing model_type in checkpoint: {self.model_type}. "
                f"Pass model_type_override explicitly."
            )
        self.is_multitask = self.model_type in MULTITASK_MODELS
        self.model = MODEL_FACTORIES[self.model_type](dropout=0.0)
        self.model.load_state_dict(payload["model_state"])
        self.model.to(self.device).eval()

    # ---- encoding ------------------------------------------------------

    def _encode(self, kyoku_log, log_index, player_id) -> torch.Tensor:
        encoder = StateEncoderV2(kyoku_log, player_id)
        state = encoder.encode(log_index)
        if not torch.is_tensor(state):
            state = torch.as_tensor(state)
        return state.unsqueeze(0).to(self.device)

    def _forward_heads(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return {head_name: logits} even for single-head checkpoints."""
        with torch.no_grad():
            out = self.model(x)
        if isinstance(out, dict):
            return out
        # Single-head DiscardModel: expose the one head
        return {"discard": out}

    # ---- decision endpoints -------------------------------------------

    def decide_discard(self, kyoku_log, log_index, player_id,
                       forbid_tiles: Optional[List[int]] = None) -> Dict:
        """Pick the best discard tile (34-dim id -> string).

        Args:
            forbid_tiles: optional list of 34-dim ids to mask out (e.g. tiles
                not in hand). Logits at those positions are set to -inf.
        """
        x = self._encode(kyoku_log, log_index, player_id)
        heads = self._forward_heads(x)
        logits = heads["discard"][0]
        if forbid_tiles:
            logits = logits.clone()
            for t in forbid_tiles:
                logits[t] = float("-inf")
        tile_id = int(logits.argmax().item())
        return {
            "action": "discard",
            "tile_id": tile_id,
            "tile": _ID_TO_TILE_34[tile_id],
            "confidence": float(F.softmax(logits, dim=-1)[tile_id]),
        }

    def decide_riichi(self, kyoku_log, log_index, player_id) -> bool:
        """Return True iff the riichi head prefers declaration (multi-task only)."""
        if not self.is_multitask:
            return False
        x = self._encode(kyoku_log, log_index, player_id)
        heads = self._forward_heads(x)
        if "riichi" not in heads:
            return False
        probs = F.softmax(heads["riichi"][0], dim=-1)
        return bool(probs.argmax().item() == 1)

    def decide_call(self, kyoku_log, discard_index, my_player_id,
                    my_hand_str: str) -> Dict:
        """Decide chi/pon/kan/pass after an opponent's dapai.

        Masks impossible calls using :mod:`mahjong_rules`.
        """
        move = kyoku_log[discard_index]
        if "dapai" not in move:
            return {"action": "pass"}
        discarder = move["dapai"]["l"]
        if discarder == my_player_id:
            return {"action": "pass"}

        tile_str = move["dapai"]["p"].replace("*", "").replace("_", "")
        tile_37 = FEATURE_TILE_MAP.get(tile_str)
        if tile_37 is None:
            return {"action": "pass"}

        hand37 = hand_counter_from_str(my_hand_str)
        # Chi is only legal from kamicha
        from_kamicha = ((discarder + 1) % 4) == my_player_id
        chi_ok = can_chi(hand37, tile_37, from_shimocha=not from_kamicha)
        pon_ok = can_pon(hand37, tile_37)
        kan_ok = can_daiminkan(hand37, tile_37)

        if not (chi_ok or pon_ok or kan_ok):
            return {"action": "pass"}
        if not self.is_multitask:
            # Single-head checkpoint can't decide calls; default to pass.
            return {"action": "pass"}

        x = self._encode(kyoku_log, discard_index, my_player_id)
        heads = self._forward_heads(x)

        def _yes(head):
            if head not in heads:
                return 0.0
            return float(F.softmax(heads[head][0], dim=-1)[1])

        options = []
        if kan_ok:
            options.append(("daiminkan", _yes("kan")))
        if pon_ok:
            options.append(("pon", _yes("pon")))
        if chi_ok:
            options.append(("chi", _yes("chi")))
        # Pick highest-confidence call above 0.5; otherwise pass.
        options.sort(key=lambda x: x[1], reverse=True)
        best_action, best_prob = options[0]
        if best_prob >= 0.5:
            return {"action": best_action, "confidence": best_prob}
        return {"action": "pass"}

    def decide_ankan(self, kyoku_log, log_index, player_id,
                     my_hand_str: str) -> bool:
        """Return True iff ankan is possible and the kan head prefers it."""
        if not self.is_multitask:
            return False
        hand37 = hand_counter_from_str(my_hand_str)
        if not can_ankan(hand37):
            return False
        x = self._encode(kyoku_log, log_index, player_id)
        heads = self._forward_heads(x)
        if "kan" not in heads:
            return False
        probs = F.softmax(heads["kan"][0], dim=-1)
        return bool(probs.argmax().item() == 1)

    # ---- high-level convenience ---------------------------------------

    def on_zimo(self, kyoku_log, log_index, player_id,
                my_hand_str: Optional[str] = None) -> Dict:
        """Top-level choice after self-draw: discard / riichi / ankan."""
        # Ankan first (most committing)
        if my_hand_str and self.decide_ankan(kyoku_log, log_index, player_id, my_hand_str):
            return {"action": "ankan"}
        if self.decide_riichi(kyoku_log, log_index, player_id):
            tile = self.decide_discard(kyoku_log, log_index, player_id)
            tile["action"] = "riichi"
            return tile
        return self.decide_discard(kyoku_log, log_index, player_id)

    def on_opponent_dapai(self, kyoku_log, log_index, my_player_id,
                          my_hand_str: str) -> Dict:
        """Alias for :meth:`decide_call`."""
        return self.decide_call(kyoku_log, log_index, my_player_id, my_hand_str)
