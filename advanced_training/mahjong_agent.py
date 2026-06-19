"""
Inference wrapper that turns a trained checkpoint into a playable agent.

The agent accepts a running kyoku log (same shape used at training time) and
returns a decision dict whose ``"action"`` value is a 牌譜形式 action name
matching the training head namespace:

    ``"dapai"``  - discard a tile (``"tile"``/``"tile_id"`` included).
    ``"riichi"`` - declare riichi together with a discard.
    ``"fulou"``  - call on an opponent's discard. The specific call type is
                   reported via the additional ``"call_type"`` field, whose
                   value is ``"chi"``, ``"pon"``, or ``"daiminkan"``.
    ``"gang"``   - declare ankan (or kakan, when supported).
    ``"hule"``   - declare a win (not currently emitted by these methods).
    ``"pass"``   - take no action.

Other keys (``"tile"``, ``"tile_id"``, ``"confidence"``) are included where
relevant. Returned dicts are *agent decisions*, not raw 牌譜形式 log records
such as ``{"dapai": ...}``.

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

from mahjong_ai_features import (  # noqa: E402
    FEATURE_SCHEMA_VERSION,
    FEATURE_TILE_MAP,
    StateEncoderV2,
)
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
        saved_schema = payload.get("config", {}).get("feature_schema_version")
        if saved_schema != FEATURE_SCHEMA_VERSION:
            raise RuntimeError(
                "Checkpoint feature schema is incompatible with the corrected encoder "
                f"(saved={saved_schema!r}, required={FEATURE_SCHEMA_VERSION!r}). "
                "Retrain the model before using MahjongAgent."
            )
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
        # Single-head DiscardModel: expose as "dapai" head.
        return {"dapai": out}

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
        logits = heads["dapai"][0]
        if forbid_tiles:
            logits = logits.clone()
            for t in forbid_tiles:
                logits[t] = float("-inf")
        tile_id = int(logits.argmax().item())
        return {
            "action": "dapai",
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

        # StateEncoderV2.encode(idx) returns state *before* kyoku_log[idx], so
        # pass ``discard_index + 1`` to include the triggering discard in the
        # river/last-discard features.  This matches the fulou-positive
        # training convention (state encoded at the fulou event index).
        x = self._encode(kyoku_log, discard_index + 1, my_player_id)
        heads = self._forward_heads(x)

        if "fulou" not in heads:
            return {"action": "pass"}

        # fulou head: 0=pass, 1=chi, 2=pon, 3=daiminkan
        probs = F.softmax(heads["fulou"][0], dim=-1)
        # Mask out impossible *calls* (chi/pon/daiminkan). "pass" is always a
        # legal option and should compete on its learned probability, so keep
        # its value untouched here.
        masked = probs.clone()
        if not chi_ok:
            masked[1] = 0.0
        if not pon_ok:
            masked[2] = 0.0
        if not kan_ok:
            masked[3] = 0.0

        best_idx = int(masked.argmax().item())
        best_prob = float(probs[best_idx])
        # Map fulou-head class index -> 牌譜形式 sub-type.
        _IDX_TO_CALL_TYPE = {1: "chi", 2: "pon", 3: "daiminkan"}
        if best_idx == 0 or best_prob < 0.5:
            return {"action": "pass"}
        # Report all fulou variants under the unified ``"fulou"`` action so
        # the returned namespace stays aligned with the training head names.
        return {
            "action": "fulou",
            "call_type": _IDX_TO_CALL_TYPE[best_idx],
            "confidence": best_prob,
        }

    def decide_ankan(self, kyoku_log, log_index, player_id,
                     my_hand_str: str) -> bool:
        """Return True iff ankan is possible and the gang head prefers it.

        gang head: 0=pass, 1=ankan, 2=kakan.
        """
        if not self.is_multitask:
            return False
        hand37 = hand_counter_from_str(my_hand_str)
        if not can_ankan(hand37):
            return False
        x = self._encode(kyoku_log, log_index, player_id)
        heads = self._forward_heads(x)
        if "gang" not in heads:
            return False
        probs = F.softmax(heads["gang"][0], dim=-1)
        # Prefer ankan (idx=1) over pass (idx=0); ignore kakan (idx=2) here.
        return bool(probs[1] > probs[0])

    # ---- high-level convenience ---------------------------------------

    def on_zimo(self, kyoku_log, log_index, player_id,
                my_hand_str: Optional[str] = None) -> Dict:
        """Top-level choice after a self-draw at ``kyoku_log[log_index]``.

        ``log_index`` is the index of the just-observed zimo event.  Because
        :meth:`StateEncoderV2.encode` returns the state *before* ``log_index``,
        the internal decision helpers are invoked with ``log_index + 1`` so
        the encoded state includes the drawn tile.
        """
        state_index = log_index + 1
        # Ankan first (most committing)
        if my_hand_str and self.decide_ankan(kyoku_log, state_index, player_id, my_hand_str):
            return {"action": "gang"}
        if self.decide_riichi(kyoku_log, state_index, player_id):
            tile = self.decide_discard(kyoku_log, state_index, player_id)
            tile["action"] = "riichi"
            return tile
        return self.decide_discard(kyoku_log, state_index, player_id)

    def on_opponent_dapai(self, kyoku_log, log_index, my_player_id,
                          my_hand_str: str) -> Dict:
        """Alias for :meth:`decide_call`.

        ``log_index`` is the index of the observed opponent ``dapai`` event;
        :meth:`decide_call` internally encodes the state including that
        discard.
        """
        return self.decide_call(kyoku_log, log_index, my_player_id, my_hand_str)
