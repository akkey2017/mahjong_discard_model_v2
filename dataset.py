"""
Dataset handling for mahjong game records.

Each *sample* is a tuple ``(kyoku_log, log_index, player_id, action_type, label)``
where ``action_type`` and ``label`` follow 牌譜形式 (game record format) key names:

    "dapai"   -> label: 0..33 (tile index; 34 standard tiles)
    "riichi"  -> label: 0/1   (declare riichi or not; negative emitted for every dapai)
    "fulou"   -> label: 0=pass, 1=chi, 2=pon, 3=daiminkan
    "gang"    -> label: 0=pass/no-gang, 1=ankan, 2=kakan
    "hule"    -> label: 1=win (tsumo or ron; no negatives synthesized)

Negative samples for fulou (label=0) can be synthesized via
``include_fulou_negatives``. Gang negatives (label=0) are emitted automatically
for every dapai where ankan was possible but not taken.

Breaking change: older dataset code exposed discard-only samples with the
action type ``"discard"``. This module now uses 牌譜形式 names such as
``"dapai"`` (including when ``collect_all_actions=False``). Callers that
previously filtered for ``"discard"`` must update to ``"dapai"``.
"""

import hashlib
import json
import re
import sys
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from mahjong_ai_features import (
    StateEncoderV2,
    _process_single_number,
    FEATURE_TILE_MAP,
    _make_pai_counter_list_from,
    _fulou_to_pais,
)
from mahjong_rules import can_ankan, can_chi, can_pon, can_daiminkan, tile_count, normalize_red_five


# ----- helpers for fulou/gang classification ------------------------------


def _strip_direction(meld_str: str) -> str:
    """Remove '+', '-', '=' direction markers from a meld string."""
    return re.sub(r"[+\-=]", "", meld_str)


def classify_fulou(meld_str: str) -> str:
    """Classify a 'fulou' meld record into 'chi'/'pon'/'daiminkan'.

    The ``m`` field in fulou records is e.g.:
        's1-23'    : chi (3 tiles forming a run; one direction mark)
        'p5=55'    : pon (3 identical tiles; one direction mark)
        'z3333+'   : daiminkan (4 identical tiles; one direction mark)

    The distinction is made by tile count (3 vs 4) and whether the tiles are
    identical (pon) or form a run (chi).
    """
    cleaned = _strip_direction(meld_str)
    # parse tiles by suit prefix
    tiles = []
    suit = ""
    for c in cleaned:
        if c in "mpsz":
            suit = c
        elif c.isdigit():
            tiles.append(f"{suit}{c}")
    if len(tiles) == 4:
        return "daiminkan"
    # 3 tiles: pon if all equal (normalizing red-5s 'm0'->'m5' etc.)
    def _norm(t):
        if len(t) == 2 and t[1] == "0":
            return f"{t[0]}5"
        return t
    norm = [_norm(t) for t in tiles]
    if len(set(norm)) == 1:
        return "pon"
    return "chi"


def classify_gang(meld_str: str) -> str:
    """Classify an (ankan/kakan) gang meld. 'gang' records never include fulou.

    Heuristic:
      - Kakan ``m`` contains direction marker (e.g. 'p555=0') because the
        original pon was from another player.
      - Ankan ``m`` has no direction marker (e.g. 'p5555').
    """
    return "kakan" if re.search(r"[+\-=]", meld_str) else "ankan"


# ----- kyoku-level sample extraction --------------------------------------


def _fulou_source_seat(p_id: int, meld_str: str) -> int:
    """Resolve the seat (0..3) the called tile came from.

    Direction markers follow kobalab's convention:
        ``-``: kamicha (upper seat, ``(p_id + 3) % 4``)
        ``=``: toimen  (across,      ``(p_id + 2) % 4``)
        ``+``: shimocha (lower seat, ``(p_id + 1) % 4``)
    """
    if "-" in meld_str:
        return (p_id + 3) % 4
    if "=" in meld_str:
        return (p_id + 2) % 4
    return (p_id + 1) % 4  # '+' (or fallback)


def _generate_fulou_negatives(kyoku_log, discard_index, hands):
    """Yield fulou=0 (pass) negatives for players who could have called but didn't.

    ``hands`` must reflect the state *just before* ``kyoku_log[discard_index]``
    (i.e. the incremental state maintained by the caller — no reconstruction).
    """
    move = kyoku_log[discard_index]
    if "dapai" not in move:
        return
    discarder = move["dapai"]["l"]
    tile_str = move["dapai"]["p"].replace("*", "").replace("_", "")
    tile_37 = FEATURE_TILE_MAP.get(tile_str)
    if tile_37 is None:
        return

    # Look at the immediate next event to decide if the discard was claimed or
    # resolved by ron before any other player could act.
    called_by = None
    for k in range(discard_index + 1, len(kyoku_log)):
        nxt = kyoku_log[k]
        if "fulou" in nxt:
            called_by = nxt["fulou"]["l"]
        elif "hule" in nxt:
            # Ron pre-empts all other calls — no valid "didn't call" negatives.
            return
        break  # only inspect the immediate next event

    # StateEncoderV2.encode(idx) returns state *before* kyoku_log[idx], so
    # use ``discard_index + 1`` here to make the encoded state include the
    # triggering discard (river/last-discard features).  Positive fulou
    # samples are emitted at the fulou event index — which is typically
    # ``discard_index + 1`` — and therefore see the same state.
    encode_index = discard_index + 1
    for seat in range(4):
        if seat == discarder or seat == called_by:
            continue
        from_shimocha = ((discarder + 1) % 4) != seat
        chi_ok = (not from_shimocha) and can_chi(hands[seat], tile_37, from_shimocha=False)
        pon_ok = can_pon(hands[seat], tile_37)
        kan_ok = can_daiminkan(hands[seat], tile_37)
        if kan_ok or pon_ok or chi_ok:
            yield (kyoku_log, encode_index, seat, "fulou", 0)


def _has_kakan_option(hand37, pon_tiles):
    """True if the player can add to any of their existing pon melds (kakan).

    ``pon_tiles`` is a list of normalized tile_37 ids (red-5 collapsed to 5)
    recorded when each pon was made.  ``tile_count`` handles the red-5 / normal-5
    equivalence so both m0 and m5 in hand count toward a m5 pon.
    """
    for pon_tile in pon_tiles:
        if tile_count(hand37, pon_tile) >= 1:
            return True
    return False


def _extract_samples_from_kyoku(kyoku_log, collect_all_actions=False,
                                include_fulou_negatives=False):
    """Yield samples for a single kyoku.

    Args:
        kyoku_log: list of move dicts (starts with {'qipai': ...}).
        collect_all_actions: if False, only emit 'dapai' samples.
        include_fulou_negatives: if True, synthesize fulou-pass samples (label 0)
            for other players who could have called but didn't.

    When ``collect_all_actions=True`` also emits:
      - riichi 0/1 labels for every dapai
      - gang=0 negatives for every dapai where the discarder could have done
        ankan *or* kakan but chose to discard instead
      - fulou / gang / hule positive samples
    """
    if not kyoku_log or "qipai" not in kyoku_log[0]:
        return

    _FULOU_LABEL = {"chi": 1, "pon": 2, "daiminkan": 3}
    _GANG_LABEL = {"ankan": 1, "kakan": 2}

    # Incrementally maintained game state.  We only need this when emitting
    # anything beyond plain dapai samples.
    if collect_all_actions:
        hands = [_make_pai_counter_list_from(h) for h in kyoku_log[0]["qipai"]["shoupai"]]
        rivers = [[], [], [], []]
        # Normalized pon tile per player (for kakan feasibility checks).
        pon_melds = [[], [], [], []]
    else:
        hands = rivers = pon_melds = None

    for i, move in enumerate(kyoku_log):
        if "dapai" in move:
            p_id = move["dapai"]["l"]
            raw = move["dapai"]["p"]
            tile_str = raw.replace("*", "").replace("_", "")
            if tile_str not in FEATURE_TILE_MAP:
                # Unknown tile string — skip emitting samples for this move.
                # The hand counter will drift slightly but ankan/daiminkan
                # feasibility checks tolerate that.
                continue
            tile_id_37 = FEATURE_TILE_MAP[tile_str]
            label = _process_single_number(tile_id_37)
            yield (kyoku_log, i, p_id, "dapai", label)

            if collect_all_actions:
                yield (kyoku_log, i, p_id, "riichi", 1 if "*" in raw else 0)

                # Gang negative: could have done ankan or kakan but chose to discard.
                if can_ankan(hands[p_id]) or _has_kakan_option(hands[p_id], pon_melds[p_id]):
                    yield (kyoku_log, i, p_id, "gang", 0)

                if include_fulou_negatives:
                    yield from _generate_fulou_negatives(kyoku_log, i, hands)

                # Advance state: tile leaves hand, enters river.
                hands[p_id][tile_id_37] = max(0, hands[p_id][tile_id_37] - 1)
                rivers[p_id].append(tile_id_37)

        elif "zimo" in move:
            if collect_all_actions:
                p = move["zimo"]["l"]
                t = FEATURE_TILE_MAP.get(move["zimo"]["p"])
                if t is not None:
                    hands[p][t] += 1

        elif "gangzimo" in move:
            if collect_all_actions:
                p = move["gangzimo"]["l"]
                t = FEATURE_TILE_MAP.get(move["gangzimo"]["p"])
                if t is not None:
                    hands[p][t] += 1

        elif "fulou" in move:
            p_id = move["fulou"]["l"]
            try:
                call_type = classify_fulou(move["fulou"]["m"])
            except Exception:
                continue
            if collect_all_actions:
                yield (kyoku_log, i, p_id, "fulou", _FULOU_LABEL[call_type])

                # Advance state: remove consumed tiles, pop river of source seat.
                meld = move["fulou"]["m"]
                tiles = _fulou_to_pais(meld)
                if tiles:
                    from_p_id = _fulou_source_seat(p_id, meld)
                    consumed = list(tiles)
                    if rivers[from_p_id]:
                        taken = rivers[from_p_id].pop()
                        if taken in consumed:
                            consumed.remove(taken)
                    for t in consumed:
                        hands[p_id][t] = max(0, hands[p_id][t] - 1)
                    if call_type == "pon":
                        # Record the normalized pon tile for future kakan checks.
                        pon_melds[p_id].append(normalize_red_five(tiles[0]))

        elif "gang" in move:
            p_id = move["gang"]["l"]
            try:
                gang_type = classify_gang(move["gang"]["m"])
            except Exception:
                continue
            if collect_all_actions:
                yield (kyoku_log, i, p_id, "gang", _GANG_LABEL[gang_type])

                meld = move["gang"]["m"]
                tiles = _fulou_to_pais(meld)
                if tiles:
                    if gang_type == "kakan":
                        kakan_tile = tiles[-1]
                        hands[p_id][kakan_tile] = max(0, hands[p_id][kakan_tile] - 1)
                        # Remove the upgraded pon from tracking.
                        norm = normalize_red_five(kakan_tile)
                        if norm in pon_melds[p_id]:
                            pon_melds[p_id].remove(norm)
                    else:  # ankan
                        for t in tiles:
                            hands[p_id][t] = max(0, hands[p_id][t] - 1)

        elif collect_all_actions and "hule" in move:
            # NOTE: we currently only emit positive ``hule`` samples (label=1).
            # The training loop sets the ``hule`` task weight to 0 by default
            # so the 2-class head is not pushed toward "always win"; negative
            # samples (when a player *could* have declared ron/tsumo but did
            # not) require ron/tenpai-from-opponent-discard detection and are
            # out of scope for this change.
            p_id = move["hule"]["l"]
            yield (kyoku_log, i, p_id, "hule", 1)


class MahjongDataset(Dataset):
    """Dataset for mahjong game records.

    Args:
        zip_path: Path to ZIP file containing game logs (*.txt with JSON).
        max_files: Maximum number of files to load from the ZIP.
        samples: Pre-loaded samples (for creating subsets).
        verbose: Whether to show progress during loading.
        collect_all_actions: If True, also emit riichi/chi/pon/kan/agari
            samples. If False (default), only 'dapai' samples are kept.
    """

    def __init__(
        self,
        zip_path=None,
        max_files=10000,
        samples=None,
        verbose=True,
        collect_all_actions=False,
        include_fulou_negatives=False,
    ):
        if samples is not None:
            self.samples = samples
            self.game_ids = [None] * len(samples)
            return

        self.samples = []
        # Parallel array: source archive filename for each sample (for game-level splits)
        self.game_ids = []

        if zip_path is None:
            return

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                txts = [n for n in zf.namelist() if n.endswith(".txt")]
                iterator = txts[:max_files]
                if verbose:
                    iterator = tqdm(
                        iterator,
                        desc="Loading game records",
                        file=sys.stderr,
                        dynamic_ncols=True,
                        mininterval=0.1,
                    )

                for name in iterator:
                    try:
                        raw = zf.read(name).decode("utf-8")
                        game_data = json.loads(raw)
                        if "log" not in game_data:
                            continue
                        game_id = name
                        for kyoku_log in game_data["log"]:
                            for sample in _extract_samples_from_kyoku(
                                kyoku_log,
                                collect_all_actions=collect_all_actions,
                                include_fulou_negatives=include_fulou_negatives,
                            ):
                                self.samples.append(sample)
                                self.game_ids.append(game_id)
                    except json.JSONDecodeError:
                        if verbose:
                            print(f"Warning: Failed to parse JSON in {name}")
                        continue
                    except Exception as e:
                        if verbose:
                            print(f"Warning: Error processing {name}: {e}")
                        continue

        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {zip_path}")
        except zipfile.BadZipFile:
            raise ValueError(f"Invalid ZIP file: {zip_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        kyoku_log, log_idx, p_id, action_type, label = self.samples[idx]
        encoder = StateEncoderV2(kyoku_log, p_id)
        state_tensor = encoder.encode(log_idx)
        return state_tensor, label, action_type

    def filter_by_action(self, action_type):
        """Return a new dataset containing only samples of a specific action type.

        Accepts the legacy alias ``"discard"`` as a synonym for ``"dapai"`` so
        older callers (train.py, evaluate_model.py, mahjong_ai_coatnet_v2.py)
        keep working after the rename to 牌譜形式 keys.
        """
        if action_type == "discard":
            action_type = "dapai"
        filtered_samples = []
        filtered_games = []
        for s, g in zip(self.samples, self.game_ids):
            if s[3] == action_type:
                filtered_samples.append(s)
                filtered_games.append(g)
        ds = MahjongDataset(samples=filtered_samples, verbose=False)
        ds.game_ids = filtered_games
        return ds

    def get_statistics(self):
        if not self.samples:
            return {"total": 0, "action_counts": {}}
        action_counts = {}
        for sample in self.samples:
            action = sample[3]
            action_counts[action] = action_counts.get(action, 0) + 1
        return {
            "total": len(self.samples),
            "action_counts": action_counts,
        }


# ----- collation & split helpers ------------------------------------------


def multitask_collate(batch):
    """Collate samples with heterogeneous action types.

    Returns (states, labels, action_types_as_list_of_str) so downstream code
    can mask losses by action type.
    """
    states = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    actions = [b[2] for b in batch]
    return states, labels, actions


def _game_level_split(dataset, train_ratio, seed):
    """Split samples so no single game_id appears in both train and val."""
    game_ids = dataset.game_ids
    if not any(gid is not None for gid in game_ids):
        # fall back to sample-level split
        return None

    # Deterministic hash-based bucketing
    train_idx, val_idx = [], []
    for i, gid in enumerate(game_ids):
        key = f"{seed}:{gid}"
        digest = int(hashlib.md5(key.encode()).hexdigest(), 16)
        bucket = (digest % 1000) / 1000.0
        (train_idx if bucket < train_ratio else val_idx).append(i)
    return train_idx, val_idx


def create_dataloaders(
    dataset,
    train_ratio=0.9,
    batch_size=64,
    num_workers=2,
    pin_memory=True,
    seed=42,
    split_by_game=False,
):
    """Create train and validation DataLoaders for a single-action dataset."""
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    if split_by_game:
        split = _game_level_split(dataset, train_ratio, seed)
    else:
        split = None

    if split is None:
        train_size = int(len(dataset) * train_ratio)
        val_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(seed)
        train_set, val_set = random_split(
            dataset, [train_size, val_size], generator=generator
        )
    else:
        train_idx, val_idx = split
        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def create_multitask_dataloaders(
    dataset,
    train_ratio=0.9,
    batch_size=64,
    num_workers=2,
    pin_memory=True,
    seed=42,
    split_by_game=False,
):
    """Train/val DataLoaders that preserve action_type labels (list of str)."""
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    if split_by_game:
        split = _game_level_split(dataset, train_ratio, seed)
    else:
        split = None

    if split is None:
        train_size = int(len(dataset) * train_ratio)
        val_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(seed)
        train_set, val_set = random_split(
            dataset, [train_size, val_size], generator=generator
        )
    else:
        train_idx, val_idx = split
        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=multitask_collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=multitask_collate,
    )
    return train_loader, val_loader
