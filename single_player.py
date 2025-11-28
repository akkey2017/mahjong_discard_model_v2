"""
Single-player mahjong simulation with AI discard decisions.

This module allows users to observe the AI making discard decisions
in a simulated single-player mahjong game (一人麻雀).
"""

import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np

from models import create_coatnet_model, create_resnet_model, create_vit_model


# ==================== Tile Definitions ====================

# Full tile set (using 37-dimensional representation internally)
# m0, m1-9: Manzu (萬子) - 0 is red 5
# p0, p1-9: Pinzu (筒子) - 0 is red 5
# s0, s1-9: Souzu (索子) - 0 is red 5
# z1-7: Jihai (字牌) - East, South, West, North, White, Green, Red

# Tile string to 37-dim ID mapping
TILE_TO_ID_37 = {
    **{f"m{i}": i for i in range(1, 10)},
    "m0": 0,  # Red 5m
    **{f"p{i}": i + 10 for i in range(1, 10)},
    "p0": 10,  # Red 5p
    **{f"s{i}": i + 20 for i in range(1, 10)},
    "s0": 20,  # Red 5s
    **{f"z{i}": i + 29 for i in range(1, 8)},
}

# 37-dim ID to tile string
ID_37_TO_TILE = {v: k for k, v in TILE_TO_ID_37.items()}

# 34-dim ID to tile string (used for model output)
ID_34_TO_TILE = {
    **{i: f"m{i+1}" for i in range(9)},
    **{i + 9: f"p{i+1}" for i in range(9)},
    **{i + 18: f"s{i+1}" for i in range(9)},
    **{i + 27: f"z{i+1}" for i in range(7)},
}

# Tile display names (Japanese/emoji style)
TILE_DISPLAY = {
    "m1": "一萬", "m2": "二萬", "m3": "三萬", "m4": "四萬", "m5": "五萬",
    "m6": "六萬", "m7": "七萬", "m8": "八萬", "m9": "九萬", "m0": "🔴五萬",
    "p1": "一筒", "p2": "二筒", "p3": "三筒", "p4": "四筒", "p5": "五筒",
    "p6": "六筒", "p7": "七筒", "p8": "八筒", "p9": "九筒", "p0": "🔴五筒",
    "s1": "一索", "s2": "二索", "s3": "三索", "s4": "四索", "s5": "五索",
    "s6": "六索", "s7": "七索", "s8": "八索", "s9": "九索", "s0": "🔴五索",
    "z1": "東", "z2": "南", "z3": "西", "z4": "北",
    "z5": "白", "z6": "發", "z7": "中",
}

# Compact tile display (using Unicode Mahjong tiles when possible)
TILE_COMPACT = {
    "m1": "🀇", "m2": "🀈", "m3": "🀉", "m4": "🀊", "m5": "🀋",
    "m6": "🀌", "m7": "🀍", "m8": "🀎", "m9": "🀏", "m0": "🀋r",
    "p1": "🀙", "p2": "🀚", "p3": "🀛", "p4": "🀜", "p5": "🀝",
    "p6": "🀞", "p7": "🀟", "p8": "🀠", "p9": "🀡", "p0": "🀝r",
    "s1": "🀐", "s2": "🀑", "s3": "🀒", "s4": "🀓", "s5": "🀔",
    "s6": "🀕", "s7": "🀖", "s8": "🀗", "s9": "🀘", "s0": "🀔r",
    "z1": "🀀", "z2": "🀁", "z3": "🀂", "z4": "🀃",
    "z5": "🀆", "z6": "🀅", "z7": "🀄",
}


def id_37_to_34(tile_id_37):
    """Convert 37-dim tile ID to 34-dim ID (merge red 5 with normal 5)."""
    if tile_id_37 < 10:  # Manzu
        return tile_id_37 - 1 if tile_id_37 != 0 else 4
    elif tile_id_37 < 20:  # Pinzu
        return tile_id_37 - 11 + 9 if tile_id_37 != 10 else 13
    elif tile_id_37 < 30:  # Souzu
        return tile_id_37 - 21 + 18 if tile_id_37 != 20 else 22
    else:  # Jihai
        return tile_id_37 - 30 + 27


def id_34_to_37(tile_id_34):
    """Convert 34-dim tile ID to 37-dim ID (use normal tile, not red)."""
    if tile_id_34 < 9:  # Manzu
        return tile_id_34 + 1
    elif tile_id_34 < 18:  # Pinzu
        return tile_id_34 - 9 + 11
    elif tile_id_34 < 27:  # Souzu
        return tile_id_34 - 18 + 21
    else:  # Jihai
        return tile_id_34 - 27 + 30


# ==================== Wall and Hand Management ====================


def create_wall(seed=None):
    """
    Create a shuffled wall of 136 tiles.
    
    Returns:
        List of tile IDs (37-dim representation)
    """
    if seed is not None:
        random.seed(seed)
    
    wall = []
    
    # Add 4 of each tile (except red 5s)
    for suit in ['m', 'p', 's']:
        for num in range(1, 10):
            tile_id = TILE_TO_ID_37[f"{suit}{num}"]
            if num == 5:
                # 3 normal 5s, 1 red 5
                wall.extend([tile_id] * 3)
                red_id = TILE_TO_ID_37[f"{suit}0"]
                wall.append(red_id)
            else:
                wall.extend([tile_id] * 4)
    
    # Add 4 of each honor tile
    for num in range(1, 8):
        tile_id = TILE_TO_ID_37[f"z{num}"]
        wall.extend([tile_id] * 4)
    
    random.shuffle(wall)
    return wall


def tile_sort_key(tile_id):
    """Get sort key for a tile ID for consistent hand display."""
    # Sort by suit (m=0, p=1, s=2, z=3) then by number (red 5 sorts as 5)
    return (tile_id // 10, tile_id % 10 if tile_id % 10 != 0 else 5)


def sort_tiles(tiles):
    """Sort a list of tile IDs for consistent display."""
    return sorted(tiles, key=tile_sort_key)


def deal_initial_hand(wall, player_count=1):
    """
    Deal initial hands from the wall.
    
    Args:
        wall: List of tile IDs
        player_count: Number of players (1-4)
    
    Returns:
        List of hands (each hand is a list of tile IDs)
    """
    hands = [[] for _ in range(player_count)]
    
    # Deal 13 tiles to each player
    for _ in range(13):
        for i in range(player_count):
            if wall:
                hands[i].append(wall.pop())
    
    # Sort hands for display
    for hand in hands:
        hand.sort(key=tile_sort_key)
    
    return hands


def draw_tile(wall):
    """Draw a tile from the wall."""
    if wall:
        return wall.pop()
    return None


def format_hand(hand, use_compact=False):
    """Format a hand for display."""
    display_map = TILE_COMPACT if use_compact else TILE_DISPLAY
    tiles = []
    for tile_id in sort_tiles(hand):
        tile_str = ID_37_TO_TILE[tile_id]
        tiles.append(display_map.get(tile_str, tile_str))
    return " ".join(tiles)


def format_hand_simple(hand):
    """Format hand in simple m/p/s/z notation."""
    result = []
    current_suit = None
    suit_order = {0: 'm', 1: 'p', 2: 's', 3: 'z'}
    
    # Group tiles by suit
    suits = {s: [] for s in 'mpsz'}
    for tile_id in hand:
        tile_str = ID_37_TO_TILE[tile_id]
        suit = tile_str[0]
        num = tile_str[1]
        suits[suit].append(num)
    
    # Format each suit
    for suit in 'mpsz':
        if suits[suit]:
            nums = sorted(suits[suit], key=lambda x: int(x) if x != '0' else 5)
            result.append("".join(nums) + suit)
    
    return "".join(result)


# ==================== State Encoding for AI ====================


def encode_state_for_ai(hand, discards, dora_indicator, turn=1, wall_remaining=70):
    """
    Encode the current game state for the AI model.
    
    This creates a simplified state tensor matching the expected (380, 4, 9) format.
    
    Args:
        hand: List of tile IDs (37-dim) in current hand
        discards: List of discarded tile IDs
        dora_indicator: Dora indicator tile ID
        turn: Current turn number
        wall_remaining: Number of tiles remaining in wall
    
    Returns:
        Tensor of shape (380, 4, 9)
    """
    num_channels = 380
    tensor = np.zeros((num_channels, 4, 9), dtype=np.float32)
    ch_offset = 0
    
    # Helper to encode tiles into tensor
    def encode_tiles(ch_start, tiles_37, max_ch=4):
        """Encode tiles into the tensor."""
        # Convert 37-dim counts to 34-dim
        tiles_34 = [0] * 34
        for i, count in enumerate(tiles_37):
            if count > 0:
                tiles_34[id_37_to_34(i)] += count
        
        # m, p, s suits (0-26)
        for suit in range(3):
            for num in range(9):
                count = tiles_34[suit * 9 + num]
                for c in range(min(count, 4)):
                    tensor[ch_start + c, suit, num] = 1.0
        
        # z (honors) at index 27-33
        for i in range(7):
            count = tiles_34[27 + i]
            for c in range(min(count, 4)):
                tensor[ch_start + c, 3, i] = 1.0
        
        # Red tiles (if tracking separately)
        if max_ch >= 7:
            for suit_idx, red_idx in enumerate([0, 10, 20]):
                if tiles_37[red_idx] > 0:
                    tensor[ch_start + 4 + suit_idx, suit_idx, 4] = 1.0
    
    # Convert hand to 37-dim counts
    hand_37 = [0] * 37
    for tile_id in hand:
        hand_37[tile_id] += 1
    
    # A. Hand (7ch for player 0, 7ch each for dummy opponents = 28ch total)
    encode_tiles(ch_offset, hand_37, max_ch=7)
    ch_offset += 28  # Skip all 4 players' hand channels
    
    # B. Melds (16ch × 4 = 64ch) - No melds in single-player simplified game
    ch_offset += 64
    
    # C. Ankan (4ch × 4 = 16ch) - No ankans in single-player simplified game
    ch_offset += 16
    
    # D. River/discards (7ch × 4 = 28ch)
    discard_37 = [0] * 37
    for tile_id in discards:
        discard_37[tile_id] += 1
    encode_tiles(ch_offset, discard_37, max_ch=7)
    ch_offset += 28
    
    # E. Riichi status (1ch × 4 = 4ch) - Not in riichi
    ch_offset += 4
    
    # F. Dora (4ch)
    if dora_indicator is not None:
        # Calculate actual dora from indicator and encode directly
        dora_id = calculate_dora(dora_indicator)
        if dora_id is not None:
            dora_34_idx = id_37_to_34(dora_id)
            # Encode dora position in tensor
            if dora_34_idx < 27:  # Number tiles (m, p, s)
                suit = dora_34_idx // 9
                num = dora_34_idx % 9
                tensor[ch_offset, suit, num] = 1.0
            else:  # Honor tiles
                tensor[ch_offset, 3, dora_34_idx - 27] = 1.0
    ch_offset += 4
    
    # G. Round info (9ch)
    # Round wind (3ch)
    tensor[ch_offset, :, :] = 1.0  # East round
    ch_offset += 3
    # Round number (4ch)
    tensor[ch_offset, :, :] = 1.0  # First round
    ch_offset += 4
    # Honba (1ch)
    ch_offset += 1
    # Riichi sticks (1ch)
    ch_offset += 1
    
    # H. Player scores (4ch)
    for i in range(4):
        tensor[ch_offset, :, :] = 0.25  # 25000 normalized
        ch_offset += 1
    
    # I. Seat winds (16ch)
    tensor[ch_offset, :, :] = 1.0  # Player is East
    ch_offset += 16
    
    # J. Remaining tiles (1ch)
    tensor[ch_offset, :, :] = wall_remaining / 70.0
    ch_offset += 1
    
    # K. Visible tiles (7ch)
    visible_37 = discard_37.copy()  # Only discards are visible in single-player
    encode_tiles(ch_offset, visible_37, max_ch=7)
    ch_offset += 7
    
    # L. Ura-dora (4ch) - hidden
    ch_offset += 4
    
    # M. Furiten status (4ch)
    ch_offset += 4
    
    # N. Last discard (7ch)
    if discards:
        last_37 = [0] * 37
        last_37[discards[-1]] = 1
        encode_tiles(ch_offset, last_37, max_ch=7)
    ch_offset += 7
    
    # O. Riichi turn (4ch)
    ch_offset += 4
    
    # P. Ippatsu (4ch)
    ch_offset += 4
    
    # Q. Double riichi (1ch)
    ch_offset += 1
    
    # R. First turn (1ch)
    if turn == 1:
        tensor[ch_offset, :, :] = 1.0
    ch_offset += 1
    
    # S. Haitei/Houtei proximity (1ch)
    if wall_remaining < 5:
        tensor[ch_offset, :, :] = 1.0
    ch_offset += 1
    
    # T. Dora counts per player (4ch)
    ch_offset += 4
    
    # U. Unseen tile counts (7ch)
    unseen_37 = [4] * 37
    unseen_37[0] = unseen_37[10] = unseen_37[20] = 1  # Red fives
    for tile_id in hand:
        unseen_37[tile_id] = max(0, unseen_37[tile_id] - 1)
    for tile_id in discards:
        unseen_37[tile_id] = max(0, unseen_37[tile_id] - 1)
    encode_tiles(ch_offset, unseen_37, max_ch=7)
    ch_offset += 7
    
    # V. Genbutsu/safe tiles (28ch)
    ch_offset += 28
    
    # W. Turn number (1ch)
    tensor[ch_offset, :, :] = turn / 20.0
    ch_offset += 1
    
    # X. Each player's last discard (28ch)
    ch_offset += 28
    
    # Y. Honba count (1ch)
    ch_offset += 1
    
    # Z. Meld counts (4ch)
    ch_offset += 4
    
    # Rest is padding/future use
    
    return torch.from_numpy(tensor)


def calculate_dora(indicator_id):
    """Calculate the dora tile from the indicator."""
    if indicator_id is None:
        return None
    
    tile_str = ID_37_TO_TILE[indicator_id]
    suit = tile_str[0]
    num_str = tile_str[1]
    
    if suit in 'mps':
        # Number tiles: next number (9 -> 1)
        num = int(num_str) if num_str != '0' else 5
        if num == 9:
            next_num = 1
        else:
            next_num = num + 1
        return TILE_TO_ID_37[f"{suit}{next_num}"]
    else:
        # Honor tiles
        num = int(num_str)
        if num <= 4:  # Winds: E->S->W->N->E
            next_num = (num % 4) + 1
        else:  # Dragons: W->G->R->W
            next_num = ((num - 5) % 3) + 5
        return TILE_TO_ID_37[f"z{next_num}"]


# ==================== AI Decision Making ====================


def get_ai_discard(model, state_tensor, hand, device, top_k=5):
    """
    Get the AI's discard decision.
    
    Args:
        model: Trained discard model
        state_tensor: Encoded game state
        hand: Current hand (list of 37-dim tile IDs)
        device: PyTorch device
        top_k: Number of top predictions to return
    
    Returns:
        Tuple of (chosen_tile_id_37, predictions)
        predictions is a list of (tile_id_34, probability)
    """
    model.eval()
    
    with torch.no_grad():
        # Add batch dimension
        input_tensor = state_tensor.unsqueeze(0).to(device)
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)[0]
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
        predictions = [
            (idx.item(), prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]
    
    # Find the highest-probability tile that's actually in hand
    hand_34_set = set(id_37_to_34(tile_id) for tile_id in hand)
    
    # Get all probabilities and sort by probability
    all_probs = [(i, probabilities[i].item()) for i in range(34)]
    all_probs.sort(key=lambda x: x[1], reverse=True)
    
    chosen_34 = None
    for tile_34, prob in all_probs:
        if tile_34 in hand_34_set:
            chosen_34 = tile_34
            break
    
    if chosen_34 is None:
        # Fallback: pick random tile from hand (should be extremely rare)
        # This could happen if model outputs are corrupted or hand is empty
        import sys
        print("⚠️ Warning: No valid tile prediction found, using random selection", 
              file=sys.stderr)
        chosen_34 = id_37_to_34(random.choice(hand))
    
    # Convert chosen 34-dim ID back to 37-dim (prefer red if available)
    chosen_37 = None
    for tile_id in hand:
        if id_37_to_34(tile_id) == chosen_34:
            chosen_37 = tile_id
            break
    
    return chosen_37, predictions


# ==================== Game Simulation ====================


def run_single_player_game(model, device, max_turns=18, seed=None, verbose=True, compact=False, interactive=False):
    """
    Run a single-player mahjong game simulation.
    
    Args:
        model: Trained discard model
        device: PyTorch device
        max_turns: Maximum number of turns to play
        seed: Random seed for reproducibility
        verbose: Whether to print game progress
        compact: Use compact tile display
        interactive: Wait for user input between turns
    
    Returns:
        Dictionary containing game history
    """
    # Initialize game
    wall = create_wall(seed)
    
    # Set aside dora indicator (from dead wall)
    dora_indicator = wall.pop()
    dora_tile = calculate_dora(dora_indicator)
    
    # Deal initial hand
    hands = deal_initial_hand(wall, player_count=1)
    hand = hands[0]
    
    # Game state
    discards = []
    history = []
    
    if verbose:
        print("\n" + "=" * 70)
        print("🀄 一人麻雀 (Single-Player Mahjong) with AI 🀄")
        print("=" * 70)
        print(f"\n📍 Dora indicator: {TILE_DISPLAY.get(ID_37_TO_TILE[dora_indicator], '?')}")
        dora_str = ID_37_TO_TILE[dora_tile] if dora_tile else "?"
        print(f"   Dora: {TILE_DISPLAY.get(dora_str, '?')}")
        print(f"🎴 Tiles in wall: {len(wall)}")
        print("\n" + "-" * 70)
        print("Initial hand (13 tiles):")
        print(f"  {format_hand(hand, compact)}")
        print(f"  [{format_hand_simple(hand)}]")
        print("-" * 70 + "\n")
    
    # Game loop
    for turn in range(1, max_turns + 1):
        # Draw a tile
        drawn = draw_tile(wall)
        if drawn is None:
            if verbose:
                print("\n🌊 Wall is empty! Game over.")
            break
        
        hand.append(drawn)
        
        # Encode state
        state = encode_state_for_ai(
            hand, discards, dora_indicator,
            turn=turn, wall_remaining=len(wall)
        )
        
        # Get AI decision
        chosen_37, predictions = get_ai_discard(model, state, hand, device)
        
        # Remove chosen tile from hand
        hand.remove(chosen_37)
        discards.append(chosen_37)
        
        # Record history
        turn_record = {
            'turn': turn,
            'drawn': drawn,
            'discarded': chosen_37,
            'predictions': predictions,
            'hand_after': hand.copy(),
        }
        history.append(turn_record)
        
        if verbose:
            drawn_str = ID_37_TO_TILE[drawn]
            discard_str = ID_37_TO_TILE[chosen_37]
            
            print(f"Turn {turn}:")
            print(f"  📥 Drew: {TILE_DISPLAY.get(drawn_str, drawn_str)}")
            print(f"  🎯 AI discards: {TILE_DISPLAY.get(discard_str, discard_str)}")
            print(f"  📊 Top predictions:")
            for i, (tile_34, prob) in enumerate(predictions[:3]):
                tile_str = ID_34_TO_TILE[tile_34]
                marker = "✓" if tile_34 == id_37_to_34(chosen_37) else " "
                print(f"     {marker} {i+1}. {TILE_DISPLAY.get(tile_str, tile_str):<8} ({prob:.1%})")
            
            print(f"  🃏 Hand: [{format_hand_simple(hand)}]")
            print(f"  🎴 Wall: {len(wall)} tiles remaining")
            print()
            
            if interactive:
                try:
                    input("  [Press Enter to continue, Ctrl+C to quit]")
                    print()
                except KeyboardInterrupt:
                    print("\n\n🛑 Game interrupted by user.")
                    break
        
        # Sort hand for nicer display
        hand.sort(key=tile_sort_key)
    
    if verbose:
        print("-" * 70)
        print("Final hand:")
        print(f"  {format_hand(hand, compact)}")
        print(f"  [{format_hand_simple(hand)}]")
        print("\nDiscards:")
        print(f"  {format_hand(discards, compact)}")
        print(f"  [{format_hand_simple(discards)}]")
        print("=" * 70)
    
    return {
        'initial_hand': hands[0],
        'final_hand': hand,
        'discards': discards,
        'dora_indicator': dora_indicator,
        'dora': dora_tile,
        'history': history,
    }


# ==================== Main Entry Point ====================


def infer_model_type_from_path(model_path):
    """
    Infer model type from the model file path.
    
    Looks for architecture names (coatnet, resnet, vit) in the filename.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Inferred model type ('coatnet', 'resnet', 'vit') or None if not found
    """
    import os
    filename = os.path.basename(model_path).lower()
    
    # Check for architecture names in the filename
    if 'vit' in filename:
        return 'vit'
    elif 'resnet' in filename:
        return 'resnet'
    elif 'coatnet' in filename:
        return 'coatnet'
    
    return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Single-player mahjong with AI discard decisions (一人麻雀)'
    )
    
    parser.add_argument('--model-path', type=str, default='discard_model_coatnet.pth',
                       help='Path to trained model weights')
    parser.add_argument('--model-type', type=str, default=None,
                       choices=['coatnet', 'resnet', 'vit'],
                       help='Model architecture type (auto-detected from filename if not specified)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cuda/cpu)')
    parser.add_argument('--turns', type=int, default=18,
                       help='Maximum number of turns to play')
    parser.add_argument('--games', type=int, default=1,
                       help='Number of games to simulate')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--compact', action='store_true',
                       help='Use compact Unicode tile display')
    parser.add_argument('--quiet', action='store_true',
                       help='Only show summary (no turn-by-turn output)')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive mode: press Enter to advance each turn')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"🖥️  Device: {device}")
    
    # Determine model type: use explicit --model-type if provided, otherwise infer from filename
    if args.model_type is not None:
        model_type = args.model_type
    else:
        model_type = infer_model_type_from_path(args.model_path)
        if model_type is None:
            model_type = 'coatnet'  # Default fallback
            print(f"⚠️  Could not infer model type from filename, defaulting to '{model_type}'")
        else:
            print(f"🔍 Auto-detected model type: {model_type}")
    
    # Load model
    print(f"📂 Loading model from '{args.model_path}'...")
    
    try:
        if model_type == 'coatnet':
            model = create_coatnet_model(dropout=0.0)
        elif model_type == 'resnet':
            model = create_resnet_model(dropout=0.0)
        elif model_type == 'vit':
            model = create_vit_model(dropout=0.0)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
        
        # Validate that the state dict keys match the model architecture
        model_keys = set(model.state_dict().keys())
        loaded_keys = set(state_dict.keys())
        if model_keys != loaded_keys:
            missing = model_keys - loaded_keys
            extra = loaded_keys - model_keys
            error_msg = []
            if missing:
                error_msg.append(f"Missing keys: {list(missing)[:5]}...")
            if extra:
                error_msg.append(f"Extra keys: {list(extra)[:5]}...")
            raise RuntimeError(f"Model architecture mismatch. {' '.join(error_msg)}")
        
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully!\n")
        
    except FileNotFoundError:
        print(f"❌ Error: Model file '{args.model_path}' not found.")
        print("   Please train a model first using train.py, or specify a valid model path.")
        print("\n💡 Tip: You can run without a trained model to see a demo with random predictions.")
        
        # Offer to run in demo mode
        print("\n🎮 Running in DEMO mode with untrained model...")
        if model_type == 'coatnet':
            model = create_coatnet_model(dropout=0.0)
        elif model_type == 'resnet':
            model = create_resnet_model(dropout=0.0)
        else:
            model = create_vit_model(dropout=0.0)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Run games
    for game_num in range(args.games):
        if args.games > 1:
            print(f"\n{'#' * 70}")
            print(f"# Game {game_num + 1} / {args.games}")
            print(f"{'#' * 70}")
        
        # Use a robust seed derivation to avoid correlation between consecutive games
        seed = (args.seed * 10000 + game_num) if args.seed is not None else None
        result = run_single_player_game(
            model, device,
            max_turns=args.turns,
            seed=seed,
            verbose=not args.quiet,
            compact=args.compact,
            interactive=args.interactive
        )
        
        if args.quiet:
            print(f"Game {game_num + 1}: {len(result['history'])} turns played")


if __name__ == '__main__':
    main()
