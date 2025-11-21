import numpy as np
import re
import torch

# --- 定数と設定 (特徴量エンジニアリング用) ---

# 37次元の牌表現（m0, p0, s0 を赤ドラとして扱う）
# m0, m1-9, p0, p1-9, s0, s1-9, z1-7
FEATURE_TILE_MAP = {
    **{f"m{i}": i for i in range(1, 10)},
    "m0": 0,
    **{f"p{i}": i + 10 for i in range(1, 10)},
    "p0": 10,
    **{f"s{i}": i + 20 for i in range(1, 10)},
    "s0": 20,
    **{f"z{i}": i + 29 for i in range(1, 8)},
}

# 逆引きマップ
FEATURE_ID_TO_TILE = {v: k for k, v in FEATURE_TILE_MAP.items()}

# --- utils.py由来のヘルパー関数 ---

def _make_pai_counter_list_from(shoupai_str: str) -> list:
    """牌姿文字列を37次元のリストに変換"""
    pai_list = [0] * 37
    current_suit = ''
    for pai in shoupai_str:
        if pai in 'mpsz':
            current_suit = pai
        else:
            key = current_suit + pai
            if key in FEATURE_TILE_MAP:
                pai_list[FEATURE_TILE_MAP[key]] += 1
    return pai_list

def _fulou_to_pais(fulou_str: str) -> list:
    """副露文字列を37次元IDのリストに変換"""
    head = fulou_str[0]
    pais = [head + item for item in fulou_str if item not in '+-=mpsz']
    pais_tiles = []
    for pai in pais:
        if pai in FEATURE_TILE_MAP:
            pais_tiles.append(FEATURE_TILE_MAP[pai])
    return pais_tiles

def _make_dora_list_from(dora_indicators: list) -> list:
    """ドラ表示牌(文字列)のリストを、ドラ牌(37次元ID)のリストに変換"""
    dora_tiles = []
    for indicator_str in dora_indicators:
        if indicator_str not in FEATURE_TILE_MAP:
            continue
        dora_display = FEATURE_TILE_MAP[indicator_str]
        
        dora_num = -1
        # 数牌
        if dora_display < 30:
            suit_base = (dora_display // 10) * 10
            num = dora_display % 10
            if num == 0: # 赤5 -> 6
                dora_num = suit_base + 6
            elif num < 9: # 1-8 -> 2-9
                dora_num = dora_display + 1
            elif num == 9: # 9 -> 1
                dora_num = suit_base + 1
        # 字牌
        elif dora_display < 34: # 東南西北
            dora_num = 30 if dora_display == 33 else dora_display + 1
        else: # 白發中
            dora_num = 34 if dora_display == 36 else dora_display + 1
            
        if dora_num != -1:
            dora_tiles.append(dora_num)
            
    return dora_tiles

def _process_single_number(tile_37: int) -> int:
    """37次元IDを34次元IDに変換 (赤5はただの5に)"""
    if tile_37 < 10: # m
        return tile_37 -1 if tile_37 != 0 else 4
    if tile_37 < 20: # p
        return tile_37 -11 + 9 if tile_37 != 10 else 13
    if tile_37 < 30: # s
        return tile_37 -21 + 18 if tile_37 != 20 else 22
    return tile_37 - 30 + 27 # z

class StateEncoderV2:
    """
    `make_data.ipynb`のロジックに基づき、ゲーム状態を (380, 4, 9) のテンソルに変換する。
    次元: (チャンネル, 牌の種類, 牌の数)
    """

    def __init__(self, kyoku_log, player_id):
        self.kyoku_log = kyoku_log
        self.player_id = player_id
        self.qipai = kyoku_log[0]['qipai']
        self.num_channels = 380

    def _get_player_offset(self, p_id):
        return (p_id - self.player_id + 4) % 4

    def encode(self, log_index_in_kyoku):
        # --- 1. 指定された局面までの状態を再現 ---
        
        # 配牌
        hands = [_make_pai_counter_list_from(h) for h in self.qipai['shoupai']]
        
        dora_indicators = [FEATURE_TILE_MAP.get(self.qipai['baopai'])]
        rivers = [[], [], [], []]
        melds = [[], [], [], []] # 公開された副露
        ankan = [[], [], [], []] # 暗槓
        reach_status = [0] * 4

        for i in range(1, log_index_in_kyoku + 1):
            move = self.kyoku_log[i]
            
            if 'zimo' in move or 'gangzimo' in move:
                key = 'zimo' if 'zimo' in move else 'gangzimo'
                p_id = move[key]['l']
                tile_str = move[key]['p']
                if tile_str in FEATURE_TILE_MAP:
                    hands[p_id][FEATURE_TILE_MAP[tile_str]] += 1
            
            elif 'dapai' in move:
                p_id = move['dapai']['l']
                tile_str = move['dapai']['p']
                tile_id = FEATURE_TILE_MAP.get(tile_str.replace('*','').replace('_',''))
                if tile_id is not None:
                    hands[p_id][tile_id] -= 1
                    rivers[p_id].append(tile_id)
                    if '*' in tile_str:
                        reach_status[p_id] = 1

            elif 'fulou' in move:
                p_id = move['fulou']['l']
                meld_str = move['fulou']['m']
                meld_tiles = _fulou_to_pais(meld_str)
                melds[p_id].append(meld_tiles)
                
                # 手牌から消費された牌を減算
                # 誰から鳴いたか特定
                from_p_id = (p_id + 3) % 4 if '-' in meld_str else (p_id + 2) % 4 if '=' in meld_str else (p_id + 1) % 4
                taken_tile = rivers[from_p_id].pop()
                
                consumed_tiles = meld_tiles.copy()
                consumed_tiles.remove(taken_tile)
                for tile in consumed_tiles:
                    hands[p_id][tile] -=1

            elif 'gang' in move:
                p_id = move['gang']['l']
                meld_str = move['gang']['m']
                meld_tiles = _fulou_to_pais(meld_str)
                
                if any(c in meld_str for c in '-+='): # 加槓
                    kakan_tile = meld_tiles[0]
                    hands[p_id][kakan_tile] -= 1
                    # 既存のポンをカンに更新
                    for m in melds[p_id]:
                        if len(m) == 3 and m[0] == kakan_tile:
                            m.append(kakan_tile)
                            break
                else: # 暗槓
                    ankan[p_id].append(meld_tiles)
                    for tile in meld_tiles:
                        hands[p_id][tile] -= 1
            
            elif 'kaigang' in move:
                dora_indicators.append(FEATURE_TILE_MAP.get(move['kaigang']['baopai']))

        # --- 2. 特徴量テンソルの作成 ---
        final_tensor = np.zeros((self.num_channels, 4, 9), dtype=np.float32)
        ch_offset = 0

        # 各プレイヤーの相対位置を計算
        player_indices = [self._get_player_offset(i) for i in range(4)]
        
        # A. 手牌 (7ch * 4人 = 28ch)
        for p_idx in player_indices:
            hand_37 = hands[p_idx]
            hand_red = [hand_37[0], hand_37[10], hand_37[20]]
            hand_34 = self._convert_to_34_dim(hand_37)
            self._encode_tiles(final_tensor, ch_offset, hand_34, hand_red, is_red_channel=True)
            ch_offset += 7
        
        # B. 副露 (16ch * 4人 = 64ch)
        for p_idx in player_indices:
            player_melds = melds[p_idx]
            self._encode_melds(final_tensor, ch_offset, player_melds)
            ch_offset += 16
        
        # C. 暗槓 (4ch * 4人 = 16ch)
        for p_idx in player_indices:
            player_ankans = ankan[p_idx]
            # 暗槓は副露と同じ構造だがチャンネル数が違う
            self._encode_melds(final_tensor, ch_offset, player_ankans, is_ankan=True)
            ch_offset += 4

        # ... (以降の特徴量も同様にエンコード) ...
        # NOTE: ipynbの全ての特徴量を実装すると非常に長大になるため、主要なものに絞っています。
        #       必要に応じてここに追加してください。
        
        # D. 河 (捨て牌) (7ch * 4人 = 28ch)
        for p_idx in player_indices:
            river_37 = [0] * 37
            for tile in rivers[p_idx]: river_37[tile] += 1
            river_red = [river_37[0], river_37[10], river_37[20]]
            river_34 = self._convert_to_34_dim(river_37)
            self._encode_tiles(final_tensor, ch_offset, river_34, river_red, is_red_channel=True)
            ch_offset += 7
        
        # E. リーチ (1ch * 4人 = 4ch)
        for p_idx in player_indices:
            if reach_status[p_idx] == 1:
                final_tensor[ch_offset, :, :] = 1.0
            ch_offset += 1

        # F. ドラ (4ch)
        dora_tiles = _make_dora_list_from([FEATURE_ID_TO_TILE.get(d) for d in dora_indicators if d is not None])
        dora_34 = [0] * 34
        for tile in dora_tiles:
             dora_34[_process_single_number(tile)] += 1
        self._encode_tiles(final_tensor, ch_offset, dora_34, [0,0,0], is_red_channel=False)
        ch_offset += 4
        
        # G. 局情報 (場風3ch + 局数4ch + 本場1ch + 供託1ch = 9ch)
        # 場風
        final_tensor[ch_offset + self.qipai['zhuangfeng'], :, :] = 1.0
        ch_offset += 3
        # 局数
        final_tensor[ch_offset + self.qipai['jushu'], :, :] = 1.0
        ch_offset += 4
        # 本場
        final_tensor[ch_offset, :, :] = self.qipai['changbang'] / 5.0 # 正規化
        ch_offset += 1
        # 供託
        final_tensor[ch_offset, :, :] = self.qipai['lizhibang'] / 4.0 # 正規化
        ch_offset += 1

        # 残りのチャンネルは0埋め
        
        return torch.from_numpy(final_tensor)

    def _convert_to_34_dim(self, hand_37):
        hand_34 = [0] * 34
        for i, count in enumerate(hand_37):
            if count > 0:
                hand_34[_process_single_number(i)] += count
        return hand_34

    def _encode_tiles(self, tensor, ch_offset, tiles_34, reds, is_red_channel=True):
        """34次元の牌リストを (4,9) の形にエンコード"""
        max_ch = 7 if is_red_channel else 4
        # m, p, s
        for suit in range(3):
            for num in range(9):
                count = tiles_34[suit * 9 + num]
                for i in range(min(count, 4)):
                    tensor[ch_offset + i, suit, num] = 1.0
        # z
        for i in range(7):
            count = tiles_34[27 + i]
            for c in range(min(count, 4)):
                tensor[ch_offset + c, 3, i] = 1.0
        # red
        if is_red_channel:
            for i in range(3):
                if reds[i] > 0:
                    tensor[ch_offset + 4 + i, i, 4] = 1.0

    def _encode_melds(self, tensor, ch_offset, melds, is_ankan=False):
        """副露リストをエンコード"""
        max_melds = 4
        ch_per_meld = 4 if not is_ankan else 1

        for i, meld_37 in enumerate(melds[:max_melds]):
            meld_34 = [0] * 34
            for tile in meld_37:
                meld_34[_process_single_number(tile)] += 1
            
            # 副露牌を4chで表現
            self._encode_tiles(tensor, ch_offset + i * ch_per_meld, meld_34, [0,0,0], is_red_channel=False)
