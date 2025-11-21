import json
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split, Dataset
from einops import rearrange

# トレーニング時に使用したクラスと関数を再定義またはインポートします
# (モデルの構造が完全に一致している必要があるため)
from mahjong_ai_features import StateEncoderV2, _process_single_number, FEATURE_TILE_MAP

# --- 1. モデルアーキテクチャの再定義 ---
# mahjong_ai_coatnet_v2.py からモデル定義をそのままコピーします
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super().__init__()
        self.stride = stride
        hidden_dim = in_channels * expansion_factor
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        layers = []
        if expansion_factor != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU()
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        self.conv = nn.Sequential(*layers)
    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class CoAtNet(nn.Module):
    def __init__(self, in_channels, out_channels_list, num_blocks_list, expansion_factor=4, heads=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_list[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels_list[0]),
            nn.GELU()
        )
        self.stage1 = self._make_stage(MBConv, num_blocks_list[0], out_channels_list[0], out_channels_list[0], 1, expansion_factor)
        self.stage2 = self._make_stage(MBConv, num_blocks_list[1], out_channels_list[0], out_channels_list[1], 1, expansion_factor)
        self.stage3_conv = nn.Conv2d(out_channels_list[1], out_channels_list[2], 1)
        self.stage3_transformer = nn.Sequential(*[TransformerBlock(dim=out_channels_list[2], heads=heads) for _ in range(num_blocks_list[2])])
    def _make_stage(self, block, num_blocks, in_ch, out_ch, stride, expansion):
        layers = [block(in_ch, out_ch, stride, expansion)]
        layers.extend([block(out_ch, out_ch, 1, expansion) for _ in range(num_blocks - 1)])
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.stem(x); x = self.stage1(x); x = self.stage2(x); x = self.stage3_conv(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.stage3_transformer(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=4, w=9)
        return x

class DiscardModel(nn.Module):
    def __init__(self, backbone, final_channels):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(final_channels, 34)
    def forward(self, x):
        x = self.backbone(x); x = self.pool(x); x = self.flat(x)
        return self.fc(x)

# --- 2. 評価用データセットと評価指標の定義 ---
class MahjongDataset(Dataset):
    def __init__(self, zip_path=None, max_files=10000, samples=None):
        if samples is not None: self.samples = samples; return
        self.samples = []
        if zip_path is None: return
        with zipfile.ZipFile(zip_path) as zf:
            txts = [n for n in zf.namelist() if n.endswith('.txt')]
            for name in tqdm(txts[:max_files], desc="Loading game records for evaluation"):
                try:
                    raw = zf.read(name).decode("utf-8")
                    game_data = json.loads(raw)
                    if 'log' not in game_data: continue
                    for kyoku_log in game_data['log']:
                        if not kyoku_log or 'qipai' not in kyoku_log[0]: continue
                        for i, move in enumerate(kyoku_log):
                            if 'dapai' in move:
                                p_id = move['dapai']['l']
                                tile_str = move['dapai']['p'].replace('*','').replace('_','')
                                if tile_str in FEATURE_TILE_MAP:
                                    label = _process_single_number(FEATURE_TILE_MAP[tile_str])
                                    self.samples.append((kyoku_log, i, p_id, 'discard', label))
                except Exception: continue
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        kyoku_log, log_idx, p_id, action_type, label = self.samples[idx]
        encoder = StateEncoderV2(kyoku_log, p_id)
        state_tensor = encoder.encode(log_idx)
        return state_tensor, label, action_type

class TopKAccuracy:
    def __init__(self, k=3): self.k, self.correct, self.total = k, 0, 0
    def update(self, preds, labels):
        _, top_k_preds = preds.topk(self.k, dim=1)
        self.correct += torch.any(top_k_preds == labels.view(-1, 1), dim=1).sum().item()
        self.total += labels.size(0)
    def compute(self): return self.correct / self.total if self.total > 0 else 0.0
    def reset(self): self.correct, self.total = 0, 0

# --- 3. 評価実行とデモ ---
if __name__ == '__main__':
    # --- 設定 ---
    MODEL_PATH = "discard_model_coatnet_v2.pth"  # 学習済みモデルのパス
    DATA_ZIP_PATH = "data2022.zip"               # 評価用データのパス
    NUM_FILES_TO_EVAL = 200                      # 評価に使用するファイル数
    BATCH_SIZE = 64

    # --- 準備 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"デバイスを準備しています: {device}")

    # モデルの骨格を定義
    coatnet_params = {
        "in_channels": 380, "out_channels_list": [64, 96, 128],
        "num_blocks_list": [2, 2, 4], "expansion_factor": 4, "heads": 4
    }
    coatnet_backbone = CoAtNet(**coatnet_params)
    model = DiscardModel(coatnet_backbone, final_channels=coatnet_params["out_channels_list"][-1])

    # 学習済みの重みをロード
    try:
        # 警告を解消するため weights_only=True を追加
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        print(f"モデルの重みを '{MODEL_PATH}' から正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: モデルファイル '{MODEL_PATH}' が見つかりません。パスを確認してください。")
        exit()
    except Exception as e:
        print(f"モデルの読み込み中にエラーが発生しました: {e}")
        exit()
        
    model.to(device)
    model.eval() # 評価モードに設定

    # 評価用データローダーを作成
    full_dataset = MahjongDataset(DATA_ZIP_PATH, max_files=NUM_FILES_TO_EVAL)
    discard_dataset = MahjongDataset(samples=[s for s in full_dataset.samples if s[3] == 'discard'])
    
    # 訓練時と同じ分割を行う（再現性のため）
    generator = torch.Generator().manual_seed(42)
    train_size = int(len(discard_dataset) * 0.9)
    val_size = len(discard_dataset) - train_size
    _, val_set = random_split(discard_dataset, [train_size, val_size], generator=generator)
    
    eval_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    print(f"評価データセットを準備しました。サンプル数: {len(val_set)}")

    # 損失関数と評価指標
    loss_fn = nn.CrossEntropyLoss()
    top1_acc = TopKAccuracy(k=1)
    top3_acc = TopKAccuracy(k=3)
    total_loss = 0

    # --- 評価ループ ---
    with torch.no_grad():
        for xb, yb, _ in tqdm(eval_loader, desc="Evaluating"):
            xb, yb = xb.to(device), yb.to(device)
            
            # 予測
            out = model(xb)
            
            # 損失と正解率を計算
            loss = loss_fn(out, yb)
            total_loss += loss.item()
            top1_acc.update(out, yb)
            top3_acc.update(out, yb)

    # --- 結果表示 ---
    print("\n--- 評価結果 ---")
    print(f"平均損失: {total_loss / len(eval_loader):.4f}")
    print(f"Top-1 正解率: {top1_acc.compute():.4f} ({top1_acc.correct}/{top1_acc.total})")
    print(f"Top-3 正解率: {top3_acc.compute():.4f} ({top3_acc.correct}/{top3_acc.total})")
    print("--------------------")

    # --- 推論デモ ---
    if len(val_set) > 0:
        print("\n--- 推論デモ ---")
        # 評価セットからランダムに1つデータを取得
        sample_idx = torch.randint(0, len(val_set), (1,)).item()
        xb_sample, yb_sample, _ = val_set[sample_idx]
        
        # 34次元IDから牌文字列への逆引きマップ
        ID_TO_TILE_34 = {v: k for k, v in {
            **{f"m{i}": i-1 for i in range(1,10)}, **{f"p{i}": i-1+9 for i in range(1,10)},
            **{f"s{i}": i-1+18 for i in range(1,10)}, **{f"z{i}": i-1+27 for i in range(1,8)}
        }.items()}

        with torch.no_grad():
            # バッチ次元を追加してモデルに入力
            out_sample = model(xb_sample.unsqueeze(0).to(device))
            
            # 確率に変換し、上位5件を取得
            probabilities = F.softmax(out_sample, dim=1)
            top5_probs, top5_indices = torch.topk(probabilities, 5)

            print(f"サンプル {sample_idx} の局面で推論を実行します。")
            actual_discard = ID_TO_TILE_34.get(yb_sample, "不明")
            print(f"正解の打牌: {actual_discard}")
            
            print("\nモデルの予測 (Top 5):")
            for i in range(top5_indices.size(1)):
                pred_tile = ID_TO_TILE_34.get(top5_indices[0, i].item(), "不明")
                prob = top5_probs[0, i].item()
                print(f"  {i+1}. {pred_tile:<4} (予測確率: {prob:.2%})")
        print("--------------------")

