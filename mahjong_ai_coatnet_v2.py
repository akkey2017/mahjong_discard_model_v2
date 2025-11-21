import json
import re
import zipfile
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from einops import rearrange, repeat


# 刷新された特徴量エンコーダと、関連するヘルパー関数をインポート
from mahjong_ai_features import StateEncoderV2, _process_single_number, FEATURE_TILE_MAP

# --- データセット ---
class MahjongDataset(Dataset):
    def __init__(self, zip_path=None, max_files=10000, samples=None):
        if samples is not None:
            self.samples = samples
            return
        self.samples = []
        if zip_path is None: return
        with zipfile.ZipFile(zip_path) as zf:
            txts = [n for n in zf.namelist() if n.endswith('.txt')]
            for name in tqdm(txts[:max_files], desc="Loading game records"):
                try:
                    raw = zf.read(name).decode("utf-8")
                    game_data = json.loads(raw)
                    if 'log' not in game_data: continue
                    for kyoku_log in game_data['log']:
                        # qipaiがない局はスキップ
                        if not kyoku_log or 'qipai' not in kyoku_log[0]:
                            continue
                        for i, move in enumerate(kyoku_log):
                            if 'dapai' in move:
                                p_id = move['dapai']['l']
                                tile_str = move['dapai']['p'].replace('*','').replace('_','')
                                # 特徴量ファイルからインポートしたTILE_MAPと関数でラベルを作成
                                if tile_str in FEATURE_TILE_MAP:
                                    tile_id_37 = FEATURE_TILE_MAP[tile_str]
                                    label = _process_single_number(tile_id_37)
                                    self.samples.append((kyoku_log, i, p_id, 'discard', label))
                except Exception as e:
                    # print(f"Error processing {name}: {e}")
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        kyoku_log, log_idx, p_id, action_type, label = self.samples[idx]
        encoder = StateEncoderV2(kyoku_log, p_id)
        state_tensor = encoder.encode(log_idx)
        return state_tensor, label, action_type

# --- 4. モデルアーキテクチャ (CoAtNet) ---
# CoAtNetの構成要素 (変更なし)
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

# CoAtNet本体 (2D入力対応に更新)
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

    def forward(self, x): # x: (B, C, 4, 9)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3_conv(x)
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
        x = self.backbone(x)
        x = self.pool(x)
        x = self.flat(x)
        return self.fc(x)

# --- 評価指標 (変更なし) ---
class TopKAccuracy:
    def __init__(self, k=3): self.k, self.correct, self.total = k, 0, 0
    def update(self, preds, labels):
        _, top_k_preds = preds.topk(self.k, dim=1)
        self.correct += torch.any(top_k_preds == labels.view(-1, 1), dim=1).sum().item()
        self.total += labels.size(0)
    def compute(self): return self.correct / self.total if self.total > 0 else 0.0
    def reset(self): self.correct, self.total = 0, 0

# --- メイン処理 ---
if __name__ == '__main__':
    zip_path = "data2023.zip"
    full_dataset = MahjongDataset(zip_path, max_files=2000)
    print(f"Total samples found: {len(full_dataset)}")

    discard_samples = [s for s in full_dataset.samples if s[3] == 'discard']
    discard_dataset = MahjongDataset(samples=discard_samples)
    print(f"Discard samples: {len(discard_dataset)}")

    if len(discard_dataset) > 0:
        train_size = int(len(discard_dataset) * 0.9)
        val_size = len(discard_dataset) - train_size
        train_set, val_set = random_split(discard_dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device {device}")

        coatnet_params = {
            "in_channels": 380, "out_channels_list": [64, 96, 128],
            "num_blocks_list": [2, 2, 4], "expansion_factor": 4, "heads": 4
        }
        coatnet_backbone = CoAtNet(**coatnet_params).to(device)
        discard_model = DiscardModel(coatnet_backbone, final_channels=coatnet_params["out_channels_list"][-1]).to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(discard_model.parameters(), lr=1e-4, weight_decay=1e-2)
        top1_acc, top3_acc = TopKAccuracy(k=1), TopKAccuracy(k=3)

        epochs = 10
        for epoch in range(epochs):
            discard_model.train()
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            for xb, yb, _ in pbar:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = discard_model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

            discard_model.eval()
            top1_acc.reset(); top3_acc.reset()
            with torch.no_grad():
                pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
                for xb, yb, _ in pbar_val:
                    xb, yb = xb.to(device), yb.to(device)
                    out = discard_model(xb)
                    top1_acc.update(out, yb); top3_acc.update(out, yb)

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | "
                  f"Val Top-1 Acc: {top1_acc.compute():.4f} | Val Top-3 Acc: {top3_acc.compute():.4f}")

        torch.save(discard_model.state_dict(), "discard_model_coatnet_v2_2000.pth")
        print("\nSaved trained model to discard_model_coatnet_v2_2000.pth")

