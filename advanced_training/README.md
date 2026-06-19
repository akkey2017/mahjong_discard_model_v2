# Advanced Training (Multi-ZIP + Large Models)

大規模モデル（CoAtNet Large / ResNet Large / ViT Large）、およびマルチタスク版（打牌+リーチ+副露+和了）を学習するためのワークスペース。複数ZIPを一度に読み込み、実験ごとにディレクトリを切って管理する。

---

## ディレクトリ構成

学習を始めると `runs/<model>_<timestamp>/` が自動で作られ、以下が保存される:

```
runs/vit_large_20260417_143022/
├── config.json        # 起動時のCLI引数一式
├── metrics.csv        # エポック毎の train/val loss, top-k 精度, 学習率
├── training.log       # stdoutと同じ内容のログ
├── best_model.pth     # monitorメトリックで最良のチェックポイント
├── last_model.pth     # 直近のチェックポイント (--resumeで再開可能)
└── summary.json       # 実行後の要約
```

チェックポイントは `{"model_state", "model_type", "config", "extra"}` のリッチな dict で保存される。評価スクリプトはチェックポイントから自動でアーキテクチャを復元する（ファイル名推定は不要）。

---

## 学習

### 基本的な使い方（打牌モデル、ViT Large）

```bash
python advanced_training/train_large.py \
    --data data2023.zip data2022.zip \
    --model vit_large \
    --epochs 20 \
    --batch-size 128 \
    --lr 3e-4 \
    --amp \
    --label-smoothing 0.05 \
    --warmup-epochs 3
```

### マルチタスク（打牌 + リーチ + チー/ポン/カン + 和了）

```bash
python advanced_training/train_large.py \
    --data data2023.zip data2022.zip \
    --model vit_multitask_large \
    --epochs 20 \
    --batch-size 128 \
    --fulou-negatives \
    --amp
```

`--fulou-negatives` で「鳴けたのに鳴かなかった」負例を自動生成する。

### Threadripper + RTX PRO 6000 Blackwell ワークステーション例

24コアCPU・大容量RAM・Blackwell GPU のような学習用ワークステーションでは、
DataLoader を強め、BF16 AMP / TF32 / EMA を有効にした設定から始める。
まず `--profile-batches` で数バッチだけ実測し、VRAM 使用量を見て
`--batch-size` と `--accumulation-steps` を上げ下げする。
詳細な調整手順は [`WORKSTATION_GUIDE.md`](WORKSTATION_GUIDE.md) を参照。

```bash
python advanced_training/train_large.py \
    --data data2023.zip data2022.zip \
    --model vit_multitask_large \
    --epochs 30 \
    --batch-size 192 \
    --accumulation-steps 2 \
    --lr 8e-5 \
    --weight-decay 1e-2 \
    --scheduler warmup_cosine \
    --warmup-epochs 3 \
    --amp \
    --amp-dtype bf16 \
    --tf32 \
    --ema-decay 0.999 \
    --fulou-negatives \
    --split-by-game \
    --num-workers 12 \
    --prefetch-factor 4 \
    --persistent-workers \
    --profile-batches 20
```

プロファイルで余裕があれば `--profile-batches` を外して本番実行する。
PyTorch 2 系で安定する環境では `--compile --compile-mode reduce-overhead`
も追加候補。

### 主な引数

**データ**
- `--data`: 学習に使うZIPを複数指定可能
- `--max-files-per-zip`: 各ZIPから読む最大ファイル数
- `--split-by-game`: 対局単位で train/val を分割（デフォルトはサンプル単位のランダム分割、リーク注意）
- `--fulou-negatives`: マルチタスク時に副露パスの負例を生成

**モデル**
- `--model`: `coatnet_large` / `resnet_large` / `vit_large` / `coatnet_multitask_large` / `resnet_multitask_large` / `vit_multitask_large`
- `--dropout`: ドロップアウト率

**学習**
- `--epochs`, `--batch-size`, `--lr`, `--weight-decay`, `--optimizer`
- `--scheduler`: `warmup_cosine` (デフォルト) / `cosine` / `plateau` / `none`
- `--warmup-epochs`: warmup_cosine のウォームアップエポック数
- `--label-smoothing`: ラベルスムージング (打牌ヘッド)
- `--max-grad-norm`: 勾配クリップ閾値
- `--accumulation-steps`: 勾配累積ステップ数（実効バッチサイズ拡大）
- `--amp`: mixed precision (AMP) 有効化
- `--amp-dtype`: AMP の dtype (`bf16` / `fp16` / `auto`)
- `--ema-decay`: EMAの減衰率（0で無効）
- `--monitor-metric`: best checkpoint / early stopping に使う検証指標
- `--dapai-weight`, `--riichi-weight`, `--fulou-weight`, `--gang-weight`, `--hule-weight`: マルチタスク loss の重み

**システム最適化**
- `--prefetch-factor`, `--persistent-workers`, `--drop-last`: DataLoader のスループット調整
- `--tf32`, `--cudnn-benchmark`: NVIDIA GPU 向け高速化
- `--compile`, `--compile-mode`: `torch.compile` の有効化
- `--profile-batches`: 指定バッチ数だけ学習して samples/sec とピークVRAMをログ出力

**実験管理**
- `--run-dir`: 実験ルート（デフォルト `runs`）
- `--run-name`: 実行ディレクトリ名の上書き
- `--resume`: 既存ランから再開（run dir か `last_model.pth` を指定）

---

## 評価

```bash
python advanced_training/evaluate_large.py \
    --model-path runs/vit_large_20260417_143022/best_model.pth \
    --data data2022.zip \
    --output-csv per_tile.csv \
    --show-demo
```

チェックポイントから自動でアーキテクチャを復元。以下を出力する:
- Loss, Top-1/3/5 (またはマルチタスクならタスク別 accuracy/loss)
- **Per-tile accuracy** (34 牌ごとの正答率 + 数牌/字牌別の集計)
- **Confusion summary** (よく間違える正解→予測ペア上位)
- `--show-demo` 指定時は推論デモ

---

## 対局エージェント (MahjongAgent)

```python
from advanced_training.mahjong_agent import MahjongAgent

agent = MahjongAgent("runs/vit_multitask_large_.../best_model.pth")

# 自分のツモ後の行動
decision = agent.on_zimo(kyoku_log, log_index=..., player_id=0,
                         my_hand_str="m178p36s15578z356")

# 他家の打牌に対する鳴き判断
decision = agent.on_opponent_dapai(kyoku_log, log_index=..., my_player_id=0,
                                   my_hand_str="m178p36s15578z356")
```

牌譜フォーマットと同じ形式で判断結果を返すので、`single_player.py` やオンライン対局サーバと連携可能。

---

## モデル

| モデル | バックボーン | 主な特徴 |
|---|---|---|
| `coatnet_large` | CoAtNet | チャネル [128,192,256]、ブロック [3,3,6]、ヘッド 8 |
| `resnet_large` | ResNet | チャネル [128,192,256,320]、ブロック [3,4,6,3] |
| `vit_large` | ViT | embed 512、深さ 8、CLSトークン・trunc_normal初期化・Stochastic Depth・Flash Attention (SDPA) |
| `coatnet_multitask_large` 等 | 同上 | 共有バックボーン + 5ヘッド（`dapai` / `riichi` / `fulou` / `gang` / `hule`）。`fulou` は鳴き（`chi` / `pon` など）を扱う multi-class head、`gang` も multi-class head |
