# Advanced Training (Multi-ZIP + Large Models)

Qiita記事の高パラメータ化を意識した実験用ワークスペースです。複数の学習用ZIPをまとめて読み込み、より大きなモデル（CNN/Transformer/CoAtNet）を学習できます。

## 使い方

```bash
python advanced_training/train_large.py \
  --data data2023.zip data2022.zip \
  --model coatnet_large \
  --epochs 15 \
  --batch-size 96 \
  --lr 8e-5 \
  --save-best
```

### 主な引数
- `--data`: 学習に使うZIPを複数指定可能（空白区切り）
- `--max-files-per-zip`: 各ZIPから読む最大ファイル数
- `--model`: `coatnet_large` / `resnet_large` / `vit_large`
- その他: `--scheduler`, `--early-stopping`, `--max-grad-norm`, `--output` などは従来と同様

## モデル
- **CoAtNet Large**: チャネル幅・ブロック数を増やし、ヘッド数も拡大
- **ResNet Large**: より深い層と広いチャネル幅
- **ViT Large**: 512次元埋め込み・8層構成のTransformer

## データ
`advanced_training/multizip_dataset.py` で複数ZIPをまとめてロードし、元の特徴量エンコーダをそのまま利用します。集計はサンプル数とZIPごとの内訳を表示します。

## 評価 (evaluate_large.py)

学習済みの大規模モデル（CoAtNet Large, ResNet Large, ViT Large）を評価するためのスクリプトです。

### 使い方

```bash
python advanced_training/evaluate_large.py \
  --model-path large_discard_model_coatnet_large.pth \
  --data data2022.zip \
  --batch-size 64 \
  --show-demo
```

### 主な引数
- `--model-path`: 学習済みモデルのパス
- `--model-type`: モデルタイプ (`coatnet_large` / `resnet_large` / `vit_large`)。指定しない場合はファイル名から自動検出
- `--data`: 評価に使うZIPを複数指定可能（空白区切り）
- `--max-files-per-zip`: 各ZIPから読む最大ファイル数（デフォルト: 200）
- `--batch-size`: バッチサイズ（デフォルト: 64）
- `--train-ratio`: 学習用データの割合（残りが検証用、デフォルト: 0.9）
- `--show-demo`: 推論デモを表示
- `--num-demo-samples`: デモに表示するサンプル数（デフォルト: 5）

### 出力
- Average Loss
- Top-1 / Top-3 / Top-5 Accuracy
- 推論デモ（`--show-demo` 指定時）
