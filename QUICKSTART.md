# クイックスタートガイド (Quick Start Guide)

## 必要な環境 (Requirements)

```bash
pip install torch torchvision torchaudio
pip install einops tqdm numpy
```

## 基本的な使い方 (Basic Usage)

### 1. データの準備 (Prepare Data)

ゲームログのZIPファイル（例: `data2023.zip`）をプロジェクトルートに配置してください。

### 2. モデルのトレーニング (Train a Model)

#### CoAtNetモデル（推奨）

```bash
python train.py --data data2023.zip --max-files 2000 --epochs 10
```

実行後、`discard_model_coatnet.pth` が生成されます。

#### ResNetモデル（高速）

```bash
python train.py --model resnet --data data2023.zip --epochs 15
```

#### Vision Transformerモデル（大量データ向け）

```bash
python train.py --model vit --data data2023.zip --max-files 5000 --epochs 30
```

### 3. モデルの評価 (Evaluate a Model)

```bash
python evaluate_model.py \
  --model-path discard_model_coatnet.pth \
  --model-type coatnet \
  --data data2022.zip \
  --show-demo
```

## 代表的なコマンド例 (Common Commands)

### 小規模データセットで試す

```bash
# 500ゲーム、5エポックで素早くテスト
python train.py --max-files 500 --epochs 5 --batch-size 32
```

### 本格的なトレーニング

```bash
# 5000ゲーム、30エポック、早期停止あり
python train.py \
  --max-files 5000 \
  --epochs 30 \
  --early-stopping 5 \
  --save-best \
  --output best_model.pth
```

### 高精度モデルのトレーニング

```bash
# ドロップアウト、学習率スケジューリング、勾配クリッピング使用
python train.py \
  --model coatnet \
  --max-files 10000 \
  --epochs 50 \
  --batch-size 64 \
  --lr 1e-4 \
  --dropout 0.15 \
  --scheduler cosine \
  --max-grad-norm 1.0 \
  --early-stopping 7 \
  --save-best
```

### 複数モデルの比較

```bash
# CoAtNet
python train.py --model coatnet --output model_coatnet.pth --epochs 20

# ResNet
python train.py --model resnet --output model_resnet.pth --epochs 20

# Vision Transformer
python train.py --model vit --output model_vit.pth --epochs 20

# それぞれ評価
python evaluate_model.py --model-path model_coatnet.pth --model-type coatnet --show-demo
python evaluate_model.py --model-path model_resnet.pth --model-type resnet --show-demo
python evaluate_model.py --model-path model_vit.pth --model-type vit --show-demo
```

## トラブルシューティング (Troubleshooting)

### メモリ不足エラー

```bash
# バッチサイズを減らす
python train.py --batch-size 32

# またはワーカー数を減らす
python train.py --num-workers 0
```

### CUDA利用可能だがCPUで実行したい

```bash
python train.py --device cpu
```

### データが見つからない

```bash
# データパスを明示的に指定
python train.py --data /path/to/data2023.zip
```

### 学習が収束しない

```bash
# 学習率を下げる
python train.py --lr 5e-5

# または重み減衰を調整
python train.py --weight-decay 1e-3
```

## パラメータチューニングのヒント (Tuning Tips)

### データ量に応じた推奨設定

**少量（< 1000ゲーム）**
```bash
python train.py --model resnet --epochs 15 --lr 2e-4 --dropout 0.1
```

**中量（1000-5000ゲーム）**
```bash
python train.py --model coatnet --epochs 25 --lr 1e-4 --dropout 0.1
```

**大量（> 5000ゲーム）**
```bash
python train.py --model vit --epochs 40 --lr 5e-5 --dropout 0.2
```

### 過学習の兆候がある場合

```bash
# ドロップアウトを増やす
python train.py --dropout 0.2

# 早期停止のpatienceを減らす
python train.py --early-stopping 3

# 重み減衰を増やす
python train.py --weight-decay 2e-2
```

### 学習が遅い場合

```bash
# バッチサイズを増やす（GPU使用時）
python train.py --batch-size 128

# ワーカー数を増やす
python train.py --num-workers 4
```

## よくある質問 (FAQ)

**Q: どのモデルを使うべきですか？**

A: 一般的にはCoAtNetがバランスが良く推奨です。高速推論が必要ならResNet、大量データがあるならVision Transformerを検討してください。

**Q: 学習にどのくらい時間がかかりますか？**

A: データ量とハードウェアに依存しますが、2000ゲームで10エポックの場合：
- GPU使用時: 15-30分
- CPU使用時: 2-4時間

**Q: 旧スクリプトとの互換性はありますか？**

A: はい。`mahjong_ai_coatnet_v2.py` は引き続き使用可能です。新機能が必要な場合は `train.py` を使用してください。

**Q: モデルのパラメータ数は？**

A: 
- CoAtNet: 約200K-300K
- ResNet: 約150K-250K
- Vision Transformer: 約400K-600K

**Q: カスタムハイパーパラメータで実験したい**

A: すべてのパラメータはコマンドライン引数で設定可能です。`python train.py --help` で全オプションを確認できます。

## 次のステップ (Next Steps)

1. **README.md** で詳細なドキュメントを確認
2. **ALGORITHM_IMPROVEMENTS.md** でアルゴリズムの詳細を学習
3. 異なるハイパーパラメータで実験
4. 複数モデルのアンサンブルを試す

## サポート (Support)

問題や質問がある場合は、GitHubのIssuesで報告してください。
