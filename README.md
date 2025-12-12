# 麻雀AI打牌モデル (Mahjong Discard Model v2)

深層学習を用いた麻雀の打牌予測モデル。CoAtNet、ResNet、Vision Transformerなど複数のアーキテクチャをサポート。

## 📁 プロジェクト構成

```
mahjong_discard_model_v2/
├── models.py                    # モデルアーキテクチャ定義
├── dataset.py                   # データセット処理
├── utils.py                     # トレーニング・評価ユーティリティ
├── train.py                     # メイントレーニングスクリプト
├── evaluate_model.py            # モデル評価スクリプト
├── single_player.py             # 一人麻雀シミュレーション（AIプレイ観察）
├── mahjong_ai_features.py       # 特徴量エンコーダ
├── mahjong_ai_coatnet_v2.py    # 旧トレーニングスクリプト（リファクタリング済み）
├── sequential_models/           # 🆕 シーケンスモデル（一局の流れを学習）
│   ├── sequence_models.py       # LSTM/Transformerモデル
│   ├── sequence_dataset.py      # シーケンスデータセット
│   ├── sequence_train.py        # シーケンスモデル用トレーニング
│   └── README.md                # 詳細ドキュメント
├── advanced_training/           # 🧪 大規模モデル + 複数ZIP対応の実験用ワークスペース
│   ├── train_large.py           # マルチZIP対応の大型モデル学習CLI
│   ├── large_models.py          # パラメータ数を増やしたCNN/Transformer/CoAtNet
│   └── multizip_dataset.py      # 複数ZIPの統合データセット
└── README.md                    # このファイル
```

## 🎯 特徴

### サポートされるモデルアーキテクチャ

#### 従来モデル（1ステップ予測）
1. **CoAtNet** (デフォルト)
   - 畳み込みとTransformerを組み合わせたハイブリッドモデル
   - 局所的特徴と大域的依存関係の両方を効果的に学習

2. **ResNet**
   - 残差接続を用いた深層畳み込みネットワーク
   - 安定した学習と高速な推論

3. **Vision Transformer (ViT)**
   - 純粋なTransformerベースのモデル
   - パッチベースの処理で大域的な関係性を捉える

#### 🆕 シーケンスモデル（一局の流れを学習）
4. **LSTM Sequential Model**
   - LSTMを用いた時系列モデル
   - 一局を通じての打牌シーケンスを学習
   - 過去のターンの文脈を保持

5. **Transformer Sequential Model**
   - Transformer Decoderを用いた自己回帰モデル
   - Self-Attentionで全ステップ間の関係を学習
   - 長い依存関係を効果的に捉える

### 機能強化

- ✅ **学習率スケジューリング**: Cosine Annealing、ReduceLROnPlateauをサポート
- ✅ **早期停止**: 過学習を防ぐための早期停止機能
- ✅ **モデルチェックポイント**: 最良モデルの自動保存
- ✅ **勾配クリッピング**: 学習の安定性向上
- ✅ **Dropout**: 過学習防止のための正則化
- ✅ **コマンドライン引数**: 柔軟な設定変更
- ✅ **エラーハンドリング**: 堅牢なエラー処理
- ✅ **一人麻雀シミュレーション**: AIの打牌判断をリアルタイムで観察
- ✅ 🆕 **シーケンスモデル**: 一局の流れを学習するLSTM/Transformerモデル

## 🚀 使い方

### 1. トレーニング

#### 基本的な使い方（CoAtNet）

```bash
python train.py --data data2023.zip --max-files 2000 --epochs 10
```

#### ResNetモデルでトレーニング

```bash
python train.py --model resnet --data data2023.zip --epochs 15 --lr 2e-4
```

#### Vision Transformerモデルでトレーニング

```bash
python train.py --model vit --data data2023.zip --epochs 20 --dropout 0.2
```

#### 高度な設定例

```bash
python train.py \
  --model coatnet \
  --data data2023.zip \
  --max-files 5000 \
  --epochs 30 \
  --batch-size 128 \
  --lr 1e-4 \
  --weight-decay 1e-2 \
  --dropout 0.1 \
  --optimizer adamw \
  --scheduler cosine \
  --max-grad-norm 1.0 \
  --early-stopping 5 \
  --save-best \
  --output best_model.pth
```

#### 🧪 複数ZIP + 大型モデルでトレーニング

```bash
python advanced_training/train_large.py \
  --data data2023.zip data2022.zip \
  --model coatnet_large \
  --epochs 15 \
  --save-best
```

### 2. 評価

#### 基本的な評価

```bash
python evaluate_model.py --model-path discard_model_coatnet.pth --data data2022.zip
```

#### デモ付き評価

```bash
python evaluate_model.py \
  --model-path discard_model_coatnet.pth \
  --model-type coatnet \
  --data data2022.zip \
  --show-demo \
  --num-demo-samples 10
```

### 3. 一人麻雀シミュレーション（AIプレイ観察）

一人麻雀モードでは、AIが実際にツモって打牌を選択する様子を観察できます。

#### 基本的な使い方

```bash
python single_player.py --model-path discard_model_coatnet.pth
```

#### インタラクティブモード（ターンごとに一時停止）

```bash
python single_player.py --model-path discard_model_coatnet.pth --interactive
```

#### 複数ゲームをシミュレート

```bash
python single_player.py \
  --model-path discard_model_coatnet.pth \
  --games 5 \
  --turns 18 \
  --seed 42
```

#### パラメータ一覧

- `--model-path`: 学習済みモデルファイルパス
- `--model-type`: モデルアーキテクチャ（`coatnet`, `resnet`, `vit`）
- `--turns`: 最大ターン数（デフォルト: 18）
- `--games`: シミュレートするゲーム数（デフォルト: 1）
- `--seed`: 乱数シード（再現性のため）
- `--compact`: Unicode麻雀牌絵文字を使用
- `--interactive`: インタラクティブモード（Enterキーで次のターンへ）
- `--quiet`: サマリーのみ表示

### 4. 旧スクリプト（互換性のため）

```bash
python mahjong_ai_coatnet_v2.py
```

### 5. 🆕 シーケンスモデルのトレーニング

一局の流れを学習するシーケンスモデルを使用する場合：

#### LSTMモデル
```bash
cd sequential_models
python sequence_train.py --model lstm --data ../data2023.zip --epochs 20
```

#### Transformerモデル
```bash
cd sequential_models
python sequence_train.py --model transformer --data ../data2023.zip --epochs 30
```

詳細は [sequential_models/README.md](sequential_models/README.md) を参照してください。

## 📊 トレーニングパラメータ

### データパラメータ
- `--data`: トレーニングデータのZIPファイルパス
- `--max-files`: 読み込む最大ファイル数
- `--train-ratio`: トレーニングデータの割合（デフォルト: 0.9）

### モデルパラメータ
- `--model`: モデルアーキテクチャ（`coatnet`, `resnet`, `vit`）
- `--dropout`: ドロップアウト率（デフォルト: 0.1）

### トレーニングパラメータ
- `--epochs`: エポック数
- `--batch-size`: バッチサイズ
- `--lr`: 学習率
- `--weight-decay`: 重み減衰
- `--optimizer`: オプティマイザ（`adam`, `adamw`, `sgd`）
- `--scheduler`: 学習率スケジューラ（`cosine`, `plateau`, `none`）
- `--max-grad-norm`: 勾配クリッピングの閾値

### 正則化
- `--early-stopping`: 早期停止のpatience（0で無効）

### その他
- `--num-workers`: データローディングワーカー数
- `--device`: 使用デバイス（`auto`, `cuda`, `cpu`）
- `--seed`: 乱数シード
- `--output`: 出力モデルパス
- `--save-best`: 最良モデルのみ保存

## 🔧 リファクタリング内容

### コード品質の改善

1. **モジュール化**
   - モデル定義を`models.py`に分離
   - データセット処理を`dataset.py`に分離
   - ユーティリティを`utils.py`に分離

2. **コードの重複排除**
   - `evaluate_model.py`と`mahjong_ai_coatnet_v2.py`間のモデル定義の重複を解消
   - 共通ロジックを関数として抽出

3. **エラーハンドリング**
   - データ読み込み時の例外処理を改善
   - ファイルが見つからない場合の適切なエラーメッセージ

4. **設定の柔軟性**
   - ハードコードされた値をコマンドライン引数化
   - 設定可能なハイパーパラメータの追加

5. **ドキュメント**
   - 各モジュール、クラス、関数にdocstringを追加
   - README作成

### アルゴリズム改善

1. **学習率スケジューリング**
   - Cosine Annealingスケジューラ
   - ReduceLROnPlateauスケジューラ

2. **正則化技術**
   - Dropoutのサポート
   - 勾配クリッピング

3. **トレーニング効率**
   - 早期停止機能
   - モデルチェックポイント

4. **新しいアーキテクチャ**
   - ResNetモデルの追加
   - Vision Transformerモデルの追加

## 🏗️ モデルアーキテクチャ詳細

### CoAtNet
- **Stage 1-2**: MBConv（Mobile Inverted Bottleneck Convolution）
- **Stage 3**: Transformer blocks
- **特徴**: 畳み込みの局所的特徴抽出とTransformerの大域的依存関係学習を融合

### ResNet
- **4つのレイヤー**: 各レイヤーに複数の残差ブロック
- **特徴**: スキップ接続により勾配消失問題を解決

### Vision Transformer
- **パッチ埋め込み**: 入力をパッチに分割
- **Transformerエンコーダ**: 自己注意機構による特徴抽出
- **特徴**: 純粋な注意機構ベースのアーキテクチャ

## 📈 パフォーマンス指標

トレーニングと評価では以下の指標を使用：

- **損失（Loss）**: CrossEntropyLoss
- **Top-1精度**: モデルの第1予測が正解である割合
- **Top-3精度**: モデルの上位3予測に正解が含まれる割合
- **Top-5精度**: モデルの上位5予測に正解が含まれる割合

## 🔍 今後の改善案

1. **データ拡張**
   - ランダムな盤面回転
   - ノイズ付加

2. **アンサンブル学習**
   - 複数モデルの予測を組み合わせ

3. **より高度な特徴量**
   - 待ち牌の情報
   - シャンテン数
   - 期待値計算

4. **転移学習**
   - 事前学習済みモデルの活用

5. **ハイパーパラメータチューニング**
   - グリッドサーチやBayesian最適化

## 📝 ライセンス

このプロジェクトのライセンスについては、リポジトリのオーナーにお問い合わせください。

## 🤝 貢献

バグ報告や機能リクエストは、GitHubのIssuesでお願いします。
