# 連続ゲームフロー学習モデル (Sequential Game Flow Learning)

このディレクトリには、麻雀の一局全体の流れを学習するためのシーケンスモデルが含まれています。

## 📊 従来モデルとの違い

### 従来モデル (Original)
- **入力**: ある瞬間のゲーム状態 `(380, 4, 9)`
- **出力**: どの牌を切るか（34クラス分類）
- **アプローチ**: 各打牌判断を独立したサンプルとして学習

### シーケンスモデル (Sequential)
- **入力**: 一局を通じての状態シーケンス `(seq_len, 380, 4, 9)`
- **出力**: 各ステップでの打牌予測 `(seq_len, 34)`
- **アプローチ**: 一局の流れ（連続した打牌判断）をシーケンスとして学習

## 🏗️ アーキテクチャ

### LSTMモデル
```
入力シーケンス (batch, seq_len, 380, 4, 9)
    ↓
StateEncoder (各状態を圧縮)
    ↓
(batch, seq_len, hidden_dim)
    ↓
LSTM (num_layers層)
    ↓
(batch, seq_len, hidden_dim)
    ↓
FC層
    ↓
出力 (batch, seq_len, 34)
```

**特徴**:
- 時系列の依存関係を学習
- 隠れ状態により過去の情報を保持
- 双方向LSTMも使用可能
- 推論時は1ステップずつ予測可能

### Transformerモデル
```
入力シーケンス (batch, seq_len, 380, 4, 9)
    ↓
StateEncoder (各状態を圧縮)
    ↓
(batch, seq_len, d_model)
    ↓
Positional Encoding
    ↓
Transformer Decoder (num_layers層, causal mask)
    ↓
(batch, seq_len, d_model)
    ↓
FC層
    ↓
出力 (batch, seq_len, 34)
```

**特徴**:
- Self-Attentionで全てのステップ間の関係を学習
- Causal maskにより自己回帰的な予測
- 並列学習が可能（LSTMより高速）
- より長い依存関係を捉えやすい

## 🚀 使い方

### 1. トレーニング

#### LSTMモデルでトレーニング
```bash
cd sequential_models
python sequence_train.py --model lstm --data ../data2023.zip --epochs 20
```

#### Transformerモデルでトレーニング
```bash
python sequence_train.py --model transformer --data ../data2023.zip --epochs 30
```

#### 詳細な設定例
```bash
python sequence_train.py \
  --model lstm \
  --data ../data2023.zip \
  --max-files 3000 \
  --max-seq-len 25 \
  --epochs 30 \
  --batch-size 32 \
  --hidden-dim 256 \
  --num-layers 2 \
  --lr 1e-4 \
  --weight-decay 1e-2 \
  --dropout 0.1 \
  --scheduler cosine \
  --max-grad-norm 1.0 \
  --early-stopping 5 \
  --save-best \
  --output best_lstm_model.pth
```

### 2. パラメータ

#### データパラメータ
| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `--data` | `data2023.zip` | トレーニングデータのZIPファイル |
| `--max-files` | `2000` | 読み込む最大ファイル数 |
| `--max-seq-len` | `30` | 最大シーケンス長 |
| `--train-ratio` | `0.9` | トレーニングデータの割合 |

#### モデルパラメータ
| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `--model` | `lstm` | モデルタイプ (`lstm`, `transformer`) |
| `--hidden-dim` | `256` | 隠れ層の次元 |
| `--num-layers` | `2` | 層数 |
| `--nhead` | `8` | 注意ヘッド数 (Transformerのみ) |
| `--dropout` | `0.1` | ドロップアウト率 |
| `--bidirectional` | `False` | 双方向LSTM (LSTMのみ) |

#### トレーニングパラメータ
| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `--epochs` | `20` | エポック数 |
| `--batch-size` | `32` | バッチサイズ |
| `--lr` | `1e-4` | 学習率 |
| `--weight-decay` | `1e-2` | 重み減衰 |
| `--optimizer` | `adamw` | オプティマイザ |
| `--scheduler` | `cosine` | 学習率スケジューラ |
| `--max-grad-norm` | `1.0` | 勾配クリッピング閾値 |
| `--early-stopping` | `5` | 早期停止 patience |

## 📁 ファイル構成

```
sequential_models/
├── __init__.py           # パッケージ初期化
├── sequence_dataset.py   # シーケンスデータセット
├── sequence_models.py    # LSTM/Transformerモデル
├── sequence_train.py     # トレーニングスクリプト
└── README.md             # このファイル
```

## 🔄 データ構造

### シーケンスデータセット
```python
# 1つのシーケンス = 1人のプレイヤーの1局分の全打牌
{
    'kyoku_log': [...],      # 局のログデータ
    'player_id': 0,          # プレイヤーID (0-3)
    'actions': [             # (log_index, label) のリスト
        (5, 12),   # 5番目のログ位置で牌ID 12を打牌
        (8, 7),    # 8番目のログ位置で牌ID 7を打牌
        ...
    ]
}
```

### モデル入出力
```python
# 入力
states: (batch, seq_len, 380, 4, 9)  # ゲーム状態シーケンス
lengths: (batch,)                      # 実際のシーケンス長

# 出力
logits: (batch, seq_len, 34)          # 各ステップの予測
```

## 📈 期待される効果

1. **時系列パターンの学習**
   - 序盤/中盤/終盤で異なる戦略
   - 前のターンの選択が次の選択に影響

2. **文脈を考慮した判断**
   - 単なる「この状態でどれを切る？」ではなく
   - 「この流れでこの状態になった時、どれを切る？」

3. **一貫性のある打牌**
   - 一局を通じての戦略の一貫性
   - 河の作り方や待ち取りの一貫性

## 🔬 技術詳細

### StateEncoder
各ゲーム状態 `(380, 4, 9)` を固定長ベクトルに圧縮:
- 3層の畳み込み層 (128 → 256 → hidden_dim)
- BatchNorm + GELU活性化
- Global Average Pooling

### シーケンス処理
- **LSTM**: 順序を保った逐次処理、隠れ状態で文脈を保持
- **Transformer**: Self-Attentionで全ステップ間の関係を並列計算

### マスキング
- **パディングマスク**: 可変長シーケンスに対応
- **因果マスク** (Transformer): 未来の情報を見ないよう制限

## 💡 今後の改善案

1. **マルチタスク学習**
   - 打牌予測 + リーチ判断 + 副露判断

2. **Encoder-Decoder構造**
   - 他家の行動もエンコード
   - より豊かな文脈理解

3. **強化学習との統合**
   - 教師あり学習で事前学習
   - 強化学習でファインチューニング

4. **アンサンブル**
   - 従来モデルとシーケンスモデルの組み合わせ
