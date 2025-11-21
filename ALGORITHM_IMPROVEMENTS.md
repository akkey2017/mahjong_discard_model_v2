# アルゴリズム改善と比較

## 概要

このドキュメントでは、リファクタリングにおけるアルゴリズム的な改善点と、新たに追加したモデルアーキテクチャの比較を説明します。

## 1. オリジナルCoAtNetモデルの改善点

### 1.1 Dropoutの追加

**問題点**: 過学習のリスク

**改善**: 
- Attention層にDropoutを追加
- TransformerBlockのFFN層にDropoutを追加
- DiscardModelの最終層前にDropoutを追加

```python
# Before
self.to_out = nn.Linear(dim, dim)

# After
self.to_out = nn.Sequential(
    nn.Linear(dim, dim),
    nn.Dropout(dropout)
)
```

**効果**: モデルの汎化性能が向上し、過学習を防ぐ

### 1.2 動的な空間次元への対応

**問題点**: ハードコードされた空間次元（h=4, w=9）

**改善**:
```python
# Before
x = rearrange(x, 'b (h w) c -> b c h w', h=4, w=9)

# After
b, c, h, w = x.shape
x = rearrange(x, 'b c h w -> b (h w) c')
x = self.stage3_transformer(x)
x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
```

**効果**: 異なる入力サイズに対応可能

### 1.3 学習の安定化

**追加機能**:
1. **勾配クリッピング**: 勾配爆発を防ぐ
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
   ```

2. **学習率スケジューリング**: 学習の収束を改善
   - Cosine Annealing: 学習率を周期的に変化
   - ReduceLROnPlateau: 検証精度が向上しない場合に学習率を削減

3. **早期停止**: 過学習を検出して学習を停止

## 2. 新しいアーキテクチャの追加

### 2.1 ResNetモデル

**設計思想**: 残差接続による深層学習

**アーキテクチャ**:
```
Input (B, 380, 4, 9)
    ↓
Stem: Conv2d + BatchNorm + GELU
    ↓
Layer1: 2× ResidualBlock (64 channels)
    ↓
Layer2: 2× ResidualBlock (96 channels)
    ↓
Layer3: 4× ResidualBlock (128 channels)
    ↓
Layer4: 2× ResidualBlock (160 channels)
    ↓
GlobalAvgPool → FC(160 → 34)
```

**特徴**:
- ✅ **学習の安定性**: スキップ接続により勾配消失を防ぐ
- ✅ **高速な推論**: 純粋な畳み込みベース
- ✅ **パラメータ効率**: CoAtNetより少ないパラメータ数

**適用場面**:
- リアルタイム推論が必要な場合
- 計算リソースが限られている場合
- 局所的なパターン認識が重要な場合

### 2.2 Vision Transformerモデル

**設計思想**: 純粋な注意機構ベースの学習

**アーキテクチャ**:
```
Input (B, 380, 4, 9)
    ↓
PatchEmbedding: Conv2d(patch_size=1×1)
    ↓
Add Positional Embedding
    ↓
6× TransformerBlock
    ↓
LayerNorm
    ↓
Reshape → GlobalAvgPool → FC(256 → 34)
```

**特徴**:
- ✅ **大域的な関係性**: 全ての位置間の関係を学習
- ✅ **柔軟な表現力**: 注意機構による適応的な特徴抽出
- ✅ **スケーラビリティ**: データ量に応じて性能が向上

**適用場面**:
- 大量のデータが利用可能な場合
- 複雑な大域的パターンが重要な場合
- 計算リソースが十分にある場合

### 2.3 CoAtNet（改良版）

**設計思想**: CNNとTransformerのハイブリッド

**アーキテクチャ**:
```
Input (B, 380, 4, 9)
    ↓
Stem: Conv2d + BatchNorm + GELU
    ↓
Stage1: 2× MBConv (64 channels)
    ↓
Stage2: 2× MBConv (96 channels)
    ↓
Stage3: Conv2d(96 → 128) + 4× TransformerBlock
    ↓
GlobalAvgPool → FC(128 → 34)
```

**特徴**:
- ✅ **バランスの取れた性能**: 局所と大域の両方を学習
- ✅ **効率的な計算**: 必要な箇所のみTransformerを使用
- ✅ **汎用性**: 様々なタスクに適応

**適用場面**:
- バランスの取れた性能が必要な場合
- 中程度のデータ量の場合
- デフォルトの選択肢として推奨

## 3. モデル比較表

| 項目 | CoAtNet | ResNet | Vision Transformer |
|------|---------|--------|-------------------|
| **パラメータ数** | 中 | 小 | 大 |
| **推論速度** | 中 | 速い | 遅い |
| **学習速度** | 中 | 速い | 遅い |
| **メモリ使用量** | 中 | 少ない | 多い |
| **局所特徴抽出** | ◎ | ◎ | △ |
| **大域的依存関係** | ◎ | △ | ◎ |
| **データ効率** | ◎ | ◎ | △ |
| **スケーラビリティ** | ○ | △ | ◎ |

## 4. トレーニング戦略の改善

### 4.1 学習率スケジューリング

**Cosine Annealing**:
```python
# 学習率がコサインカーブに従って減少
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
```

**利点**:
- 初期は高速に学習
- 後期は細かい調整
- 局所最適解からの脱出

**ReduceLROnPlateau**:
```python
# 検証精度が向上しない場合に学習率を削減
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
```

**利点**:
- 適応的な学習率調整
- 過学習の防止
- 収束の改善

### 4.2 正則化技術

**勾配クリッピング**:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm=1.0)
```

**効果**: 
- 勾配爆発の防止
- 学習の安定化

**Dropout**:
```python
# 0.1 ~ 0.2 が一般的
model = create_coatnet_model(dropout=0.1)
```

**効果**:
- 過学習の防止
- アンサンブル効果

### 4.3 早期停止

```python
early_stopping = EarlyStopping(patience=5, mode='max')
if early_stopping(val_accuracy):
    break  # 学習を停止
```

**効果**:
- 計算時間の削減
- 過学習の防止
- 最適なエポック数の自動決定

## 5. 推奨されるハイパーパラメータ

### CoAtNet
```bash
python train.py \
  --model coatnet \
  --epochs 30 \
  --batch-size 64 \
  --lr 1e-4 \
  --weight-decay 1e-2 \
  --dropout 0.1 \
  --scheduler cosine \
  --max-grad-norm 1.0 \
  --early-stopping 5
```

### ResNet
```bash
python train.py \
  --model resnet \
  --epochs 25 \
  --batch-size 128 \
  --lr 2e-4 \
  --weight-decay 1e-2 \
  --dropout 0.1 \
  --scheduler cosine \
  --max-grad-norm 1.0 \
  --early-stopping 5
```

### Vision Transformer
```bash
python train.py \
  --model vit \
  --epochs 50 \
  --batch-size 32 \
  --lr 5e-5 \
  --weight-decay 1e-2 \
  --dropout 0.2 \
  --scheduler cosine \
  --max-grad-norm 1.0 \
  --early-stopping 7
```

## 6. 実験結果の期待値

### データ量による性能予測

**少量データ（< 1000ゲーム）**:
1. ResNet（推奨）
2. CoAtNet
3. Vision Transformer

**中量データ（1000-5000ゲーム）**:
1. CoAtNet（推奨）
2. ResNet
3. Vision Transformer

**大量データ（> 5000ゲーム）**:
1. Vision Transformer（推奨）
2. CoAtNet
3. ResNet

### 計算リソースによる選択

**CPU推論**:
1. ResNet（推奨）- 最速
2. CoAtNet - やや遅い
3. Vision Transformer - 遅い

**GPU推論**:
1. CoAtNet（推奨）- バランス良い
2. ResNet - 最速
3. Vision Transformer - やや遅い

## 7. さらなる改善の可能性

### 7.1 アンサンブル学習
```python
# 複数モデルの予測を組み合わせ
coatnet_pred = coatnet_model(x)
resnet_pred = resnet_model(x)
vit_pred = vit_model(x)
ensemble_pred = (coatnet_pred + resnet_pred + vit_pred) / 3
```

### 7.2 知識蒸留
```python
# 大きなモデル（Teacher）から小さなモデル（Student）へ知識を転移
# Vision Transformer → ResNet
```

### 7.3 マルチタスク学習
```python
# 打牌予測 + リーチ判断 + 和了確率予測を同時に学習
```

### 7.4 自己注意の可視化
```python
# どの牌に注目しているかを可視化
# デバッグと解釈性の向上
```

## 8. まとめ

このリファクタリングにより:

1. ✅ **コードの品質向上**: モジュール化、エラーハンドリング、ドキュメント
2. ✅ **学習の改善**: スケジューリング、正則化、早期停止
3. ✅ **柔軟性の向上**: 複数のモデル、設定可能なパラメータ
4. ✅ **新しい選択肢**: ResNet、Vision Transformerの追加

各モデルには長所と短所があり、タスクやデータに応じて適切なものを選択することが重要です。
