# Model Information

This document provides detailed information about the model architectures available in this repository, including parameter counts.

## Summary Table

| Model | Factory Function | Parameters | Parameters (M) |
|-------|------------------|------------|----------------|
| CoAtNet (Standard) | `create_coatnet_model` | 1,224,738 | 1.22M |
| ResNet (Standard) | `create_resnet_model` | 2,738,658 | 2.74M |
| Vision Transformer (Standard) | `create_vit_model` | 4,854,562 | 4.85M |
| CoAtNet (Large) | `create_large_coatnet` | 7,026,850 | 7.03M |
| ResNet (Large) | `create_large_resnet` | 16,324,770 | 16.32M |
| Vision Transformer (Large) | `create_large_vit` | 25,451,042 | 25.45M |

## Vision Transformer Comparison

以下は Vision Transformer の標準版と大型版の比較表です。

| 項目 | 標準 ViT (`create_vit_model`) | 大型 ViT (`create_large_vit`) |
| --- | --- | --- |
| 入力チャンネル (`in_channels`) | 380 | 380 |
| クラス数 (`num_classes`) | 34 | 34 |
| Embedding 次元 (`embed_dim`) | 256 | 512 |
| 深さ (`depth`) | 6 | 8 |
| Attention ヘッド数 (`heads`) | 8 | 8 |
| パッチサイズ (`patch_size`) | (1, 1) | (1, 1) |
| ドロップアウト既定値 | 0.0 | 0.1 |
| 出力チャネル（分類直前, `final_channels`） | 256 | 512 |
| ファクトリ関数 | `create_vit_model` | `create_large_vit` |
| 学習スクリプトでのモデル指定 | `--model vit` (`train.py`) | `--model vit_large` (`advanced_training/train_large.py`) |
| **パラメータ数** | **4,854,562 (4.85M)** | **25,451,042 (25.45M)** |
| パラメータ比率 | 1.0x (基準) | 5.24x |

### 補足

- 標準版は主に単一ZIP用の `train.py` で利用（デフォルトドロップアウト0.0）
- 大型版は複数ZIP＋大規模学習用の `advanced_training/train_large.py` で利用（デフォルトドロップアウト0.1、バッチやエポックも大きめ）
- 大型版は標準版の約5.24倍のパラメータ数を持つ

## Model Configurations

### Standard Models

#### CoAtNet (Standard)
- **Factory**: `create_coatnet_model`
- **Script**: `train.py --model coatnet`
- **Configuration**:
  - in_channels: 380
  - num_classes: 34
  - out_channels_list: [64, 96, 128]
  - num_blocks_list: [2, 2, 4]
  - expansion_factor: 4
  - heads: 4
  - dropout: 0.0 (default)
  - final_channels: 128
- **Parameters**: 1,224,738 (1.22M)

#### ResNet (Standard)
- **Factory**: `create_resnet_model`
- **Script**: `train.py --model resnet`
- **Configuration**:
  - in_channels: 380
  - num_classes: 34
  - num_blocks_list: [2, 2, 4, 2]
  - channels_list: [64, 96, 128, 160]
  - dropout: 0.0 (default)
  - final_channels: 160
- **Parameters**: 2,738,658 (2.74M)

#### Vision Transformer (Standard)
- **Factory**: `create_vit_model`
- **Script**: `train.py --model vit`
- **Configuration**:
  - in_channels: 380
  - num_classes: 34
  - embed_dim: 256
  - depth: 6
  - heads: 8
  - patch_size: (1, 1)
  - dropout: 0.0 (default)
  - final_channels: 256
- **Parameters**: 4,854,562 (4.85M)

### Large Models

#### CoAtNet (Large)
- **Factory**: `create_large_coatnet`
- **Script**: `advanced_training/train_large.py --model coatnet_large`
- **Configuration**:
  - in_channels: 380
  - num_classes: 34
  - out_channels_list: [128, 192, 256]
  - num_blocks_list: [3, 3, 6]
  - expansion_factor: 6
  - heads: 8
  - dropout: 0.1 (default)
  - final_channels: 256
- **Parameters**: 7,026,850 (7.03M)

#### ResNet (Large)
- **Factory**: `create_large_resnet`
- **Script**: `advanced_training/train_large.py --model resnet_large`
- **Configuration**:
  - in_channels: 380
  - num_classes: 34
  - num_blocks_list: [3, 4, 6, 3]
  - channels_list: [128, 192, 256, 320]
  - dropout: 0.1 (default)
  - final_channels: 320
- **Parameters**: 16,324,770 (16.32M)

#### Vision Transformer (Large)
- **Factory**: `create_large_vit`
- **Script**: `advanced_training/train_large.py --model vit_large`
- **Configuration**:
  - in_channels: 380
  - num_classes: 34
  - embed_dim: 512
  - depth: 8
  - heads: 8
  - patch_size: (1, 1)
  - dropout: 0.1 (default)
  - final_channels: 512
- **Parameters**: 25,451,042 (25.45M)

## Generating This Information

To regenerate or verify this information, run:

```bash
python model_info.py
```

This script will output detailed information about all models including:
- Parameter counts (total and trainable)
- Architecture configurations
- Comparison tables in markdown format
