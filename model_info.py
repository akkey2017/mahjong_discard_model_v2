#!/usr/bin/env python3
"""
Model Information Utility

This script outputs detailed information about the available model architectures,
including parameter counts, layer configurations, and architecture summaries.
"""

import sys
from pathlib import Path

# Ensure repository root is importable
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch


def count_parameters(model):
    """Count the total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_num(n):
    """Format a number with commas for readability."""
    return f"{n:,}"


def print_model_info(name, model, config):
    """Print detailed information about a model."""
    total, trainable = count_parameters(model)
    
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"\nParameter Count:")
    print(f"  Total:     {format_num(total)} ({total / 1e6:.2f}M)")
    print(f"  Trainable: {format_num(trainable)} ({trainable / 1e6:.2f}M)")
    
    return total, trainable


def get_all_model_info():
    """Get information about all available models."""
    from models import create_coatnet_model, create_resnet_model, create_vit_model
    from advanced_training.large_models import (
        create_large_coatnet,
        create_large_resnet,
        create_large_vit,
    )
    
    models_info = []
    
    # Standard models
    models_info.append({
        "name": "CoAtNet (Standard)",
        "factory": "create_coatnet_model",
        "script": "train.py --model coatnet",
        "model": create_coatnet_model(),
        "config": {
            "in_channels": 380,
            "num_classes": 34,
            "out_channels_list": "[64, 96, 128]",
            "num_blocks_list": "[2, 2, 4]",
            "expansion_factor": 4,
            "heads": 4,
            "dropout": "0.0 (default)",
            "final_channels": 128,
        }
    })
    
    models_info.append({
        "name": "ResNet (Standard)",
        "factory": "create_resnet_model",
        "script": "train.py --model resnet",
        "model": create_resnet_model(),
        "config": {
            "in_channels": 380,
            "num_classes": 34,
            "num_blocks_list": "[2, 2, 4, 2]",
            "channels_list": "[64, 96, 128, 160]",
            "dropout": "0.0 (default)",
            "final_channels": 160,
        }
    })
    
    models_info.append({
        "name": "Vision Transformer (Standard)",
        "factory": "create_vit_model",
        "script": "train.py --model vit",
        "model": create_vit_model(),
        "config": {
            "in_channels": 380,
            "num_classes": 34,
            "embed_dim": 256,
            "depth": 6,
            "heads": 8,
            "patch_size": "(1, 1)",
            "dropout": "0.0 (default)",
            "final_channels": 256,
        }
    })
    
    # Large models
    models_info.append({
        "name": "CoAtNet (Large)",
        "factory": "create_large_coatnet",
        "script": "advanced_training/train_large.py --model coatnet_large",
        "model": create_large_coatnet(),
        "config": {
            "in_channels": 380,
            "num_classes": 34,
            "out_channels_list": "[128, 192, 256]",
            "num_blocks_list": "[3, 3, 6]",
            "expansion_factor": 6,
            "heads": 8,
            "dropout": "0.1 (default)",
            "final_channels": 256,
        }
    })
    
    models_info.append({
        "name": "ResNet (Large)",
        "factory": "create_large_resnet",
        "script": "advanced_training/train_large.py --model resnet_large",
        "model": create_large_resnet(),
        "config": {
            "in_channels": 380,
            "num_classes": 34,
            "num_blocks_list": "[3, 4, 6, 3]",
            "channels_list": "[128, 192, 256, 320]",
            "dropout": "0.1 (default)",
            "final_channels": 320,
        }
    })
    
    models_info.append({
        "name": "Vision Transformer (Large)",
        "factory": "create_large_vit",
        "script": "advanced_training/train_large.py --model vit_large",
        "model": create_large_vit(),
        "config": {
            "in_channels": 380,
            "num_classes": 34,
            "embed_dim": 512,
            "depth": 8,
            "heads": 8,
            "patch_size": "(1, 1)",
            "dropout": "0.1 (default)",
            "final_channels": 512,
        }
    })
    
    return models_info


def generate_markdown_table(models_info):
    """Generate a markdown table comparing all models."""
    print("\n" + "="*60)
    print("Markdown Table (for MODEL_INFO.md)")
    print("="*60 + "\n")
    
    # Header
    print("| Model | Factory Function | Parameters | Parameters (M) |")
    print("|-------|------------------|------------|----------------|")
    
    # Rows
    for info in models_info:
        total, _ = count_parameters(info["model"])
        print(f"| {info['name']} | `{info['factory']}` | {format_num(total)} | {total / 1e6:.2f}M |")


def generate_vit_comparison_table(models_info):
    """Generate a comparison table specifically for ViT models."""
    vit_standard = None
    vit_large = None
    
    for info in models_info:
        if info["factory"] == "create_vit_model":
            vit_standard = info
        elif info["factory"] == "create_large_vit":
            vit_large = info
    
    if not vit_standard or not vit_large:
        return
    
    std_total, _ = count_parameters(vit_standard["model"])
    large_total, _ = count_parameters(vit_large["model"])
    
    print("\n" + "="*60)
    print("Vision Transformer Comparison Table")
    print("="*60 + "\n")
    
    print("| 項目 | 標準 ViT (`create_vit_model`) | 大型 ViT (`create_large_vit`) |")
    print("| --- | --- | --- |")
    print("| 入力チャンネル (`in_channels`) | 380 | 380 |")
    print("| クラス数 (`num_classes`) | 34 | 34 |")
    print("| Embedding 次元 (`embed_dim`) | 256 | 512 |")
    print("| 深さ (`depth`) | 6 | 8 |")
    print("| Attention ヘッド数 (`heads`) | 8 | 8 |")
    print("| パッチサイズ (`patch_size`) | (1, 1) | (1, 1) |")
    print("| ドロップアウト既定値 | 0.0 | 0.1 |")
    print("| 出力チャネル（分類直前, `final_channels`） | 256 | 512 |")
    print("| ファクトリ関数 | `create_vit_model` | `create_large_vit` |")
    print("| 学習スクリプトでのモデル指定 | `--model vit` (`train.py`) | `--model vit_large` (`advanced_training/train_large.py`) |")
    print(f"| **パラメータ数** | **{format_num(std_total)} ({std_total / 1e6:.2f}M)** | **{format_num(large_total)} ({large_total / 1e6:.2f}M)** |")
    print(f"| パラメータ比率 | 1.0x (基準) | {large_total / std_total:.2f}x |")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Mahjong Discard Model - Model Information")
    print("=" * 60)
    
    models_info = get_all_model_info()
    
    # Print detailed info for each model
    for info in models_info:
        print_model_info(info["name"], info["model"], info["config"])
    
    # Generate comparison tables
    generate_markdown_table(models_info)
    generate_vit_comparison_table(models_info)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
