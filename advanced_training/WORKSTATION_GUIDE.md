# Workstation Guide: `vit_multitask_large`

このガイドは、Threadripper 9960X / 192GiB RAM / RTX PRO 6000 Blackwell
Workstation Edition のような単一GPUワークステーションで
`advanced_training/train_large.py --model vit_multitask_large` を回すための
手順メモです。

## 1. まず確認すること

```bash
nvidia-smi
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY
```

- PyTorch 2 系なら `--compile` を試す価値がある。
- Blackwell / Ada / Ampere 世代では、まず `--amp --amp-dtype bf16 --tf32`
  を基本設定にする。
- データ ZIP は NVMe SSD 上に置く。ネットワークドライブや遅いHDDでは
  GPU が DataLoader 待ちになりやすい。

## 2. プロファイルから始める

最初から長時間学習せず、20バッチだけ実行して速度とピークVRAMを見る。

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

ログに出る `samples/sec` と `peak allocated VRAM` を見て調整する。

## 3. 調整の目安

| 状況 | 調整 |
|---|---|
| VRAM に大きく余裕がある | `--batch-size` を 224 → 256 → 320 と上げる |
| OOM する | `--batch-size` を下げる、または `--accumulation-steps` を上げて実効バッチを維持 |
| GPU 使用率が低い | `--num-workers` を 12 → 16 → 20、`--prefetch-factor` を 4 → 6 に上げる |
| CPU/RAM 使用量が高すぎる | `--num-workers` または `--prefetch-factor` を下げる |
| loss が不安定 | `--lr` を `5e-5` に下げる、`--max-grad-norm 1.0` を維持 |
| `fulou` 精度を重視 | `--fulou-weight 0.6` などで副露 head の重みを上げる |

## 4. 本番学習例

プロファイルで問題なければ `--profile-batches` を外して本番実行する。

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
    --run-name vit_mt_blackwell_bf16
```

## 5. `torch.compile` の使い方

PyTorch 2 系で動く場合のみ、上のコマンドに追加する。

```bash
--compile --compile-mode reduce-overhead
```

注意:

- 初回 epoch は compile により遅くなることがある。
- 環境によっては `max-autotune` が速いが、compile 時間とメモリ使用量が増える。
- 問題が出た場合は `--compile` を外して BF16 / TF32 / DataLoader 調整だけで回す。

## 6. 監視指標とマルチタスク重み

デフォルトの best checkpoint 監視指標は、マルチタスクでは `top1_acc`。
特定 head を重視したい場合は、例えば以下のようにする。

```bash
--monitor-metric fulou_acc --fulou-weight 0.6
```

利用可能な検証メトリクスは `metrics.csv` の `val_` 接頭辞付きカラムで確認する。
存在しない metric を指定した場合、学習ログに warning が出て best checkpoint と
early stopping はその epoch では更新されない。

## 7. Resume

途中再開は run directory か `last_model.pth` を指定する。

```bash
python advanced_training/train_large.py \
    --data data2023.zip data2022.zip \
    --resume runs/vit_mt_blackwell_bf16
```

resume 時は保存済み config の `model` が優先される。データ引数は現在の CLI から
渡す必要がある。
