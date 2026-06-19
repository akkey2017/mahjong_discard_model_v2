# Workstation Guide: `vit_multitask_large`

このガイドは、Threadripper 9960X / 192GiB RAM / RTX PRO 6000 Blackwell
Workstation Edition のような単一GPUワークステーションで
`advanced_training/train_large.py --model vit_multitask_large` を回すための
手順メモです。

この構成はモデル学習には十分強力ですが、最初から DataLoader の worker 数と
先読み量を大きくすると、初回バッチの生成中に停止したように見えることがあります。
まず安全な設定で一巡させ、その後に1項目ずつ増やしてください。

## 1. 実行環境を確認する

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
    print("compiled architectures", torch.cuda.get_arch_list())
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    print("bf16 matmul", (x @ x).shape)
PY
```

- CUDA が利用でき、BF16 の行列積まで完了することを確認する。
- Blackwell では、インストール済み PyTorch / CUDA がGPUを正式に扱えることを
  確認する。GPU名が表示されるだけでは十分ではない。
- 基本設定は `--amp --amp-dtype bf16 --tf32` とする。
- データ ZIP はローカル NVMe SSD 上に置く。ネットワークドライブやHDDでは
  GPU が DataLoader 待ちになりやすい。
- TRX50 / Threadripper 9960X は4チャネルメモリ対応。3枚構成でも動作するが、
  CPU側の特徴量生成を重視する場合は、同一仕様のDIMMを4枚そろえて全チャネルを
  使用する構成が望ましい。

## 2. 最初は安全なスモークテストを行う

最初から20バッチを実行せず、2バッチだけでデータ読み込みとGPU学習が通ることを
確認します。`--profile-batches` はバッチを逐次処理し、DataLoader待ちを含む進捗、
初回バッチ時間、データ待ち時間、GPU転送・計算時間を個別に表示します。

```bash
python advanced_training/train_large.py \
    --data data2023.zip data2022.zip \
    --model vit_multitask_large \
    --epochs 30 \
    --batch-size 256 \
    --accumulation-steps 1 \
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
    --num-workers 4 \
    --prefetch-factor 2 \
    --persistent-workers \
    --profile-batches 2
```

最初の確認では `--compile` を付けません。ログに `Profile: batches=2` と
`peak allocated VRAM` が出れば、データ読み込み・forward・backwardは通っています。

## 3. 停止したように見える場合の切り分け

### `Task weights: ...` が最後の表示になっている

`--profile-batches` を外すか、DataLoader を単一プロセスにして確認します。

```bash
--num-workers 0 --no-persistent-workers
```

これで `Epoch 1/...` と `Training` が進む場合、モデルやGPUではなく
DataLoader の worker 起動・先読み・特徴量生成がボトルネックです。

### GPU使用率が0%に近い

- CPU使用率が高い: 特徴量生成待ち。worker 数を段階的に増やす。
- RAMまたはSwap使用量が増え続ける: worker 数と `prefetch-factor` を下げる。
- CPU使用率も低い: worker停止、ストレージ待ち、プロセス異常を確認する。
- `--compile` 使用時だけ発生する: `--compile` を外して再確認する。

監視例:

```bash
watch -n 1 nvidia-smi
htop
free -h
```

## 4. 段階的に調整する

一度に複数の設定を変えず、同じ `--profile-batches` 数で比較してください。

1. `--num-workers 4` で基準値を取る。
2. worker 数を `4 → 8 → 12` と増やす。
3. 最も速い worker 数のまま、batch sizeを `256 → 384 → 512` と増やす。
4. VRAM不足時のみ batch sizeを戻すか、`--accumulation-steps` を増やす。
5. 最後に `--compile --compile-mode reduce-overhead` を単独で比較する。

| 状況 | 調整 |
|---|---|
| VRAMに十分な余裕がある | `--batch-size` を 256 → 384 → 512 と上げる |
| OOMする | batch sizeを下げる。必要なら勾配累積で実効バッチを維持する |
| GPU使用率が低くCPUに余裕がある | `--num-workers` を 4 → 8 → 12 と上げる |
| CPU/RAM/Swap負荷が高い | worker 数を下げ、`--prefetch-factor 2` を維持する |
| 初回バッチだけ極端に遅い | worker起動、先読み、compile時間を個別に確認する |
| lossが不安定 | `--lr 5e-5` を試し、`--max-grad-norm 1.0` を維持する |
| `fulou` 精度を重視 | `--fulou-weight 0.6` などで副露headの重みを上げる |

`prefetch-factor` は「workerごとの先読みバッチ数」です。例えば
`num-workers=12, prefetch-factor=4` は最大48バッチを先読みするため、
最初からこの値を使わないでください。

## 5. 本番学習の開始例

スモークテスト後は `--profile-batches` を外して本番学習します。以下は安全寄りの
開始点であり、プロファイル結果に応じて batch size と worker 数を変更します。

```bash
python advanced_training/train_large.py \
    --data data2023.zip data2022.zip \
    --model vit_multitask_large \
    --epochs 30 \
    --batch-size 256 \
    --accumulation-steps 1 \
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
    --num-workers 4 \
    --prefetch-factor 2 \
    --persistent-workers \
    --run-name vit_mt_blackwell_bf16
```

## 6. `torch.compile` の使い方

通常学習が正常に動き、速度の基準値を取得してから追加します。

```bash
--compile --compile-mode reduce-overhead
```

注意:

- 初回forwardはコンパイルにより長く止まったように見えることがある。
- `max-autotune` はコンパイル時間とメモリ使用量が増えるため、最初は使わない。
- エラー、停止、速度低下があれば外し、BF16 / TF32 / DataLoader調整を優先する。

## 7. 監視指標とマルチタスク重み

デフォルトのbest checkpoint監視指標は、マルチタスクでは `top1_acc` です。
特定headを重視する場合は、例えば以下のようにします。

```bash
--monitor-metric fulou_acc --fulou-weight 0.6
```

利用可能な検証メトリクスは `metrics.csv` の `val_` 接頭辞付きカラムで確認します。
存在しないmetricを指定した場合、そのepochではbest checkpointとearly stoppingは
更新されません。

## 8. Resume

途中再開はrun directoryか `last_model.pth` を指定します。

```bash
python advanced_training/train_large.py \
    --data data2023.zip data2022.zip \
    --resume runs/vit_mt_blackwell_bf16
```

resume時は保存済みの学習・データ生成設定を復元します。現在のCLIで明示した引数は
上書きとして扱われるため、例えば総epoch数を延長する場合は `--epochs 40` を追加します。
`--data`、`--device`、出力先など実行環境固有の引数は現在のCLI値を使います。
