# Large-Scale Mahjong ViT Training Plan

作成日: 2026-07-13  
対象リポジトリ: `akkey2017/mahjong_discard_model_v2`  
対象ブランチ: `codex/mahjong_discard_model_v2`

---

## 1. 目的

本計画の目的は、1年あたり約14万牌譜、10年分で約140万牌譜、1000万局以上の牌譜を対象として、単一GPUワークステーション上で安定かつ高速に教師あり学習を実行できる麻雀AI学習基盤を構築することである。

最終的には以下を達成する。

- 10年分の牌譜をRAMへ全展開せず処理できる
- 牌譜の前処理を再利用可能にし、毎EpochでJSONを再解析しない
- 1局を先頭から1回だけ処理し、局面再構築の重複計算をなくす
- RTX PRO 6000 Blackwellを継続的に高稼働させる
- ViTを中心とした単一のモデル系統へ整理する
- 打牌、リーチ、副露、槓、和了を同一状態から予測する
- 現在局面だけでなく、必要に応じて局内の時系列も学習できる
- データ形式・特徴量形式・モデル形式にバージョンを付与し、再現可能な学習を行う
- 学習、評価、再開、データ追加を長期運用できる

---

## 2. 対象ハードウェア

想定する主な構成は以下である。

- CPU: AMD Ryzen Threadripper 9960X 24コア
- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- RAM: 192 GiB
- Storage: 3 TB NVMe SSD
- Motherboard: GIGABYTE TRX50 AERO D

この構成ではGPU演算能力とVRAM容量に大きな余裕がある一方、次の要素が先にボトルネックになりやすい。

1. PythonによるJSON解析
2. 牌譜からの状態再構築
3. CPU側での特徴量生成
4. NVMeの読み書き
5. DataLoaderのプロセス間転送
6. CPUメモリ帯域

したがって、最適化はGPUモデルだけでなく、データ形式・前処理・並列化・転送まで含めて行う。

---

## 3. リポジトリ方針

### 結論

**新しいリポジトリは作らず、現在のリポジトリを継続利用する。**

理由は以下である。

- 現行ブランチには他家の非公開手牌を隠す修正が入っている
- 対象プレイヤー基準の相対席順修正が入っている
- 局ログをサンプルごとに直接持たず、局ID参照にする改善が入っている
- 現在のViT、マルチタスクhead、評価、checkpoint処理を再利用できる
- 過去の実験・Issue・PR・変更履歴を保てる
- 旧実装との一致テストを行いやすい

ただし、mainへ直接大規模変更を入れず、専用ブランチで段階的に実装する。

推奨ブランチ名:

```text
feature/large-scale-vit-pipeline
```

または、変更範囲が大きい場合:

```text
rewrite/streaming-vit-v3
```

現行の動作基準を残すため、着手前にtagを付ける。

```bash
git tag baseline-codex-2026-07
git push origin baseline-codex-2026-07
```

新しいリポジトリを検討するのは、以下の場合に限る。

- 既存CLIやcheckpoint互換性を完全に捨てる
- 学習基盤を独立ライブラリとして公開する
- 既存コードを一切再利用しない
- 推論Botと学習基盤を別プロジェクトとして運用する

現時点ではそこまで分離する必要はない。

---

## 4. アルゴリズム方針

### 4.1 最初の段階では学習アルゴリズムを変えない

最初の大規模化では、以下を維持する。

- 教師あり模倣学習
- 現在局面からの行動予測
- 入力形式: `C × 4 × 9`
- backbone: Vision Transformer
- multi-task head:
  - 打牌
  - リーチ
  - 副露
  - 槓
  - 和了
- task-masked CrossEntropyLoss
- BF16 mixed precision

変更するのはデータ供給方式である。

```text
旧:
ZIPを全展開
→ 全局ログをRAM保持
→ 全サンプルdescriptorをRAM保持
→ __getitem__ごとに局を先頭から再生
→ 380ch Tensorを生成
→ GPUへ転送

新:
ZIPを一度だけ正規化
→ コンパクトな局・イベントshardへ保存
→ 複数workerが異なる局を順方向に1回だけ処理
→ バッチを先読み
→ GPUへ継続供給
```

### 4.2 モデルはViTへ一本化する

CoAtNetとResNetは、新しい大規模学習パイプラインでは必須としない。

維持するモデル候補:

- `vit_small`
- `vit_base`
- `vit_large`
- `vit_multitask_large`
- 将来の`vit_temporal_multitask`

最低限必要なのは`vit_multitask_large`のみである。

モデル生成は設定オブジェクトへ統一する。

```python
@dataclass
class ViTConfig:
    in_channels: int
    embed_dim: int
    depth: int
    heads: int
    patch_size: tuple[int, int]
    dropout: float
    drop_path: float
```

### 4.3 局の流れの扱い

現行モデルは、対象局面までの出来事を状態へ集約しているため、現在局面の状況は学習できる。

一方、次のような正確な順序情報は十分には保持していない。

- 捨て牌の順番
- 各打牌の巡目
- 手出し・ツモ切りの連続
- リーチ前の打牌推移
- 方針変化
- 直近数巡の速度感
- 同一プレイヤーの行動系列

そのため、最終段階ではSnapshot ViTにTemporal Transformerを追加できる構成を採用する。

```text
Snapshot State
    ↓
Snapshot ViT
    ↓
state embedding ─────┐
                     ├→ fusion → task heads
Event History        │
    ↓                │
Temporal Transformer ┘
```

ただし、最初から時系列モデルへ変更しない。

順序は以下とする。

1. 現行ViTと同じ入出力で大規模データ基盤を完成させる
2. 10年分学習のViT baselineを取得する
3. 時系列モデルを追加する
4. 同じvalidation setで比較する

---

## 5. 目標アーキテクチャ

```text
Raw ZIP files
    ↓
Dataset Scanner
    ↓
Normalizer / Validator
    ↓
Compact Round Shards
    ↓
Streaming IterableDataset
    ↓
Incremental Round Processor
    ↓
Batch Builder
    ↓
Pinned Memory Queue
    ↓
Optional GPU Feature Expander
    ↓
Snapshot ViT
    ↓
Optional Temporal Transformer
    ↓
Multi-task Heads
    ↓
Loss / Metrics / Checkpoint
```

---

## 6. データ形式

### 6.1 Raw JSONを学習時に直接使用しない

Raw ZIPは原本として保持する。

学習時には、事前に正規化したshardを使用する。

推奨構造:

```text
data/
├── raw/
│   ├── data2015.zip
│   ├── data2016.zip
│   └── ...
├── prepared/
│   └── schema_v1/
│       ├── manifest.json
│       ├── train/
│       │   ├── shard_000000/
│       │   ├── shard_000001/
│       │   └── ...
│       ├── validation/
│       └── test/
└── cache/
```

### 6.2 Shard内容

初期案:

```text
shard_000000/
├── rounds.npy
├── events.npy
├── offsets.npy
├── metadata.npy
└── checksum.json
```

- `rounds.npy`: 局開始時の情報
- `events.npy`: 正規化されたイベント列
- `offsets.npy`: 各局のイベント開始・終了位置
- `metadata.npy`: game ID、year、split、ルール情報など
- `checksum.json`: shard内容の検証情報

### 6.3 Compact event schema

例:

```python
event_dtype = np.dtype([
    ("type", np.uint8),
    ("player", np.uint8),
    ("tile", np.uint8),
    ("flags", np.uint8),
    ("meld_offset", np.uint32),
])
```

候補イベント:

- qipai
- zimo
- dapai
- fulou
- gang
- gangzimo
- kaigang
- lizhi
- hule
- pingju

JSONの辞書・文字列をそのまま保存せず、可能な限り整数へ正規化する。

### 6.4 Split

train、validation、testはゲーム単位で固定する。

```python
bucket = stable_hash(f"{archive}:{game_id}") % 10000
```

例:

- train: 0–9799
- validation: 9800–9899
- test: 9900–9999

同じ対局の局面が複数splitへ入らないようにする。

分割規則はmanifestへ保存する。

---

## 7. Incremental Round Processor

### 7.1 目的

現行の`StateEncoderV2.encode(log_idx)`は、局面ごとに局の先頭から再生する。

新実装では、1局を1回だけ順方向に処理する。

```python
state = RoundState.from_qipai(qipai)

for event in events:
    samples = label_builder.before_event(state, event)
    yield from samples
    state.apply(event)
```

### 7.2 管理する状態

```python
@dataclass
class RoundState:
    hands
    rivers
    melds
    ankans
    scores
    riichi_status
    riichi_turn
    ippatsu_status
    furiten_status
    dora_indicators
    visible_tiles
    last_discard
    last_discard_by_player
    draw_count
    turn_count
    honba
    kyotaku
    dealer
    round_wind
```

### 7.3 正確性

旧Encoderとの一致テストを用意する。

テスト対象:

- 配牌直後
- 通常ツモ・打牌
- 赤ドラ
- ツモ切り
- リーチ
- チー
- ポン
- 大明槓
- 暗槓
- 加槓
- 槓ドラ
- ロン
- ツモ
- 流局

最初の段階では、旧Encoderと同じ特徴量を生成する。

その後、旧Encoderの近似・未実装箇所を修正する。

---

## 8. 特徴量schema

### 8.1 Schema version

特徴量形式には必ずversionを付与する。

例:

```text
snapshot-v2-incremental
```

checkpointとprepared datasetの両方に保存する。

互換性のないschema同士はresume・評価を禁止する。

### 8.2 現行特徴量の見直し

現行380chには予約用ゼロチャンネルが含まれる。

大規模学習前に以下を確認する。

- 使用中のチャンネル数
- 常にゼロのチャンネル
- 常に同じ値のチャンネル
- 不正確な近似特徴
- 情報漏洩の有無
- 重複特徴
- 推論時に取得できない特徴

修正候補:

- 正確な残り牌数
- 正確な巡目
- リーチ宣言巡目
- 一発状態
- 手出し・ツモ切り
- フリテン
- 各打牌の巡目
- 鳴きによる状態更新
- 暗槓の公開情報
- 自分以外の非公開情報の完全排除

### 8.3 CPU生成とGPU生成

最初はCPU上で現行互換のTensorを作る。

その後、ベンチマーク結果に応じて、compact stateからGPU上で特徴量を展開する。

GPU Feature Expanderを採用する条件:

- DataLoader待ちが総時間の10%以上
- CPU使用率が高い
- GPU使用率が低い
- compact state転送がdense Tensor転送より十分小さい
- 旧CPU Encoderとの一致テストを通過できる

採用しない条件:

- CPU生成でGPU使用率が十分高い
- 実装複雑性に対して効果が小さい
- 変換処理が`torch.compile`を不安定にする

---

## 9. サンプルschema

### 9.1 1状態1レコード

現行は同一局面でもタスクごとに別サンプルになり得る。

新形式では1状態に複数のラベルを持たせる。

```python
@dataclass
class MultiTaskTarget:
    dapai_label: int
    dapai_mask: bool

    riichi_label: int
    riichi_mask: bool

    fulou_label: int
    fulou_mask: bool

    gang_label: int
    gang_mask: bool

    hule_label: int
    hule_mask: bool
```

利点:

- 同一状態の重複特徴量生成を防ぐ
- 同じ状態をbackboneへ複数回通さない
- バッチ内のPython文字列処理を削減する
- 全head常時出力にできる
- `torch.compile`と相性がよい

### 9.2 負例サンプリング

負例は全件保存せず、設定可能にする。

```yaml
negative_sampling:
  riichi:
    keep_probability: 0.25
  fulou:
    max_negative_per_positive: 4
  gang:
    keep_probability: 1.0
  hule:
    keep_probability: 1.0
```

sampling seedをmanifestへ保存する。

validation/testでは原則として固定かつ全件評価する。

---

## 10. Streaming Dataset

### 10.1 IterableDataset

全サンプル一覧をRAMへ持たない。

Datasetが保持するのは以下のみとする。

- manifest
- shard path
- split
- seed
- worker ID
- epoch ID
- sampling configuration

### 10.2 Worker分割

```python
worker_info = torch.utils.data.get_worker_info()
my_shards = shards[worker_id::num_workers]
```

worker間で同じshardを重複処理しない。

### 10.3 シャッフル

完全ランダムshuffleは巨大データでは高価である。

次の2段階shuffleを用いる。

1. shard順をshuffle
2. shard内の局順をshuffle bufferで並べ替える

例:

```text
shuffle_buffer_rounds = 8192
```

### 10.4 Worker内バッチ生成

workerから1サンプルずつ返すのではなく、完成したbatchを返す方式も比較する。

```python
DataLoader(
    dataset,
    batch_size=None,
    num_workers=12,
    persistent_workers=True,
    prefetch_factor=2,
    pin_memory=True,
)
```

---

## 11. 学習ループ

### 11.1 Stepベース

10年分では1 Epochが巨大になるため、stepベースへ変更する。

主要引数:

```text
--max-steps
--warmup-steps
--validate-every
--checkpoint-every
--log-every
--samples-per-virtual-epoch
```

例:

```bash
python -m training.train_vit \
  --data-manifest data/prepared/schema_v1/manifest.json \
  --model vit_large \
  --max-steps 300000 \
  --warmup-steps 10000 \
  --validate-every 5000 \
  --checkpoint-every 5000 \
  --batch-size 1024 \
  --amp-dtype bf16
```

### 11.2 GPU設定

初期値:

```text
AMP: BF16
TF32: enabled
cudnn.benchmark: enabled
pin_memory: enabled
non_blocking transfer: enabled
drop_last: enabled
```

`torch.compile`は最後に比較する。

### 11.3 Batch size探索

候補:

```text
256
512
1024
2048
4096
```

最大スループットだけでなく、validation精度も比較する。

勾配累積は、物理batchを大きくできない場合にのみ使う。

### 11.4 Worker数探索

候補:

```text
4
8
12
16
20
```

Threadripper 9960Xは24コアなので、12を初期値とする。

環境変数:

```bash
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
```

---

## 12. 時系列モデル

### 12.1 実装時期

Snapshot ViT baseline完成後に実装する。

### 12.2 Event token

1イベントを次のembeddingの和で表現する。

```text
event type embedding
relative player embedding
tile embedding
flags embedding
turn embedding
position embedding
```

対象イベント:

- ツモ
- 打牌
- リーチ
- チー
- ポン
- 槓
- 槓ドラ
- 和了
- 流局

### 12.3 履歴長

候補:

```text
32
64
128
256 events
```

まず64または128で比較する。

### 12.4 Causal mask

判断時点より未来のイベントは入力しない。

学習サンプル生成時に履歴を切り出し、未来情報漏洩テストを追加する。

### 12.5 比較実験

最低限次を比較する。

- Snapshot ViT
- Snapshot ViT + last 64 events
- Snapshot ViT + last 128 events

評価指標:

- dapai top-1 / top-3
- riichi precision / recall / F1
- fulou precision / recall / F1
- gang precision / recall / F1
- log loss
- calibration
- 局面区分別精度
- 巡目別精度
- リーチ者有無別精度
- 副露数別精度

---

## 13. 評価設計

### 13.1 データ分割

- train: 過去データ中心
- validation: 固定hash split
- test: 固定hash split
- temporal holdout: 最新年を丸ごと保持する追加評価

例:

```text
train: 2015–2023
validation: 2015–2023の固定hash subset
test: 2024
```

ランダムsplitだけでなく、年代差への汎化も確認する。

### 13.2 指標

打牌:

- top-1
- top-3
- top-5
- cross entropy
- 牌種別accuracy

二値判断:

- precision
- recall
- F1
- ROC-AUC
- PR-AUC
- calibration error

全体:

- task別loss
- task別sample数
- 年別精度
- ルール別精度
- プレイヤー段位別精度
- 序盤・中盤・終盤別精度

### 13.3 比較可能性

各runへ以下を保存する。

```text
git commit
dataset manifest hash
feature schema version
target schema version
model config
optimizer config
scheduler config
seed
PyTorch version
CUDA version
GPU name
```

---

## 14. 自動ベンチマーク

### 14.1 計測項目

- JSON files/sec
- games/sec
- rounds/sec
- events/sec
- generated states/sec
- training samples/sec
- time to first batch
- DataLoader wait time
- CPU encode time
- H2D transfer time
- GPU forward time
- GPU backward time
- GPU utilization
- peak VRAM
- parent RSS
- worker RSS
- total RAM
- disk read throughput
- disk write throughput

### 14.2 合格基準

目標:

```text
GPU utilization: 平均85%以上
DataLoader wait: 総時間の10%未満
Swap usage: 0
RAM headroom: 30 GiB以上
worker起動時のDataset複製: なし
resume再現性: あり
```

性能目標は実測後に更新する。

---

## 15. ストレージ計画

3TB NVMeを前提とする。

元ZIPは1年あたり約500MB〜1GBであり、10年分でも約5〜10GBに収まる見込みである。そのため、raw ZIP自体の容量は大きな問題にならない。

3TBあれば、以下を同一ストレージ上に保持できる可能性が高い。

- OS、Python、CUDA、Docker環境
- repository
- 10年分のraw ZIP
- compact形式のprepared shard
- checkpoint
- 実験ログ
- 前処理中の一時ファイル
- 複数schemaの比較用データ

ただし、Denseな`[N, 380, 4, 9]`特徴量を全局面分保存すると、サンプル数によっては数TB以上になるため、全件保存は行わない。

保存形式はcompact event shardを基本とし、特徴量は学習時にIncremental EncoderまたはGPU Feature Expanderで生成する。

運用上は約300GB以上の空き容量を残すことを目安とする。`scan_dataset.py`と1年分の実変換によってprepared shardの容量を測定し、10年分の予測使用量がストレージ上限へ近づく場合のみ追加SSDを検討する。

現時点では追加SSDを前提としない。

---

## 16. ディレクトリ再構成案

```text
mahjong_discard_model_v2/
├── src/
│   └── mahjong_ai/
│       ├── data/
│       │   ├── scan.py
│       │   ├── normalize.py
│       │   ├── shard_format.py
│       │   ├── streaming_dataset.py
│       │   ├── batch_builder.py
│       │   └── split.py
│       ├── state/
│       │   ├── round_state.py
│       │   ├── incremental_encoder.py
│       │   ├── feature_schema.py
│       │   └── rules.py
│       ├── models/
│       │   ├── vit.py
│       │   ├── temporal_transformer.py
│       │   ├── multitask.py
│       │   └── config.py
│       ├── training/
│       │   ├── train.py
│       │   ├── evaluate.py
│       │   ├── losses.py
│       │   ├── metrics.py
│       │   ├── checkpoint.py
│       │   └── profiler.py
│       └── inference/
│           └── agent.py
├── scripts/
│   ├── scan_dataset.py
│   ├── prepare_dataset.py
│   ├── benchmark_pipeline.py
│   └── train_vit.py
├── tests/
│   ├── data/
│   ├── state/
│   ├── models/
│   └── training/
├── configs/
│   ├── prepare_schema_v1.yaml
│   ├── vit_large.yaml
│   └── temporal_vit_large.yaml
├── PLAN.md
└── pyproject.toml
```

既存コードはすぐ削除せず、移行期間中は`legacy/`または現行位置に残す。

---

## 17. 実装ロードマップ

# Phase 0: Baseline固定

目的: 現行ブランチの挙動を記録する。

実装:

- tag作成
- 現行ViTの短時間学習
- 現行validation指標保存
- 1万ファイルで速度計測
- RAM、VRAM、DataLoader待ち計測
- 既知の不具合一覧作成

成果物:

- `baseline_metrics.json`
- `baseline_profile.json`
- baseline checkpoint
- dataset sample fixture

完了条件:

- 同じseedで再実行可能
- 現行出力を比較基準として保存済み

---

# Phase 1: Dataset Scanner

目的: 実データの規模を正確に把握する。

実装:

- `scripts/scan_dataset.py`
- ファイル数集計
- 対局数集計
- 局数集計
- イベント数集計
- task候補数集計
- malformed JSON集計
- 年別・ルール別統計
- 推定prepared容量
- 1コア・複数コア速度測定

完了条件:

- 1年14万ファイルを最後までscanできる
- 10年分の推定時間と容量を算出できる
- 不正データ率を把握できる

---

# Phase 2: Incremental State Engine

目的: 1局を1回だけ処理する。

実装:

- `RoundState`
- `apply_event`
- `snapshot`
- 正確な巡目
- 正確な残り牌数
- リーチ宣言巡目
- 一発状態
- 手出し・ツモ切り
- 見えている牌
- 相対席順
- 非公開情報のmask

完了条件:

- 旧Encoderと互換対象特徴が一致する
- 代表局面testを通過する
- 旧実装より局処理速度が十分に向上する

---

# Phase 3: Normalizer and Shard Writer

目的: Raw JSONを一度だけcompact形式へ変換する。

実装:

- `prepare_dataset.py`
- multiprocessing
- shard writer
- manifest
- checksum
- resume
- corrupted file isolation
- progress log
- train/validation/test split

完了条件:

- 1年分を中断・再開可能
- shardのchecksum検証が通る
- 同じ入力・seedで同じmanifestを生成する
- 192GiB RAMを使い切らない

---

# Phase 4: Streaming Dataset

目的: 全データをRAMへ置かず学習する。

実装:

- `IterableDataset`
- worker shard assignment
- shard shuffle
- shuffle buffer
- persistent worker
- prefetch
- pinned memory
- deterministic seed
- distributed-ready worker partition

完了条件:

- worker数に比例してDataset本体のRAMが増えない
- 1、4、8、12、16 workerで完走する
- duplicate sampleがない
- missing sampleがない
- epoch/step再開が可能

---

# Phase 5: Multi-task Sample Unification

目的: 同一状態の重複計算をなくす。

実装:

- 1状態1レコード
- label tensor
- mask tensor
- 全head常時forward
- negative sampling
- task別weight
- task別metrics

完了条件:

- 現行task件数と整合する
- 同一状態のbackbone重複実行がなくなる
- task別lossが正常に収束する

---

# Phase 6: ViT-only Training Stack

目的: モデル系統をViTへ整理する。

実装:

- ViTConfig
- vit_small/base/large
- multitask heads
- BF16
- TF32
- step-based scheduler
- checkpoint
- resume
- EMA
- gradient clipping
- profiling
- compile切替

完了条件:

- CoAtNet/ResNetに依存せず学習可能
- 現行ViT checkpoint互換方針が明文化されている
- 1年分の学習を安定完走できる

---

# Phase 7: Workstation Auto-Tuning

目的: 現在のワークステーションで最速設定を決める。

探索:

- workers: 4, 8, 12, 16, 20
- batch: 256, 512, 1024, 2048, 4096
- prefetch: 1, 2, 4
- compile: on/off
- CPU dense encode / GPU feature expand
- shard size
- compression方式

完了条件:

- 推奨設定をJSONへ保存
- GPU利用率85%以上を目標に調整
- RAMに30GiB以上余裕を残す
- Swap 0

---

# Phase 8: 10-year Snapshot ViT Baseline

目的: 時系列モデル追加前の基準モデルを作る。

実装:

- 10年分prepared dataset
- Snapshot ViT学習
- 年代holdout評価
- task別評価
- calibration
- error analysis

完了条件:

- 10年分を最後まで学習できる
- checkpointからresumeできる
- 固定test setの評価結果が保存される
- 年別性能差が可視化される

---

# Phase 9: Temporal Transformer

目的: 局の流れを明示的に学習する。

実装:

- event tokenizer
- event embedding
- causal Temporal Transformer
- Snapshot ViTとのfusion
- 履歴長32/64/128比較
- history mask
- padding
- future leakage test

完了条件:

- Snapshot ViT baselineを上回るか検証済み
- 時系列追加による速度・VRAM増加を計測済み
- 精度向上が小さい場合はSnapshot ViTを主系統として維持する

---

# Phase 10: Inference Integration

目的: 学習モデルを対局Botで使用する。

実装:

- 同じIncremental RoundStateを推論で使用
- checkpoint schema検証
- compact event history
- action legality mask
- latency benchmark
- batchなし単局面推論
- fallback behavior

完了条件:

- 学習時と推論時の特徴量が一致する
- 他家の非公開情報を使用しない
- 合法手のみ返す
- 実戦で必要な推論時間内に収まる

---

# Phase 11: 継続運用

目的: 毎年の牌譜追加と再学習を容易にする。

実装:

- 新しいZIPだけ追加prepare
- manifest merge
- schema migration
- scheduled benchmark
- experiment registry
- dataset lineage
- model registry
- reproducibility report

完了条件:

- 新年データを全再変換せず追加できる
- 古いcheckpointとの互換性を自動判定できる
- 主要runを再現できる

---

## 18. テスト戦略

### Unit tests

- tile normalization
- red five handling
- relative seat
- meld parsing
- chi/pon/kan classification
- state transition
- riichi state
- ippatsu cancellation
- remaining tile count
- target generation
- negative sampling
- split hash

### Regression tests

- 旧Encoder vs 新Encoder
- 現行branch fixtureとの出力比較
- task counts
- deterministic preparation
- deterministic validation
- resume後のloss trajectory

### Integration tests

- 小さいZIPからprepare
- prepared dataから学習
- checkpoint保存
- resume
- evaluate
- inference

### Performance tests

- samples/sec
- RAM
- VRAM
- first batch time
- worker scaling
- shard size
- GPU utilization

---

## 19. リスク

### データ形式の誤解

対策:

- malformed sampleを隔離
- 年別統計を比較
- representative fixtureを保存
- rule variantをmetadataに持つ

### 情報漏洩

対策:

- 他家の非公開手牌をmask
- 判断時点より後のイベントを除外
- testで未来イベントを入れないことを確認
- 推論時取得可能な情報だけを使用

### 負例の偏り

対策:

- task別sample数を記録
- negative sampling設定を保存
- accuracyだけでなくPR-AUCを見る
- validationでは固定分布を使う

### SSD容量不足

対策:

- dense特徴量を保存しない
- compact event shard
- 追加NVMeを前提にする
- cacheに上限を設ける

### GPUがCPU待ちになる

対策:

- Incremental Encoder
- multiprocessing
- worker内batch生成
- pinned memory
- prefetch
- GPU Feature Expanderを比較

### 変更範囲が大きすぎる

対策:

- PhaseごとにPRを分ける
- 旧実装との一致テスト
- mainへ一括mergeしない
- baseline tagを保持する

---

## 20. PR分割案

### PR 1

```text
Dataset scanner and baseline profiling
```

### PR 2

```text
Incremental round state and regression tests
```

### PR 3

```text
Compact shard format and dataset preparation
```

### PR 4

```text
Streaming IterableDataset and worker partitioning
```

### PR 5

```text
Unified multi-task targets
```

### PR 6

```text
ViT-only step-based training stack
```

### PR 7

```text
Workstation benchmark and auto-tuning
```

### PR 8

```text
Temporal event transformer
```

### PR 9

```text
Inference integration and schema validation
```

---

## 21. 最初に着手するタスク

最初の実装対象は次の4点とする。

1. 現行ブランチへbaseline tagを付ける
2. `scan_dataset.py`を作る
3. 現行Encoderのfixtureを保存する
4. `IncrementalRoundState`の最小実装を作る

最初の成功条件:

```text
1万ファイルをscan
→ 実サンプル数と推定10年規模を算出
→ 1000局面で旧Encoderと新Encoderを比較
→ 新Encoderが同等出力をより高速に生成
```

この結果を見て、shard size、圧縮形式、worker数、追加SSD容量を確定する。

---

## 22. 最終判断

本プロジェクトは現在のリポジトリで継続する。

モデルはViTへ一本化する。

最初はアルゴリズムを変えず、データ処理基盤だけを大規模化する。

10年分のSnapshot ViT baseline完成後、局の流れを明示的に扱うTemporal Transformerを追加し、効果を実測する。

最終的な完成形は以下である。

```text
10年分Raw牌譜
→ 再利用可能なcompact shard
→ Incremental state processing
→ Streaming batch pipeline
→ Snapshot ViT
→ Optional Temporal Transformer
→ Multi-task prediction
→ Reproducible evaluation
→ Real-time inference
```
