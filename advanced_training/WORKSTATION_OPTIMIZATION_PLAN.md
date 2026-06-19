# Workstation Optimization Plan

`advanced_training/train_large.py` のワークステーション実行を、停止箇所が分かりやすく、
大規模データでもメモリ使用量を制御しやすい実装にするための変更計画です。

## 実装状況

- 完了: 非公開手牌のマスク、相対席順の修正
- 完了: Resume config復元、評価データ全件化、評価条件の統一
- 完了: zero-weight headの監視指標除外、CUDA preflight
- 完了: profileの逐次処理、初回バッチ・DataLoader・計算時間の可視化
- 一部完了: 局ログを一度だけ保持するcompact sample descriptor
- 未実装: プロセス合計RSSの自動計測、自動パラメータ探索、特徴量生成キャッシュ

## 目的

- `Task weights: ...` の後で無表示になり、停止と誤認される状態をなくす。
- DataLoader起動、初回バッチ生成、GPU計算、`torch.compile` を個別に計測する。
- worker数を増やした際のRAM増加を抑える。
- Blackwell環境の不適合を学習開始前に明確なエラーとして検出する。

## Phase 1: 可視性と安全性（優先度: 高）

1. `_profile_training_batches` の無表示収集を廃止する。
   - バッチをリストへ全件保存せず、上限付きイテレータから逐次学習する。
   - `Preparing profile batch 1/N` または専用progress barを表示する。
   - データ待ち時間とGPU学習時間を分けてログへ記録する。
2. 初回バッチのタイムアウト診断に必要なログを追加する。
   - DataLoader iterator作成開始、最初のバッチ受信、最初のGPU転送を記録する。
   - worker数、総先読みバッチ数、概算先読みサンプル数を表示する。
3. CUDA preflightを追加する。
   - GPU名、compute capability、PyTorchが含むCUDA architectureを表示する。
   - 小さなBF16 forward/backwardを実行し、失敗時は学習前に終了する。
   - `print_model_summary` の例外を表示だけで握り潰さず、CUDA実行エラーは失敗扱いにする。

### 検証

- `num_workers=0` と `num_workers>0` の両方でprofileが進捗表示される。
- profileバッチ数を増やしてもCPU側に全バッチを保持しない。
- CUDA非対応環境では `Task weights` より前に原因を含むエラーで終了する。

## Phase 2: DataLoaderメモリ構造（優先度: 高）

1. `MahjongDataset.samples` をコンパクト化する。
   - 現在の各sampleが `kyoku_log` を直接参照する形式をやめる。
   - 局ログは一度だけ保持し、sampleは `(game/kyoku ID, log_idx, player_id,
     action_type, label)` のような小さいdescriptorにする。
2. 複数ZIP結合時も局ログを重複させない。
3. worker起動方式ごとのRAM使用量を計測する。
   - Linuxの既定方式と `spawn` を比較し、対応可能ならCLI選択肢を追加する。

### 検証

- 同じseedで分割件数、action counts、各sampleのtensor/label/actionが変更前と一致する。
- 1、4、8、12 workersで親・子プロセスの合計RSSを比較する。
- 既存の単一タスク、マルチタスク、game-level splitのテストを通す。

## Phase 3: 自動ベンチマーク（優先度: 中）

1. worker数とbatch sizeを候補リストから比較する短時間ベンチマークを追加する。
2. 以下をCSVまたはJSONへ保存する。
   - time-to-first-batch
   - samples/sec
   - GPU利用率またはGPU待機時間
   - peak VRAM
   - 親・workerのpeak RAM
3. 最速値だけでなく、RAM使用量に上限を設けた推奨設定を出す。

## Phase 4: 追加最適化（優先度: 低、計測後に判断）

- StateEncoderのCPUプロファイルを取り、局面再構築の重複計算を減らす。
- 同一局面・プレイヤーで共有可能な中間特徴だけをキャッシュする。
- 非同期GPU転送やbatch生成済みtensorの保存を比較する。
- `torch.compile` のモード別結果を記録し、Blackwell向け既定値の変更要否を判断する。

キャッシュや事前tensor化はディスク容量とデータ更新手順を複雑にするため、Phase 1と
Phase 2の計測でCPU特徴量生成が主要ボトルネックだと確認できた場合のみ実施します。
