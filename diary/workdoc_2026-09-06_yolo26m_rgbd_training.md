# YOLO26m RGB-D 学習 作業計画書兼記録書

**日付:** 2026年09月06日

**リポジトリ:** `YOPO_clone` (`rgb-d` branch)

**作業者:** Codex

## 1. 作業目的

DEIMv2 JAXで学習済みのYOLO26m backboneだけをYOPOのRGB branchへ厳密に
移植し、既存depth/fusion/2D・3D headを保ったRGB-DモデルをFULL学習する。
検証済みbest 1個だけを再現可能かつ公開安全な形で配布する。

### 1.1 ゴール要求分析

- **直截的な目的:** YOLO26m backboneを持つDEIMv2の成果を、YOPO RGB-D学習へ
  一貫して接続し、実用的な2D/3D性能を得る。
- **明示要求:** YOLO26m相当、公式initial weight由来、AMUSE、32GB GPUを
  活用する最大safe batch、FULL学習、best-only、diary/README、圧縮、release。
- **暗黙制約:** native YOLO headは使わない。private path・dataset・credential・run
  transcriptをcommitしない。失敗runをproduction表記しない。
- **非ゴール:** YOLO head側PAN/FPNの再導入、別optimizer比較、OBB専用派生、
  validationを使ったcalibration、全checkpointの配布。
- **成功条件:** 250-leaf transfer、capacity/resume/FULL、独立181画像validation、
  primaryと2D guardの同一epoch通過、model-only再読込、公開成果物の一致。

### 1.2 サブゴール構造

| ID | サブゴール | 成果物 | 検証 |
|---|---|---|---|
| SG-1 | backboneを厳密移植 | converter、YOLO26m RGB branch | 250/250 leaf、3-level parity |
| SG-2 | RGB-D境界を安定化 | train-only calibration | 7 leaf限定、holdout RMSE改善 |
| SG-3 | 最大safe batchでFULL学習 | epoch-10 best | finite、early-stop、181画像評価 |
| SG-4 | best-only配布 | model、config、metrics、manifest | SHA、再読込、公開scan |

### 1.3 トレーサビリティ

| Trace ID | 要求 | 証跡 |
|---|---|---|
| TR-TRANSFER | DEIMv2 backboneだけを移植 | `tests/test_yolo26m_jax_backbone_transfer.py`、設計書 |
| TR-TRAIN | RGB-D FULL学習 | resolved config、training metrics |
| TR-QUALITY | 2D/3D採否 | epoch-10独立validation JSON、下表 |
| TR-DELIVERY | best-only・公開安全 | model SHA、manifest、release archive |

## 2. 作業内容

### フェーズ1: 調査・設計

公式YOLO26m layers 0-10、採用DEIMv2 JAX tree、YOPO stage-8 stateを照合し、
250-leaf transferとstage-8 allowlistを固定した。詳細は
[`design_yolo26m_rgbd_transfer.md`](design_yolo26m_rgbd_transfer.md)を正本とする。

### フェーズ2: 実装・初期化

PyTorch YOLO26m RGB branch、strict converter、三段feature parity、初期weight
生成、train-onlyの7-leaf境界calibrationをtest-firstで実装した。

### フェーズ3: 学習・検証・配布

B24/BF16/AMUSEでFULL学習し、epoch 10を選択した。model-only成果物を元の
checkpointと同じcanonical stateで再生成し、そのファイルから独立評価した。

## 3. 作業チェックリスト

### 手順1: strict backbone transfer
- [x] 🖐 **操作**: JAX `ema::backbone/**`をYOPOのYOLO26m RGB branchへ移す。
- [x] 🔎 **確認**: 250/250 leaf、extra 0、P3/P4/P5 shapeと数値parityを確認する。
- [x] 🧪 **テスト**: `uv run pytest tests/test_yolo26m_jax_backbone_transfer.py -q`を成功させる。
- [x] 🛠 **エラー時対処**: missing/extra/shape/non-finite時は変換を中止し、mappingを明示修正する。

### 手順2: RGB-D境界calibration
- [x] 🖐 **操作**: train splitだけでprojection 3、depth adapter 3、beta 1をfitする。
- [x] 🔎 **確認**: 変更keyが7個だけで、3 levelすべてholdout RMSEが改善する。
- [x] 🧪 **テスト**: calibration正常系・rank不足・shape・non-finiteのfocused testを成功させる。
- [x] 🛠 **エラー時対処**: rank/finite条件を満たさない場合はartifactを出さずfail closedとする。

### 手順3: capacity、resume、FULL学習
- [x] 🖐 **操作**: B24/BF16/AMUSEで200-update/resume gate後にFULL学習する。
- [x] 🔎 **確認**: VRAM上限内、finite、checkpoint/resume成功、規定patience終了を確認する。
- [x] 🧪 **テスト**: 200→201 updateを別processで再開し、FULL epoch-10 bestを181画像評価する。
- [x] 🛠 **エラー時対処**: OOMなら直前safe batchへ戻し、non-finiteならcheckpointを採用しない。

### 手順4: best-only成果物
- [x] 🖐 **操作**: epoch-10 primary bestからmodel-only checkpoint 1個を生成する。
- [x] 🔎 **確認**: 1,167 tensor、non-finite 0、canonical state SHA一致を確認する。
- [x] 🧪 **テスト**: 配布checkpoint自体を`tools/test.py`で181画像再評価しexit 0を確認する。
- [x] 🛠 **エラー時対処**: metric/state SHA不一致時はreleaseせず元checkpointから再生成する。

## 4. 再現コマンド

repo rootで`just setup`を先に実行し、datasetをconfig記載の相対配置へ置く。
以下の変数は利用者自身のcheckpointを指し、未設定なら即時停止する。

```bash
: "${DEIMV2_BEST:?set DEIMV2_BEST to the accepted DEIMv2 JAX checkpoint}"
: "${STAGE8_BEST:?set STAGE8_BEST to the validated YOPO stage-8 checkpoint}"
: "${YOPO_BEST:?set YOPO_BEST to yopo_yolo26m_rgbd_best_epoch10.pth}"

uv run python tools/model_converters/prepare_yolo26m_rgbd_checkpoint.py \
  --jax-checkpoint "$DEIMV2_BEST" \
  --stage8-checkpoint "$STAGE8_BEST" \
  --target-config configs/yopo/nocs_fruits_2026_rgbd_yolo26m_stage1_full.py \
  --output work_dirs/yolo26m_rgbd_stage1_initial.pth \
  --weights ema --seed 20260903

uv run python tools/model_converters/calibrate_yolo26m_rgbd_frontend.py \
  --student-config configs/yopo/nocs_fruits_2026_rgbd_yolo26m_stage1_full.py \
  --student-checkpoint work_dirs/yolo26m_rgbd_stage1_initial.pth \
  --teacher-config configs/yopo/nocs_fruits_2026_rgbd_shared_stage8_sensor_depth_anchor.py \
  --teacher-checkpoint "$STAGE8_BEST" \
  --output work_dirs/yolo26m_rgbd_frontend_calibrated_train64.pth \
  --device cuda:0 --calibration-batches 16 --holdout-batches 4 \
  --batch-size 4 --samples-per-level 2048 --ridge 0.0001 \
  --depth-ridge 0.000001 --seed 20260906

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python tools/train.py \
  configs/yopo/nocs_fruits_2026_rgbd_yolo26m_stage2_calibrated_full.py \
  --work-dir work_dirs/yopo_yolo26m_rgbd_stage2_calibrated_full

uv run python tools/test.py \
  configs/yopo/nocs_fruits_2026_rgbd_yolo26m_stage2_calibrated_full.py \
  "$YOPO_BEST"
```

## 5. 結果

| Metric | stage-8 baseline | selected epoch 10 | delta |
|---|---:|---:|---:|
| HBB AP50 | 0.8714 | 0.8622128367 | -0.009187 |
| ellipse mAP50 | 0.8706 | 0.8613967896 | -0.009203 |
| projection AP50 | 0.8388 | 0.8342097371 | -0.004590 |
| shared AP25 (primary) | 0.2208 | 0.2679015474 | +0.047102 |
| strict 3D IoU25 | 0.1960 | 0.1914560553 | -0.004544 |
| OBB mAP50 | — | 0.3515181839 | — |

追加値はHBB AP50:95 `0.5734277508`、invalid prediction `0`。epoch 10 / iter
570の1,167 tensorは全finite。model-only checkpoint SHA256は
`831c82632adf1beff15327f0f633570c08cf18d4b3e06660b97792b5157b4ad0`、
canonical state SHA256は
`1f1750189f6b12edfc745a70a3b8fd11fd89ade9aa2d7bc65bb85556d296eab8`。
移植実装はcommit `3ee9970`、train-only境界calibrationはcommit `b80417a`。

## 6. 完了の定義

- [x] 250-leaf transferと三段feature parityが成立する。
- [x] B24 capacity、200→201 resume、FULL/early-stopがfiniteに完了する。
- [x] epoch 10がprimaryと全2D guardを同一validationで満たす。
- [x] model-only checkpoint 1個が元stateと一致し、独立評価で再現する。
- [x] private path・dataset・credential・session transcriptを成果物へ含めない。

## 7. 作業記録

**重要な注意事項:**

- 作業開始前に`date "+%Y-%m-%d %H:%M:%S %Z%z"`で時刻を確認する。
- 各作業項目の開始時と完了時に、成功・失敗・発見事項を記録する。
- エラー時は暗黙fallbackせず、原因、採否、再開点を明記する。
- DRY/KISS/SOLIDを保ち、性能原因に寄与しない証憑専用作業を増やさない。

| 日時 (JST) | 作業 | 結果 |
|---|---|---|
| 2026-09-06 00:24 | strict transfer | focused test成功、250 leaf、3-level parity成立 |
| 2026-09-06 00:26 | capacity | B24採用、B25は29,346 MiB上限超過で棄却 |
| 2026-09-06 00:37 | resume gate | 200 update完了、別processで201へ正常resume |
| 2026-09-06 01:49 | uncalibrated FULL | finite完走したが全promotion guard不合格 |
| 2026-09-06 02:03 | boundary calibration | train-only 7 leaf、holdout 3/3改善、validation gate合格 |
| 2026-09-06 02:51 | calibrated FULL | epoch30でpatience 4、exit 0、epoch10を選択 |
| 2026-09-06 02:59 | delivery validation | model-only checkpointから181画像指標を再現、exit 0 |
