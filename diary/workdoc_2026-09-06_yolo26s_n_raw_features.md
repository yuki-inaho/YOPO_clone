# RGB=s / depth=n raw feature転用：作業書・設計・結果

日付: 2026-09-06（実行記録はJST）。対象: `YOPO_clone`、branch `rgb-d`。
作業者: Codex。本書は公開用の作業書兼設計・結果要約である。

## 1. 目的と要求

### 1.1 ゴール要求分析

YOPOのRGB branchをYOLO26s、depth branchをYOLO26nとし、各scaleのDEIMv2
事前学習から**raw backboneとHybridEncoder/neck**を転用してRGB-D 2D/3Dを学習する。
DEIMのtask decoder/headやnative YOLO検出headは転用しない。

- TR-RAW: EMAではなく`params::backbone/`と`params::encoder/`を厳密転送する。
- TR-ARCH: 各枝のencoder後に融合する非対称モデルを構築する。
- TR-TRAIN: uv、AMUSE/BF16、32GB GPUの実測safe batchで段階学習・独立評価する。
- TR-QUALITY: 主指標と補助指標を分け、採用checkpointの独立評価による再現を確認する。
- TR-CURRICULUM: 過去の成功・不採用実験を参考に、必要な段だけ実行する。
- TR-DELIVERY: 設計、比較、再実行手順をdiaryへ記録し、source/testを公開安全に管理する。

非ゴールはDEIM/OBBの再学習、EMAへの暗黙切替、全checkpointの配布、
性能条件に達するまで無制限に実験を増設すること。
成功条件は転送/parity、実データgate/resume、設定終端までの学習、
同一checkpointでの採否、採用bestの181枚独立評価である。性能改善は保証しない。

## 2. アーキテクチャ設計（TR-RAW / TR-ARCH）

| branch | backbone / 入力 | backbone出力channel | 転用encoder / 出力 |
|---|---|---|---|
| RGB | YOLO26s layers 0–10 / RGB 3ch | 256 / 256 / 512 | sのDEIM HybridEncoder / 256×3段 |
| depth | YOLO26n layers 0–10 / depth 1ch | 128 / 128 / 256 | nのDEIM HybridEncoder / 256×3段 |

各encoderはprojection→AIFI→top-down FPN→PAN、8 heads、FFN 1024、AIFI 1層。
三段はstride 8/16/32。対応levelごとに`rgb + beta * depth_adapter(depth)`で融合し、
`EncodedPyramidNeck`の1×1 projection三段と派生P6を通してYOPO task stackへ渡す。
CoP depth contextにもdepth encoder後の256-channel三段を使う。

depth stemはraw RGB kernelを入力channel方向に**合計**して1chへ変換する。
これはdepthを3chへ複製して元の畳み込みに渡す場合と等価であり、平均ではない。
BN統計はfreeze、affineと両枝backbone/encoderを含む全parameterは学習対象。
入力はRGB/255、depthはmetres。3D camera geometryを壊すMosaic/MixUpは使わない。

転送元は両scaleともDEIMのstep 3750。checkpoint自体はEMA指標で選ばれたbestに
rawも同梱されたもので、独立に選定した「raw-best」とは主張しない。
各枝backbone 200 source leaf、encoder 23 source leafを転送する。
YOPO stage8からtask部とP6の570 leafを再利用する。

### 2.1 初期境界の適合

raw転送の数値parityが成立しても、以前のYOPO headが期待する特徴表現には一致しない。
未調整の5epoch gateはHBB AP50=.0078645、shared AP25≈0だった。
そこで**YOPO追加学習前のraw転送初期値から**train64枚でridge fitし、別train16枚で確認した。
ラベルとvalidationはfitに使わず、以下の4 leafだけを変えた。

- `neck.projections.{0,1,2}.weight`
- `bbox_head.depth_query_sampler.output_projection.0.weight`

三段のdepth線形写像は、bilinear ROI sampling・poolingと可換なので既存output linearへ
折り込む。両枝のraw backbone/encoderはこの適合処理ではbyte単位で不変。
その後の通常学習では両枝も更新する。
train holdoutの相対RMSEは融合三段1.178/1.343/1.259→.601/.536/.338、
depth context三段89.30/80.21/176.73→.435/.064/.031へ減少した。
調整済み初期値（0 update）の181枚評価はHBB .7984、shared .2181。
未調整5epochとは学習履歴が異なり、同stepの対照実験とは表記しない。

## 3. カリキュラムと実行結果（TR-TRAIN / TR-CURRICULUM）

| 段階 | 実行内容 | 終端・採否 |
|---|---|---|
| A | raw転送、境界4 leafのtrain-only適合 | parity 12/12合格、調整済み初期値を採用 |
| B | B25・AMUSE/BF16・base/aux LR 5e-5、5epoch gateから全state resume | 40epoch / 2,160 updateでpatience4早期終了、**epoch20を採用** |
| C | B bestからweights-only、fresh optimizer、LRを1e-5へ低下 | 15epoch / 810 update完走、shared改善条件未達で不採用 |

Bは50epoch上限、validationは5epochごと、monitorは`projection/shared_AP_25`、
min_delta=.001、patience4。warmup100、外部scheduler/auto LR scalingなし。
Cはloss、全層更新、paramwise倍率、batchを維持した最大15epochの仕上げ。
Cのepoch5/10/15を一時保持し、同一候補で次の全条件を要求した。

1. shared AP25 ≥ B best + .001。
2. HBB/ellipse/projection AP50、strict 3D IoU25がそれぞれB best − .005以上。
3. 全metric finite、invalid prediction=0。

3候補とも条件2/3は通過したが条件1を満たさなかった。Cを棄却し、B bestを維持した。
異なるepochの2D/3D bestを組み合わせた成績は作らない。

### 3.1 同一181枚validationでの比較

| モデル / checkpoint | HBB AP50 | ellipse mAP50 | projection AP50 | shared AP25（主指標） | strict 3D IoU25 |
|---|---:|---:|---:|---:|---:|
| 従来HGNet stage8 / epoch5 | .8714 | .8706 | .8388 | .2208 | .1960 |
| 既存YOLO26m RGB + HGNet depth / epoch10 | .8622 | .8614 | .8342 | .2679 | .1915 |
| **今回s/n raw features / B epoch20（採用）** | **.8483** | **.8482** | **.8251** | **.3158** | **.1877** |
| 今回C epoch15（不採用） | .8594 | .8592 | .8308 | .2857 | .2064 |

採用bestのHBB AP50:95は.5777265412、invalid=0。
共有matchingのAP25はstrict 3D指標とは別物である。主指標は向上したが、
採用bestのHBB/ellipse/strict 3Dは従来stage8を下回るため「全面的な性能向上」としない。
Cは2D/strict 3Dの別のtrade-offを示したが、事前の主指標条件では不採用。
従来mの詳細は[過去作業書](workdoc_2026-09-06_yolo26m_rgbd_training.md)を参照。
native Rotatedデータの別split/画像数のAPとは直接比較しない。

### 3.2 parameterとVRAM

| 今回の部分 | parameter数 |
|---|---:|
| RGB YOLO26s backbone | 5,441,984 |
| RGB DEIM HybridEncoder | 5,770,496 |
| depth YOLO26n 1ch backbone | 1,365,184 |
| depth DEIM HybridEncoder | 5,639,424 |
| その他の融合・neck・YOPO task部 | 18,306,568 |
| **今回合計** | **36,523,656** |
| 従来HGNet stage8合計 | 34,405,710 |

今回合計は従来より2,117,946（約6.16%）増。backboneがs/nでもencoderを各枝に
配置するため、モデル全体の小型化にはなっていない。
RTX 5090総32,607 MiBに対してdriver上限29,346 MiBを設定した。
B27/B26は短い容量試験を通過しても長時間/resumeで上限を超えたためB25へ下げた。
採用Bのdriver peakは28,466 MiB、Cは28,052 MiBで、両runとも上限超過なし・exit0。

### 3.3 findings / struggles / tips

| 発見・難所 | 対処・次回の注意 |
|---|---|
| parity成功だけではYOPO task境界の適合を保証しない | raw枝を保ち、train-onlyで接続4 leafを適合してからgateする |
| depth contextの振幅差が大きかった | 融合特徴だけでなくCoP depth入力も比較する |
| JAX fixtureの最初の差はexporterの既定4-head設定によるもの | source manifestの実8-head encoder設定を読む。入力は32倍数にする |
| Bはe20がbestで、その後2D改善と3D低下が分離 | latestを採用せず同一epochの主指標・補助指標を確認する |
| 奥行き方向誤差中央値はB e20 8.485mm→e30 15.351mm | 2D APだけでなくray誤差も監視する。低LRだけで主指標が改善するとは限らない |
| 過去stage9 shape-onlyとstage10 reverse-KLDは不採用 | 同じ条件を再演せず、sensor anchorと学習済みheadを保った最小3段にした |
| 短時間capacityと長時間/resumeのpeakは異なる | driverを連続監視し、実gateを通ったB25を使う |

## 4. 再実行手順

以下はYOPO repo rootで実行する。環境未構築なら既存の`just setup`を使う。
datasetをconfig記載の相対`data/`配置へ置く。DEIM変数は`manifest.json`と
`arrays.npz`のあるディレクトリ、YOPO変数はcheckpointファイルを指定する。
converterは出力が既にあると停止する。再実験は別出力名と対応する`load_from`、
別work_dirを明示し、既存bestへ上書きしない。

```sh
: "${DEIM_S_BEST:?set to the DEIM YOLO26s source checkpoint directory}"
: "${DEIM_N_BEST:?set to the DEIM YOLO26n source checkpoint directory}"
: "${YOPO_STAGE8_BEST:?set to the YOPO stage8 task checkpoint}"

uv run python tools/model_converters/prepare_yolo26_raw_features.py \
  --rgb "$DEIM_S_BEST" --depth "$DEIM_N_BEST" --task "$YOPO_STAGE8_BEST" \
  --config configs/yopo/nocs_fruits_2026_rgbd_yolo26s_n_raw_features_full.py \
  --output work_dirs/yolo26s_n_raw_features_initial.pth

OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 uv run python \
  tools/model_converters/calibrate_yolo26_raw_feature_boundary.py \
  --student-config configs/yopo/nocs_fruits_2026_rgbd_yolo26s_n_raw_features_full.py \
  --student-checkpoint work_dirs/yolo26s_n_raw_features_initial.pth \
  --teacher-config configs/yopo/nocs_fruits_2026_rgbd_shared_stage8_sensor_depth_anchor.py \
  --teacher-checkpoint "$YOPO_STAGE8_BEST" \
  --output work_dirs/yolo26s_n_raw_features_calibrated.pth

OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 uv run python tools/train.py \
  configs/yopo/nocs_fruits_2026_rgbd_yolo26s_n_raw_features_calibrated_full.py \
  --work-dir work_dirs/raw_features_reproduction_gate \
  --cfg-options train_cfg.max_epochs=5 default_hooks.checkpoint.interval=1

OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 uv run python tools/train.py \
  configs/yopo/nocs_fruits_2026_rgbd_yolo26s_n_raw_features_calibrated_full.py \
  --work-dir work_dirs/raw_features_reproduction_full \
  --resume work_dirs/raw_features_reproduction_gate/epoch_5.pth

: "${YOPO_STAGE_B_BEST:?set to the selected primary best of the completed B run}"
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 uv run python tools/test.py \
  configs/yopo/nocs_fruits_2026_rgbd_yolo26s_n_raw_features_calibrated_full.py \
  "$YOPO_STAGE_B_BEST" --work-dir work_dirs/raw_features_reproduction_best_eval

OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 uv run python tools/train.py \
  configs/yopo/nocs_fruits_2026_rgbd_yolo26s_n_raw_features_refine.py \
  --work-dir work_dirs/raw_features_reproduction_refine \
  --cfg-options load_from="$YOPO_STAGE_B_BEST"
```

再実行時は別端末の`nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv -l 1`
でもメモリを監視する。記載peakは今回環境の測定値であり、他環境での上限保証ではない。
今回は対象training processだけを停止する監視wrapperで29,346 MiBを強制した。
上記は学習本体の再現コマンドで、監視wrapperは含まない。
上限超過/非finite時は対象runを止め、batchを下げてそのrunのlatestから再開する。
異なる段階へ移るときはbestをweights-onlyで使い、optimizerを引き継がない。

JAX fixture生成は**DEIM repoのuv環境**で行う。スクリプトはこのrepoの
`tools/model_converters/dump_yolo26_raw_features.py`。引数は
`--checkpoint`、`--scale s`または`n`、`--output`、depth側のみ`--depth`。
`JAX_PLATFORMS=cpu`でsourceを評価し、RGBを`rgb.npz`、depthを`depth.npz`として同じ
fixtureディレクトリへ保存する。YOPOでは次を実行する。

```sh
: "${RAW_FEATURE_FIXTURES:?set to the directory containing rgb.npz and depth.npz}"
OMP_NUM_THREADS=4 uv run python tools/model_converters/check_yolo26_raw_features.py \
  --config configs/yopo/nocs_fruits_2026_rgbd_yolo26s_n_raw_features_full.py \
  --checkpoint work_dirs/yolo26s_n_raw_features_initial.pth \
  --fixtures "$RAW_FEATURE_FIXTURES" --output work_dirs/raw_features_parity.json

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 uv run pytest \
  tests/test_yolo26_asymmetric_features.py \
  tests/test_portable_hybrid_encoder_neck.py \
  tests/test_yolo26m_jax_backbone_transfer.py \
  tests/test_jax_feature_transfer.py tests/test_shared_rgbd_yopo_config.py -q
```

## 5. 成果物と照合

採用モデル（weight-only、1,095 tensor）:
`work_dirs/yopo_yolo26s_n_raw_features_calibrated_full_20260906/best_ellipsoid_shared_AP_25_epoch_20.pth`。
SHA256: `227da40d4ec559f4792cac0dcde9cf3fea8afce5ded43a1853fcfc0bc460510d`。
対応configは`configs/yopo/nocs_fruits_2026_rgbd_yolo26s_n_raw_features_calibrated_full.py`。
モデル/data/runログはGitへ含めず、sourceと本書を管理する。今回新しいreleaseは作成していない。

| 入力・中間成果物 | SHA256 |
|---|---|
| DEIM s arrays.npz / step3750 raw | `7d01b0c5607ea85df17e1205b51239e0ecf3e62269b65cab77ab957d2c7b2a20` |
| DEIM n arrays.npz / step3750 raw | `c37033a602202214207e662f7d762e24f1393a4783e8f86a7bad6d83f95c5960` |
| YOPO stage8 task parent | `96c543a89e7fe3c433064f75bb0a0bb912997ca5304c3da11153ec9bcea1a0a3` |
| 未調整raw初期値 | `ff54e3af466ba2e0cfc5ac444b76d5da75a8e58ae241fa2f470edd6aaef205d1` |
| train-only調整済み初期値 | `b409fd7a2aae0ab0fb2da22e1439d45d37e9345a7a9c942bec54247db5a3e084` |

監査入口（いずれもrepo相対、ローカル実験成果物）:

- 転送/適合: `work_dirs/yolo26s_n_raw_features_initial.pth.json`、`work_dirs/yolo26s_n_raw_features_calibrated.pth.json`。
- parity: `work_dirs/yolo26s_n_raw_features_parity.json`。RGB/depth×2size×3level、12/12、最大絶対誤差7.82013e-5（atol1e-4/rtol1e-3）。
- B: `work_dirs/yopo_yolo26s_n_raw_features_calibrated_full_20260906/20260906_210612/vis_data/scalars.json`。
- 独立評価: `work_dirs/yopo_yolo26s_n_raw_features_b_best_eval_20260906/20260906_220020/20260906_220020.json`。
- C: `work_dirs/yopo_yolo26s_n_raw_features_refine_20260906/20260906_220609/vis_data/scalars.json`。
- 各本学習work_dirの`guard_report.json`がexit code・driver peakを記録する。

## 6. 完了確認

以下は実行済み内容の公開用要約。試験は56 passed、採用モデルの独立評価はexit0。

- [x] TR-RAW/ARCH: raw-only転送、depth stem sum、三段feature parityとstrict loadを確認した。
- [x] TR-TRAIN: gate/resume、Bの設定early-stop、Cの15epoch上限まで実学習した。
- [x] TR-CURRICULUM: C全3候補を同時guardで棄却し、B epoch20を採用した。
- [x] TR-QUALITY: 採用モデルの181枚独立評価でHBB/mAP/ellipse/projection/shared/strict3Dの6指標が学習時と完全一致した。

## 7. 作業記録

注意: 作業開始前は`date "+%Y-%m-%d %H:%M:%S %Z%z"`を確認する。
チェック完了時は記録を更新する。DRY/KISS/SOLIDを守り、暗黙fallbackや
証憑だけを目的にした工程を増やさない。失敗/不採用/未達を成功と混同しない。

| 日時（JST） | 内容 | 結果 |
|---|---|---|
| 09-06 20:11–20:28 | 要求・実装・raw転送 | RGB=s/depth=n、各枝encoderまで転用。48件の初回回帰と12/12 parity成功 |
| 09-06 20:33–20:48 | 容量・未調整gate | B27/26を上限超過で停止、B25を採用。finiteだが精度不足 |
| 09-06 20:55–21:06 | 境界4 leaf調整と再gate | raw両枝不変、train-only。5epoch shared=.2501でFULLへ |
| 09-06 21:21–21:25 | 過去実績から計画改訂 | write/reviewでA→B→Cを固定。旧shape-only/reverse-KLD不採用を反映 |
| 09-06 21:28 | B epoch20 | shared=.3158、今回の主指標best |
| 09-06 21:56–22:02 | B終端・独立評価 | 40epochでpatience4、exit0。e20の1,095 tensor全finite、strict load、主要6指標差0 |
| 09-06 22:03–22:05 | C設定・回帰 | config未存在RED→GREEN、loss/倍率/全層更新不変、56 passed |
| 09-06 22:06–22:29 | C実学習・採否 | 15epoch/810update、exit0。3候補ともshared改善条件未達、B bestを維持 |
| 09-06 22:30以降 | 公開引継ぎ | 本書へraw範囲・比較・知見・再現手順を集約。公開内容に実データ/絶対private path/run transcriptは含めない |
| 09-06 22:40 | 公開前レビュー・品質 | write/reviewで転用範囲・段階・採否・実コマンドを照合、最終PASS。13 Pythonのruff check/format、diff-check成功。private pattern該当0 |
