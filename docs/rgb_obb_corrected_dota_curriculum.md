# Human-corrected DOTAによるRGB-only 2D OBB追加学習

## 1. 目的と非目的

対象は、RGB画像だけを入力し、class scoreと2D oriented bounding box
`(cx, cy, width, height, angle)`だけを出力する独立detectorである。Depth encoder、
3D translation、3D size、3D rotation、CoP/parallel pose headはこの学習に含めない。

学習後のHGNetV2-B2 backboneはYOPOのRGB branchと同型なので、既存の
`RGBBackboneTransferHook`で後段のRGB-Dモデルへ移植できる。一方、standaloneの
`RotatedDeformableDETRHead`とRGB-Dモデルのquery-level OBB auxiliary headは別構造
なので、head全体を暗黙に移植しない。

## 2. データ契約

正本は次のhuman-corrected DOTA packageとする。

```text
/workspace/YOPO_clone/datasets/
  fruit_obb_o2deim-stage3sf-cvat-corrected_
  train1368-box94605_test181-box9034_20260704/
```

修正前のCVAT送付用COCOは95,199/9,126 boxes、修正後DOTAは94,605/9,034
boxesである。追加学習には後者だけを使う。

| split | images | boxes | boxes/image median | max boxes/image |
|---|---:|---:|---:|---:|
| train | 1,368 | 94,605 | 68 | 129 |
| test-as-validation | 181 | 9,034 | 42 | 118 |

全画像は800x600。全103,639 label rowは10列、finite、正面積、class=`tomato`、
difficulty=`0`で、画像/label stem欠損とdecode不能画像は0。CVATの境界物体により
train 1,982 boxes、test 191 boxesはquad頂点が最大約9.3 pxだけ画面外へ出る。
これは回転矩形として有効なので、事前clipで角度を変形せずaugmentation/prediction
境界処理へ委ねる。

testはteacher predictionを人手修正したsplitで、完全に独立なholdoutではない。
ここでのmetricは追加学習checkpointの相対選択に使い、未知環境への汎化性能とは
表現しない。

## 3. 再利用可能なloader契約

新しい汎用`DOTAOBBDataset`を導入し、既存`DOTATomatoDataset`は互換wrapperとして
残す。汎用loaderは次をconfigから選べるものとする。

- `metainfo.classes`: 任意の1クラスまたは多クラス名。
- `img_suffixes`: `.jpg/.jpeg/.png`等の探索順。
- `img_shape`: 既知なら`(height, width)`、`None`なら画像から取得。
- `strict_loading=True`: class不一致、9/10列以外、非finite、退化quad、画像欠損、
  label/image stem不一致、decode不能、宣言shape不一致を開始前に例外化。
- suffixは大文字小文字を区別せず、`jpg`と`.jpg`のどちらの設定も正規化する。
- `strict_loading=False`: 既存datasetのsilent-skip挙動を必要な場合だけ維持。

今回のconfigは`classes=('tomato',)`、`img_shape=(600,800)`、strictを明示する。
既存loaderの既定class `stem`をそのまま使うと全GTが捨てられるため禁止する。

## 4. モデル設定

```text
RGB 800x600
  -> HGNetV2-B2 backbone
  -> ChannelMapper
  -> 4-layer Deformable DETR encoder/decoder
  -> RotatedDeformableDETRHead (1 class, 150 queries)
  -> score + (cx,cy,w,h,angle)
```

`num_queries=150`とする。100 queryではtrain 155画像、validation 9画像でGT数が
query上限を超え、構造的なrecall上限を作る。最大GTは129/118なので150で全画像を
収容できる。既存100-query checkpointは先頭100行を完全保持し、追加50 queryを
seed 3407、learned-query標準偏差の1% noiseで決定論的に初期化する。

backboneはpretrained/既存checkpointを読み、stemだけを凍結する
`freeze_at=0, freeze_stem_only=True`とする。HGNetV2 stage 1--4、neck、transformer、
OBB headは更新対象であり、「RGB branchを追加学習する」を満たす。Normは学習可能。

train/validationとも800x600のidentity `Resize`を明示し、画像geometryを変えず
`scale_factor`を必ず生成する。augmentationはhorizontal/vertical/diagonal flipだけ
から開始し、dense objectを切り落とすcropは導入しない。

## 5. curriculum

### Phase 0: baselineと容量gate

1. 既存RIoU checkpointを100->150 queryへ展開する。
2. corrected validation 181画像で学習前metricを測る。
3. 1 epoch smokeで全loss/grad finite、OOMなし、GT数9,034一致を確認する。

baselineを測定せず追加学習後だけを報告しない。

### Phase 1: linear Rotated IoU adaptation（5 epochs）

human-corrected label geometryへ直接合わせる段階。classification Focal、RBox L1に
加え、overlap lossはlinear Rotated IoUを使う。既存checkpointもRotated IoUで学習
済みなので、loss familyを変えずに補正済みlabelへ適応できる。Hungarian costは
Focal + RBox L1 + KLD/GDとし、初期overlapが低いqueryにもfiniteな対応を与える。
`-log(IoU)`は過去にnon-finite gradientを生じたため使わない。

- batch: 32。batch 20 smokeのpeak 14,646 MiBを根拠に、32 GiB GPUで余裕を
  残しつつthroughputを上げる。32でOOMの場合だけ24へ戻す。
- FP16 AMP、static loss scale 1.0。
- Muon LR `1e-4`、ScheduleFree LR `5e-6`。
- validation/checkpoint: every epoch、`rbbox_mAP_50`でbest保存。

### Phase 2: GWD refinement（15 epochs）

Phase 1 bestから開始し、最終overlapを滑らかなGaussian Wasserstein Distanceへ
切り替える。回転・細長さ・中心ずれをGaussian geometryとして連続的に最適化し、
Rotated IoUの境界付近で生じる不連続性を抑えて仕上げる。assignment costは
両PhaseでGD/KLDのまま維持し、loss切替時にmatching policyまで同時に変えない。

- Muon LR `5e-5`、ScheduleFree LR `2.5e-6`。
- epoch 12でLRを0.1倍。
- validation/checkpoint: every epoch、`rbbox_mAP_50`でbest保存。

Phase 2が悪化した場合はPhase 1 bestをproduction候補として保持し、後段を採用した
ことにしない。

## 6. 評価・採用gate

同じ181画像、IoU=0.50、score threshold=0.05で次を比較する。

| gate | 判定 |
|---|---|
| safety | NaN/Inf/OOM/loader skipが0、GT=9,034 |
| primary | `rbbox_mAP_50`がbaselineを上回る |
| coverage | recallを併記し、query不足による欠落がない |
| geometry | `rbbox_mean_matched_rIoU`を併記する |
| selection | Phase 0/1/2のうちmAP最高を選び、悪化runは不採用と記録 |
| visual | score>=0.05のOBBを少なくとも6画像へ描画し、角度/縮尺を目視確認 |

validation tuning後のscore thresholdを独立test性能として扱わない。最終checkpoint、
prediction、PNGは`/workspace/YOPO_clone/work_dirs`へ保存し、Gitにはloader、config、
test、docsだけを含める。

## 7. 2026-08-25 実行結果

RTX 5090 32 GiB、PyTorch 2.8.0+cu128、uv環境で全curriculumを実行した。
batch 20のGWD smokeはpeak 14,646 MiB、正式batch 32のRotated IoU smokeは
23,297 MiBで、全runにNaN/Inf/OOMはなかった。loaderのNumPy配列list変換警告は
instance bboxを標準listへ正規化して解消し、worker終了待ちはこのcurriculumで
`persistent_workers=False`を明示して解消した。

| 段階 | 採用epoch | mAP50 | recall@.50 | precision@.50 | matched rIoU |
|---|---:|---:|---:|---:|---:|
| 100→150 query baseline | - | 0.0732 | 0.3478 | 0.1161 | 0.6245 |
| Phase 1 Rotated IoU best | 4/5 | 0.0917 | 0.3689 | 0.1228 | 0.6249 |
| Phase 2 GWD best | 15/15 | **0.0960** | **0.3745** | **0.1246** | **0.6270** |

最終mAP50はbaselineから`+0.0228`、相対`+31.1%`。GWD stageはRotated IoU
stageを`+0.0043`上回ったため、ユーザー指定のRotated IoU→GWD順でPhase 2 bestを
採用する。GT件数は全評価で9,034と一致した。

```text
final checkpoint:
  work_dirs/rddetr_tomato_obb_corrected_gwd_stage2/selected_best.pth
  -> best_rbbox_mAP_50_epoch_15.pth
  sha256 b09d4ac57e4f0ee7b5ac8f73858e5861e691ecba98069fc518f98046063886c0
prediction dump:
  work_dirs/rddetr_tomato_obb_corrected_final_eval/predictions.pkl
  sha256 163a3a93d8b6331933b4c95f7470dc981ec871280c495f41d8f214ebfe787fb4
visual evidence:
  work_dirs/rddetr_tomato_obb_corrected_final_eval/overlays_score0p05/
  score >= 0.05、各画像最大150 query、6 PNG + manifest.json
```

可視化は「認識している候補を隠さない」目的でscore 0.05を使うため、各画像で150
queryすべてが描画された。これは精度の主張ではなくcoverage確認用である。定量採用は
上表の同一metric条件で行う。
