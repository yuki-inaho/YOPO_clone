# NMS-Free RGB-D Query Selection Architecture

**作成日:** 2026-08-25
**対象:** `YOPO_clone` / custom-fruit RGB-D / Q150 / 1 class
**状態:** architecture decision + v1/v2 ablation completed

## 1. 目的

2D BBOX、2D OBB、depth、3D size、rotationを同じquery identityへ保持しつつ、
推論時NMSなしで各物体を一意に出力できるscore/assignment契約を定義する。

本書が解く問題は「可視化で箱が少ない」こと自体ではない。現checkpointでは
confidence `0.5`のrecallが低く、`0.2`ではほぼ全150 queryが残る。従って、
固定thresholdやNMSだけではなく、queryのobjectnessと局在品質を学習時に整合
させる必要がある。

## 2. 参照資料の位置付け

- ユーザー指定資料: [YOLO26: An Analysis of NMS-Free End to End Framework](https://arxiv.org/abs/2601.12882)
- 公式資料: [Ultralytics YOLO26](https://arxiv.org/abs/2606.03748)
- 公式training recipe: [YOLO26 Training Recipe](https://docs.ultralytics.com/guides/yolo26-training-recipe)
- set predictionの基礎: [DETR](https://arxiv.org/abs/2005.12872)
- 現headの基礎: [DINO](https://arxiv.org/abs/2203.03605)
- score/localization alignment: [Rank-DETR](https://arxiv.org/abs/2310.08854)
- quality target: [Generalized Focal Loss](https://arxiv.org/abs/2006.04388)

`2601.12882`は公式実装論文ではなく、公開情報を整理した二次分析である。その
ためSTAL、ProgLoss、MuSGDの記述をそのまま現repoの仕様とはしない。公式資料で
確認できる次の原則だけを設計入力にする。

1. inference用one-to-one headはNMSなしで使う。
2. one-to-many headはtraining-only auxiliaryとして分離する。
3. classification confidenceとbox qualityを揃える。
4. training-only経路をexport/inference graphへ残さない。

## 3. 現行データ・query契約

| split | images | objects | median objects | p95 | max | Q150超過 |
|---|---:|---:|---:|---:|---:|---:|
| train | 300 | 25,004 | 83.0 | 119.0 | 147 | 0 |
| val | 50 | 3,894 | 80.5 | 106.75 | 115 | 0 |

Q150による理論recall上限はtrain/valとも`1.0`である。従って、現時点でquery数を
増やしても、confidence過密やduplicateの根因は解消しない。

現行headは次の契約を持つ。

```text
150 decoder queries
  -> one-to-one Hungarian assignment
  -> sigmoid FocalLoss (fruit/background implicit)
  -> bbox / center / OBB / z / size / rotation per query
  -> top-150 inference output
```

これは構造上すでにDETR型NMS-freeである。問題はNMSの欠如ではなく、一意性と
順位を成立させるscore supervisionが弱いことである。

## 4. 2026-08-25のscore診断

最終teacher-free dumpの7,500 queryを、2D IoU Hungarian、IoU `>=0.5`で
positive/negativeへ分けた。

| 指標 | 実測 |
|---|---:|
| matched positive query | 2,461 |
| remaining query | 5,039 |
| positive score median | 0.4381 |
| negative score median | 0.3946 |
| positive/negative score AUC | 0.6819 |
| scoreとbest 2D IoUのSpearman相関 | 0.4794 |

正例と負例のscore分布が大きく重なっている。`0.5`は高精度側だけを残すが、
全GTに対するrecallを`0.1153`まで落とす。`0.2`はrecall `0.6320`を維持するが、
7,488 queryを残してprecisionは`0.3287`である。

診断用class-agnostic 2D NMSの結果は次のとおりである。

| confidence | NMS IoU | mean kept/image | precision | recall |
|---:|---:|---:|---:|---:|
| 0.2 | none | 149.76 | 0.3287 | 0.6320 |
| 0.2 | 0.3 | 81.02 | 0.5771 | 0.6004 |
| 0.2 | 0.5 | 87.14 | 0.5485 | 0.6138 |
| 0.2 | 0.7 | 100.92 | 0.4822 | 0.6248 |
| 0.5 | none | 13.96 | 0.6433 | 0.1153 |

val GT同士でIoU `>0.5`のpairは0なので、現valではNMS 0.5のdistinct-object
抑制リスクは小さい。ただしNMS 0.5でもrecallは`0.6320 -> 0.6138`へ低下する。
NMSは現checkpointの可視化/暫定運用には有効だが、最終architectureにはしない。

## 5. NMS-free成立条件

query `q`のfruit scoreを `p_q`、2D boxを `b_q`、対応GTを `g_j`とする。
NMS-free inferenceには次の三条件が必要である。

### 5.1 Coverage

各GTへ少なくとも1 queryが十分な局在品質で到達する。

```text
for every g_j, exists q: IoU(b_q, g_j) >= tau_loc
```

### 5.2 Uniqueness

Hungarianで選ばれた1 queryだけをpositiveとし、同じ物体を表すunmatched duplicate
はbackground target 0とする。推論時NMSではなくtraining lossで重複を抑える。

### 5.3 Quality-aligned ranking

binary target `1`だけでは、局在IoU `0.51`と`0.90`を同じscoreへ押す。そこで
positive targetをdetached aligned IoUとする。

```text
y_q = IoU(stopgrad(b_q), g_j)  if q is Hungarian-positive
y_q = 0                         otherwise
```

classification logit `z_q`にはQuality Focal Lossを適用する。

```text
L_quality(q) = BCEWithLogits(z_q, y_q) * |sigmoid(z_q) - y_q|^beta
```

これにより、正確なpositiveは高score、粗いpositiveは中score、unmatched duplicateと
background queryは0へ学習される。quality targetはdetachし、classification lossが
bboxを高scoreにするためだけに不正変形する経路を作らない。

## 6. 採用architecture: QA-O2O v1

最初のablationでは新しいCNN detectorやYOLO headを追加しない。現DINO decoderの
one-to-one構造を保持し、classification supervisionだけをquality-alignedへ変える。

```text
RGB-D backbone / encoder / DINO decoder
                  |
                  +-> bbox b_q -----------+
                  |                        |
                  +-> class logit z_q      +-> Hungarian one-to-one
                  |                        |
                  +-> center/OBB/z/size/R  |
                                           v
                         y_q = detached aligned 2D IoU or 0
                                           |
                                           v
                                  Quality Focal Loss

inference: sigmoid(z_q) -> threshold/top-k -> same query's 2D/OBB/3D outputs
           no NMS
```

この変更はparameter shapeを変えないため、最終checkpointからstrict-compatibleに
再開できる。matching costは最初のablationでは既存FocalLossCostを保持し、assignment
とlossを同時変更しない。

## 7. 2D OBB・3Dとの責務境界

1. query生存判定は2D quality-aligned scoreだけで行う。
2. OBB GWDはHungarian-positiveだけを教師とする。
3. background queryのOBB covarianceは未定義なので、selectionへ使わない。
4. retained query indexを変換せず、同じindexの`bbox/center/OBB/T/size/rotation`を返す。
5. 3D lossはpositive queryだけに適用し、score targetへ3D IoUを初回から混ぜない。

この境界により、2D object existence、2D orientation、3D geometryの失敗を別々に
診断できる。

## 8. Training PDCA

### A. time-control

最終checkpointから既存FocalLossのまま同epoch数を継続する。

### B. QA-O2O

同じcheckpoint、seed、optimizer、epoch、全geometry lossで、FocalLossだけを
QualityFocalLossへ変更する。

最初は5 epochで比較し、各epochでteacher-free dumpを作る。NMSは評価にも推論にも
入れず、診断表だけでNMS counterfactualを残す。

## 9. 採用gate

BはAに対して次を同時に満たす場合だけ採用する。

### 9.1 NMS-free score gate

- positive/negative score AUC `>= max(0.75, A)`
- score--best-IoU Spearman `>= max(0.55, A)`
- conf 0.2、NMSなしprecision `>= 0.45`
- conf 0.2、NMSなしrecall `>= A - 0.01`
- NMSなしrecallとNMS 0.5 recallの差 `<= 0.01`

### 9.2 Detection/3D維持gate

- AP50 `>= A`
- 2D IoU Hungarian recall@0.5 `>= A - 0.01`
- 3D IoU@0.50/.75 `>= A - 0.005`
- pose 10deg/10cm `>= A - 0.005`
- matched SO(3) median/p75を悪化させない
- invalid/NaN/Inf/OOM 0

score calibrationだけ改善して3D query identityを壊すrunは不採用とする。

## 10. v1失敗時だけ行うv2

QA-O2Oだけでcoverageが改善しない場合、YOLO26のdual-head原則をDINOへ適合させる。

```text
shared decoder hidden states
   +-> one-to-one quality head -> inference/export
   +-> one-to-many auxiliary head -> training only, export時削除
```

auxiliary headはGTごとtop-k queryへ2D class/bbox教師を与え、shared representationの
coverageを改善する。one-to-one headとparameterを共有せず、同一logitへpositiveと
negativeの矛盾した教師を与えない。auxiliary weightは後半で0へ減衰する。

STALは現Hungarianが全GTを必ず割り当て、Q150も全GTを収容できるためv1には不要で
ある。MuSGDもscore misalignmentの単一原因ではないため同時変更しない。

## 11. Rollbackと成果物契約

- 既存最終checkpointは上書きしない。
- QA-O2Oは独立config/work_dirへ保存する。
- inference dumpにはselection前の全Q150を保持し、threshold別診断を再計算可能にする。
- 可視化は`raw all`、`NMS counterfactual`、`NMS-free selected`を別directoryにする。
- 失敗runもconfig、log、checkpoint、report hashを保持する。

## 12. 事前実験時点の結論

現モデルへ必要なのはNMS moduleの常設ではない。DINOのone-to-one assignmentを
活かしながら、binary fruit confidenceをlocalization-aware quality scoreへ変える
ことが第一変更である。2D NMS 0.5は比較対照と可視化にだけ使用し、QA-O2Oがgateを
通ればdeployment pathから除外する。

## 13. 5 epoch A/B実測

全runを同じ最終checkpoint、seed `3407`、optimizer、geometry loss、5 epochから開始
した。推論はteacher-freeで全Q150を保存し、NMSなしの値を正とした。

| run | AP50 | IoU@.50 | IoU@.75 | pose 10deg/10cm |
|---|---:|---:|---:|---:|
| A: binary Focal time-control | **0.3520** | 0.5143 | **0.1653** | 0.3491 |
| B: aligned-IoU QFL | 0.3500 | **0.5201** | 0.1618 | **0.3492** |
| C: Focal + training-only O2M weight 0.1 | 0.3510 | 0.5143 | 0.1643 | **0.3498** |

QFLはIoU@.50を`+0.0058`改善したが、AP50を`-0.0020`、IoU@.75を`-0.0035`
低下させた。O2Mはgeometryをほぼ維持したが、AP50を`-0.0010`低下させた。どちらも
事前定義した「Aを落とさない」gateを通過しない。

連続SO(3)ではA/B/Cのmedianがそれぞれ`7.6079/7.6179/7.6073 deg`、p75が
`14.9531/14.9596/14.9798 deg`であり、B/Cとも同時改善ではない。

## 14. score/NMS-free gate実測

| run | positive/negative AUC | score--IoU Spearman | conf .2 precision | conf .2 recall | mean kept | NMS .5 recall差 |
|---|---:|---:|---:|---:|---:|---:|
| A: Focal | **0.68124** | 0.48138 | 0.33009 | **0.63508** | 149.84 | 0.01772 |
| B: QFL | 0.67996 | **0.51192** | **0.34229** | 0.63431 | **144.32** | **0.01695** |
| C: O2M | 0.68090 | 0.47926 | 0.33017 | **0.63508** | 149.80 | 0.01798 |

QFLはlocalization順位相関を改善したが、positive/background分離は改善しなかった。
O2M weight 0.1も主headのscore分布を変えなかった。従ってB/Cをproductionへ採用せず、
追加のQFL/O2M組合せやweight sweepも行わない。

## 15. v2実装契約

失敗結果も再利用可能にするため、Cのtraining-only O2M構造はdefault-offで実装した。

1. `o2m_aux_topk=0`が既定で、既存model/checkpoint挙動は不変。
2. 有効時だけ主headとは別の`o2m_cls_branch/o2m_reg_branch`を生成する。
3. Hungarian seedでquery capacity内の全GTへ最低1 queryを与える。
4. 2D L1+IoU shortlistからGTごと最大2 queryまで、query重複なしで追加する。
5. auxiliary lossは最終decoder hidden stateへだけ戻し、main predictor parameterへは戻さない。
6. `forward`、`predict`、teacher-free dumpは主one-to-one headだけを使う。
7. smokeのweight 0.5は約5.1 sec/iterかつ3Dが揺れたため不採用。shortlist化とweight
   0.1で約1.4 sec/iter、peak約24.3 GiBへ戻した。

この構造は公式YOLO26の「one-to-manyはtraining-only」という責務境界を満たすが、
今回のデータで性能gateを通ったことを意味しない。

## 16. 最終採否と運用境界

- production候補はAのFocal5 checkpoint。既存DINO one-to-one推論なので構造上NMS-free。
- ただしscore分離gateは未達で、conf `.2`の実用的なNMS-free selected setは未完成。
- 2D NMS `.5`は診断/可視化fallbackに限る。同じquery indexの2D/OBB/3Dを一括保持し、
  3D単独NMSや再indexは行わない。
- `render_rgbd_training_result_overlays.py --all-predictions --score-threshold .2
  --nms-iou-threshold .5`でこのfallbackを再現できる。
- 次の学習仮説はlossの追加ではなく、positive/unmatched logit分布を直接分離する
  rank-aware one-to-one objectiveまたはencoder proposal rankingの再設計とする。

主要成果物:

| artifact | SHA-256 |
|---|---|
| Focal5 checkpoint | `ceb68bf6d70575930eab83f6936b04a22a84033b253dc7316e499308ce7e754b` |
| QFL5 checkpoint | `b792ab4410cf098573c32db5c2571a12d65a1646607c350a68cefb2502bbc69e` |
| O2M5 checkpoint | `4f4461c616ffe293ee09469678364a959462cf6af15d86eac3038adfb9205f25` |
| query report | `324a40973efcfd968c27cbce2e1fe894d153b84d00e881648263e3f47930083d` |
| Focal/QFL rotation report | `c9b0efd46933ba82a3df3f659d6d749be24cbe85d35cb137520b36f8259007f7` |
| Focal/O2M rotation report | `370f0d18d6ed3c4defebe2e3f50fedb28995c2954a9b91894af935021a034d35` |
