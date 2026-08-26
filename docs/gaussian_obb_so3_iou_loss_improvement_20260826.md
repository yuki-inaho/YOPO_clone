# YOPO Gaussian OBB・SO(3) loss／評価改善設計と作業記録

最終更新: 2026-08-26  
対象: `rgb-d` / `DINO9DCenter2DPoseHead` / fruit RGB-D 736x512  
状態: 実装・単体試験完了、checkpoint同一条件probeを実行中

## 1. 結論

今回の改善は、既存bestの検出性能を壊さず、次の三層を別componentとして扱う。

1. raw rotation 6Dの表現有効性を`Stiefel`制約で保証する。
2. 2D compact Gaussian OBBの形状・向きを`KFIoU`で直接最適化する。
3. 3D評価を、従来実装から任意SO(3) cuboidの交差体積に置き換える。

semanticな3D回転loss、2D Gaussian loss、3D OBB IoUを一つの式へ混ぜない。役割と
failure modeが異なるため、設定で独立に有効化し、loss keyも分離する。

## 2. 固定baseline

現行FULLはユーザー指示によりepoch 90のtrain中に一度だけSIGINTし、正常終了した。
epoch 85の保存済みbestを共通初期値とする。

| 項目 | 値 |
|---|---:|
| checkpoint | `best_AP50_95_epoch_85.pth` |
| AP50:95 | 0.1807451571 |
| AP50 | 0.4755973816 |
| AP75 | 0.0934972242 |
| exact 3D IoU@0.10 | 0.0333094954 |
| exact 3D IoU@0.25 | 0.0130703658 |
| exact 3D IoU@0.50 | 0.0009250595 |
| exact 3D IoU@0.75 | 0.0000004344 |
| pose 10deg/10cm（exact IoU matching） | 0.1688508292 |

artifact root:

```text
/home/kasm-user/Desktop/YOPO_clone_artifacts/work_dirs/
  stage10_2d_anchor_mal_full_schedulefree_fixed_v2/
```

注意: 従来報告していた`3d_iou_*`は、後述の実装不具合により正しい3D OBB IoUではない。
上表の3D値は同じepoch-85 checkpointを新実装で再評価した唯一のbaselineである。pose APも
IoU@0.10 matchingを使うため、旧metricで得たpose値とは比較しない。

## 3. 情報源と適用範囲

### 3.1 Rotated IoU / OpenMMLab

- [IoU Loss for 2D/3D Object Detection](https://arxiv.org/abs/1908.03851)
- [MMDetection3D RotatedIoU3DLoss](https://github.com/open-mmlab/mmdetection3d/blob/1.0/mmdet3d/models/losses/rotated_iou_loss.py)
- [MMCV PR 1854](https://github.com/open-mmlab/mmcv/pull/1854)

論文の3D式とOpenMMLab実装は、7D box
`(x, y, z, w, l, h, alpha)`を対象とする。xy平面のrotated rectangle交差面積とz方向の
overlapを掛けるyaw-only方式であり、YOPOの6D rotationが表す任意SO(3)へ直接流用できない。

一方、次の構成は任意SO(3)へ一般化できる。

- 一方のbox内にある他方のcornerを集める。
- 一方のedgeと他方のfaceの交点を双方向に集める。
- 得られる交差多面体のvertexを重複除去する。
- 3D convex hull volumeをintersection volumeとする。

この一般化は本実装の設計であり、1908.03851がfull SO(3)を提案したという意味ではない。
現環境のMMCV 2.2.0には`diff_iou_rotated_3d`が存在するが、同じyaw-only契約なので、full
SO(3)主評価／主lossとしては用いない。

### 3.2 Gaussian OBB

- [Gaussian representation and 3-D generalization](https://arxiv.org/abs/2209.10839)
- [KFIoU Loss](https://arxiv.org/abs/2201.12558)

現行2D OBB headは`(cx, cy, w, h, theta)`ではなく、
`(cx, cy, sigma_xx, sigma_xy, sigma_yy)`を出す。したがってGaussian集合間距離を使うことが
表現契約に一致する。KFIoUはcenterを含まないため、HBB/center2D lossを必ず併用する。

### 3.3 rotation表現と観測可能性

- [RSAR](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_RSAR_Restricted_State_Angle_Resolver_and_Rotated_SAR_Benchmark_CVPR_2025_paper.html)
- `/home/kasm-user/Downloads/SPAN_full_3D_OBB_formulation_ja.tex`
- `/home/kasm-user/Downloads/ChatGPT-YOPOの定式化とアーキテクチャ.md`

raw 6D frameの直交性と、GTに対するsemantic SO(3)誤差は別問題である。前者はStiefel
constraint、後者は現行`AMPStableRotation3DLoss`で扱う。category symmetryや軸の意味が
明示されていない現データで、任意の24回転や連続回転対称性を推測してlossへ入れない。

## 4. loss設計

### 4.1 raw 6D Stiefel constraint

raw predictionを`a1, a2 in R^3`、`A=[a1 a2] in R^(3x2)`とする。

```math
L_frame = SmoothL1(A^T A - I_2)
```

契約:

- 正規直交な2列では厳密に0。
- Gram--Schmidt後のSO(3) lossを置換しない。
- AMP下でもGram計算はFP32。
- main decoderの推論に使うraw streamだけに適用する。
- chain modeではCoP rotation、parallel/auxiliary modeではparallel rotationを対象とする。
- DN、encoder、使われないaux-chainへ暗黙に適用しない。
- `loss_rotation_frame`としてsemantic rotation lossと別に記録する。

既知の限界:

- raw frameが完全な0では勾配も0になる。
- 2列が完全に同一だと対称な勾配になる。
- 新規学習ではidentity相当の非平行biasが望ましいが、既存checkpoint継続時にbiasを変更しない。

### 4.2 compact 2D Gaussian KFIoU

pred/target covarianceを`Sigma_p, Sigma_t`とし、Kalman fusion covarianceを

```math
Sigma_cap = (Sigma_p^-1 + Sigma_t^-1)^-1
```

とする。逆行列を直接作らず、

```math
det(Sigma_cap) = det(Sigma_p) det(Sigma_t) / det(Sigma_p + Sigma_t)
```

をCholesky log-determinantで計算する。2D volume定数は比で相殺される。raw KFIoUは同一
covarianceでも最大`1/3`なので、

```math
S_KF = 3 * V_cap / (V_p + V_t - V_cap)
L_KF = 1 - clamp(S_KF, 0, 1)
```

と正規化し、同一covarianceのlossを0にする。

契約:

- centerは有限性のみ検証し、lossへ含めない。
- HBB L1/GIoU、center2D lossを維持する。
- SPD/finiteをpositive pairだけ検証する。
- invalid positiveは既定でfail-fastする。
- AMP下のgeometryはFP32、FP64 inputはFP64を維持する。
- 現行`GaussianGWDLoss`との同時加算ではなく、isolated probeでは置換する。

### 4.3 semantic SO(3)

現行`AMPStableRotation3DLoss`を主回転監督として維持する。Stiefelは表現のconditioning、
KFIoUは2D OBB covarianceのshape/orientation、semantic SO(3)は物体座標軸の姿勢を扱う。
同じ「回転」に見えても監督対象が異なる。

## 5. exact arbitrary-SO(3) 3D OBB IoU

boxをcenter `c`、full size `s > 0`、local-to-world rotation `R in SO(3)`で定義する。

```math
B(c,R,s) = {x | abs(R^T (x-c)) <= s/2}
```

二boxのintersection vertex候補は次だけで十分である。

1. box 1の8 cornerのうちbox 2内部のもの。
2. box 2の8 cornerのうちbox 1内部のもの。
3. box 1の12 edgeとbox 2の6 faceの有効交点。
4. box 2の12 edgeとbox 1の6 faceの有効交点。

候補をtolerance付きでdeduplicateし、full-rankなら`scipy.spatial.ConvexHull`で体積を求める。
face/edge/point contactは体積0とする。

```math
IoU_3D = V_intersection / (prod(s_1) + prod(s_2) - V_intersection)
```

数値契約:

- pairごとに一方のcenterへ再中心化し、大きなworld座標での桁落ちを抑える。
- 15軸SATで非交差pairを厳密に早期棄却してから交差頂点を構成する。
- standalone helperはsize<=0、NaN/Inf、非SO(3) matrixを黙って0にせず拒否する。
- metric adapterだけは有限な非正prediction sideをIoU 0のfalse positiveとして保持する。
  invalid GT、非有限値、non-similarity RTはvalidationをfail-fastする。
- identical、disjoint、既知axis overlap、90度回転、包含、面接触をunit testする。
- independentなhalf-space intersection + LP oracleとrandom caseで比較する。
- 評価用NumPy/SciPy実装であり、training lossとして逆伝播しない。

### 5.1 既存metricの不具合

従来`compute_3d_iou`は、変換後corner shape `(3,8)`へ`amin/amax(axis=0)`を適用していた。
これは3空間軸のAABBを作らず、8 cornerごとの値を作るため、AABB近似としても成立しない。
正しいAABBなら`axis=1`だが、今回はAABB修正を最終解にせず、上記exact OBBへ置換する。

RTの上左3x3にuniform scaleが埋め込まれている既存NOCS契約は、

```text
scale = cbrt(det(RT[:3,:3]))
R = RT[:3,:3] / scale
effective_size = abs(scale) * stored_size
center = RT[:3,3]
```

として明示的に分解する。

## 6. 実験設計

全probeは同じepoch-85 model-only checkpoint、seed、data、selection policyから開始し、別work_dirへ
保存する。新loss以外の変更を混ぜない。

| probe | 変更 | 目的 |
|---|---|---|
| exact re-eval | metricのみ | 正しい3D baselineを固定 |
| KFIoU | compact GWDをKFIoUへ置換 | 2D OBB covarianceの高IoU整合 |
| Stiefel | `loss_rotation_frame`追加 | raw 6D conditioningと3D姿勢改善 |
| combined | 両単独probeがgate通過時のみ | 相補性を確認 |

短期gate:

- 全loss/gradient/metricがfinite。
- AP50:95がbaselineから0.002超悪化しない。
- AP75または正しい3D OBB IoUの少なくとも一方が改善する。
- `loss_rotation_frame`がsemantic rotationを押しのける規模にならない。
- KFIoUの初期loss scaleを現行GWDと比較し、weightを決める。

gate不通過ならそのcomponentを採用せず、長期学習へ進めない。二つの改善を同時に入れて原因を
不明にしない。

## 7. 実装一覧

| component | path | config registry |
|---|---|---|
| compact KFIoU | `yopo/models/losses/gaussian_kfiou_loss.py` | `GaussianKFIoULoss` |
| Stiefel loss | `yopo/models/losses/rotation_6d_stiefel_loss.py` | `Rotation6DStiefelLoss` |
| Stiefel head integration | `yopo/models/dense_pose_heads/dino_9d_center2d_posehead.py` | `loss_rotation_frame` |
| exact 3D OBB IoU | `yopo/evaluation/metrics/oriented_box_iou_3d.py` | evaluation helper |
| probe共通設定 | `configs/yopo/nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_stage11_geometry_loss_probe5_base.py` | LR 3e-6 / 5 epoch |
| KFIoU probe | `configs/yopo/nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_stage11_kfiou_probe5.py` | weight 1.0 |
| Stiefel probe | `configs/yopo/nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_stage11_stiefel_probe5.py` | weight 0.05 |
| native geometry guard | `yopo/datasets/transforms/identity_geometry.py` | `AssertIdentityImageGeometry` |
| native 736x512 base | `configs/yopo/nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_stage12_native736x512_base.py` | loss-neutral |
| native capacity smoke | `configs/yopo/nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_stage12_native736x512_capacity_smoke.py` | b16 / 2 iter |
| native FULL template | `configs/yopo/nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_stage12_native736x512_full.py` | final loss反映前 |

既定値は`loss_rotation_frame=None`なので、既存config/checkpointの挙動とparameter keyを変えない。

## 8. 検証記録

### 2026-08-26: component tests

- loss focused tests 33 passed、head・CoP・projectionを含む関連回帰129 passed。
- `GaussianKFIoULoss`はrandom SPD 512組でexplicit precision-fusion参照と一致し、最大絶対
  誤差`7.77e-16`。
- RTX 5090のFP16/BF16 forward/backwardで両新lossの内部FP32計算とfinite gradientを確認。
- `Rotation6DStiefelLoss`: 新規13 tests、既存AMP rotationとの合同15 passed。
- arbitrary-SO(3) helper・NOCS統合・既存metric回帰は合計62 passed。
- random 250 casesをindependent HalfspaceIntersection + LP oracleと比較し、最大絶対誤差
  `8.88e-16`。
- 実データ3,074,143 pair相当のexact matchingは8 workerで151秒（epoch-85全validation実測）。

### 2026-08-26: epoch-85 corrected baselineとKFIoU校正

- corrected re-evaluationはexit 0。2D APは旧評価と完全一致した。
- corrected exact `3d_iou_0.50=0.0009250595`。旧3D値の過大評価を確認した。
- KFIoU weight 0.25の20-iteration校正では`loss_obb_aux`が約0.010/decoder layerで弱すぎた。
- weight 1.0では約0.040/decoder layer。既存GWD約0.095より小さいが有効な比較強度として
  採用し、同じepoch-85 model-only checkpointから5-epoch probeを開始した。

### 2026-08-26: KFIoU独立5-epoch probe

全epochがfinite、exit 0。baselineは本書2章のcorrected epoch-85値である。

| 選択 | epoch | AP50:95 | AP50 | AP75 | exact IoU@0.25 | exact IoU@0.50 | pose 10deg/10cm |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 0.1807452 | 0.4755974 | 0.0934972 | 0.0130704 | 0.0009251 | 0.1688508 |
| AP best | 3 | 0.1814040 | 0.4781575 | 0.0928645 | 0.0129024 | 0.0008369 | 0.1688701 |
| 3D/AP75 balance | 2 | 0.1808667 | 0.4761290 | 0.0940177 | 0.0131439 | 0.0010090 | 0.1659265 |

epoch 2はAP50:95を維持し、AP75とexact IoU@0.25/@0.50を同時に改善したため単独gateを通過。
epoch 3はAP50系bestだが3Dは下がる。最終FULLのseed候補は原因を明確にするため
`best_3d_iou_0.50_epoch_2.pth`を優先し、Stiefel単独結果と比較して確定する。

### 2026-08-26: native 736x512 geometry contract

- resize/padを除き、identity guardでRGB-Dを`(B,4,512,736)`へfail-fast固定した。
- 実train/validation sampleでHBB、OBB Gaussian、intrinsic、center2D、rotation、translation、
  size、z、Tが不変であることを確認した。
- 新規7 tests、関連込み23 tests、Ruffが成功した。
- capacityはb16から測り、目標peakは約29.5--31.0 GiB。OOMまたは31 GiB超ならb15、
  29.5 GiB未満ならb17を一度だけ試す。
- FULLは初期50 epoch、val interval 5、LR 3e-6、AP50:95の20-epoch停滞で早期終了する。
  改善継続時だけ25 epochずつ、最大100までexact-resumeする。

### 2026-08-26: claude-mem

従来observerは廃止済みモデル設定により8回連続失敗していた。ユーザー指定に従い
OpenRouterの追従alias`~deepseek/deepseek-v4-flash-latest`へ変更した。以降は設計判断、metric
semantic、probe結果、採否だけを節目ごとに記録し、逐次ログを大量保存しない。新sessionで
実モデル`deepseek/deepseek-v4-flash-0731`への解決と観測DB保存をログ確認した。

## 9. 未完了checklist

- [x] 現行FULLを保存済みbestで安全終了した。
- [x] compact KFIoUを独立実装・単体検証した。
- [x] Stiefel lossを独立実装し、headへoptional統合した。
- [x] 任意SO(3) exact OBB IoU helperをoracle検証した。
- [x] corrected IoUをNOCS validationへ接続する。
- [x] epoch-85 bestをcorrected IoUで再評価する。
- [ ] 同一bestからKFIoU / Stiefelを独立probeする。
- [ ] gate判定と採用bestを本書・claude-memへ追記する。

## 10. 最終FULLのDefinition of Done: ネイティブ736x512・高VRAM占有

単独probeの採否を確定した後、選んだcheckpointを初期値として次を満たす追加FULL学習を行う。
ユーザーの最新指定は元データと同じ736x512である。設定上の`scale`だけでなく、modelへ渡る
tensor shapeをDoDの判定対象とする。

- [ ] dataloader＋preprocessor後のtrain/validation tensor shapeを実測し、`(B,4,512,736)`を
  自動testで保証する。
- [ ] 不要なresize/paddingで画素・座標を変えず、RGB、raw depth、HBB、2D OBB Gaussian、3D pose
  annotation、camera intrinsicが元736x512データと同じ座標系にあることをtestする。
- [ ] RTX 5090で2-iteration capacity smokeを行い、finite forward/backward、OOMなし、最大
  allocated/reserved VRAMを記録する。
- [ ] 現行640x445・batch 20より画素数が約32%増えるため、batch 15または16から測定を始める。
  OOMを避ける安全余白を残しつつVRAMを高占有する最大値を採用する。batchを変える場合もLRを
  盲目的に上げず、成熟checkpoint用の保守的LRで短期安定性を先に確認する。
- [ ] KFIoU/Stiefel単独gateを通ったcomponentだけを最終configへ含める。両方通過した場合もcombined
  1-epoch gateを通すまで同時採用しない。
- [ ] corrected exact 3D OBB IoU、AP50:95、AP75、pose APを毎評価時に保存し、旧3D metric値とは
  比較しない。
- [ ] FULLのbest/last checkpoint、resolved config、metrics JSON、VRAM実測、起動コマンド、採否理由を
  artifact root、本書、DeepSeek版claude-memへ記録する。
