# YOPO Gaussian OBB・SO(3) loss／評価改善設計と作業記録

最終更新: 2026-08-26
対象: `rgb-d` / `DINO9DCenter2DPoseHead` / fruit RGB-D 736x512
状態: component実装・同一条件probe・2-phase評価A/B・native 736x512 capacity/1-epoch gate完了、FULL epoch 20でユーザー指定停止

## 1. 結論

今回の改善は、既存bestの検出性能を壊さず、次の三層を別componentとして扱う。

1. raw rotation 6Dの表現有効性を`Stiefel`制約で保証する。
2. 2D compact Gaussian OBBの形状・向きを`KFIoU`で直接最適化する。
3. 3D評価を、従来実装から任意SO(3) cuboidの交差体積に置き換える。

semanticな3D回転loss、2D Gaussian loss、3D OBB IoUを一つの式へ混ぜない。役割と
failure modeが異なるため、設定で独立に有効化し、loss keyも分離する。

同一epoch-85 checkpointからの5-epoch比較では、KFIoU epoch 2だけがAP75とexact 3D
IoU@0.25/@0.50を同時に改善した。Stiefelはfiniteかつ2D APを維持したが、同じepochで
3D/poseを一貫して改善しなかったため、Stage-12にはKFIoUだけを採用する。

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

### 5.2 proof-safe 2-phase評価

all-pairs exactは正しいが、validation 330画像で約307万pairをscalar convex intersectionへ
送るため遅い。そこで、各pairを次の二段階で評価する。

1. broad phase: world軸、prediction OBBの3軸、GT OBBの3軸へ両boxを射影し、それぞれの
   interval overlap積`J_Q`を求める。
2. narrow phase: exact IoUが最小評価閾値へ到達し得るpairだけ、既存のSAT＋intersection
   vertex＋ConvexHullへ送る。

任意の正規直交frame `Q`で真のintersectionは射影overlap直方体に含まれる。したがって、

```math
U = min(V_p, V_g, J_world, J_Rp, J_Rg)
```

はintersection volumeの上界であり、IoUはintersectionに対して単調増加なので、

```math
IoU_exact <= IoU_ub = U / (V_p + V_g - U)
```

となる。raw AABB IoUは分母もAABB体積へ変わるため上界ではなく、枝刈りには使わない。
center distanceが外接球半径和を超えるpairも、安全な数値guard付きで`U=0`にできる。

実装契約:

- exact kernelと同じabsolute/relative geometry toleranceに加え、dot/sum/productへ外向き
  FP64 guardを入れる。
- full evaluatorはexact値を`float32 overlaps`へ格納してからstrict `>`比較する。そのため
  `float32(IoU_ub)`を+inf側へさらに1 ULP広げ、同じ比較領域で候補判定する。
- bottle/bowl/can、handle-hidden mugはpairごとのlocal-y canonicalizationでAABBが変わり得る。
  この対称pairは枝刈りせず全てexactへ送る。
- fast pathで0へ省略するのは、全評価閾値未満と証明されたraw overlapだけである。
  GT/pred matching、AP、pose selectionはall-exactと同一である。
- `NOCSMetric(two_phase_3d_iou=True)`を既定とし、`False`で診断用all-exact A/Bへ戻せる。
- `compute_3d_matches`直呼びはraw overlap契約を守るためall-exactを既定とする。

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
| 2-phase NOCS統合 | `yopo/evaluation/metrics/nocs_metric.py` | `two_phase_3d_iou=True/False` |
| probe共通設定 | `configs/yopo/nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_stage11_geometry_loss_probe5_base.py` | LR 3e-6 / 5 epoch |
| KFIoU probe | `configs/yopo/nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_stage11_kfiou_probe5.py` | weight 1.0 |
| Stiefel probe | `configs/yopo/nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_stage11_stiefel_probe5.py` | weight 0.05 |
| native geometry guard | `yopo/datasets/transforms/identity_geometry.py` | `AssertIdentityImageGeometry` |
| native 736x512 base | `configs/yopo/nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_stage12_native736x512_base.py` | loss-neutral |
| selected KFIoU base | `configs/yopo/nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_stage12_native736x512_kfiou_base.py` | KFIoU w1 / LR 3e-6 |
| selected capacity smoke | `configs/yopo/nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_stage12_native736x512_kfiou_capacity_smoke.py` | b16 / 2 iter |
| selected 1-epoch gate | `configs/yopo/nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_stage12_native736x512_kfiou_gate1.py` | valあり / checkpointなし |
| selected FULL | `configs/yopo/nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_stage12_native736x512_kfiou_full.py` | 50 epoch / val 5 |

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
- 2-phase、float32 threshold境界、native/config契約を含む独立再検証は92 passed、Ruff成功。

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
`best_3d_iou_0.50_epoch_2.pth`を採用する。

### 2026-08-26: Stiefel独立5-epoch probeと採否

全epochがfinite、exit 0。最も3Dが良いepoch 2と、最もAP50:95が良いepoch 3を比較した。

| 選択 | epoch | AP50:95 | AP50 | AP75 | exact IoU@0.25 | exact IoU@0.50 | pose 10deg/10cm |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0 | 0.1807452 | 0.4755974 | 0.0934972 | 0.0130704 | 0.0009251 | 0.1688508 |
| 3D best | 2 | 0.1806993 | 0.4758989 | 0.0933582 | 0.0131003 | 0.0010000 | 0.1638137 |
| AP best | 3 | 0.1813078 | 0.4781244 | 0.0923864 | 0.0127292 | 0.0008735 | 0.1702460 |

epoch 2の3D改善は小さく、AP75とposeが低下した。epoch 3は2D APとposeが伸びる一方、AP75と
3D IoUが低下した。同条件KFIoU epoch 2がAP75と3Dを同時に改善しているため、Stiefelは今回の
最終構成へ採用しない。両component通過条件を満たさないため、combined probeも行わない。

### 2026-08-26: 2-phase exact評価の全validation A/B

保存済みprediction artifact
`stage10_fair_eval_control_epoch1/raw_predictions.pkl`（SHA-256
`0269270d521ac6e9dbe3652bfa75639abb230204e177587963bcaf95ed88115b`）を両経路で共有した。
score threshold 0.2、2D NMS IoU 0.5、8 worker、3D threshold 0.10/0.25/0.50/0.75を固定した。

| 項目 | all-exact | 2-phase |
|---|---:|---:|
| images / total pairs | 330 / 3,074,143 | 330 / 3,074,143 |
| exact narrow-phase pairs | 3,074,143 | 6,022 (0.1959%) |
| matching wall time | 184.4966 s | 3.1199 s |
| speedup | 1.0x | 59.14x |

- 全画像のGT match、prediction match、score-sort indexはbitwise一致した。
- 省略されたbelow-threshold overlap entryは7,760件。IoU@0.10を超える誤枝刈りは0件で、
  省略entryの最大all-exact IoUは0.0995931だった。
- match digestは`3c766eb992b4e916300f4598d10593eb055fd535ad884ce4d83afb7c3939b4bc`。
- 0.10/0.25/0.50/0.75それぞれについてfloat32丸めmidpointの直下・一致・直上を回帰試験した。

### 2026-08-26: native 736x512 geometry contract

- resize/padを除き、identity guardでRGB-Dを`(B,4,512,736)`へfail-fast固定した。
- 実train/validation sampleでHBB、OBB Gaussian、intrinsic、center2D、rotation、translation、
  size、z、Tが不変であることを確認した。
- 新規7 tests、関連込み23 tests、Ruffが成功した。
- capacityはb16・2 iterationで正常終了した。RTX 5090の総VRAM 32,607 MiBに対し、
  `nvidia-smi`のprocess peakは30,516 MiB（29.80 GiB）、MMEngineのpeak allocated表示は
  28,119 MiBだった。全lossとgradient normはfiniteでOOMもなく、目標29.5--31.0 GiB内なので
  b16を採用した。残り約2.04 GiBを安全余白とし、b17の追加試行は行わない。
- FULLは初期50 epoch、val interval 5、LR 3e-6、AP50:95の20-epoch停滞で早期終了する。
  改善継続時だけ25 epochずつ、最大100までexact-resumeする。
- val evaluatorは`two_phase_3d_iou=True`を明示し、corrected exact metricを維持したまま
  broad-phaseで高速化する。

### 2026-08-26: native 736x512 1-epoch quality gate

Stage-11 KFIoU epoch 2 model-only checkpointを同じ初期値にし、batch 16、LR 3e-6で1 epochだけ
学習した。75/75 iterationと330画像validationをexit code 0で完走し、全loss/gradientはfiniteだった。
validation時のexact narrow phaseは2,680,428 pair中5,278 pair（0.20%）だけを計算した。

| metric | old-resolution selected seed | native gate epoch 1 | delta |
|---|---:|---:|---:|
| AP50:95 | 0.180867 | 0.1930 | +0.0121 |
| AP50 | 0.476129 | 0.4851 | +0.0090 |
| AP75 | 0.094018 | 0.1102 | +0.0162 |
| exact 3D IoU@0.10 | 0.033796 | 0.0330 | -0.0008 |
| exact 3D IoU@0.25 | 0.013144 | 0.0126 | -0.0005 |
| exact 3D IoU@0.50 | 0.001009 | 0.0008 | -0.0002 |
| pose AP 10 degree / 5 cm | 0.1382 | 0.1497 | +0.0115 |
| pose AP 10 degree / 10 cm | 0.1659 | 0.1832 | +0.0173 |

2D APとpose APは明確に改善し、3D IoUの低下は1 epochのgeometry transitionとして小さい。
lossも安定して低下したためFULL開始gateをpassとする。gateは品質判定専用でcheckpointを重複保存せず、
FULLも同じStage-11 KFIoU epoch 2 model-only checkpointから独立に開始する。

FULLは2026-08-26 14:26 UTCに開始した。artifact rootは
`/home/kasm-user/Desktop/YOPO_clone_artifacts/work_dirs/stage12_native736x512_kfiou_full_b16_from_stage11e2`
である。`/workspace`の空きが241 MiBしかないため、checkpointを含む新規出力は空き約93 GiBの
Desktop側に限定した。起動直後にmodel-only checkpointのloadとfresh optimizerをログ確認し、
epoch 1の最初の30 iterationはcapacity/gateと同じfinite loss軌跡を再現している。
長時間側のprocess VRAMは31,498 MiBで安定したが、display等を含むdevice空きは約603 MiBまで
下がるため、FULL中はほかのGPU workloadを併用しない。OOM時のみbatch 15へ下げ、LR 3e-6は
据え置く。学習には未使用だった継承`test_pipeline`定義の640x480 resizeも除去し、下流の推論configが
TestLoopを有効化した場合もtrain/validationと同じnative 736x512 identity geometryを再利用できるようにした。
Stage-12 FULLのTestLoop自体は従来どおり無効であり、学習中のvalidationには影響しない。

最初の実運用gateであるepoch 5 validationは全主要指標でnative 1-epoch gateを上回った。

| metric | native gate epoch 1 | FULL epoch 5 | delta |
|---|---:|---:|---:|
| AP50:95 | 0.1930 | 0.195789 | +0.002789 |
| AP50 | 0.4851 | 0.490007 | +0.004907 |
| AP75 | 0.1102 | 0.112533 | +0.002333 |
| exact 3D IoU@0.10 | 0.0330 | 0.035299 | +0.002299 |
| exact 3D IoU@0.25 | 0.0126 | 0.013923 | +0.001323 |
| exact 3D IoU@0.50 | 0.0008 | 0.000970 | +0.000170 |
| pose AP 10 degree / 5 cm | 0.1497 | 0.150016 | +0.000316 |

OOM、Traceback、NaN/Infはなく、MMEngine peak allocatedは28,479 MiBだった。`epoch_5.pth`、
`best_AP50_95_epoch_5.pth`、`best_AP75_epoch_5.pth`、`best_3d_iou_0.25_epoch_5.pth`の
生成を確認した。unused test pipeline修正後のfocused config/geometry testは11件すべて成功した。

epoch 10でも改善が継続した。epoch 5比でAP50:95は+0.003976、AP75は+0.003542、exact
3D IoU@0.25は+0.000597、pose 10 degree / 5 cmは+0.007573だった。3D IoU@0.50だけ
-0.0000032だが、閾値の高い希少matchで生じた無視できる規模の変動である。

| metric | FULL epoch 5 | FULL epoch 10 | delta |
|---|---:|---:|---:|
| AP50:95 | 0.195789 | 0.199764 | +0.003976 |
| AP50 | 0.490007 | 0.497277 | +0.007270 |
| AP75 | 0.112533 | 0.116075 | +0.003542 |
| exact 3D IoU@0.10 | 0.035299 | 0.037048 | +0.001749 |
| exact 3D IoU@0.25 | 0.013923 | 0.014520 | +0.000597 |
| exact 3D IoU@0.50 | 0.000970 | 0.000967 | -0.000003 |
| pose AP 10 degree / 5 cm | 0.150016 | 0.157588 | +0.007573 |

exact narrow phaseは2,724,655 pair中5,599 pair（0.21%）。`epoch_10.pth`と3種の
epoch-10 best checkpointへ正常更新され、NaN/Inf/OOM/例外はない。Stage-12本体processは
31,498 MiBで安定した。監視中、一時的な別CPU指定processが498 MiBのCUDA contextを確保して
device peak 32,012 MiBまで上げたが終了済みであり、FULLには介入していない。

### 2026-08-26: LR再確認とepoch 15 exact-resume

ユーザー指摘を受け、LRを再監査した。現在の`3e-6`はscratch学習向けではなく、成熟したStage-11
bestを全parameter ScheduleFreeでfine-tuneするための値である。過去の同系3D warm-start単一因子
gateでは、`5e-5`で3D IoU@0.50が0.1690から0.0952、`1e-5`でも0.1266へ低下し、`1e-6`だけ
0.1692を維持した。別のfresh runではScheduleFree `2.5e-3`がepoch 13でHungarian cost非finiteに
なった。したがってMuonのscratch向け`1e-3`やScheduleFree `1e-4`を現在の全層warm-startへ
そのまま移植せず、改善中の`3e-6`を維持する。

会話turnの中断に伴い、元のunified processはepoch 15 validationのexact 3D集計中に外部終了した。
Python/CUDA OOM/NaN/Inf/kernel OOMの証跡はなく、2D tableまでは出たが最終metric dictは未生成なので
epoch 15 validationを正式値に使わない。validation前に保存された`epoch_15.pth`は497,643,043 bytes、
`state_dict/optimizer/param_schedulers/message_hub/meta`を持ち、metaはepoch 15、iter 1125だった。
tmux `yopo_stage12_full_resume`でこのcheckpointからexact-resumeし、ログの`resumed epoch: 15,
iter: 1125`、epoch 16のLR 3e-6、finite loss/gradientを確認した。以後の正式な比較点はepoch 20とする。

epoch 20 validationは完了し、ユーザー指示により直後に停止した。epoch 21は約30 iterationだけ
RAM上で更新されたがcheckpoint保存前なので不採用であり、SIGINT後にGPU解放を確認した。

| metric | FULL epoch 10 | FULL epoch 20 | delta |
|---|---:|---:|---:|
| AP50:95 | 0.199764 | 0.202231 | +0.002467 |
| AP50 | 0.497277 | 0.504579 | +0.007302 |
| AP75 | 0.116075 | 0.118236 | +0.002162 |
| exact 3D IoU@0.10 | 0.037048 | 0.038728 | +0.001679 |
| exact 3D IoU@0.25 | 0.014520 | 0.015142 | +0.000622 |
| exact 3D IoU@0.50 | 0.000967 | 0.001069 | +0.000102 |
| pose AP 10 degree / 5 cm | 0.157588 | 0.153256 | -0.004332 |

2D APとexact 3D overlapはすべてepoch 10を上回った。一方pose APはepoch 10の方が良く、用途別に
checkpointを使い分ける余地がある。epoch 20でAP50:95/AP75/3D IoU@0.25の全bestが更新されたため、
標準採用品はepoch 20とする。resumable checkpointのSHA-256は
`515deacba10da10526d9159afa0fa5f351fcb8ba86cb83d94902fa00ef18912e`。詳細metric、best path、
各checksumはartifact rootの`stage12_epoch20_summary.json`へself-containedに保存した。

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
- [x] 同一bestからKFIoU / Stiefelを独立probeする。
- [x] gate判定を行い、KFIoU epoch 2を採用、Stiefelを非採用とする。
- [x] proof-safe 2-phase評価を実装し、全330画像でall-exactとのmatching完全一致を確認する。
- [x] 採用bestと2-phase A/B結果をDeepSeek V4 Flash版claude-memへ追記する。

## 10. 最終FULLのDefinition of Done: ネイティブ736x512・高VRAM占有

単独probeの採否を確定した後、選んだcheckpointを初期値として次を満たす追加FULL学習を行う。
ユーザーの最新指定は元データと同じ736x512である。設定上の`scale`だけでなく、modelへ渡る
tensor shapeをDoDの判定対象とする。

- [x] dataloader＋preprocessor後のtrain/validation tensor shapeを実測し、`(B,4,512,736)`を
  自動testで保証する。
- [x] 不要なresize/paddingで画素・座標を変えず、RGB、raw depth、HBB、2D OBB Gaussian、3D pose
  annotation、camera intrinsicが元736x512データと同じ座標系にあることをtestする。
- [x] RTX 5090でbatch 16の2-iteration capacity smokeを行い、finite forward/backward、OOMなしを
  確認した。最大process VRAMは30,516 / 32,607 MiB、MMEngine peak allocatedは28,119 MiB。
- [x] 現行640x445・batch 20より画素数が約32%増える条件でbatch 16を採用した。約2.04 GiBの
  安全余白を残しつつ目標使用量29.5--31.0 GiB内に入ったため、追加のbatch探索は打ち切った。
  LRは成熟checkpoint用の保守値3e-6を維持し、次に1-epoch品質gateで安定性を確認する。
- [x] KFIoU/Stiefel単独gateを通ったcomponentだけを最終configへ含める。今回はKFIoUだけを採用し、
  Stiefelとcombinedは含めない。
- [x] corrected exact 3D OBB IoUの2-phase経路をall-exactと全件A/Bし、metric-equivalenceを保証する。
- [x] corrected exact 3D OBB IoU、AP50:95、AP75、pose APを完了した各正式評価時に保存し、旧3D metric値とは
  比較しない。
- [x] FULLのbest/last checkpoint、resolved config、metrics JSON、VRAM実測、起動コマンド、採否理由を
  artifact root、本書、DeepSeek版claude-memへ記録する。
