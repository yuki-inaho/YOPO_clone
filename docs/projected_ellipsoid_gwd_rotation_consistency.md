# 2D OBB–3D Ellipsoid Projection Consistency for Fruit Rotation

作成日: 2026-08-25 JST
対象: `/home/kasm-user/Desktop/YOPO_clone` の custom fruit RGB-D / Q150 CoP model

## 1. 目的と非目的

### 1.1 目的

既に学習できている2D BBOX、2D center、depth、3D sizeを土台として、予測した3D姿勢を画像へ投影した結果と、観測された2D OBBを幾何的に一致させる。これにより、2D OBBの向き・楕円率が持つ情報を3D rotation branchへ直接逆伝播する。

中心となる制約は次である。

```text
predicted (center, depth, size, rotation)
    -> camera-frame 3D ellipsoid
    -> perspective projection with known K
    -> 2D conic / ellipse
    -> 2D Gaussian (center, covariance)
    -> GWD against observed GT OBB Gaussian
```

これは、3D quadricを2D conicへ射影して画像上の観測と比較する [QuadricSLAM](https://arxiv.org/abs/1804.04011) の射影則と、果実を矩形より楕円として扱う [Ellipse R-CNN](https://arxiv.org/abs/2001.11584) の考え方を、現在の単画像RGB-D detectorへ微分可能な補助lossとして適用するものである。

### 1.2 非目的

- 2D OBBだけから完全な3D rotationを一意に復元することは目的にしない。
- 既存の3D geodesic rotation lossを置換しない。
- 初回ablationではHungarian assignmentへprojection costを追加しない。
- 初回ablationではprojection lossから2D center、depth、sizeへ勾配を流さない。
- 別データで学習したOBB modelの予測をNOCS画像の正解として扱わない。

## 2. 現在の前提と実測済みのデータ契約

### 2.1 画像・カメラ・単位

- RGB/depth画像: 640×480 pixel。
- camera intrinsic:

  ```text
  fx = 443.9066
  fy = 449.1953
  cx = 321.3503
  cy = 230.8687
  ```

- translation/depth/size: metre。
- 2D BBOX/center/OBB: pixel。
- rotation: camera-frame内の右手系3×3行列。network出力は6D表現からGram–SchmidtでSO(3)へ変換する。
- sizeの3成分はlocal frameの3軸に対応する。generator上の軸並びと実pklの対応はprojection診断で全順列を監査し、trainで決めた対応をvalへ固定する。

### 2.2 OBB座標scale

実pklの `obb_cxcywha_rad` は元の800×600座標で、現在のRGBとaxis-aligned bboxは640×480座標である。実サンプルでは

```text
OBB center (581.5217, 377.3438) * 0.8
    = bbox center (465.2174, 301.8750)
```

と一致した。したがって、raw OBBの `(cx, cy, w, h)` はdataset adapterで `(0.8, 0.8)` 倍してから使用する。角度は等方resizeでは不変である。このscaleは暗黙値にせずconfigへ明示し、全train/valでcenter residualを検証する。

### 2.3 既存基準値

- Q150 Stage4 teacher-free AP50: `0.335`
- 2D oracle IoU≥0.50: `0.6091`
- nearest-center median: `2.86 px`
- 3D IoU@0.50: `0.4239`
- translation-only 360°/10cm: `0.5770`
- pose 10°/10cm: `0.0036`

projection lossの目的は最後のrotation依存指標を改善し、前5項目を実質的に維持することである。

## 3. 3D ellipsoidの定義

queryごとに、camera-frame translationを `t in R^3`、rotationを `R in SO(3)`、full axis lengthsを `s=(sx, sy, sz)` とする。semi-axisは

```text
a = s / 2
```

であり、camera-frameのellipsoid shape matrixを

```text
A = R diag(a_x^2, a_y^2, a_z^2) R^T
```

と定義する。ellipsoid表面は

```text
(X - t)^T A^{-1} (X - t) = 1
```

を満たす。

homogeneous primal quadric `Q` の逆行列と比例するdual quadricは、scale不定性を無視して

```text
Q* = [[A - t t^T, -t],
      [    -t^T, -1]]
```

となる。

## 4. 既知Kによる厳密なperspective projection

object poseは既にcamera-frameにあるため、camera projection matrixは

```text
P = K [I | 0]
```

である。QuadricSLAMと同じdual projection則により、画像上のdual conicは

```text
C* = P Q* P^T
   = K (A - t t^T) K^T
```

となる。primal conicは任意scaleを除いて

```text
C = inverse(C*)
```

で得られる。`C` を

```text
C = [[M, m],
     [m^T, c]]
```

と2+1次元へ分割すると、投影楕円のpixel中心 `mu` とsemi-axis covariance/shape matrix `Sigma` は

```text
mu       = - inverse(M) m
c_center = c - m^T inverse(M) m
Sigma    = (-c_center) inverse(M)
```

である。`Sigma` の固有値は投影楕円のsemi-axis lengthの二乗、固有ベクトルは画像面内の軸方向になる。

この式はcuboidの8頂点へmin/maxを取る近似と異なり、ellipsoidのperspective projectionを直接表し、rotationとsizeに対して微分可能である。

## 5. GT 2D OBBのGaussian表現

scale補正・augmentation反映後のGT OBBを

```text
b* = (cx, cy, w, h, theta)
```

とする。2D rotation matrixを `R2(theta)` として、GT Gaussianを

```text
mu*    = (cx, cy)
Sigma* = R2(theta) diag((w/2)^2, (h/2)^2) R2(theta)^T
```

と定義する。

この表現は次に対して不変である。

```text
(w, h, theta) == (h, w, theta + pi/2)
theta         == theta + pi
```

したがって、angle wrap、width/height交換、OpenCV angle conventionの枝分かれをlossへ持ち込まない。

### 5.1 resizeとhorizontal flip

OBBは早い段階で `(mu*, Sigma*)` に変換して保持する。画像座標変換 `x' = D x` に対し、

```text
mu'    = D mu
Sigma' = D Sigma D^T
```

と更新する。

- resize: `D = diag(scale_x, scale_y)`
- horizontal flip: 中心は既存bbox規約と同じ幅基準で反転し、covarianceには `F=diag(-1,1)` を適用する。

これにより、angleを再正規化せずaugmentation後も一貫したtargetを得る。

## 6. Projection GWD loss

予測投影 `(mu, Sigma)` とGT `(mu*, Sigma*)` のGaussian Wasserstein distanceを使う。

```text
d_GWD^2 = ||mu - mu*||_2^2
          + Tr(Sigma + Sigma*
               - 2 (Sigma^(1/2) Sigma* Sigma^(1/2))^(1/2))
```

2×2 SPD matrixではrepo既存 `GDLoss/gwd_loss` と同じtrace/determinant閉形式を使う。object scale差を抑えるため既存GWDと同じ正規化と `log1p` postprocessを用いる。

最終lossは

```text
L_projection = lambda_projection * mean_positive(GWD)
```

とする。初期値は `lambda_projection=1.0` とし、loss実測scaleを確認してからのみ変更する。

## 7. 識別可能性と対称性

### 7.1 単視点の限界

2D ellipseは3D rotationの全自由度を一意に決めない。projectionは

```text
A = R diag(a^2) R^T
```

だけに依存するため、次の変換を区別できない。

- 各local axisの符号反転に対応する180°対称性。
- sizeと軸を同時に置換する変換。
- 等しいsemi-axisを持つ部分空間内の任意rotation。

よってprojection lossは3D geodesic supervisionの代替ではなく、画像観測と矛盾するrotationを減らす補助制約である。

### 7.2 円形・球形の果実

`sx≈sy≈sz` なら `A≈alpha I` となり、projectionはrotationに依存しない。このときrotation gradientが弱くなるのは欠陥ではなく、観測から向きを識別できないという正しい挙動である。

GT OBBでも `w≈h` なら `Sigma*≈beta I` となり、GWDのangle依存は自然に消える。したがって初回実装ではhardなaspect-ratio maskを使わない。診断reportではanisotropy bin別にgradientと誤差を出し、必要な場合だけ連続weightを追加する。

## 8. 勾配の隔離と学習順

projection lossはcenter、depth、size、rotationのすべてで小さくできるため、初回から全branchへ流すと、rotation誤差をsizeの歪みで吸収する可能性がある。

初回ablationではmatched GTの中心・depth・sizeをnuisance geometryとして固定し、予測rotationだけを射影する。

```text
t_for_projection = t_gt
s_for_projection = s_gt
R_for_projection = R
```

これにより、学習初期または更新直後の予測depth/sizeが幾何学的に不正でも、projection lossが正例を黙って捨てたり、rotation以外の誤差を混入させたりしない。projection lossの勾配はrotation chainだけへ流し、2D BBOX/center、depth、sizeは従来lossで学習を継続する。rotation-onlyで改善を確認した後に限り、予測geometryを使う条件やsize joint-gradientを別ablationとして試す。

### 8.1 query-level OBB auxiliary head

direct projectionだけでrotation改善が検出できない場合は、rotation stage feature `h_R` に軽量な3出力headを付ける。出力 `(a,b,c)` から

```text
L = [[softplus(a)+eps, 0],
     [b, softplus(c)+eps]]
Sigma_obb = L L^T
```

とするため、予測covarianceは構成上strict SPDである。GT OBB covarianceは画像幅・高さで正規化し、center項を除いたnormalized GWDを適用する。これは2D center lossの二重計上を避け、楕円のaxis lengthとorientationだけをrotation stage featureへ教師するためである。aux出力はassignment、推論、3D decodeへ使わない。従ってcheckpointにaux parameterが追加されてもteacher-free inference schemaは不変である。

## 9. Assignmentとloss適用範囲

- 既存のclass/BBOX/GIoU/translation/rotationによるHungarian assignmentを維持する。
- assignment後のpositive queryと対応GT OBBにprojection lossを計算する。
- 初回はprojectionをmatching costへ入れない。対応付け自体が変わると純粋なrotation改善を判別できないためである。
- decoder各層のpositive queryへ適用するが、encoder proposalとdenoising queryは初回対象外とする。
- 評価・推論時にaux teacherやGT OBBを必要としない。

## 10. 数値安定性と明示的error contract

全quadric/conic/GWD geometryはautocastを無効にしてFP32で計算する。

各positiveについて次を検証する。

1. `z > max(s)/2 + eps`
2. `det(R) > 0` かつfinite
3. `C*`, `C`, `M`, `Sigma`, `Sigma*` がfinite
4. `Sigma`, `Sigma*` の最小固有値が `eps` より大きい
5. OBBの `w,h > 0`

実装上のclampは、有限な正例を数値演算の定義域へ収めるためだけに使う。
正例が非finite、非SPD、またはカメラ包含条件違反なら、main trainingでは
`fail_on_invalid=True`により件数付き`RuntimeError`で停止し、暗黙にweight 0へ
落として継続しない。diagnosticでは件数と割合もreportし、real-data smokeで
invalidが1件でもあれば本走を開始しない。

## 11. 事前診断gate

学習実装前にtrainで軸対応を決定し、valには固定して次を確認する。

### 11.1 OBB座標gate

- 推定scaleがx/yとも `0.8 ± 1e-4`。
- scale補正後のOBB centerとaxis-aligned bbox centerのp99誤差 `<= 1e-3 px`。
- nonfinite/非正size OBBが0件。

### 11.2 rotation情報gate

同じtranslation/sizeでrotationだけを変え、投影GWDを比較する。

- `paired`: 対応するGT rotation。
- `canonical`: camera anchorに固定したrotation。
- `shuffled`: 別instanceのGT rotationを決定論的に割り当てた対照。

採用条件はvalで同時に

- paired median GWD `<= 0.90 * shuffled median`
- pairedがshuffledより小さいinstance割合 `>= 0.60`
- paired median GWD `<= 0.95 * canonical median`

とする。満たさない場合、raw OBBは3D rotationを拘束する教師として弱いので、projection loss本走は行わず、OBB aux-onlyまたはdepth ROI形状特徴の改善へ戻る。

### 11.3 size-axis対応gate

sizeの全6順列をtrain paired projectionで比較する。generator sourceの並び以外がmedian GWDを10%以上改善し、かつ同じ順列がvalでも最良なら、annotation axis mapping defectとして別途明示修正する。それ以外はstored orderを維持する。valを使って順列を選ばない。

## 12. TDD要件

最低限、次を自動testで固定する。

1. 800×600→640×480のOBB scaleが0.8として復元される。
2. `(w,h,theta)` と `(h,w,theta+pi/2)` が同一Gaussian/GWDになる。
3. 球を光軸上へ置いた解析解とdual-quadric projectionが一致する。
4. anisotropic ellipsoidではrotationへfiniteかつnon-zero gradientが届く。
5. sphereではrotation gradientが0または数値許容以下になる。
6. matched GT center/depth/sizeを用いたとき、projection gradientがrotationだけへ届く。
7. resize/flip後のGaussian targetが明示的な点変換と一致する。
8. teacher-free推論の出力schemaと数値がprojection loss追加前から変わらない。
9. 正例の射影またはtarget Gaussianが非finite/非SPDなら件数付き例外で停止する。

## 13. PDCA実験行列と採用条件

全候補はQ150 Stage4 bestから同じepoch数、batch、seed、validationで比較する。

| ID | 条件 | 目的 |
|---|---|---|
| A | 現Stage4をそのまま5 epoch継続 | 時間経過だけのcontrol |
| B | projection GWD、rotation-only gradient | 主仮説 |
| C | B + query-level OBB aux head | OBB表現学習の追加効果 |
| D | B + size joint-gradient | B成功後だけ実施 |

候補の最低維持gate:

- AP50 `>= 0.328`（現best 0.335の約98%）
- translation-only 360°/10cm `>= 0.548`（現0.577の95%）
- 3D IoU@0.50 `>= 0.4239`
- invalid projection 0件、NaN/Inf/OOM 0件

主なrotation改善判定は、次の3条件を同時に満たすこととする。

- pose 10°/10cmがcontrol Aより改善する。
- 3D IoU@0.50または0.75がcontrol A以上である。
- matched-pair rotation errorのmedian/p75がcontrol Aより低い。

同率なら構造が単純なBを採用する。

## 14. 成果物と状態

| 成果物 | 状態 | 正本または証跡 |
|---|---|---|
| 理論 | 完了 | 本文書 |
| dataset correlation report | 完了 | `work_dirs/diagnostics/obb_projection_correlation/report.json` |
| projection diagnostic overlay | 未生成 | 数値診断・学習採否には使用していない。将来、画像境界でのconic形状を目視監査する場合の任意成果物として残す |
| projection utility/loss、OBB target pipeline、config | 完了 | `yopo/models/losses/projected_ellipsoid_loss.py`、custom fruit pipeline、`configs/yopo/*projection_gwd*.py` |
| math/data-transform/gradient/inference contract tests | 完了 | 最終対象回帰88件pass。全`tests/`のうち13件は今回と無関係な旧artifact欠落でfail |
| A–C config、log、checkpoint、prediction dump、比較JSON | 完了 | 各`work_dirs/nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_*`と`work_dirs/diagnostics/control_vs_projection_obb_aux_rotation/report.json` |
| 作業記録 | 完了 | `temp/workdoc_Aug25-2026_cop_chain_obb_center_distillation.md` |

## 15. 2026-08-25の実験結果

全runを同じQ150 Stage4 best、5 epoch、seed 3407、50-image validationから開始した。連続rotation値はprediction dumpをGTへ2D xyxy IoU Hungarianで一対一対応させ、IoU `>=0.5` の組だけについて、対称性で商を取らないSO(3) geodesic errorを計算した。

| run | AP50 | 3D IoU@.50 | 3D IoU@.75 | pose 10deg/10cm | translation 10cm | rotation median | rotation p75 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: continuation control | 0.3360 | 0.4450 | 0.1170 | 0.0254 | 0.6192 | 20.0218 deg | 29.7952 deg |
| B: direct projection, weight 1 | 0.3360 | 0.4441 | - | 0.0253 | 0.6170 | 20.0400 deg | 29.8273 deg |
| C5: B + OBB aux, weight 5 | 0.3360 | 0.4498 | 0.1203 | 0.0240 | 0.6180 | 20.3831 deg | 30.2036 deg |
| C1: B + OBB aux, weight 1 | 0.3370 | 0.4503 | 0.1189 | 0.0251 | 0.6211 | 20.1490 deg | 29.8071 deg |

`-` は当該値を比較artifactへ固定していないことを表し、0とは解釈しない。C1のteacher-free validationは同じAP/IoU/pose値を再現した。projection/aux lossは全updateでfinite、invalid projectionは0、peak GPU memoryは約24.3 GiBであった。

### 15.1 仮説ごとの判定

- BはAに対してIoU、pose、matched rotation median/p75のいずれも改善せず、direct projection単独のrotation改善仮説は不採用とする。
- C5はIoUを改善したが、matched rotationとposeを悪化させた。auxiliary weight 5はrotation目的には過大である。
- C1はC5よりmatched rotationを改善し、AP50、IoU@.50、translation 10cmで4 run中最高になった。しかしAに対するrotation medianは`+0.1272 deg`、p75は`+0.0119 deg`、pose 10deg/10cmは`-0.0003`であり、rotation改善条件を満たさない。
- 従ってC1 checkpointは「best 3D IoU候補」として保存するが、「OBBでrotationが改善したcheckpoint」とは呼ばない。rotation目的でのweight sweepはここで打ち切る。

比較の正本は次である。

- `work_dirs/diagnostics/control_vs_projection_obb_aux_rotation/report.json`
- C1 checkpoint: `work_dirs/nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_projection_gwd_obb_aux_w1/best_3d_iou_0.50_epoch_5.pth`
- C1 prediction dump: `work_dirs/nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_projection_gwd_obb_aux_w1/predictions.pkl`

## 16. 中間結論（5 epoch直接loss実験時点）

2D OBBと3D poseの一貫性を定義する方法として、3D ellipsoidを既知cameraで2D conicへ投影し、観測OBBとGWDで比較する定式化自体はwell-definedであり、データ相関gateと微分可能性testも通過した。一方、実測ではdirect projectionもquery-level covariance auxも3D rotationを改善しなかった。つまり「OBBにrotation情報がある」ことと「現在のloss/head経路がその情報を3D rotation精度へ変換できる」ことは別であり、後者は支持されていない。

OBB auxは3D IoUとtranslationをわずかに改善したため、rotation featureの表現正則化またはsize/depthとの共有表現には寄与した可能性がある。ただしこれは実測からの推論であり、rotation改善の証拠ではない。

## 17. 次の検証可能な仮説

次にrotationを狙う場合は、単なるloss weight調整を繰り返さず、次のいずれかを独立ablationにする。

1. OBB covarianceから得たanisotropyを用い、向きが観測可能なinstanceだけdirect projection gradientを連続的に強める。
2. aux covarianceを予測するだけでなく、そのcompact covarianceをrotation regressionへ明示的に入力し、推論時にも同じ経路を通す。
3. annotationの物体対称性を定義し、SO(3)評価とtraining lossの双方を同じsymmetry quotientへ揃える。

どの条件でも、A controlを再利用するだけでなく、変更点を1つに限定し、本文13節のrotation median/p75とpose gateを満たすまでrotation改善とは判定しない。

## 18. 異方性重み付きprojectionの定義と結果

GT OBB covarianceの固有値を `lambda_max >= lambda_min > 0` とし、

```text
a = (lambda_max - lambda_min) / (lambda_max + lambda_min)
w_i = a_i^p / mean_positive(a^p)
```

と定義する。`p=0` は従来の一様weightと厳密に一致し、実験では
`p=1` とした。batch内の正例平均weightを1に固定するため、projection
loss全体のscaleを変えず、向きの観測可能性だけを連続的に反映する。
hard maskは使用しない。

事前診断では、valのanisotropy平均/中央値は`0.2207/0.2093`であった。
paired/shuffled projection GWDの中央値比は`a<0.1`で`0.857`、
`0.4<=a<0.5`で`0.334`となり、anisotropyと
`shuffled-paired` gainの相関は`0.3075`であった。従って、GT幾何の
観測可能性指標としての`a`は支持された。

一方、同じStage4 bestからの5 epoch学習では次になった。

| run | AP50 | IoU@.50 | IoU@.75 | pose 10deg/10cm | translation 10cm | rotation median | rotation p75 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: control | 0.3360 | 0.4450 | 0.1170 | 0.0254 | 0.6192 | 20.0218 deg | 29.7952 deg |
| E: anisotropy-weighted projection | 0.3360 | 0.4451 | 0.1168 | 0.0253 | 0.6164 | 20.0909 deg | 29.8102 deg |

Eはinvalid projection 0、NaN/Inf/OOM 0、peak約24.2 GiBで完走したが、
controlに対してpose、translation、rotation median/p75の全てで改善せず、
不採用とする。これは「GT OBBのanisotropyがrotation観測可能性を表す」
ことを否定しないが、「projection gradientをその値で再重み付けすれば現head
の3D rotationが改善する」という学習仮説を否定する。

## 19. OBB covarianceからrotationへの明示的conditioning

aux lossだけではOBB出力がrotation推論経路に入らないため、予測covariance
`Sigma`から次のspin-2 descriptorを作る。

```text
q(Sigma) = ((Sigma_xx - Sigma_yy) / tr(Sigma),
            2 Sigma_xy / tr(Sigma))
```

`q`の偏角はOBB axis angleの2倍、ノルムはanisotropyである。そのため
180度異なる同一軸を同じ値へ写し、円形に近いOBBでは方向入力が自然に0へ
近づく。予測器はrotation stage直前のfeatureからstrict-SPD covarianceを
生成し、`Linear(2,256)(q)`をrotation stage inputへ残差加算する。推論でも
同じ自己予測経路を通り、GT OBBやteacherを必要としない。

TDDでは次を固定した。

1. `q`が共分散の一様scaleに不変である。
2. conditioning有効時、rotation出力の勾配がOBB predictorへ到達する。
3. legacy OBB auxでは同じ勾配が到達しない。
4. OBB lossなし、またはrotationが最終CoP stageでない設定を拒否する。

実測結果は次である。

| run | AP50 | IoU@.50 | IoU@.75 | pose 10deg/10cm | translation 10cm | rotation median | rotation p75 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: control | 0.3360 | 0.4450 | 0.1170 | 0.0254 | 0.6192 | 20.0218 deg | 29.7952 deg |
| F: OBB-conditioned rotation | 0.3360 | 0.4456 | 0.1173 | 0.0252 | 0.6217 | 20.1361 deg | 29.7263 deg |

FはIoU@.50/.75、translation 10cm、rotation p75をわずかに改善したが、
poseとrotation median/meanを悪化させた。paired mean deltaはcontrol比
`+0.0376 deg`、median deltaは`+0.0234 deg`であり、本文13節の同時gateを
満たさないためrotation改善としては不採用とする。

異方性別paired deltaでは、`a<0.1`のmedianだけ`-0.0341 deg`で改善した。
`a>=0.1`の全binではmean/medianとも悪化し、`0.4<=a<0.5`ではmedian
`+0.0647 deg`であった。高観測性ほど改善するという期待と逆なので、次の
ボトルネック候補はconditioningの有無ではなく、自己予測したOBB axisの
精度である。この解釈は現時点では推論であり、予測`q`とGT `q*`の角度・
anisotropy誤差を直接測って確定する。

## 20. 更新後の成果物と次のgate

- E checkpoint: `work_dirs/nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_projection_gwd_anisotropy/best_3d_iou_0.50_epoch_5.pth`
- F checkpoint: `work_dirs/nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_obb_rotation_conditioning/best_3d_iou_0.50_epoch_5.pth`
- 最終比較: `work_dirs/diagnostics/control_vs_projection_obb_conditioned_rotation/report.json`
- conditioning config: `configs/yopo/nocs_custom_fruit_rgbd_3dbbox_cop_q150_stage4_obb_rotation_conditioning.py`

次の実装変更前に、Fのmatched positiveについて予測OBBとGT OBBの
double-angle alignment、anisotropy calibration、GWDを測る。高anisotropy
GTでaxis alignmentが不足するならOBB branchの表現/教師を改善し、十分なら
3D symmetry quotientまたは2D axisから3D rotationへの非一意性を次の主因と
する。診断なしの追加weight sweepは行わない。

## 21. 予測OBB axisの直接診断

診断configでのみ最終queryの正規化Gaussianをprediction dumpへ追加し、通常の
推論schemaは変更しなかった。2D IoU Hungarian、IoU `>=0.5`の2,372組で、
予測/GT descriptorのcosineをdouble-angle alignment、
`0.5*acos(alignment)`を180度同一視したaxis errorとして測定した。

| run | predicted anisotropy median | anisotropy correlation | weighted alignment | axis error median | normalized GWD median |
|---|---:|---:|---:|---:|---:|
| C1: legacy post-rotation OBB aux | 0.0732 | 0.0248 | 0.3432 | 33.50 deg | 0.1334 |
| F: pre-rotation OBB conditioning | 0.0526 | -0.0047 | -0.1108 | 47.41 deg | 0.2426 |

GT anisotropy medianは約`0.187`であり、両runともanisotropyを過小推定し、
instance間相関もほぼ0である。さらにFはaxis alignment自体が負で、legacy
より明確に悪い。FではOBBをrotation stage featureの形成前に予測したため、
OBB教師をrotationへ入力できても、入力された`q`が観測軸を表していない。
これはFの高anisotropy binでrotationが改善しなかった結果を直接説明する。

次の単一変更は二段refinementとする。まず従来どおりrotation stageを完了し、
そのpost-rotation featureからOBB covarianceを予測する。その`q`を同featureへ
残差加算した後、既存rotation output layerを再適用して最終rotationを得る。
これにより、C1と同じOBB予測位置を保持したまま、OBB→rotationの明示経路だけ
を追加できる。採用gateは本文13節を変更しない。

## 22. post-rotation refinementと対称性診断

二段refinementは、各decoder層で従来のrotation stageを一度完了したfeature
`h_R`からOBB covarianceを予測し、19節のdescriptor `q`を同じfeatureへ
残差加算して、共有rotation output layerを再適用する。

```text
Sigma_hat = OBBHead(h_R)
q_hat = q(Sigma_hat)
h_refined = h_R + Linear(q_hat)
R_hat = RotationOut(h_refined)
```

OBB predictorの位置をlegacy C1と同じに保ち、OBBからrotationへの明示経路
だけを追加する。zero embedではOBBとrotation出力がlegacyとbitwise一致し、
rotation lossからOBB predictorへ有限・非zero gradientが届くことをtestで固定
した。

元Stage4 bestからの5 epoch実験では、AP50 `0.3360`、IoU@.50/.75
`0.4402/0.1188`、pose 10deg/10cm `0.0245`、translation 10cm `0.6163`、
SO(3) median/p75 `20.1508/29.7091 deg`であった。p75はcontrolより改善したが、
medianとposeが悪化し、13節の同時gateを満たさなかった。従って、粗いOBB
表現のままrefinementだけを足す仮説は不採用である。

別に、annotationを変更せず評価上の対称性だけで説明できるかを診断した。
180度C2をx/y/z軸へ適用してもSO(3)誤差は変化せず、連続SO2 quotientでは
medianがx/y/z軸で`15.39/13.62/16.83 deg`まで下がった。しかし直交二軸の
relative size difference中央値は`0.180/0.197/0.171`であり、物体は厳密な
軸対称ではない。従って、根拠なくsymmetry quotientをtraining/evaluationへ
導入することはしない。

## 23. 2D OBB foundationの定義と品質gate

21節の診断は、3D回転接続以前にOBB表現自体が未学習であることを示した。
このボトルネックを単独で解くため、元Stage4 bestから次の20 epochを学習した。

- 有効: class、2D bbox L1/GIoU、2D center、query-level OBB GWD weight 5
- 無効: z、size、rotation、direct projection、全distillation
- validation: 5 epochごと、seed 3407、Q150、batch 22、FP32
- 採用gate: AP50 `>=0.328`、axis error median `<33.50 deg`、weighted
  alignment `>0.3432`、anisotropy correlation `>0.0248`

3D目的を0にしたので、この段階の3D IoU/poseは採否指標にしない。20 epoch後、
AP50は`0.354`、recallは`0.643`となった。matched OBB品質は次のとおりである。

| run | matched | axis error median | weighted alignment | anisotropy corr. | normalized GWD median |
|---|---:|---:|---:|---:|---:|
| legacy C1 | 2,372 | 33.50 deg | 0.3432 | 0.0248 | 0.1334 |
| 2D OBB foundation | 2,441 | 12.17 deg | 0.7232 | 0.1061 | 0.1010 |

全gateを大差で通過した。これは「OBB supervisionが存在する」だけでは不十分で、
2D detector/decoder表現をOBB目的で十分収束させてから3D回転へ進む必要がある
ことを示す。20 epoch checkpointとdiagnostic dumpは、それぞれSHA-256
`2b3831e2...ee188`、`042b960c...441e`である。

## 24. 2D foundation後の3D再開と最終結果

まず、同じ2D OBB foundationから5 epochのmatched A/Bを実行した。両runは
z/size/rotation、distillation、OBB weight 1を同一にし、失敗済みdirect
projectionを無効化した。Bだけが22節のpost-rotation refinementを持つ。

| run | AP50 | IoU@.50 | IoU@.75 | pose 10deg/10cm | trans. 10cm | SO(3) median | SO(3) p75 |
|---|---:|---:|---:|---:|---:|---:|---:|
| foundation control | 0.350 | 0.4238 | 0.1093 | 0.0242 | 0.6120 | 20.2022 deg | 29.9801 deg |
| foundation refinement | 0.349 | 0.4257 | 0.1095 | 0.0245 | 0.6134 | 20.1921 deg | 29.9745 deg |

Bは13節の最低維持gateとrotation改善3条件をすべて通過した。2,415 matched
pairのrotation delta中央値は`-0.0075 deg`、改善率は`54.1%`であり、効果は
小さいが符号は整合した。GT anisotropy `0.4--0.5`ではdelta中央値
`-0.0318 deg`、改善率`64.7%`で、方向が観測しやすいinstanceほど効果が
大きかった。ここでrefinement経路を採用した。

その後、採用runから15 epochを追加し、5 epochごとに同一validationを行った。

| 追加epoch | AP50 | IoU@.50 | IoU@.75 | pose 10deg/10cm | trans. 10cm |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.354 | 0.4711 | 0.1285 | 0.0397 | 0.6325 |
| 10 | 0.357 | 0.4845 | 0.1489 | 0.1504 | 0.6447 |
| 15 | 0.351 | 0.4853 | 0.1556 | 0.2696 | 0.6471 |

IoU基準の最終bestは追加epoch 15である。teacher-free再評価は上表を再現し、
SO(3) errorはmatched 2,459組でmean/median/p75
`13.5510/9.7454/16.8337 deg`となった。旧Stage4 continuation controlとの
共通2,343組ではpaired delta mean/median `-8.8207/-7.1166 deg`、改善率
`86.77%`である。旧controlのAP50/IoU@.50/IoU@.75/pose/trans.
`0.336/0.4450/0.1170/0.0254/0.6192`も全て上回る。

3D再学習後のOBB axis error medianは`17.30 deg`へ戻ったが、legacy
`33.50 deg`より良く、anisotropy correlation `0.1513`、weighted alignment
`0.6426`、GWD median `0.1099`もlegacyを上回る。従って、3D改善がOBB表現の
完全な消失で得られたものではない。

## 25. 最終結論と利用する成果物

本データでは、3D ellipsoidの2D投影とOBB GWDの定式化は情報gateを通る一方、
direct projection loss、異方性weight、未収束OBBからのconditioningは回転を
改善しなかった。成功した順序は次である。

```text
高品質2D detection
  -> 2D OBB GWDを十分収束
  -> post-rotation OBB descriptorで小さくrefine
  -> full 3D Stage-4を十分継続
```

したがって、今回の大幅な最終改善を「投影loss単体の効果」とは解釈しない。
実測が支持するのは、2D-first curriculum、学習済みOBB表現、明示的だが小さい
OBB-to-rotation経路、十分な3D recovery scheduleの組合せである。短期matched
ablationだけがrefinementの因果効果を分離し、15 epoch継続結果は採用pipeline
全体の到達性能を表す。

- 最終checkpoint:
  `work_dirs/nocs_custom_fruit_rgbd_2d_obb_foundation_stage4_refinement_continue15/best_3d_iou_0.50_epoch_15.pth`
  （SHA-256 `df4791a4...0aed3`）
- teacher-free prediction + OBB dump:
  `work_dirs/nocs_custom_fruit_rgbd_2d_obb_foundation_stage4_refinement_continue15/predictions_with_obb.pkl`
  （SHA-256 `a4bae6b0...461c`）
- 最終rotation比較:
  `work_dirs/diagnostics/obb_foundation_final_rotation/report.json`
  （SHA-256 `756c2fc3...e8e9`）
- 最終OBB診断:
  `work_dirs/diagnostics/predicted_obb_alignment_full/final_3d_report.json`
  （SHA-256 `712b8717...9f1`）

通常推論では`expose_obb_aux_predictions=False`のままとし、診断configだけが
`obb_gaussians`を出力する。最終checkpointはteacher-free inference configへ
明示的に渡して使用し、training configのteacher pathへ依存させない。

2D queryのobjectness、duplicate suppression、NMS-free selectionは
`docs/nms_free_rgbd_query_architecture.md`を正本とする。本書のOBB/3D整合lossは
Hungarian-positive queryへ適用し、background queryの未教師OBBをselectionへ
使用しない。
