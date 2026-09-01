# 作業書: RGB-D GauCho-3D 楕円体検出（YOPO `rgb-d`）

本書がこの作業の唯一の真実源 (SSOT) である。要求、設計判断、チェックリスト、DoD、作業記録を
ここへ集約する。`docs/` へ出すのは確定した設計と結果だけとする。

- 作成: 2026-08-31 22:19:35 JST+0900
- 対象リポジトリ: `/home/kasm-user/Desktop/YOPO_clone`（branch `rgb-d`）
- 参照実装: `/home/kasm-user/Desktop/rotated_rtmdet_jax`（branch `cu12-ellipse`）
- 理論正本:
  - `/home/kasm-user/Desktop/Downloads/aug31 (1)/RGBD_GauCho3D_Ellipsoid_DualQuadric_Design_ja.pdf`（A4 26頁）
  - `/home/kasm-user/Desktop/Downloads/aug31 (1)/ChatGPT-3D_GauCho_Ellipse_R-CNN_技術設計ノート.md`
  - `/home/kasm-user/Desktop/data/ellipse/Amodal_GauCho_Ellipse_RCNN_Technical_Note.pdf`（2D 版・先行）

---

## 1. 目的と要求

### 1.1 ユーザー目的

`rotated_rtmdet_jax` の `cu12-ellipse` で確立した 2D GauCho（角度を回帰せず Cholesky 因子から
SPD Gaussian を直接構成する）を、YOPO `rgb-d` の RGB-D 経路へ 3 次元へ拡張する。RGB-D 1 枚から
物体ごとに 2D amodal 楕円と metric 3D 楕円体を同時推定し、両者を射影幾何と深度観測で自己整合させる。

### 1.2 Definition of Done（ユーザー指定・2026-08-31 追加）

| ID | 条件 | 判定方法 |
|---|---|---|
| **DoD-A** | 2D ellipse 検出性能が `rotated_rtmdet_jax` の 2D GauCho と同等 | ellipse envelope 対 GT OBB の rotated mAP50。rtmdet 側の実測は raw **0.851443**（`docs/GAUCHO_ELLIPSE_2025_2026.md`、2025+2026 800x600 DOTA, batch 48, FULL epoch 22） |
| **DoD-B** | 良好な 3D 認識性能 | 3D IoU AP @0.10/@0.25/@0.50 と pose AP 10deg/10cm。現行 compact CoP ベースライン（`YOPO_best_models_20260826/README.md`）を上回ること |

**DoD-A の比較条件について（重要・未解決）**: rtmdet の 0.851443 は
**2025+2026 の 800x600 DOTA（train 2,541 / valid 511, class `tomato`）** で得た値である。
一方 YOPO 側で今回利用できる RGB-D 3D OBB データは
**Jun30-2025 736x512（train 1,196 / valid 330, class `stem`）** であり、クラスも枚数も分割も異なる。
したがって 0.851443 との直接比較は成立しない。取り得る選択肢は §5 に記す。この判断はユーザー確認事項。

### 1.3 要求追跡 (R1-R10、理論正本 表1 に対応)

| ID | 要求 | 本作業での充足 |
|---|---|---|
| R1 | 2D GauCho を 3D 楕円体へ拡張 | `gaucho3d_geometry.cholesky3d_from_raw` / `scale_shape_cholesky3d` |
| R2 | RGB-D を metric 3D 推定へ使う | 充足。ただし**設計書の式(13)(14) ではなく YOPO 既存の方式**で満たしている（§X 参照） |
| R3 | 2D ellipse head と 3D 出力を整合 | `project_ellipsoid_dual_quadric`（`Q* -> C* = P Q* P^T` のみ） |
| R4 | 直交 2D GauCho を重ねる案の検証 | `dual_plane_cholesky3d` / `cholesky3d_to_dual_plane`、`rc=0` 縮約の欠損を数値提示 |
| R5 | pose/shape と対称性 | **未達**。SPD は正本だが、継承元の `AMPStableRotation3DLoss`（weight 5.0）が全 GauCho stage で有効であり、設計書 §8.4 が「置かない」とする full SO(3) pose loss をそのまま学習している（§X 参照） |
| R6 | アモーダルと可視の分離 | amodal ellipse と visible mask を別 field。depth equality は visible 画素限定 |
| R7 | head 間の共謀防止 | stop-gradient 規則、段階学習、独立教師 2 系統以上 |
| R8 | 数値的に定義済み | 契約試験（§4）、validity の fail-closed |
| R9 | 既存 2D GauCho の利点維持 | 角度非回帰、inverse-free KLD、ellipse envelope 評価 |
| R10 | 設計判断の切り分け | A0-A9 アブレーション |

### 1.4 制約

1. 編集スコープは `yopo/models/losses/`、`yopo/models/dense_pose_heads/`、`configs/yopo/`、
   `tests/`、`temp/`、`docs/` に限る。`pyproject.toml` と `justfile` は変更しない。
2. `yopo/` 本体の既存経路（CoP / parallel の 9D pose）は退行させない。新経路はフラグで分離する。
3. 実行は必ずリポジトリ root から。`init_default_scope('yopo')` 規約を守る。
4. 暗黙 fallback 禁止。invalid は typed status で除外し、固定値で埋めない。

---

## 2. 環境の実状（2026-08-31 時点）

| 項目 | 状態 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, **compute cap 8.9 (sm_89)**, 24 GB |
| `pyproject.toml` の mmcv | `mmcv-2.2.0-0sm120t280`（**sm_120 = RTX 5090 専用**） |
| 対処 | 同一リリースの `mmcv-2.2.0-0sm89t280-cp310-cp310-linux_x86_64.whl` を `wheels/` に取得済み。`pyproject.toml` は変更せず venv 側だけ差し替える |
| VRAM | 24 GB。設計書の実測 batch 48 は 32 GB 前提のため、そのままでは載らない（§5 参照） |
| `uv sync` | 実行中（回線が約 2 MB/s で共有されており長時間かかる） |

---

## 3. 実装方針（理論正本 §13 の確定推奨に従う）

1. **3D 正本**: `Ellipsoid3D(t, Sigma)`。標準 head は scale–shape 3D Cholesky。
   radii/主軸/quaternion は診断用派生量に降格する。
2. **2D 正本**: whole-object/amodal `Ellipse2D(mu, S)`。visible mask とは別 field。
3. **2D/3D 結合**: shared feature + 独立 decoder。`Q* -> C* = P Q* P^T` を唯一の正確な投影主経路とする。
   透視 Jacobian による共分散輸送、8頂点投影、AABB envelope は使わない。
4. **段階**: Independent 基線 → stop-gradient した soft conditioning。hard stack は標準にしない。
5. **loss**: 3D は inverse-free Cholesky KLD を主、GWD を併用可。2D は既存 `GaussianGWDLoss` を再利用。
6. **対称性**: sphere で pose なし、spheroid で無向 axis。full SO(3) pose loss を置かない。

### 3.1 head 結合方式（理論正本 表2）

| 方式 | 採否 |
|---|---|
| Independent | **基線として必須**。まずこれで正しさを確立する |
| Soft-conditioned | **推奨最終形** |
| Hard-stacked | 初期実装には非推奨（透視近似誤差が直接伝播）。A9 でのみ評価 |
| Dual-plane | 3D decoder 内部の parameterization option として提供 |

---

## 4. 原子的チェックリスト

### Phase 1: 幾何コア

- [x] `yopo/models/losses/gaucho3d_geometry.py` を作成
- [x] 3D Cholesky chart（式9）と逆写像
- [x] scale–shape chart（式10,11）と逆写像
- [x] dual-plane GauCho（式16,18）と標準 Cholesky との相互変換
- [x] block-Cholesky（式21）
- [x] dual quadric（式31）、投影（式32）、dual conic -> Gaussian（式33,34）
- [x] front margin（式35）、ray frame（式25,26）、ray-ellipsoid 解析根（式43,44）
- [x] `decode_center_from_anchor`（式13,14）— 実装・単体試験済み。**意図的に未接続**（YOPO 既存方式を採用、§X 参照）
- [x] 対称性クラス分類
- [x] 倍精度契約試験 17 項目すべて PASS（§4.1）

### Phase 2: 損失

- [x] 3D KLD（式39、inverse-free Cholesky solve）
- [x] 3D GWD（式37,38）
- [x] dual-quadric 投影整合（式40）
- [x] conic 残差（式41）
- [x] ray-surface（式45）／free-space（式46）
- [x] `MODELS` 登録と `losses/__init__.py` への追加

### Phase 3: head 統合

- [x] `DINO9DCenter2DPoseHead` に GauCho-3D branch を追加（既定 off）
- [x] 出力契約 `Ellipsoid3D` / `Ellipse2D` の実装（`expose_gaucho_predictions`）
- [~] stop-gradient 規則 — §9.2 の規則2（centre/depth anchor の detach）が未適用
- [x] 推論経路と validity の fail-closed

### Phase 4: 設定・試験・学習

- [x] 契約試験を `tests/test_gaucho3d.py` として固定
- [x] Independent 基線 config
- [x] smoke train（Stage B、暫定環境）
- [~] 本学習と DoD-A / DoD-B 評価（Stage A 完了、Stage B 実行中、Stage C 待機）

### 4.1 契約試験の実測（2026-08-31、float64、torch 2.1 隔離環境）

| 検査 | sample | 結果 | 閾値 |
|---|---:|---:|---:|
| direct Cholesky 最小固有値 | 20,000 | 3.179e-12 | > 0 |
| direct Cholesky raw 往復 | 20,000 | 1.110e-16 | < 1e-12 |
| dual-plane L 往復 | 20,000 | 9.159e-15 | < 1e-10 |
| standard -> dual-plane -> standard | 20,000 | 4.526e-14 | < 1e-10 |
| dual-plane Sigma 往復 | 20,000 | 2.728e-12 | < 1e-10 |
| scale–shape det 恒等式 | 10,000 | 3.640e-12 | < 1e-10 |
| scale–shape 往復 | 10,000 | 2.665e-15 | < 1e-10 |
| dual conic canonical 往復 | 5,000 | 5.821e-11 | < 1e-9 |
| `P Q* P^T == K(Sigma-tt^T)K^T` | 5,000 | 1.164e-10 | < 1e-9 |
| ray root surface 残差 | hit のみ | 4.867e-12 | < 1e-9 |
| near <= far 順序 | hit のみ | 全件正 | — |
| SE(3) center / SPD 往復 | 10,000 | 8.438e-15 / 9.237e-14 | < 1e-10 |
| 主軸 sign 反転 / 置換の Sigma 不変性 | 1,000 | 0.0 / 0.0 | < 1e-12 |
| front_margin と `C33 < 0` の一致 | 5,000 | 1.0000 | = 1 |

診断: `rc = 0` 固定の二平面のみ縮約モデルは、full-SPD に対して相対 Frobenius 誤差の中央値
**8.764e-2**。二つの直交 2D GauCho だけでは一般形に 1 自由度不足するという理論正本の主張を再現した
（正本の記載値 0.175 とは sampling 分布が異なるため絶対値は一致しない。符号と結論は一致）。

---

## 5. 未解決の判断事項（ユーザー確認を要する）

1. **DoD-A の比較条件**: §1.2 のとおり rtmdet の 0.851443 とは data split もクラスも異なる。
   - (a) YOPO 側の Jun30-2025 `stem` valid 330 枚で rtmdet ellipse checkpoint を評価し、
     同一 split の対照値を作ってから比較する（推奨。追加学習不要で対照が取れる）
   - (b) rtmdet が使った 2025+2026 800x600 DOTA を YOPO 側にも用意する（データが手元にない）
   - (c) 「同等」を絶対値ではなく「同一 split で rtmdet を下回らない」と定義する
2. **VRAM**: 設計書の batch 48 は 32 GB 実測。24 GB の RTX 4090 では batch probe をやり直す必要がある。
3. **3D 教師の限界**: 単眼 RGB-D では背面形状が一意に決まらない。本データの 3D OBB は
   `projection-tight` で生成された疑似教師であり、真の楕円体 GT ではない。
   したがって DoD-B は「疑似教師に対する一致度」であり、真の 3D 形状精度は主張できない。

---

## 6. 作業記録

### 2026-08-31 22:19:35 JST+0900 — 開始

- 環境調査完了。RTX 4090 = sm_89 と `pyproject.toml` の sm_120 固定の不一致を検出、sm_89 wheel を確保。
- 理論正本 26 頁と 2D 参照実装（`geometry/ellipse.py` の GauCho chart、
  `losses.py` の `_gaussian_kld_from_cholesky`）を読了。
- Phase 1 完了: `yopo/models/losses/gaucho3d_geometry.py` を新規作成。

### 2026-08-31 22:19 — Phase 1 セルフレビューで検出した欠陥と修正

1. **`cholesky3d_to_scale_shape` の `rho` が壊れる（重大・修正済み）**
   3 つの Cholesky 対角の**積**へ `clamp_min(eps=1e-7)` を掛けていた。metric 半径が
   センチメートル級（`l11*l22*l33` が 1e-8 台）の果実では floor に張り付き、
   scale–shape 往復の最大誤差が **1.372e+00** になった。対数和
   `log_rho = (log l11 + log l22 + log l33)/3` へ置換し、**2.665e-15** へ改善。
   併せて `log` の floor を物理値と無関係な `_LOG_FLOOR = 1e-30` へ分離した。
   `DEFAULT_EPS` を長さと体積の両方に流用していたのが根本原因である。
2. **`cholesky3d_from_raw` の死んだ行（修正済み）**: `if False else None` の残骸を削除。
3. **`rho0` のブロードキャスト処理が場当たり（修正済み）**: `_as_broadcastable` へ集約し、
   不正な shape は黙って reshape せず `ValueError` にする。
4. **未使用 import（修正済み）**: `math` を削除。


### 2026-08-31 22:4x — Phase 2 完了（損失）

新規 `yopo/models/losses/gaucho3d_loss.py`。`MODELS` へ登録し `losses/__init__.py` へ追加。

| クラス | 対応式 | 役割 |
|---|---|---|
| `Ellipsoid3DKLDLoss` | 式39 | **3D 主目的**。三角 solve のみで逆行列を作らない |
| `Ellipsoid3DGWDLoss` | 式37,38 | 3D 補助 |
| `DualQuadricProjectionGWDLoss` | 式40 | `Q* -> C* = P Q* P^T` による 2D-3D 整合。teacher を既定 detach |
| `DualConicResidualLoss` | 式41 | conic 残差（scale/sign を canonical 化） |
| `RayEllipsoidSurfaceLoss` | 式42-45 | visible 画素限定の front-surface 一致 |
| `EllipsoidFreeSpaceLoss` | 式46 | 観測 surface より手前の free-space 侵入を片側罰則 |

試験は `tests/test_gaucho3d.py`（28 件）。torch 2.1 の隔離環境で **26 passed / 2 failed**。

#### Phase 2 セルフレビューで検出した欠陥と修正

1. **近球極限で GWD 勾配が NaN（重大・修正済み）**
   設計書 §11.3 の stress test 1「`r1 -> r2 -> r3` の近球極限で勾配が有限か」で再現した。
   原因は二つ。
   - `_symmetric_sqrt` の `torch.linalg.eigh` の backward は固有値差で除算する。球・回転楕円体
     （＝トマトが実際に取る形状）では固有値が縮退し 0 除算になる。target は定数なので
     `torch.no_grad()` + `detach()` に変更し、特異点が勾配へ到達しないようにした。
   - 予測が正解に一致したとき距離が 0 になり、`sqrt` の微分が発散していた。設計書 式(38) に
     従い**二乗距離**へ bounded 変換を掛ける形へ変更（0 で微分 0）。
     併せて `cross_eigenvalues` の floor を `0` から `eps` へ変更した。
2. **`normalize` が no-op だった（修正済み）**: `Ellipsoid3DKLDLoss` の `normalize` 引数は
   `distance = distance` で何もしていなかった。KLD は中心項が予測形状下の Mahalanobis なので
   元から metric サイズ不変であり、正規化は不要。引数ごと削除した。
3. **`include_center=False` の実装が不透明だった（修正済み）**: 「predicted の代わりに target の
   中心を渡して差を 0 にする」という裏技だった。`ellipsoid_kld_from_cholesky` の明示引数へ変更。
4. **毎 iteration の GPU 同期（修正済み）**: `_finalize` が常に `.item()` を呼んでいた。
   `fail_on_invalid=True` のときだけ同期するようにした。
5. **テスト側の誤り（修正済み）**: bounded loss の上界は `< 1` ではなく `<= 1`。
   浮動小数点では上界に到達する（それが意図した cap）。

#### 未実行の 2 件（環境制約であり実装欠陥ではない）

`test_projection_loss_is_minimal_at_the_true_ellipsoid` と
`test_projection_loss_detaches_its_two_dimensional_teacher` は、隔離環境の torch 2.1.0 に
`Tensor.all(dim=tuple)` が無いため実行できない。呼び出し先は**既存**の
`projected_ellipsoid_loss.gaussian_wasserstein_distance` であり、本作業で追加した経路ではない。
リポジトリが固定する torch 2.8.0 では利用可能な API のため、`uv sync` 完了後に実 venv で再実行する。
古い torch へ合わせて既存コードを書き換えることはしない。

### 2026-08-31 23:0x — Phase 3（head 統合）

`DINO9DCenter2DPoseHead` へ GauCho-3D branch を追加した。既定 `gaucho_ellipsoid=False` で、
既存 config の挙動は不変である。

- `_init_layers`: `reg_ellipsoid_branch`（class ごと 6 値）。zero-init で全 query が
  size prior の等方楕円体から始まる。
- `forward`: 返り値の**末尾**へ `all_layers_outputs_ellipsoid` を追加。既存の位置参照
  （推論の `outs[:6]`、OBB 診断の `outs[9]`）は不変。
- `loss_by_feat` / `loss_by_feat_simple` / `loss_by_feat_single`: `obb_aux` と同じ
  「任意の per-layer テンソル」経路に揃えた。
- 3D 中心は既存の 2D 中心・深度枝を再利用する（第二の translation 推定器を作らない）。
  GauCho branch は `Sigma` だけを供給する = Independent 基線。
- GT は annotation の `(R, size)` から `Sigma_g` を構成する。ネットワークは `R` を回帰しない。

#### Phase 3 セルフレビューで検出した回帰と修正

`forward` の返り値を 10 → 11 要素にしたことで、**既存テスト 9 件が壊れた**。
変更を `git stash` して同一 shim でベースラインを取り直し、環境由来の失敗と切り分けて特定した。

| 原因 | 修正 |
|---|---|
| `loss_by_feat` / `loss_by_feat_simple` / `loss_by_feat_single` に**必須**位置引数を追加した。既存テストは位置引数で呼んでいる | 新引数をすべて末尾のデフォルト付きへ移動 |
| `loss_by_feat_simple(*head(...))` と呼ぶテストがあり、11 要素目が `batch_gt_instances` に流れ込んだ | `all_layers_ellipsoid_preds` を **11 番目の位置**に置き、`batch_gt_instances` / `batch_img_metas` を既定 `None` + 明示検証へ |
| `multi_apply` は keyword を layer ごとに変えられない | 明示的な per-layer ループへ置換 |
| test double が歴史的な 12-tuple を返す | `loss_by_feat_simple` は 12/14 の両方を受理する |
| `forward` の要素数を直接アサートするテスト 1 件 | 意図した契約変更なのでアサートを更新し、新スロットが `None` であることの検証を追加 |

#### 回帰確認（同一 shim でのベースライン比較）

| テストファイル | baseline | 変更後 |
|---|---|---|
| `test_rgbd_cop_prediction_mode` | 26 passed | **26 passed** |
| `test_rotation_6d_stiefel_head_integration` | 6 passed | **6 passed** |
| `test_rgbd_nms_free_quality` | 4 passed | **4 passed** |
| `test_dino_9d_center2d_no_dn` | 3 passed | **3 passed** |
| `test_rgbd_3dbbox_geometry_diagnostics` | 4 passed | **4 passed** |
| `test_matchability_head_integration` | 4 passed | **4 passed** |
| `test_matchability_quality` | 3 passed | **3 passed** |
| `test_projected_ellipsoid_gwd_loss` | 2 passed | **2 passed** |

**回帰ゼロ**。残る失敗は全ファイルでベースラインと同一であり、隔離環境の torch 2.1 が
`Tensor.all(dim=tuple)` 等を持たないことに起因する既存の環境制約である。

### 2026-08-31 23:22 — 2D GauCho amodal ellipse head と curriculum config

DoD-A（2D ellipse 性能）に必要なため、3D と並行して 2D GauCho head を追加した。

- `gaucho3d_geometry.py`: 2D chart `scale_shape_cholesky2d` / 逆写像 / `gaussian_to_ellipse2d`
  （表示・NMS・評価専用のデコード）/ `obb_gaussian_to_cholesky2d`。
- `gaucho3d_loss.py`: `Ellipse2DKLDLoss`（2x2 の前進代入で書いた inverse-free KLD）。
- head: `reg_ellipse2d_branch`（class ごと 5 値 `(tx, ty, r, u, v)`）。

**dense head との違いの扱い**: 参照実装は FPN cell の stride から `rho_Q = s/2` を得る。
DETR query には stride が無いため、query 自身の**予測 box**（`detach` 済み）から
`rho_Q = 0.5 * sqrt(w*h)` と参照中心を作る。detach しているので ellipse の目的関数が
自分の参照枠を通して box branch を作り替えることはない（テストで検証）。

**教師が参照実装と完全一致であることの確認**:
`yopo/datasets/pose_estimation/nocs_custom_fruit_dataset.py:94` の `obb_gaussian` は
`Sigma = R diag((w/2)^2, (h/2)^2) R^T`、すなわち **OBB の最大面積内接楕円**である。
これは参照実装の `rboxes_to_inscribed_ellipses` + `ellipses_to_gaussian_geometry` と同一量。
したがって DoD-A は同一の目的関数・同一の教師定義での比較になる。
base config は両 pipeline で既に `with_obb_gaussian=True` を設定済みで、追加の pipeline 変更は不要。

#### この段階で検出した欠陥（修正済み）

**`obb_gaussians` 教師が全ゼロになる（重大）**: `_get_targets_single` は
`loss_projection` か `loss_obb_aux` が有効なときだけ `obb_gaussian_targets` を埋めていた。
新しい `loss_ellipse2d` / `loss_ellipsoid_projection` だけを有効にすると、
**例外も警告も出ないまま全ゼロの教師で学習してしまう**。
`_requires_obb_gaussians` プロパティへ集約し、4 つの consumer すべてで gate するようにした。

#### curriculum config

| config | 有効な目的関数 |
|---|---|
| `nocs_fruits_Jun30_2025_rgbd_gaucho_stageA_ellipse2d.py` | 2D GauCho KLD のみ（3D off） |
| `nocs_fruits_Jun30_2025_rgbd_gaucho_stageB_ellipsoid.py` | + 3D direct KLD |
| `nocs_fruits_Jun30_2025_rgbd_gaucho_stageC_projection.py` | + dual-quadric 投影整合（teacher detach） |
| `temp/smoke_gaucho3d_1iter.py` | 1 epoch / batch 1 の GPU smoke |

head は `loss_ellipsoid_projection` 単独を**拒否**する（`ValueError`）。
投影整合だけでは 2D と 3D が同じ誤った楕円へ共倒れできるため、direct 3D 教師との併用を強制する。

#### 試験状況

`tests/test_gaucho3d.py` は **46 passed / 2 failed**。失敗 2 件は隔離環境の torch 2.1 に
`Tensor.all(dim=tuple)` が無いことによるもので、呼び出し先は既存の
`projected_ellipsoid_loss.gaussian_wasserstein_distance`。実 venv（torch 2.8）で再実行する。

head 周りの既存テストは全 8 ファイルでベースラインと一致（**回帰ゼロ**）。

### 2026-08-31 23:32 — Stage B smoke train 成功（暫定環境）

`uv sync` の完了を待たずに smoke を通すため、**暫定環境**を組んだ。ユーザー判断による。

#### 暫定環境の構成

`~/Desktop/venv/rotated-rtmdet-mmrotate-sandbox/.venv`（別プロジェクトの venv、**読み取り専用で使用**）に
不足していたのは `plyfile` と `scikit-learn` だけだった。これを `--target` で
`YOPO_clone/.smoke-deps` へ分離インストールし、`PYTHONPATH` で合成した。

| 項目 | 値 |
|---|---|
| torch | 2.1.0+cu121（pinned は 2.8.0+cu128） |
| mmcv / mmengine | 2.2.0 / 0.10.7（pinned と同一） |
| numpy | 1.26.4（`numpy<2` 要件を満たす） |
| GPU | RTX 4090, cap (8,9) |
| mmcv CUDA ops | **動作確認済み**（`nms` と `diff_iou_rotated_2d` を GPU で実行、rotated IoU = 0.3913） |

注意点:

1. `--target` は既定で numpy 2.2.6 / scipy を引き込む。`PYTHONPATH` は site-packages より
   優先されるため、そのままだと `numpy<2` 要件を破る。`--no-deps` と
   `scikit-learn==1.5.2` で回避した。
2. `torch.serialization.add_safe_globals` は torch 2.4 以降の API。リポジトリ本体は変更せず、
   scratchpad の wrapper で no-op を注入した（pinned 環境では不要になる）。
3. **`.venv` には触れていない**。進行中の `uv sync` と競合しない。

#### smoke 結果（`temp/smoke_gaucho_stageB_1iter.py`、実 RGB-D データ 4 frame）

```
Epoch(train) [1][1/4]  loss_ellipse2d: 1.6533  loss_ellipsoid: 1.9634
Epoch(train) [1][2/4]  loss_ellipse2d: 1.6496  loss_ellipsoid: 1.8234
Epoch(train) [1][3/4]  loss_ellipse2d: 1.6454  loss_ellipsoid: 1.8223
Epoch(train) [1][4/4]  loss_ellipse2d: 1.6446  loss_ellipsoid: 1.8222
grad_norm: 21536.2861       Saving checkpoint -> epoch_1.pth (390 MB)
```

確認できたこと:

- dataloader -> Cholesky chart -> KLD -> backward -> checkpoint の全経路が実データで動く。
- `loss_ellipse2d` と `loss_ellipsoid` が **6 つの decoder layer すべて**（`d0`〜`d4` と最終層）
  に出ており、per-layer 経路の配線が正しい。
- 両 loss とも単調減少し、値・勾配ともに有限。
- VRAM は batch 1 で約 1.36 GB。

#### smoke で判明した設定上の注意（修正済み）

- `default_hooks.checkpoint` を単純 merge すると、継承元の `TopKCheckpointHook` の
  `topk` キーが残り、`CheckpointHook` へ渡って `TypeError` になる。`_delete_=True` が要る。
- base config の depth backbone は MAE 事前学習 checkpoint を参照する。smoke には不要なので
  `init_cfg=None` で無効化した（幾何経路の検証が目的であり精度は主張しない）。

#### この環境で実行できないもの

Stage C（`DualQuadricProjectionGWDLoss`）は既存の
`projected_ellipsoid_loss.gaussian_wasserstein_distance` を経由し、そこで使われる
`Tensor.all(dim=tuple)` が torch 2.1 に無い。pinned 環境（torch 2.8）完成後に実行する。

### 2026-08-31 23:38 — batch/VRAM probe と推論・評価経路

#### batch probe（RTX 4090 24 GB、torch 2.1、FP32、`num_queries=100`）

| batch | peak VRAM | imgs/s |
|---:|---:|---:|
| 2 | 2,122 MB | 26.0 |
| 4 | 3,945 MB | 33.3 |
| 8 | 7,764 MB | 36.8 |
| 12 | 11,457 MB | 27.6 |
| 16 | 15,472 MB | 36.9 |
| 20 | 19,252 MB | 31.0 |
| 24 | **OOM** | — |

約 960 MB/枚で線形。最大安定 batch は 20。設計書の batch 48 は 32 GB + BF16 AMP 前提であり、
24 GB・FP32 のここでは成立しない。pinned 環境（BF16 AMP）で取り直す。

#### 推論経路

`expose_gaucho_predictions=True` で `predict` が次を返す。

- `ellipses` `(a,b,cx,cy,theta)` / `ellipse_gaussians` / **`ellipse_obb`**（楕円の外接回転矩形）
- `ellipsoid_shapes` (Sigma) / `ellipsoid_centers` / `ellipsoid_symmetry` /
  `ellipsoid_front_margin` / `projected_ellipses`（dual-quadric 投影）

角度は表示・NMS・評価でだけ現れる。学習経路には一切入らない。

#### 評価指標

`yopo/evaluation/metrics/ellipse_rotated_iou_metric.py` に
`EllipseEnvelopeRotatedIoUMetric` を追加。既存 `RotatedIoUMetric` を継承し、
予測は `ellipse_obb`、GT は `obb_gaussians` を 5D 回転矩形へデコードして rotated mAP50 を出す。
**参照実装と同一の定義**（学習済み楕円の外接矩形 対 GT OBB）。
軸平行 40x20 OBB で往復が厳密一致することを確認済み。

#### 較正学習の 1 回目で検出した設定不整合（修正済み・重要）

継承元 `nocs_custom_fruit_rgbd_3dbbox_transfer.py` は
**`num_queries=100` と `test_cfg.max_per_img=300`** を組み合わせている。この組は
学習中は無害だが、検証で `cls_score.view(-1).topk(300)` が 100 要素に対して呼ばれ
`RuntimeError: selected index k out of range` で落ちる。

さらに本データは 1 フレームに最大約 148 個の stem を含むため、100 query では
そもそも表現できない。同じデータ向けの
`nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_curriculum_base.py` は
`max_objects = 256` を `num_queries` と `max_per_img` の**両方**へ入れている。
Stage A へ同じ値を入れ、Stage B/C と smoke へ継承させた。

これは本作業で追加した経路の欠陥ではなく、継承元の設定不整合である。

### 2026-08-31 23:5x — 推論経路の欠陥と、試験カバレッジの回復

#### 較正学習 2 回目で検出した欠陥（自分の追加箇所・修正済み）

`_attach_gaucho_predictions` が 3D 中心を `result.T[:, :, 3]` から取っていた。
`T` は同次 4x4 なのでこの列は長さ 4 になり、
`ValueError: center must end in 3 values, got (256, 4)` で検証が落ちた。
同じ camera-frame 中心は `results.translations` に `(N, 3)` でそのまま入っているので、
曖昧なスライスをやめて直接使うよう変更した。

**この欠陥は既存の試験では捕まらなかった**（推論経路に試験が無かった）。
再発防止として `tests/test_gaucho3d.py` へ推論経路の試験を追加した。

- `predict` が `ellipses` / `ellipse_gaussians` / `ellipse_obb` / `ellipsoid_shapes` /
  `ellipsoid_centers` / `ellipsoid_symmetry` / `ellipsoid_front_margin` /
  `projected_ellipses` / `projected_valid` を正しい shape で出すこと。
- `ellipse_obb` が「出力された楕円そのものの 2a x 2b」であること（独立に回帰した箱ではない）。
- `ellipsoid_centers == translations`（4x4 の列を取ると黙って壊れる箇所の固定）。
- `expose_gaucho_predictions=False` のとき、これらの field が**存在しない**こと。
- 完全一致予測に対し `EllipseEnvelopeRotatedIoUMetric` が **AP = 1.0** を返すこと。
- 予測 field 欠落時に `expose_gaucho_predictions` を促す `KeyError` を出すこと。

#### 試験カバレッジの回復（重要）

隔離環境に `sklearn` が無いため `yopo.evaluation` の import が失敗し、
**それが多数の既存試験の失敗の正体だった**。`.smoke-deps` を試験の `PYTHONPATH` へ加えたところ、
`test_rgbd_cop_prediction_mode` は 26 passed から **53 passed** になった。
それまで「環境由来の既存失敗」と扱っていた分の大半がこれである。

#### 回帰確認（sklearn を通した状態での再測定）

| テストファイル | baseline | 変更後 |
|---|---|---|
| `test_rgbd_cop_prediction_mode` | 53 passed / 1 failed | **53 passed / 1 failed** |
| `test_matchability_head_integration` | 4 / 1 | **4 / 1** |
| `test_matchability_quality` | 3 / 7 | **3 / 7** |
| `test_projected_ellipsoid_gwd_loss` | 2 / 9 | **2 / 9** |
| `test_rgbd_nms_free_quality` | 7 / 0 | **7 / 0** |

**全ファイルで完全一致＝回帰ゼロ**。`tests/test_gaucho3d.py` は 50 passed / 2 failed
（失敗 2 件は torch 2.1 の `Tensor.all(dim=tuple)` 非対応のみ）。

#### 暫定成果物の扱い

`YOPO_clone/.smoke-deps/`（`plyfile` / `scikit-learn` / `joblib` / `threadpoolctl` /
`cloudpickle`）は暫定環境専用であり、**コミット対象ではない**。
pinned 環境の `uv sync` 完了後に削除する。

---

## 2026-09-01 — pinned 環境の完成と、事前学習の載せ替え

### pinned 環境

`uv sync` 完了。`just env-doctor`:

```
torch 2.8.0+cu128 | torchvision 0.23.0+cu128 | mmcv 2.2.0 | mmengine 0.10.7 | yopo 3.3.0
cuda_available True | device NVIDIA GeForce RTX 4090
```

**§2 で予告した sm_120 問題は実際に発生した**。pinned wheel のまま CUDA op を叩くと
`no kernel image is available for execution on the device`。確保しておいた
`mmcv-2.2.0-0sm89t280` へ venv 側だけ差し替えて解決（`MMCV_CUDA_OPS_OK_AFTER_SM89_SWAP`)。
`pyproject.toml` は未変更。リポジトリ自身の smoke も `SMOKE INFER OK`。

暫定環境で必要だった 3 つの回避策はすべて不要になった。

| 試験 | pinned 環境での結果 |
|---|---|
| `tests/test_gaucho3d.py` | **54 passed / 0 failed**（torch 2.1 で保留だった 2 件を含む） |
| head 周り回帰 8 ファイル | 114 passed / 1 failed |

残る 1 件 `test_chain_pre_decoder_skips_encoder_pose_branches` は `git stash` で確認したところ
**変更前後で同一に失敗＝既存の失敗**であり回帰ではない。

### 事前学習の載せ替え（DoD の前提条件）

`nocs_custom_fruit_rgbd_3dbbox_transfer` 系が参照する初期化成果物は**2つとも新規クローンに無い**。
実際にスクラッチ学習すると epoch 2 で ellipse mAP50 = 0.001、3D IoU@25 = 0.001 だった。

`YOPO_best_models_20260826/large_2025_2026_compact_cop_best3d_epoch5.pth` を調べたところ、
24.62M params・**num_classes = 1**・**256 queries**・RGB-D・自身の val で 2D AP50 0.6929 で、
本データにそのまま使える。Stage 設定を **compact 系へ載せ替え**（モデルは compact chain、
データ・パイプライン・per-frame intrinsics は Jun30 変換）。照合結果:

```
missing: 30  (すべて GauCho の新規枝)   other missing: 0   unexpected: 0
```

事前学習の全テンソルが過不足なく載る。ランダム初期化は新規 GauCho head だけであり、
設計書の「head warm-up → FULL fine-tune」の形になった。

### 検出した欠陥 3 件

1. **指標キーの誤り（修正済み）**: `EarlyStoppingHook`/`CheckpointHook` に `ellipse/AP` を
   指定していたが、`RotatedIoUMetric` が返すのは `ellipse/rbbox_mAP_50`。検証直後に
   `KeyError` で停止した。

2. **報告の取り違え（訂正済み）**: 最初の検証で出た AP テーブル `0.645` を ellipse 指標の値として
   報告したが、実際は **`NOCSMetric` の `AP50 = 0.6451`**（既存 2D 検出 AP、warm-start 由来）だった。
   同形式の AP テーブルが 2 つ出るため取り違えた。私の ellipse 指標は
   **`ellipse/rbbox_mAP_50 = 0.0014`** だった。

3. **座標系の不一致（重大・修正済み）**:
   `ellipse/rbbox_mean_matched_rIoU = 0.5847` に対し `recall = 0.0499` という不自然な組から疑い、
   実データで確認した。

   ```
   ori_shape (512, 736) | img_shape (445, 640) | scale_factor (0.8696, 0.8691)
   gt obb 中心 x 範囲: 143.7 .. 632.6   -> img_shape 幅 640 に収まる = リサイズ後空間
   ```

   `gt_instances.obb_gaussians` は val pipeline の `ResizeforPose` 後の 640x445 空間、
   一方 `predict(rescale=True)` の予測は元画像 736x512 空間。1.15 倍のずれで、画像中心付近の
   少数だけが IoU 0.5 を超え、それ以外は全滅していた。

   指標側で GT を予測と同じ frame へ写す `rescale_compact_gaussian` を追加。
   デコード後の `(w,h,theta)` ではなく **Gaussian のまま `mu -> S mu`, `Sigma -> S Sigma S^T`**
   で変換するため、x/y のスケールが異なっても厳密である。

   **既存 `RotatedIoUMetric` も同じ潜在問題を持つ**（`pred_instances.bboxes` は rescale 済み、
   `gt_instances.bboxes` はリサイズ後）。compact 系の 800x600 設定は
   `AssertIdentityImageGeometry` でリサイズが無く `scale_factor == 1` のため顕在化していない。
   今回は自分の指標のみ修正し、既存の挙動は変更していない。

### 学習の実行条件（pinned）

batch 16 / bfloat16 AMP で **VRAM ピーク 10.9 GB**（24 GB に対し余裕大）。
1 epoch 75 iteration、20 epoch の ETA 約 25 分。
検出器は compact chain の base LR 5e-5、唯一ランダム初期化の
`bbox_head.reg_ellipse2d_branch` だけ `lr_mult=10`。

### 2026-09-01 — DoD-A の比較基準が確定した

§1.2 と §5 で「未解決」としていた比較条件は、取得したリリース資産の `__metadata__` を
読むことで確定した。各 checkpoint は学習時の config を丸ごと埋め込んでいる。

| checkpoint | dataset | class | 種類 | mAP50 | eval protocol |
|---|---|---|---|---:|---|
| `rtmdet-hgnetv2-b1--stem1--512x736` | **`tomato_jun30`** | **stem** | 回転OBB | **0.8170** | score_thr .1 / NMS IoU .2 / pre 2000 / max_det 2000 |
| `gaucho-ellipse` (v0.4.0) | `tomato_2025_2026_800x600_dota` | tomato | ellipse 包絡 | 0.8514 | score_thr .05 / NMS IoU .2 / pre 2000 / max_det 2000 |
| `o2deim--stem1` | `o2deim_lr_probe_300_50` | stem | 回転OBB | 0.7277 | score_thr .01 / NMS IoU .3 |

rtmdet 側の `conf/data/tomato_jun30.yaml` の `root` は
`/workspace/data/tomato_obb_detection/fruits_detection_data_Jun30-2025_dota`、
`class_names: [stem]`。これは今回 Downloads から展開した DOTA データそのものであり、
YOPO 側 RGB-D データの元データである。

**結論**:

1. **0.8514 は比較対象にならない。** 別データ（tomato 2025+2026 800x600）で測られており、
   その config は `root: null`＝データを実行時に外部供給する形なので、手元に無い。
   §5 の選択肢 (b) は実行不可能である。
2. **同一データでの rtmdet 実測は 0.8170** である。DoD-A の対照値はこれとする。
   ただし回転 OBB を直接回帰したモデルであり、ellipse 包絡ではない。
3. rtmdet の ellipse モデルを我々の stem データで評価することは可能だが、
   tomato 学習の out-of-domain 適用になるため、参考値以上には扱えない。

#### 比較条件の差分（結果に併記すること）

| | rtmdet 0.8170 | 現行 YOPO |
|---|---|---|
| 入力解像度 | 512x736 | 640x445（736x512 をリサイズ） |
| NMS | IoU 0.2 / pre 2000 / max_det 2000 | **なし**、256 query |
| 出力 | 回転 OBB の直接回帰 | 学習済み楕円の外接矩形 |
| score_thr | 0.1 | 0.05（`RotatedIoUMetric` では AP 自体に影響しない） |

DETR は one-to-one 割当を前提とするため NMS 無しが設計上正しいが、
厳密に条件を揃えた回転 NMS 付きの数値も別途取る。

### 2026-09-01 01:0x — 参照値の再現と rtmdet 環境の欠陥

#### 参照値をビット一致で再現した

checkpoint 埋め込みの評価プロトコルをそのまま適用して、リリース値を再現した。

```
RESULT          {"AP50/class_0": 0.8170062340102078, "mAP50": 0.8170062303543091}
embedded metric:                                     mAP50 0.8170062303543091
protocol: score_thr 0.1 / NMS IoU 0.2 / nms_pre 2000 / max_detections 2000
model: hgnetv2_b1_tomato_full, input 512x736, num_classes 1
```

これにより次が同時に証明された。

1. Downloads から展開・配置した DOTA データが、rtmdet 側が使ったデータと**同一**である。
2. 評価プロトコルの再現が正しい。
3. **DoD-A の対照値は 0.8170062** で確定（同一データ・同一クラス `stem`）。

再現スクリプトは `<scratchpad>/rtmdet_eval.py`。評価の knob は config から発明せず、
checkpoint に埋め込まれた値を読んで適用する。

#### rtmdet の pixi 環境の欠陥（回避策を確立）

`pixi run smoke` と `pixi run train-smoke` は GPU 実行が全滅していた。

```
CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed
cudnn_status: CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED
Unable to load any of {libcudnn_engines_runtime_compiled.so.9.24.0, ...}
```

原因は、cuDNN 9 が sublibrary を**実行時 dlopen** するのに対し、pixi の activation が
pip の `site-packages/nvidia/*/lib` を loader パスへ追加しないこと。`.so` 自体は存在する。

`LD_LIBRARY_PATH` へ 13 個の nvidia lib ディレクトリを通すことで解決
（`<scratchpad>/rtmdet_env.sh`）。これで上記の参照値再現も smoke も通った。

| rtmdet 試験 | 結果 |
|---|---|
| `pixi run test` | 199 passed, 3 skipped, 148 deselected |
| `pixi run smoke` | `"successful": true` |
| `pixi run train-smoke` | 完走（epoch=1 step=4） |

#### Stage A warm-up の頭打ちと、その原因

| epoch | ellipse mAP50 | 検出器 AP50 | 比 | matched rIoU |
|---:|---:|---:|---:|---:|
| 2 | 0.5994 | 0.6445 | 93.0% | 0.6860 |
| 4 | 0.6186 | 0.6476 | 95.5% | 0.6869 |
| 6 | 0.6223 | 0.6525 | 95.4% | 0.6856 |
| 8 | 0.6263 | 0.6564 | 95.4% | 0.6870 |
| 10 | 0.6304 | 0.6543 | **96.3%** | 0.6885 |

**ellipse AP は検出器自身の AP の 96% に張り付いている。** 楕円表現側に伸びしろは無く、
残る差は検出器が本データへ適応していないことに由来する。released checkpoint は
2025+2026 tomato 学習であり、本データは stem・別解像度である。

これは想定内で、現段階は設計書の **warm-up** に相当する（新規 head だけ `lr_mult 10`、
検出器は 5e-5 据え置き）。参照実装も同じ経過をたどっている。

| 参照実装 (rtmdet GauCho) | raw mAP50 |
|---|---:|
| warm-up 3 epoch | 0.004 |
| FULL epoch 1 | 0.405 |
| FULL epoch 8 | 0.782 |
| FULL epoch 22 | **0.851** |

#### FULL 段階の設定根拠

`temp/train_gaucho_compact_stageA_full.py`。

- **LR 1e-4**: リポジトリ自身の値。compact curriculum は stage1-3 が 1e-4、
  最終 refinement の stage4 だけ 5e-5。発明した値ではない。
- **ellipse head の `lr_mult` を 10 → 1.0**: warm-up は終わったので全 parameter を均等に更新する。
  参照実装も warm-up の head LR (Muon .005) より FULL (.001) を下げ、対象を全体へ広げている。
- 40 epoch、`ellipse/rbbox_mAP_50` で patience 6。

### 2026-09-01 01:2x — compact 系の Stage B/C を追加

DoD-B のために、3D 経路も事前学習が載る側へ用意した。既存の
`nocs_fruits_Jun30_2025_rgbd_gaucho_stageB/C` は transfer 系に乗っており初期化成果物が無い。

| config | 有効な目的関数 | 初期化 |
|---|---|---|
| `..._compact_stageA` | 2D GauCho KLD | released compact checkpoint |
| `temp/train_gaucho_compact_stageA_full` | 同上・全 parameter を LR 1e-4 | Stage A best |
| `..._compact_stageB` | + 3D 楕円体 KLD | 前段 best（起動時指定） |
| `..._compact_stageC` | + dual-quadric 投影整合（teacher detach） | 前段 best（起動時指定） |

Stage C の照合:

```
params (M): 25.96
missing 60 | ellipse2d 30 | ellipsoid 30 | other 0 | unexpected 0
```

2D・3D の GauCho 枝だけが新規で、事前学習は全テンソル過不足なく載る。

Stage B/C の `load_from = None` は意図的である。released checkpoint を書くと前段の学習が
黙って捨てられるため、前段の best を起動時に渡す。試験でこの契約を検証している。

`tests/test_gaucho3d.py`: **57 passed / 0 failed**。

### 2026-09-01 01:5x — FULL 段階の失敗と、入力解像度の特定

#### FULL 段階（LR 1e-4、全 parameter）は伸びなかった

| epoch | ellipse mAP50 | 検出器 AP50 |
|---:|---:|---:|
| warm-up best (ep18) | **0.6419** | 0.6596 |
| FULL 2 | 0.6267 | 0.6541 |
| FULL 8 | 0.6164 | 0.6574 |
| FULL 16 | 0.6305 | 0.6607 |

16 epoch 回して warm-up best を一度も超えなかった。「warm-up が飽和したので FULL へ進めば伸びる」
という見立ては外れた。LR やスケジュールをいじる前に、データを実測して原因を探した。

#### 原因は入力解像度だった（実測で特定）

val 20 frame・679 個の GT OBB をリサイズ後空間で実測した。

```
w 中央値 29.6 px (p10 18.5 / p90 45.3)
h 中央値 23.4 px (p10 13.7 / p90 37.2)
最小辺が 16 px 未満の物体: 18.1%
```

対象の 2 割近くが 16 px 未満の極小物体である。

| | rtmdet 0.8170 | それまでの YOPO |
|---|---|---|
| 入力 | **512x736（ネイティブ）** | 640x445（0.87 倍に縮小） |
| 画素数 | 375k | 285k（**24% 減**） |

小物体検出が最も解像度に敏感な領域で 24% の画素を捨てていた。
compact checkpoint は 800x600 学習であり、640x445 より 736x512 のほうが事前学習分布にも近い。
さらに **compact chain の元設定は `AssertIdentityImageGeometry` でリサイズを一切していない**。
640x480 への縮小は transfer 系のパイプラインを流用した副作用であって、設計上の要請ではなかった。

#### ネイティブ解像度で再実行した結果

`temp/train_gaucho_compact_native.py`。warm-up best から継続、736x512、batch 12。

| | 640x445 warm-up best | 640x445 FULL ep2 | **736x512 native ep2** | **native ep4** | **native ep6** |
|---|---:|---:|---:|---:|---:|
| 検出器 AP50 | 0.6596 | 0.6541 | 0.6958 | **0.7018** | — |
| ellipse recall | 0.7919 | 0.7881 | 0.8094 | **0.8164** | — |
| ellipse mAP50 | 0.6419 | 0.6267 | 0.6360 | 0.6504 | **0.6533** |

検出器 AP50 が **+4.2 ポイント**改善し、ellipse も warm-up best を更新した。仮説は確認された。

`ResizeforPose` を削除せず native サイズへ設定したのは、削ると `scale_factor` が metainfo から
消え、`rescale=True` の推論と ellipse 指標の frame 変換の両方が壊れるためである。
この設定では `scale_factor == (1.0, 1.0)` になり、機構はそのまま保たれる。

VRAM は batch 12 / native で 10.1 GB。まだ batch を上げる余地がある。

---

## 2026-09-01 03:xx — リファクタリング・実装改善レビュー

5 観点（重複・head 統合・効率・頑健性・API/試験）を並列でレビューし、各指摘を
敵対的検証にかけた。**10 件中 5 件確認 / 5 件棄却**。棄却例は次のとおりで、
検証が機能していることの裏付けになる。

- 「評価指標が楕円デコードを再実装している」→ 重複は事実だが欠陥の枠組みが誤り
- 「arity 許容が None を loss dict へ書き込む」→ そのパスは到達不能
- 「`_backproject` が query ごとに LU 分解している」→ 形は正しいが実コストの主張が誤り
- 「decoder 層ごとに 11 回の device 同期」→ 機構は正しいが影響の見積もりが誤り
- 「`Sigma - t t^T` で桁落ちする」→ 数値的核心は真だが shipped code では到達不能

### 適用した修正

#### 1. Stage B/C が初回イテレーションで死ぬ（最重要・未実行のため未発見だった）

compact 設定は `AmpScheduleFreeOptimWrapper` の `dtype="bfloat16"` を継承する。
autocast は matmul を降格するが **`torch.linalg.*` には触れない**。そして CUDA には
cholesky / triangular solve / LU / eigh の bfloat16 カーネルが存在しない。
検証エージェントが実 GPU（torch 2.8.0+cu128, RTX 4090）で再現している。

```
ellipsoid_from_rotation_size(fp32 入力) は autocast 下で bfloat16 を返す
  -> torch.linalg.cholesky
  -> NotImplementedError: "cholesky_cusolver" not implemented for 'BFloat16'
```

予測側の Cholesky が助かっていたのは偶然で、`scale_shape_cholesky3d` が `.exp()` を
呼び、autocast の fp32 promote 対象だったためである。

修正: リポジトリ既存の `_recover_translation` と同じ float32 island の idiom を、
`_backproject` と 3D ブロック全体へ適用。**損失モジュール呼び出しまで island に含める**
（部分修正だと数行先で次の未対応カーネルが落ちる）。`_gaucho_query_intrinsics` へ渡す
reference も明示的に `.float()` する（`_intrinsic_matrix` が reference の dtype へ K を
キャストするため）。損失は float32 のまま返す（bf16 の兄弟と加算すれば自動昇格する）。

#### 2. Stage B/C の intrinsic が別フレーム（先に自分で見つけた座標系バグと同類型）

`nocs_fruits_Jun30_2025_rgbd_gaucho_stageB/C` は `train_intrinsic_to_image_space` を
設定しておらず、既定 False。パイプラインは `ResizeforPose` + `ResizeOBBGaussians` で
教師をリサイズ後空間へ移すのに、dual quadric は**元画像の K** で投影されていた。
中心は同じ K で往復するため誤差が**形状項だけに隠れ**、metric スケールで教師される
`loss_ellipsoid` と綱引きする。

修正: 該当 2 設定へ `train_intrinsic_to_image_space=True` を設定。加えて、
**head が `loss_ellipsoid_projection` 有効かつ当該フラグ False のとき `ValueError` で
拒否する fail-closed ガード**を追加した。将来の設定でも同じ罠を踏めない。

#### 3. size prior が chart の clamp の内側にあった

`direct` / `dual_plane` では prior を raw に加算してから ±`log_clip` で clip していた。
結果、(a) 可動域が prior 依存で非対称になり（prior 0.03 で縮小 4.49 nats / 拡大 11.5 nats）、
`gaucho_size_prior <= exp(-8)` では zero-init が完全に clip されて勾配が消える、
(b) bias が log 対角の 3 スロットにしか入らないため shear 項だけ prior が効かない。

修正: `rho0 * chol(raw)` の後乗算へ変更。`L -> rho0 L` は `Sigma -> rho0^2 Sigma` であり、
全成分が等しくスケールする。`scale_shape` が元から clip の**後**に rho を掛けていたので、
これで 3 つの chart が本当に交換可能になった。

#### 4. 2D デコードが学習経路と推論経路に二重定義

`loss_by_feat_single` と `_attach_gaucho_predictions` に同じ 7 行が独立に存在した。
片方だけ変えても**全試験が緑のまま**、学習が最適化する楕円と評価が採点する楕円が
食い違う。3D 側は既に `_gaucho_cholesky` で単一化されていたので、同じ形へ揃えた
（`_decode_gaucho_ellipse2d`）。`detach` は呼び出し側の判断として残す。

### 検証

| | |
|---|---|
| `tests/test_gaucho3d.py` | **62 passed / 0 failed**（各指摘に回帰試験を追加） |
| head 周り既存 8 ファイル | 114 passed / 1 failed（**既存の失敗**、新規回帰ゼロ） |

追加した回帰試験:

- 学習経路と推論経路が同一のデコードを使うこと
- size prior が shear を含む全成分をスケールすること、極小 prior でも clip されないこと
- **bfloat16 autocast 下で 3D 損失が有限かつ backward できること**（未カバーだった）
- `loss_ellipsoid_projection` が image-space intrinsic を要求して fail-closed すること
- 3D を有効にする全 4 設定が intrinsic frame を固定していること

---

## 2026-09-01 04:xx — curriculum の実行状況と、暫定成果物の整理

### 実行構成（native 736x512、pinned 環境）

```
Stage A (2D GauCho)  ->  Stage B (3D 楕円体)  ->  Stage C (dual-quadric 投影整合)
  best 0.6533 (ep6)        実行中                   B 完了後に自動起動
```

Stage B / Stage C は `temp/train_gaucho_stageB_native.py` /
`temp/train_gaucho_stageC_native.py`。いずれも前段の best から起動する。

### Stage B（3D 経路の初実行）

レビューの bfloat16 修正が入るまで**起動すらできなかった**経路である。

| epoch | 3D IoU@0.25 | 3D IoU@0.50 | 2D ellipse mAP50 | 検出器 AP50 |
|---:|---:|---:|---:|---:|
| 2 | 0.0220 | 0.0020 | 0.6353 | 0.6865 |
| 4 | 0.0228 | — | 0.6409 | 0.6980 |
| 6 | 0.0228 | — | — | — |

`loss_ellipsoid` は 1.4522 -> 1.2864 と単調低下しており、inverse-free Cholesky KLD が
実データで機能している。**3D 枝を足しても 2D ellipse は退行していない**
（0.6353 -> 0.6409）。設計書が要求する「2D と 3D を独立 head として保持する」構成が
効いている。

released compact model の自己申告値は 3D IoU@0.25 = 1.66% / @0.50 = 0.17% だが、
**検証 split が異なるため直接比較はできない**。同一 split の対照ではない。

### 暫定成果物の整理

pinned 環境が自己完結していることを確認した上で削除した。

- `YOPO_clone/.smoke-deps/`（48 MB、暫定環境専用の pip target）
- `temp/calib_gaucho_stageB_interim.py`（torch 2.1 用 fp32 較正）
- `temp/train_gaucho_compact_stageA_warmup.py`（fp32 / batch 6 の暫定版）

後 2 者は pinned 環境では**誤った設定**になるため、残すと次に実行する人の罠になる。
削除後も `tests/test_gaucho3d.py` は 62 passed。

`temp/smoke_gaucho_stageB_1iter.py` は残し、pinned 環境では Stage C も動く旨を追記した。

### 上流 YOPO SwinL checkpoint について

`Broken pipe` が繰り返し発生し、892 MB 中 41 MB で停滞している。回線側の問題である。
これらは上流の**単眼 RGB** モデルであり `rgb-d` ブランチの作業には使わない。
リポジトリ自身の `just smoke-infer` が要求する R50 (205 MB) は取得・検証済みで、
DoD には影響しない。

---

## 2026-09-01 — curriculum 全段の完走と最終結果

### 各段の最良値（すべて native 736x512、pinned 環境、val 330 frame）

| 段 | 目的関数 | best 指標 | 2D ellipse mAP50 | 3D IoU@0.25 | 検出器 AP50 |
|---|---|---|---:|---:|---:|
| Stage A warm-up (640x445) | 2D KLD | ep18 | 0.6419 | — | 0.6596 |
| **native 736x512** | 2D KLD | ep6 | **0.6533** | — | **0.7018** |
| Stage B | + 3D KLD | ep18 | 0.6443 (ep22) | **0.0295** | 0.6881 |
| Stage C | + dual-quadric 投影 | ep8 | 0.6213 (ep20) | 0.0294 | 0.6805 |

### 結論

1. **2D GauCho は成功している。** 角度・軸長を一切回帰せず、検出器が見つけた物体の
   **95〜97%** を楕円として再現する（matched rotated IoU 0.69、単調改善）。
2. **3D GauCho も機能する。** Stage B で 3D IoU@0.25 が 0.0220 -> **0.0295（+34%）**。
   損失は inverse-free Cholesky KLD のみで、回転も軸長も角度も回帰していない。
   2D と 3D を独立 head として保持したため、3D 追加による 2D の退行は起きなかった
   （中盤で 0.625 まで下がるが ep22 で 0.6443 へ復帰、matched IoU は単調改善）。
3. **Stage C の dual-quadric 投影整合は、この設定では効果が無かった。**
   3D は 0.0295 -> 0.0294 と横ばい、2D は 0.6443 -> 0.6213 と低下した。
   設計書自身がこれをアブレーション A2「正確な 2D-3D 整合の効果」として
   *測定すべき対象*に位置づけており、前提として仮定していない。
   本データにおける答えは「loss_weight 1.0 では利得なし」である。
   重みを下げる、Stage C をさらに後段へ回す、あるいは surface/free-space 項
   （実装済み・未使用）と併用する余地は残る。
4. **DoD-A は未達。** 目標 0.8170（同一データでの rtmdet 実測、ビット一致で再現済み）
   に対し最良 **0.6533**。ただし残差の所在は特定できている（下記）。
5. **DoD-B は判定不能。** released compact model の 3D IoU@0.25 = 1.66% に対し
   本実装は 2.95% だが、**検証 split が異なるため優劣を主張できない**。
   同一 split の対照値が存在しない。

### DoD-A 未達の所在

ellipse mAP50 は一貫して検出器自身の AP50 の **95〜97%** で推移した。
すなわち律速は楕円表現ではなく検出器の絶対性能である。

| | rtmdet 0.8170 | 本実装 0.6533 |
|---|---|---|
| 検出器 | このデータ専用に学習した dense RTMDet | 2025+2026 tomato 学習の compact DETR を stem へ転移 |
| 検出器自身の AP50 | (= 0.8170、OBB 直接回帰) | 0.7018 |
| 出力 | 回転 OBB を直接回帰 | 学習済み楕円の外接矩形 |
| NMS | IoU 0.2 / max_det 2000 | なし（DETR の one-to-one） |

楕円表現の妥当性と検出器の絶対性能は分離して評価できる状態にある。
DoD-A を「同一検出器上で楕円が OBB 直接回帰に匹敵するか」と読むなら、
本実装は検出器 AP の 97% を再現しており妥当と言える。
「絶対値 0.8170 に到達するか」と読むなら未達であり、そのためには
検出器そのものをこのデータで学習し直す必要がある。

### 最終試験状況

| | |
|---|---|
| `tests/test_gaucho3d.py` | **62 passed / 0 failed** |
| head 周り既存 8 ファイル | 114 passed / 1 failed（**既存の失敗**、新規回帰ゼロ） |

---

## §X 2026-09-01 — 理論整合性レビューと、本作業書の記載訂正

設計書 26 頁を 5 分割し、式番号レベルで実装と突き合わせた。各指摘は敵対的検証にかけ、
**9 件確認 / 1 件棄却**。棄却された 1 件（式(28)(29) soft lift の不在）は、
Table 2 の段階方針として明示的に文書化済みの意図的判断と判定された。

### 本作業書に誤りがあった（訂正済み）

自分で `grep` して確認した事実:

```
decode_center_from_anchor : 0 callers（geometry モジュール外）
ray_frame                 : 0 callers
loss_rotation             : AMPStableRotation3DLoss, weight 5.0  ← Stage B で有効
```

- **R2 について（ユーザー判断により確定）**: 設計書の式(13)(14) は使わず、
  **YOPO 既存のアーキテクチャ・方法論をそのまま踏襲する**。これは意図的な選択であり
  欠陥ではない。詳細は §X.1。
- **R5 は「充足」ではなかった。** 設計書 §8.4 は「標準仕様では Σ を直接学び、
  full SO(3) pose loss を置かない」と明記するが、継承元 compact chain の
  `AMPStableRotation3DLoss`（weight 5.0）が全 GauCho stage で有効である。
  設計書が置かないとするものを学習していた。
- **§9.2 規則2 未適用。** ellipsoid 損失が 2D 中心・depth head へ逆伝播している。
- Phase 2 で surface/free-space を「実装済み・未使用」と明記する規約を自分で作りながら、
  `decode_center_from_anchor` と `ray_frame` に同じ規約を適用していなかった。
  記載の非対称性であり、これが誤りの本質である。

### 確認された乖離（9 件）

| 重大度 | 種別 | 内容 | 設計書 |
|---|---|---|---|
| — | 意図的逸脱 | 式(13)(14) depth anchor を使わず YOPO 既存方式を採用（§X.1） | §3.3, 付録C step 3 |
| high | 未実装 | ray frame 式(25)-(27) が死にコード。`Sigma_c = Br Sigma_r Br^T` を適用していない | §5.4 |
| high | 未実装 | 式(42)-(45) ray-surface 損失に呼び出し元が無く、`w_j` も無い | §7.4, R2/R6 |
| high | **矛盾** | Table 3 の対称性 gating が死にコードで、full SO(3) pose loss を全 stage で学習 | §8.1/8.4, 式(52) |
| high | 乖離 | 式(35) front margin が検査として強制されていない。完全に camera 背後の楕円体が projection validity を通る | §12.2, §6.4, 付録C step 5 |
| medium | 未実装 | 式(20) `rc = 0` 縮約 dual-plane を選択できない（A4/A5 アブレーション不能） | §4.3, §13 item 10 |
| medium | 乖離 | 式(45) の分母。ray を mask すると再重み付けでなく loss 全体が再スケールされる | §7.4 |
| medium | 乖離 | §9.2 規則2: centre/depth anchor が detach されていない | §9.2 |
| — | 意図的逸脱 | 付録D の invalid-depth 受入行は、意図的に未接続の関数を試験している（§X.1） | 付録D |

### 判断

3 件（front margin gate、式(45) 分母、`rc=0` オプション）は適用済み。§X.2 参照。

`loss_rotation` は**ユーザー判断により現状維持**とする。設計書 §8.4 からの
意図的な逸脱として記録する。継承元 compact chain の学習済み重みはこの損失の下で
最適化されており、外した影響は測らないと分からないため、変更は別作業とする。

### §X.1 R2 の扱い — YOPO 既存方式の踏襲（確定）

**ユーザー判断により、式(13)(14) の depth anchor は採用せず、YOPO 既存の
アーキテクチャ・方法論をそのまま用いる。** 以下はその裏付けとして自分で確認した事実である。

#### GauCho の 3D 中心は、YOPO 既存の 9D pose と同一機構である

既存 YOPO（本作業以前から存在）:

```
_recover_translation(K, centers_2d_h, z_pred)   posehead.py:2348
  -> depth = exp(z) if use_log_z else z
  -> rays  = solve(K, q~)
  -> return depth * rays
呼び出し元: :2395 (matching), :2734 (inference)  ← いずれも既存コード
```

本作業の GauCho 3D 中心:

```
_backproject(pixels, depth, K)                  posehead.py:1907
  -> 同一の depth * K^-1 q~
```

すなわち **第二の translation 推定器を作らず、既存の学習済み経路を共有している**。
設計書 Table 2 の Independent 方式（共有特徴 + 独立 decoder）に合致し、
head の docstring と Stage B/C config にも当初からその旨を記載している。

#### depth は 2 経路で入っている（レビューの主張は誤りだった）

理論整合性レビューは「`cop_fusion_mode` は既定 `'residual'` のままなので z 枝は depth 特徴を
サンプリングすらしていない」と述べ、私はそれを検証せず一度作業書へ取り込んだ。**誤りである。**
実際にモデルを構築して確認した:

```
use_cop_chain           : True
cop_fusion_mode         : depth_dense      <- 既定ではない
requires_depth_features : True
has depth_query_sampler : True
backbone                : RGBDResidualBackbone
```

したがって depth は次の 2 経路で metric 推定に寄与している。

1. `ConcatRawDepthToImage` → `RGBDResidualBackbone` の 4ch 入力
2. `cop_fusion_mode='depth_dense'` の `MultiScaleDepthQuerySampler` による
   CoP chain への dense depth 特徴注入

R2「RGB-D を metric 3D 推定へ使う」は**この構成で充足している**。

#### 結論

`decode_center_from_anchor` と `ray_frame` は実装・単体試験済みのまま
**意図的に未接続**とする。設計書の anchor 方式を試したくなった場合の入口として残す。
付録D の invalid-depth 受入行も同様の位置づけである。

**教訓**: レビュー結果を自分で検証せずに作業書へ取り込んだ。`cop_fusion_mode` は
実際にモデルを構築すれば 1 コマンドで確認できた。エージェントの指摘であっても
一次情報に当たってから記録すること。

### §X.2 適用した修正（3 件）と、評価の公平化

#### 1. 式(35) front margin を validity へ組み込んだ

`C*_33 < 0` だけでは前方判定にならない。`C*_33 = Sigma_zz - t_z^2` は
**camera 背後でも**、`|t_z|` が物体自身の奥行き広がりを超えれば負になる。自分で再現した:

```
t_z=+1.0 | C33=-0.9991  | front_margin=+0.9700 | valid=True   (正しい)
t_z=-1.0 | C33=-0.9991  | front_margin=-1.0300 | valid=True   <- 背後なのに通る
t_z=-5.0 | C33=-24.9991 | front_margin=-5.0300 | valid=True   <- 完全に背後でも通る
```

`front_margin` は正しく負値を返していたのに、検査に使われていなかった。
`project_ellipsoid_dual_quadric` に `z_min` を追加し `valid & (front_margin > z_min)` とした。

**学習への影響なし**を確認済み: 現実的な前方配置 2000 サンプルで受理率 1.0000。

#### 2. 式(45) の分母を重み付き平均へ

`_finalize` 経由の一般 `mean` は全 ray 数で割るため、visible mask で遮蔽物を除外するほど
loss 全体が小さくなっていた。設計書の `sum_j w_j rho_H / (sum_j w_j + eps)` は
**寄与した重み**で割る。`reduction=="mean"` かつ `avg_factor` 未指定のとき重み付き平均にした。
試験で「1 ray」と「同じ 1 ray + mask した 3 ray」が同値になることを固定した。

#### 3. 式(20) `rc = 0` 縮約 dual-plane を選択可能にした

`dual_plane_cholesky3d(raw, fix_rc_zero=True)`。アブレーション A4/A5
（二平面のみ vs 相関を学習する完全形）が実行できるようになった。
試験で `l32 == 0`、SPD 維持、第6 raw 値が不活性、そして
**full 形との相対 Frobenius 誤差の中央値が 1e-2 を超える**（＝1 自由度を実際に失う）ことを固定。

#### 4. 評価の公平化（別件・重大）

`EllipseEnvelopeRotatedIoUMetric` に opt-in の class-aware rotated NMS を追加した
（既定 `None` ＝ 現状維持なので、既存の記録値との比較可能性は保たれる）。

理由: `NOCSMetric` は既定で `score_thr=0.2` と `nms_cfg(iou_threshold=0.5)` を適用するのに、
ellipse 指標は NMS を一切適用せず 256 query を全部出していた。**比較が非対称だった。**
参照実装 0.8170 も rotated NMS IoU 0.2 で測られている。

同一 checkpoint の予測を dump して再スコアした実測値（自分で計測）:

| NMS IoU | mAP50 | recall | precision | 検出数 |
|---:|---:|---:|---:|---:|
| なし（従来の報告値） | 0.6533 | 0.8171 | 0.2421 | 84,480 |
| **0.2（参照実装と同一）** | **0.7113** | 0.7824 | 0.4768 | 40,937 |
| 0.65 | 0.7247 | 0.8075 | 0.4153 | 48,644 |

出力の 34% が既にマッチ済み GT の重複だった。`topk` による切り詰めでは改善しない
（200 → 0.6476）ことから、原因は検出数ではなく重複の順位づけである。

**これはモデルの改善ではなく測定の公平化である。** 0.65 は評価 split 上で選んだ値なので
見出しにはしない。参照実装と同一プロトコル（IoU 0.2）での値 **0.7113** を採用する。

この結果、「ellipse は検出器 AP の 95-97% で頭打ち＝検出器が律速」という
これまでの結論は**根拠を失う**。NMS 込みでは ellipse (0.7113) が
検出器の報告値 (0.7018) を上回る。

### §X.3 最終試験状況

| | |
|---|---|
| `tests/test_gaucho3d.py` | **65 passed / 0 failed** |
| head 周り既存 8 ファイル | 114 passed / 1 failed（**既存の失敗**、新規回帰ゼロ） |

---

## §Y 2026-09-01 — ユーザーレビューを受けた確定作業

### Y.1 R2 レビューの撤回と、ハイブリッド実装としての明記

R2 の当初レビューは撤回する。現行 GauCho3D は既存 YOPO の RGB-D / CoP / 3D 中心復元方式を
正しく継承している。depth の入り方は次の 4 経路であり、「第4チャネルだけ」は不正確だった。

1. RGB 3ch と depth 1ch を分割し HGNetV2-B1 / B0 へ（`yopo/models/backbones/dual_rgbd.py:128`）
2. 各スケールで `RGB + beta * depth_adapter(depth)` として融合
3. 融合前 depth pyramid を query box 内で 3x3 sampling し depth_dense CoP chain へ
4. 3D 中心は既存 YOPO どおり `t = z K^-1 [u,v,1]^T`（`posehead.py:2348`）

GauCho3D も同じ center2D / z 経路を再利用する（`posehead.py:2206`）。
式(13)(14) の per-pixel depth anchor を使わないのは**意図的設計差**である。

**本実装は「純粋な GauCho3D」ではなく、GauCho shape と既存 semantic pose を併用する
ハイブリッドである。** Stage B/C の 4 config すべての docstring に明記した。
「no rotation is regressed」は GauCho branch 単体では正しいが、モデル全体では
`loss_rotation`（weight 5.0）が有効であるため、表現を訂正した。

### Y.2 mmcv の `nms_rotated` は `labels` を無視する（実測）

`labels` 引数は存在するが機能しない。実測:

```
labels [0, 0] -> keep [0]
labels [0, 1] -> keep [0]   <- 別クラスでも抑制される
```

したがって「class-aware」という当初の記述は誤りだった。`batched_nms` と同じ
座標オフセット方式（クラスごとに平面上の別領域へ移す）で自前実装し、
単体試験 5 件で固定した。

### Y.3 単一 checkpoint での全指標（Stage B best, epoch 18）

段ごとの表は 3D best と 2D best が別 epoch の値を混在させており、運用点ではなかった。
一つの checkpoint について両スコアリング規約で再測定した。

| 指標 | 値 |
|---|---:|
| `ellipse_raw/rbbox_mAP_50`（NMS なし、score_thr .05） | 0.6346 |
| **`ellipse_ref/rbbox_mAP_50`（NMS IoU .2、score_thr .1）** | **0.7148** |
| `ellipse_ref/rbbox_precision_50` | 0.5114 |
| `ellipse_ref/rbbox_recall_50` | 0.7776 |
| `ellipse_ref/rbbox_mean_matched_rIoU` | 0.6993 |
| 検出器 `AP50` | 0.6865 |
| `3d_iou_0.10 / 0.25 / 0.50` | 0.0721 / **0.0295** / 0.0026 |
| `pose 10 degree, 5cm / 10cm` | 0.1630 / 0.1987 |

raw と reference-aligned を別 prefix (`ellipse_raw` / `ellipse_ref`) で併記する形にした。

### Y.4 DoD-B baseline（同一 split・同一評価器）

公開 compact CoP checkpoint を **我々の Jun30 val 330 枚**、同じ dataloader、
同じ評価器で評価した。GauCho branch は off（この checkpoint に該当重みが無いため）。

| 指標 | 公開 CoP（我々の split） | Stage B best | 差 |
|---|---:|---:|---:|
| `3d_iou_0.10` | 0.0084 | **0.0721** | **8.6x** |
| `3d_iou_0.25` | 0.0032 | **0.0295** | **9.2x** |
| `3d_iou_0.50` | 0.0002 | **0.0026** | **11.2x** |
| `pose 10deg/5cm` | 0.1314 | **0.1630** | +3.2 pt |
| `pose 10deg/10cm` | 0.1480 | **0.1987** | +5.1 pt |
| 検出器 `AP50` | 0.6615 | 0.6865 | +2.5 pt |

**これが DoD-B の初めて成立する比較である。** 公開値 0.0166 は別 split の自己申告値であり、
同一 split では 0.0032 まで落ちる。GauCho-3D はその上で 3D IoU@0.25 を **9.2 倍**にした。

ただし検出器 AP50 も +2.5pt 改善しているため、3D の利得のすべてが GauCho shape に
帰属するわけではない。純粋な寄与を出すには Y.5 のアブレーションが要る。

### Y.5 残作業（優先順）

1. `loss_rotation` on/off の一要因アブレーション（同一 Stage A checkpoint、同一 seed /
   epoch / 評価条件。center/z detach や projection weight を同時に変えない）
2. `rc=0` を head / config へ配線してから A4/A5
3. visible mask を用意するまで ray-surface / free-space は有効化しない
4. 主系は Stage B best。Stage C weight 1 は採用しない
5. 次の学習前に merged config / split manifest / checkpoint SHA / UV 環境を固定
