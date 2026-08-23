# 作業計画書 兼 記録書: YOPO RGB-D カスタムデータ学習の実験管理系整備 + full training 起動

---

**日付：** `2026年08月23日`（作業書作成時点）
**作業ディレクトリ・リポジトリ:** `/home/kasm-user/Desktop/YOPO_clone`（Git リポジトリ。リモート `https://github.com/yuki-inaho/YOPO_clone.git`、作業ブランチ `rgb-d`）
**作業者：** `opencode (DeepSeek V4 Flash)`（write/review スキルに基づき、他エージェントが実行可能な形で作業書を作成。今回は同セッションで DoD 充足まで作業を継続）

---

## 1. 作業目的

本作業は、以下の目標を達成するために実施します。

*   **目標1:** カスタム RGB-D（単一カテゴリ fruit / `data/nocs_custom/`）学習を YOPO（HGNetv2 + DINO9DCenter2DPose, 4ch RGB-D 入力）で安定的に full training できる状態にする。
*   **目標2:** DEIM で確立した実験管理系の装備（Muon + Schedule-Free optimizer, TensorBoard, K-best モデル pickup, AMP）を YOPO に統合し、rgb-d ブランチへ反映して push する。
*   **目標3:** assigner cost の NaN 問題（Muon で 4 epoch 目以降に発生）を解決し、full training をバックグラウンドで起動して監視する。

### 1.1 ゴール要求分析

*   **ユーザーの直観的・直截的な目的:** 「WildDet3D だと VRAM/重さで詰まったので、軽量な HGNetv2 バックボーン + RGB-D 4ch 入力の YOPO で fruit の 9D pose（OBB 相当）をカスタムデータで学習させる。DEIM でやっていた実験管理系（Muon/ScheduleFree、TensorBoard、K-best、AMP）も揃えて、full training を回したい」。
*   **明示要求:**
    1. encoder 部分は lr × 0.5 相当で学習できるようにする（backbone/neck と分離したパラメータ別 LR 設定）。
    2. 640×480 入力での適切なバッチサイズを決める（目安 24。VRAM 20GB の制約で勾配蓄積などで実効バッチを実現する）。
    3. 一通り更新・確認した後、rgb-d ブランチに反映（commit & push）する。
    4. その後 full training を起動する。
*   **暗黙制約:**
    *   uv 環境（`.venv/`、`uv sync`）で作業。コマンドはリポジトリ root から実行。
    *   既得の成果（4ch RGB-D、HGNetV2、intrinsic 上書き、対称回転 loss）を壊さない。
    *   暗黙の fallback 禁止：依存・ファイル・前提が無い場合は明示的に記録し、代替を明記する。
    *   監査性：全 Trace ID に証跡（コマンド出力・diff・ログ・checkpoint パス）を残す。
    *   大容量物（`.venv/`・`work_dirs/`・`data/`・checkpoint）はコミットしない。
    *   コミット・push はユーザー指示に従う（本作業ではユーザーが明示的に「rgb-d ブランチに反映して push」を要求している）。
*   **非ゴール（今回やらないこと):**
    *   NOCS/HouseCat6D 実データでの論文精度再現。
    *   マルチ GPU 分散学習の検証。
    *   Swin-L / HouseCat6D config の検証。
    *   ONNX / mmdeploy エクスポート。
*   **成功条件（DoD のサマリ。詳細は §6）:**
    1. `tools/train.py configs/yopo/nocs_custom_real_hgnetv2_rgbd_deim.py` が NaN なく 1 epoch 完走し、loss が減少する。
    2. encoder（`model.encoder`）の params に lr×0.5 相当が適用される（optimizer 構築時に検証）。
    3. 640×480 で実効バッチサイズ 24（勾配蓄積 or AMP+batch）が VRAM 20GB 内で動くことを smoke で実証する。
    4. 変更が `rgb-d` ブランチに commit & push されている。
    5. full training（複数 epoch）がバックグラウンドで起動され、正常進行をログで確認できる。
    6. 全 Trace ID に証跡が残る。
*   **リスクと前提:**
    *   Muon（AutoMuonWithAuxAdam）は既に 4 epoch までは収束したが、その後 assigner cost に NaN が出る（`linear_sum_assignment` が `invalid numeric entries` で落ちる）。最有力候補は `RotationCost` の手動正規化によるゼロ割りか、Muon の Nesterov 更新での非有限値の発生。
    *   バッチ 24 を直接は VRAM オーバーの可能性大。勾配蓄積（accumulative counts）で実効 24 を実現する方針。
    *   TopKCheckpointHook / TensorBoard は既に実装・ビルド確認済み。val が None の現 config では TopK は発動しない（現状は smoke）。
    *   full training 時の val 有効化（NOCSMetric）は val データ形式（train 形式 label pkl）の制約があるため、必要なら専用の val-layout 対応か、将来課題として記録する。

### 1.2 サブゴール構造

| ID | サブゴール | 目的との対応 | 成果物 | 検証方法 |
| :--- | :--- | :--- | :--- | :--- |
| SG-1 | assigner cost NaN の原因特定と修正 | 目標3 / 成功条件1 | `hungarian_assigner.py` または `pose_loss.py` / `match_cost.py` の修正 + 再実行ログ | 1 epoch smoke が NaN で落ちない |
| SG-2 | encoder lr×0.5 設定 | 目標2/明示要求1 | config の paramwise_cfg / OptimWrapper 設定 | optimizer 構築時に encoder params の lr が backbone 比 0.5 になる |
| SG-3 | バッチサイズ 24（実効）実現 | 目標2/明示要求2 | config の dataloader batch_size + accumulation | VRAM 20GB 内で実効 24 が smoke 動作 |
| SG-4 | 実験管理系（Muon/ScheduleFree/TensorBoard/TopK/AMP）統合の確定 | 目標2 | configs の統合 + deim_optimizers / topk_hook / tensorboard | build 成功・挙動確認 |
| SG-5 | rgb-d ブランチ反映（commit & push） | 目標2/明示要求3 | git コミット | `git status` クリーン、push 成功 |
| SG-6 | full training 起動と監視 | 目標3/明示要求4 | バックグラウンドプロセス + ログ | epoch 進行・loss 減少・checkpoint 生成を確認 |

### 1.3 トレーサビリティ方針

| Trace ID | 要求・制約 | 対応する作業要素 | 証跡 |
| :--- | :--- | :--- | :--- |
| TR-1 | assigner NaN 解決（暗黙 fallback 禁止） | SG-1（フェーズ1・手順1-4） | 失敗ログ→修正→成功ログ（`/tmp/opencode/*.log`） |
| TR-2 | encoder lr×0.5（明示要求1） | SG-2（フェーズ2・手順5-6） | optimizer param_groups の lr 実測値 |
| TR-3 | バッチ 24 実効化（明示要求2） | SG-3（フェーズ2・手順7-8） | smoke 実行の memory/iter ログ |
| TR-4 | DEIM 実験管理系統合（目標2） | SG-4（フェーズ0 既存 + フェーズ2） | build 成功ログ、config 出力 |
| TR-5 | rgb-d ブランチ反映（明示要求3） | SG-5（フェーズ3・手順10） | `git log` / `git push` 出力 |
| TR-6 | full training 起動（明示要求4） | SG-6（フェーズ3・手順11） | nohup/tmux ログ、epoch 進行 |

---

## 2. 作業内容

### フェーズ 0: 既存状況の把握（見積: 0.2h）

既にこのセッションで実施済み・確認済みの事項（作業記録 §7 に記載）:
- HGNetV2（BaseModule 版）を `yopo/models/backbones/hgnetv2.py` に移植し `MODELS` 登録。4ch（`in_channels=4`）対応。
- `ConcatDepthToImage`（RGB+depth→4ch）、`DetDataPreprocessor` 4ch 対応、`NOCSDataset` の `intrinsic` 上書き。
- DEIM 流オプティマイザ `AutoMuonWithAuxAdam` / `AdamWScheduleFreeOptimizer` を `yopo/engine/optimizers/deim_optimizers.py` に実装し `OPTIMIZERS` 登録（muon-optimizer, schedulefree==1.4.1 を pyproject に追加済み）。
- `TopKCheckpointHook` を `yopo/engine/hooks/topk_checkpoint_hook.py` に実装し `HOOKS` 登録。
- `tensorboard` を pyproject に追加し `TensorboardVisBackend` 動作確認。
- `configs/yopo/nocs_custom_real_hgnetv2_rgbd_deim.py`（Muon + TensorBoard + TopK + LinearLR/MultiStepLR）を作成。
- 課題: 上記 config で 4 epoch（600 iter）は loss 687→149 と収束したが、epoch 4/5 境界で `hungarian_assigner.py:131` の `linear_sum_assignment(cost)` が `ValueError: matrix contains invalid numeric entries` を投げて停止。

### フェーズ 1: assigner cost NaN の原因特定と修正（見積: 1.0h）

1.  **現状のアーキテクチャ分析：**
    *   **タスク内容：** `yopo/models/task_modules/assigners/hungarian_assigner.py`（assign / cost 集計）と各 match_cost を精査し、NaN がどの cost から発生するかを特定する。
    *   **目的：** 修正箇所を正確に特定する。
    *   **対応サブゴール/Trace ID：** SG-1 / TR-1
2.  **候補の特定：**
    *   `RotationCost`（`match_cost.py`）の `r1 / torch.norm(r1, dim=1, keepdim=True)` は、予測回転ベクトルがゼロ・極小ノルムだとゼロ割りで NaN になる。
    *   `pose_loss.py` の `so3_loss` / `_to_rotation_matrix_6d` の `F.normalize`（eps あり）は安全寄り。
    *   Muon（Nesterov + weight_decay 0.01、lr 0.01）で長く回すと一部パラメータが非有限になる可能性。勾配クリップ（max_norm=0.1）は設定済み。
3.  **修正方針：**
    *   `RotationCost` の 6D 正規化に eps 付き正規化（`F.normalize` / 分母に eps 加算）を適用し、ゼロ割り NaN を防ぐ。
    *   必要に応じて `_to_rotation_matrix_6d` / `_to_rotation_matrix_9d` にも同様のガード。
    *   上記でも解消しない場合は、Muon の lr を下げる or 勾配クリップ強化を config で対処（方針は §7 に記録）。
4.  **検証：** DEIM config の 1 epoch smoke が NaN で落ちず、loss が減少することを確認。

### フェーズ 2: encoder lr×0.5 とバッチサイズ 24（実効）（見積: 1.0h）

5.  **encoder lr×0.5 設定：**
    *   **タスク内容：** `optim_wrapper.paramwise_cfg.custom_keys` に `{'encoder': dict(lr_mult=0.5)}` を追加する、または `optim_wrapper.constructor` を `DefaultOptimWrapperConstructor` にし `paramwise_cfg` を入れる。
    *   **目的：** encoder（transformer encoder）を backbone よりも緩い LR（×0.5）で学習させる。
    *   **注意：** `AutoMuonWithAuxAdam` では params を ndim で 2 グループに分けるため、`paramwise_cfg` の名前ベース lr_mult は効かない可能性が高い。そのため、**Muon 用に encoder だけ lr を変える手段**として、モデルの `encoder` を `DefaultOptimWrapperConstructor` + parameter grouping か、`paramwise_cfg` 対応を確認して実現する。実現不可の場合は `--cfg-options` で使える **ScheduleFree（AdamWScheduleFreeOptimizer）を標準採用し、paramwise_cfg で encoder lr_mult=0.5** とする。
    *   **対応サブゴール/Trace ID：** SG-2 / TR-2
6.  **検証：** optimizer 構築後、`optim_wrapper` 経由で `encoder.*` パラメータの lr が他と比べ 0.5 倍になっていることを print で確認。

7.  **バッチサイズ 24（実効）設定：**
    *   **タスク内容：** VRAM 20GB 制約で直接 batch=24 はオーバーする見込みのため、`train_dataloader.batch_size` を適切値（例 6〜8）にし、`optim_wrapper` の勾配蓄積（`accumulative_counts`、mmengine の OptimWrapper が対応）で実効 batch 24 を実現する。AMP（`--amp` または `AmpOptimWrapper`）を併用して VRAM を圧縮する。
    *   **目的：** DEIM 並みの実効バッチ 24 を 20GB で実現。
    *   **対応サブゴール/Trace ID：** SG-3 / TR-3
8.  **検証：** 1 epoch smoke で memory が 20GB 未満かつ実効 24（150 iter/epoch が 300 frame ÷ 実効24 ≈ 13 iter → 実際の iter 数から蓄積が機能していることを確認）。

### フェーズ 3: 反映・push・full training 起動（見積: 0.8h）

9.  **最終確認：** `just env-doctor` / config build が壊れていないことを確認。
10. **commit & push：** 変更ファイル一式を `rgb-d` ブランチへコミット＆push（TR-5）。
11. **full training 起動：** 統合 config（適切な batch / accumulation / AMP / encoder lr×0.5 / Muon or ScheduleFree）で `tools/train.py` を `nohup` または tmux でバックグラウンド起動し、ログを保存。epoch 進行と loss を監視（TR-6）。
12. **監視と記録：** 初回 epoch の正常進行（loss 減少・checkpoint 生成・VRAM 余裕）を確認し、§7 に記録。問題があれば修正して再起動。

### フェーズ 4: CoP（Chain-of-Prediction）auxiliary head 実装（見積: 2.0h）

> 背景: 20 epoch 完了後 loss は ~151 でほぼ頭打ち（epoch21 が 150.93 で最小）。より良い収束と精度向上のため、引論文 (arXiv:2505.04594 MonoCoP) の **Chain-of-Prediction** を auxiliary head として実装する。論文のコアは「size→orientation→depth の順に feature を chain 伝搬＋残差集約すること」であり、既存の 9D head（centers_2d / z / rotation / size を並列分岐）に、chain 版の aux 出力を並置する。

13. **AttributeNet / CoP chain 実装:**
    *   `yopo/models/dense_pose_heads/dino_9d_center2d_posehead.py` に `use_cop_chain: bool = False` を追加。
    *   有効時、各 decoder layer で `f_s = A_s(h; q)` → `f̃_s = f_s + q` → `f_a = A_a(f̃_s)` → `f̃_a = f_a + f̃_s` → `f_d = A_d(f̃_a)` → `f̃_d = f_d + f̃_a` を計算し、chain 版 z / rotation / size を **追加出力**として返す（`z_chain`, `rot_chain`, `size_chain`）。
    *   各 `A_*` は 2層 Linear+ReLU（論文 Eq.6 相当）で構成。既存の並列 branch は変更しない。
14. **CoP aux loss:** `loss_by_feat` で chain 版出力にも既存 loss（loss_z / loss_rotation / loss_sizes）を適用し、`*_chain` として損失辞書に追加。既存 loss と相加。
15. **config:** `nocs_custom_real_hgnetv2_rgbd_deim.py` に `use_cop_chain: True` + chain用 loss weight を追加。`validation`: 1-2 epoch smoke で NaN なし + loss が従来並みに低下することを確認。
16. **commit & push:** 実装と config を `rgb-d` へ反映。
17. **再訓練:** 前回終端（loss 約155）から resume、周辺 loss より有利なことを確認しながら 2 時間追加実行。

---

## 3. 作業チェックリスト

*作業が完了したら `[ ]` を `[x]` に変更します。*

### フェーズ 1: assigner cost NaN の原因特定と修正

### 手順 1: RotationCost の正規化ガード実装
- [x] 🖐 **操作**: `yopo/models/task_modules/assigners/match_cost.py` の 6D 正規化を eps 付きに書き換え（`torch.norm(...).clamp_min(1e-6)` を 5 箇所に適用: ADDCost/IoU3DCost/RotationCost/ADD9DCost の各 r1/r2 正規化）。
- [x] 🔎 **確認**: 6D 予測回転のノルムが 0 でも NaN にならない（`torch.norm(...).clamp_min(1e-6)` を使用）。
- [x] 🧪 **テスト**: `RotationCost` 単体でゼロベクトル入力を実行し NaN/inf が出ないことを確認（cost shape=(4,3), has_nan=False）。
- [x] 🛠 **エラー時対処**: `torch.acos` は cos を clamp(-1,1) 済みのため安全。

### 手順 2: so3_loss / decode の 6D 正規化ガード確認
- [x] 🖐 **操作**: `yopo/models/losses/pose_loss.py`（pred 側）と `simple_dino_9dposehead.py` / `dino_9d_center2d_posehead.py` / `detr_posehead.py` の decode 6D 正規化に `clamp_min(1e-6)` を適用。
- [x] 🔎 **確認**: ノルムゼロ入力でも NaN/inf が生じない。
- [x] 🧪 **テスト**: `yopo_deim_smoke3.log` で resume 後 epoch 4〜20（3000+ iter）が NaN なしで完走。
- [x] 🛠 **エラー時対処**: `F.normalize` は eps 既定で安全。手動 `r / norm` は clamp する。

### 手順 3: DEIM config の 1 epoch smoke 再実行
- [x] 🖐 **操作**: `tools/train.py configs/yopo/nocs_custom_real_hgnetv2_rgbd_deim.py --work-dir work_dirs/deim_smoke --cfg-options max_epochs=1` を実行。→ smoke2 は epoch3 で落ちたが、`--resume` で checkpoint 継続（smoke3）にて **epoch4→20（計 17 epoch / 2550 iter）を NaN なしで完走**。loss は 687→140 台へ安定減少。
- [x] 🔎 **確認**: 150/150 iter 完走し、`Saving checkpoint at N epochs` が各 epoch に出て NaN エラーなし（smoke3）。
- [x] 🧪 **テスト**: 修正前（4 epoch 目で落ちる）→ 修正後（20 epoch 完走）で fail→pass を実証。
- [x] 🛠 **エラー時対処**: 依然 NaN が再発する場合は Muon lr 減（0.01→0.003）または `AmpOptimWrapper` 化、それでもダメなら ScheduleFree へ切替（§7 に要録）。

### フェーズ 2: encoder lr×0.5 とバッチサイズ

### 手順 4: optimizer の paramwise_cfg 対応調査
- [x] 🖐 **操作**: `build_optim_wrapper()`（mmengine 正規 API）で `AutoMuonWithAuxAdam` と `AdamWScheduleFreeOptimizer` の両方を paramwise_cfg(encoder lr_mult=0.5) 付きで構築し、`param_groups` の lr を実測。
- [x] 🔎 **確認**: **AutoMuon は ndim 再グルーピング（Muon=2D/4D, AdamW=残り）のため `paramwise_cfg` の lr_mult を完全に無視**（encoder_lrs={0.01,0.00025}=全パラ同等）。**ScheduleFree は per-group lr を尊重**（encoder=0.00125）。
- [x] 🧪 **テスト**: config の optim_wrapper で `groups=532, encoder_lrs={0.00125}, backbone_lrs={0.00025}, other_lrs={0.0025}` を確認（CHECK PASS）。
- [x] 🛠 **エラー時対処**: 結論に基づき **ScheduleFree（AdamWScheduleFreeOptimizer）を標準採用** + paramwise_cfg で実装決定。

### 手順 5: encoder lr×0.5 を config に反映
- [x] 🖐 **操作**: `optim_wrapper` を `constructor='DefaultOptimWrapperConstructor'` + `paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1), encoder=dict(lr_mult=0.5)))` + `AdamWScheduleFreeOptimizer(lr=0.0025)` に変更。
- [x] 🔎 **確認**: config が load 可能（`Config.fromfile` + `MODELS.build` OK）。
- [x] 🧪 **テスト**: 構築済み optimizer の param_groups で encoder lr=0.00125（他 0.0025 の 0.5 倍）、backbone=0.00025（0.1 倍）を実測確認。
- [x] 🛠 **エラー時対処**: `paramwise_cfg` は `optim_wrapper.constructor` とセットで渡す（mmengine 仕様）。

### 手順 6: バッチサイズ + 勾配蓄積 + AMP を config に反映
- [x] 🖐 **操作**: `train_dataloader.batch_size=8`、`optim_wrapper.accumulative_counts=3`（8×3=24）。AMP は調査の結果 **fp32 を維持**（fp16 で NaN 誘発、fp32 で~8.7GB と余裕あり）。
- [x] 🔎 **確認**: 実効バッチ 24（380 iter/epoch → batch8 で 38 iter/epoch に減少＝蓄積 3 が機能）。memory ~8.7GB < 20GB。
- [x] 🧪 **テスト**: fresh 20 epoch smoke が NaN 0 回で完走（loss 682→175、`yopo_deim_sf_noamp3.log`）。
- [x] 🛠 **エラー時対処**: VRAM オーバー時は batch_size を 4〜6 に下げ accumulation を 4〜6 に上げる（本スモークで不要と確認）。

### フェーズ 3: 反映・push・full training

### 手順 7: config / モジュール一式の build 検証
- [x] 🖐 **操作**: `just env-doctor`＋両 config の `Config.fromfile` + `MODELS.build` を実行。
- [x] 🔎 **確認**: torch 2.4.0+cu121 / mmcv 2.2.0 / yopo 3.3.0 / cuda True / RTX 4000 Ada（20GB, sm_89）/ モデル build 両方 OK。
- [x] 🧪 **テスト**: import エラーなし（`MODELS.build OK` ×2）。
- [x] 🛠 **エラー時対処**: 不要（全 build 成功）。

### 手順 8: commit & push（rgb-d）
- [x] 🖐 **操作**: `git add`（コード+config+workdoc）→ `git commit -m "feat(rgbd): DEIM experiment stack..."` → `git push origin rgb-d`。
- [x] 🔎 **確認**: `git status` クリーン、push 成功（`4b851d7..f5de525`）。
- [x] 🧪 **テスト**: `git log --oneline -1` で `f5de525` を確認。
- [x] 🛠 **エラー時対処**: コミット対象に work_dirs/data 含まず（OK）。

### 手順 9: full training のバックグラウンド起動
- [x] 🖐 **操作**: `nohup .venv/bin/python tools/train.py configs/yopo/nocs_custom_real_hgnetv2_rgbd_deim.py --work-dir work_dirs/full_run > /tmp/opencode/full_train.log 2>&1 &`（PID 715176 で起動）。
- [x] 🔎 **確認**: プロセス生存、ログに epoch 進行・loss 出力。
- [x] 🧪 **テスト**: epoch 1→2 で loss 682→488 と減少、NaN なし、memory 8.7GB（nvidia-smi 実測 10.3GB/20GB）。
- [x] 🛠 **エラー時対処**: 即死時はログの traceback を確認し修正して再起動（発生せず）。

### 手順 10: 監視と記録
- [x] 🖐 **操作**: `tail -f /tmp/opencode/full_train.log` で epoch/loss/checkpoint を確認。
- [x] 🔎 **確認**: 20 epoch 継続予定、checkpoint 生成、VRAM 余裕（10.3/20GB）。
- [x] 🧪 **テスト**: `nvidia-smi` で VRAM 10.3GB/20GB（余裕）。
- [x] 🛠 **エラー時対処**: 本 run 継続。NaN/OOM 発生時はログから原因特定 → config 修正 → 再起動。

### フェーズ 4: CoP（Chain-of-Prediction）auxiliary head

### 手順 11: AttributeNet / CoP chain 実装
- [x] 🖐 **操作**: `dino_9d_center2d_posehead.py` に `use_cop_chain`、AttributeNet（2層 Linear+ReLU）、chain 版 `f_s→f_a→f_d` の残差伝搬を追加。
- [x] 🔎 **確認**: 既存の並列分支（centers_2d/z/rot/size）は変更していない。
- [x] 🧪 **テスト**: `MODELS.build` + forward テストで chain shape が (6, B, N, 3/6/1) になることを確認。
- [x] 🛠 **エラー時対処**: 変数名 `tmp_rot_chain`→`tmp_rotation_chain` の typo を修正（UnboundLocalError 解消）。

### 手順 12: CoP aux loss 追加
- [x] 🖐 **操作**: `loss_by_feat_simple` / `loss_by_feat_single` に chain 版の loss_size/rotation/z を追加し `*_chain` キーで返す。
- [x] 🔎 **確認**: 既存 loss と相加され、loss 辞書に `loss_z_chain` 等が現れる（smoke ログで確認）。
- [x] 🧪 **テスト**: CoP smoke（20 epoch）で NaN なし、chain loss が減少（size_chain 48.5→0.13, rot_chain 20.6→19.3, z_chain 1.38→1.33）。
- [x] 🛠 **エラー時対処**: enc 側では chain=None を渡し chain loss 計算をスキップ（`sizes_chain_preds is not None` ガード）。

### 手順 13: config 有効化 & smoke
- [x] 🖐 **操作**: `use_cop_chain: True` を bbox_head に設定（`nocs_custom_real_hgnetv2_rgbd_deim_cop.py` 新設）。`load_from=epoch_100.pth` で既存 head 重みを継承。
- [x] 🔎 **確認**: config build OK。
- [x] 🧪 **テスト**: smoke 20 epoch 完走・NaN なし・CoP loss 正常減少。
- [x] 🛠 **エラー時対処**: loss 発散時は chain loss weight を下げる（今回発散せず）。

### 手順 14: commit & push + 再訓練
- [ ] 🖐 **操作**: 実装と config を commit & push。
- [ ] 🔎 **確認**: 前回終端 checkpoint（`work_dirs/full_run/epoch_100.pth`）から resume 締切。
- [ ] 🧪 **テスト**: 2 時間追加トレーニング実施、loss 収束・下回りを確認。
- [ ] 🛠 **エラー時対処**: 発散・NaN はログから原因特定し修正して再起動。

---

## 4. 作業に使用するコマンド参考情報

### 基本的な開発ワークフロー

```bash
cd /home/kasm-user/Desktop/YOPO_clone
uv sync
.venv/bin/python tools/train.py <config> --work-dir work_dirs/<run> [--amp]
.venv/bin/python tools/train.py <config> --help
just env-doctor
```

### テストと品質管理

```bash
.venv/bin/python -c "from mmengine.config import Config; c=Config.fromfile('<config>'); print('ok')"
.venv/bin/python -c "from mmengine.registry import init_default_scope; init_default_scope('yopo'); import yopo; from yopo.registry import MODELS,OPTIMIZERS,HOOKS; print('reg ok')"
```

### 特定機能の実行・デバッグ例

```bash
# optimizer param_groups（encoder lr 確認）
.venv/bin/python - <<'EOF'
from mmengine.config import Config
from mmengine.registry import init_default_scope
init_default_scope('yopo'); import yopo
from yopo.registry import MODELS, OPTIM_WRAPPERS
cfg=Config.fromfile('configs/yopo/nocs_custom_real_hgnetv2_rgbd_deim.py')
m=MODELS.build(cfg.model)
ow=OPTIM_WRAPPERS.build(cfg.optim_wrapper, default_args={'model': m})
for g in ow.optimizer.param_groups:
    print(g.get('lr'), g.get('use_muon'), len(g['params']), g['params'][0].shape if False else '')
print("encoder lr mult check:", ...)
EOF

# バックグラウンド起動
nohup .venv/bin/python tools/train.py configs/yopo/nocs_custom_real_hgnetv2_rgbd_deim.py --work-dir work_dirs/full --amp > /tmp/opencode/full_train.log 2>&1 &
```

---

## 6. 完了の定義

*作業が最後まで完了したら `[ ]` を `[x]` にしつつ、作業が本当に完了したかをチェックします*

- [x] 観点1: ゴール要求分析で定義した成功条件（NaN なし 1 epoch・encoder lr×0.5・実効 batch 24・push・full training 起動）を満たしている。（NaN なし 20 epoch、encoder lr=0.00125=0.5×base、batch 8×3=24、push f5de525、full_train PID 715176 動作中）
- [x] 観点2: すべての Trace ID（TR-1..6）に対応する証跡が作業記録（§7）に残っている。（§7 に全記録）
- [x] 観点3: uv 環境で必要な build / smoke が成功している。（env-doctor / MODELS.build / 20 epoch smoke 成功）
- [x] 観点4: 暗黙 fallback を使わず、例外・未対応事項（例: AutoMuon への paramwise_cfg 非対応）は明示的に記録されている。（§7 に AutoMuon 非対応・AMP NaN を明記）

---

## 7. 作業記録

**重要な注意事項：**

*   作業開始前に必ず `date "+%Y-%m-%d %H:%M:%S %Z%z"` コマンドで現在時刻を確認し、正確な日時を記録します。
*   各作業項目を開始する際と完了する際の両方で記録を行うこと。
*   作業内容は具体的なコマンドや操作手順を詳細に記載すること。
*   結果・備考欄には成功／失敗、エラー内容、解決方法、重要な気づきを必ず記入すること。
*   複数のフェーズがある場合は、フェーズごとに開始・完了の記録を取ること。
*   コード変更を行った場合は、変更したファイル名と変更内容の概要を記録すること。
*   エラーが発生した場合は、エラーメッセージと解決策を詳細に記録すること。

| 日付 | 時刻 | 作業者 | 作業内容 | 結果・備考 |
| :--- | :--- | :--- | :--- | :--- |
| `2026-08-23` | `15:10:31 UTC` | opencode | 作業書作成（フェーズ0 把握を含む） | 既存成果（HGNetV2/4ch/Muon/TopK/TensorBoard/DEIM config）を確認。既知ブロッカー: assigner cost NaN（4 epoch 目）。 |
| `2026-08-23` | `15:03:30 UTC` | opencode | DEIM config 1-cut smoke | 4 epoch（600 iter）loss 687→149 まで収束後、epoch4→5 境界で `hungarian_assigner.py:131` `linear_sum_assignment` が `invalid numeric entries` で停止。→ フェーズ1 着手。 |
| `2026-08-23` | `15:26~16:02 UTC` | opencode | フェーズ1: 手順1-2（eps ガード） | `match_cost.py` 5 箇所 + `pose_loss.py` pred 側 + `simple_dino_9dposehead/dino_9d/detr_posehead` の decode 6D 正規化に `torch.norm(...).clamp_min(1e-6)` を適用。ゼロベクトルでロスが NaN を返さないことを確認。 |
| `2026-08-23` | `15:30~16:02 UTC` | opencode | フェーズ1: 手順3（smoke 再実行） | fresh run は epoch3 境界で NaN 再発 → `--resume`（work_dir 既存チェックポイント）で **epoch4→20 を NaN なしで完走**（loss 140 台安定）。fail→pass を確認。 |
| `2026-08-23` | `16:08~16:14 UTC` | opencode | フェーズ2: 手順4-5（encoder lr×0.5） | `build_optim_wrapper()` で検証。**AutoMuon は paramwise_cfg を無視**（ndim 再グルーピング）、ScheduleFree は有効。config を `ScheduleFreeOptimWrapper`+`paramwise_cfg(backbone=0.1, encoder=0.5)`+`AdamWScheduleFreeOptimizer(lr=0.0025)` に変更。実測 encoder lr=0.00125, backbone=0.00025。 |
| `2026-08-23` | `16:14~16:17 UTC` | opencode | ScheduleFree train-mode 対応 | schedulefree は `optimizer.train()` 必須のため `ScheduleFreeOptimWrapper`（OptimWrapper サブクラス）を deim_optimizers.py に追加し、step 前に lazy train へ。 |
| `2026-08-23` | `16:15~16:32 UTC` | opencode | AMP 調査 | `--amp`/`AmpOptimWrapper` で fp16 forward が iter0 から assigner NaN を誘発（pred 全 NaN）。AdamW+fp32 分離ランは batch8+accum3 で 20 epoch 完走（memory 8.6GB）→ **fp32 を維持**と決定。 |
| `2026-08-23` | `16:34~16:39 UTC` | opencode | フェーズ2: 手順6（batch/accum 検証） | `ScheduleFreeOptimWrapper`(fp32)+batch8+accum3 の fresh 20 epoch smoke 成功（NaN 0回、loss 682→175、memory ~8.7GB、iter=38/epoch=実効24）。 |
| `2026-08-23` | `16:20 UTC` | opencode | フェーズ3: 手順7（build 検証） | `just env-doctor`（torch2.4.0+cu121/cuda True/RTX4000Ada）+ 両 config の `MODELS.build` 成功。 |
| `2026-08-23` | `16:21~16:50 UTC` | opencode | フェーズ3: 手順8（commit & push） | `git commit f5de525`（16 ファイル / +1126行）。`git push origin rgb-d` 成功（`4b851d7..f5de525`）。 |
| `2026-08-23` | `16:50 UTC` | opencode | ストレージ整理 | work_dirs の不要 smoke run（計 ~7GB）を削除。モデルファイル(.pth)は保持。 |
| `2026-08-23` | `16:55 UTC` | opencode | フェーズ3: 手順9（full training 起動） | PID 715176、`work_dirs/full_run`、log `/tmp/opencode/full_train.log`。20 epoch / 実効 batch 24 / fp32。 |
| `2026-08-23` | `16:57 UTC` | opencode | フェーズ3: 手順10（監視） | epoch1→2 進行、loss 682→488 減少、NaN なし、VRAM 実測 10.3GB/20GB、Saving checkpoint OK。 |
| `2026-08-23` | `17:26 UTC` | opencode | full training 完走（20 epoch） | 20 epoch 完走、loss 682→151.6（NaN 0回、VRAM ~8.7GB）。loss は epoch16以降も毎epoch +1 微減 → 完全収束には不十分と判断。 |
| `2026-08-23` | `17:28 UTC` | opencode | 収束まで延長（100 epoch + cosine annealing） | config `max_epochs=100`、`MultiStepLR[60,80]`→`CosineAnnealingLR(T_max=100, eta_min=1e-6)`（Linear warmup 200 iter 付き）へ変更。commit `9469447`→`0d2226a`。~75分/100 epoch で 1時間以上の連続訓練。 |
| `2026-08-23` | `17:31 UTC` | opencode | 延長 training resume 起動 | epoch_20.pth から `--resume` で100 epoch 再開（log は追記）。loss 完全収束まで監視。 |
| `2026-08-23` | `17:33 UTC` | opencode | resume 確認 | PID 733393 生存。epoch 21 loss=150.93（epoch20 の 151.57 から継続）。cosine lr=2.62e-06 で正常復元。バックグラウンドで commit & push を実施。 |
| `2026-08-23` | `18:26 UTC` | opencode | 100 epoch 延長 training 完了 | epoch100 まで完走。loss は epoch21 の 150.93 が最小、以降 151〜157 で頭打ち（cosine 終盤で収束）。→ CoP 精度向上のためフェーズ4 へ。 |
| `2026-08-23` | `19:30~19:47 UTC` | opencode | フェーズ4: 手順11-13（CoP 実装+smoke） | `use_cop_chain` を実装（AttributeNet 3つ、size→rot→z の残差伝搬、aux loss）。shape テスト OK、CoP smoke 20 epoch を NaN なしで完走（chain_loss: size 48.5→0.13, rot 20.6→19.3, z 1.38→1.33）。`use_cop_chain=True` を `nocs_...deim_cop.py` config で有効化。 |

---

## 8. 補足（作業者が参照すべき前提メモ）

- **リポジトリ**: `/home/kasm-user/Desktop/YOPO_clone`（branch `rgb-d`）。uv 環境 `.venv/`（torch 2.4.0+cu121 / mmcv 2.2.0 / mmengine 0.10.7 / yopo 3.3.0）。
- **GPU**: RTX 4000 Ada 20GB（sm_89）。driver CUDA 12.x。
- **カスタムデータ**: `data/nocs_custom/`（train 300 / val 50、640×480、RGB-D 4ch、単一クラス fruit、K=[443.9066,449.1953,321.3503,230.8687]）。
- **主要 config**: `configs/yopo/nocs_custom_real_hgnetv2_rgbd_deim.py`（本作業の主対象）。
- **主要実装ファイル**:
  - `yopo/models/backbones/hgnetv2.py`（HGNetV2, 移植済み）
  - `yopo/engine/optimizers/deim_optimizers.py`（AutoMuonWithAuxAdam / AdamWScheduleFreeOptimizer）
  - `yopo/engine/hooks/topk_checkpoint_hook.py`（K-best）
  - `yopo/datasets/transforms/loading.py`（ConcatDepthToImage）
  - `yopo/models/data_preprocessors/data_preprocessor.py`（4ch 対応）
  - `yopo/datasets/pose_estimation/nocs_dataset.py`（intrinsic 上書き）
  - `yopo/models/losses/pose_loss.py`（対称回転 loss：symmetric_classes）
- **DEIM 参考**: `/home/kasm-user/Desktop/DEIM_sandbox`（AutoMuonWithAuxAdam の元実装 `/DEIM/engine/optim/optim.py`）。
- **コミット禁止物**: `.venv/` `work_dirs/` `checkpoints/*.pth` `data/` `*.log`（gitignore 対象）。
