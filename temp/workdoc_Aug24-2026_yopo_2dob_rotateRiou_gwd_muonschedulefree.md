# 作業計画書 兼 記録書: YOPO 2D OBB 学習（共通 transformer encoder + 2D OBB head、rotate IoU→GWD、Muon+ScheduleFree、HGNetV2-B2 軽量 backbone）

---

**日付：** `2026年08月24日`
**作業ディレクトリ・リポジトリ:** `/home/kasm-user/Desktop/YOPO_clone`（Git リポジトリ。リモート `https://github.com/yuki-inaho/YOPO_clone.git`、作業ブランチ `rgb-d`）
**作業者：** `opencode (DeepSeek V4 Flash)`（write-workdoc-uv スキルに基づき作成。ほかエージェントが実行可能な程度に詳細化）
**前作業からの引き継ぎ:** claude-mem observation id=17 に現状を記録済み

---

## 1. 作業目的

本作業は、以下の目標を達成するために実施します。

*   **目標1:** 従来の RGB-D 3D pose 学習（YOPO の loss_z=1.25 頭打ち・既存デコードバグ）を一旦中断し、**先に 2D OBB（回転矩形）検出** を、YOPO の**共通の transformer encoder（DeformableDETR）から派生出力する 2D OBB head** で正常に学習できることを確認する。
*   **目標2:** 2D OBB head の学習が **rotate IoU 損失 → GWD（Gaussian Wasserstein Distance）損失** の双方で回る（NaN なし・loss が減少する）ことを確認する。
*   **目標3:** 学習に **Muon + ScheduleFree**（DEIM_sandbox 方式）を使う。3D 出力は今回はしない。
*   **目標4:** backbone は ResNet50 などではなく **HGNetV2（YOPO が持つ軽量 pretrained 付き）** を使い、流用するのは transformer の **encoder/decoder（head）のみ** であることを明確にする。
*   **目標5:** 本実装は **YOPO リポジトリ内で self-contained / self-consistent** になるようにする（外部 mmrotate 依存を持ち込まない）。

**注意：** 本作業書のサーミー（仮置き）で、YOPO 内への rotate 実装の移植（RotatedBoxes / rbbox_overlaps / RotatedIoULoss / GDLoss / RotatedDeformableDETRHead / DOTATomatoDataset / match_cost）と、Muon+ScheduleFree の `MuonScheduleFreeOptimizer` 実装は**既に行われています**。「ブロッカー」として Banck している DefaultOptimWrapperConstructor との整合を解決し、FULL 学習に進むのが主目的。

### 1.1 ゴール要求分析

*   **ユーザーの直観的・直截的な目的:** 「YOPO の 3D pose 学習は成果が出ない（z=1.25 頭打ち・デコードバグ）ので、いったん 3D は置いといて、まず 2D OBB 検出を共通 encoder から正しく学習できるかまで確認してほしい。バックボーンは軽量な HGNetV2 を使い、学習は Muon+ScheduleFree で。」
*   **明示要求:**
    1. 3D 出力はしない。2D OBB（cx, cy, w, h, angle）出力のみ。
    2. 2D OBB head は共通の transformer encoder から派生出力する（3D head と encoder を共有する想定の第一歩）。
    3. 損失は **rotate IoU → GWD** の順で確認する。
    4. optimizer は **Muon + ScheduleFree**（DEIM_sandbox 方式）。従来 YOPO で使っていた AdamW 系は使わない。
    5. backbone は **HGNetV2（軽量・pretrained 付き）**。決して ResNet50 を流用しないこと。流用は encoder/decoder のみ。
    6. YOPO リポジトリ内で完結（self-contained）させる。
*   **暗黙制約:**
    *   uv 環境（`.venv/`）で作業。コマンドはリポジトリ root から。torch 2.4.0+cu121 / mmcv 2.2.0 / mmengine 0.10.7。
    *   監査性: 全 Trace ID に証跡（コマンド出力・ログ・checkpoint パス）を残す。
    *   暗黙の fallback 禁止。無いものは明示的に記録し代替を明記する。
    *   `.venv/ work_dirs/ data/ *.log` はコミットしない。コミット・push はユーザー指示時のみ。
    *   media：`schedulefree==1.4.1`、`muon-optimizer`（KellerJordan/Muon）は pyproject に導入済み。
*   **非ゴール（今回やらないこと):**
    *   3D OBB / 3D pose 出力（SPAN 拡張や深度デュアルパスは今回対象外）。
    *   depth ブランチ。今回は RGB のみ（ユーザー指示で depth を無効にした）。
    *   画像回転データオーグメンテーション（RandomRotate）の実装は、まず smoke では省略し、必要なら後で追加。RandomFlip は rbox 対応済み。
    *   ONNX / OpenVINO エクスポート、mmdeploy。
    *   YOPO の 3D pose 用 head（DINO9DCenter2DPoseHead）の修正（既知のデコードバグは対象外）。
*   **成功条件:**
    1. 2D OBB モデルが build でき、train/val forward が NaN なしで完走する。
    2. rotate IoU 損失で smoke（数 epoch）が loss 減少（特に loss_bbox が下がる）ことを確認する。
    3. rotate IoU から **GWD** に切替えた config で、同様に smoke が loss 減少するを確認する。
    4. **Muon + ScheduleFree**（MuonScheduleFreeOptimizer）が DefaultOptimWrapperConstructor 経由で build でき、学習が回る（ブロッカー解消）。
    5. FULL 学習（50 epoch、HGNetV2-B2 + GWD + MuonScheduleFree）で loss が収束し、チェックポイントが保存される。
    6. 全 Trace ID に証跡が残る。
*   **リスクと前提:**
    *   **KNOWN ブロッカー**: `DefaultOptimWrapperConstructor` が `MuonScheduleFreeOptimizer` を `inspect.signature(optimizer_cls)` で解決できず `TypeError: None is not callable object`。→ オプティマイザの build 方式の再設計が必要。
    *   schedulefree の `AdamWScheduleFreeOptimizer`（既存 deim 実装）は `__new__` で `schedulefree.AdamWScheduleFree` を返すため constructor が通るが、MuonScheduleFreeOptimizer は普通の `Optimizer` 継承 + 内蔵 `_sf` のため通らない。→ 解決策は本作業書 フェーズ2 に記載。
    *   `RotatedIoULoss` は `diff_iou_rotated_2d`（mmcv.ops）を使い、初期 IoU≈0 で loss≈10。学習初期は loss_iou が高止まりするのは正常。
    *   GWD の `loss_iou` は log1p 距離で ~1.0 前後に落ち着く。主な収束指標は `loss_bbox`（L1、cxcywhr）。
    *   HGNetV2-B2 の `PPHGNetV2_B2_stage1.pth` は `~/.cache/torch/hub/checkpoints/` にある。

### 1.2 サブゴール構造

| ID | サブゴール | 目的との対応 | 成果物 | 検証方法 |
| :--- | :--- | :--- | :--- | :--- |
| SG-1 | 現状の移植コード・config・データを精査し、Muon+ScheduleFree ブロッカーの原因を特定 | 目標3 / ブロッカー | 原因分析メモ | ブロッカーが再現する config と traceback の記録 |
| SG-2 | Muon+ScheduleFree（MuonScheduleFreeOptimizer）を DefaultOptimWrapperConstructor で build できるよう修正 | 目標3 | 修正済み deim_optimizers.py | `tools/train.py` で build→学習が回る |
| SG-3 | rotate IoU 版 smoke（数 epoch）で loss 減少を確認 | 目標2 | smoke ログ | loss_bbox が減少 |
| SG-4 | GWD 版 smoke（数 epoch）で loss 減少を確認 | 目標2 | smoke ログ | loss_bbox が減少 |
| SG-5 | FULL 学習（50 epoch、GWD + MuonScheduleFree）で収束・checkpoint 保存 | 目標1,4,5 | checkpoint + ログ | loss 収束、checkpoint 存在 |
| SG-6 | コミット・push、claude-mem、作業記録 | 監査性 | commit・push・claude-mem id | git log / SELECT MAX(id) |

### 1.3 トレーサビリティ方針

| Trace ID | 要求・制約 | 対応する作業要素 | 証跡 |
| :--- | :--- | :--- | :--- |
| TR-1 | 2D OBB head を共有 encoder から派生（明示要求2） | SG-3,4,5 / フェーズ1-3 | forward ログ、smoke ログ |
| TR-2 | rotate IoU→GWD（明示要求3） | SG-3,4 / フェーズ2,3 | 両 config の smoke ログ（loss_bbox 減少） |
| TR-3 | Muon+ScheduleFree（明示要求4） | SG-1,2 / フェーズ2 | build 成功ログ、学習 step ログ |
| TR-4 | HGNetV2 軽量 backbone、ResNet50 を使わない（明示要求5） | SG-5 / フェーズ1,3 | config の backbone 設定、ログ |
| TR-5 | YOPO self-contained（明示要求6） | 全 SG | 移植ファイル一覧、import ログ |
| TR-6 | 監査性・証跡 | SG-6 / フェーズ4 | git log、claude-mem id、作業記録 |

---

## 2. 作業内容

### フェーズ 1: 現状把握とブロッカー特定（見積: 0.8h）

このフェーズでは、前作業者が移植した rotate 実装と config を精査し、Muon+ScheduleFree のブロッカー（DefaultOptimWrapperConstructor の解決失敗）を特定します。

1.  **移植済みファイルの確認：**
    *   **タスク内容：** 以下のファイルが正しく存在し、YOPO 内で import 可能か確認する。
        - `yopo/structures/bbox/rotated_boxes.py`（RotatedBoxes, reg=5, le90）
        - `yopo/structures/bbox/bbox_overlaps.py`（`rbbox_overlaps` 追加済み）
        - `yopo/models/losses/rotated_iou_loss.py`（RotatedIoULoss）
        - `yopo/models/losses/gaussian_dist_loss.py`（GDLoss: gwd/kld）
        - `yopo/models/dense_heads/rotated_deformable_detr_head.py`（RotatedDeformableDETRHead）＋ `rotated_detr_head.py`
        - `yopo/models/task_modules/assigners/match_cost.py`（RBoxL1Cost / CenterL1Cost / GDCost / RotatedIoUCost）
        - `yopo/datasets/dota_tomato.py`（DOTATomatoDataset）
    *   **目的：** 移植が漏れなく行われているか、import が通るかを確認し、前提を固める。
    *   **対応サブゴール/Trace ID：** SG-1 / TR-5
2.  **既存 config の確認：**
    *   **タスク内容：** `configs/yopo/rotated_deformable_detr_tomato_obb.py`（rotate IoU）と `..._obb_gwd.py`（GWD）を読み、現状を記録する。特に `optim_wrapper`（MuonScheduleFree）と backbone（HGNetV2-B2）設定。
    *   **目的：** smoke／FULL 実行時の起点を正確に把握する。
    *   **対応サブゴール/Trace ID：** SG-1 / TR-1, TR-4
3.  **ブロッカーの再現と原因特定：**
    *   **タスク内容：** `tools/train.py configs/yopo/rotated_deformable_detr_tomato_obb_gwd.py --cfg-options train_cfg.max_epochs=1` を実行し、`DefaultOptimWrapperConstructor` の `inspect.signature(optimizer_cls)` で `TypeError: None is not callable object` が再現するか確認する。
    *   **目的：** Muon+ScheduleFree の build が何故壊れるか、正確な原因（optimizer_cls が None になる理由）を特定する。
    *   **対応サブゴール/Trace ID：** SG-1, SG-2 / TR-3
4.  **MuonScheduleFreeOptimizer の build 方式の解決策を選定：**
    *   **タスク内容：** `DefaultOptimWrapperConstructor`（`mmengine/optim/optimizer/default_constructor.py` の `__call__`）が `optimizer_cls = self.optimizer_cfg['type']` → `OPTIMIZERS.switch_scope_and_registry(None)` 下で `registry.get(...)` し、`inspect.signature(optimizer_cls)` へ渡す。**`switch_scope_and_registry(None)` はスコープを None（デフォルト）に切替えるため、custom scope=yopo の登録が見えなくなり `registry.get` が None を返す**のが `None is not callable object` の最有力原因。次を確認し、採用案を本書に明記する。
        - **確認A**: `default_constructor` が解決する時だけ None になるのか、直接 `OPTIMIZERS.switch_scope_and_registry(None)` でも None になるのかを判別。`from yopo.registry import OPTIMIZERS` と `from mmengine.registry import OPTIMIZERS` を両方試す。
        - **採用候補**:
          - 案B（推奨）: `DefaultOptimWrapperConstructor` を**使わない**。`optim_wrapper = dict(type='ScheduleFreeOptimWrapper', constructor=None, optimizer=dict(type='MuonScheduleFreeOptimizer', params=model.parameters(), ...))` とするが、config 内で仮パラメータを参照できず複雑 → より現実的には、`Runner` の `build_optim_wrapper` が `OptimWrapper` に対して `optimizer` に `params` を注入する方式を利用し、`constructor=None` を明示。
          - 案A: `MuonScheduleFreeOptimizer` に `register` 済みのまま、`__new__` を追加して実 optimizer（`torch.optim.Optimizer` サブクラス）を返す形にし、`DefaultOptimWrapperConstructor` が解決できるようにする（既存 `AdamWScheduleFreeOptimizer` と同じ pattern。ただし Muon と ScheduleFree が同居するので、返す実オブジェクトは両者を束ねたカスタム Optimizer にする）。
          - 案C: `optim_wrapper.optimizer.type` をフルパス（`'yopo.engine.optimizers.deim_optimizers.MuonScheduleFreeOptimizer'`）にして、スコープ切替後も解決できるようにする。
    *   **目的：** ブロッカーを解消し、Muon+ScheduleFree が mmengine Runner で回るようにする。**実装はフェーズ2 手順5 で行う。**
    *   **対応サブゴール/Trace ID：** SG-2 / TR-3

### フェーズ 2: Muon+ScheduleFree の build 修正と rotate IoU smoke（見積: 1.5h）

このフェーズでは、フェーズ1 で選定した方針で MuonScheduleFreeOptimizer の build 問題を解決し、rotate IoU 版の smoke を回します。

5.  **MuonScheduleFreeOptimizer の build 修正：**
    *   **タスク内容：** フェーズ1 手順4 で選定した案に基づき、`yopo/engine/optimizers/deim_optimizers.py` の `MuonScheduleFreeOptimizer` を修正する。最も有望なのは **案B**（`constructor=None` の専用 build）または **案A**（`__new__` で実optimum返却）。
    *   **目的：** `tools/train.py` でモデル build → optimizer build → 学習 が通るようにする。
    *   **対応サブゴール/Trace ID：** SG-2 / TR-3
6.  **rotate IoU 版の 1 epoch smoke：**
    *   **タスク内容：** `configs/yopo/rotated_deformable_detr_tomato_obb.py` で 1 epoch smoke を実行（`--cfg-options train_cfg.max_epochs=1`）。backbone は HGNetV2-B2、optim は MuonScheduleFree（修正後）。
    *   **目的：** NaN なしで学習が回ることを確認。loss_bbox / loss_iou の初期値を記録。
    *   **対応サブゴール/Trace ID：** SG-3 / TR-1, TR-2
7.  **rotate IoU 版 数 epoch（3〜5 ep）で loss 減少確認：**
    *   **タスク内容：** 5 epoch で回し、loss_bbox が減少することを確認（前作業の GWD 5ep では loss 13.3→10.4、loss_bbox 2.08→1.21）。
    *   **目的：** rotate IoU でも学習が機能する（loss が下がる）ことを実証する。
    *   **対応サブゴール/Trace ID：** SG-3 / TR-2

### フェーズ 3: GWD smoke と FULL 学習（見積: 2.0h）

このフェーズでは、GWD 版に切替えて smoke した後、FULL 学習（50 epoch）を実行します。

8.  **GWD 版 smoke（数 epoch）：**
    *   **タスク内容：** `configs/yopo/rotated_deformable_detr_tomato_obb_gwd.py`（GWD）で 5 epoch smoke。backbone=HGNetV2-B2、optim=MuonScheduleFree。
    *   **目的：** GWD でも loss_bbox が減少することを確認（前作業で確認済みだが、Muon+ScheduleFree 化後の再確認）。
    *   **対応サブゴール/Trace ID：** SG-4 / TR-2
9.  **FULL 学習（50 epoch）：**
    *   **タスク内容：** GWD config で 50 epoch をバックグラウンド実行。`--work-dir work_dirs/rddetr_tomato_gwd_full`。loss_bbox / loss_iou の推移を監視。
    *   **目的：** 学習が収束し、checkpoint が保存されることを確認（成功条件5）。
    *   **対応サブゴール/Trace ID：** SG-5 / TR-1, TR-2, TR-4
10. **checkpoint / ログ検証：**
    *   **タスク内容：** 保存された checkpoint（`epoch_50.pth`）とログを確認。loss 収束・NaN なしを確認。
    *   **目的：** FULL 学習完走の証跡を残す。
    *   **対応サブゴール/Trace ID：** SG-5 / TR-6

### フェーズ 4: 反映・記録（見積: 0.5h）

11. **commit & push：** 移植分と config、作業書を `rgb-d` へ反映（ユーザー指示時のみ push）。
12. **claude-mem 更新：** 2D OBB 学習完了（rotate IoU / GWD / Muon+ScheduleFree / FULL 収束）を記録。
13. **handover / 作業記録：** 本作業書の作業記録テーブルに全フェーズの開始・完了・結果を追記。

---

## 3. 作業チェックリスト

*作業を完了したら `[ ]` を `[x]` に変更します。*

### フェーズ 1: 現状把握とブロッカー特定

### 手順 1: 移植済みファイルの import 確認
- [x] 🖐 **操作**: `yopo/structures/bbox/rotated_boxes.py`, `bbox_overlaps.py`（rbbox_overlaps）, `losses/rotated_iou_loss.py`, `losses/gaussian_dist_loss.py`, `dense_heads/rotated_deformable_detr_head.py`, `task_modules/assigners/match_cost.py`, `datasets/dota_tomato.py` が存在し、`MODELS`/`TASK_UTILS`/`DATASETS`/`OPTIMIZERS` registry に登録されているか、`init_default_scope('yopo')` + import で確認。
- [x] 🔎 **確認**: 全クラスが import・build でき、registry に存在する。
- [x] 🧪 **テスト**: `.venv/bin/python -c "from yopo.registry import MODELS, TASK_UTILS, DATASETS, OPTIMIZERS; print('RotatedDeformableDETRHead' in MODELS, 'RBoxL1Cost' in TASK_UTILS, 'DOTATomatoDataset' in DATASETS, 'MuonScheduleFreeOptimizer' in OPTIMIZERS)"` が全て True。
- [x] 🛠 **エラー時対処**: ModuleNotFoundError が出たら、該当ファイルの import 文を YOPO パス（`yopo.*`）に修正。

### 手順 2: 既存 config の確認
- [x] 🖐 **操作**: `configs/yopo/rotated_deformable_detr_tomato_obb.py` と `..._obb_gwd.py` の `backbone` / `encoder` / `decoder` / `bbox_head` / `optim_wrapper` を読み、差分を記録。
- [x] 🔎 **確認**: backbone は HGNetV2-B2、loss_iou は rotate IoU 版 / GWD 版でそれぞれ異なる。optim_wrapper は MuonScheduleFreeOptimizer。
- [x] 🧪 **テスト**: `.venv/bin/python -c "from mmengine.config import Config; c=Config.fromfile('configs/yopo/rotated_deformable_detr_tomato_obb_gwd.py'); print(c.model.backbone.name, c.optim_wrapper.optimizer.type)"` が `B2`, `yopo.engine.optimizers.deim_optimizers.MuonScheduleFreeOptimizer`。
- [x] 🛠 **エラー時対処**: config 読めない場合は import 文のミスを確認。

### 手順 3: Muon+ScheduleFree ブロッカー再現・原因特定
- [x] 🖐 **操作**: `tools/train.py configs/yopo/rotated_deformable_detr_tomato_obb_gwd.py --cfg-options train_cfg.max_epochs=1` を実行し、traceback を記録。
- [x] 🔎 **確認**: `DefaultOptimWrapperConstructor` / `inspect.signature` / `TypeError: None is not callable object` が再現する。
- [x] 🧪 **テスト**: 修正前 1 epoch smoke が optimizer build 前に失敗する（ブロッカー再現）ことを確認。修正後の成功テストは手順5・6で実施する。
- [x] 🛠 **エラー時対処**: `optimizer_cls` の解決が失敗している場合は、`optim_wrapper.optimizer.type` の registry 解決（`OPTIMIZERS.switch_scope_and_registry(None)` 相当）が通るかを確認。

### 手順 4: build 方式の解決策選定
- [x] 🖐 **操作**: 手順3 で特定した原因に基づき、案A（`__new__` で実optimum返却）/ 案B（`constructor=None` 専用 wrapper）/ 案C を比較し、採用案を本書に明記。
- [x] 🔎 **確認**: 採用案と理由（他案を捨てた理由）が記載されている。
- [x] 🧪 **テスト**: 採用案を実装前の「既存失敗テスト」として、失敗理由が明確。
- [x] 🛠 **エラー時対処**: 複数案で迷う場合は、実装コスト・リスクを表にしてユーザー判断を仰ぐ。

**採用決定（2026-08-24 08:05 UTC）:** 案Cを採用する。`default_scope=None` を維持し、optimizer と wrapper の両方を完全修飾 type で指定することで `DefaultOptimWrapperConstructor` を正しく利用する。案Bの `constructor=None` は MMEngine 0.10.7 で registry type として無効であり、実測で `TypeError: type must be a str or valid type, but got NoneType` となった。案Aの `__new__` は既に複合 Optimizer として持つ Muon/ScheduleFree の状態管理を複雑化するため採用しない。あわせて、複合 Optimizer の公開 `param_groups` に ScheduleFree パラメータを含め、勾配消去・scheduler・checkpoint state を一貫させる。

### フェーズ 2: Muon+ScheduleFree 修正と rotate IoU smoke

### 手順 5: MuonScheduleFreeOptimizer の build 修正
- [x] 🖐 **操作**: 手順4 で採用した案C（完全修飾 type + `DefaultOptimWrapperConstructor`）を `configs/yopo/rotated_deformable_detr_tomato_obb.py`、`..._obb_gwd.py` と `yopo/engine/optimizers/deim_optimizers.py` に反映。複合 optimizer の公開 param_groups・ScheduleFree state・scheduler 同期を追加。
- [x] 🔎 **確認**: `tools/train.py` と同じ config/model build 経路で optimizer build が成功する。`MuonScheduleFreeOptimizer` の Muon パラメータ（2D/4D）と ScheduleFree パラメータ（ベクトル）の両方が `optimizer.param_groups` に渡る。
- [x] 🧪 **テスト**: 早期に失敗していた 1 epoch smoke（`--cfg-options train_cfg.max_epochs=1`）が、修正後に build を通過して学習 step へ進む（失敗→成功）。
- [x] 🛠 **エラー時対処**: まだ `inspect.signature` エラーが出るなら、`optimizer_cfg['type']` の値と `OPTIMIZERS.get(...)` を直接 print して None の有無を特定。`switch_scope_and_registry(None)` 由来なら案C（フルパス type）に切替。

### 手順 6: rotate IoU 版 1 epoch smoke
- [x] 🖐 **操作**: `tools/train.py configs/yopo/rotated_deformable_detr_tomato_obb.py --work-dir work_dirs/riou_muonsf_1ep --cfg-options train_cfg.max_epochs=1`
- [x] 🔎 **確認**: 1 epoch（1249枚 / batch8 = 157 iter 相当）完走、NaN なし。
- [x] 🧪 **テスト**: epoch1 終了時の `loss_bbox` / `loss_iou` を record。
- [x] 🛠 **エラー時対処**: NaN は Muon の lr 高すぎ等を疑い `muon_lr` を下げる。OOM は batch を下げる。

### 手順 7: rotate IoU 版 5 epoch で loss 減少確認
- [x] 🖐 **操作**: 5 epoch でバックグラウンド実行（`--cfg-options train_cfg.max_epochs=5`）。
- [x] 🔎 **確認**: `loss_bbox` が epoch 経過で減少する。
- [x] 🧪 **テスト**: epoch1 vs epoch5 の `loss_bbox` を比較し、減少を確認。
- [x] 🛠 **エラー時対処**: loss 減少しない場合は lr 調整（`muon_lr` / `sf_lr`）や Muon 実装を疑う。

### フェーズ 3: GWD smoke と FULL 学習

### 手順 8: GWD 版 5 epoch smoke
- [x] 🖐 **操作**: `tools/train.py configs/yopo/rotated_deformable_detr_tomato_obb_gwd.py --cfg-options train_cfg.max_epochs=5`
- [x] 🔎 **確認**: loss_bbox が減少（前作業では 2.08→1.21）。
- [x] 🧪 **テスト**: epoch1 vs epoch5 の loss_bbox 比較。
- [x] 🛠 **エラー時対処**: GWD 特有なら `tau`/`fun` を調整。

### 手順 9: FULL 学習（50 epoch）
- [x] 🖐 **操作**: `nohup .venv/bin/python tools/train.py configs/yopo/rotated_deformable_detr_tomato_obb_gwd.py --work-dir work_dirs/rddetr_tomato_gwd_full > /tmp/opencode/gwd_full.log 2>&1 &`（detach 推奨・例: `setsid ... &`）
- [x] 🔎 **確認**: 進捗（`loss_bbox` 収束、NaN なし）を periodic に確認。
- [x] 🧪 **テスト**: 50 epoch 完了後、現行 FULL run の `work_dirs/rddetr_tomato_gwd_amp_eval_full/epoch_50.pth` が存在。
- [x] 🛠 **エラー時対処**: 即死はログの traceback から修正後再起動。

### 手順 10: checkpoint / ログ検証
- [x] 🖐 **操作**: `work_dirs/rddetr_tomato_gwd_amp_eval_full/` の `epoch_50.pth` とログを確認。
- [x] 🔎 **確認**: loss 収束、NaN なし、checkpoint 保存。
- [x] 🧪 **テスト**: `ls work_dirs/rddetr_tomato_gwd_amp_eval_full/epoch_*.pth`。
- [x] 🛠 **エラー時対処**: checkpoint が無い場合は学習失敗原因を確認。

### 手順 10-A: 学習済み checkpoint の 2D OBB RGB オーバーレイ
- [x] 🖐 **操作**: 50 epoch 後、`rbbox_mAP_50` が最高の今回の checkpoint を使い、validation RGB 画像へ予測 2D OBB（回転矩形・score）を重畳して PNG を出力。
- [x] 🔎 **確認**: 出力画像が実データの RGB を背景にし、予測 OBB が画像座標・回転角へ正しく描画されている。
- [x] 🧪 **テスト**: 少なくとも3枚の PNG と出力元 checkpoint path を確認し、生成物を作業記録へ記載。
- [x] 🛠 **エラー時対処**: config/checkpoint の class 定義または可視化器の rbox 形式が不一致なら、まず単一画像 inference の tensor shape・角度単位を検証する。

### フェーズ 4: 反映・記録

### 手順 11: commit & push（ユーザー指示時のみ）
- [ ] 🖐 **操作**: `git add`（対象ファイルのみ）→ commit → `git push origin rgb-d`（指示時のみ）。
- [ ] 🔎 **確認**: `work_dirs/ data/ *.log` が含まれないこと。
- [ ] 🧪 **テスト**: `git log --oneline -1`。
- [ ] 🛠 **エラー時対処**: 不要物混入を防ぐため `git status` を確認し、明示対象だけ add。

### 手順 12: claude-mem 更新
- [ ] 🖐 **操作**: 2D OBB 学習完了（rotate IoU / GWD / Muon+ScheduleFree / FULL 収束）を observation に追加。
- [ ] 🔎 **確認**: 新 id が追加される。
- [ ] 🧪 **テスト**: `SELECT MAX(id) FROM observations` が増える。
- [ ] 🛠 **エラー時対処**: created_at_epoch 等 NOT NULL 制約に注意（INSERT カラム充足）。

### 手順 13: 作業記録の締め
- [ ] 🖐 **操作**: 本章の作業記録テーブルに全フェーズの開始・完了・結果を追記。
- [ ] 🔎 **確認**: 注意事項（時刻・両端記録・結果備考）を遵守。
- [ ] 🧪 **テスト**: 全チェックリストが `[x]`。
- [ ] 🛠 **エラー時対処**: 未記録項目は完了後すぐ補完。

---

## 4. 作業に使用するコマンド参考情報

### 基本的な開発ワークフロー

```bash
cd /home/kasm-user/Desktop/YOPO_clone
just env-doctor          # 環境確認（torch/cuda/GPU）
# uv sync は通常不要（すでに構築済み・schedulefree/muon 導入済み）
```

### モデル build 検証

```bash
.venv/bin/python - <<'EOF'
from mmengine.registry import init_default_scope
init_default_scope('yopo')
from mmengine.config import Config
from yopo.registry import MODELS, TASK_UTILS, DATASETS, OPTIMIZERS
print('head', 'RotatedDeformableDETRHead' in MODELS)
print('cost', 'RBoxL1Cost' in TASK_UTILS)
print('ds', 'DOTATomatoDataset' in DATASETS)
print('opt', 'MuonScheduleFreeOptimizer' in OPTIMIZERS)
cfg = Config.fromfile('configs/yopo/rotated_deformable_detr_tomato_obb_gwd.py')
m = MODELS.build(cfg.model)
print('model OK', type(m).__name__)
EOF
```

### 1 epoch smoke（rotate IoU 版 / GWD 版）

```bash
# rotate IoU 版
.venv/bin/python tools/train.py configs/yopo/rotated_deformable_detr_tomato_obb.py \
    --work-dir work_dirs/riou_muonsf_1ep --cfg-options train_cfg.max_epochs=1

# GWD 版
.venv/bin/python tools/train.py configs/yopo/rotated_deformable_detr_tomato_obb_gwd.py \
    --work-dir work_dirs/gwd_muonsf_1ep --cfg-options train_cfg.max_epochs=1
```

### FULL 学習（50 epoch、GWD + MuonScheduleFree）

```bash
setsid bash -c '.venv/bin/python tools/train.py configs/yopo/rotated_deformable_detr_tomato_obb_gwd.py \
    --work-dir work_dirs/rddetr_tomato_gwd_full > /tmp/opencode/gwd_full.log 2>&1' \
    < /dev/null > /dev/null 2>&1 &
# 進捗確認
grep -aoE "loss_bbox: [0-9.]+" /tmp/opencode/gwd_full.log | tail
```

### ブロッカー（Muon+ScheduleFree）のデバッグ

```bash
.venv/bin/python - <<'EOF'
from yopo.registry import OPTIMIZERS
print('resolved:', OPTIMIZERS.get('MuonScheduleFreeOptimizer'))
# switch_scope_and_registry(None) での解決を再現
with OPTIMIZERS.switch_scope_and_registry(None) as reg:
    print('under switch_current:', reg.get('MuonScheduleFreeOptimizer'))
EOF
```

### claude-mem 確認

```bash
python3 -c "import sqlite3; c=sqlite3.connect('/home/kasm-user/.claude-mem/claude-mem.db'); print(c.execute('SELECT MAX(id) FROM observations').fetchone())"
```

---

## 5. 対象データ／参考リソース

- **データ**: `/workspace/data/tomato_obb_detection/fruits_detection_data_Jun30-2025_dota`（DOTA 形式、`stem` 1クラス、train 1249 / valid 330 / test 330、画像 736x512）。ラベルは 1行が `` x1 y1 x2 y2 x3 y3 x4 y4 stem 0 ``。qbox（8頂点）は DOTATomatoDataset 内で `cv2.minAreaRect` により rbox（cx,cy,w,h,angle[rad]）へ変換済み。
- **pretrained**: `~/.cache/torch/hub/checkpoints/PPHGNetV2_B2_stage1.pth`（HGNetV2-B2）。`dino-4scale_r50...pth`（ResNet50 用、今回は backbone に使わない）。
- **参考実装（DEIM_sandbox 方式）**: `/home/kasm-user/Desktop/DEIM_sandbox` の `DEIM/engine/optim/optim.py`（AutoMuonWithAuxAdam: muon_lr=0.005, momentum=0.95, ns_steps=5, adam_lr=0.00025 / AdamWScheduleFreeOptimizer: sf_lr=0.00025, betas(0.9,0.95)）。config 例: `configs/deim_dfine/deim_hgnetv2_m_coco_tomato_timm_muon.yml`。
- **既知の前作業結果（GWD 5ep・Had AdamW）**: loss 13.3→10.4、loss_bbox 2.08→1.21、loss_iou(GWD)~1.0 横ばい。

---

## 6. 完了の定義

*作業が最後まで完了したら `[ ]` を `[x]` にしつつ、作業が本当に完了したかをチェックします*

- [ ] 観点1: 2D OBB（共通 encoder + head）が build でき、rotate IoU / GWD 両方で loss_bbox が減少（成功条件1,2,3）。
- [ ] 観点2: Muon + ScheduleFree（MuonScheduleFreeOptimizer）が build でき、学習が回る（成功条件4、ブロッカー解消）。
- [ ] 観点3: FULL 学習（50 epoch、HGNetV2-B2 + GWD + MuonScheduleFree）で loss が収束し、`epoch_50.pth` が保存される（成功条件5）。
- [ ] 観点4: 全 Trace ID（TR-1..6）に対応する証跡が作業記録（§7）に残っている。
- [ ] 観点5: 暗黙 fallback を使わず、例外・未対応事項（例: ブロッカー、Muon 実装の制限）は明示的に記録されている。

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
| `2026-08-24` | `07:47:59 UTC` | opencode | 作業書作成（write-workdoc-uv）+ claude-mem 記録 | 2D OBB 学習（rotate IoU→GWD、Muon+ScheduleFree、HGNetV2-B2）の作業書を作成。claude-mem に id=17 で現状を記録（前作業からの引き継ぎ）。 |
| `2026-08-24` | `08:03:03 UTC` | Codex | フェーズ1再開・SG-2 調査開始 | `just env-doctor` 成功（RTX 4000 Ada / torch 2.4.0+cu121）。短縮名の optimizer/wrapper は `default_scope=None` 下の MMEngine registry で解決できず、1 epoch smoke は optimizer build 前に `TypeError: None is not a callable object` で停止。さらに既存の複合 optimizer は ScheduleFree 側パラメータを `param_groups` に含めず、`zero_grad()` と checkpoint state が欠落することを最小テストで確認。 |
| `2026-08-24` | `08:05:13 UTC` | Codex | チェックリスト逐次実行開始 | 作業書のパスと手順を再確認。以後、未完了チェックボックスを上から 1 つずつ処理し、各完了直後にこの記録を更新する。 |
| `2026-08-24` | `08:05:13 UTC` | Codex | 手順1・操作完了: 移植済み rotate 実装の存在・import | 8 対象ファイルがすべて存在。`.venv/bin/python` で RotatedBoxes / rbbox_overlaps / rotate losses / rotated heads / assigner costs / DOTATomatoDataset の対象モジュールを全て import 成功。 |
| `2026-08-24` | `08:05:13 UTC` | Codex | 手順1・確認完了: registry 解決 | `RotatedDeformableDETRHead`、`RotatedIoULoss`、`GDLoss`、`RBoxL1Cost`、`GDCost`、`DOTATomatoDataset`、`MuonScheduleFreeOptimizer` が各 YOPO registry で全て解決した。 |
| `2026-08-24` | `08:05:13 UTC` | Codex | 手順1・テスト完了: 最小 registry チェック | 作業書指定の 4 判定が `True True True True`。 |
| `2026-08-24` | `08:05:13 UTC` | Codex | 手順1・エラー時対処確認 | ModuleNotFoundError は発生しなかったため修正不要。失敗時の修正方針（`yopo.*` import へ統一）は作業書どおり維持。 |
| `2026-08-24` | `08:05:13 UTC` | Codex | 手順2・操作完了: rotate IoU/GWD config 差分確認 | 両者とも RotatedDeformableDETRHead と Muon+ScheduleFree 指定。GWD は HGNetV2-B2 / 4-layer transformer / GWD、rotate IoU は ResNet50 / 6-layer transformer / RotatedIoULoss のままで、HGNetV2-B2 要件に不一致を発見した。 |
| `2026-08-24` | `08:05:13 UTC` | Codex | 手順2・確認中: HGNetV2-B2 への統一 | rotate IoU config を GWD と同じ HGNetV2-B2 / 4-layer shared transformer / batch 8 に修正し、loss のみ RotatedIoULoss とした。初回の確認スクリプトは config が解決済み class object を返す点を誤って文字列比較し assertion 失敗。実装の不具合ではないため class identity 比較で再検証する。 |
| `2026-08-24` | `08:05:13 UTC` | Codex | 手順2・確認完了: config 要件 | 両 config は HGNetV2-B2、encoder/decoder は各4層、head は RotatedDeformableDETRHead。rotate IoU 側は RotatedIoULoss、GWD 側は GDLoss、双方 MuonScheduleFreeOptimizer を使用することを class identity で確認。 |
| `2026-08-24` | `08:05:13 UTC` | Codex | 手順2・テスト完了: Config 再読込 | GWD config は `B2 yopo.engine.optimizers.deim_optimizers.MuonScheduleFreeOptimizer` を返し、完全修飾名を用いた registry 解決を確認。 |
| `2026-08-24` | `08:05:13 UTC` | Codex | 手順2・エラー時対処確認 | Config.fromfile は両 config で成功。import エラーは無く、対処は不要。 |
| `2026-08-24` | `08:05:13 UTC` | Codex | 手順3・操作完了: 修正前ブロッカー証跡 | `/tmp/opencode/rddetr_muonsf_1ep.log` に `DefaultOptimWrapperConstructor` の line 308、`inspect.signature(optimizer_cls)`、`TypeError: None is not a callable object` を確認。生成 work dir は config copy のみで checkpoint 無し。 |
| `2026-08-24` | `08:05:13 UTC` | Codex | 手順3・確認完了: registry 解決失敗の原因 | `mmengine.registry.OPTIMIZERS.get('MuonScheduleFreeOptimizer')` は `None`、完全修飾名は MuonScheduleFreeOptimizer class を返す。default_scope=None 下で短縮名を使ったことが traceback の直接原因。 |
| `2026-08-24` | `08:05:13 UTC` | Codex | 手順3・テスト完了: 修正前 smoke 回帰証跡 | 旧 `/tmp/opencode/rddetr_muonsf_1ep.log` に TypeError を確認し、`work_dirs/rddetr_tomato_muonsf_1ep/epoch_1.pth` が無いことを確認。現在の修正済み config を失敗させる再現は行わない。 |
| `2026-08-24` | `08:05:13 UTC` | Codex | 手順3・エラー時対処完了 | `optim_wrapper.optimizer.type` と global registry を直接照合し、短縮名→完全修飾名への切替で解決することを確定した。 |
| `2026-08-24` | `08:05:13 UTC` | Codex | 手順4・操作完了: build 方針選定 | `constructor=None` は MMEngine で `NoneType` エラー、完全修飾 type + DefaultOptimWrapperConstructor は複合 optimizer build 成功。案C を採用し、ScheduleFree 側の grad/state/scheduler 連携も補正する方針を作業書に追記。 |
| `2026-08-24` | `08:05:13 UTC` | Codex | 手順4・確認完了 | 作業書 §3 手順4直後に案Cの採用理由、案A/Bの不採用理由、複合 optimizer の状態管理要件が記載されていることを確認。 |
| `2026-08-24` | `08:05:13 UTC` | Codex | 手順4・テスト完了: 短縮 type の失敗 | 最小 Linear model でも `optimizer.type='MuonScheduleFreeOptimizer'` は `DefaultOptimWrapperConstructor` の `TypeError: None is not a callable object` を再現。完全修飾名が必要なことを回帰テストとして確定。 |
| `2026-08-24` | `08:05:13 UTC` | Codex | 手順4・エラー時対処確認 | 案B/C を MMEngine 0.10.7 の実 builder で比較し、案Cだけが成功したため判断の曖昧さは無い。追加のユーザー判断は不要。 |
| `2026-08-24` | `08:05:13 UTC` | Codex | 手順5・操作完了: 複合 optimizer 修正 | 両 OBB config の optimizer/wrapper type を完全修飾化。MuonScheduleFreeOptimizer は vector(ScheduleFree) param group を公開し、scheduler 同期、state_dict/load_state_dict を実装。`git diff --check` 成功。 |
| `2026-08-24` | `08:05:13 UTC` | Codex | 手順5・確認完了: 実 OBB model optimizer build | GWD config の HGNetV2-B2 OBB model で ScheduleFreeOptimWrapper/MuonScheduleFreeOptimizer を build。Muon group 17,476,064 要素、ScheduleFree group 60,776 要素で、両者が public param_groups に存在した。 |
| `2026-08-24` | `08:13:30 UTC` | Codex | 定期記録: 行動カウント更新 | 手順1〜4は全チェック済み。手順5は実装反映と実 OBB model の optimizer build まで成功し、次は修正後 1 epoch smoke で学習 step への到達を検証する段階。 |
| `2026-08-24` | `08:15:48 UTC` | Codex | 手順5・テスト完了: GWD 1 epoch smoke | `work_dirs/gwd_muonsf_1ep` で 157/157 iter 完走、`epoch_1.pth` 保存。loss_bbox は iter 50 の 2.5854 から iter 150 の 1.0196 へ低下、loss_iou は約0.88、NaN/traceback 無し、GPU memory 9,213 MiB。 |
| `2026-08-24` | `08:15:48 UTC` | Codex | 手順5・エラー時対処確認 | `DefaultOptimWrapperConstructor` の inspect.signature エラーは再発しなかった。短縮 type が原因だったことは手順3/4で証明済みで、完全修飾 type により解消。 |
| `2026-08-24` | `08:18:34 UTC` | Codex | 手順6・操作完了: rotate IoU 1 epoch smoke | `work_dirs/riou_muonsf_1ep` で 157/157 iter 完走、`epoch_1.pth` 保存。loss_bbox は iter 50 の3.2900→iter 150 の2.8799、rotate IoU loss は13.1912→10.7236、NaN/traceback無し、GPU memory 9,213 MiB。 |
| `2026-08-24` | `08:18:34 UTC` | Codex | 手順6・確認完了: 1 epoch 完走・NaN検査 | `Epoch(train) [1][150/157]`、`epoch_1.pth`、ログ内の NaN/traceback 不在を機械検査して確認。 |
| `2026-08-24` | `08:18:34 UTC` | Codex | 手順6・テスト完了: epoch1 metric | 最終 logger window: loss_bbox=2.8799、loss_iou=10.7236、loss_cls=0.2019、grad_norm=329.1164。初期 IoU≈0 のため高めの loss_iou は作業書の既知挙動と整合。 |
| `2026-08-24` | `08:18:34 UTC` | Codex | 手順6・エラー時対処確認 | NaN/OOM は発生せず、batch=8、Muon lr=0.005 を維持。対処不要。 |
| `2026-08-24` | `08:19:29 UTC` | Codex | 手順7・操作完了: rotate IoU 5 epoch 起動 | `setsid timeout 1800 tools/train.py ... --work-dir work_dirs/riou_muonsf_5ep --cfg-options train_cfg.max_epochs=5` を PID 1385855 で起動。stdout/stderr は `/tmp/opencode/riou_muonsf_5ep.log`。 |
| `2026-08-24` | `08:29:57 UTC` | Codex | 手順7・確認完了: rotate IoU 5 epoch 収束確認 | 5 epoch 完走、epoch_1〜5.pth 保存。各 epoch 最終 logger window の loss_bbox は 2.9420, 2.9074, 2.9527, 2.9934, 2.7385 で、epoch1→5 に 0.2035 低下。NaN/traceback 無し。 |
| `2026-08-24` | `08:29:57 UTC` | Codex | 定期記録: 行動カウント更新 | 手順7の5 epoch rotate IoU run が完走。次は epoch1/5 metric をテスト項目として記録し、GWD 5 epoch smoke へ進む。 |
| `2026-08-24` | `08:29:57 UTC` | Codex | 手順7・テスト完了: epoch1/5 main loss_bbox 比較 | `Epoch(train)` 行の主 ` loss_bbox: ` を decoder 補助損失と区別して抽出。epoch1=2.9420、epoch5=2.7385（-0.2035）。補助 `d2.loss_bbox` を誤取得しないことも確認。 |
| `2026-08-24` | `08:29:57 UTC` | Codex | 手順7・エラー時対処確認 | 5 epoch で主 loss_bbox が減少したため、muon_lr/sf_lr の調整は不要。 |
| `2026-08-24` | `08:31:11 UTC` | Codex | 手順8・操作完了: GWD 5 epoch 起動 | `setsid timeout 1800 tools/train.py ... --work-dir work_dirs/gwd_muonsf_5ep --cfg-options train_cfg.max_epochs=5` を PID 1394484 で起動。stdout/stderr は `/tmp/opencode/gwd_muonsf_5ep.log`。 |
| `2026-08-24` | `08:41:22 UTC` | Codex | 手順8・確認完了: GWD 5 epoch の loss_bbox 低下 | 主 `loss_bbox` は epoch1=1.0512、epoch2=0.9117、epoch3=0.8657、epoch4=0.8611、epoch5=0.8377。epoch1→5 で 0.2135 低下し、Traceback/RuntimeError/ERROR は検出されなかった。 |
| `2026-08-24` | `08:41:22 UTC` | Codex | 手順8・テスト完了: GWD epoch1/5 比較 | `Epoch(train)` の主損失を抽出し epoch1=1.0512、epoch5=0.8377、差分=-0.2135 を確認。`epoch_1.pth` と `epoch_5.pth` の両方が存在し、比較は合格。 |
| `2026-08-24` | `08:41:22 UTC` | Codex | 手順8・エラー時対処確認 | `GWDLoss(fun='log1p', tau=1)` で 5 epoch を NaN/例外なく完走し、loss_bbox も低下。`tau`/`fun` の調整は不要。 |
| `2026-08-24` | `08:42:29 UTC` | Codex | 手順9・操作完了: GWD FULL 50 epoch 起動 | `setsid timeout 14400 .venv/bin/python tools/train.py configs/yopo/rotated_deformable_detr_tomato_obb_gwd.py --work-dir work_dirs/rddetr_tomato_gwd_full` を PID 1403303 で起動。stdout/stderr は `/tmp/opencode/gwd_full.log`。 |
| `2026-08-24` | `08:44:55 UTC` | Codex | 定期記録／手順9・確認完了: FULL 学習の初期進捗 | epoch1 の主 loss_bbox は iter50=2.5598、iter100=1.4791、iter150=0.9996 と低下。PID 1403303 は稼働中で、ログに NaN/Traceback/RuntimeError/ERROR は無い。行動カウントの定期リフレッシュ時点の状況を記録。 |
| `2026-08-24` | `09:08:30 UTC` | Codex | 定期記録: FULL 学習継続状況 | PID 1403303 は稼働中。epoch_1〜13.pth の 13 checkpoint を確認し、ログに NaN/Traceback/RuntimeError/ERROR は無い。主 loss_bbox は epoch1 終盤=0.9996、epoch5 終盤=0.8374 まで低下。 |
| `2026-08-24` | `09:24:28 UTC` | Codex | 手順9・中断記録: FULL 学習の外部要因による停止 | epoch22 の iter150 までは NaN/OOM 無しで進行したが、`work_dirs/rddetr_tomato_gwd_full/20260824_084240/vis_data/20260824_084240.json` が存在しないため LoggerHook が `FileNotFoundError` で停止。親 work-dir も消失し、epoch1〜21 checkpoint を含む成果物を確認できない。学習・GWD 実装の例外ではなく作業ディレクトリ削除が直接原因。 |
| `2026-08-24` | `09:25:46 UTC` | Codex | 手順9・再発防止／再開 | ユーザーの storage cleanup により前 run の work-dir が削除されたことを確認。GWD config に `CheckpointHook(interval=1, max_keep_ckpts=2, save_last=True)` を追加し、metric evaluator が無いので Top-K ではなく直近 2 checkpoint のみに制限。Config/Hook の生成検証後、PID 1436381 で FULL 50 epoch を最初から再起動。 |
| `2026-08-24` | `09:27:40 UTC` | Codex | 定期記録: FULL 再実行の初期進捗 | 再実行 PID 1436381 は epoch1・iter60/157 まで進行、主 loss_bbox=2.3359。NaN/Traceback/RuntimeError/ERROR は検出されず、checkpoint はまだ epoch 終了前のため未生成。 |
| `2026-08-24` | `09:35 UTC` | Codex | 追加要件: AMP・実 rIoU evaluator・容量制御 | ユーザー要請により `RotatedIoUMetric`（rbox mAP@0.50／matched rIoU）、valid 330 枚、5 epoch 間隔 validation、実 rIoU Top-2 checkpoint を追加。bf16 AMP の batch32 probe は `ms_deform_attn_forward_cuda` が BFloat16 未対応で停止したため、数値問題ではなく CUDA 演算子互換性として fp16 AMP + dynamic loss scaling に変更して再試行する。 |
| `2026-08-24` | `09:37 UTC` | Codex | AMP 容量測定 | fp16 AMP は batch32 で計算可能だが、19.45 GiB 使用後に追加 314 MiB を確保できず OOM。GPU 利用率を高く保ちつつ allocation 変動に耐える batch30 を採用し、1 epoch smoke で有限 loss・実使用量・rIoU evaluator を確認する。 |
| `2026-08-24` | `09:38 UTC` | Codex | AMP 容量測定（継続） | batch30 も 19.34 GiB 使用、free 204 MiB に対し追加 294 MiB が必要で OOM。allocated 18.75 GiB に対し reserved-unallocated 403 MiB があるため、次は batch を維持して `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を指定し、allocator 断片化を避ける smoke を行う。 |
| `2026-08-24` | `09:39 UTC` | Codex | AMP 容量測定（容量上限確定） | `expandable_segments:True` を指定しても batch30 は 19.31 GiB 使用後に追加 588 MiB を確保できず OOM。allocator 断片化ではなく VRAM 容量不足と判定し、fp16 AMP batch24 へ下げて有限 loss と安定した高使用率を確認する。 |
| `2026-08-24` | `09:41:42 UTC` | Codex | 定期記録: AMP 幾何演算の修正検証 | fp16 autocast で失敗した `torch.det`/LU のため、GDLoss と Hungarian GDCost を局所 fp32 化。実 CUDA test で GWD loss=float32・有限、pred gradient=有限、GDCost=float32・有限を確認。backbone/transformer は AMP のまま維持する。 |
| `2026-08-24` | `09:51 UTC` | Codex | AMP/evaluator smoke の中間結果 | fp16 AMP batch24、static loss scale=1.0、局所 fp32 GWD/GDCost、AMP target dtype 修正により 1 epoch を finite grad_norm（23.56→16.41）で完走。VRAM は 17.4〜17.8 GiB / 20.0 GiB。valid 330枚の実評価は `rbbox_mAP_50=0.0032`、recall=0.0466、precision=0.0391、matched rIoU=0.5738（epoch1 初期値）を出力し、metric key の供給を確認。 |
| `2026-08-24` | `10:00 UTC` | Codex | AMP/evaluator 5 epoch smoke 完了 | fp16 AMP batch24（VRAM 17.4 GiB、static loss scale=1.0）で 5 epoch を NaN/OOM/例外なく完走。epoch5終盤は loss_bbox=0.8340、GWD loss_iou=0.7983、grad_norm=20.0931。valid 330枚で `rbbox_mAP_50=0.0126`、recall=0.1040、precision=0.0874、matched rIoU=0.5902。`best_rbbox_mAP_50_epoch_5.pth`、`topk_epoch_6_greater0.0126.pth`、`epoch_5.pth`（各約72MB）が保存され、TopK pool=1/2 を確認。 |
| `2026-08-24` | `10:24:50 UTC` | Codex | 定期記録: FULL AMP/evaluator epoch15 | `work_dirs/rddetr_tomato_gwd_amp_eval_full` の50 epoch本走行は epoch17 に進行。実 rIoU validation は epoch5: mAP@0.50=0.0093 / matched rIoU=0.5872、epoch10: **0.0158** / **0.5949**（現best）、epoch15: 0.0114 / 0.5897。epoch15終盤の loss_bbox=0.7527、GWD loss_iou=0.7561、grad_norm=34.2974 は有限。Top-K は epoch5 の低値を削除し epoch10・15の2本だけを保持、GPU使用量は17,429 MiB。 |
| `2026-08-24` | `10:43:12 UTC` | Codex | 定期記録: FULL AMP/evaluator epoch29 | FULL run は epoch29 を完了し、NaN/Traceback/RuntimeError/OOM は0件、GPU使用量は17,429 MiBで安定。epoch20 の mAP@0.50=0.0086 / matched rIoU=0.5815 は一時低下したが、epoch25 は **mAP@0.50=0.0191、matched rIoU=0.6011** に上昇してbestを更新。訓練損失と実検出品質の非単調性を踏まえ、Top-K は `rbbox_mAP_50` で選別し、容量上限を維持している。 |
| `2026-08-24` | `11:02:05 UTC` | Codex | 定期記録: epoch43 とユーザー追加要件 | FULL AMP/evaluator run は epoch43 に進行し、NaN/OOM/例外0件。ユーザー要請により手順10-Aを追加し、50 epoch完了後に今回の `rbbox_mAP_50` 最高 checkpoint を流用して validation RGB 画像へ予測2D OBB（回転矩形・score）を重畳した PNG を少なくとも3枚出力・検証する。旧full runの削除済み path は現行 `rddetr_tomato_gwd_amp_eval_full` に訂正。 |
| `2026-08-24` | `11:13:14 UTC` | Codex | 手順9・テスト完了: FULL epoch_50 checkpoint | `work_dirs/rddetr_tomato_gwd_amp_eval_full/epoch_50.pth`（73,031,570 bytes）と `best_rbbox_mAP_50_epoch_50.pth`（73,031,186 bytes）の存在を確認。最終実評価は **rbbox_mAP@0.50=0.0242**、recall=0.1525、precision=0.1281、matched rIoU=0.6031。epoch50が新bestで、学習プロセスは正常終了。 |
| `2026-08-24` | `11:13:53 UTC` | Codex | 手順9・エラー時対処確認完了 | FULL log 全体で `Traceback`、`RuntimeError`、`OutOfMemoryError`、`NaN`、`FileNotFoundError` はすべて0件。局所fp32化したGWD/GDCost、fp16 AMP static scale=1.0、batch24の構成で修正・再起動を要する失敗は発生しなかった。 |
| `2026-08-24` | `11:14:09 UTC` | Codex | 手順10・操作完了: checkpoint/log 確認 | `epoch_50.pth` と `best_rbbox_mAP_50_epoch_50.pth` は各70MBで存在。最終ログは epoch50 の checkpoint 保存、validation 42/42、mAP@0.50=0.0242、Top-K 更新までを記録している。 |
| `2026-08-24` | `11:14:30 UTC` | Codex | 手順10・確認完了: 収束・数値安定・保存 | 主 `loss_bbox` は epoch1 iter50=2.5419 から epoch50 iter50=0.6537へ低下（約74%減）、GWD loss_iou=0.9362→0.6975。FULL log にNaN/例外はなく、epoch50/best checkpoint保存を確認。途中の抽出regexは項目順の仮定不一致だったため、logger原文を直接検証して結論を確定。 |
| `2026-08-24` | `11:14:51 UTC` | Codex | 手順10・テスト完了: epoch checkpoint 列挙 | `find .../epoch_*.pth` は `epoch_50.pth` 1本（73,031,570 bytes）を返した。`max_keep_ckpts=1` の想定どおり、最終epoch checkpointがストレージを不必要に占有していない。 |
| `2026-08-24` | `11:15:07 UTC` | Codex | 手順10・エラー時対処確認完了 | `epoch_50.pth` は非空で、ログに `Saving checkpoint at 50 epochs` と最終 `rbbox_mAP_50=0.0242` が存在。checkpoint 欠落・学習失敗の追加調査は不要。 |
| `2026-08-24` | `11:15:40 UTC` | Codex | 定期記録: 手順10-A overlay 準備 | 学習は50 epochで正常完了。現行 config の test pipeline は valid RGB（736×512）に適用でき、`best_rbbox_mAP_50_epoch_50.pth` を推論に使う方針を確認。標準可視化器は回転矩形に特化していないため、予測 `cx,cy,w,h,angle` をOpenCV回転四角形として描画する専用スクリプトを次に実装する。 |
| `2026-08-24` | `11:15:40 UTC` | Codex | 手順10-A・操作完了: RGB OBB overlay 生成 | 追加した `tools/analysis_tools/visualize_rotated_obb_predictions.py` により、`best_rbbox_mAP_50_epoch_50.pth` をGPU推論へ使用。valid RGB 3枚にscore≥0.05の上位15予測を `cx,cy,w,h,angle(rad)` の回転式で描画し、`work_dirs/rddetr_tomato_gwd_amp_eval_full/obb_overlays/*.png` とcheckpoint/configを含む `manifest.json` を出力。 |
| `2026-08-24` | `11:15:40 UTC` | Codex | 手順10-A・確認完了: RGB上の回転OBB可視確認 | 出力3枚を実表示で確認。いずれもトマト棚のRGB画像を背景に、果実位置・傾きに追従する緑の回転矩形とscoreを描画できた。score≥0.05・上位15件のため密集果実ではラベルが一部重なるが、座標系・angle(rad)→四隅変換は視覚的に整合。 |
| `2026-08-24` | `11:17:53 UTC` | Codex | 手順10-A・テスト完了: PNG/manifest 検証 | `obb_overlays/manifest.json` を検証し、PNGは3枚、manifest画像数も3、全出力ファイルが存在することを確認。各画像は15予測を描画し、出力元は絶対pathで `best_rbbox_mAP_50_epoch_50.pth`（mAP@0.50=0.0242）と明記されている。 |
| `2026-08-24` | `11:18:11 UTC` | Codex | 手順10-A・エラー時対処確認完了 | `py_compile` 成功、manifestの各 `boxes_xywha_rad` / `scores` 件数は `num_drawn` と一致。実行時のconfig/checkpoint class不一致・rbox tensor/angle単位エラーは発生しなかった。 |
| `2026-08-24` | `10:00 UTC` | Codex | 手順9・FULL 再開: AMP/evaluator 構成 | 旧 FP32 run と成果物を混同しないよう、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 付きで PID 1473165 を `work_dirs/rddetr_tomato_gwd_amp_eval_full` に起動。fp16 AMP batch24、static loss scale=1.0、5 epoch ごと valid 実 rIoU、Top-2＋最終 checkpoint（weights-only）で 50 epoch を実行する。 |
| `YYYY-MM-DD` | `HH:MM:SS TZ` | `作業者名` | フェーズ1開始: `[タスク名]` | 作業計画書確認完了、`[タスク]`の要件を把握 |
| `YYYY-MM-DD` | `HH:MM:SS TZ` | `作業者名` | フェーズ2開始: `[タスク名]` | `[実行コマンド]` で build / smoke 検証 |
| `YYYY-MM-DD` | `HH:MM:SS TZ` | `作業者名` | フェーズ3開始: `[FULL 学習]` | NaN 有無・loss_bbox 推移・memory |
| `YYYY-MM-DD` | `HH:MM:SS TZ` | `作業者名` | フェーズ4: `[commit・push・記録]` | コミット hash・claude-mem id・checkpoint パス |

---

## 8. 補足（作業者が参照すべき前提メモ）

- **リポジトリ**: `/home/kasm-user/Desktop/YOPO_clone`（branch `rgb-d`）。uv 環境 `.venv/`（torch 2.4.0+cu121 / mmcv 2.2.0 / mmengine 0.10.7）。GPU RTX 4000 Ada 20GB。
- **YOPO の rotate 実装移植状況（前作業者による、本作業の前提）**:
  - `yopo/structures/bbox/rotated_boxes.py`: `RotatedBoxes`（box_dim=5、`regularize_boxes` le90 対応、`rbox2corner`/`corner2rbox`/`overlaps`）。`yopo/structures/bbox/__init__.py` に `RotatedBoxes` と `rbbox_overlaps` を export。
  - `yopo/structures/bbox/bbox_overlaps.py`: `rbbox_overlaps`（`mmcv.ops.box_iou_rotated` ラッパ。coordinate clamp・wh clamp 付き）。
  - `yopo/models/losses/rotated_iou_loss.py`: `RotatedIoULoss`（`diff_iou_rotated_2d`, mode=log/linear/square）。`losses/__init__.py` に登録。
  - `yopo/models/losses/gaussian_dist_loss.py`: `GDLoss`（loss_type: gwd/kld/jd/kld_symmax/kld_symmin）。`gwd` は `sqrt` 引数を取らない点に注意（kld のみ sqrt）。`losses/__init__.py` に登録。
  - `yopo/models/dense_heads/rotated_detr_head.py`: `RotatedDETRHead`（reg_dim=5、angle_factor、`_get_targets_single` で rbox 正規化 + regularize、`_predict_by_feat_single` 5次元デコード）。
  - `yopo/models/dense_heads/rotated_deformable_detr_head.py`: `RotatedDeformableDETRHead`（`DeformableDETRHead` + `RotatedDETRHead` の多重継承。`_init_layers` で reg=5、forward で angle を残差に足さない。**`_init_layers` 末尾に `self.init_weights()` を追加済み** = reg bias を [0,0,-2,-2,-2] にする）。
  - `yopo/models/task_modules/assigners/match_cost.py`: `RBoxL1Cost` / `CenterL1Cost` / `GDCost`（kld） / `RotatedIoUCost`。`assigners/__init__.py` に export。
  - `yopo/datasets/dota_tomato.py`: `DOTATomatoDataset`（qbox→rbox 変換、`filter_data`、`get_cat_ids`）。
  - configs: `rotated_deformable_detr_tomato_obb.py`（rotate IoU）/ `..._obb_gwd.py`（GWD + MuonScheduleFree 期待）。
- **既知の主な問題**:
  - **ブロッカー**: `MuonScheduleFreeOptimizer`（`yopo/engine/optimizers/deim_optimizers.py`）は単体では BUILD/STEP/EVAL 成功するが、`DefaultOptimWrapperConstructor` 経由で `inspect.signature` が `None is not callable` になる。→ フェーズ1手順4 / フェーズ2手順5 で build 方式を解決する。
  - `RotatedIoULoss` の初期 loss は高め（~10、IoU≈0）で正常。収束は `loss_bbox` で判断。
  - GWD の `loss_iou` は log1p で ~1.0 に落ち着く。GWD は `sqrt` 引数を取らないため config に `sqrt` を入れない。
  - 従来 YOPO の 3D pose（DINO9DCenter2DPoseHead）は使用しない（本作業は 2D OBB のみ）。
- **mmrotate 由来の移植**: 元の mmrotate 実装は `/workspace/Project/rotated-rtmdet-mmrotate-sandbox/mmrotate/` にあり、import を `yopo.*` に書き換えた。細部の diff はそれらのファイルを参照。
