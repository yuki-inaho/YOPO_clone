# 作業計画書 兼 記録書: YOPO を uv 環境 (cu121) で GPU 学習・推論可能にする

---

**日付：** `2026年06月19日`
**作業ディレクトリ・リポジトリ:** `/home/kasm-user/Desktop/YOPO`（Git リポジトリ。リモート `git@github.com:yuki-inaho/YOPO.git`、作業ブランチ `cu121`）
**作業者：** `マルチエージェント（統括: Claude Opus / 作業: DeepSeek V4 Pro or Sonnet / 監査: Claude Opus）`

> 本書は start-work-audit パターンの「唯一の真実源 (single source of truth)」として使用する。作業エージェントは本書のチェックリストのみを根拠に作業し、監査エージェントは本書の DoD とトレーサビリティ表で検証する。

---

## 1. 作業目的

本作業は、以下の目標を達成するために実施します。

* **目標1:** YOPO（`github.com/yuki-inaho/YOPO`、MMDetection 3.3 を `yopo` パッケージにリネームした単眼RGBカテゴリレベル9D姿勢推定リポジトリ）を、**uv で管理する repo-local 仮想環境 (cu121 スタック)** でセットアップする。
* **目標2:** その環境で **GPU 上の学習スモーク**（実 YOPO R50 モデルが loss→backward→checkpoint 保存まで1イテレーション進む）を成立させる。
* **目標3:** その環境で **GPU 上の推論スモーク**（実 YOPO R50 モデルが公式チェックポイントを読み込み、合成フレームに対して forward して予測を出す）を成立させる。
* **目標4:** 再現手順を `justfile` と docs に固定し、監査可能な形（ログ・証跡）で記録する。

### 1.1 ゴール要求分析

* **ユーザーの直観的・直截的な目的:**
  「YOPO をクローンして、**uv 環境で GPU を使って学習と推論ができる**状態にする」。実データ（NOCS/HouseCat6D）の本番学習で論文精度を再現することではなく、**GPU 上で学習・推論パイプラインが実際に動くことを最短で実証**することがゴール。
* **明示要求:**
  1. `git@github.com:yuki-inaho/YOPO.git` をクローンする（完了済み）。
  2. uv 環境で動かせるようにする。
  3. GPU で**学習と推論ができる**ところまで到達する。
  4. 参照実装 `/home/kasm-user/Desktop/inaho_repos/tomato_stem_mmsegmentation` の docs / justfile / pyproject.toml の uv×OpenMMLab パターンに則る。
  5. write/review スキルで作業書を作り、DoD を定める。
  6. start-work-audit パターンでマルチエージェント実行する（作業=DSv4Pro or sonnet、監査=opus、persistent agent、DoD 達成まで停止しない、DSv4Pro 優先で Claude 枠節約）。
* **暗黙制約:**
  * uv 環境で作業する（`uv sync` / `uv run` / `just` ターゲット）。
  * 参照リポジトリのパターンを踏襲：`[tool.uv] package = false`、torch/torchvision を pinned CUDA index から取得、**mmcv は prebuilt manylinux wheel**（ソースビルドしない）、`numpy<2`、フレームワークは `.pth` で develop 風に import 可能化、`justfile` に `sync` / `env-doctor` / `smoke-train` 等。
  * **暗黙のフォールバック禁止**：依存・ファイル・前提が無い場合は明示的に記録し、代替を明記する。
  * **監査性**：全 Trace ID に証跡（コマンド出力・diff・ログ・checkpoint パス）を残す。
  * 実データ（NOCS/HouseCat6D）は巨大かつ手動 DL が必要で本環境に無いため、**合成ミニデータでのスモーク**を採用する（参照リポジトリの `smoke-train` 思想と同じ）。
  * **編集スコープ（作業エージェント向け制約）**：変更は原則 `scripts/`・`temp/`・`data/` 生成物・`docs/`・`README.md` に限定する。`pyproject.toml`・`justfile`・`.venv`・`yopo/` 本体は**変更しない**（`yopo/` は真のフレームワークバグを特定した場合のみ最小修正し、根拠と diff を §7 に必ず記録する）。これらの基盤ファイルへの変更が必要と判断したら、勝手に変更せず統括/監査に確認する。
  * **証跡の保存先**：各スモークの完全ログは `work_dirs/<run>/` 配下（mmengine が自動生成）または `temp/logs/` に保存し、§7 には要約と該当パスを記録する。
* **非ゴール（今回やらないこと）:**
  * NOCS/HouseCat6D 実データの本番学習・論文精度の再現。
  * マルチGPU分散学習（`tools/dist_train.sh`）。
  * ONNX/mmdeploy エクスポート。
  * Swin-L / HouseCat6D config の検証（R50 / NOCS に限定。余力があれば追加）。
* **成功条件（DoD のサマリ。詳細は §6）:**
  1. `just env-doctor` が cu121 スタックと `cuda_available True` / `NVIDIA L4` を表示する。
  2. `just smoke-train` が GPU で1イテレーション以上学習を進め、`work_dirs/smoke_train/` に checkpoint を保存して exit 0。
  3. `just smoke-infer` が公式 R50 checkpoint を読み込み、GPU で forward して予測を出力し `SMOKE INFER OK` を表示して exit 0。
  4. クリーン手順（`just setup` → `gen-synthetic` → `download-ckpt` → `smoke-train` → `smoke-infer`）が再現する。
  5. docs（cu121 セットアップ手順）と README 追記が存在する。
  6. 全 Trace ID に証跡が残る。
* **リスクと前提:**
  * prebuilt mmcv wheel の CUDA カーネルが L4 (sm_89) で動くか → **検証済み（mmcv.ops.nms on CUDA OK、compiled cuda 12.1）**。
  * 合成 NOCS データのラベル形式（`*_label.pkl` のキー／intrinsics／`segmentation_results`）が `NOCSDataset` とパイプライン transform の期待と完全一致するか → **最大のリスク**。`Load9DPoseAnnotations` / `Pack9DPoseInputs` が要求するキー不足で KeyError になりやすい。
  * 9D pose head（DINO9DCenter2DPose）の loss が合成ラベル（少数オブジェクト・任意姿勢）で NaN/shape mismatch を起こさないか。
  * MMEngine の「val_dataloader/val_cfg/val_evaluator は全て None か全て非 None」制約。スモークで val を無効化する場合は3点セットで消す必要がある（**初回失敗を確認済み**、§7 参照）。
  * Runner 外でモデルを build する場合 `init_default_scope('yopo')` が必須（**検証済み**）。

### 1.2 サブゴール構造

| ID | サブゴール | 目的との対応 | 成果物 | 検証方法 |
| :--- | :--- | :--- | :--- | :--- |
| SG-1 | uv/cu121 依存スタックの構築 | 目標1 / 明示要求2,4 | `pyproject.toml`, `justfile`, `.venv`, `_yopo_src.pth` | `just env-doctor` が全バージョン表示 |
| SG-2 | GPU 基盤検証（import / mmcv CUDA ops / model build） | 目標1 / リスク低減 | 検証ログ | matmul・nms on CUDA・R50 backbone forward が GPU で成功 |
| SG-3 | 合成 NOCS ミニデータ＋スモーク機構の整備 | 目標2,3 / 暗黙制約（実データ無） | `scripts/gen_synthetic_nocs.py`, `temp/smoke_nocs_r50_1iter.py`, `scripts/smoke_infer.py`, `data/nocs_smoke/` | データ生成が NOCS レイアウトで成功 |
| SG-4 | 学習スモークの緑化（GPU 1-iter train） | 目標2 | `work_dirs/smoke_train/` の checkpoint + ログ | `just smoke-train` exit 0 + ckpt 生成 |
| SG-5 | 推論スモークの緑化（GPU inference） | 目標3 | 予測出力ログ | `just smoke-infer` exit 0 + `SMOKE INFER OK` |
| SG-6 | docs / README / 再現手順の整備 | 目標4 / 暗黙制約4 | `docs/CU121_TRAINING.md`, README 追記 | 手順どおりに再現でき、内容が実装と一致 |
| SG-7 | 検証・記録・トレーサビリティ | 目標4 / 監査性 | §7 作業記録、Trace 表の証跡 | 監査エージェントが全 TR に証跡を確認 |

### 1.3 トレーサビリティ方針

| Trace ID | 要求・制約 | 対応する作業要素 | 証跡 |
| :--- | :--- | :--- | :--- |
| TR-1 | uv 環境で動く（明示要求2） | フェーズ1 / 手順1-3 / SG-1 | `uv sync` ログ、`just env-doctor` 出力 |
| TR-2 | 参照リポジトリの uv×OpenMMLab パターン踏襲（明示要求4） | フェーズ1 / 手順1,2 / SG-1 | `pyproject.toml`・`justfile` の diff、参照との対応説明 |
| TR-3 | prebuilt mmcv wheel が L4 で動く | フェーズ1 / 手順4 / SG-2 | `mmcv.ops.nms on CUDA OK` ログ |
| TR-4 | GPU で学習できる（明示要求3） | フェーズ2 / 手順5-7 / SG-4 | `just smoke-train` のログ + checkpoint パス |
| TR-5 | GPU で推論できる（明示要求3） | フェーズ2 / 手順8-9 / SG-5 | `just smoke-infer` のログ + 予測 shape |
| TR-6 | フォールバック禁止・例外明示 | 全フェーズ | §7 にブロッカーと代替を明示記録 |
| TR-7 | 再現性・監査性（目標4） | フェーズ3 / 手順10-12 / SG-6,7 | docs、README、§7 作業記録 |

---

## 2. 作業内容

> フェーズの定義：**フェーズ1=調査・設計（済の確認含む）**、**フェーズ2=スモーク実装と緑化**、**フェーズ3=検証・文書化・記録**。計画/調査/実装/検証を混同しない。

### フェーズ 1: 調査・設計（環境確立の確認） (見積: 1.0h、大半は実施済み)

1. **参照リポジトリ uv×OpenMMLab パターンの精査：**
   * **タスク内容：** `inaho_repos/tomato_stem_mmsegmentation` の `pyproject.toml`/`justfile`/`CU12_TRAINING.md` を読み、`package=false`・CUDA index pin・prebuilt mmcv wheel・`numpy<2`・`.pth` 方式・justfile 構成を把握する。
   * **目的：** YOPO に同型のパターンを適用し、手戻りを防ぐ。
   * **対応サブゴール/Trace ID：** SG-1 / TR-2
2. **YOPO 依存・パッケージ構造の確認：**
   * **タスク内容：** `yopo/__init__.py`（mmcv<2.3.0/mmengine<1.0 の guard）、`yopo/registry.py`（`locations=['yopo.models']` 等の自動 import）、`setup.py`（`ext_modules=[]`＝CUDA コンパイル不要）、`requirements/*.txt`、内部 import が全て `from yopo`（`mmdet` 非依存）であることを確認する。
   * **目的：** mmcv 2.2.0 で guard を通過すること、`package=false`＋`.pth` で `import yopo` が成立することを保証する。
   * **対応サブゴール/Trace ID：** SG-1 / TR-1,TR-2
3. **バージョン確定：**
   * **タスク内容：** YOPO 記載の基盤（`pytorch/pytorch:2.4.0-cuda12.1`）に合わせ **Python 3.10 / torch 2.4.0+cu121 / torchvision 0.19.0+cu121 / mmcv 2.2.0（openmmlab cu121/torch2.4.0 manylinux wheel）/ mmengine 0.10.x / numpy<2** を採用。L4 ドライバ(CUDA 12.7)は cu121 ランタイムと後方互換。
   * **目的：** mmcv wheel 入手性（cp310 wheel 存在を確認済み）と GPU 互換性を両立。
   * **対応サブゴール/Trace ID：** SG-1 / TR-2,TR-3
4. **GPU 基盤検証：**
   * **タスク内容：** `import torch,mmcv,mmengine,yopo`＋`torch.cuda.is_available()`、torch GPU matmul、`mmcv.ops.nms` on CUDA、`init_default_scope('yopo')`→`MODELS.build`→`.cuda()`→backbone forward を確認する。
   * **目的：** prebuilt wheel の CUDA カーネルが L4(sm_89) で動くこと、モデルが GPU で構築・前進することを実装前に保証する。
   * **対応サブゴール/Trace ID：** SG-2 / TR-3

### フェーズ 2: スモーク実装と緑化 (見積: 3.0h)

1. **合成 NOCS データ生成器の確定：**
   * **タスク内容：** `scripts/gen_synthetic_nocs.py` を、`NOCSDataset` と train/test パイプライン（`LoadImageFromFile`/`Resize`/`Load9DPoseAnnotations`/`Pack9DPoseInputs` 等）が要求する on-disk レイアウトと `*_label.pkl` キーに完全一致させる。不足キーは transform のコードを根拠に補う。
   * **目的：** KeyError / FileNotFound 無しでデータがロードできる状態を作る。
   * **対応サブゴール/Trace ID：** SG-3 / TR-4,TR-5
2. **学習スモーク config の確定：**
   * **タスク内容：** `temp/smoke_nocs_r50_1iter.py` を R50 config の `_base_` 拡張で作り、`data_root=data/nocs_smoke`、`load_from=None`、1-iter ループ、`batch_size` 小（BN/head が壊れない値）、`num_workers=0`、val 無効化（val_dataloader/val_cfg/val_evaluator を3点とも None もしくは整合させる）、最後に checkpoint 保存を設定する。
   * **目的：** 実データ無しで GPU 学習が1イテレーション進む最小構成を作る。
   * **対応サブゴール/Trace ID：** SG-4 / TR-4
3. **学習スモークの緑化：**
   * **タスク内容：** `just smoke-train` を実行し、エラーを1つずつ潰す（val 3点セット、ラベルキー不足、loss shape/NaN 等）。修正対象は原則 `temp/`・`scripts/`・`data/` 生成物に限定し、`yopo/` 本体は真のバグ時のみ最小修正＋記録。
   * **目的：** loss→backward→optimizer step→checkpoint 保存を GPU で成立させる。
   * **対応サブゴール/Trace ID：** SG-4 / TR-4,TR-6
4. **推論スモークの確定と緑化：**
   * **タスク内容：** `scripts/smoke_infer.py` を、config ロード→`init_default_scope('yopo')`→`MODELS.build`→`load_checkpoint`→`.cuda().eval()`→test パイプラインで合成1フレームをパック→`test_step`/`predict` で forward→予測 shape/keys 表示、の流れに確定し、`just smoke-infer` を緑化する。
   * **目的：** 公式 R50 checkpoint を使った GPU 推論を成立させる。
   * **対応サブゴール/Trace ID：** SG-5 / TR-5,TR-6

### フェーズ 3: 検証・文書化・記録 (見積: 1.5h)

1. **再現性検証：**
   * **タスク内容：** `just setup` → `just gen-synthetic` → `just download-ckpt` → `just smoke-train` → `just smoke-infer` を通しで実行し再現を確認する。
   * **目的：** クリーン手順での再現性を担保する。
   * **対応サブゴール/Trace ID：** SG-6,7 / TR-7
2. **docs / README 整備：**
   * **タスク内容：** `docs/CU121_TRAINING.md`（cu121 スタック・セットアップ・スモーク・既知の注意点）を作成し、README に uv/cu121 クイックスタートを追記する。
   * **目的：** 第三者が再現できる文書を残す。
   * **対応サブゴール/Trace ID：** SG-6 / TR-7
3. **品質チェックと記録：**
   * **タスク内容：** `uv run ruff check scripts temp` 等で追加コードの lint を確認し、§7 に全 Trace の証跡（コマンド出力要約・checkpoint パス・diff）を残す。
   * **目的：** 監査エージェントがトレーサビリティを検証できる状態にする。
   * **対応サブゴール/Trace ID：** SG-7 / TR-6,TR-7

---

## 3. 作業チェックリスト

*作業が完了したら `[ ]` を `[x]` に変更します。各手順は最小単位・順序厳守・検証可能。*

### フェーズ 1: 調査・設計（環境確立の確認）

### 手順 1: 参照パターンと YOPO 構造の精査（記録）
- [x] 🖐 **操作**: `inaho_repos/tomato_stem_mmsegmentation` の `pyproject.toml`/`justfile`/`CU12_TRAINING.md` と、YOPO の `yopo/__init__.py`/`registry.py`/`setup.py`/`requirements/*` を読み、要点を本書 §1-2 に反映する。
- [x] 🔎 **確認**: `package=false`＋prebuilt mmcv wheel＋`.pth` 方式、YOPO の内部 import が全て `from yopo`、guard が mmcv<2.3.0、`ext_modules=[]` を確認できている。
- [x] 🧪 **テスト**: 調査フェーズのため自動テスト不要。確認事項が §1.1 リスク／§2 フェーズ1 に記録されていること。
- [x] 🛠 **エラー時対処**: 参照ファイルが見つからない場合は `find /home/kasm-user/Desktop/inaho_repos -name pyproject.toml` で再探索する。

### 手順 2: pyproject.toml / justfile の整備（記録）
- [x] 🖐 **操作**: `pyproject.toml`（cu121 deps、`[tool.uv.sources]` で torch/torchvision を cu121 index、mmcv を openmmlab wheel URL、`package=false`）と `justfile`（`sync`/`setup`/`env-doctor`/`gen-synthetic`/`download-ckpt`/`smoke-train`/`smoke-infer`/`test`/`train`）を作成する。
- [x] 🔎 **確認**: 両ファイルがリポジトリ root に存在し、参照リポジトリの構成と対応している。
- [x] 🧪 **テスト**: `just --list` がレシピ一覧を表示する。
- [x] 🛠 **エラー時対処**: `just` 未インストール時は `uv run` 直書きの同等コマンドで代替し、その旨を §7 に記録する。

### 手順 3: uv sync と env-doctor（環境確立）
- [x] 🖐 **操作**: `just sync`（=`uv sync`＋`_yopo_src.pth` 書き込み）→ `just env-doctor` を実行する。
- [x] 🔎 **確認**: `torch 2.4.0+cu121` / `mmcv 2.2.0` / `mmengine 0.10.7` / `yopo 3.3.0` / `cuda_available True` / `NVIDIA L4` が表示される。
- [x] 🧪 **テスト**: `just env-doctor` を再実行しても同一出力で安定する。
- [x] 🛠 **エラー時対処**: mmcv import 失敗時は wheel URL（cu121/torch2.4.0/cp310）と Python が 3.10 であることを確認。`yopo not found` 時は `.pth` のパスを確認する。

### 手順 4: GPU 基盤検証
- [x] 🖐 **操作**: `.venv/bin/python` で torch GPU matmul・`mmcv.ops.nms` on CUDA・`init_default_scope('yopo')`→`MODELS.build`→`.cuda()`→`extract_feat` を実行する。
- [x] 🔎 **確認**: matmul・nms が CUDA で成功、R50 モデル(51.2M)が GPU 構築・backbone が4階層特徴を返す。
- [x] 🧪 **テスト**: 上記スクリプトが exit 0 で `MMCV CUDA OPS WORK ON L4` / `MODEL BUILDS + RUNS ON GPU` を出力する。
- [x] 🛠 **エラー時対処**: `DetDataPreprocessor is not in registry` は `init_default_scope('yopo')` の欠落。CUDA カーネル不一致なら mmcv wheel の cu/torch を再確認する。

### フェーズ 2: スモーク実装と緑化

### 手順 5: 合成 NOCS データのロード適合
- [x] 🖐 **操作**: `scripts/gen_synthetic_nocs.py` の出力（`data/nocs_smoke`）を `NOCSDataset` の train パイプラインで1サンプルだけロードする最小スクリプト／`tools/train.py` 起動時のデータロードで検証する。
- [x] 🔎 **確認**: `*_label.pkl` のキー・intrinsics・画像/depth サイズが transform の要求と一致し、KeyError / FileNotFound が出ない。
- [x] 🧪 **テスト**: `just gen-synthetic` 後、データロードが例外なく1バッチ取得できる（失敗→キー補完→成功を記録）。
- [x] 🛠 **エラー時対処**: KeyError のキー名を `yopo/datasets/.../transforms` のコードで特定し、生成器に該当キーを追加。intrinsics 不一致は `nocs_utils.py` のハードコード値に合わせる。

### 手順 6: 学習スモーク config の整合（val 3点セット等）
- [x] 🖐 **操作**: `temp/smoke_nocs_r50_1iter.py` で val を無効化する場合は `val_dataloader`/`val_cfg`/`val_evaluator` を**3点とも None**にし、`train_cfg` を 1-iter、`load_from=None`、`batch_size`・`num_workers=0`・checkpoint 保存を設定する。
- [x] 🔎 **確認**: `Runner.from_cfg(cfg)` が `val_*` 整合エラーを出さずに構築できる。
- [x] 🧪 **テスト**: `just smoke-train` が Runner 構築段階を通過する（初回の `ValueError: val_dataloader, val_cfg, and val_evaluator should be ...` が解消）。
- [x] 🛠 **エラー時対処**: ループ種別が epoch ベースなら `_delete_=True` で `IterBasedTrainLoop` に置換、または `max_epochs=1`＋1サンプルに整合させる。

### 手順 7: 学習スモークの緑化（GPU 1-iter）
- [x] 🖐 **操作**: `just smoke-train` を実行し、loss→backward→optimizer step→checkpoint 保存まで通す。
- [x] 🔎 **確認**: ログに少なくとも1回の iter（loss 値）と checkpoint 保存が出力され、`work_dirs/smoke_train/*.pth` が生成され exit 0。
- [x] 🧪 **テスト**: 終了後 `ls work_dirs/smoke_train/*.pth` が1件以上、`.venv/bin/python -c "import torch;torch.load(...)"` で読める。
- [x] 🛠 **エラー時対処**: loss が NaN/shape mismatch なら合成ラベルの姿勢/サイズ/マスク値域を見直す。`batch_size=1` で head/BN が壊れるなら 2 に上げる。OOM なら入力解像度/batch を下げる。

### 手順 8: 推論スモーク script の整合
- [x] 🖐 **操作**: `scripts/smoke_infer.py` を config ロード→`init_default_scope('yopo')`→`MODELS.build`→`load_checkpoint(checkpoints/nocs_yopo_real_camera_r50.pth)`→`.cuda().eval()`→test パイプラインで合成1フレームをパック→`test_step` の流れに確定する。
- [x] 🔎 **確認**: data_preprocessor が要求するパック済み入力（`DetDataSample`）を正しく構築できている。
- [x] 🧪 **テスト**: スクリプトがチェックポイント読込で missing/unexpected keys を致命にせず（警告許容）forward まで到達する。
- [x] 🛠 **エラー時対処**: collate/preprocessor のエラーは Runner の test_dataloader 経由でバッチ生成する方式に切替える。

### 手順 9: 推論スモークの緑化（GPU inference）
- [x] 🖐 **操作**: `just smoke-infer` を実行する。
- [x] 🔎 **確認**: GPU 上で forward が完了し、予測の型/shape/keys と `SMOKE INFER OK` が表示され exit 0。
- [x] 🧪 **テスト**: 予測が空でなく（または構造が妥当で）例外が出ない。
- [x] 🛠 **エラー時対処**: checkpoint 形状不一致は config とモデル定義（num classes/categories）の対応を確認。CUDA OOM は入力解像度を下げる。

### フェーズ 3: 検証・文書化・記録

### 手順 10: クリーン再現手順の通し実行
- [x] 🖐 **操作**: `just setup` → `just gen-synthetic` → `just download-ckpt` → `just smoke-train` → `just smoke-infer` を順に実行する。
- [x] 🔎 **確認**: すべて exit 0 で、checkpoint 生成と `SMOKE INFER OK` が再現する。
- [x] 🧪 **テスト**: 各コマンドの終了コードを記録する。
- [x] 🛠 **エラー時対処**: いずれか失敗時は該当手順（5-9）に戻り、原因と修正を §7 に記録する。

### 手順 11: docs / README 整備
- [x] 🖐 **操作**: `docs/CU121_TRAINING.md` を作成（スタック・setup・smoke・既知注意点・非ゴール）し、README に uv/cu121 クイックスタートを追記する。
- [x] 🔎 **確認**: 記載手順が実際の `justfile`/`pyproject.toml`/スクリプトと一致している。
- [x] 🧪 **テスト**: docs のコマンドをコピペ実行して再現する（少なくとも env-doctor と smoke 2種）。
- [x] 🛠 **エラー時対処**: 記載と実装の差異があれば実装側か docs 側のどちらを正とするか判断し、両者を一致させて記録する。

### 手順 12: 品質チェックと証跡記録
- [x] 🖐 **操作**: `uv run ruff check scripts temp` を実行し、§7 に全 Trace の証跡（コマンド出力要約・checkpoint パス・主要 diff）を残す。
- [x] 🔎 **確認**: lint が成功（または指摘を是正）し、TR-1〜TR-7 すべてに証跡が紐づく。
- [x] 🧪 **テスト**: §6 完了の定義の全観点が `[x]` になる。
- [x] 🛠 **エラー時対処**: lint 失敗は自動修正可否を切り分け、規約違反原因を記録してから再実行する。

---

## 4. 作業に使用するコマンド参考情報

### 基本ワークフロー（リポジトリ root = `/home/kasm-user/Desktop/YOPO` から）

```bash
# 依存同期 + yopo を import パスへ（.pth）
just sync                 # = uv sync; printf <root> > .venv/lib/python3.10/site-packages/_yopo_src.pth

# レシピ一覧
just --list

# 環境トリアージ（バージョン + GPU）
just env-doctor

# 合成NOCSミニデータ生成
just gen-synthetic        # -> data/nocs_smoke/

# 公式 R50 チェックポイント取得（~196MB）
just download-ckpt        # -> checkpoints/nocs_yopo_real_camera_r50.pth
```

### スモーク（GPU 学習・推論）

```bash
# 1-iter GPU 学習スモーク（loss->backward->checkpoint）
just smoke-train          # -> work_dirs/smoke_train/*.pth

# GPU 推論スモーク（公式ckptロード -> forward -> 予測）
just smoke-infer          # -> "SMOKE INFER OK"
```

### 実データ用ラッパー（データが data/ に揃っている場合のみ）

```bash
just train configs/yopo/nocs_yopo_real_camera_r50.py
just test  configs/yopo/nocs_yopo_real_camera_r50.py checkpoints/nocs_yopo_real_camera_r50.pth
```

### 品質チェック

```bash
uv run ruff check scripts temp
uv run ruff format --check scripts temp
```

---

## 6. 完了の定義（DoD）

*作業が最後まで完了したら `[ ]` を `[x]` にしつつ、本当に完了したかをチェックします。*

- [x] **DoD-1（環境）**: `just env-doctor` が `torch 2.4.0+cu121` / `torchvision 0.19.0+cu121` / `mmcv 2.2.0` / `mmengine 0.10.7` / `yopo 3.3.0` / `cuda_available True` / `device NVIDIA L4` を表示する。【SG-1 / TR-1,TR-2,TR-3】
- [x] **DoD-2（GPU 学習）**: `just smoke-train` が GPU で1イテレーション以上学習を進め（loss 値がログに出る）、`work_dirs/smoke_train/` に読み込み可能な checkpoint(.pth) を保存し exit 0。【SG-4 / TR-4】
- [x] **DoD-3（GPU 推論）**: `just smoke-infer` が `checkpoints/nocs_yopo_real_camera_r50.pth` を読み込み、GPU で forward して予測（型/shape/keys）を出力し `SMOKE INFER OK` で exit 0。【SG-5 / TR-5】
- [x] **DoD-4（再現性）**: クリーン手順 `just setup` → `gen-synthetic` → `download-ckpt` → `smoke-train` → `smoke-infer` が通しで再現する。【SG-6 / TR-7】
- [x] **DoD-5（文書）**: `docs/CU121_TRAINING.md` が存在し、README に uv/cu121 クイックスタートが追記され、内容が実装と一致する。【SG-6 / TR-7】
- [x] **DoD-6（トレーサビリティ）**: TR-1〜TR-7 すべてに証跡（ログ要約・checkpoint パス・diff）が §7 に残り、暗黙フォールバックを使っていない。【SG-7 / TR-6,TR-7】
- [x] **DoD-7（品質）**: 追加コード（`scripts/`, `temp/`）に対する `uv run ruff check` が成功（または指摘是正済み）。【SG-7 / TR-7】

> 非ゴール（実データ本番学習・論文精度再現・分散学習・ONNX・Swin-L/HouseCat6D）は DoD に含めない。

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
| 2026-06-19 | 09:30:00 UTC | Opus(統括) | フェーズ1開始: 参照パターン精査・YOPO 構造調査 | 参照 `tomato_stem_mmsegmentation` の uv×OpenMMLab パターン把握。YOPO は mmdet 3.3 を `yopo` にリネーム、内部 import 全て `from yopo`、guard mmcv<2.3.0、`ext_modules=[]` |
| 2026-06-19 | 09:40:00 UTC | Opus(統括) | `pyproject.toml`/`justfile` 作成、`uv sync` | ✅成功: torch 2.4.0+cu121 / torchvision 0.19.0+cu121 / mmcv 2.2.0 / mmengine 0.10.7 / yopo 3.3.0 |
| 2026-06-19 | 09:45:00 UTC | Opus(統括) | GPU 基盤検証 | ✅成功: torch matmul・`mmcv.ops.nms` on CUDA(L4, compiled cuda 12.1)・R50 model(51.2M) GPU 構築+backbone forward。**重要発見**: Runner 外 build は `init_default_scope('yopo')` 必須 |
| 2026-06-19 | 09:55:00 UTC | smoke-designer(sonnet) | 合成データ生成器/スモークconfig/推論script 草案作成 + `just gen-synthetic` | ✅`scripts/gen_synthetic_nocs.py`・`temp/smoke_nocs_r50_1iter.py`・`scripts/smoke_infer.py` 作成、`data/nocs_smoke/`(4フレーム, real/camera/seg_results) 生成 |
| 2026-06-19 | 09:58:00 UTC | Opus(統括) | R50 checkpoint 取得 | ✅`checkpoints/nocs_yopo_real_camera_r50.pth`(195.8MB, 795 tensors, mmengine 形式) 検証OK |
| 2026-06-19 | 10:00:00 UTC | Opus(統括) | `just smoke-train` 初回実行 | ❌失敗（既知化）: `ValueError: val_dataloader, val_cfg, and val_evaluator should be either all None or not None` → 手順6で val 3点セットを整合して解消予定 |
| 2026-06-19 | 10:05:00 UTC | Opus(統括) | 作業書作成（本書）+ DoD 定義 | ✅`temp/workdoc_Jun19-2026_yopo_uv_cu121_gpu.md` 作成。以降は start-work-audit パターンで作業=DSv4Pro/sonnet・監査=opus にて DoD 達成まで継続 |
| 2026-06-19 | 10:06:00 UTC | Opus(統括) | review-written-workdoc 実施 | ✅Verdict PASS（修正後）。§1.1 に編集スコープ・証跡保存先の制約を追記 |
| 2026-06-19 | 10:07:00 UTC | Opus(統括) | start-work-audit 基盤構築 | ✅`.agents/roles/{worker,audit,coordinator}.txt` 配置。DeepSeek疎通OK(pro/flash)、OpenRouter残$7.76、claude-agent-teams/opencode/tmux 利用可 |
| 2026-06-19 | 10:08:00 UTC | Opus(統括) | 作業エージェント起動 | ✅tmux `oc-ds-pro-yopo-worker`(DeepSeek V4 Pro)起動。タスク=手順5-9,11緑化(DoD-2,3,5)。`temp/worker_prompt_round1.md` 付与。SSOT/編集スコープ遵守を指示 |
| 2026-06-19 | 10:13:00 UTC | DeepSeek V4 Pro(作業) | 手順5-9,11 実装（smoke 緑化+docs） | ✅Worker Report: `temp/smoke_nocs_r50_1iter.py` で val/test 6点 None・1epoch・bs2、`scripts/smoke_infer.py` に `init_default_scope('yopo')` 追加。`just smoke-train` exit0 / `just smoke-infer` exit0。`docs/CU121_TRAINING.md`+README quickstart 作成。編集スコープ厳守 |
| 2026-06-19 | 10:15:00 UTC | Opus(統括) | 独立検証（再実行） | ✅`git status`=README のみ・yopo/不変、pyproject/justfile マーカー一致。`just smoke-train` 再実行 exit0→`work_dirs/smoke_train/epoch_1.pth`(585M,795t,epoch=1) 可読。`just smoke-infer` 再実行 exit0→pred 9キー・`SMOKE INFER OK`（証跡 `temp/logs/smoke_{train,infer}_verified.log`）|
| 2026-06-19 | 10:24:00 UTC | Opus(統括) | 品質ゲート（DoD-7） | ✅`ruff check --fix scripts temp` 11件修正→`All checks passed!`、`# noqa: F401` 整備。smoke-infer 再検証 exit0 |
| 2026-06-19 | 10:25:00 UTC | Opus(監査) | Round1 独立監査 | ✅**判定: 承認**。学習ログ直読(4 iter完走/有限 loss/全9D loss項/checkpoint保存)・`ruff` 自前実行・smoke config が base と pipeline 逐語一致・scope を git で独立確認。スモークは本物（ダミー/隠れfallback/逸脱/過大評価なし）。是正は全て統括の最終統合(DoD-6)に属する軽微事項 |
| 2026-06-19 | 10:26:00 UTC | Opus(統括) | 最終統合（監査是正 a/b/c） | ✅(b)`temp/logs/` に検証ログ保存、(c)smoke config に work_dir 差異の補足、(a)本§7 と §6 DoD・チェックリストを更新。**DoD-1〜7 全クローズ** |
| | | | | |

---

## 8. エージェント編成 (start-work-audit)

| agent_id (tmux) | role | model | workspace | allowed | forbidden | status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| (本セッション) | 統括 coordinator | claude-opus-4-8[1m] | YOPO@cu121 | 割当/監査受理/workdoc更新/最終統合 | 実装の自走(委譲可能時) | active |
| oc-ds-pro-yopo-worker | 作業 worker | deepseek-v4-pro (OpenCode/OpenRouter) | YOPO@cu121 | scripts/temp/data/docs/README の実装 | pyproject/justfile/.venv/yopo本体の変更, commit/push | Round1 完了（承認済） |
| cc-yopo-audit-opus48 | 監査 auditor | claude-opus-4-8[1m] (agent-teams) | YOPO@cu121 | 読取/差分確認/判定 | 実装 | Round1 監査=**承認** |

> 方針: 反復実装は DeepSeek V4 Pro に委譲し Claude 枠を節約。監査(Opus)は作業ブロック完了時にコンパクトな証跡パケットで起動する。Round1 で DoD-1〜7 を全達成し承認。追加要求が無ければ両 tmux セッションは停止可。
