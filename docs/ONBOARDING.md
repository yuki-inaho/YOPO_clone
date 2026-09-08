# LLMオンボーディングサマリー

> このドキュメントは、新任LLMエージェントがYOPO_cloneで作業を始めるための初期資料です。現時点で確認済みのリポジトリ状態、セットアップ手順、データ形式、学習・評価・可視化方法、注意点をまとめます。

## 1. プロジェクト概要と目的
- **プロジェクト名称・領域:** YOPO: You Only Pose Once。Monocular RGB category-level 9D multi-object pose estimation。
- **最終成果物:** RGB画像から複数物体のカテゴリ、2D bbox、3D translation、3D rotation、3D sizeを推定する学習・評価・推論環境。
- **ビジネス背景・価値:** RGB-onlyでカテゴリレベル9D姿勢推定を行うことで、深度センサなしのロボット認識・物体操作・AR/検査用途への適用可能性を高める。
- **現時点の進捗サマリ:** リポジトリは `~/Desktop/YOPO_clone` にクローン済み。uv仮想環境 `.venv` を作成済み。PyTorch 2.4.0 + CUDA 12.1、mmcv 2.2.0、mmengine 0.10.7、YOPO editable install済み。`tools/train.py --help` と `tools/test.py --help` は起動確認済み。

## 2. クリティカルな要求・制約
> 「壊してはいけない」品質・仕様ラインを箇条書きで列挙します。
- `data/`, `temp/`, `.venv/`, `work_dirs/`, checkpoint `*.pth` はリモートに含めない。`.gitignore` に `temp/` を追加済み。
- YOPO poseラベルは object/canonical座標から camera座標への姿勢として扱う。`p_cam = R @ p_obj + t` を前提にする。
- camera座標は `+X=画像右`, `+Y=画像下`, `+Z=カメラ前方` として扱う。投影は `u = fx * X / Z + cx`, `v = fy * Y / Z + cy`。
- NOCS / HouseCat6D の評価では translation 差分を `* 100` して cm にするため、通常は m 単位を前提にする。BOP由来データなど mm 単位の場合は変換が必要。
- Python 3.8環境では `list[float]` や `tuple[int, int]` の実行時評価で落ちるため、型注釈は `typing.List` か `from __future__ import annotations` で互換化する。
- `setup.py` のCUDA extension用torch importは遅延されており、metadata生成時にはtorchを要求しない。editable installは通常 `uv pip install --no-build-isolation -e .` で行える。
- `cityscapesscripts` は optional 依存だが Python 3.8で古い `typing` backport を要求して `uv pip check` を壊すため、この環境では外している。

## 3. 参照すべき合意済み資料
> 新任エージェントが必ず確認すべき一次資料の一覧です。パスと役割を記載します。

| 種別 | ファイル/リンク | 概要・用途 |
|------|------------------|------------|
| 要求定義書 | TBD | 独立した要求定義書は未作成。現状は `README.md` と本ドキュメントを参照する。 |
| 要件定義書 | `README.md` | 公式概要、環境、データ配置、学習・評価コマンド、Model Zoo。 |
| WBS / 進捗 | TBD | 独立したWBSは未作成。作業履歴はgit diffと会話ログを参照する。 |
| テスト資産 | `tools/train.py`, `tools/test.py`, `configs/yopo/*.py` | 学習・評価・可視化の実行入口。 |
| 既知課題リスト | 本ドキュメント「クリティカルな要求・制約」 | Python 3.8互換、optional依存、データ単位、BOP変換時の注意。 |
| データセット実装 | `yopo/datasets/pose_estimation/nocs_dataset.py` | NOCS形式の読み込み、ラベルキー、pose変換。 |
| データセット実装 | `yopo/datasets/pose_estimation/housecat6d_dataset.py` | HouseCat6D形式の読み込み、ラベルキー、pose変換。 |
| 可視化実装 | `yopo/visualization/pose_visualizer.py` | bbox、pose軸、3D cuboidのオーバーレイ描画。 |
| 評価実装 | `yopo/evaluation/metrics/nocs_metric.py`, `yopo/evaluation/metrics/housecat6d_metric.py` | 3D IoU、degree/cm評価、結果dump。 |
| 外部仕様 | https://mmdetection3d.readthedocs.io/en/latest/user_guides/coord_sys_tutorial.html | OpenMMLab camera座標系の確認。 |
| 外部仕様 | https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md | BOP `cam_R_m2c`, `cam_t_m2c` の model-to-camera 定義。 |

## 4. タスク境界（任せること / 任せないこと）
### 任せるタスク（例）
- uv環境の再構築、依存関係確認、`tools/train.py` / `tools/test.py` の起動確認。
- NOCS / HouseCat6D形式のデータ配置確認、ラベルpklのキー検査、サンプル読み込み検証。
- configを用いた学習・評価・可視化コマンドの組み立て、batch sizeやworker数の調整。
- Python 3.8互換や軽微な起動不具合の修正。
- 推論結果のdump、オーバーレイ画像保存、評価ログの整理。

### 任せないタスク（例）
- 未確認データをNOCS / HouseCat6D互換と断定すること。
- mm単位、m単位、camera/world座標の変換を根拠なしに決めること。
- ユーザーデータ、checkpoint、`temp/`、`data/`、`.venv/` をgit管理へ追加すること。
- 大規模なモデル構造変更や評価指標変更を、検証計画なしに実施すること。
- 外部データセットの利用規約、権利、配布可否を確認せずに再配布すること。

## 5. インタラクション方針
- **回答スタイル:** 日本語で簡潔に、見出しと箇条書きを中心にする。コマンドやパスはコード表記する。
- **回答手順:** まず現状確認、次に根拠となるファイル/行、最後に実行コマンドまたは次アクションを示す。
- **禁止事項・注意:** 未確定事項を断定しない。座標系・単位・ラベル形式は必ず実コードまたは一次資料で確認する。ユーザー未承認の破壊的git操作は禁止。
- **秘匿情報の扱い:** データセット、checkpoint、実験結果、認証情報、SSH鍵、APIキーは出力・コミットしない。必要なら `.gitignore` と `git status` で確認する。

## 6. 試行タスク（オンボーディング演習）
> 小さな検証タスクを2〜3件記載してください。理解度を確認するために実施します。
1. `source .venv/bin/activate` 後に `python tools/train.py --help` と `python tools/test.py --help` が通ることを確認する。
2. `configs/yopo/nocs_yopo_real_camera_r50.py` の `data_root`, `train_dataloader`, `val_dataloader`, `visualizer` を読み、NOCSの学習splitと評価splitを説明する。
3. 任意のラベルpklを1件読み、`class_ids`, `bboxes`, `translations`, `rotations`, `sizes/scales` または `gt_scales` がYOPO実装の期待と一致するか確認する。

## 7. 運用ルール・変更管理
- **ドキュメント更新時の記載ルール:** 実行コマンド、確認結果、参照ファイル、未確認事項を分けて記載する。推測は「推定」「未確認」と明示する。
- **TBDの扱い:** 未作成資料・未確認仕様はTBDとして残し、根拠が得られた時点で更新する。
- **レビュー/承認フロー:** コード修正後は `git diff` と関連コマンドの実行結果を確認する。大きな仕様変更、学習条件変更、評価指標変更は事前にユーザー確認する。
- **その他の運用ルール:** `data/`, `temp/`, `work_dirs/`, `.venv/` は作業用領域。成果物として残す必要がある場合も、git管理対象に含める前に確認する。

---

### 付録: 参考情報
- **主要リポジトリ/ディレクトリ:** `~/Desktop/YOPO_clone`, `configs/yopo/`, `yopo/datasets/pose_estimation/`, `yopo/models/dense_pose_heads/`, `yopo/evaluation/metrics/`, `yopo/visualization/`。
- **代表的なコマンド:**
  ```bash
  cd ~/Desktop/YOPO_clone
  source .venv/bin/activate

  python tools/train.py configs/yopo/nocs_yopo_real_camera_r50.py
  python tools/train.py configs/yopo/housecat6d_yopo_r50.py

  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash tools/dist_train.sh configs/yopo/nocs_yopo_real_camera_r50.py 4 --auto-scale-lr

  python tools/test.py configs/yopo/nocs_yopo_real_camera_r50.py <checkpoint.pth>
  python tools/test.py configs/yopo/nocs_yopo_real_camera_r50.py <checkpoint.pth> --show-dir overlay_results

  uv pip check --python .venv/bin/python
  ```
- **依存ライブラリ:** Python 3.8.10, uv, PyTorch 2.4.0+cu121, torchvision 0.19.0+cu121, mmcv 2.2.0, mmengine 0.10.7, openmim, pycocotools, scipy, shapely, terminaltables, plyfileなど。
- **セットアップ手順メモ:**
  ```bash
  uv venv --python 3.8 .venv
  uv pip install --python .venv/bin/python -U pip setuptools wheel
  uv pip install --python .venv/bin/python torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
  uv pip install --python .venv/bin/python openmim mmengine
  PATH="$PWD/.venv/bin:$PATH" .venv/bin/mim install mmcv==2.2.0
  uv pip install --python .venv/bin/python --no-build-isolation -e .
  uv pip install --python .venv/bin/python -r requirements.txt
  uv pip uninstall --python .venv/bin/python typing cityscapesscripts
  uv pip check --python .venv/bin/python
  ```
- **学習可能な主要データ形式:**
  - NOCS: `data/nocs/` 配下。`camera_train + real_train` で学習、`real_test` で評価。
  - HouseCat6D: `data/housecat6d/` 配下。`scene*` で学習、`test_scene*` で評価。
- **pose定義:** `translation=[X,Y,Z]` はcamera座標の物体中心。`rotation` はobject/canonical座標からcamera座標への3x3回転行列。YOPO内部では回転行列の第1列・第2列を連結した6D表現で学習する。
- **連絡先/責任者:** TBD。

> ※このドキュメントは必要に応じて拡張・縮退して構いません。記入済みのドキュメントはバージョン管理してください。
