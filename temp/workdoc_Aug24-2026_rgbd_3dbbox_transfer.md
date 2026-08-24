# 作業計画書 兼 記録書: RGB-D 3D BBOX transfer learning

---

**日付：** 2026年08月24日  
**作業ディレクトリ・リポジトリ:** `/home/kasm-user/Desktop/YOPO_clone`（Git repository `rgb-d` branch）  
**作業者：** Codex

---

## 1. 作業目的

本作業は、直前に完走した tomato 2D rotated-OBB detector の有効な重みを安全に再利用し、`clipboard.txt` に記載された RGB-D データで 3D BBOX を学習・評価できる、監査可能な訓練系を構築して実行するものである。

* **目標1:** `/home/kasm-user/Desktop/clipboard.txt` から、対象データ、3D BBOX の表現、split、評価条件を一次情報として確定する。
* **目標2:** 既存の RGB-D/MAE 実装を壊さず、RGB encoder の凍結と 2D OBB checkpoint の部分 transfer を明示した 3D BBOX 設計を実装する。
* **目標3:** depth encoder には、3D BBOX supervision 前に利用可能な深度のみで行う MAE 型自己教師あり事前学習を、成立条件が満たされる場合に限定して実施する。
* **目標4:** 実データで GPU 訓練・実 evaluator・可視化・再現可能な checkpoint/ログを残し、未承認 commit/push はしない。

### 1.1 ゴール要求分析

* **ユーザーの直観的・直截的な目的:** 2D の学習済み認識能力を足場に RGB-D から 3D BBOX を予測したい。RGB encoder をむやみに壊さず、depth は必要なら MAE で表現学習してから使いたい。
* **明示要求:** (a) まず `clipboard.txt` を読む、(b) 上の RIoU 作業完了後に着手、(c) 現在学習済みモデルの重みを部分的に持ち込む、(d) RGB-D 入力から 3D BBOX を出力、(e) RGB branch/encoder を凍結、(f) depth MAE 自己教師ありを必要に応じて導入、(g) clipboard 記載データで実学習する。
* **暗黙制約:** Python は repo-local uv/.venv を使う。`justfile` の `just train`/`just test` を優先する。既存の user-owned RGB-D/MAE dirty files（`dual_rgbd.py`、`mae_depth.py`、関連 config）を stage/reset/上書きしない。checkpoint のキー一致は実測してから行い、暗黙 fallback や random-init を transfer 成功と呼ばない。GPU VRAM、AMP、NaN/Inf/OOM、データリークを監視する。
* **非ゴール:** 2D OBB RIoU 実験の再実行、clipboard にないデータの推測利用、カメラ内部パラメータや 3D coordinate convention の勝手な変更、RGB encoder 全層の fine-tune、無断 commit/push。
* **成功条件:** clipboard のデータ契約を文書化し、RGB-D 3D BBOX model/dataset/loss/evaluator を config から構築できる。RGB encoder freeze と checkpoint transfer の loaded/missing/unexpected keys を記録する。深度 MAE は実行可能な深度データがある場合のみ事前学習 loss を有限にして checkpoint 化する。3D BBOX training は実 dataset の train/validation を通過して checkpoint と evaluator 数値を残す。予測を RGB に 3D BBOX 投影して目視する。
* **リスクと前提:** clipboard のデータが NOCS/Camera/Real、独自 RGB-D DOTA、または別 schema のどれかは未確認である。3D BBOX target（center/size/rotation、9D pose、corner）は一次情報と既存 evaluator に合わせる。2D OBB detector（HGNetV2 B2）と既存 RGB-D model の backbone が異なる場合、名前一致 transfer は行わず、adapter または明示的な compatible subset のみを採用する。depth MAE は深度無効値/尺度/解像度を検査してから有効化する。

### 1.2 サブゴール構造

| ID | サブゴール | 目的との対応 | 成果物 | 検証方法 |
| :--- | :--- | :--- | :--- | :--- |
| SG-1 | clipboard と既存実装の調査 | 目標1、TR-1/2 | データ契約・現状分析 | file/schema/config の実測記録 |
| SG-2 | transfer/freeze/MAE の設計 | 目標2/3、TR-3/4 | 設計追記・checkpoint mapping | strict な key/shape report |
| SG-3 | RGB-D 3D BBOX 実装と unit smoke | 目標2、TR-5 | config/model/dataset tests | fail→pass、finite loss/gradient |
| SG-4 | depth MAE 事前学習 | 目標3、TR-6 | depth MAE checkpoint/log | masking/reconstruction finite check |
| SG-5 | 3D BBOX 実訓練・評価・可視化 | 目標4、TR-7/8 | work_dir/checkpoint/metrics/overlay | real val/evaluator/artifact |
| SG-6 | 監査・引継ぎ | 目標4、TR-9 | 作業記録・worktree audit | unchecked=0、DoD 確認 |

### 1.3 トレーサビリティ方針

| Trace ID | 要求・制約 | 対応する作業要素 | 証跡 |
| :--- | :--- | :--- | :--- |
| TR-1 | clipboard のデータを一次情報とする | 手順1 | clipboard 抽出、作業記録 |
| TR-2 | 既存 RGB-D/MAE を保護して現状把握 | 手順2 | git status、source/config 分析 |
| TR-3 | 2D 重みの部分 transfer | 手順3、6 | key/shape mapping、load log |
| TR-4 | RGB encoder freeze、depth MAE は条件付き | 手順4、7 | requires_grad report、MAE log |
| TR-5 | RGB-D→3D BBOX 入出力契約 | 手順5、6 | unit test/config build |
| TR-6 | depth MAE は有効深度のみ | 手順7 | mask/loss/checkpoint report |
| TR-7 | 実データ 3D BBOX 学習と evaluator | 手順8、9 | train/val log、checkpoint |
| TR-8 | RGB 上の 3D BBOX 可視化 | 手順10 | images/manifest、目視記録 |
| TR-9 | dirty worktree・無断 push 禁止 | 手順11、DoD | git status、作業記録 |

---

## 2. 作業内容

### フェーズ 1: 一次情報・データ契約の調査

clipboard と現在のデータ/config を読み、入力 RGB/depth、3D target、camera intrinsics、split、評価形式を具体化する。既存 user-owned RGB-D/MAE worktree をこの段階では編集しない。

### フェーズ 2: 設計・実装

互換性を実測して 2D checkpoint の RGB-compatible subset のみを transfer し、RGB encoder を凍結する。3D target schema に合わせた RGB-D model、loss、evaluator、dataset adapter を最小限に実装する。depth データが検証を通った場合だけ MAE pretraining を独立 work_dir で実装・実行する。

### フェーズ 3: 実訓練・検証

先に tiny GPU smoke、次に実データの 3D BBOX 学習・real validation を行う。checkpoint を使用して RGB 投影上の 3D BBOX を出力・目視し、数値と artifact を作業記録へ残す。

---

## 3. 作業チェックリスト

*作業が完了したら `[ ]` を `[x]` に変更し、直後に「作業記録」へ結果を追記する。各項目は上から一つずつ実行する。*

### フェーズ 1: 一次情報・データ契約の調査

### 手順 1: clipboard の 3D BBOX データ契約を確定する
- [x] 🖐 **操作**: `/home/kasm-user/Desktop/clipboard.txt` を全文読み、3D BBOX 対象データ、path、RGB/depth 対応、annotation schema、camera convention、split、既知の会話/作業状態を抽出する。
- [x] 🔎 **確認**: 一次情報に記載された path と repository 内の候補が存在し、推測ではなく抽出値として作業記録に残る。
- [x] 🧪 **テスト**: `clipboard_contract_inventory` として必須項目（data root、RGB/depth、3D label、評価または不明理由）が全て値または明示的な欠落理由を持つことを確認する。
- [x] 🛠 **エラー時対処**: clipboard が欠落/曖昧なら内容を捏造せず、関連 config/workdoc/log を再探索しても契約が確定しない点だけをユーザーへ報告する。

### 手順 2: 既存 RGB-D/MAE とデータ loader を実測する
- [x] 🖐 **操作**: `configs/yopo/nocs_custom_real_hgnetv2_rgbd*.py`、`yopo/models/backbones/dual_rgbd.py`、`yopo/models/detectors/mae_depth.py`、対応 dataset/evaluator/test を読み、実装済み機能と user-owned dirty files を分離して記録する。
- [x] 🔎 **確認**: RGB-D tensor shape、3D target representation、既存 checkpoint/config、depth invalid-value 処理、評価器の責務が source path/line とともに記録される。
- [x] 🧪 **テスト**: `rgbd_existing_config_inventory` として各 config を build-only で parse し、dataset path へのアクセス有無と import error を記録する。
- [x] 🛠 **エラー時対処**: config/import が壊れている場合は user-owned file を上書きせず、最小再現・欠落 module・既存 workdoc を記録して adapter を別ファイルに分離する。non-finite 3D target がある場合も同様に、原因 instance と既存実装を記録して専用 adapter/filter の方針を次の手順2aで検証する。

### 手順 2a: non-finite rotation target を原因別に隔離する
- [x] 🖐 **操作**: non-finite 21 target の frame/instance/raw rotation/`r_norm` を列挙し、`sym_ids` による対称化前後を比較する。
- [x] 🔎 **確認**: NaN が raw annotation、対称化、又は transform のどこで生じるかを数値で特定し、valid target の保持数を記録する。
- [x] 🧪 **テスト**: `test_nocs_rotation_finite_contract` を先に作り、現状21件の nonfinite を再現して失敗させる。
- [x] 🛠 **エラー時対処**: raw rotation 自体が壊れていれば該当 instance を明示除外する専用 adapter を設計する。対称化だけが原因なら `r_norm` epsilon guard を既存 user-owned sourceではなく専用 subclass/adapter で実装し、raw targetを勝手に置換しない。

### 手順 3: 2D OBB checkpoint の転用可能 subset を決める
- [x] 🖐 **操作**: `work_dirs/rddetr_tomato_riou_linear_ft20/best_rbbox_mAP_50_epoch_20.pth` と採用予定 RGB-D model の state_dict key/shape を比較し、RGB backbone/neck で厳密一致する subset と不一致理由を設計記録へ追記する。
- [x] 🔎 **確認**: transfer 対象、除外対象（query/head/depth/fusion/3D head）、loaded/missing/unexpected keys、random init 箇所が明示される。
- [x] 🧪 **テスト**: `partial_transfer_schema` を先に実行し、未知 key/shape mismatch を失敗として検出し、意図した allow-list 後だけ成功することを確認する。
- [x] 🛠 **エラー時対処**: HGNetV2 B2 と採用 RGB encoder が非互換なら名前一致の部分 load をしない。互換 encoder config を新設するか、既存 compatible checkpoint を選ぶ判断を作業書に追記する。

### 手順 4: freeze/optimizer/MAE 実行可否を設計する
- [x] 🖐 **操作**: RGB branch/encoder の freeze 範囲、3D/fusion/depth の trainable 範囲、optimizer parameter groups、depth MAE の開始条件（深度枚数・無効値率・camera/depth scale）を設計として固定する。
- [x] 🔎 **確認**: `requires_grad=False` の target module、trainable parameter count、MAE を skip する客観条件、3D target loss/evaluator が明記される。
- [x] 🧪 **テスト**: `freeze_parameter_contract` として RGB target params に gradient が無く、depth/fusion/3D head には gradient がある unit test を予定する。
- [x] 🛠 **エラー時対処**: 「RGB branch」が model 内で特定できない場合、attribute name を推測せず module tree と config を記録して freeze API を明示的に実装する。

### フェーズ 2: 実装・smoke

### 手順 5: clipboard schema に適合する RGB-D 3D BBOX adapter を実装する
- [x] 🖐 **操作**: 手順1/2で確定した schema に対してのみ、RGB/depth 同期・intrinsics・3D BBOX target を `data_sample` へ渡す dataset/transform/adapter を新規ファイルまたは明示的な専用 config に実装する。
- [x] 🔎 **確認**: 1 sample の RGB/depth shapes、dtype、invalid depth mask、3D target tensor、coordinate convention が assert で確認できる。
- [x] 🧪 **テスト**: `test_rgbd_3dbbox_sample_contract` を先に追加して schema 不一致で失敗させ、adapter 実装後に pass することを `uv run pytest` で確認する。
- [x] 🛠 **エラー時対処**: RGB/depth basename、解像度、intrinsics、label coordinate が不一致なら無言補正をせず、対応表/変換式/除外数を記録してから最小 adapter を追加する。

### 手順 6: transfer と RGB freeze を持つ 3D BBOX model/config を実装する
- [x] 🖐 **操作**: 専用 model/config に allow-list partial load、RGB encoder freeze、depth/fusion/3D BBOX head、loss/evaluator を実装し、run 名・`load_from`・`resume=False` を固定する。
- [x] 🔎 **確認**: config build で RGB-D forward/loss が構築され、load report は手順3の allow-list と一致し、RGB params が optimizer に入らない。
- [x] 🧪 **テスト**: `test_rgbd_3dbbox_transfer_freeze` を fail→pass させ、partial load report、requires_grad、one-batch finite loss/backward を確認する。
- [x] 🛠 **エラー時対処**: transfer 後に missing/unexpected key が allow-list 外なら run を開始せず、key mapping または encoder 選択を修正する。GWD/2D detector を3D modelの偽 checkpointとして使わない。

### 手順 7: 条件を満たす場合のみ depth MAE を事前学習する
- [x] 🖐 **操作**: 手順4の開始条件を通過した場合だけ、RGB frozen の独立 depth MAE config を作り、masked valid-depth reconstruction の tiny GPU smoke→有限 loss checkpoint を実行する。条件不成立なら skip の根拠を記録する。
- [x] 🔎 **確認**: MAE input/mask は invalid depth を reconstruction loss から除外し、RGB branch に gradient が流れず、MAE checkpoint/use-or-skip が明示される。
- [x] 🧪 **テスト**: `test_depth_mae_valid_mask` を fail→pass させ、all-invalid depth で NaN にならず、valid depth の reconstruction loss が有限であることを確認する。
- [x] 🛠 **エラー時対処**: depth scale/invalid value が未確定、又は MAE が finite にならない場合は 3D BBOX 本走に偽の MAE checkpoint を渡さず、MAE を明示 skip して supervised depth branch のみを検証する。

### フェーズ 3: 実訓練・評価・可視化

### 手順 8: RGB-D 3D BBOX の one-batch AMP smoke を通す
- [x] 🖐 **操作**: 専用 work_dir で batch 1 相当の `train_cfg.max_iters`/`max_epochs` override を使い、AMP forward/backward/checkpoint を実行する。
- [x] 🔎 **確認**: GPU memory、RGB freeze、depth/fusion/3D head gradient、finite loss、checkpoint 保存、NaN/Inf/OOM 不在を log に残す。
- [x] 🧪 **テスト**: `rgbd_3dbbox_amp_smoke` として intentional pre-implementation failure から、実 dataset one-batch exit 0 へ移行することを確認する。
- [x] 🛠 **エラー時対処**: OOM は batch/resize を最小化して再現し、RGB freeze/transfer を外さずに原因を記録する。NaN は depth validity・coordinate scale・loss denominator を順に診断する。

### 手順 9: 実データ 3D BBOX training と real evaluator を完走する
- [x] 🖐 **操作**: smoke 成功後、clipboard 確定 split で専用 work_dir に本訓練を一度だけ起動し、定期 evaluator、best weights-only checkpoint、GPU/数値監視を保存する。
- [x] 🔎 **確認**: config の 3D BBOX metric（translation/size/rotation/IoU 等、clipboard schemaに適合）が validation 全件を処理し、best checkpoint と epoch 別値が log に残る。
- [x] 🧪 **テスト**: `rgbd_3dbbox_e2e` として exit 0、finite loss/grad、real validation、best checkpoint 非空を確認する。
- [x] 🛠 **エラー時対処**: non-finite、data leak、evaluator schema mismatch は run を停止し、last safe checkpoint/data sample/error を記録する。2D mAP を3D品質の代替指標にしない。

### 手順 10: RGB 上へ 3D BBOX を投影して目視する
- [x] 🖐 **操作**: best 3D checkpoint を使い、clipboard の intrinsics/coordinate convention に従う RGB projection overlay を3枚と manifest に出力する。
- [x] 🔎 **確認**: RGB、投影した 3D box corners/edges、score、camera convention、input/checkpoint、予測数が artifact/manifest に残る。
- [x] 🧪 **テスト**: `rgbd_3dbbox_overlay_artifact` として PNG 3枚、manifest、画像サイズを列挙し、実際に目視して投影の鏡映・軸・尺度取り違えがないことを確認する。
- [x] 🛠 **エラー時対処**: projection が画面外/鏡映/NaN の場合、intrinsics、depth unit、camera-to-object transform、corner ordering を一つずつ検証し、描画閾値変更で問題を隠さない。

### 手順 11: worktree・再現性・完了定義を監査する
- [x] 🖐 **操作**: `git status --short`、checkpoint/log/overlay、config、test 結果、作業書を監査し、user-owned dirty files と本作業ファイルを分離して記録する。
- [x] 🔎 **確認**: 無断 stage/reset/commit/push が無く、全 Trace ID に証跡、MAE use/skip と 3D evaluator の未対応事項が残る。
- [x] 🧪 **テスト**: `rgbd_3dbbox_workdoc_audit` として実装 checklist と完了定義を確認後、未完了項目が無いことを検証する。
- [x] 🛠 **エラー時対処**: 未完了項目、artifact 欠落、未解決 error、又は clipboard 契約欠落があれば完了にせず、該当手順を四項目へ細分化して理由と再開条件を作業記録へ残す。

---

## 4. 作業に使用するコマンド参考情報

```bash
# 環境と repository の確認
just env-doctor
uv run python -c "import yopo; print(yopo.__version__)"

# config build と既存テスト候補の探索
uv run python -c "from mmengine.config import Config; Config.fromfile('<config>')"
uv run pytest <target-test> -q

# 訓練・評価（手順で確定した専用 config/work_dir のみを使用）
just train <config> <work_dir>
just test <config> <checkpoint>

# 学習監視
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
```

---

## 6. 完了の定義

*全手順を終えた後、以下を `[x]` にし、最終監査で成果物と作業記録を照合する。*

- [x] 観点1: clipboard 契約に適合した RGB-D 入力・3D BBOX target・evaluator を実データで構築し、RGB encoder freeze と checkpoint transfer の実測証跡がある。
- [x] 観点2: depth MAE は valid-depth 条件を満たして finite checkpoint を作るか、成立しない理由を明示して安全に skip している。
- [x] 観点3: real 3D BBOX validation、best checkpoint、RGB projection overlay 3枚、metrics/manifest が存在し、実際に確認している。
- [x] 観点4: uv tests/AMP smoke/worktree audit が成功し、暗黙 fallback・無断 commit/push・user-owned dirty file の変更がない。

---

## 7. 作業記録

**重要な注意事項：**

* 作業開始前に必ず `date "+%Y-%m-%d %H:%M:%S %Z%z"` コマンドで現在時刻を確認し、正確な日時を記録する。
* 各作業項目を開始する際と完了する際の両方で記録を行うこと。
* 作業内容は具体的なコマンドや操作手順を詳細に記載すること。
* 結果・備考欄には成功／失敗、エラー内容、解決方法、重要な気づきを必ず記入すること。
* 複数のフェーズがある場合は、フェーズごとに開始・完了の記録を取ること。
* コード変更を行った場合は、変更したファイル名と変更内容の概要を記録すること。
* エラーが発生した場合は、エラーメッセージと解決策を詳細に記録すること。

| 日付 | 時刻 | 作業者 | 作業内容 | 結果・備考 |
| :--- | :--- | :--- | :--- | :--- |
| 2026-08-24 | 13:04:29 UTC | Codex | 新規 RGB-D 3D BBOX 作業開始・時刻記録 | 直前の RIoU workdoc は全チェック完了。`date`、repo root、uv/justfile、existing RGB-D/MAE candidates、`/home/kasm-user/Desktop/clipboard.txt` の存在を確認。本書を作成し、次の未完了は手順1の clipboard 一次情報読み取り。 |
| 2026-08-24 | 13:06:55 UTC | Codex | 手順1・操作完了: clipboard 一次情報読取 | `/home/kasm-user/Desktop/clipboard.txt`（57 lines, 2,886 bytes）を全文読取。対象は `data/nocs_custom/real` の train 300/val 50、RGB 640×480、uint16-mm depth、NOCS pkl（class_ids/bboxes/translations[m]/rotations[3x3]/sizes[m]/scales/追加2D OBB）。single fruit class。既知の未解決事項は contour_v2 sizes と、YOPO `NOCSDataset.SPLIT_INFO` が実機 K `[443.9,449.2,321.35,230.87]` 未反映のままの点。 |
| 2026-08-24 | 13:07:57 UTC | Codex | 手順1・確認完了: data contract 実体 | split list の実 path は `data/nocs_custom/real/train_list.txt`（300行）/`test_list.txt`（50行）。各 line（`scene_1_train/0000` 等）について color/depth/label の3ファイル全件存在を assert。clipboard の list tree 表示は一段浅いが、データ契約は実体と一致する。 |
| 2026-08-24 | 13:07:57 UTC | Codex | 手順1・テスト完了: clipboard_contract_inventory | data root、train/val split、RGB/depth format、3D BBOX target、未反映の self K を値として inventory。evaluation metric は clipboard に指定が無いことを `EXPLICITLY_UNSPECIFIED_IN_CLIPBOARD` として扱い、推測で補わないことを確認。 |
| 2026-08-24 | 13:07:57 UTC | Codex | 手順1・エラー時対処確認完了 | clipboard の list path は一段浅い表示だったが、`rg --files` で `real/` 配下を特定し実体で解消。評価器名は一次情報に無いため捏造せず、手順2で既存 `NOCSMetric`/custom evaluator を source/config から特定する再開条件を固定。 |
| 2026-08-24 | 13:07:57 UTC | Codex | 手順2・操作完了: existing RGB-D/MAE inventory | `NOCSDataset` は custom intrinsic override、4ch RGB+depth loader、target translation/rotation-6D/size を実装。`NOCSMetric` は3D IoU/poseを評価。user-owned `RGBDDualBackbone` は RGB=B2/depth=B0→256ch fusion、`..._deim_cop_dual.py` は custom K + NOCSMetric + MAE checkpointを統合済み。ただし RGB は `freeze_norm` のみ、MAE `F.l1_loss(pred[mask==0],depth[mask==0])` は invalid depth を除外しない。既存 workdoc は MAE20 epoch成功・中心/z/size decode不具合を記録しており、対象 files を上書きしない。 |
| 2026-08-24 | 13:07:57 UTC | Codex | 手順2・確認完了: config/dataset 実測 | `register_all_modules()` 後に dual/deim-cop-dual/test config と dataset を build。train=300、val=50、RGB/depthと self K、translation[3]/rotation6D/size[3]、`NOCSMetric`、MAE encoder checkpoint（7,529,082 bytes）を確認。plain dual config は val evaluator無し、integrated dual は NOCSMetric。`NOCSDataset` symmetry transform の `r_norm=0` RuntimeWarning を検出したため、全sample finite target scan を次項で必須化。 |
| 2026-08-24 | 13:07:57 UTC | Codex | 手順2・テスト完了: rgbd_existing_config_inventory | `register_all_modules()` を使えば全3 config/dataset は build 可能。全train scan で 25,099 target 中21件が nonfinite rotation、sampled depth zero fraction=0.324946 を検出し assertion は意図どおり失敗。これは import/path の不備ではなく `NOCSDataset` 対称化の `r_norm=0` 由来の data-contract error であり、本訓練前の修正対象として次項へ引継ぐ。 |
| 2026-08-24 | 13:07:57 UTC | Codex | 手順2・エラー時対処完了 | build-only の registry error は `register_all_modules()` で解決。user-owned `NOCSDataset`/dual/MAE source は変更せず、21 nonfinite rotation の原因・filter/subclass設計を手順2aへ追加して訓練開始を保留。 |
| 2026-08-24 | 13:07:57 UTC | Codex | 手順2a・操作完了: nonfinite target 原因列挙 | 21件すべて raw pkl rotation は有限で、`[[0,0,1],[0,-1,0],[1,0,0]]`（符号ゼロ差のみ）を持つ。fruit class_id=0 は `sym_ids` に入るため、既存対称化の `theta_x=R00+R22=0`、`theta_y=R02-R20=0`、`r_norm=0` 除算で NaN になる。例: `scene_1_train/0028` instance86。raw annotation破損ではない。 |
| 2026-08-24 | 13:07:57 UTC | Codex | 手順2a・確認完了: symmetry 契約不一致 | raw annotation は全25,099 targetで finite、対称化後だけ21件が nonfinite。既存 `NOCSDataset.sym_ids=[0,1,3]` は元のNOCSクラス indexで、custom single-class fruit label0 を誤って対称化する。専用 adapter で custom fruit の symmetry を明示し、少なくとも raw finite 25,099 target を保持する設計を採用する。 |
| 2026-08-24 | 13:15:27 UTC | Codex | 定期状況・行動カウント reset | `date '+%Y-%m-%d %H:%M:%S %Z%z'` で時刻を再確認。RGB-D 3D BBOX はまだ訓練を開始せず、次の未完了チェック「`test_nocs_rotation_finite_contract`」を、raw finite 25,099件と旧dataset変換後nonfinite 21件を再現するテストから一項目ずつ進める。 |
| 2026-08-24 | 13:18:37 UTC | Codex | 手順2a・テスト完了: test_nocs_rotation_finite_contract | 新規 `tests/test_nocs_rotation_contract.py` を追加。`uv run pytest -q tests/test_nocs_rotation_contract.py` は 1 passed（8.16s）。raw pkl の rotation 25,099件は全て finite、stock `NOCSDataset` の parse target は21件 nonfinite と assertion で再現した。RuntimeWarning は意図した既知defectであり、次の専用 fruit adapter による解消対象。 |
| 2026-08-24 | 13:20:40 UTC | Codex | 手順2a・エラー時対処完了: custom fruit symmetry adapter | `yopo/datasets/pose_estimation/nocs_custom_fruit_dataset.py` を新設。`NOCSCustomFruitDataset` は fruit METAINFO と `sym_ids=[]` を既定にし、parent初期化中にも stock `[0,1,3]` が parse に漏れないよう virtual parse hook で custom IDs を適用する。raw labelを置換/除外しない。TDDとして module 未作成時の `ModuleNotFoundError` を確認後、`uv run pytest -q tests/test_nocs_rotation_contract.py` が 2 passed（9.94s）、adapter target 25,099件すべて finite を確認。 |
| 2026-08-24 | 13:22:11 UTC | Codex | 手順3・操作完了: 2D→dual RGB transfer schema 比較 | `best_rbbox_mAP_50_epoch_20.pth`（state_dict 560 key）と `DINO9DCenter2DPose/RGBDDualBackbone`（1,131 key）を build 比較。`backbone.*`→`backbone.rgb_backbone.*` の 300 key は全て同shapeで転用可能。sourceのBN `num_batches_tracked` 60 key は targetに非存在のallow-list除外。neck conv は source in-ch `[384,768,1536]` 対 target `[256,256,256]` で4件shape不一致、3D encoder/decoder/head/query等は別taskなので転用しない。 |
| 2026-08-24 | 13:22:32 UTC | Codex | 手順3・確認完了: transfer allow-list | load対象は mapping後の `backbone.rgb_backbone.*` 300 tensorのみ。random init は depth B0、fusion projection、dual後neck、3D encoder/decoder/query/head。2D neck/encoder/decoder/head/query/`level_embed` はshapeが偶然一致するものも3D転用対象にしない。最終load reportは、allow-list外を0件とし、全non-RGB targetを意図したmissingとして分類する loader schema test で確定する。 |
| 2026-08-24 | 13:22:55 UTC | Codex | 定期状況・行動カウント reset | 手順2aのfruit adapter により3D rotation targetは25,099/25,099 finite。手順3は2D RIoU checkpoint→dual RGB B2 の厳密一致300 keyを確定し、次の未完了は未知key/shape mismatchを拒否する `partial_transfer_schema` のTDD。まだ訓練・commit/pushは開始していない。 |
| 2026-08-24 | 13:24:15 UTC | Codex | 手順3・テスト完了: partial_transfer_schema | TDDとして `tests/test_rgb_backbone_transfer.py` を先に追加し utility未作成時の `ModuleNotFoundError` を確認後、`yopo/utils/partial_checkpoint.py` を実装。`uv run pytest -q tests/test_rgb_backbone_transfer.py` は1 passed（5.81s）。実checkpointで300 key選択/BN counter60 key明示除外を確認し、注入した`backbone.unexpected.weight`とstem weight shape mismatchはいずれも `ValueError` で停止する。 |
| 2026-08-24 | 13:24:34 UTC | Codex | 手順3・エラー時対処完了: encoder compatibility gate | HGNetV2-B2 RGB encoder は300/300 strict key-shape一致のため非互換分岐は非発動。`build_rgb_backbone_transfer_state` が今後のunknown/shape/missing RGB keyを例外化するため、名前一致だけの部分loadは不可能。もしcheckpoint又はtarget B2を変更して例外になれば、load/runを開始せず compatible encoder/checkpoint の新比較を手順3から再実行する。 |
| 2026-08-24 | 13:26:50 UTC | Codex | 手順4・操作完了: freeze/optimizer/MAE 設計 | RGB B2 `backbone.rgb_backbone`（60 parameter tensors, 5,999,848 params）をmodel構築時から全freeze+evalにしoptimizerから除外。trainableはdepth B0 1,837,776、RGB/depth projection 688,896/459,520、fuse 393,984、neck/encoder/decoder/head等。DefaultOptimWrapperConstructor は `requires_grad=False` をskipすることをsource確認。raw depth は全300frame非空、有効率58.4–71.2%（median67.8%）、125–4,359mm、Kはclipboard値。既存fill loaderは sampleで−0.135m/zero4,255を生みmaskを失うため不採用。専用loaderはraw uint16-mm→mと`raw>0` maskを保持し、supervisedは4ch、MAEだけ5ch（mask付）にする。MAEはこのcontractでvalid masked L1を使う条件を満たす。NOCSMetricはfruit metainfoで非対称の3D IoU/poseを評価する。 |
| 2026-08-24 | 13:27:14 UTC | Codex | 手順4・確認完了: parameter/MAE/evaluator contract | Freeze対象は `model.backbone.rgb_backbone` の全parameter、実測5,999,848。depth+fusionは3,380,176、neck+encoder+decoder+bbox head+otherは25,522,753をtrainableにする（計28,902,929）。MAEをskipする客観条件は、depth file欠落/空frame、raw valid率median<0.50、mm scale/K未確定、又はvalid-mask lossがnonfiniteのいずれか。今回はいずれも不成立でMAE実行可。3D target lossはbbox/center2d/z/rotation6D/size、real evaluatorはfruit class非対称のNOCSMetric（3D IoU/rotation/translation閾値）に限定し、2D mAPは代用しない。 |
| 2026-08-24 | 13:29:10 UTC | Codex | 手順4・テスト完了: freeze_parameter_contract | `tests/test_rgbd_freeze_contract.py` をTDD追加し、subclass未実装ではModuleNotFound。`FrozenRGBDDualBackbone` 新設後、初回は inherited depth `freeze_at=0` が depth stemをfreezeする assertion failureを検出。専用runでは `depth_backbone.freeze_at=-1` を必須overrideとし、`uv run pytest -q tests/test_rgbd_freeze_contract.py` は1 passed（5.80s）。RGB全param grad=None/eval、depth/fuse/3D headはrequires_gradかつfinite最小lossのgrad有りを確認。 |
| 2026-08-24 | 13:29:28 UTC | Codex | 手順4・エラー時対処完了: explicit RGB freeze API | 実model treeでRGB=`backbone.rgb_backbone`、depth=`backbone.depth_backbone`、fusion=`rgb_proj/depth_proj/fuse`を確認。既存user-owned `dual_rgbd.py`は変更せず、新規 `yopo/models/backbones/frozen_rgbd.py` の `FrozenRGBDDualBackbone` が構築時`rgb_backbone.requires_grad_(False)`、`train()`後`rgb_backbone.eval()`を保証する。unknown attributeの推測は無い。 |
| 2026-08-24 | 13:30:10 UTC | Codex | 定期状況・行動カウント reset | 手順3の2D→RGB B2 strict transfer（300 key）と手順4のRGB-only freeze contract（RGB 5,999,848 param frozen、depth/fusion/3D trainable）を完了。既存depth補完がraw validityを失い負値も生むことを発見したため、手順5では補完なしraw mm→m+maskを専用transform/configとしてTDD実装する。 |
| 2026-08-24 | 13:32:44 UTC | Codex | 手順5・操作完了: raw-depth RGB-D 3D BBOX adapter | 新規 `raw_depth.py` に `LoadRawDepthImageWithValidMask`（uint16-mm→m、inpaint無し、`raw>0` bool mask）と`ConcatRawDepthToImage`を実装。新規 `nocs_custom_fruit_rgbd_3dbbox_transfer.py` は fruit adapter、raw 4ch pipeline、self K、pose-aware horizontal flip、HSVなし、`FrozenRGBDDualBackbone`、depth `freeze_at=-1`、rotation symmetric_classes=[]を明示。既存user-owned config/sourceは変更せず、旧mask無しMAE checkpointは不使用。 |
| 2026-08-24 | 13:33:03 UTC | Codex | 手順5・確認完了: RGB-D/3D target sample assertions | val sampleで raw depth shape=480×640、dtype=float32、`depth_valid_mask == (uint16 PNG > 0)`、depth値は0以上m単位をassert。pack後inputは`(4,480,640)` float32、K=`[443.9066,449.1953,321.3503,230.8687]`。非empty sampleのlabels、translation[-1]=3、rotation[-1]=6、size[-1]=3をassertし、全target finite。`uv run pytest -q tests/test_rgbd_3dbbox_sample_contract.py`=1 passed（6.03s）。 |
| 2026-08-24 | 13:33:36 UTC | Codex | 手順5・テスト完了: test_rgbd_3dbbox_sample_contract | TDD初回は`raw_depth` module未作成で `ModuleNotFoundError`。実装後 `uv run pytest -q tests/test_rgbd_3dbbox_sample_contract.py` は1 passed。testはraw depth mask・mm→m・4ch入力・self K・3D shape/finiteを検証し、既存補完loaderへfallbackしない。 |
| 2026-08-24 | 13:33:56 UTC | Codex | 手順5・エラー時対処完了: explicit raw-depth mismatch handling | 対象sampleでRGB/depthとも480×640、basename対応、K、3D label座標は一致し不一致分岐は非発動。`ConcatRawDepthToImage` はRGB/depth/mask shape不一致を詳細値付き`ValueError`にし、unsupported depth encoding/file欠落も明示例外。無言resize、補完、instance除外は実施していない。再発時は対応表と変換式/除外数を記録して手順5から再開する。 |
| 2026-08-24 | 13:37:10 UTC | Codex | 手順6・操作完了: strict transfer + frozen 3D model | 新規`RGBBackboneTransferHook`を`custom_hooks`へ接続。`before_train`で2D checkpointを読み、strict selectorの300 RGB tensorsだけをloadし、non-RGB missing集合とunexpected=0を照合、`partial_transfer_report.json`をwork_dirへ出力する。modelは`FrozenRGBDDualBackbone`、depth B0/fusion/neck/6-layer 3D encoder-decoder/headをtrainable、fruit非対称loss/evaluator。専用configは`load_from=None`/`resume=False`、旧MAE checkpoint不使用。 |
| 2026-08-24 | 13:37:29 UTC | Codex | 手順6・確認完了: config/load/optimizer contract | `RGBBackboneTransferHook` test reportはloaded=300、target RGB missing=0、unexpected=[]、非RGB missing=831。実config `ScheduleFreeOptimWrapper`/`DefaultOptimWrapperConstructor`はlogでRGB全parameterを`requires_grad=False`としてskipし、optimizer parameter IDにRGBが無いことをassert。pseudo-collateしたreal val sampleでmodel lossをbuildしfinite/backwardできた。 |
| 2026-08-24 | 13:37:47 UTC | Codex | 手順6・テスト完了: test_rgbd_3dbbox_transfer_freeze | TDD初回はhook未作成でModuleNotFound。次に簡略optimizerがfrozen paramsをgroupに残すことを検出し、実runの`paramwise_cfg` optimizerで検証へ変更。data preprocessorは単体listでなく`pseudo_collate`が必要なTypeErrorを修正。最終`uv run pytest -q tests/test_rgbd_3dbbox_transfer_freeze.py`は1 passed（8.82s）で、strict report/RGB optimizer除外/real val one-batch finite loss/backwardを確認。 |
| 2026-08-24 | 13:38:11 UTC | Codex | 手順6・エラー時対処完了: strict pre-run gate | 今回はselectorのunknown/shape/missing RGB key=0、hook unexpected=0で正常通過。sourceからロードするのは `backbone.*`→`backbone.rgb_backbone.*` の300 tensorだけで、2D GWD/RIoU detectorのhead/query/neck/encoderを3D checkpointとして偽装しない。将来allow-list外が出ればhookが例外でrun前に止めるので、key mapping/encoder比較を手順3から修正する。 |
| 2026-08-24 | 13:39:20 UTC | Codex | 定期状況・行動カウント reset | 手順5/6まで完了。GPUは2/20,475MiBで空き、実3D one-batch loss/backwardもfinite。step7の旧`enc_b0_mae.pth`はplain HGNet 252 keyだがvalid-mask無しで学習されたため再利用しない。raw depth 300frame非空、median valid率67.8%を満たすため、5ch（RGB/raw depth/valid mask）入力の新規valid-mask MAE tiny smokeを開始する。 |
| 2026-08-24 | 13:44:56 UTC | Codex | 手順7・操作完了: valid-mask depth MAE smoke | `ValidMaskMAEDepth`、5ch preprocessor、MAE config、encoder extractorを新設。初回AMP runは旧4ch preprocessorが5ch meanを拒否、次はcommon `bgr_to_rgb`引数をsubclassが未受理で停止し、安全に修正。AMP runはexit0/checkpoint finiteだが`grad_norm=inf`のため採用しない。FP32 `work_dirs/rgbd3d_validmask_mae_fp32_smoke` は19iter exit0、loss=2.2293、grad_norm=121.1083、`epoch_1.pth`を保存。converterでfinite B0 encoder 210 keyを`enc_b0_validmask_mae.pth`へ抽出し、3D depth targetと210/210 strict一致後configへ接続。 |
| 2026-08-24 | 13:45:17 UTC | Codex | 手順7・確認完了: MAE valid-mask/use contract | `ValidMaskMAEDepth.loss` はinput ch3 depthだけをencodeし、ch4の`>0` valid maskとrandom masked pixelの積集合だけをL1に入れる。all-invalidはgraph接続finite zero。MAEにRGB branchは存在しないためRGB gradientは不可能。採用品はFP32 `enc_b0_validmask_mae.pth`（210 finite tensor）、不採用品はAMP grad_norm=inf checkpointと旧mask無しcheckpoint。3D config Pretrained init後に210/210 `torch.equal`、全finite、RGB frozenを確認。 |
| 2026-08-24 | 13:45:56 UTC | Codex | 手順7・テスト完了: test_depth_mae_valid_mask | 初回はvalid-mask model未作成でModuleNotFound。`uv run pytest -q tests/test_depth_mae_valid_mask.py`は実装後1 passed（4.70s）。validかつmaskedの1pixelだけのL1=1.0とgradient、invalid/maskedまたはvalid/unmaskedは寄与0、all-invalid loss=0.0 finiteをassert。 |
| 2026-08-24 | 13:46:15 UTC | Codex | 手順7・エラー時対処完了: MAE finite checkpoint gate | depth scale=raw uint16 mm→m、invalid=raw zeroを確定。AMP MAEは`grad_norm=inf`なのでcheckpointがfiniteでも採用拒否し、work_dirに残るが3D configから参照しない。FP32 runはfinite loss/gradと210/210 strict encoder loadを満たすため採用。もしFP32もnonfiniteならMAEを明示skipしdepth branch random/supervisedのみで次手順へ進む条件だった。 |
| 2026-08-24 | 13:48:47 UTC | Codex | 定期状況・行動カウント reset | valid-mask MAE（FP32 finite B0）まで完了。手順8 fp16 AMP smokeはone-batch loss=1416.6322/checkpoint/exit0だが`grad_norm=nan`を検出したため成功扱いにせず、main trainingを保留。GPUは20GB空き。Ada GPU対応のBF16 AMPで同じstrict transfer/freeze/MAE configを再検証し、finite gradが確認できた時だけbatch拡大へ進む。 |
| 2026-08-24 | 13:51:31 UTC | Codex | 手順8・NaN再現と切り分け開始 | fp16 AMPのstatic `loss_scale=1.0` one-iteration run (`work_dirs/rgbd3d_fp16_scale1_amp_smoke`) はexit 0/checkpoint保存/損失1495.3098 finiteだが`grad_norm=nan`。BF16はAda上の`ms_deform_attn_forward_cuda`がBFloat16未実装のためforward前に停止。データ・target・transfer/freezeを変えず、3D multi-branch初期loss約1.5e3に対してAMPのstatic scaleを0.001へ下げる再現実験を次に行う。非finiteのcheckpointは本走へ使用しない。 |
| 2026-08-24 | 13:59:36 UTC | Codex | 手順8・操作完了: 実データfp16 AMP one-batch | `configs/yopo/nocs_custom_fruit_rgbd_3dbbox_amp_smoke.py` を `work_dirs/rgbd3d_fp16_stable_amp_smoke` で1iter実行。`AMPStableRotation3DLoss`はfp32 geodesic、safe acos（|cos|=0.9999で線形外挿）、safe 6D normalize（eps=0.01）でfp16逆伝播の特異点だけを有界化。exit 0、loss=1398.8627、grad_norm=36016.9219、`epoch_1.pth`=134MiBを保存。checkpoint state 1,131 tensorは全finite。 |
| 2026-08-24 | 14:00:07 UTC | Codex | 手順8・確認完了: transfer/freeze/gradient/GPU | `partial_transfer_report.json`はloaded=300、missing target RGB=0、unexpected=0、意図したnon-RGB missing=831。実batch再測定でRGB 60 parameter tensorはrequires_grad=0/grad=0、depth 42/`rgb_proj` 6/head 332 tensorには全てfinite gradient。smoke logはfinite loss/grad_normとcheckpoint保存のみでNaN/Inf/OOM/traceback無し。終了後GPU使用量2/20,475MiB（訓練時log表示885MiB）。 |
| 2026-08-24 | 14:01:31 UTC | Codex | 手順8・テスト完了: rgbd_3dbbox_amp_smoke | TDD初期は`AMPStableRotation3DLoss`未実装で`ModuleNotFoundError`、次にzero 6D predictionで`[-inf,0,...,-inf]`を再現。safe acosだけでは実batch415 tensorがnonfiniteだったため、safe 6D normalizeも追加。最終`uv run pytest -q tests/test_amp_rotation_loss.py tests/test_rgbd_3dbbox_transfer_freeze.py tests/test_rgbd_3dbbox_sample_contract.py`は4 passed（8.46s）。実run `rgbd3d_fp16_stable_amp_smoke` もexit 0のone-batch checkpointへ移行。 |
| 2026-08-24 | 14:01:31 UTC | Codex | 定期状況・行動カウント reset | 手順8は操作/確認/テストまで完了。fp16 AMPはfloat32で安定化したrotation geometryだけを局所使用し、実batch loss/grad/checkpointはfinite。次の未完了はNaN/Inf/OOMエラー時対処の記録で、原因（fp16 6D rotationのacos/normalization特異点）と再発時の診断順を固定後、手順9の本訓練バッチサイズをVRAM実測で決める。 |
| 2026-08-24 | 14:01:31 UTC | Codex | 手順8・エラー時対処完了: fp16 rotation特異点 | OOMは未再現（1batch log memory=885MiB）で、RGB freeze/300-key transferは保った。非finiteはraw depth validity、m単位coordinate、3D targetがすべてfiniteの後もrotation群だけで発生することをloss群別backwardで特定。`AMPStableRotation3DLoss`はunsafe acosとzero 6D normalizationをfloat32 + 有界gradientへ局所置換し、`acos_clamp`/`normalize_eps`/rotation幅/対称指定を明示検証する。再発時は(1)raw depth mask/target finite、(2)coordinate scale、(3)loss group別grad、(4)rotation eps/clamp、(5)batch/resize OOMの順で診断し、非finite checkpointは本走に使用しない。 |
| 2026-08-24 | 14:04:01 UTC | Codex | 手順9・本走batch容量preflight | fp16 one-iterのbatch24 (`rgbd3d_fp16_batch24_capacity_probe`) はexit0、log memory=16,984MiB/20,475MiB、loss=1506.7811、grad_norm=33379.2617 finite。batch28はdecoder cross-attentionで追加88MiB確保時にOOM（CUDA実測19.53/19.55GiB使用、free14.38MiB）となった。RGB freeze/transferと入力サイズは維持し、batch27を最大安全候補として次に検証する。 |
| 2026-08-24 | 14:06:59 UTC | Codex | 手順9・本走batch27停止と再分割 | batch27 one-iter probeは19,279MiBで成功したが、20epoch本走の第1iter後に第2iterでdecoder FFNの追加40MiB確保時OOM（19.51/19.55GiB、free26.38MiB）。第1iter自体はloss=1437.5588/grad_norm=36242.1094 finiteで、データ/数値ではなく反復間の最大activation/allocator余裕不足。開始runはcheckpoint未保存で採用しない。batch26を4iter、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`で再検証し、そこでもOOMならbatch25へ下げる。 |
| 2026-08-24 | 14:07:59 UTC | Codex | 手順9・batch26連続capacity成功 | `rgbd3d_fp16_batch26_4iter_capacity` を`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`で4iter実行しexit0。memoryは18,431→19,034→18,853→19,326MiB（peak=94.4%）、loss=1422.22→962.19、grad_norm=36,504→14,870で全finite、NaN/Inf/OOM無し。batch26を20epoch configへ固定し、batch27/28と中断runは本訓練成果に採用しない。 |
| 2026-08-24 | 14:10:24 UTC | Codex | 手順9・real evaluator 初回失敗と修正 | `rgbd3d_amp26_20ep_eval` はepoch 1の12/12 updateをfinite loss/gradで完了（最終loss=748.5353、peak memory=19,326MiB）したが、初回validationの`DINO9DCenter2DPoseHead.predict`で `cls_score.view(-1).topk(max_per_img)` が `RuntimeError: selected index k out of range`。モデルは1 fruit class×100 queries=100 scoreに対しbase config由来`max_per_img=300`だった。confidence thresholdやNOCSMetricの数値ではなく、評価前のquery-cap不整合。TDD `tests/test_rgbd_3dbbox_predict_query_cap.py` は修正前`300 <= 100`で失敗を再現、main configに`model.test_cfg.max_per_img=100`を明示後1 passed。partial runはcheckpoint未保存なので成果扱いにせず、新work_dirで20epochを再走する。 |
| 2026-08-24 | 14:14:06 UTC | Codex | 定期状況・手順9 evaluator 実測 | 修正版`rgbd3d_amp26_20ep_eval_q100`はRGB transfer=300、RGB freeze、batch26 fp16 AMPでepoch 1を完走。train lossは1421.4634→749.3517、grad_normは全finite、train peak memory=19,326MiB（94.4%）。50 real val imageのNOCSMetricが初めて完走し、`AP50=0.0000`、`3d_iou_0.10/0.25/0.50/0.75=0.4695/0.3297/0.1053/0.0159`、best `best_3d_iou_0.50_epoch_1.pth`を保存。これはconfidence thresholdで落ちた結果でなく全100 queryを評価した初期epoch値であり、低値の改善可否は20epoch時点で判定する。 |
| 2026-08-24 | 14:24:04 UTC | Codex | 手順9・本走中断と数値例外の証跡 | `rgbd3d_amp26_20ep_eval_q100`はepoch 1–12のreal validationを完走し、bestはepoch 1の`3d_iou_0.50=0.1053`。epoch 13の1/12（loss=167.5927、grad_norm=242.2864、memory=18,657MiB）後、ptyの標準出力にHungarian assignment (`linear_sum_assignment(cost)`) の `ValueError: matrix contains invalid numeric entries` が出てprocess exit 1。run file logにはtraceback本文がflushされずepoch13 1/12で止まるため、session stderr/exit codeを一次証跡とする。これはquery-cap/evaluator errorではなく訓練時costの非有限化。best checkpointを保持し、checkpoint未保存の後半重みは採用せず、再走前にraw target・transform target・prediction/cost finiteをbatch単位で再現して原因を特定する。 |
| 2026-08-24 | 14:29:42 UTC | Codex | 手順9・数値例外の診断 gate/TDD | `tests/test_rgbd_3dbbox_train_transform_finite.py`でRandomFlipを含む全300 train sampleのbbox/center2d/z/rotation/sizeが有限（1 passed）を再確認し、target/transform起因を除外。HungarianAssignerの従来動作は非有限costをSciPyの曖昧なValueErrorへ渡すことをTDDで再現（fail）後、cost名・shape/finiteness/range・pred/gt tensor fieldsを含む`FloatingPointError`で停止するgateを追加。`uv run pytest -q tests/test_hungarian_assigner_finite_cost.py tests/test_rgbd_3dbbox_train_transform_finite.py`=2 passed。解決後main configの実効optimizerは`AdamWScheduleFreeOptimizer(lr=0.0025, weight_decay=0.0001)`、clip max_norm=0.1であり、DETR系3D transferとして過大なLRが有力な非有限候補。 |
| 2026-08-24 | 14:29:42 UTC | Codex | 定期状況・行動カウント reset | 手順9の本走はquery-cap修正後、epoch12まで実evaluatorを完走したがepoch13 update後にcost非有限で終了。data/transform全300件はfiniteで、assignmentにcost別fail-fast gateを追加済み。実効LR=0.0025が高いため、次は同じAMP/batch26/freeze/transferで安定LRを短期再現し、finiteかつ評価可能と確認後に20epoch本走を再開する。 |
| 2026-08-24 | 14:44:05 UTC | Codex | 定期状況・行動カウント reset | `date '+%Y-%m-%d %H:%M:%S %Z%z'` を再確認。安定LR（2e-4）の `rgbd3d_amp26_lr2e4_20ep_eval` はepoch14のreal evaluatorまで完走し、全train loss/grad_normは有限、Hungarian finite-cost gateの例外・NaN・Inf・OOMなし。batch26のmemory peakは19,326/20,475MiB（94.4%）。epoch14の AP50=0.0000、3d_iou_0.10/0.25/0.50/0.75=0.6485/0.4405/0.1690/0.0211で、IoU@0.50はepoch9の0.1673を上回る暫定最高。AP50はconfidence表示閾値ではなく全100 queryの順位評価で依然低く、20epoch最終評価後に品質判定する。 |
| 2026-08-24 | 14:49:04 UTC | Codex | 手順9・操作完了: 安定LR 20epoch本走 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python tools/train.py configs/yopo/nocs_custom_fruit_rgbd_3dbbox_amp_20ep.py --work-dir work_dirs/rgbd3d_amp26_lr2e4_20ep_eval` を一度だけ起動し exit 0。clipboard確定 split（train300/val50）、RGB 300-key strict transfer + frozen RGB、valid-mask MAE depth encoder、fp16 AMP/batch26でepoch1–20と各epochのreal evaluatorを完走。train peak memory=19,326/20,475MiB（94.4%）、loss/grad_normは全有限、NaN/Inf/OOM/Hungarian finite-cost gate例外なし。best weights-only は `best_3d_iou_0.50_epoch_14.pth`（140,803,743 bytes、IoU@0.50=0.1690）、`partial_transfer_report.json`も存在。最終epoch20はAP50=0.0010、IoU@0.10/0.25/0.50/0.75=0.4574/0.3045/0.1143/0.0174。 |
| 2026-08-24 | 14:50:17 UTC | Codex | 定期状況・行動カウント reset | `date '+%Y-%m-%d %H:%M:%S %Z%z'` を再確認。手順9の操作チェックを完了済み。main configの`CheckpointHook`は`save_best=3d_iou_0.50`・weights-only・`save_last=False`、`max_per_img=100`を明示。次の一項目「確認」でscalarsの20 evaluation行、50 val全件、translation/size/rotation/IoU metric、best checkpointの対応を読み取り専用で照合する。 |
| 2026-08-24 | 14:50:17 UTC | Codex | 手順9・確認完了: real evaluator と best 監査 | `scalars.json`から AP50/3d_iou の validation 行を20件抽出（step=epoch1–20）し、各50 val imageを全処理したNOCSMetricのIoUとpose（rotation/translation閾値）を確認。epoch14がIoU@0.50=0.169004で最高、対応するbest fileは140,803,743 bytesで存在する。最終epoch20はAP50=0.001、IoU@0.10/0.25/0.50/0.75=0.4574/0.3045/0.1143/0.0174。configは`max_per_img=100`で100query×1class上限に一致し、CheckpointHookは`save_best=3d_iou_0.50`/`save_optimizer=False`/`save_last=False`。これにより2D描画閾値・mAPを3D qualityの代理値にはしていない。 |
| 2026-08-24 | 14:50:17 UTC | Codex | 手順9・テスト完了: rgbd_3dbbox_e2e | 新規`tests/test_rgbd_3dbbox_e2e.py`を追加し、`uv run pytest -q tests/test_rgbd_3dbbox_e2e.py`は1 passed（0.03s）。run scalarsについて20×12=240 train recordのloss/grad_norm全有限、epoch1–20の20 validation record、AP50と4種3D IoU全有限をassert。transfer reportのloaded=300/unexpected=[]、epoch14 best checkpointの非空も検証した。 |
| 2026-08-24 | 14:50:17 UTC | Codex | 手順9・エラー時対処完了: numerical/evaluator regression | 高LR=0.0025 runはepoch13でnonfinite Hungarian costを検出してexit1、best epoch1を保持し後半重みを不採用とした。全300 train target finite scan、assignerの診断gate、100-query cap、保守LR=2e-4を実装・適用して新runをexit0へ移行。`uv run pytest -q tests/test_hungarian_assigner_finite_cost.py tests/test_rgbd_3dbbox_train_transform_finite.py tests/test_rgbd_3dbbox_predict_query_cap.py tests/test_rgbd_3dbbox_stable_optimizer_config.py tests/test_rgbd_3dbbox_e2e.py`は5 passed（14.25s、非阻害DeprecationWarning 1件）。AP50は全query順位評価であり、3D quality はNOCS 3D IoU/poseで判断することを継続する。 |
| 2026-08-24 | 14:53:55 UTC | Codex | 定期状況・手順10 dump 経路の細分化 | `tools/test.py ... --out epoch14_predictions.pkl` を試行したが、本configは`val_dataloader`/`val_evaluator`のみで`test_dataloader`が無く、`runner.test_evaluator`取得時に`AttributeError: 'NoneType' object has no attribute 'evaluator'`でdump前に安全停止。checkpoint・config・datasetは不変、予測artifactは未作成。test loopを偽装せず、既存val loopに`DumpDetResults`を追加してepoch14 checkpointからprediction dumpを作る専用toolへ操作を細分化する。 |
| 2026-08-24 | 14:55:37 UTC | Codex | 手順10・操作完了: best 3D box RGB projection | 新規`dump_rgbd_3dbbox_predictions.py`でtest-only configのtest-loop不在を避け、val loop + `DumpDetResults`によりbest epoch14から50件`epoch14_predictions.pkl`（1,193,957 bytes）をexit0で生成。再評価はIoU@0.50=0.1690でbest値と一致。新規`render_rgbd_3dbbox_overlays.py`は各imageのscore最大queryを1件（confidence thresholdなし）選び、`T` object→camera[m]、size[m]、clipboard Kで8 cornerを投影。`epoch14_overlays/`へPNG3枚（698–711KB）とmanifestを保存。目視で全boxは有限・画面内、鏡映/軸反転なし。boxが小さいのは現モデルの低い検出品質をそのまま示す。 |
| 2026-08-24 | 14:55:37 UTC | Codex | 手順10・確認完了: artifact/manifest schema | manifestをJSON読み取りし、camera convention=`T maps object-frame metres to camera-frame metres; project with K and z>0`、selection=`top-1 query per image, no confidence threshold`、images=3を確認。各PNGはquery=0、score=0.545585/0.539836/0.536968、8 projected corners、visible corners=8、画像=640×480。config/checkpoint/dumpの絶対path、3×3 K、4×4 T、size[m]、camera-space corner[m]、2D corner[px]を各entryに保存済み。 |
| 2026-08-24 | 14:58:53 UTC | Codex | 定期状況・行動カウント reset | `date '+%Y-%m-%d %H:%M:%S %Z%z'` を再確認。手順10はbest epoch14のreal validation再推論、thresholdなしtop-1投影、PNG3枚とmanifest、目視、schema確認まで完了。次の未完了は`rgbd_3dbbox_overlay_artifact`で、PNG数・サイズ・yellow edge実在・manifestのK/T/corner/scoreを自動検証する。 |
| 2026-08-24 | 14:58:53 UTC | Codex | 手順10・テスト完了: rgbd_3dbbox_overlay_artifact | 新規`tests/test_rgbd_3dbbox_overlay_artifact.py`を追加。`uv run pytest -q tests/test_rgbd_3dbbox_overlay_artifact.py`は1 passed（0.29s）。3 overlay PNGをdecodeし、manifest記載の各640×480と一致、yellow投影edgeの実pixel、epoch14 checkpoint名、thresholdなしtop-1規則、score finite、K=3×3/T=4×4/size=3/corners=8×2、visible corners=8をassert。3画像は実際に目視し、投影の鏡映・軸・尺度取り違えは見つからなかった。 |
| 2026-08-24 | 14:58:53 UTC | Codex | 手順10・エラー時対処完了: projection guard | 新規`tests/test_rgbd_3dbbox_projection_guard.py`を追加し、`uv run pytest -q tests/test_rgbd_3dbbox_projection_guard.py`は1 passed（0.38s）。`render_rgbd_3dbbox_overlays.py`はKが4値/3×3以外、camera depth z≤1e-6、非有限camera/projected corner、サイズ/T形状不一致を例外にする。今回の3 artifactはK=clipboard値、T=object→camera[m]、z>0、8corner全画面内でguard非発動。score thresholdは選択・描画に使っていない。画面外/鏡映時はK→m単位→T→corner orderingの順で再検証する。 |
| 2026-08-24 | 14:58:53 UTC | Codex | 手順11・操作完了: worktree/artifact 分離監査 | `git status --short`、tracked diff、work_dir file list、関連test listを読み取り専用で監査。本作業はtracked `hungarian_assigner.py` finite-cost gate、未追跡の3D config/dataset/rawdepth/freeze/MAE/transfer/loss/utils/テスト/overlay tools/workdoc。成果物は20epoch log=1,041,331 bytes、best epoch14 checkpoint=140,803,743 bytes、transfer report=4,411 bytes、dump=1,193,957 bytes、PNG3枚、manifest=9,713 bytes。開始時からのuser-owned dirtyは`backbones/__init__.py`、`detectors/__init__.py`、`rotated_iou_loss.py`、既存dual_rgbd/mae_depth、既存dual/RIoU config/workdocで、stage/reset/commit/pushは一切していない。 |
| 2026-08-24 | 14:58:53 UTC | Codex | 手順11・確認完了: authority/trace audit | `git diff --cached --name-only`は空、HEAD=`f03ccc59...`（11:42の既存 rotated OBB work-record commit）で本作業中のstage/reset/commit/push無し。transfer reportはloaded=300/missing target RGB=[]/unexpected=[]。workdocにMAEのFP32採用・AMP不採用根拠、20epoch NOCS evaluator、低いAP50を閾値問題と誤認しない事項、overlay path/K/T/score/選択規則、error再開条件を記録。未対応事項はモデル品質（AP50最大0.002、IoU@0.50最高0.169）であり、完了判定を高精度主張に拡張していない。 |
| 2026-08-24 | 15:04:02 UTC | Codex | 定期状況・行動カウント reset | 作業書全文と開始以降のworktree/artifactを照合済み。新規の`test_rgbd_3dbbox_workdoc_audit.py`は手順1–10に未完了印がないこと、strict RGB transfer=300/missing=0/unexpected=0、epoch14 best checkpoint、prediction dump、thresholdなしoverlay manifest 3件を自動確認する。次に関連テスト一式を実行し、手順11のテスト欄を確定する。 |
| 2026-08-24 | 15:04:02 UTC | Codex | 手順11・テスト完了: rgbd_3dbbox_workdoc_audit | 初回の関連suiteは16 passed/監査1 failedで、説明文中のリテラル`[ ]`を誤検出したことを再現。task行だけを対象に修正後、`uv run pytest -q tests/test_rgbd_3dbbox_workdoc_audit.py`=1 passed。手順1–10に未完了taskがないこと、transfer report=`loaded_key_count=300`/missing RGB=[]/unexpected=[]、epoch14 best checkpoint>100MB、prediction dump>1MB、thresholdなしtop-1 overlay manifest=3件を実証した。完了定義の四項目は手順11のerror欄を確定後に個別照合し、全印完了後にstrict再検査する。 |
| 2026-08-24 | 15:04:02 UTC | Codex | 手順11・エラー時対処完了: unresolved-error gate | `git diff --check`は成功。作業書の未完了taskは本項目と完了定義4項目だけで、final artifactsは監査testで存在・schemaを確認済み。過去のtest-loop不在、100-query上限超過、高LR=0.0025のnonfinite Hungarian cost、overlay guardは各手順で安全停止・原因修正・再検証済みで未解決errorではない。AP50最大0.002/IoU@0.50最高0.169はconfidence thresholdでもartifact障害でもなく、production品質に未達という性能リスクとして完了照合に明示する。stage/reset/commit/pushは行わず、開始時dirtyと本作業untrackedを分離した。 |
| 2026-08-24 | 15:04:02 UTC | Codex | 完了定義・観点1達成 | `nocs_custom_fruit_rgbd_3dbbox_transfer.py`でraw uint16-mm→m+valid maskの4ch入力、K、3D target、fruit非対称rotation、`FrozenRGBDDualBackbone`、`RGBBackboneTransferHook`を確認。main configはNOCSMetric real validationと100query capを継承。`uv run pytest -q tests/test_rgbd_3dbbox_sample_contract.py tests/test_rgbd_3dbbox_transfer_freeze.py`=2 passedで実data sample/finite 3D loss/backward、RGB freeze、strict reportを再確認した。reportは300 loaded/target RGB missingなし/unexpectedなし。 |
| 2026-08-24 | 15:04:02 UTC | Codex | 完了定義・観点2達成 | `uv run pytest -q tests/test_depth_mae_valid_mask.py`=1 passed。FP32 valid-mask MAE抽出品`work_dirs/rgbd3d_validmask_mae_fp32_smoke/enc_b0_validmask_mae.pth`は210 tensor全finiteで、transfer configのdepth B0 init checkpointとして接続済み。invalid depth除外/all-invalid finite zeroのcontractをtestし、AMP MAEの`grad_norm=inf` checkpointは明示不採用のままにした。 |
| 2026-08-24 | 15:04:02 UTC | Codex | 完了定義・観点3達成 | `uv run pytest -q tests/test_rgbd_3dbbox_e2e.py tests/test_rgbd_3dbbox_overlay_artifact.py tests/test_rgbd_3dbbox_projection_guard.py`=3 passed。E2Eはepoch1–20全validation/finite recordsとepoch14 bestを検証、overlay testは640×480 PNG3枚・yellow edge・K/T/cornersを検証した。manifestはbest epoch14 checkpoint、`top-1 query per image, no confidence threshold`、score=0.545585/0.539836/0.536968を記録し、既に3枚を目視確認済み。 |
| 2026-08-24 | 15:04:02 UTC | Codex | 完了定義・観点4達成 | 17関連testを再実行して全pass（既知warning 22件のみ）。`git diff --check`成功、`git diff --cached --quiet`成功、HEADは作業前の`f03ccc59`で、本作業中のstage/reset/commit/pushは0件。AMP実訓練はbatch26 peak=19,326/20,475MiB（94.4%）、NaN/Inf/OOMなし。既存user-ownedの`backbones/__init__.py`、`detectors/__init__.py`、`rotated_iou_loss.py`、dual RGB-D/旧config/workdocは保持し、本作業のnew filesと`hungarian_assigner.py`診断gateを混在させず記録した。 |
| 2026-08-24 | 15:08:29 UTC | Codex | 最終完了照合 | `rgbd_3dbbox_workdoc_audit`を、手順11・完了定義を含む全task行の未完了ゼロを確認するstrict条件へ更新し、`uv run pytest -q tests/test_rgbd_3dbbox_workdoc_audit.py`=1 passed。`rg -n -- '^- \\[ \\]' workdoc`の出力も空。全チェックリストと完了定義を満たすが、3D AP50最大0.002/IoU@0.50最高0.169で精度はproduction基準に達しておらず、成果を高精度と誤認しない。 |

## 8. 完了照合・調査分析サマリ

### 8.1 照合対象と現状

- 対象作業書: `temp/workdoc_Aug24-2026_rgbd_3dbbox_transfer.md`
- 照合時点: 2026-08-24 15:08:29 UTC
- 照合対象: RGB-D入力/3D BBOX adapter、RGB B2の部分転送とfreeze、valid-mask depth MAE、AMP実訓練、NOCS real evaluator、overlay、関連test、worktree権限。
- 現状: 実装・20epoch学習・可視化・再現性の証跡は揃っている。品質値は低く、研究用の機能完了であってproduction品質の検出器完了ではない。

### 8.2 完了定義の照合

| 完了定義 | 状態 | 根拠 |
|---|---|---|
| 観点1: RGB-D/3D target/evaluator、freeze、transfer | 達成 | raw depth+maskの4ch adapter、`FrozenRGBDDualBackbone`、strict report（loaded=300、target RGB missing=0、unexpected=0）、real-sample test。 |
| 観点2: valid-depth MAE の安全なuse/skip | 達成 | FP32 valid-mask MAE checkpointは210 tensor全finiteでdepth encoderへ接続。AMP版のgrad inf checkpointは不採用。 |
| 観点3: real validation/best/overlay | 達成 | 50 val image×20epochのNOCSMetric、epoch14 best weights、prediction dump、RGB overlay 3枚とmanifestを確認。 |
| 観点4: test/AMP/worktree audit | 達成 | 関連17 test pass、batch26 peak 19,326/20,475MiB、stage空、無断reset/commit/pushなし。 |

### 8.3 主な証跡

- `work_dirs/rgbd3d_amp26_lr2e4_20ep_eval/20260824_143138/20260824_143138.log`: epoch 1–20のtrain/evaluator、NaN/Inf/OOMなし。
- `work_dirs/rgbd3d_amp26_lr2e4_20ep_eval/best_3d_iou_0.50_epoch_14.pth`: best 3D IoU@0.50=0.169004、140,803,743 bytes。
- `work_dirs/rgbd3d_amp26_lr2e4_20ep_eval/partial_transfer_report.json`: RGB B2の厳密300 key転送。
- `work_dirs/rgbd3d_amp26_lr2e4_20ep_eval/epoch14_predictions.pkl` と `epoch14_overlays/manifest.json`: top-1/no confidence threshold、K/T/cornersを含む3投影。
- `uv run pytest -q` の関連17 test: 17 passed（既知の非阻害warning 22件）。

### 8.4 差分・不整合の分析

- **一部達成（品質）**: 3D AP50は最高0.002、3D IoU@0.50は最高0.169004（epoch14）。最終epoch20はAP50=0.001、IoU@0.50=0.1143である。
- これは描画用confidence thresholdではない。評価は1 class×100 queryの全候補を`max_per_img=100`で順位評価し、overlayもthresholdなしtop-1を描画している。
- 高LR=0.0025の前runはepoch13でHungarian cost非有限となり不採用。LR=2e-4とfinite-cost gateで20epoch完走へ改善したが、精度自体の改善目標は作業書に定義されていなかった。

### 8.5 原因・判断根拠

- 100 queryに対しreal valのGTは多く、初期のfrozen RGB +新規depth/fusion/3D headを20epochだけ学習した条件では、3D pose/size/translationの整合がまだ弱いと推定される。
- 低APを閾値・evaluator破綻と扱わない根拠は、query-cap修正後の20回real validation、finite metrics、best checkpoint、投影artifact、e2e testである。
- そのため品質改善には長期最適化、loss/assignmentの再調整、データ/annotation監査、必要なら段階的unfreezeを別タスクとして実施すべきである。現taskでは精度を偽装する閾値変更やsilent fallbackを行っていない。

### 8.6 残課題と次の作業

- **未達**: production用途に耐える3D検出精度。受入閾値は未定義だが、現AP50/IoU@0.50をそのまま採用できる値ではない。
- 次タスク候補: (1) label座標・size/pose分布の定量監査、(2) best epoch14から低LRでより長く学習、(3) depth/fusion loss・query assignmentの診断、(4) RGB最終stageだけの段階的unfreezeを比較する。各候補は新しい受入metricを先に定めてから実施する。

### 8.7 最終結論

**条件付き完了。** 作業書の全チェックリストと完了定義は証跡付きで達成し、RGB-D→3D BBOXの実行経路、AMPによる約94.4% VRAM利用、実evaluator、best重み、投影画像、テストは再現可能である。一方で数値品質は未達のため、モデルを高精度・production-readyとは報告しない。
