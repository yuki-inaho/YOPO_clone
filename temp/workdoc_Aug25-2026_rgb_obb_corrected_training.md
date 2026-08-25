# 作業計画書 兼 記録書: 補正済みDOTAによるRGB-only 2D OBB追加学習

---

**日付：** 2026年08月25日

**作業ディレクトリ・リポジトリ:** `/home/kasm-user/Desktop/YOPO_clone`（Git repository、branch `rgb-d`、uv）

**重い成果物の保存先:** `/workspace/YOPO_clone/work_dirs`（repository内`work_dirs` symlink経由）

**作業者：** Codex

---

## 1. 作業目的

補正済みtomato OBBデータを用い、YOPOのRGB branchと同型のHGNetV2-B2を含む
RGB-only 2D OBB detectorを追加学習する。loaderを特定データ専用に固定せず、class、
画像拡張子、画像shape、strict validationをconfigから変更できる形へ整備する。

### 1.1 ゴール要求分析

* **ユーザーの直観的・直截的な目的:** 3D BBOXへ進む前にRGB encoder/decoderと2D
  OBBの質を上げ、別のRGB/OBBデータにも再利用できる学習基盤を作る。
* **明示要求:** RGB branch＋2D OBBだけを追加学習する。設定とcurriculumを先に整理
  する。loader等を再利用性・柔軟性のある形にする。loss curriculumは
  Rotated IoU→GWDとする。32 GiB GPUを使い高速にPDCAする。作業記録を残す。
* **暗黙制約:** uv環境、TDD、既存RGB-D/3D headを混ぜない、重いdataset/checkpointを
  Gitへ入れない、補正済みDOTAを正本にする、最大129 GTをquery不足で欠落させない、
  暗黙fallbackを行わない、検証bestを選んで最終epochを盲目的に採用しない。
* **非ゴール:** depth mapのないデータをRGB-Dとして偽装する、3D translation/size/
  rotationやCoP/parallel headをこのrunで更新する、test-as-validationを独立holdout性能と
  表現する、低confidence表示だけでモデル品質を主張する。
* **成功条件:** (1) 汎用`DOTAOBBDataset`の正常・異常系testが通る、(2) 旧
  `DOTATomatoDataset`が互換で残る、(3) 150 query・RGB-only・strict loaderのconfigが
  解決できる、(4) batch 32でfinite/OOMなし、(5) Rotated IoU→GWDを完走、(6) mAP50が
  baselineを上回る、(7) checkpoint/prediction/PNG/hash/再実行手順が残る。
* **リスクと前提:** test splitはteacher predictionを人手修正したもので完全独立では
  ない。150 queryの全候補をscore 0.05で描くと可視化は混雑する。境界物体のquadは
  最大約9 px画面外だが正面積のため、事前clipで角度を壊さない。

### 1.2 サブゴール構造

| ID | サブゴール | 目的との対応 | 成果物 | 検証方法 |
| :--- | :--- | :--- | :--- | :--- |
| SG-1 | データと既存checkpointを監査 | 正本・query容量の確定 | dataset統計、query expansion | 画像/label/row/shape集計、hash |
| SG-2 | 汎用DOTA OBB loaderをTDD実装 | 再利用性・柔軟性 | `DOTAOBBDataset`、互換wrapper | `tests/test_dota_obb_dataset.py` |
| SG-3 | RGB-only curriculumを設定化 | loss順とscopeの固定 | Phase 1/2/smoke/inference config、docs | Config contract test |
| SG-4 | GPUでfull training・評価 | 2D OBB品質向上 | best checkpoint、prediction、metric | baseline/Phase 1/2比較 |
| SG-5 | 証跡と引継ぎを固定 | 監査性・再現性 | workdoc、manifest、SHA-256 | file/hash/command確認 |

### 1.3 トレーサビリティ方針

| Trace ID | 要求・制約 | 対応する作業要素 | 証跡 |
| :--- | :--- | :--- | :--- |
| TR-1 | loaderの再利用性 | SG-2、手順3–5 | loader source、10 tests |
| TR-2 | RGB＋2D OBB限定 | SG-3、手順6 | resolved config、`depth`非包含test |
| TR-3 | Rotated IoU→GWD | SG-3/4、手順7–9 | docs、2 config、train logs |
| TR-4 | 32 GiBで高速PDCA | SG-4、手順7 | batch 32、peak 23,297 MiB |
| TR-5 | 品質改善とbest選択 | SG-4、手順8–10 | mAP50 0.0732→0.0960 |
| TR-6 | 作業記録・再現性 | SG-5、手順11–12 | 本書、hash、commit/push記録 |

---

## 2. 作業内容

### フェーズ1: 調査・設計

補正済みDOTA archiveを`/workspace/YOPO_clone/datasets`へ安全に展開し、1,368 train
images/94,605 boxes、181 test images/9,034 boxes、800x600、最大129/118 boxes per image
を確認した。既存100-query checkpointは構造的recall上限を持つため、seed 3407で
150 queryへ決定論的に展開する。RGB-D・depth・3D headは対象外とした。

### フェーズ2: loader・config実装

`DOTAOBBDataset`へ`metainfo.classes`、`img_suffixes`、`img_shape`、
`strict_loading`を公開した。strict時はunknown class、9/10列以外、非finite、退化quad、
stem不一致、decode不能、宣言shape不一致を明示例外にする。suffixは大文字小文字と
先頭`.`の有無を正規化する。bboxをPython listへ正規化し、LoadAnnotationsでの
list-of-ndarray Tensor変換警告を除いた。旧loaderは`stem` classの互換wrapperで残した。

Phase 1はlinear Rotated IoU 5 epoch、Phase 2はGWD 15 epochとし、Phase 1 bestだけを
固定名`selected_best.pth`でPhase 2へ渡す。HGNet stemだけ凍結し、stage 1–4、neck、
transformer、OBB headは更新する。train/valは800x600 identity resize、150 query、
batch 32、FP16、MuonScheduleFreeを使う。

### フェーズ3: GPU検証・full training

100→150 query baselineを測定し、batch 20 GWD smokeとbatch 32 Rotated IoU smokeで
finite、VRAM、速度を確認した。DataLoader workerが終了時に待ち続ける挙動を確認し、
当curriculumでは`persistent_workers=False`を明示した。正式にRotated IoU 5 epoch、
そのbestからGWD 15 epochを実行し、毎epochの`rbbox_mAP_50`でbestを保存した。

### フェーズ4: 最終評価・可視化・監査

GWD bestをteacher-free inference configで再評価してprediction pickleを出力した。
score 0.05以上、最大150 queryを6画像へ描き、checkpoint/box/scoreに紐づくmanifestを
保存した。PyTorch 2.6+の`weights_only=True`互換のため、test CLIと可視化CLIへ既存の
MMEngine safe-global登録helperを適用した。

---

## 3. 作業チェックリスト

### フェーズ1: 調査・設計

### 手順 1: 補正済みDOTA正本を検証する
- [x] 🖐 **操作**: archiveを`/workspace/YOPO_clone/datasets`へ展開し、画像・label・row・shape・class・difficultyを全件集計する。
- [x] 🔎 **確認**: train 1,368/94,605、test 181/9,034、全画像800x600、class tomato、decode不能0である。
- [x] 🧪 **テスト**: 全103,639 rowがfiniteかつ正面積で、画像/label stem差分が0であることを検査する。
- [x] 🛠 **エラー時対処**: archive path traversalやpair欠損があれば展開・学習を中止し、対象pathと件数を記録する。

### 手順 2: query容量と既存checkpointを監査する
- [x] 🖐 **操作**: 画像ごとのGT数とcheckpointの`query_embedding` shapeを確認する。
- [x] 🔎 **確認**: 最大GT 129に対して既存100 queryが不足し、150 queryなら全画像を収容できる。
- [x] 🧪 **テスト**: 先頭100 queryを保持し、追加50 queryをseed 3407で生成したcheckpointのshape/hashを確認する。
- [x] 🛠 **エラー時対処**: checkpoint key/shape不一致は暗黙skipせず、expansion CLIで対象keyを明示する。

### フェーズ2: loader・config実装

### 手順 3: 汎用loaderの契約testを先に追加する
- [x] 🖐 **操作**: `tests/test_dota_obb_dataset.py`へclass/suffix/auto-shape/strict異常系/互換性testを追加する。
- [x] 🔎 **確認**: 実装前は`DOTAOBBDataset` ImportErrorで失敗し、必要な公開APIが固定される。
- [x] 🧪 **テスト**: `uv run pytest -q tests/test_dota_obb_dataset.py`を実装前fail、実装後passへ変える。
- [x] 🛠 **エラー時対処**: registry import漏れなら`yopo/datasets/__init__.py`のexportとcustom importを確認する。

### 手順 4: `DOTAOBBDataset`を実装する
- [x] 🖐 **操作**: `yopo/datasets/dota_tomato.py`へconfigurable/strict loaderを実装する。
- [x] 🔎 **確認**: class mapping、9/10列、finite、difficulty、degenerate、stem、decode、shapeを明示的に扱う。
- [x] 🧪 **テスト**: PNG/uppercase JPG/auto shape/unknown class/malformed/nonfinite/degenerate/unpaired/shape mismatchを確認する。
- [x] 🛠 **エラー時対処**: strict falseの互換経路は旧datasetだけで利用し、新configでは必ずstrict trueを明示する。

### 手順 5: 旧loader互換とTensor変換効率を保つ
- [x] 🖐 **操作**: `DOTATomatoDataset`を`stem` class、736x512、JPG、strict falseのwrapperとして残し、bboxをlistで返す。
- [x] 🔎 **確認**: 旧METAINFOが変わらず、新runでlist-of-ndarray警告が消える。
- [x] 🧪 **テスト**: legacy登録testと実GPU train logの警告有無を確認する。
- [x] 🛠 **エラー時対処**: core BaseBoxesを広域変更せず、dataset境界で標準形式へ正規化して影響範囲を限定する。

### 手順 6: RGB-only curriculum configを固定する
- [x] 🖐 **操作**: Phase 1/2/smoke/inferenceの4 configと`docs/rgb_obb_corrected_dota_curriculum.md`を作る。
- [x] 🔎 **確認**: RGB 3ch、150 query、tomato strict loader、Rotated IoU→GWD、depth/3D非包含である。
- [x] 🧪 **テスト**: Config contract testでloss type、epoch、LR、dataset、load順を検証する。
- [x] 🛠 **エラー時対処**: config継承で古いdict keyが残る場合は`_delete_=True`を使い、resolved configを確認する。

### フェーズ3: GPU検証・full training

### 手順 7: baselineとbatch 32 smokeを測る
- [x] 🖐 **操作**: query-expanded checkpointを181画像で評価し、1 epoch smokeをbatch 32で実行する。
- [x] 🔎 **確認**: baseline mAP50 0.0732、smoke peak 23,297 MiB、finite/OOMなし、GT 9,034である。
- [x] 🧪 **テスト**: loss/grad/metricがfiniteで、smoke後mAP50 0.0870となることをlogで確認する。
- [x] 🛠 **エラー時対処**: OOM時だけbatch 24へ戻し、非finite時はloss modeとAMPを分離診断する。

### 手順 8: Phase 1 Rotated IoUを5 epoch学習する
- [x] 🖐 **操作**: `uv run python tools/train.py configs/yopo/rotated_deformable_detr_tomato_obb_corrected_riou_stage1.py --work-dir work_dirs/rddetr_tomato_obb_corrected_riou_stage1`を実行する。
- [x] 🔎 **確認**: epoch 4 bestがmAP50 0.0917でbaselineを上回る。
- [x] 🧪 **テスト**: 全epochでNaN/Inf/OOMなし、GT 9,034、checkpoint hashを確認する。
- [x] 🛠 **エラー時対処**: 最終epochが悪化した場合は`save_best`のepoch 4を固定名へsymlinkする。

### 手順 9: Phase 2 GWDを15 epoch学習する
- [x] 🖐 **操作**: Phase 1 `selected_best.pth`からGWD configを15 epoch実行する。
- [x] 🔎 **確認**: epoch 15 bestがmAP50 0.0960でPhase 1 bestを上回る。
- [x] 🧪 **テスト**: epoch 12後のLR低下、全loss finite、peak 23,297 MiB、GT 9,034を確認する。
- [x] 🛠 **エラー時対処**: GWDが悪化した場合はPhase 1 bestを採用し、後段を採用したことにしない。

### フェーズ4: 最終評価・記録

### 手順 10: bestをteacher-freeで再評価する
- [x] 🖐 **操作**: inference configとGWD bestで`tools/test.py --out predictions.pkl`を実行する。
- [x] 🔎 **確認**: mAP50 0.0960、recall 0.3745、precision 0.1246、matched rIoU 0.6270を再現する。
- [x] 🧪 **テスト**: train中best metricと独立test CLI出力が一致する。
- [x] 🛠 **エラー時対処**: PyTorch weights-only error時は信頼済みMMEngine型だけを既存safe-global helperで登録する。

### 手順 11: 低threshold可視化とmanifestを生成する
- [x] 🖐 **操作**: score 0.05、max 150で6画像へOBBとscoreを描画する。
- [x] 🔎 **確認**: 6 PNGとmanifestが存在し、各画像150 queryのbox/score/checkpointが追跡できる。
- [x] 🧪 **テスト**: PNG decode、manifest JSON、ファイル数、目視で角度・座標系を確認する。
- [x] 🛠 **エラー時対処**: 表示が混雑する場合もcoverage証跡は保持し、用途別表示はthreshold/max-detsをCLIで変更する。

### 手順 12: 回帰test・Git・hashを監査する
- [ ] 🖐 **操作**: 対象pytest/ruffを実行し、tracked差分だけをcommitして`origin/rgb-d`へpushする。
- [ ] 🔎 **確認**: test/lint成功、dataset/checkpoint/work_dirsをcommitしていない、local HEADとoriginが一致する。
- [ ] 🧪 **テスト**: `uv run pytest -q tests/test_dota_obb_dataset.py`と関連OBB test、`uv run ruff check`を実行する。
- [ ] 🛠 **エラー時対処**: unrelated user変更はstageせず、push競合時はfetch後に履歴を確認して非破壊的に解決する。

---

## 4. 作業に使用するコマンド参考情報

```bash
cd /home/kasm-user/Desktop/YOPO_clone
uv run pytest -q tests/test_dota_obb_dataset.py

uv run python tools/train.py \
  configs/yopo/rotated_deformable_detr_tomato_obb_corrected_riou_stage1.py \
  --work-dir work_dirs/rddetr_tomato_obb_corrected_riou_stage1

uv run python tools/train.py \
  configs/yopo/rotated_deformable_detr_tomato_obb_corrected_gwd_stage2.py \
  --work-dir work_dirs/rddetr_tomato_obb_corrected_gwd_stage2

uv run python tools/test.py \
  configs/yopo/rotated_deformable_detr_tomato_obb_corrected_inference.py \
  work_dirs/rddetr_tomato_obb_corrected_gwd_stage2/selected_best.pth \
  --work-dir work_dirs/rddetr_tomato_obb_corrected_final_eval \
  --out work_dirs/rddetr_tomato_obb_corrected_final_eval/predictions.pkl
```

## 5. 最新成果物と証跡

| 種別 | 正本 | metric/hash |
| :--- | :--- | :--- |
| 設計 | `docs/rgb_obb_corrected_dota_curriculum.md` | data契約、loss順、gate、実測値 |
| Phase 1 best | `work_dirs/rddetr_tomato_obb_corrected_riou_stage1/best_rbbox_mAP_50_epoch_4.pth` | mAP50 0.0917、SHA-256 `a717c10cdf9a2b4ca9f902d8ee1d52e9b91c2917697b864ea32d5d79d70b4409` |
| final best | `work_dirs/rddetr_tomato_obb_corrected_gwd_stage2/best_rbbox_mAP_50_epoch_15.pth` | mAP50 0.0960、SHA-256 `b09d4ac57e4f0ee7b5ac8f73858e5861e691ecba98069fc518f98046063886c0` |
| prediction | `work_dirs/rddetr_tomato_obb_corrected_final_eval/predictions.pkl` | SHA-256 `163a3a93d8b6331933b4c95f7470dc981ec871280c495f41d8f214ebfe787fb4` |
| visual | `work_dirs/rddetr_tomato_obb_corrected_final_eval/overlays_score0p05/` | 6 PNG、manifest SHA-256 `238ce79e780af6a10a13245223ba6177e3c8d389ebef5e2d7bf8f92bedf71ccd` |

---

## 6. 完了の定義

- [x] 観点1（SG-1/TR-2）: 補正済みDOTAの件数・shape・class・query容量が全件検証済み。
- [x] 観点2（SG-2/TR-1）: 汎用loaderと旧loader互換が正常・異常系testで固定済み。
- [x] 観点3（SG-3/TR-2/3）: RGB-onlyかつRotated IoU→GWDのscope/load順がconfigとdocsで一致。
- [x] 観点4（SG-4/TR-4/5）: batch 32でfinite/OOMなし、mAP50が0.0732から0.0960へ改善。
- [x] 観点5（SG-5/TR-6）: checkpoint/prediction/PNG/manifest/hash/再実行コマンドが記録済み。
- [ ] 観点6（SG-5/TR-6）: 最終回帰test・lint・commit・pushが完了しoriginと一致。

---

## 7. 作業記録

**重要な注意事項：**

* 作業開始前に必ず `date "+%Y-%m-%d %H:%M:%S %Z%z"` コマンドで現在時刻を確認し、正確な日時を記録します。
* 各作業項目を開始する際と完了する際の両方で記録を行うこと。
* 作業内容は具体的なコマンドや操作手順を詳細に記載すること。
* 結果・備考欄には成功／失敗、エラー内容、解決方法、重要な気づきを必ず記入すること。
* 複数のフェーズがある場合は、フェーズごとに開始・完了の記録を取ること。
* コード変更を行った場合は、変更したファイル名と変更内容の概要を記録すること。
* エラーが発生した場合は、エラーメッセージと解決策を詳細に記録すること。
* 本書作成時の時刻確認は`2026-08-25 19:16:48 JST+0900`。それ以前の実行時刻はMMEngine logのJSTを転記した。

| 日付 | 時刻 | 作業者 | 作業内容 | 結果・備考 |
| :--- | :--- | :--- | :--- | :--- |
| 2026-08-25 | 18:54 JST | Codex | baseline評価完了 | q150既存重みでmAP50 0.0732、recall 0.3478、GT 9,034。`tools/test.py`へsafe globals登録を追加。 |
| 2026-08-25 | 18:54 JST | Codex | GWD batch 20 smoke | peak 14,646 MiB、finite、1 epoch mAP50 0.0875。正式loss順変更前の容量証跡として保持。 |
| 2026-08-25 | 18:57 JST | Codex | ユーザー指定でcurriculum反転 | 正式順をRotated IoU→GWDへ変更。config名、load順、docs、contract testを更新し9 passed。 |
| 2026-08-25 | 18:58 JST | Codex | Rotated IoU batch 32 smoke | peak 23,297 MiB、finite/OOMなし、mAP50 0.0870。32 GiB内で正式batch採用。 |
| 2026-08-25 | 19:00 JST | Codex | loader実運用改善 | bbox list正規化、uppercase suffix、strict decode/shape確認を追加。10 tests passed。persistent worker終了待ちはconfigでfalseにして解消。 |
| 2026-08-25 | 19:00–19:03 JST | Codex | Phase 1 Rotated IoU full | 5 epoch完走。epoch 4 best mAP50 0.0917、recall 0.3689、matched rIoU 0.6249。 |
| 2026-08-25 | 19:04–19:14 JST | Codex | Phase 2 GWD full | 15 epoch完走。epoch 15 best mAP50 0.0960、recall 0.3745、precision 0.1246、matched rIoU 0.6270。 |
| 2026-08-25 | 19:15 JST | Codex | teacher-free最終評価 | train bestと同じmetricを再現し、1,390,039-byte prediction pickleを保存。 |
| 2026-08-25 | 19:15 JST | Codex | 可視化初回失敗 | PyTorch weights-onlyで`HistoryBuffer`未allowlist。可視化CLIへ既存safe-global helperを追加。 |
| 2026-08-25 | 19:16 JST | Codex | 可視化再実行 | score 0.05、最大150で6 PNGとmanifest生成。1枚を原寸目視し、画像座標でOBBが描かれることを確認。 |
| 2026-08-25 | 19:16 JST | Codex | workdoc作成開始 | `write-workdoc-uv`のtemplate、repository/justfile/uv構成を確認し、本書へ証跡を集約。 |
