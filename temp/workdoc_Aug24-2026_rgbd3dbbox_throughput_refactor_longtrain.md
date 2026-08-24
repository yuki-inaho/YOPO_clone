# 作業計画書 兼 記録書

---

**日付：** 2026年08月24日  
**作業ディレクトリ・リポジトリ:** `/home/kasm-user/Desktop/YOPO_clone`（Git repository）  
**作業者：** Codex

---

## 1. 作業目的

本作業は、完了済みのRGB-D 3D BBOX transfer学習を、数値意味を保ったまま短時間化・保守しやすくリファクタリングし、その実測根拠を得てから精度改善と長時間学習へ安全に移行するために実施する。

* **目標1:** 実データRGB-D 3D BBOX学習の時間内訳（data、forward/backward、optimizer、validation/checkpoint）を再現可能に測定し、ボトルネックだけを改善する。
* **目標2:** training/benchmark/long-run configの重複をDRYに整理し、意味を変えない高速化をTDDで守る。
* **目標3:** 現best重みを起点に、実NOCS evaluatorと品質ゲートを維持した長時間fine-tuneを実行し、精度・安定性・artifactを報告する。

### 1.1 ゴール要求分析

* **ユーザーの直観的・直截的な目的:** 20GB VRAMを有効利用して速く回る訓練基盤に整えたうえで、見せかけのconfidence調整ではなく本当に精度の高いRGB-D→3D BBOXモデルを長く学習したい。
* **明示要求:** 学習終了後に学習時間を短くする施策とリファクタリングを行い、その後に精度向上と長時間学習へ移る。
* **暗黙制約:** `uv`環境、既存NOCS fruit split（train 300/val 50）、raw uint16-mm→m+valid mask、RGB B2 300-key strict transferとfreeze、FP16 AMP/batch26、NOCSMetric、暗黙fallback禁止、user-owned dirty filesの保護、TDD/DRY/KISS/SOLID、checkpoint容量を無駄に増やさないこと。
* **現ベースライン:** 3Dは`rgbd3d_amp26_lr2e4_20ep_eval`で20epoch完走、peak VRAM=19,326/20,475MiB、best epoch14 3D IoU@0.50=0.169004、AP50最大=0.002。先行2D OBBはepoch20 mAP@0.5=0.033770で、score threshold 0.01/0.10の再評価とも同値の0.033770であり、低値は描画/score thresholdだけの問題ではない。
* **成功条件:** (a) 変更前後でRGB freeze/transfer/data contract/finite loss/evaluatorを保つ、(b) 実測に基づく速度改善またはcompute-boundである根拠を記録する、(c) long runは無限に回さず品質ゲート・最良weights-only checkpoint・real validationを持つ、(d) baselineを上回れない場合も数値と原因を隠さない。
* **非ゴール:** GTを変える、valをtrainへ混入する、confidence thresholdだけでmetricを良く見せる、RGB freezeを無断解除する、既存user-owned dirtyをstage/reset/commit/pushする、未検証の高速化を本走へ混ぜること。
* **リスクと前提:** batch26は約94.4% VRAMでbatch27は継続step OOM、旧LR=0.0025はepoch13でHungarian cost非有限。長時間runはLR=2e-4以下・finite-cost gate・隔離work_dirを維持する。最適化が10%未満なら複雑化を採用せず、測定結果をもってcompute-boundと結論する。

### 1.2 サブゴール構造

| ID | サブゴール | 目的との対応 | 成果物 | 検証方法 |
| :--- | :--- | :--- | :--- |
| SG-1 | 学習速度を分解計測する | 目標1 / TR-1 | baseline benchmark JSON/CSV、GPU/CPU/data時間 | 同一seed・同一batch26の複数step中央値とvariance |
| SG-2 | 意味を保つ高速化を実装する | 目標1 / TR-2 | profiler hook/config、速度改善patch | freeze/transfer/finite/evaluator回帰testとbefore/after比較 |
| SG-3 | 設定・ツールをリファクタリングする | 目標2 / TR-3 | 共通base configまたは小さなhelper、tests | config build、重複削減、ruff/対象test |
| SG-4 | 品質改善可能な長時間fine-tuneを作る | 目標3 / TR-4 | long-run config、best checkpoint、scalars | real validationの品質ゲートとbest metric |
| SG-5 | 監査可能に完了する | 全目標 / TR-5 | overlays、work record、final audit | artifact/test/worktree照合 |

### 1.3 トレーサビリティ方針

| Trace ID | 要求・制約 | 対応する作業要素 | 証跡 |
| :--- | :--- | :--- | :--- |
| TR-1 | 測らずに高速化しない | 手順1–3 | baseline/after profiler記録、GPU統計 |
| TR-2 | 数値意味を保つ高速化 | 手順4–7 | TDD、finite/freeze/transfer/evaluator test |
| TR-3 | DRY/KISS/SOLIDのリファクタリング | 手順5–7 | diff、config build、ruff/pytest |
| TR-4 | 真实3D品質を改善して長時間学習 | 手順8–10 | long-run scalars、best weights、NOCSMetric |
| TR-5 | 閾値偽装・無断変更なし | 手順10–12 | overlay manifest、worktree audit、作業記録 |

---

## 2. 作業内容

### フェーズ 1: 計測・設計 (SG-1 / TR-1, TR-2, TR-3)

1. 既存20epoch log/configとデータpipelineを読み、train/validation/checkpointの時間・I/O・GPU使用量を分離する。
2. 同一batch26、同一モデル入力で短いprofiler benchmarkを作り、中央値・p95・GPUメモリをJSONに保存する。
3. 結果に基づき、data-boundならdecode/loader、validation-boundなら評価頻度、compute-boundなら不要な同期・重複処理だけを候補にする。精度・数学・batchサイズを無根拠に変えない。

### フェーズ 2: 高速化・リファクタリング (SG-2 / SG-3, TR-2, TR-3)

1. benchmark用計測hook/toolを小さく追加し、通常trainへの副作用をconfig opt-inに限定する。
2. 重複したRGB-D 3D configをbase/overrideに整理し、smoke/benchmark/long-runの差だけを各ファイルに残す。
3. profilerで特定した単一ボトルネックを一つずつ変更し、before/afterの同一条件比較と数値contract testで採否を決める。

### フェーズ 3: 精度改善・長時間学習・監査 (SG-4 / SG-5, TR-4, TR-5)

1. best epoch14を明示起点にしたlong-finetune configを作り、LR/validation interval/checkpoint保持数・品質中間ゲートを固定する。
2. 短いquality gateを通過してからlong runを起動する。NaN/Inf/OOM、Hungarian finite-cost gate、metric低下の各停止/継続条件を記録する。
3. 最良モデルを実NOCS evaluator、thresholdなしoverlay、test、worktree監査で確認する。

### 2.4 計測に基づく採用方針（2026-08-24）

* **採用する短時間化:** long fine-tuneだけは`val_interval=5`とし、NOCSMetricを5epochごとに全50 validation imageへ実行する。baseline epoch10–20の実測はtrain約31秒、validation+metric集計16秒、計47秒/epochである。80epochなら毎epoch評価の約3,760秒に対し、5epoch評価は約2,736秒（約27%短縮）を見込む。
* **採用しない候補:** data waitはwarm-up後中央値0.06185秒/compute 2.63411秒（約2.3%）であり、loader/decodeの複雑な変更は10%短縮の根拠を満たさない。batch27は継続stepでOOM、batch26を維持する。モデル数学、raw-depth contract、RGB freeze、strict transfer、loss、metric、score thresholdは変更しない。
* **品質保護:** quality gateはepoch5で初回実評価し、以後5epochごとにbest `3d_iou_0.50` を判定する。validationを省略するepochもtrain loss/grad/finite-cost gateを全step監視し、NaN/Inf/OOMなら直ちにlong run不採用とする。
* **リファクタリング境界:** benchmark hookはopt-inで通常/long configに入れない。config共通化は次手順で解決済みdict同値testを通過した場合だけ採用する。

---

## 3. 作業チェックリスト

*作業が完了したら `[ ]` を `[x]` に変更し、直後に「作業記録」へ結果を追記する。各項目は上から一つずつ実行する。*

### フェーズ 1: 計測・設計

### 手順 1: 速度・品質ベースラインを固定する（SG-1/TR-1,TR-4）
- [x] 🖐 **操作**: 20epoch log/scalars、config、2D threshold sweep、GPU記録を読み取り、3D train/eval時間、VRAM、AP50/3D IoUと2D mAPの基準値を本書へ記録する。
- [x] 🔎 **確認**: artifact path、epoch、score thresholdとmetricの関係、比較対象が明示され、閾値と本metricを混同していない。
- [x] 🧪 **テスト**: `test_rgbd_3dbbox_e2e`と`test_rgbd_3dbbox_workdoc_audit`を実行し、baseline artifact/scalarの有限性を確認する。
- [x] 🛠 **エラー時対処**: log/scalarが欠落ならrunを捏造せず、存在するcheckpoint/manifestと欠落範囲を記録し、計測対象を明示的に再作成する。

### 手順 2: train/eval/data時間を短期実測する（SG-1/TR-1）
- [x] 🖐 **操作**: batch26・同一seed・同一transfer/freezeでwarm-upを除く複数iterationのdata/forward-backward/optimizer/eval/checkpoint時間、GPU peakを計測する専用benchmark config/toolを作る。
- [x] 🔎 **確認**: JSON/CSVにはconfig hash、seed、batch、step別wall time中央値/p95、GPU memory、測定除外区間がある。
- [x] 🧪 **テスト**: `test_rgbd_3dbbox_throughput_benchmark_schema`をfail→passさせ、必須fieldの欠落・nonfinite・0 stepを拒否する。
- [x] 🛠 **エラー時対処**: CUDA非同期で時間が歪む場合は`torch.cuda.synchronize()`を計測境界だけに置く。OOMならbatchを変えず短いstep数に下げ、長時間runへ流用しない。

### 手順 3: 最小高速化方針を選ぶ（SG-1/TR-1,TR-2,TR-3）
- [x] 🖐 **操作**: benchmarkから最大割合の待ち時間を一つ選び、候補・採否基準（中央値で10%以上短縮、又はcompute-boundの不採用根拠）・変更対象を本書へ追記する。
- [x] 🔎 **確認**: train数学、data split、RGB freeze、batch26、score/metric計算を変えないことと、評価頻度を変える場合の比較可能性が明文化される。
- [x] 🧪 **テスト**: 設計だけのため自動testは追加せず、手順2 JSONを入力に採否計算を再現するコマンドを作業記録に残す。
- [x] 🛠 **エラー時対処**: 優位差がノイズ範囲なら最適化を採用しない。複数候補が僅差なら可読性・リスクの低い方だけを選ぶ。

### フェーズ 2: 高速化・リファクタリング

### 手順 4: opt-in throughput計測を実装する（SG-2/TR-1,TR-2）
- [x] 🖐 **操作**: 新規hook/toolへ計測責務を隔離し、通常configでは無効、benchmark configでのみJSON/CSVを出すよう実装する。
- [x] 🔎 **確認**: 計測のCUDA同期・I/Oはbenchmark時だけで、通常trainingのloss/optimizer/evaluator stateを変更しない。
- [x] 🧪 **テスト**: `test_rgbd_3dbbox_throughput_benchmark_schema`でempty/nonfinite/順序不正をfail、正常recordをpassさせる。
- [x] 🛠 **エラー時対処**: runner hook順序が不明なら既存`memory_profiler_hook.py`とmmengine APIを読み、推測したcallback名を使わない。

### 手順 5: RGB-D 3D configの重複を整理する（SG-3/TR-3）
- [x] 🖐 **操作**: transfer/smoke/capacity/20epoch configの共通部分を既存baseへ残し、run固有overrideだけにして、base configの意味を変えないリファクタリングを行う。
- [x] 🔎 **確認**: config build後にdata contract、frozen RGB、MAE depth init、max_per_img=100、AMP、optimizer/metricがbaselineと一致する。
- [x] 🧪 **テスト**: `test_rgbd_3dbbox_config_equivalence`をfail→passさせ、旧20epoch configとrefactor後の意味的fieldが等しいことを確認する。
- [x] 🛠 **エラー時対処**: config inheritanceで値が不透明なら`Config.fromfile`の解決済みdictを比較し、完全同値にできないfieldを理由付きで明示する。

### 手順 6: 実測ボトルネックだけを高速化する（SG-2/TR-1,TR-2）
- [x] 🖐 **操作**: 手順3で選んだ一つのdata/eval/不要同期ボトルネックを、public contractを変えない小さなpatchで改善する。
- [x] 🔎 **確認**: before/after同一条件benchmarkで中央値10%以上短縮、又は非採用のcompute-bound根拠がJSONと作業記録にある。
- [x] 🧪 **テスト**: 影響するdataset/runner/config testと`test_rgbd_3dbbox_config_equivalence`を実行し、finite/freeze/transfer/evaluator contractを回帰確認する。
- [x] 🛠 **エラー時対処**: speedupと数値contractが両立しない場合はpatchを本走に採用せず、計測hookだけを残して次候補を一つに再分割する。

### 手順 7: 高速化後の短期回帰を通す（SG-2/TR-2,TR-3）
- [x] 🖐 **操作**: 採用configで短期AMP smokeとreal validationを一度実行し、baselineと同一splitのfinite loss/grad、RGB freeze、3D metric、GPU peakを記録する。
- [x] 🔎 **確認**: NaN/Inf/OOM/Hungarian gate例外なし、baselineよりVRAM増加なし、real evaluatorは全50 val imageを処理する。
- [x] 🧪 **テスト**: `test_amp_rotation_loss.py`、`test_rgbd_3dbbox_transfer_freeze.py`、`test_rgbd_3dbbox_e2e.py`を実行する。
- [x] 🛠 **エラー時対処**: 数値例外は速度patchを外してbaseline contractから、OOMはbatch26を維持して計測stepを減らして再現する。silent fallbackはしない。

### フェーズ 3: 精度改善・長時間学習・監査

### 手順 8: best重み起点のlong fine-tune configを固定する（SG-4/TR-4）
- [x] 🖐 **操作**: epoch14 best weightsを明示`load_from`にした隔離long-run configを新設し、RGB freeze/MAE/3D evaluatorを保持、低LR、validation interval、best-only checkpoint、停止/継続quality gateを固定する。
- [x] 🔎 **確認**: 2D checkpointからのRGB transfer hookとfull 3D best重みのロード順が明文化され、optimizer stateをresume偽装せずfresh fine-tuneである。
- [x] 🧪 **テスト**: `test_rgbd_3dbbox_long_finetune_config`をfail→passさせ、source checkpoint、max_per_img、freeze、finite-safe optimizer、storage bounded checkpoint policyを確認する。
- [x] 🛠 **エラー時対処**: full checkpointとtransfer hookのkey衝突があればロードreportを確認し、RGBのみstrict transferのままfull model重みの優先順をコードで明示する。曖昧なら本走を起動しない。

### 手順 9: quality gateを通す（SG-4/TR-4）
- [x] 🖐 **操作**: long configを短い決め打ちepochで実行し、finite loss/grad、NOCSMetric、best 3D IoU、AP50、GPU memoryをbaselineと比較する。
- [x] 🔎 **確認**: 一度でもNaN/Inf/OOM/finite-cost gate例外が出たらlong runへ進まず、last safe checkpoint・step・tensor/cost情報を保存する。
- [x] 🧪 **テスト**: `test_rgbd_3dbbox_long_finetune_config`とrun artifact schema testを実行し、quality gateのlog/scalarが有限であることを確認する。
- [x] 🛠 **エラー時対処**: metricが短期に悪化しても単発値で停止せず、定義済みwindow/best metricで判断する。baselineより明確に不安定ならLR/loader変更を一つずつ戻す。

### 手順 10: 長時間fine-tuneと実evaluatorを完走する（SG-4/TR-4）
- [ ] 🖐 **操作**: 手順9成功後のみ隔離work_dirで長時間fine-tuneを一度起動し、定期NOCS validation、best-only weights、throughput/GPU/finite監視を保存する。
- [ ] 🔎 **確認**: 全validationが50 imageを処理し、best metric/epoch、総wall time、実効epoch時間、AP50/3D IoU、NaN/Inf/OOM不在がlog/scalarsに残る。
- [ ] 🧪 **テスト**: `test_rgbd_3dbbox_longrun_e2e`でrun record、best checkpoint、metric finite、baseline比較fieldを確認する。
- [ ] 🛠 **エラー時対処**: 中断/例外時は無断resumeせず、checkpoint/optimizer保存有無・再開可否・再開configを記録する。品質未改善でもcheckpointを削除せず数値を報告する。

### 手順 11: 最良モデルのRGB overlayを再確認する（SG-5/TR-4,TR-5）
- [ ] 🖐 **操作**: long runのbest checkpointでthresholdなしtop-1 RGB 3D BBOX overlay 3枚とmanifestを再生成し、baseline overlayと併記する。
- [ ] 🔎 **確認**: K/T/m単位/corner projection、checkpoint path、score、選択規則、画像サイズをmanifestに記録し、目視で鏡映/軸/尺度を確認する。
- [ ] 🧪 **テスト**: `test_rgbd_3dbbox_overlay_artifact.py`と`test_rgbd_3dbbox_projection_guard.py`を実行する。
- [ ] 🛠 **エラー時対処**: projection errorはK→unit→T→corner orderingで検証し、閾値変更で隠さない。品質がbaselineより悪くても比較画像を削除しない。

### 手順 12: 再現性・品質・worktreeを最終監査する（SG-5/TR-1–TR-5）
- [ ] 🖐 **操作**: benchmark、config、tests、long-run log/scalars/checkpoint/overlay、`git status --short`を照合し、速度/品質/未達を本書へ追記する。
- [ ] 🔎 **確認**: すべてのTrace IDに証跡、user-owned dirtyとの分離、無断stage/reset/commit/pushなし、受入できない精度は未達として明記される。
- [ ] 🧪 **テスト**: `rgbd_3dbbox_throughput_longrun_audit`を作成し、必須artifact・finite metrics・config provenance・workdoc未完了ゼロを検証する。
- [ ] 🛠 **エラー時対処**: artifact欠落/未解決例外/metric schema差があれば完了にせず、該当手順を四項目へ細分化してlast safe stateと再開条件を記録する。

---

## 4. 作業に使用するコマンド参考情報

```bash
# リポジトリ・環境
just env-doctor
uv run python -c "from mmengine.config import Config; Config.fromfile('configs/yopo/nocs_custom_fruit_rgbd_3dbbox_amp_20ep.py')"

# 対象test
uv run pytest -q tests/test_rgbd_3dbbox_e2e.py tests/test_rgbd_3dbbox_transfer_freeze.py

# baseline/benchmark/long-run（各手順で確定した専用config/work_dirだけを使用）
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python tools/train.py <config.py> --work-dir <work_dir>

# GPU観察
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
```

---

## 6. 完了の定義

*作業が最後まで完了したら `[ ]` を `[x]` にしつつ、作業が本当に完了したかをチェックします。*

- [ ] 観点1: baselineと高速化後の速度・VRAM・数値contractが同一条件で比較され、採否が定量根拠で説明されている。
- [ ] 観点2: config/tool refactorはDRYで、RGB freeze、strict transfer、raw-depth/MAE、AMP、NOCS evaluatorの意味を変えずtestsで守られている。
- [ ] 観点3: long fine-tuneはquality gate後に実行され、real validation、best checkpoint、metrics、thresholdなしoverlay、有限性を確認している。
- [ ] 観点4: 全Trace IDの証跡、uv tests、worktree/authority audit、未達精度と再開条件が作業記録にある。

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

| 日付 | 時刻 | 作業者 | 作業内容 | 結果・備考 |
| :--- | :--- | :--- | :--- | :--- |
| 2026-08-24 | 15:10:16 UTC | Codex | 作業開始・作業書作成 | 前作業書`workdoc_Aug24-2026_rgbd_3dbbox_transfer.md`の全checklist/DoDを完了後、ユーザー指示「短時間化→refactor→精度向上/長時間学習」に対応する独立作業書を作成。3D baselineの20epoch/artifactと2D threshold sweepを固定し、閾値偽装を非ゴールにした。 |
| 2026-08-24 | 15:10:16 UTC | Codex | 手順1・操作完了: baseline 抽出 | `rgbd3d_amp26_lr2e4_20ep_eval`のscalarsを240 train/20 val recordとして解析。train time中央値=2.8946s/step（p95=3.2397）、data_time中央値=0.3336s（p95=0.4604）、memory peak=19,326MiB。val time中央値=0.07429s/image（50 imageで約3.71s/epoch）。best epoch14は3D IoU@0.50=0.169004/AP50=0、最終epoch20は0.114321/AP50=0.001。2D best epoch20はmAP@0.5=0.033770、threshold=0.01/0.10の再testも同mAPであり、閾値だけが低値の原因ではない。 |
| 2026-08-24 | 15:13:23 UTC | Codex | 定期状況・行動カウント reset | baselineのtrainは中央値2.895s/stepに対してdata_time 0.334s（約11.5%）、validationは50 imageで約3.71s/epoch。最初の高速化候補は評価頻度ではなくforward/backward周辺の実測で決める。閾値変更でmetricを良く見せない方針を維持する。 |
| 2026-08-24 | 15:13:23 UTC | Codex | 手順1・確認完了: 比較軸の固定 | 解決済みmain configからbatch=26、workers=4/persistent、fp16 loss_scale=1.0、LR=2e-4、max_per_img=100、NOCSMetric、best=`3d_iou_0.50`、save_last=Falseを抽出。2D sweep両方は同一epoch20 checkpoint/100queryで、score_thrだけ0.01/0.10、mAP=0.033770は不変。3DはNOCS AP50/3D IoUで判断し、2D score threshold/描画設定を品質代替にしていない。 |
| 2026-08-24 | 15:13:23 UTC | Codex | 手順1・テスト完了: baseline artifact contracts | `uv run pytest -q tests/test_rgbd_3dbbox_e2e.py tests/test_rgbd_3dbbox_workdoc_audit.py`=2 passed（0.03s）。20×12のtrain finite record、20 validation metric、epoch14 best checkpoint、300-key transfer report、前作業書の全task/DoD/artifact契約を現在のfilesystemで再確認した。 |
| 2026-08-24 | 15:13:23 UTC | Codex | 手順1・エラー時対処完了: baseline availability | `20260824_143138.log`、scalars、epoch14 checkpoint、transfer report、prediction dump、manifest、2D threshold reportsはすべて存在し、E2E/auditがfinite/schemaを検証済み。欠落runの捏造や再作成は不要。以後のbenchmarkはこのartifactを変更せず、隔離work_dirのみへ出力する。 |
| 2026-08-24 | 15:21:21 UTC | Codex | 手順2・操作完了: opt-in実測benchmark | `ThroughputBenchmarkHook`/schema test/1epoch benchmark configを追加。初回v1はLoggerHook後（LOW）でCUDA peakが368MiBへreset済みとなる計測欠陥を発見し成果不採用。NORMALへ修正後のv2を同一batch26/fp16/transfer/freeze/real valで完走。`throughput_benchmark.json`はwarm-up 2除外の10 train recordでcompute median=2.64445s/p95=2.67926、data median=0.06292s/p95=1.01715、train peak=19,327MiB、val compute median=0.06536s/image、val peak=461MiBを記録。checkpointは保存せず、v2 JSONのみを以後の速度基準に採用。 |
| 2026-08-24 | 15:23:25 UTC | Codex | 定期状況・行動カウント reset | v2でtiming/peakは正しくなったが、config hash/seedが不足したため正式比較から除外。schemaをseed/hash必須へ強化し、固定seed=3407のv3を起動済み。現在train 8/12で、他のGPU taskを並列起動せず50 val image・JSON書出しを待機中。 |
| 2026-08-24 | 15:23:25 UTC | Codex | 手順2・確認完了: v3 reproducible benchmark | `work_dirs/rgbd3d_throughput_benchmark_v3/throughput_benchmark.json`をschema読み込み。provenanceはconfig、SHA256=`2aa70a2c…eff06758`、seed=3407、batch=26。warm-up=2を除くtrain 10 recordはcompute median=2.63411s/p95=2.67762、data median=0.06185s/p95=0.91356、peak=19,327MiB。val 50 recordはcompute median=0.06411s/p95=0.06532、peak=461MiB。v3は計測専用のrandom 3D head one epochなのでIoU=0をquality比較に用いず、速度基準だけに用いる。 |
| 2026-08-24 | 15:23:25 UTC | Codex | 手順2・テスト完了: throughput schema | TDD初回は`ModuleNotFoundError`。hook実装後5 passed、LoggerHookがpeakをresetする優先度欠陥を発見しNORMAL優先度testを追加、さらにconfig hash/seed必須testを追加した。最終`uv run pytest -q tests/test_rgbd_3dbbox_throughput_benchmark_schema.py`=7 passed。empty/nonfinite/zero measured/非連番record、provenance欠落、peak reset順序を拒否・固定する。 |
| 2026-08-24 | 15:23:25 UTC | Codex | 手順2・エラー時対処完了: benchmark integrity | CUDA同期はhookのbefore/after train/val境界だけに置いた。v1のLOW priorityがLoggerHookの`reset_peak_memory_stats()`後に368MiBを読む問題をNORMALへ修正し、v2のseed/hash欠落はschema必須化・v3固定seedへ修正。OOM/NaN/Inf/finite-cost gateはv3で未発生、batch26を変更して速度を偽装していない。v1/v2 artifactは診断証跡として残すが、採用benchmarkはv3だけである。 |
| 2026-08-24 | 15:23:25 UTC | Codex | 手順3・操作完了: 計測ベースの採用方針 | baseline epoch10–20 timestampを再解析し、train 30–32s（代表31s）+NOCSMetric validation/集計15–16s（代表16s）=47s/epochと確定。v3のwarm-up後data 0.06185s/compute 2.63411sからloader最適化は根拠不足と判定。long runだけ`val_interval=5`へ変更する方針を2.4節へ追記した。80epoch概算は毎epoch評価3,760s→5epoch評価2,736s（約27%短縮）で、3D数学/batch/freeze/metric/thresholdは不変。 |
| 2026-08-24 | 15:23:25 UTC | Codex | 手順3・確認完了: speed/quality contract | 2.4節を再読し、採用変更はlong fine-tuneの`val_interval=5`だけと確認。全50 val image/NOCSMetricは評価epochごとに不変で、最良metricは同じ`3d_iou_0.50`。`31*80 + 16*(80/5)=2,736`対`(31+16)*80=3,760`、短縮27.23%を再現。train数学、split、raw-depth、RGB freeze、strict transfer、batch26、score threshold/metricは変更対象から除外した。 |
| 2026-08-24 | 15:23:25 UTC | Codex | 手順3・テスト完了: 採否計算の再現 | 実装対象がない設計項目のため新規自動testは追加しない。`uv run python`でbaseline=`(31+16)*80=3,760`、interval5=`31*80+16*(80//5)=2,736`、reduction=`27.23%`を再算出。入力31/16はbaseline log timestamp、data/compute比はv3 JSONに由来し、採用根拠を再現可能にした。 |
| 2026-08-24 | 15:23:25 UTC | Codex | 手順3・エラー時対処完了: speedup selection | interval5は27.23%見込みで10%基準を上回る。loader/decodeはdata wait中央値がtrain computeの約2.3%で、優位差が基準未満のため採用せず、batch27は既知の反復OOMなので候補外。複数の高速化patchを重ねず、可読性・品質リスクの低いevaluation scheduleだけをlong configへ適用する。 |
| 2026-08-24 | 15:23:25 UTC | Codex | 手順4・操作完了: opt-in hook isolation | `ThroughputBenchmarkHook`は`throughput_benchmark_hook.py`へ隔離し、同期/MessageHub/peak/report書出しをそこだけに限定。config buildでmain `custom_hooks`にhookなし、benchmark configにhookありと確認。通常20epochや将来long runには計測同期・JSON I/Oを混入させない。 |
| 2026-08-24 | 15:23:25 UTC | Codex | 手順4・確認完了: no normal-run mutation | hookのpublic methodsはtrain/val callbackでtime、MessageHub read、CUDA peak read、benchmark file writeのみを実行する。main configはThroughputBenchmarkHookを含まず、benchmark v3は`throughput_benchmark.json`を出力済み。loss、optimizer、model weight、evaluatorの入力/metricをhookが変更する経路は無い。 |
| 2026-08-24 | 15:23:25 UTC | Codex | 手順4・テスト完了: benchmark hook schema | `uv run pytest -q tests/test_rgbd_3dbbox_throughput_benchmark_schema.py`=7 passed。初期ModuleNotFoundから、valid report、empty/nonfinite/invalid iteration、warmupゼロrecord、config hash/seed欠落、LoggerHook前priorityまでfail→passを記録。v3実runでも同schemaを出力できた。 |
| 2026-08-24 | 15:23:25 UTC | Codex | 手順4・エラー時対処完了: verified hook order | `IterTimerHook` sourceと`EpochBasedTrainLoop.run_iter`、実runのhook orderを読んでcallbackを確定。初回LOWはLoggerHook後でpeak reset済みだったため、IterTimer後/Logger前となるNORMALに修正した。未知callback名・暗黙fallbackは使わず、MessageHub scalar/explicit CUDAが無ければ例外で停止する。 |
| 2026-08-24 | 15:23:25 UTC | Codex | 手順5・操作完了: safe config DRY refactor | transfer baseにある`load_from=None`/`resume=False`をsmoke/capacity/20epoch/benchmark子configで重複宣言していたため8行を除去し、run固有overrideだけを残した。初案の外部Python factory importはMMEngineのlazy/non-lazy inheritance parseを壊したので即時撤回し、元のdirect transfer hook宣言を復元。解決済み4configのload/resumeはすべて従来値を維持する。 |
| 2026-08-24 | 15:23:25 UTC | Codex | 手順5・確認完了: resolved config equivalence | 20epoch configの解決済みdictを確認。train/valはraw-depth valid-mask pipeline、backbone=`FrozenRGBDDualBackbone`/freeze_rgb=True、depth init=finite valid-mask MAE、max_per_img=100、AMP fp16、LR=2e-4、NOCSMetric、best=`3d_iou_0.50`でbaselineと一致。設定リファクタリングはこれら意味的fieldを変更していない。 |
| 2026-08-24 | 15:23:25 UTC | Codex | 手順5・テスト完了: config equivalence | TDD初回は共通factory未実装でModuleNotFound、次にfactory importがMMEngine config parseを破壊する失敗を検出。安全にfactoryを撤回し、子configの冗長load/resumeを対象にしたtestへ再分割。除去前は1 failed/2 passed、除去後は`test_rgbd_3dbbox_config_equivalence.py`+throughput schema=10 passed。resolved contractも保持した。 |
| 2026-08-24 | 15:23:25 UTC | Codex | 手順5・エラー時対処完了: config parser compatibility | 外部utilityをconfigでimportする案はlazy/non-lazy inheritance incompatibilityのため非採用と記録。転用hookの直接dictは復元し、base継承で保証できるload/resumeだけを重複除去した。以後は`Config.fromfile`で解決済みdata/freeze/MAE/query/AMP/LR/evaluator/bestとchild defaultsを比較し、parserに依存する暗黙のfactoryを導入しない。 |
| 2026-08-24 | 15:32:43 UTC | Codex | 定期状況・行動カウント reset | interval5 configのTDDは初回FileNotFoundから4 passedへ移行。解決済みconfigはval_interval=1→5だけが差分で、batch/model/optimizerは同一。計測・safe refactorは完了し、次はこのspeed patchを短期real validationで回帰確認してからbest重み起点のlong fine-tune configへ進む。 |
| 2026-08-24 | 15:32:43 UTC | Codex | 手順6・操作完了: interval5 real speedcheck | `amp_interval5.py`を`train_cfg.max_epochs=5`/checkpoint save無効の隔離work_dirで実行。train epoch1→5 first/last=171s、epoch5 val first→metric=18s、合計190s。毎epochevaluation baseline見積5×47=235sより19.15%短く10%採否基準を満たす。50 val imageを1回全処理し、AP50=.001/IoU@.50=.0973はrandom-head short runのfinite確認値で品質比較には使わない。pthは作成されず、NaN/Inf/OOM/finite-cost例外も無い。 |
| 2026-08-24 | 15:32:43 UTC | Codex | 手順6・確認完了: measured speedup and invariants | config同値判定は`only_schedule_change=True`。5epoch baseline=235s、interval5 actual=190s、短縮=19.15%。logはepoch5 50/50 final metricを含み、Traceback/RuntimeError/ValueError/FloatingPointError/OOMは0、work_dir直下のpthは0。v3 JSONにはdata-boundでない根拠、speedcheck logには実run根拠が残る。 |
| 2026-08-24 | 15:32:43 UTC | Codex | 手順6・テスト完了: speed patch regressions | `test_rgbd_3dbbox_sample_contract.py`、`test_rgbd_3dbbox_transfer_freeze.py`、`test_rgbd_3dbbox_config_equivalence.py`、`test_rgbd_3dbbox_throughput_benchmark_schema.py`=13 passed。raw depth/3D finite、RGB freeze/strict transfer/backward、interval5同値、benchmark schemaを回帰確認。既知のDeprecationWarning等8件のみ。 |
| 2026-08-24 | 15:32:43 UTC | Codex | 手順6・エラー時対処完了: adopted speed policy | speedcheckのfinite/freeze/transfer/evaluator contractと19.15%短縮が両立したためinterval5をlong runへ採用する。もし今後long configでNaN/Inf/OOM、full50 val未実行、または10%未満となればinterval5を本走から外し、benchmark hookとv3証跡だけを残して次候補を単一変更で検証する。silent fallbackやbatch変更はしない。 |
| 2026-08-24 | 15:32:43 UTC | Codex | 手順7・操作完了: interval5 AMP real validation | speedcheckは採用config（batch26/fp16/strict RGB transfer/frozen RGB/valid-mask MAE）で5epochを完走し、epoch5でcustom_val 50 image/NOCSMetricを実行。final AP50=.001、3D IoU@.10/.25/.50/.75=.6287/.3475/.0973/.0081はfinite。random headの短期値なのでbaseline品質比較には使わず、AMP/freeze/evaluatorの実行経路確認に限定する。checkpointなし、実壁時計190s。 |
| 2026-08-24 | 15:32:43 UTC | Codex | 手順7・確認完了: finite/freeze/evaluator real run | speedcheck scalarsを解析し、60 train recordのloss/grad全有限、memory peak=19,326MiB（baseline19,326以下）、validation record=1、final IoU@.50=.097330/AP50=.001を確認。transfer reportはloaded=300/missing RGB=[]/unexpected=[]。log上のTraceback/RuntimeError/ValueError/FloatingPointError/OOM=0、epoch5 custom_val 50/50完走。 |
| 2026-08-24 | 15:42:26 UTC | Codex | 手順7・テスト | `uv run pytest -q tests/test_amp_rotation_loss.py tests/test_rgbd_3dbbox_transfer_freeze.py tests/test_rgbd_3dbbox_e2e.py` は 4 passed。AMP rotation loss、RGB転送/freeze、RGB-D 3D E2Eの回帰なし（既知の依存警告8件のみ）。 | 成功 |
| 2026-08-24 | 15:42:26 UTC | Codex | 手順7・エラー時対処 | speedcheck logの Traceback/RuntimeError/ValueError/FloatingPointError/OOM/NaN は全て0。`inf` の612件は `INFO` への単純文字列一致で、非INFO行は0。数値例外時はinterval5速度変更を外してbaseline contractへ戻し、OOM時はbatch26を維持して計測stepを減らす復旧方針を確認した。 | 成功 |
| 2026-08-24 | 15:43:15 UTC | Codex | 定期状況（行動カウント reset） | 手順1〜7を完了。V3 throughput基準値、最小安全refactor、interval5の5epoch実測（190s、baseline推定235s比19.15%短縮）、4回帰テスト成功を記録済み。以降はepoch14 best full 3D checkpointを起点に低LR/80epochのfresh long fine-tune設定、品質gate、本走、overlay、監査をこの順に進める。 | 進行中 |
| 2026-08-24 | 15:44:40 UTC | Codex | 手順8・操作 | `configs/yopo/nocs_custom_fruit_rgbd_3dbbox_long_finetune.py`を新設。epoch14 best full 3D checkpointを`load_from`、`resume=False`、batch26/fp16/frozen RGB/valid-mask MAE/NOCSMetric/max_per_img=100を継承。fresh optimizer LR=5e-5、80epoch、full 50-image validationを5epoch間隔、best-only/no optimizer/no last checkpointとした。 | 成功 |
| 2026-08-24 | 15:45:33 UTC | Codex | 手順8・確認 | MMEngine実装で`Runner.train(): load_or_resume()`が先、`EpochBasedTrainLoop.run(): before_train`が後と確認。従ってfull epoch14 checkpointを先にロードし、その後hookがRGBのみstrict再適用する。full 3Dと2D sourceのRGB 300 tensorは`torch.equal`全件true、missing=0。`resume=False`なのでoptimizer/scheduler stateを復元しないfresh fine-tune。 | 成功 |
| 2026-08-24 | 15:45:33 UTC | Codex | 手順8・テスト | 新規`test_rgbd_3dbbox_long_finetune_config.py`はconfig未作成時に1 failedを確認後、実装後に同testとconfig equivalenceで6 passed。source path、80epoch、val_interval=5、batch26、frozen RGB、max_per_img=100、fp16 wrapper LR=5e-5、best-only/no optimizer/no last、benchmark hook非混入、RGB 300 tensor一致を検証した。 | 成功 |
| 2026-08-24 | 15:45:33 UTC | Codex | 手順8・エラー時対処 | source runの`partial_transfer_report.json`はloaded=300、missing RGB=[]、unexpected=[]。full bestとの300 tensor完全一致と合わせ、strict RGB再適用によるkey衝突はない。将来この一致またはreportが崩れた場合は本走を起動せず、fullモデル優先順をコードで明示して再検証する。 | 成功 |
| 2026-08-24 | 15:47:27 UTC | Codex | 定期状況（行動カウント reset） | 手順8を全完了。手順9のquality gateを`rgbd3d_long_ft80_quality_gate`へ開始（15:46:48 UTC、epoch14 full bestをfresh load、5epoch/val interval5、seed3407）。15:47:27時点で親train processと4 workerが稼働、epoch1の更新を確認。終了後にfinite/50 val/best IoU@.50>=.14を判定する。 | 進行中 |
| 2026-08-24 | 15:50:44 UTC | Codex | 手順9・操作 | `rgbd3d_long_ft80_quality_gate`をepoch14 bestからfreshに5epoch実行し、train 60 update・custom_val 50/50を完走。NaN/Inf/OOM/finite-cost例外=0、peak GPU=19,752MiB、AP50=.0010、IoU@.10/.25/.50/.75=.4313/.2912/.0952/.0117。IoU@.50=.0952はsource best=.1690と保守下限=.14を下回ったため、このLR=5e-5候補はlong本走へ不採用。 | 失敗（品質gate） |
| 2026-08-24 | 15:50:44 UTC | Codex | 手順9・確認 | validation 50/50、train scalar/grad finite、error=0を確認。NaN/Inf/OOM/finite-cost gate例外はないためlast-safeはsource epoch14 checkpoint（IoU@.50=.169004）。ただしgate IoU@.50=.0952は−.0738/−43.7%で、品質継続条件を満たさない。この候補の80epoch実行は明示的に禁止し、source checkpointを保持する。 | 失敗（品質gate） |
| 2026-08-24 | 15:50:44 UTC | Codex | 手順9・テスト | 新規`test_rgbd_3dbbox_quality_gate_artifact.py`でgate config、60 train record、50/50 final metric、AP50/IoU有限、例外marker不在、best-only checkpoint、transfer reportを検証。初回はregex escapeの1 failedを修正し、`test_rgbd_3dbbox_long_finetune_config.py`と合わせて3 passed。 | 成功 |
| 2026-08-24 | 15:54:10 UTC | Codex | 定期状況（行動カウント reset） | source epoch14を`Runner.val()`で再評価しIoU@.50=.1690（AP50=.0000）を再現。従ってLR=5e-5 gateの.0952は評価器差ではなくfresh fine-tune候補の品質低下。`tools/test.py`はtest loop未定義で未実行停止したが、`Runner.val()`へ切替済み。既存候補を保持し、LRのみ1e-5へ下げた隔離config/gateとsingle-factor契約testを追加した。 | 進行中 |
| 2026-08-24 | 16:00:02 UTC | Codex | 手順9・品質gate再分割（LR 1e-5） | source/モデル/batch26/AMP/schedule/evaluator/checkpoint方針を固定し、LRのみ1e-5へ変更した5epoch gateを完走。finite/50 val/OOMなし、IoU@.50=.1266、AP50=.0010。5e-5の.0952より+.0314改善したがsource .1690比−25.1%、下限.14未達のため不採用。resolved optimizerは`AdamWScheduleFreeOptimizer`でMuonではなく、validationでoptimizer eval切替はない。元20epochもepoch14=.1690→epoch20=.1143で低下しており過学習域の証跡。 | 失敗（品質gate） |
| 2026-08-24 | 16:01:33 UTC | Codex | 定期状況（行動カウント reset） | LR=5e-5/.1e-5 gateはいずれも数値正常だがIoU@.50=.0952/.1266で不採用。LR=1e-6のみを変更した第三候補は16:00:45 UTCに`rgbd3d_long_ft80_lr1e6_quality_gate`へ起動、16:01:33時点で親train process+4 worker稼働、epoch1 update確認。 | 進行中 |
| 2026-08-24 | 16:05:14 UTC | Codex | 手順9・エラー時対処 | 同一5epoch/50-val windowでLRのみを5e-5→1e-5→1e-6と一つずつ戻した結果、IoU@.50=.0952→.1266→.1692。LR=1e-6はsource再評価=.1690を+.0002上回り、finite/OOMなし、best-only checkpointあり。LR=5e-5/1e-5は保持するが不採用、採用configは`nocs_custom_fruit_rgbd_3dbbox_long_finetune_lr1e6.py`。artifact schemaを3候補へ拡張後7 passed。 | 成功 |
| 2026-08-24 | 16:05:40 UTC | Codex | 手順10・本走開始 | `configs/yopo/nocs_custom_fruit_rgbd_3dbbox_long_finetune_lr1e6.py`を`work_dirs/rgbd3d_long_ft80_lr1e6`へ起動。fresh source=epoch14 full best、resume=False、batch26 fp16 AMP、LR=1e-6、80epoch、val interval=5、best-only/no optimizer/no last。epoch1のfinite update開始を確認。 | 進行中 |
| 2026-08-24 | 16:09:14 UTC | Codex | 手順10・epoch5監視 | 本走はtrain 60 updateとcustom_val 50/50を完了。AP50=.0000、IoU@.10/.25/.50/.75=.6399/.4377/.1691/.0204。quality gate LR=1e-6のIoU@.50=.1692と一致範囲、source=.1690を維持。NaN/Inf/OOM/finite-cost例外なし、VRAM peak<=19,752MiB。 | 進行中 |
| 2026-08-24 | 16:09:31 UTC | Codex | 定期状況（行動カウント reset） | 本走はepoch5 validationを通過して継続中。IoU@.50=.1691（source .1690、gate .1692と整合）、finite/OOMなし。16:09:31時点で親train processとworker群が稼働、次のfull validationはepoch10。設定変更なしで監視を続行する。 | 進行中 |
| 2026-08-24 | 16:12:34 UTC | Codex | 手順10・epoch10監視 | 本走epoch10 custom_val 50/50: AP50=.0000、IoU@.10/.25/.50/.75=.6300/.4352/.1694/.0215。epoch5=.1691/source=.1690を上回り、best-only checkpointは`best_3d_iou_0.50_epoch_10.pth`へ更新。NaN/Inf/OOM/finite-cost例外なし。 | 進行中 |
| 2026-08-24 | 16:15:56 UTC | Codex | 手順10・epoch15監視 | epoch15 custom_val 50/50: AP50=.0010、IoU@.10/.25/.50/.75=.6104/.4224/.1598/.0197。epoch10 best=.1694を下回るが、best-only checkpointはepoch10のまま保持。NaN/Inf/OOM/finite-cost例外なし。定義済みwindow/best判断に従い単発低下では中断しない。 | 進行中 |
| 2026-08-24 | 16:18:35 UTC | Codex | 定期状況（行動カウント reset） | 本走はepoch20へ進行中。epoch5/10/15 IoU@.50=.1691/.1694/.1598、best=epoch10 .1694を保持。16:18:35時点でtrain epoch20 4/12、全error marker=0。次のepoch20 validationを監視する。 | 進行中 |
| 2026-08-24 | 16:28:11 UTC | Codex | 手順10・epoch20–30監視 | epoch20/25/30 custom_val 50/50のIoU@.50=.1563/.1454/.1329（AP50=.0000/.0010/.0010）。いずれもepoch10 best=.1694未満のためbest-only checkpointはepoch10のまま、数値例外/OOMなし。epoch33まで継続中で、品質はbest metricで保護する。 | 進行中 |
| 2026-08-24 | 16:28:11 UTC | Codex | 定期状況（行動カウント reset） | 本走はepoch33以降も継続中。epoch10 best=.1694を保持し、epoch20/25/30の単発低下はcheckpointへ採用しない。NaN/Inf/OOM/finite-cost例外は0で、次のvalidationはepoch35。 | 進行中 |
| 2026-08-24 | 16:38:10 UTC | Codex | 定期状況（行動カウント reset） | 本走はepoch49へ進行中。epoch35/40/45 IoU@.50=.1282/.1206/.1178でepoch10 best=.1694を更新しないが、best-only重みは保持。NaN/Inf/OOM/finite-cost例外0。次のepoch50 validationを監視する。 | 進行中 |
| 2026-08-24 | 19:42:49 UTC | Codex | 手順10・80epoch完走確認 | `rgbd3d_long_ft80_lr1e6`は80epoch/960 train update/16回のcustom_val 50/50を完走。train初回からepoch80 metricまで3,196秒（53分16秒、39.95秒/epoch）、GPU peak=19,326MiB、error marker=0。bestはepoch10 AP50=.0000/IoU@.10/.25/.50/.75=.6300/.4352/.1694/.0215、最終epoch80はAP50=.0010/IoU@.50=.0757。後半品質低下を隠さず、`best_3d_iou_0.50_epoch_10.pth`を保持。 | 完走・bestのみ採用 |
| | | | | |
