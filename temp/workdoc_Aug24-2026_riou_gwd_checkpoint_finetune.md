# 作業計画書 兼 記録書

---

**日付：** `2026年08月24日`  
**作業ディレクトリ・リポジトリ:** `/home/kasm-user/Desktop/YOPO_clone`（branch: `rgb-d`）  
**作業者：** Codex

---

## 1. 作業目的

本日の作業は、以下の目標を達成するために実施します。

* **目標1:** 既存の Tomato 2D OBB GWD 最良 checkpoint を起点に、rotate-IoU（RIoU）損失による追加学習を再現可能な設定として用意する。
* **目標2:** fp16 AMP・約 20 GB VRAM を使う batch size 24・実データ validation/evaluator を保ったまま、RIoU の CUDA/AMP 互換性を安全に検証してから 20 epoch fine-tuning を実行する。
* **目標3:** 実測 rIoU 指標と RGB 上の 2D OBB overlay により、GWD epoch 50 の基準結果との比較を記録する。

### 1.1 ゴール要求分析

* **ユーザーの直観的・直截的な目的:** 描画時に検出が少ないことを単なる閾値の問題と混同せず、現在モデルが本当に OBB を学習しているかを確認し、さらに RIoU を使った追加学習で品質を改善・比較したい。
* **明示要求:** (1) 既存の作業を commit & push する、(2) RIoU で追加学習する、(3) AMP を有効にして 20 GB VRAM を実用的に使う、(4) 実 evaluator を有効にする、(5) 学習済みモデルの RGB/2D OBB overlay を確認する。commit/push は既に `f03ccc5` まで `origin/rgb-d` へ完了しており、本作業では未許可の追加 push を行わない。
* **暗黙制約:** `uv.lock`/`justfile` がある repo-local `.venv` を明示的に使う。暗黙 fallback はしない。GWD baseline の重みは破壊・上書きしない。ユーザー所有の未コミット RGB-D/MAE 系変更は stage/reset/edit しない。DRY/KISS/SOLID、t-wada 的に先に小さく失敗点を検出し、監査可能なコマンド・ログ・数値を残す。
* **非ゴール:** ゼロ初期化の RIoU 50 epoch ベンチマーク、データセット/アノテーションの変更、GWD checkpoint の overwrite、ユーザー変更の commit、推論描画の score threshold 変更だけで性能を主張すること。
* **成功条件:** GWD epoch 50 の model weight を読み、fresh optimizer で RIoU/AMP の 1 epoch smoke が NaN/unsupported-op なしに終了する。その後同一データ・batch 24・rIoU evaluator で 20 epoch fine-tune を終了し、epoch 5 ごとの metric、best checkpoint、baseline 比較、最良 RIoU 重みの overlay を証跡として残す。
* **リスクと前提:** `RotatedIoULoss` が fp16 autocast 下の `box_iou_rotated` を受け付けない可能性がある。問題時は loss 内の IoU 演算だけを明示的に fp32 化してから smoke をやり直し、無断で GWD へ fallback しない。追加学習は `work_dirs/rddetr_tomato_gwd_amp_eval_full/best_rbbox_mAP_50_epoch_50.pth` が存在することを前提とする。ここでは「追加学習」をこの checkpoint 起点の 20 epoch fine-tuning と解釈する。

### 1.2 サブゴール構造

| ID | サブゴール | 目的との対応 | 成果物 | 検証方法 |
| :--- | :--- | :--- | :--- | :--- |
| SG-1 | baseline と RIoU 実装・AMP 境界を監査 | 目標1・制約 | 互換性判断と設計記録 | config 構文読み込み、checkpoint の state_dict 確認 |
| SG-2 | checkpoint 起点の RIoU fine-tune config を追加 | 目標1/2 | `configs/yopo/rotated_deformable_detr_tomato_obb_riou_finetune.py` | `tools/train.py --cfg-options max_epochs=1` の config build |
| SG-3 | AMP smoke と 20 epoch 実訓練 | 目標2 | 専用 work_dir、ログ、weights-only checkpoints | loss finite、CUDA 使用、rIoU evaluator の metric |
| SG-4 | baseline 比較と視覚的検証 | 目標3 | metrics 比較と overlay PNG/manifest | 同一 `RotatedIoUMetric`、画像を目視確認 |
| SG-5 | 記録・引継ぎ | 監査性 | 本書の作業記録 | 全 Trace ID のコマンド、結果、未対応事項を記載 |

### 1.3 トレーサビリティ方針

| Trace ID | 要求・制約 | 対応する作業要素 | 証跡 |
| :--- | :--- | :--- | :--- |
| TR-1 | 追加学習は GWD baseline を起点にする | 手順1–3 | baseline checkpoint path、`load_from`、`resume=False` |
| TR-2 | RIoU を実際に使い、GWD へ隠れた fallback をしない | 手順2–5 | config の `RotatedIoULoss`、smoke/full ログ |
| TR-3 | AMP/20 GB 級 VRAM を有効に使う | 手順2・4・5 | `AmpScheduleFreeOptimWrapper`、batch 24、`nvidia-smi`/ログ |
| TR-4 | 実 evaluator の指標で評価する | 手順2・5・6 | `RotatedIoUMetric`、epoch 5 metric、best checkpoint |
| TR-5 | OBB の見た目を確認する | 手順7 | RGB overlay PNG、manifest、目視結果 |
| TR-6 | 既存作業の push とユーザー変更の保護 | 手順1・8 | `git log`/`git status`、work record |

---

## 2. 作業内容

### フェーズ 1: 調査・実験設計 (見積: 0.3h)

1. **baseline・実装境界の精査:** GWD config、RIoU loss、checkpoint、現 worktree を読み、同じモデル state を読み込めることと AMP 上の危険箇所を確認する（SG-1/TR-1/2/3/6）。
2. **設計の固定:** GWD config をコピー元にし、loss だけ RIoU、fresh optimizer、batch 24、valid 330 images、5 epoch evaluation、weights-only Top-K を固定する。20 epoch は追加学習の短い比較実験であり、ゼロ初期化比較ではない（SG-1/2）。

### フェーズ 2: 設定・小規模検証 (見積: 0.4h)

1. **専用 config の追加:** baseline の runtime/evaluator/checkpoint policy を複製し、RIoU loss と専用 work_dir を明確にする（SG-2/TR-1–4）。
2. **TDD 的 smoke:** `max_epochs=1` で「AMP RIoU の演算が未対応なら失敗する」ことを先に検出する。成功基準は finite loss と checkpoint load の明示ログである（SG-3/TR-2/3）。

### フェーズ 3: 追加学習・評価・可視化 (見積: GPU 実行時間依存)

1. **20 epoch 訓練:** smoke と同じ config で本走を起動し、5 epoch ごとに全 validation で `RotatedIoUMetric` を記録する（SG-3/TR-2–4）。
2. **比較:** baseline（epoch 50: mAP 0.0242 / recall 0.1525 / precision 0.1281 / matched rIoU 0.6031）との違いを数値で記録する。改善しない場合も失敗ではなく事実として扱う（SG-4/TR-4）。
3. **可視化・記録:** 最良 checkpoint の score threshold 0.05、max detections 15 の overlay を生成して、閾値と表示数の意味を併記する。完了定義を監査し、追加 commit/push が必要ならユーザー承認を求める（SG-4/5/TR-5/6）。

---

## 3. 作業チェックリスト

*作業が完了したら `[ ]` を `[x]` に変更し、その直後に必ず「7. 作業記録」へ同時刻の行を追加する。各行は一度に一つだけ完了にする。*

### フェーズ 1: 調査・設計フェーズ

### 手順 1: baseline、現 worktree、RIoU/AMP 境界を確定する
- [x] 🖐 **操作**: `best_rbbox_mAP_50_epoch_50.pth`、GWD/既存 RIoU config、`RotatedIoULoss`、現 `git status` を読み、checkpoint を重みのみで読める設計と保護すべきユーザー変更を記録する。
- [x] 🔎 **確認**: GWD baseline path、既存 baseline 指標、`RotatedIoULoss(mode='log')`、AMP 危険候補、今回対象外の dirty files が本書に明記されている。
- [x] 🧪 **テスト**: `.venv/bin/python -c` で checkpoint の top-level `state_dict` を読み、tensor 数が得られることを確認する。`model` key を要求しない weights-only 形式であることを確認し、失敗時はファイル不存在または checkpoint 形式不一致として止める。
- [x] 🛠 **エラー時対処**: checkpoint が無い/壊れている場合は学習を開始せず、`find work_dirs -name 'best_rbbox_mAP_50_epoch_50.pth'` で候補を再探索して作業記録に残し、解決しなければユーザーへ確認する。

### 手順 2: GWD と同一条件の RIoU fine-tune config を設計する
- [x] 🖐 **操作**: `rotated_deformable_detr_tomato_obb_gwd.py` を基準に、RIoU loss・`load_from`・`resume=False`・20 epoch・batch 24・fp16 AMP・valid/evaluator・Top-K checkpoint の具体値を新 config 用に確定する。
- [x] 🔎 **確認**: 設計差分として、loss 以外の model/data/optimizer/evaluator 条件を baseline と一致させ、optimizer state を復元せず、GWD baseline work_dir を使わないことを確認する。実 config dump による検証は手順3で実施する。
- [x] 🧪 **テスト**: 設計時の検出条件を「half input の RIoU backward が NaN/未対応なら原因を記録し、有限なら smoke へ進める」と固定する。実行結果は half input gradient が NaN となり、fp32 化が必要と確定した。
- [x] 🛠 **エラー時対処**: memory が 20 GB を超える場合は batch を勝手に下げず、GPU 使用量を記録してユーザーに選択を求める。RIoU fp16 非対応時は IoU 演算だけを fp32 にする最小修正を採用し、GWD への切替はしない。

### フェーズ 2: 設定・小規模検証

### 手順 3: 再現可能な RIoU fine-tune config を追加して構文検証する
- [x] 🖐 **操作**: `configs/yopo/rotated_deformable_detr_tomato_obb_riou_finetune.py` を追加し、手順2の値と実行コマンドを記載する。
- [x] 🔎 **確認**: `loss_iou.type='RotatedIoULoss'`、`max_epochs=20`、`AmpScheduleFreeOptimWrapper`、batch 24、`val_interval=5`、`RotatedIoUMetric`、専用 `load_from`、`resume=False` が config dump に存在する。
- [x] 🧪 **テスト**: `uv run python tools/train.py <config> --work-dir /tmp/rddetr_riou_cfgcheck --cfg-options train_cfg.max_epochs=0` を実行し、config build の成否を記録する。初回は lazy import 下の `deepcopy` が `RuntimeError` となったため、次項で `_base_` 継承に直して再検証する。
- [x] 🛠 **エラー時対処**: import/registry error は fully qualified custom optimizer path、loss export、config import を確認する。`uv run` が repo-local env を使えない場合は `just env-doctor` で理由を明示し、`.venv/bin/python` を選んだ証跡を残す。

### 手順 4: checkpoint 起点の 1 epoch AMP RIoU smoke を実行する
- [x] 🖐 **操作**: 新 config を `work_dirs/rddetr_tomato_riou_amp_smoke` に `train_cfg.max_epochs=1` で起動し、開始時の GPU usage と stdout/stderr を保存する。
- [x] 🔎 **確認**: log に `RotatedIoULoss`、checkpoint load、AMP wrapper、CUDA、finite な `loss`/`loss_iou` があり、OOM/NaN/未対応 half op なしで epoch 1 が完了する。
- [x] 🧪 **テスト**: `riou_amp_1epoch_smoke` として、実装前の「AMP RIoU 演算未検証」状態から、終了 code 0 と単語境界付き `rg` による NaN/Inf/未対応演算非検出へ変わることを確認する。
- [x] 🛠 **エラー時対処**: `box_iou_rotated` の fp16 エラーは loss 内の入力を `float()` にし autocast を局所無効化して修正・再 smoke する。OOM は nvidia-smi の実測を添え、batch 変更前にユーザーへ確認する。

### 手順 4a: warm-start RIoU の学習率を保護して再 smoke する
- [x] 🖐 **操作**: epoch 50 GWD 重みを壊した `muon_lr=0.005` / `sf_lr=0.00025` を warm-start 用に 50 分の 1（`0.0001` / `0.000005`）へ下げ、batch 24・AMP・evaluator を変えずに config と workdir を分離する。
- [x] 🔎 **確認**: 低学習率 config は GWD baseline weight・`resume=False`・RIoU loss・batch 24・fp16 AMP・5 epoch evaluator を保ち、変更箇所が optimizer の二つの LR と専用 workdir 名だけである。
- [x] 🧪 **テスト**: `riou_warmstart_lr_smoke` として 1 epoch を実行し、mAP/recall が高 LR smoke の 0.0000/0.0001 より少なくとも悪化せず、NaN/Inf/OOM なしで終わることを確認する。baseline を超えることはこの 1 epoch の必須条件にしない。
- [x] 🛠 **エラー時対処**: 低学習率でも mAP が 0 のままなら 20 epoch を開始せず、最適化器の fresh-start 挙動・loss scale・RIoU loss weight を分離して検証し、ユーザーへ選択肢を提示する。

### フェーズ 3: 追加学習・評価・可視化

### 手順 5: 20 epoch の RIoU fine-tuning を完走する
- [x] 🖐 **操作**: smoke 成功後、新 config を `work_dirs/rddetr_tomato_riou_gwd_ft20` に `max_epochs=20` で一度だけ起動し、5 epoch ごとの validation と Top-K weights-only checkpoint を保存する。
- [x] 🔎 **確認**: 初回本走を監査し、epoch 5/10/15/20 metrics が得られる前の epoch 2/20・20/53 で `grad_norm: nan` を検出したこと、loss は有限でも 20 epoch/Top-K 指標条件は未達であることを確認する。
- [x] 🧪 **テスト**: 初回 `riou_finetune_20epoch_e2e` を実行し、epoch 2 で `grad_norm: nan` を検出して 20 epoch 未到達だったため失敗として記録する。linear RIoU 回復後に別手順で同じ成功条件を再検証する。
- [x] 🛠 **エラー時対処**: 中断/クラッシュ時は work_dir と最後の epoch を記録し、原因を切り分けるまで `resume=True` にしない。checkpoint が weights-only のため、再実行時は手順と optimizer fresh の意味を明示する。

### 手順 5a: linear rotate-IoU の 2 epoch 安定性を検証する
- [x] 🖐 **操作**: RIoU を維持したまま `mode='log'` から有界な `mode='linear'` に切り替え、GWD epoch 50 weight・low warm-start LR・batch 24・fp16 AMP を保った 2 epoch run を `work_dirs/rddetr_tomato_riou_linear_2epoch_smoke` に起動する。
- [x] 🔎 **確認**: config dump が `RotatedIoULoss(mode='linear')`、`muon_lr=0.0001`、`sf_lr=0.000005`、`resume=False`、batch 24、実 rIoU evaluator を示す。
- [x] 🧪 **テスト**: `riou_linear_2epoch_stability` として epoch 2 の 53/53 iter を通過し、`grad_norm`、loss、loss_iou が finite、final validation 42/42 と exit 0 を確認する。log RIoU で NaN が出た epoch 2 を越えることを必須とする。
- [x] 🛠 **エラー時対処**: linear RIoU でも非有限 gradient が出たら、full run を開始せず、異常 batch/box を記録して RIoU gradient guard を明示実装する。loss 種別を GWD へ暗黙に戻さない。

### 手順 5b: 安定な linear rotate-IoU で 20 epoch 本走を完了する
- [x] 🖐 **操作**: 手順5a成功後、GWD epoch 50 weight を毎回初期値として `work_dirs/rddetr_tomato_riou_linear_ft20` に 20 epoch を一度だけ起動し、5 epoch ごとの evaluator と Top-K weights-only checkpoint を保存する。
- [x] 🔎 **確認**: epoch 5/10/15/20 の rIoU metrics、finite gradient/loss、best checkpoint、VRAM 実測が log に残る。
- [x] 🧪 **テスト**: `riou_linear_20epoch_e2e` として exit 0、20 epoch 到達、各 validation 330 images 完走、NaN/Inf/OOM/非有限 grad norm 非検出を確認する。
- [x] 🛠 **エラー時対処**: 中断・非有限値時は tmux session を停止し、last epoch と work_dir を記録する。`resume=True` にはせず、GWD baseline weights から変更を分離して再実行する。

### 手順 6: baseline と RIoU 追加学習を実測値で比較する
- [x] 🖐 **操作**: 最良 RIoU checkpoint の `rbbox_mAP_50`、recall、precision、matched rIoU を抽出し、GWD epoch 50 baseline（0.0242/0.1525/0.1281/0.6031）との差を本書に記録する。
- [x] 🔎 **確認**: 同一 evaluator、score threshold 0.05、IoU threshold 0.5 という比較前提と、改善/悪化/不確実性が数値で明記されている。
- [x] 🧪 **テスト**: `metrics_traceability` として run log、checkpoint 名、metric 行が相互に対応し、手入力の数値が `rg` 抽出結果と一致することを確認する。
- [x] 🛠 **エラー時対処**: best checkpoint が無い場合は final checkpoint を best と偽らず、評価可能な epoch と Top-K hook の出力を記録して原因を報告する。

### 手順 7: 最良 RIoU checkpoint の RGB/2D OBB overlay を生成・目視確認する
- [x] 🖐 **操作**: `tools/analysis_tools/visualize_rotated_obb_predictions.py` を最良 RIoU checkpoint に対し `--score-thr 0.05 --max-dets 15` で実行し、3 枚と manifest を専用 directory へ出力する。
- [x] 🔎 **確認**: PNG に RGB、角度付き OBB、score が重なり、各画像の予測数が manifest に残る。0.05 は低い描画閾値、15 は表示上限であることを記録する。
- [x] 🧪 **テスト**: `riou_overlay_artifact` として PNG 3 枚、manifest、画像サイズを列挙し、画像を実際に目視して座標/回転の取り違えがないことを確認する。
- [x] 🛠 **エラー時対処**: prediction key/shape error は checkpoint と config の model shape を確認する。検出が少ない場合は score と max-dets を先に報告し、閾値変更だけを性能改善として扱わない。

### 手順 8: 完了定義・worktree・引継ぎを監査する
- [x] 🖐 **操作**: 全チェックと `git status --short`、ログ/overlay/checkpoint の存在を確認し、本書の作業記録へ結論・未対応事項・追加 commit/push の承認要否を記載する。
- [x] 🔎 **確認**: ユーザー所有の dirty files が stage/reset/編集されておらず、全 Trace ID に証跡がある。新たな commit/push はユーザー承認なしに実施していない。
- [x] 🧪 **テスト**: `workdoc_pre_dod_audit` として、未完了が本テスト・後続のエラー時対処・完了定義4観点・最終監査だけの既知7項目であることを確認し、全体の `rg -n '^[-*] \[ \]' <workdoc>` 空確認は完了定義後の最終監査へ委譲する。
- [x] 🛠 **エラー時対処**: 未完了チェック、生成物欠落、または未解決 error があれば完了にせず、該当手順を細分化して新しい四項目を追加し、理由と再開条件を記録する。

---

## 4. 作業に使用するコマンド参考情報

この repository は `pyproject.toml`/`uv.lock`/`justfile` を持つ。通常は次を使う。既存の repo-local `.venv` を使う必要がある training command では、`just env-doctor` で同一環境であることを確認した上で明示的に `.venv/bin/python` を使う。これは fallback ではなく、`justfile` が採用している実行形式である。

```bash
# 依存関係と GPU 環境の確認
uv sync
just env-doctor

# config/RIoU 1 epoch smoke（ログは tee で保存する）
.venv/bin/python tools/train.py \
  configs/yopo/rotated_deformable_detr_tomato_obb_riou_finetune.py \
  --work-dir work_dirs/rddetr_tomato_riou_amp_smoke \
  --cfg-options train_cfg.max_epochs=1

# 20 epoch fine-tune
.venv/bin/python tools/train.py \
  configs/yopo/rotated_deformable_detr_tomato_obb_riou_finetune.py \
  --work-dir work_dirs/rddetr_tomato_riou_gwd_ft20

# 最良重みによる RGB overlay（score 0.05、最大15表示）
.venv/bin/python tools/analysis_tools/visualize_rotated_obb_predictions.py \
  configs/yopo/rotated_deformable_detr_tomato_obb_riou_finetune.py \
  work_dirs/rddetr_tomato_riou_gwd_ft20/<best-checkpoint>.pth \
  --output-dir work_dirs/rddetr_tomato_riou_gwd_ft20/obb_overlays \
  --score-thr 0.05 --max-dets 15 --num-images 3
```

監視には `nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader`、完了監査には `rg -n '^[-*] \[ \]' temp/workdoc_Aug24-2026_riou_gwd_checkpoint_finetune.md` を用いる。

---

## 6. 完了の定義

*作業が最後まで完了したら `[ ]` を `[x]` にしつつ、作業が本当に完了したかをチェックする。*

- [x] 観点1: GWD epoch 50 checkpoint 起点の RIoU 20 epoch fine-tuning が、明示的な AMP 条件で完走している。
- [x] 観点2: `RotatedIoUMetric` の epoch 別数値・最良 checkpoint・GWD baseline 比較が作業記録に残っている。
- [x] 観点3: 最良 RIoU 重みによる RGB/2D OBB overlay 3 枚と manifest を実際に確認している。
- [x] 観点4: dirty worktree を保護し、暗黙 fallback/未許可 commit・push をせず、全 Trace ID の証跡と未対応事項を記録している。

### 6.1 最終完了監査

- [x] 🧪 **テスト**: `workdoc_dod_precheck` として、自身だけが未完了であることを確認する。自身を `[x]` に更新して直後に作業記録を追記した後、postcondition として `rg -n '^[-*] \[ \]' <workdoc>` が空であること、および完了定義と成果物パスの一致を確認する。

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
| 2026-08-24 | 11:43:49 UTC | Codex | 作業開始・時刻記録 | `date '+%Y-%m-%d %H:%M:%S %Z%z'` を実行。既存 push は `f03ccc5` / `origin/rgb-d` と確認。 |
| 2026-08-24 | 11:43:49 UTC | Codex | 作業書作成前調査 | `uv.lock`、`justfile`、`pyproject.toml`、GWD/RIoU config、RIoU loss を確認。対象外の RGB-D/MAE dirty files を保護する方針を固定。 |
| 2026-08-24 | 11:43:49 UTC | Codex | 本作業書を作成 | 追加学習を「GWD epoch 50 weight 起点・fresh optimizer・RIoU 20 epoch」と明示。まだ checklist は未着手。 |
| 2026-08-24 | 11:46:26 UTC | Codex | 手順1・操作完了: baseline/worktree 調査 | checkpoint は 70 MB、`state_dict` 560 tensors、optimizer key なし。`load_from` + `resume=False` による fresh optimizer に適合。既存 RGB-D/MAE dirty files は対象外として保持。 |
| 2026-08-24 | 11:47:34 UTC | Codex | 手順1・確認完了: baseline/AMP 境界 | GWD epoch 50 validation は mAP 0.0242、recall 0.1525、precision 0.1281、matched rIoU 0.6031。RIoU は `diff_iou_rotated_2d` を autocast 無効化して呼ぶが input の `float()` は明示しないため、AMP smoke で検証する。 |
| 2026-08-24 | 11:47:34 UTC | Codex | 手順1・テスト期待値の補正 | 実 checkpoint は top-level `state_dict`（560 tensors）で `model` key は持たない正常な weights-only 形式だった。誤った `model` key 要求を補正し、同じ読み込み検証を再実行する。 |
| 2026-08-24 | 11:48:21 UTC | Codex | 手順1・テスト完了: weights-only schema | `.venv/bin/python` で `state_dict` の存在、560 tensors、optimizer key 不在を assert。`PASS weights_only_schema` を確認。 |
| 2026-08-24 | 11:48:39 UTC | Codex | 手順1・エラー時対処確認完了 | `find work_dirs -name 'best_rbbox_mAP_50_epoch_50.pth'` は期待 path 1 件のみを返した。欠落/形式不一致がないため学習停止条件は発動しない。 |
| 2026-08-24 | 11:49:06 UTC | Codex | 手順2・操作完了: RIoU FT 設計 | GPU は 20,475 MiB 中 2 MiB 使用。GWD と model/data/optimizer/evaluator を一致させ、`RotatedIoULoss(mode='log', loss_weight=2.0)`、batch 24、fp16 AMP、20 epoch、5 epoch validation、Top-K weights-only、`load_from` baseline、`resume=False`、専用 work_dir を固定。 |
| 2026-08-24 | 11:49:41 UTC | Codex | 手順2・確認完了: 設計差分 | 既存 RIoU config の batch 8・非 AMP・評価なしとの差分を確認。新 config では model/data/optimizer/evaluator は GWD baseline と揃え、意図的な差分を RIoU loss、20 epoch、baseline `load_from`、専用 work_dir に限定する。 |
| 2026-08-24 | 11:50:50 UTC | Codex | 手順2・テスト完了: AMP half RIoU 境界 | fp32 input の RIoU loss/backward は finite だったが、half input では `diff_iou_rotated_2d` 後の gradient が全 NaN。autocast 無効化だけでは input dtype を変えないことを実証し、loss 内だけの fp32 化を次項で適用する。 |
| 2026-08-24 | 11:51:30 UTC | Codex | 手順2・エラー時対処完了: RIoU fp32 geometry | `yopo/models/losses/rotated_iou_loss.py` で geometry kernel 入力を `pred.float()`/`target.float()` に限定。half input unit test は loss 0.23850466、finite fp16 gradient で成功。GWD への fallback は行っていない。 |
| 2026-08-24 | 11:52:18 UTC | Codex | 手順3・操作完了: RIoU FT config 追加 | `configs/yopo/rotated_deformable_detr_tomato_obb_riou_finetune.py` を追加。GWD config を `read_base()` で継承し、RIoU loss、20 epoch/16 epoch decay、weights-only `load_from`、`resume=False` のみを上書き。 |
| 2026-08-24 | 11:53:08 UTC | Codex | 手順3・テスト完了: 初回 config build | `uv run python tools/train.py ... max_epochs=0` を実行。`read_base()` が lazy import となり `deepcopy(model)` で `RuntimeError`。training は開始しておらず、設定記法を `_base_` の部分上書きに変更して再検証する。 |
| 2026-08-24 | 11:57:36 UTC | Codex | 手順3・確認完了: config dump | `Config.fromfile` で RIoU log loss、20 epoch、AmpScheduleFreeOptimWrapper、batch 24、val interval 5、RotatedIoUMetric、baseline `load_from`、`resume=False`、Top-K weights-only を確認。0 epoch runner build も exit 0。 |
| 2026-08-24 | 11:58:05 UTC | Codex | 手順3・エラー時対処完了: config 形式 | `read_base()` lazy import と `_base_` 通常継承の混在を廃止し、GWD runtime を明示した self-contained config に置換。`uv run` の runner build と `py_compile` が成功。 |
| 2026-08-24 | 12:01:08 UTC | Codex | 手順4・操作完了: 1 epoch AMP RIoU smoke | `work_dirs/rddetr_tomato_riou_amp_smoke` に batch 24/fp16 AMP で起動。実測 GPU は最大 19,414 MiB / 20,475 MiB。53 train iter と 42 validation iter、weights-only checkpoint 保存まで exit 0。 |
| 2026-08-24 | 12:01:50 UTC | Codex | 手順4・確認完了: AMP RIoU smoke | `/tmp/opencode/riou_amp_smoke.log` に GWD weight load、epoch 1/53 iter、validation 42/42、`train_exit=0` を確認。単語境界 scan で NaN/Inf/OOM/未対応 op は不在。評価は mAP 0.0000、matched rIoU 0.5210。 |
| 2026-08-24 | 12:02:06 UTC | Codex | 手順4・テスト完了: riou_amp_1epoch_smoke | 実装前の half gradient NaN から、fp32 geometry 修正後の 1 epoch end-to-end exit 0 へ移行。`INFO` を誤検知しない単語境界付き scan で数値 NaN/Inf/未対応演算なしを確認。 |
| 2026-08-24 | 12:02:26 UTC | Codex | 手順4・エラー時対処完了: AMP RIoU | `RotatedIoULoss` 内の geometry input fp32 化により half-gradient NaN を解消。peak VRAM 19,414 MiB で OOM はなく、batch は 24 を維持。mAP 急落は演算エラーと混同せず、手順4aで最適化強度を独立に検証する。 |
| 2026-08-24 | 12:02:59 UTC | Codex | 手順4a・操作完了: warm-start LR | fine-tune config の `muon_lr` を 0.005→0.0001、`sf_lr` を 0.00025→0.000005 に変更。high-LR smoke artifact は保持し、次の run は `rddetr_tomato_riou_warmstart_smoke` へ分離する。 |
| 2026-08-24 | 12:03:57 UTC | Codex | 手順4a・確認完了: warm-start config | config object で lr 0.0001/0.000005、batch 24、AMP wrapper、RotatedIoULoss、RotatedIoUMetric、GWD epoch 50 `load_from`、`resume=False` を assert。型は config parse 後に `RotatedIoULoss` class となるため class name で判定。 |
| 2026-08-24 | 12:06:42 UTC | Codex | 手順4a・テスト完了: riou_warmstart_lr_smoke | 1 epoch/53 train iter + 42 validation iter、exit 0。mAP 0.0277、recall 0.1635、precision 0.1374、matched rIoU 0.6051。high-LR smoke（0.0000/0.0001）より改善し、GWD baseline（0.0242/0.1525/0.1281/0.6031）もこの時点で上回る。VRAM peak は 19,424 MiB、NaN/Inf/OOM なし。 |
| 2026-08-24 | 12:07:06 UTC | Codex | 手順4a・エラー時対処確認完了 | warm-start run の mAP は 0.0277 で 0 ではなく、数値 NaN/Inf/OOM/未対応 op scan も空。optimizer/loss scale/loss weight の追加分離は不要と判断し、低 LR config で 20 epoch 本走へ進む。 |
| 2026-08-24 | 12:08:13 UTC | Codex | 手順5・操作完了: 20 epoch 本走起動 | `nohup` が実行環境の終了時に継続しないため訓練開始なしと判明し、tmux session `riou_gwd_ft20` で再起動。PID 1566341、専用 work_dir/log、low-LR config、batch 24/fp16 AMP、5 epoch validation を確認。 |
| 2026-08-24 | 12:09:40 UTC | Codex | 手順5・定期状況（行動カウント reset） | tmux 本走は epoch 1/20・20/53 iter まで正常。`loss_bbox=0.6528`、`loss_iou=9.4451`、GPU 19,406 MiB / 20,475 MiB（100% utilization）。NaN/OOM は未検出。 |
| 2026-08-24 | 12:10:53 UTC | Codex | 手順5・異常検出と安全停止 | epoch 2/20・20/53 で `grad_norm: nan`、30/53 でも再現（loss は有限）。NaN gradient 更新を防ぐため tmux session を停止し GPU 解放を確認。GWD baseline/low-LR smoke は変更しない。`mode='log'` RIoU の勾配不安定性を分離するため、linear RIoU の 2 epoch 安定性手順を追加する。 |
| 2026-08-24 | 12:11:59 UTC | Codex | 手順5・確認完了: 初回本走の停止条件 | `rddetr_tomato_riou_gwd_ft20` は epoch 1 を通過したが epoch 2/20 iter から `grad_norm: nan`、30 iter でも継続。loss は finite でも 20 epoch/5 epoch metrics は未達のため、tmux 停止が正しい安全条件と確認。 |
| 2026-08-24 | 12:12:15 UTC | Codex | 手順5・テスト完了: 初回 E2E 失敗 | training process は tmux kill により安全停止。20 epoch 到達/5 epoch evaluator/NaN非検出の成功条件を満たさず、`mode='log'` RIoU の不安定性を修正せずに resume しないことを確定。 |
| 2026-08-24 | 12:12:56 UTC | Codex | 手順5・エラー時対処完了: log RIoU 安全停止 | interrupted work_dir は `rddetr_tomato_riou_gwd_ft20`、最後の完全 epoch は 1、epoch 2/30 iter で stop。weights-only checkpoint のため `resume=True` は使わず、GWD epoch 50 weight から fresh optimizer で linear RIoU を検証する手順5a/5bを追加。 |
| 2026-08-24 | 12:13:29 UTC | Codex | 手順5a・操作完了: linear RIoU | fine-tune config の `RotatedIoULoss` を `mode='log'` から `mode='linear'` へ変更。RIoU objective、loss weight 2.0、muon/schedulefree low LR、batch 24、fp16 AMP、GWD weight 起点・fresh optimizer は維持。 |
| 2026-08-24 | 12:13:55 UTC | Codex | 手順5a・確認完了: linear config | config object で `RotatedIoULoss(mode='linear')`、lr 0.0001/0.000005、batch 24、float16 AMP、RotatedIoUMetric、GWD baseline weight、`resume=False` を assert。 |
| 2026-08-24 | 12:14:21 UTC | Codex | 手順5a・操作完了: linear 2 epoch 起動 | tmux session `riou_linear_2epoch` で 2 epoch 追加学習を起動。work_dir/log は `rddetr_tomato_riou_linear_2epoch_smoke` / `/tmp/opencode/riou_linear_2epoch.log`、fresh GWD baseline weight、batch 24/fp16 AMP。 |
| 2026-08-24 | 12:17:59 UTC | Codex | 手順5a・テスト完了: linear 2 epoch 安定性 | epoch 2/53 iter を finite grad norm 34.9–38.1 で完走、42/42 validation、exit 0。mAP 0.0280、recall 0.1640、precision 0.1378、matched rIoU 0.6054。log RIoU が grad_norm NaN を出した epoch 2 を安全に越えた。 |
| 2026-08-24 | 12:18:19 UTC | Codex | 手順5a・エラー時対処確認完了 | `grad_norm: nan/inf`、numeric NaN/Inf、OOM、未対応 op は scan で不在。linear RIoU の非有限時用 gradient guard は不要で、GWD への fallback なしに手順5b 20 epoch 本走を開始できる。 |
| 2026-08-24 | 12:18:47 UTC | Codex | 手順5b・操作完了: linear 20 epoch 本走起動 | tmux session `riou_linear_ft20`（PID 1576792）で `rddetr_tomato_riou_linear_ft20` を開始。GWD epoch 50 weights + fresh optimizer、linear RIoU、low LR、batch 24/fp16 AMP、5 epoch real evaluator、Top-K weights-only を使用。 |
| 2026-08-24 | 12:21:39 UTC | Codex | 手順5b・定期状況（行動カウント reset） | linear 本走は epoch 2/20・30/53 iter。`grad_norm=1084.3580` は大きいが有限、`loss=9.3952`、`loss_iou=1.4898`、VRAM 約19.4 GB。log RIoU で発生した NaN は再発していない。 |
| 2026-08-24 | 12:43:16 UTC | Codex | 手順5b・定期状況（行動カウント reset） | linear 本走は epoch 15/20 の実 evaluator まで完了し、NaN/Inf/OOM は未検出。mAP@0.5 は epoch 5/10/15 で 0.0299/0.0305/0.0326、recall は 0.1682/0.1700/0.1756、precision は 0.1413/0.1428/0.1475、matched rIoU は 0.6068/0.6063/0.6107。最高 mAP は GWD baseline 0.0242 を 0.0084 上回る。GPU は 19,432/20,475 MiB を使用し、epoch 20 の完走を継続監視する。 |
| 2026-08-24 | 12:49:19 UTC | Codex | 手順5b・確認完了: linear 20 epoch metrics | `/tmp/opencode/riou_linear_ft20.log` で epoch 5/10/15/20 の 42/42 evaluator を確認。mAP@0.5=0.0299/0.0305/0.0326/0.0338、recall=0.1682/0.1700/0.1756/0.1786、precision=0.1413/0.1428/0.1475/0.1500、matched rIoU=0.6068/0.6063/0.6107/0.6095。全 train log の `grad_norm` は有限で、VRAM は 19,432/20,475 MiB。最良重みは `best_rbbox_mAP_50_epoch_20.pth`。 |
| 2026-08-24 | 12:49:19 UTC | Codex | 手順5b・テスト完了: riou_linear_20epoch_e2e | `train_exit=0`、epoch 5/10/15/20 の各 `Epoch(val) [N][42/42]`、NaN/Inf/OOM marker 不在、best checkpoint 非空を確認。`yopo.registry.DATASETS` で val dataset=330 images、batch size=8、期待 42 iter を検証し、全 validation が対象全件を処理したことを確定。 |
| 2026-08-24 | 12:49:19 UTC | Codex | 手順5b・エラー時対処確認完了 | `tmux has-session -t riou_linear_ft20` は closed、最終 log は epoch 20 validation・Top-K 保存・`train_exit=0`。GPU は 2/20,475 MiB へ解放済み。中断/非有限値が無いため stop/resume 分岐は発動せず、GWD baseline 起点・fresh optimizer の分離を保った。 |
| 2026-08-24 | 12:49:19 UTC | Codex | 手順6・操作完了: baseline 比較抽出 | baseline `/tmp/opencode/gwd_amp_eval_full.log` epoch 50 と RIoU `/tmp/opencode/riou_linear_ft20.log` epoch 20 はともに 42/42 evaluator・`rbbox_num_gt=27723`。RIoU は GWD に対し mAP 0.0242→0.0338（+0.0096, +39.7%）、recall 0.1525→0.1786（+0.0261）、precision 0.1281→0.1500（+0.0219）、matched rIoU 0.6031→0.6095（+0.0064）。 |
| 2026-08-24 | 12:54:29 UTC | Codex | 手順6・確認完了: evaluator/threshold 切り分け | GWD/RIoU config はともに `RotatedIoUMetric(iou_thr=0.5, score_thr=0.05, num_classes=1)`、42 iter/330 images、GT=27,723 で一致。RIoU best checkpoint を `test_evaluator.score_thr=0.01/0.05/0.10` で再評価し、mAP/recall は全て 0.0338/0.1786（precision は 0.1500/0.1500/0.1506）。従って描画用や evaluator の conf 閾値の取り違えは mAP 低値の原因ではない。絶対性能は低い一方、GWD 比では全指標が改善している。 |
| 2026-08-24 | 12:54:29 UTC | Codex | 手順6・テスト完了: metrics_traceability | parser assert で baseline final metric `0.0242/0.1525/0.1281/0.6031`、RIoU final metric `0.0338/0.1786/0.1500/0.6095`、`train_exit=0`、log 内の `best_rbbox_mAP_50_epoch_20.pth`、非空 checkpoint（72,504,338 bytes）が一致。手入力比較値との差異なし。 |
| 2026-08-24 | 12:55:23 UTC | Codex | 定期状況（行動カウント reset）: mAP 低値の診断 | RIoU best を score_thr 0.01/0.05/0.10 で同一 IoU=0.5 再評価し、mAP/recall は全て 0.0338/0.1786（precision 0.1500/0.1500/0.1506）。conf threshold の取り違えは否定。val GT は 27,723 個/330 枚（平均84、最大214）、`num_queries=100` 超過画像は95枚で、query-only の理論 recall 上限は0.9027。query 数は次段階の改善候補だが、現 0.1786 recall の単独原因ではない。 |
| 2026-08-24 | 12:55:23 UTC | Codex | 手順6・エラー時対処確認完了 | best checkpoint は `best_rbbox_mAP_50_epoch_20.pth`（72,504,338 bytes）として存在し、epoch 20 の `0.0338` 保存ログとも一致。欠落時の final checkpoint 偽装/Top-K 代替分岐は発動しない。 |
| 2026-08-24 | 12:57:05 UTC | Codex | 手順7・操作完了: RIoU RGB/2D OBB overlay | 最良 `best_rbbox_mAP_50_epoch_20.pth` に `score_thr=0.05`、`max_dets=15` を適用し、`work_dirs/rddetr_tomato_riou_linear_ft20/obb_overlays/` に PNG 3枚と `manifest.json` を生成。各 PNG は 15 predictions を描画し、manifest は config/checkpoint/入力画像/score/xywha-radian を保持。 |
| 2026-08-24 | 12:57:05 UTC | Codex | 手順7・確認完了: manifest/描画閾値 | manifest assert により `score_threshold=0.05`、`max_dets_per_image=15`、PNG 3枚、各 `num_drawn=15`、全描画 score≥0.05 を確認（score range: 0.759–0.879 / 0.864–0.937 / 0.642–0.876）。0.05 は低い描画下限、15 は PNG の可読性用の上限で、`RotatedIoUMetric` の全検出 AP 計算とは独立。 |
| 2026-08-24 | 12:57:58 UTC | Codex | 手順7・テスト完了: riou_overlay_artifact | PNG 3枚（809,065 / 801,647 / 812,090 bytes）と manifest（7,543 bytes）を列挙し、3枚を実画像で目視。RGB 上で果実群に OBB 中心・回転・score が概ね整合し、座標系/角度単位の取り違えは見えない。密集画像では text が重なる／上端で clip するが、表示上限15件の可読性問題であり、評価用検出の欠落指標ではない。 |
| 2026-08-24 | 12:58:45 UTC | Codex | 手順7・エラー時対処確認完了 | config/checkpoint shape error は無く可視化は exit 0。生出力は 3枚とも `raw=100`、score≥0.05/0.10 も各100で、PNG の15件は `max_dets=15` の表示制限。対応 GT は 76/112/55（2枚目のみ query 上限100を超過）。閾値変更だけを性能改善と扱わず、低 mAP は conf cutoff ではなく過信した誤検出/位置精度・密集時 query 上限を含む検出品質課題と記録。 |
| 2026-08-24 | 12:59:22 UTC | Codex | 手順8・操作完了: artifact/worktree 監査 | GWD baseline/RIOU best checkpoint、両 log、overlay manifest は全て nonempty。HEAD/origin は prior push `f03ccc5` で一致し、本作業中の追加 commit/push は無し。今回の未 commit 変更は `rotated_iou_loss.py`（AMP geometry fp32化）、RIoU config、作業書。RGB-D/MAE の既存 dirty files（backbones/detectors 初期化・dual/MAE config/model）は対象外として保護。未対応は次ユーザー要求の RGB-D/3D BBOX 作業であり、当該新規変更の commit/push は明示承認待ち。 |
| 2026-08-24 | 12:59:22 UTC | Codex | 手順8・確認完了: dirty worktree/Trace ID | `git diff --cached --name-only` は空。tracked dirty は既存 `backbones/__init__.py`・`detectors/__init__.py` と本作業の `rotated_iou_loss.py`、untracked には既存 RGB-D/MAE と今回 config/workdoc が共存する状態を保全。作業書の Trace table/手順は baseline/RIoU log、work_dir、checkpoint、overlay manifest へ参照を持つ。HEAD=origin/rgb-d の `f03ccc5` で、本作業の無断 commit/push は無し。 |
| 2026-08-24 | 12:59:22 UTC | Codex | 手順8・作業書細分化: 完了監査の循環解消 | 元の `workdoc_dod_audit` は、手順8の後続項目と完了定義が未完了のまま文書全体の `[ ]` 空を要求しており循環。`workdoc_pre_dod_audit`（残存未完了が既知6項目だけ）と、完了定義後の「6.1 最終完了監査」（全体空確認）へ分離。実装・成果物には変更なし。 |
| 2026-08-24 | 12:59:22 UTC | Codex | 手順8・テスト完了: workdoc_pre_dod_audit | `rg '^[-*] \[ \]'` は細分化後に7件のみ（本テスト、後続エラー時対処、完了定義4観点、6.1最終監査）を返し、他の実装チェックは完了。最終の文書全体空確認は完了定義の全観点をチェック後に実行する。 |
| 2026-08-24 | 12:59:22 UTC | Codex | 手順8・エラー時対処確認完了 | artifact 欠落・未解決 runtime error・予定外の未完了項目は無し。文書内の最終監査循環だけは先行して細分化し、残る4観点と最終監査を順に実行する再開条件を明確化したため、未完了を隠して完了扱いにはしない。 |
| 2026-08-24 | 13:01:38 UTC | Codex | 完了定義・観点1確認 | RIoU log/config に GWD epoch 50 `load_from`、AMP `dtype='float16'`、epoch 20 `42/42` validation、`train_exit=0` を確認。linear RIoU 20 epoch fine-tuning は明示 AMP 条件で完走。 |
| 2026-08-24 | 13:01:38 UTC | Codex | 完了定義・観点2確認 | 作業記録に epoch 5/10/15/20 metrics、best `best_rbbox_mAP_50_epoch_20.pth`、GWD baseline と +0.0096 mAP の比較、threshold sensitivity/traceability が記録済み。checkpoint 実体も nonempty。 |
| 2026-08-24 | 13:01:38 UTC | Codex | 完了定義・観点3確認 | `obb_overlays` の PNG 3枚（809,065 / 801,647 / 812,090 bytes）と manifest（7,543 bytes）を確認済み。各15件描画の manifest assert と3枚の目視で、RGB/OBB/score の整合を確認。 |
| 2026-08-24 | 13:01:38 UTC | Codex | 完了定義・観点4確認 | current HEAD/origin は prior push `f03ccc5` のまま。config/log は `mode=linear`、GWD `load_from`、`resume=False`、`train_exit=0` を示し、暗黙 GWD fallback は無し。既存 RGB-D/MAE dirty files を stage/reset/編集せず、Trace/未対応（次の RGB-D/3D BBOX 作業）を記録済み。 |
| 2026-08-24 | 13:02:58 UTC | Codex | 定期状況（行動カウント reset）: 最終監査の自己参照解消 | 完了定義4観点後の `rg` は未完了1件（最終監査自身）だけを返した。最終監査も自分を含めて空を要求する自己参照だったため、`workdoc_dod_precheck`（自身のみ未完了）→ `[x]` 更新直後の postcondition 全体空確認に細分化。成果物・コードへの変更なし。 |
| 2026-08-24 | 13:02:58 UTC | Codex | 6.1 最終監査・チェック更新 | `workdoc_dod_precheck` は、自身だけが未完了であることを確認して成功。全実装手順と完了定義4観点は `[x]`。本行の直後に postcondition の全体 `[ ]` 空確認を実行する。 |
| 2026-08-24 | 13:02:58 UTC | Codex | 6.1 最終監査・postcondition 成功 | `rg -n '^[-*] \[ \]' workdoc` は空で、workdoc の全 checklist/完了定義は `[x]`。最良 RIoU checkpoint、manifest、overlay PNG 3枚は全て nonempty。成果物・metrics・worktree 保護の証跡が作業記録と一致するため、本 RIoU 追加学習作業を完了。 |
| | | | | |
