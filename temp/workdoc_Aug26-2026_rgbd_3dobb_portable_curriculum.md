# 作業計画書 兼 記録書: portable RGB-D 3D OBB curriculum

---

**日付：** 2026年08月26日
**作業ディレクトリ・リポジトリ:** `/home/kasm-user/Desktop/YOPO_clone` (`rgb-d`, uv)
**作業者：** Codex

---

## 1. 作業目的

2D OBB detectorのplateau確認後、検証済みportable RGB-Dデータ1526 frameを使い、
2D featureを保持したまま`z -> size -> rotation/projection`の順に3D OBBを学習する。

### 1.1 ゴール要求分析

- **ユーザーの直観的・直截的な目的:** 良くなった2D物体認識を土台にし、RGB-Dから実用的な
  3D OBBを出せるモデルへ段階的に育てたい。
- **明示要求:** 2Dを最大追加50 epochまたはmAP飽和まで学習し、その後、指定された
  `fruits_detection_Jun30-2025_stem_rgbd_736x512_yopo_3dobb_20260825.tar.zst`
  のデータで3D学習する。curriculumと作業記録を残す。
- **暗黙制約:** RTX 5090 32 GiB、repo-local uv環境、重い成果物は`/workspace`側の
  `work_dirs`へ保存する。RGB/depth/K/2D OBB/3D poseを同じ幾何変換で扱い、暗黙補正や
  旧データへのfallbackを禁止する。既存dirty差分をresetしない。
- **非ゴール:** 新データを旧50-frame validationと混ぜること、2D mAPだけで3D品質を
  主張すること、最大物体数を超えるGTを黙って捨てること、未検証checkpointをbestと呼ぶこと。
- **成功条件:** 1196 train / 330 validationを実loaderで読み、最大205 GTを包含する256 query
  modelを構築する。2D bestのRGB-D backbone/neckと既存3D CoP pose重みを監査可能に転送する。
  各stageでfinite lossを確認し、最終stageでfull NOCS validation、best checkpoint、
  `3d_iou_0.50`、AP50、pose 10°/10cmを記録する。
- **リスクと前提:** 新2D detectorと既存DINO pose detectorのheadは非互換なのでheadを
  名前だけで転送しない。validationは24,794 GTで旧splitより重い。Q256はQ150よりVRAMを使うため
  batch capacityを実測する。生成3D教師はprojection-tightの擬似教師であり、独立実測GTではない。

### 1.2 サブゴール構造

| ID | サブゴール | 目的との対応 | 成果物 | 検証方法 |
|---|---|---|---|---|
| SG-1 | データ・split・幾何契約を固定 | 指定データを安全に使用 | inventoryとsample contract | SHA256、全件集計、loader test |
| SG-2 | 2D featureと3D poseを合成転送 | 2D品質を3Dへ継承 | strict transfer hook/report | key/shape test、smoke report |
| SG-3 | Q256 CoP curriculumを実行 | z→size→rotation | stage configs/checkpoints | finite loss、stage gate |
| SG-4 | full validationと記録 | 実用候補を選定 | metric/log/best/overlay | 330-frame evaluatorとartifact監査 |

### 1.3 トレーサビリティ方針

| Trace ID | 要求・制約 | 対応する作業要素 | 証跡 |
|---|---|---|---|
| TR-1 | 指定portable datasetのみ使用 | 手順1–2 | archive SHA256、handoff、conversion report |
| TR-2 | 全GTを表現可能 | 手順2、4 | max GT=205、Q256 config/test |
| TR-3 | 2D featureを3Dへ継承 | 手順3–5 | composite transfer report |
| TR-4 | curriculum | 手順6–8 | z/size/full stage logs |
| TR-5 | 3D指標で選定 | 手順9–10 | NOCSMetric、best checkpoint、overlay |

## 2. 作業内容

### フェーズ1: 調査・設計

アーカイブ、handoff、全PKL集計、既存CoP実装とcheckpointを調べ、データ境界と転送境界を
確定する。新データはtrain 1196 / validation 330、3D instance 85,163、1画像最大205であるため
Q256を採用する。

### フェーズ2: 転送・curriculum実装

最終2D checkpointから同型`RGBDResidualBackbone`と`ChannelMapper`をexact transferする。
既存最良CoP checkpointからbackbone/neckを除くshape互換pose tensorを転送し、Q150の
`query_embedding`だけQ256へ決定論的に拡張する。depth-dense CoPにはresidual backboneの
生depth pyramidを明示的に256chへ射影して渡す。

curriculumは累積的に以下を有効化する。

1. Stage Z: 2D detection/center + metric depth `z`。
2. Stage Size: Stage Z + metric 3D size。
3. Stage Full: Stage Size + SO(3) rotation + 2D OBB/projection consistency。
4. Plateau: full objectiveを`3d_iou_0.50`監視で継続し、改善停止で終了。

### フェーズ3: smoke・本学習・評価

1 iteration smoke、Q256 capacity probe、各5-epoch stage、最終plateau runの順に進む。
途中stageは最終epochを次stageへ渡し、full stage以降だけ3D IoUでbestを選ぶ。

## 3. 作業チェックリスト

### フェーズ1: 調査・設計

### 手順1: アーカイブとhandoffを検証する
- [x] 🖐 **操作**: handoff全文、archive listing、checksum fileを読み、実SHA256を計算する。
- [x] 🔎 **確認**: SHA256が`dfc79f42e03214f8acf4ac5d26a3411c989578f5c1e728e48235c039c393064f`と一致する。
- [x] 🧪 **テスト**: archiveの危険パス0・zstd整合性OKというhandoff証跡と展開済みrootを照合する。
- [x] 🛠 **エラー時対処**: checksum不一致なら展開済みデータを学習に使わず、archiveを再取得する。

### 手順2: instance分布とloader契約を固定する
- [x] 🖐 **操作**: 全1526 PKLのinstance数を集計し、train/testの最大値と分位点を記録する。
- [x] 🔎 **確認**: train 60,369、validation 24,794、最大205でありQ256が全GTを包含する。
- [x] 🧪 **テスト**: `test_portable_3dobb_curriculum_config`でdata root、Q256、feature channel契約を検証する。
- [x] 🛠 **エラー時対処**: GT>256を検出した場合はquery数を増やしてcapacityを再測定し、filterしない。

### 手順3: 転送境界を確定する
- [x] 🖐 **操作**: 2D bestと既存CoP bestのstate_dict root/key数を実測する。
- [x] 🔎 **確認**: 2Dはbackbone 616/neck 12、CoPはpose側互換tensorを持ち、head同士は転送対象外とする。
- [x] 🧪 **テスト**: `test_rgbd_pose_composite_transfer`を先に追加し、未知shape mismatchを失敗させる。
- [x] 🛠 **エラー時対処**: 同名でもshape不一致のtensorはskipせず、許可対象外または設計不一致として停止する。

### フェーズ2: 転送・curriculum実装

### 手順4: residual backboneからdepth pyramidを公開する
- [x] 🖐 **操作**: `RGBDResidualBackbone.forward_with_depth_features`を追加する。
- [x] 🔎 **確認**: fused出力は通常forwardと一致し、depth出力は3 levelの生depth featureである。
- [x] 🧪 **テスト**: `test_rgbd_residual_backbone_depth_feature_contract`をfail→passさせる。
- [x] 🛠 **エラー時対処**: modalities計算を`_forward_modalities`へ集約し、二重forwardを避けた。

### 手順5: composite transferとQ256 configを実装する
- [x] 🖐 **操作**: pose/feature二つのcheckpointをstrictに合成するhookとportable curriculum base configを追加する。
- [x] 🔎 **確認**: 実checkpointで1349 tensor、Q150→256 query、target-only depth projection 4 tensorを確認した。
- [x] 🧪 **テスト**: helper unit test、3 stage config build、Q256/batch12の1 iteration GPU smokeを成功させた。
- [x] 🛠 **エラー時対処**: target-onlyは新規depth projection 4 tensorだけで、未分類missing/mismatchが0であることをreportで確認した。

### 手順6: Stage Zを実行する
- [x] 🖐 **操作**: 2D/center/zのみ有効な5-epoch configをportable train 1196 frameで実行した。
- [x] 🔎 **確認**: loss/gradはfinite、z loss 1.656→0.730、peak 28,446 MiBでepoch5 checkpointを保存した。
- [x] 🧪 **テスト**: full validationでAP50=0.0010、3D IoU25=0.2960、IoU50=0.04038、360°/10cm=0.2050を記録した。
- [x] 🛠 **エラー時対処**: OOM/non-finiteはなく、capacity probeで選定したbatch20を維持した。

### 手順7: Stage Sizeを実行する
- [x] 🖐 **操作**: Stage Z最終重みからsize lossを加えて5 epoch学習した。
- [x] 🔎 **確認**: z loss 0.730→0.648を保持しつつ、size loss 0.606→0.00634までfiniteで低下した。
- [x] 🧪 **テスト**: IoU25 0.2960→0.5522、IoU50 0.04038→0.30476と同一validationで改善した。
- [x] 🛠 **エラー時対処**: zは悪化せず、size weight/LRの差し戻しは不要と判定した。

### 手順8: Stage Fullとplateau継続を実行する
- [x] 🖐 **操作**: Stage Full/plateau後に2D経路を修復し、累積90 epochまで継続した。さらにresize/flip後の`centers_2d`と学習時Kの座標系を修正し、10 epochのgeometry repairを実行した。
- [x] 🔎 **確認**: 旧projection loss約0.750は収束限界ではなく座標不整合だった。20 validation sampleのGT自己投影で0.69793→0.05818、学習lossは約0.061へ低下した。
- [x] 🧪 **テスト**: 累積100 epochでAP50=0.5024、IoU50=0.5297、pose 10°/10cm=0.2256。さらに最大10 epochのplateau確認を行い、累積108 epochで早期終了した。
- [x] 🛠 **エラー時対処**: 追加区間のIoU50最高0.5234はStage 8の0.5297を超えず、3回連続でmin_delta=0.002を満たさなかったため、長期化せずStage 8を3D bestとして採用した。

### フェーズ3: 最終評価・監査

### 手順9: bestをfull validationして可視化する
- [x] 🖐 **操作**: 2D AP bestと3D balance bestを分離して選定し、3D bestを330-frame validationで再評価した。F1 sweep後、conf 0.31 + 2D NMS 0.3875で同一queryのOBB/3Dを12枚描画した。
- [x] 🔎 **確認**: 2D bestはStage 7 epoch35 AP50=0.5422。3D bestはStage 8 epoch10 AP50=0.5024、IoU50=0.5297、pose 10°/10cm=0.2256である。
- [x] 🧪 **テスト**: 3D best dumpは330件・2970 arrayが全finite。F1=0.62688（precision=0.67855、recall=0.58252）、PNG/manifestは12件で一致した。
- [x] 🛠 **エラー時対処**: 旧AP50=0.002はGTだけresize座標、predictionは元画像座標というevaluator不具合だった。GTを`scale_factor`で元画像座標へ戻す修正と回帰test後、同一checkpointでAP50=0.5282を確認した。

### 手順10: 作業記録と品質ゲートを完了する
- [x] 🖐 **操作**: config、log、checkpoint、metric、SHA256、GPU peak、失敗と対処を本書へ追記した。
- [x] 🔎 **確認**: TR-1〜5の証跡、旧評価の無効理由、用途別best、残課題を明示した。
- [x] 🧪 **テスト**: 最終focused pytest 112 passed、対象Ruff、`git diff --check`を実行した。
- [x] 🛠 **エラー時対処**: repo全体には今回と無関係な既知baseline failure/lint debtが残るため、今回変更範囲の品質ゲートと全体baselineを区別して記録した。

## 4. 作業に使用するコマンド参考情報

```bash
cd /home/kasm-user/Desktop/YOPO_clone
uv run pytest -q tests/test_rgbd_pose_curriculum.py
uv run python tools/train.py configs/yopo/<portable_stage>.py --work-dir work_dirs/<stage>
uv run python tools/test.py configs/yopo/<inference>.py <best.pth>
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
```

## 6. 完了の定義

- [x] 観点1: 指定portable datasetの1196/330 frameと最大205 GTを暗黙除外なく使用した。
- [x] 観点2: 2D featureと3D poseの合成転送がkey/shape/reportで証明されている。
- [x] 観点3: z→size→rotation/projection curriculumとplateau判定を実GPUで完了した。
- [x] 観点4: full 3D validation、best checkpoint、可視化、uv test/lint、作業記録が揃った。

## 7. 作業記録

**重要な注意事項：**

- 作業開始前に必ず`date "+%Y-%m-%d %H:%M:%S %Z%z"`で現在時刻を確認する。
- 各作業項目を開始する際と完了する際の両方で記録を行うこと。
- 作業内容は具体的なコマンドや操作手順を詳細に記載すること。
- 結果・備考欄には成功／失敗、エラー内容、解決方法、重要な気づきを必ず記入すること。
- 複数のフェーズがある場合は、フェーズごとに開始・完了の記録を取ること。
- コード変更を行った場合は、変更したファイル名と変更内容の概要を記録すること。
- エラーが発生した場合は、エラーメッセージと解決策を詳細に記録すること。

| 日付 | 時刻 | 作業者 | 作業内容 | 結果・備考 |
|---|---|---|---|---|
| 2026-08-26 | 01:15 JST | Codex | 2D plateau run開始 | 最大50 epoch、min_delta 0.0005、patience 12、低LRで開始。 |
| 2026-08-26 | 01:17 JST | Codex | archive/handoff検証 | SHA256一致。展開済み1526 frame、85,163 3D instanceを確認。 |
| 2026-08-26 | 01:18 JST | Codex | 全PKL instance集計 | train 60,369 / val 24,794、train max164、val max205。Q256採用。 |
| 2026-08-26 | 01:20 JST | Codex | 転送境界調査 | 2D backbone 616 + neck 12、既存CoP pose 1443 tensor。head直結は非互換のためfeature/pose分離転送と決定。 |
| 2026-08-26 | 01:25 JST | Codex | TDD実装 | residual depth feature公開とheterogeneous channel projectionを先に2 test失敗で再現し、実装後2 passed。 |
| 2026-08-26 | 01:27 JST | Codex | composite transfer/config build | 3 stageすべてQ256でbuild。実checkpointから1349/1353 tensorを選択、残り4 tensorは新depth input projection。dataset 1196、sample `(4,445,640)`、GT48。 |
| 2026-08-26 | 02:10 JST | Codex | 2D plateau run完了 | 50 epoch完走。best epoch48 mAP50=0.2011801、final=0.1998965、peak 27,306 MiB。best SHA256=`25361d0c85df1f4f4eda22644156d139ecb0368719ebb6dcc25854838aa90e1f`。3D feature sourceを同bestへ固定。 |
| 2026-08-26 | 02:12–02:15 JST | Codex | Stage Z GPU smoke/full baseline | Q256/batch12で1 update成功。loss=70.3547、z loss=1.3854、grad finite、peak 15,851 MiB。strict transfer 1349 tensor、Q150→256、target-only 4。330-frame baseline: AP50=0.0010、3D IoU25=0.1699、IoU50=0.00694。 |
| 2026-08-26 | 02:16 JST | Codex | Q256 capacity probe | batch20で1 update成功。loss=68.8530、z loss=1.4118、peak 26,317 MiB、time=2.423s/update。batch12の1.813s/updateと比較して1 epoch見込みが約181s→145sのため、full stageをbatch20へ固定。 |
| 2026-08-26 | 02:17–02:25 JST | Codex | Stage Z full | 5 epoch/300 update完走。z loss 1.656→0.730、3D IoU50 0.00694→0.04038、360°/10cm 0.0474→0.2050。checkpoint SHA256=`1647ce4b6138139a595960d32d4f2ba751edd8d8c6cc293a00ef55e73192e7ab`。 |
| 2026-08-26 | 02:28–02:36 JST | Codex | Stage Size full | 5 epoch/300 update完走。size loss 0.606→0.00634、z loss 0.648。IoU25 0.5522、IoU50 0.30476、IoU75 0.0470。checkpoint SHA256=`1bd2a1ace4cff51d12ca56289ba6e8ad76ba7e01b715413eeee2333f55686733`。 |
| 2026-08-26 | 02:38–02:47 JST | Codex | Stage Full | 5 epoch/300 update完走。rotation loss 5.81→3.161、最終z 0.6237、size 0.0067、projection 0.7497、peak 28,467 MiB。IoU25 0.5801、IoU50 0.3393、IoU75 0.0871、pose 10°/10cm 0.0341。best SHA256=`d60cdf9c5f8ebabf915a50dc3c9bd2c5a2897c2689cead1569603d3284f3bdc4`。 |
| 2026-08-26 | 02:48 JST | Codex | Plateau run開始 | Stage Full bestから最大50 epoch、5 epoch間隔validation、IoU50 min_delta=0.002、patience=3で飽和判定を開始。 |
| 2026-08-26 | 02:57 JST | Codex | Plateau epoch 5評価 | IoU25=0.5988、IoU50=0.3562、IoU75=0.0942、pose 10°/10cm=0.0413。Stage Full比IoU50 +0.0169でbest更新、継続。 |
| 2026-08-26 | 03:05 JST | Codex | Plateau epoch 10評価 | IoU25=0.6111、IoU50=0.3629、IoU75=0.0943、pose 10°/10cm=0.0459。前best比 +0.0066でmin_deltaを超え、best更新・継続。 |
| 2026-08-26 | 03:14 JST | Codex | Plateau epoch 15評価 | IoU25=0.6319、IoU50=0.3991、IoU75=0.1107、pose 10°/10cm=0.0587。前best比 +0.0362でbest更新、継続。 |
| 2026-08-26 | 03:22 JST | Codex | Plateau epoch 20評価 | IoU25=0.6483、IoU50=0.4038、IoU75=0.1084、pose 10°/10cm=0.0631。前best比 +0.0048でbest更新、継続。 |
| 2026-08-26 | 03:31 JST | Codex | Plateau epoch 25評価 | IoU25=0.6688、IoU50=0.4374、IoU75=0.1185、pose 10°/10cm=0.0690。前best比 +0.0336でbest更新、継続。 |
| 2026-08-26 | 03:39 JST | Codex | Plateau epoch 30評価 | IoU25=0.6785、IoU50=0.4454、IoU75=0.1225、pose 10°/10cm=0.0765。前best比 +0.0080でbest更新、継続。 |
| 2026-08-26 | 03:47 JST | Codex | Plateau epoch 35評価 | IoU25=0.6898、IoU50=0.4534、IoU75=0.1239、pose 10°/10cm=0.0782。前best比 +0.0080でbest更新、継続。 |
| 2026-08-26 | 03:55 JST | Codex | Plateau epoch 40評価 | IoU25=0.6954、IoU50=0.4586、IoU75=0.1253、pose 10°/10cm=0.0846。前best比 +0.0052でbest更新、継続。 |
| 2026-08-26 | 04:04 JST | Codex | Plateau epoch 45評価 | IoU25=0.7027、IoU50=0.4625、IoU75=0.1202、pose 10°/10cm=0.0884。前best比 +0.0039でbest更新、最終5 epochへ継続。 |
| 2026-08-26 | 04:12 JST | Codex | Plateau max 50完走 | 最終かつbest: IoU25=0.70575、IoU50=0.46436、IoU75=0.11633、pose 10°/10cm=0.09231、360°/10cm=0.45744、AP50=0.002。最終train loss=77.764、z=0.4358、rotation=2.4826、size=0.00630、projection=0.75027、peak 28,545 MiB。best SHA256=`d8ece5247c17dffcaccac7f50520e87c652fbfa65ba310711b5339a5b28a6f1a`。 |
| 2026-08-26 | 04:13–04:16 JST | Codex | best再評価・prediction dump | 330-frame metricを再現。prediction dump 18,696,300 bytes、SHA256=`d0318781106ce354511cec48ea5aeb4591db311b61b37b8a0717883d6241042a`。 |
| 2026-08-26 | 04:17 JST | Codex | 可視化・artifact検査 | conf 0.2 + class-agnostic 2D NMS 0.5、同一query 3D cuboidを12 PNGへ描画。330 prediction、2640 array finite、12 manifest/output一致。低閾値では全例150件上限となり、2D AP50=0.002の残課題を確認。 |
| 2026-08-26 | 04:18 JST | Codex | 最終品質ゲート | focused pytest 86 passed / 26 warnings、Ruff pass、`git diff --check` pass。 |
| 2026-08-26 | 05:02–05:54 JST | Codex | Stage 6 detection repair | 35 epoch実行。学習中AP50=0.002と表示されたが、後の座標監査で評価値が無効と判明。epoch35を修正後evaluatorで再評価しAP50=0.5282、IoU50=0.4477を確認。 |
| 2026-08-26 | 06:05–06:13 JST | Codex | 2D evaluator座標監査・修正 | GTはresize座標、predictionは元画像座標だった。`scale_factor`でGTを元画像へ戻す実装と回帰testを追加し、同一checkpointのAP50が0.002→0.5282になった。 |
| 2026-08-26 | 06:14–07:33 JST | Codex | Stage 7長期継続 | 最大65中55 epochでAP50早期終了。累積90 epoch。2D best epoch35 AP50=0.5422、3D best epoch20 IoU50=0.4528。 |
| 2026-08-26 | 07:34–07:42 JST | Codex | Stage 7 score/NMS校正 | best AP dumpで精密sweep。conf=0.30、NMS IoU=0.375、precision=0.6705、recall=0.5873、F1=0.62617。12枚を描画・目視確認。 |
| 2026-08-26 | 07:43–07:50 JST | Codex | projection座標監査・修正 | 通常`Resize`が`centers_2d`を変換せず、学習時Kも元画像座標のままだった。`ResizeforPose`とcurrent-image K変換を導入。GT自己投影loss 0.69793→0.05818、中心誤差max 0.000123 px。 |
| 2026-08-26 | 07:50–08:08 JST | Codex | Stage 8 geometry repair | 10 epoch実行し累積100 epoch。projection loss約0.061、center loss 0.761→0.0809。最終AP50=0.5024、IoU25/50/75=0.7374/0.5297/0.1756、pose 10°/10cm=0.2256。 |
| 2026-08-26 | 08:09–08:24 JST | Codex | Stage 9 plateau確認 | Stage 8 3D bestから最大10 epochで開始し、8 epochで早期終了（累積108）。IoU50は0.5234→0.5233→0.5229→0.5165でStage 8 best未達。pose 10°/10cmは最終0.2351。 |
| 2026-08-26 | 08:24–08:27 JST | Codex | 3D best再評価・最終可視化 | Stage 8 bestを再評価してAP50=0.5024、IoU50=0.5297を再現。330 prediction・2970 array finite。精密sweep bestはconf=0.31/NMS=0.3875、F1=0.62688。12 PNGを生成し先頭/中央/末尾を目視確認。 |
| 2026-08-26 | 08:28 JST | Codex | 最終品質ゲート | focused pytest 112 passed / 26 warnings、対象Ruff pass、`git diff --check` pass。 |
| | | | | |

## 8. 完了照合・調査分析サマリ

### 8.1 監査範囲

本節は2026-08-26 08:28 JST時点のrepository、config、学習log、checkpoint、prediction dump、
threshold sweep、可視化manifestを実物照合した結果である。過去logのAP50=0.002は履歴として残すが、
座標不一致evaluatorによる無効値であり、修正後の値と混同しない。

### 8.2 完了定義との照合

| 完了観点 | 判定 | 実証 |
|---|---|---|
| 1196/330 frame、最大205 GTを扱う | 達成 | Q256、validation 24,794 GT、capacity超過画像0をdump/sweepで再確認 |
| 2D featureと3D poseの合成転送 | 達成 | strict transfer 1349 tensor、Q150→Q256、target-only depth projection 4 tensor |
| curriculumと飽和判定 | 達成 | z→size→full→2D repair→geometry repair、累積100 epoch後に8 epochのplateau確認で早期終了 |
| full validation/best/可視化/test | 達成 | 330-frame再評価、用途別best、12 PNG、112 focused tests、対象lint/diff check |

### 8.3 最終成果

- **2D検出優先:** Stage 7 epoch35、AP50=0.5422。F1校正はconf=0.30/NMS=0.375で0.62617。
- **3Dバランス優先:** Stage 8 epoch10、AP50=0.5024、3D IoU50=0.5297、pose 10°/10cm=0.2256。
- **姿勢寄りの参考値:** Stage 9 epoch8はpose 10°/10cm=0.2351だが、AP50=0.4931、IoU50=0.5165へ低下したため総合bestには採用しない。
- **幾何整合性:** GT自己投影lossは0.69793から0.05818へ低下し、学習projection loss約0.061は教師由来の残差に近い。
- **最終表示条件:** Stage 8でconf=0.31/NMS=0.3875、precision=0.67855、recall=0.58252、F1=0.62688。

### 8.4 証跡

- 3D best checkpoint SHA256: `76987d9c2dd4137eed78f7ce282b078ece8279c919734d5c834da1150b74bd2b`
- prediction dump SHA256: `9e75cee9dabd78579b52fe986d6da28326bb7a31ada25363b0d8fbbace9feb97`
- F1 sweep SHA256: `34bc1b1fe9ba56ae5c1e985288b59cd9e061f811e6d3e3704941a0786d50823d`
- visualization manifest SHA256: `fa09e7d97ff58155cc8befa91e708d027eccfe2a8d491c1e71b3f32409854559`
- focused regression: `112 passed, 26 warnings in 8.79s`。対象Ruffと`git diff --check`はpass。
- repository全体の既知baselineは過去確認時`189 passed / 13 failed`、全体Ruff 510件であり、今回変更範囲が全repo debtを解消したとは主張しない。

### 8.5 残課題と結論

追加学習だけでは2D APと3D IoUが同じepochで同時最大にならないため、用途別checkpointを維持する。
Stage 9では3D IoUが改善せず、100 epochを超えた追加学習を続ける根拠はなくなった。座標不整合、
projection loss停滞、誤ったAP表示は解消済みで、指定データによるRGB-D 3D OBB curriculum、
飽和確認、再現可能な評価・校正・可視化まで完了したと判定する。
