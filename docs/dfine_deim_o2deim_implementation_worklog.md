# D-FINE / DEIM / O²-DEIM改善 実装作業記録

最終更新: 2026-08-26
設計: [dfine_deim_o2deim_rgbd_3dobb_architecture.md](./dfine_deim_o2deim_rgbd_3dobb_architecture.md)
claude-mem session: `codex-yopo-dfine-o2-20260826`

## 1. 完了の定義

- 新部品が既存挙動を既定値で変えない。
- MAL、OBB Chamfer、box-only OCDを個別に設定・試験できる。
- 現行3D headへ少なくともMALと2D-anchored assignmentを統合できる。
- config build、単体試験、CPU/GPU smokeがfiniteで完了する。
- loss、quality、matching、DNが独立componentで、configだけで組合せを変更できる。
- host固有pathをconfigへ固定せず、別machineでも同じsource/configを利用できる。
- 既存checkpointを上書きせず、独立work_dirで短期比較できる。
- FULL trainingを開始する前に採用gateとrollback条件を記録する。

## 2. 担当表

| agent_id | scope | workspace | allowed paths | forbidden | status | last update | evidence |
|---|---|---|---|---|---|---|---|
| `/root` | 統合設計、head/config接続、総合試験、GPU smoke、FULL開始 | `YOPO_clone` | docs、head、config、必要なexports/tests | user変更の巻き戻し、commit/push | active | 2026-08-26 | 最終選定focused 113 pass、ruff/diff-check、finite GPU smoke、FULL run |
| `/root/paper_2412_dfine` | standalone MAL + unit test | shared | `yopo/models/losses/matchability_aware_loss.py`, `tests/test_matchability_aware_loss.py` | exports/head/config | complete | 2026-08-26 | `MatchabilityAwareLoss`実装・統合済み |
| `/root/paper_2603_o2deim` | standalone OBB Chamfer cost + unit test | shared | `match_cost.py`, `tests/test_obb_chamfer_match_cost.py` | exports/config/head | complete | 2026-08-26 | `OBBChamferCost`実装・registry export済み |
| `/root/paper_2410_deim` | standalone box-only OCD helper + unit test | shared | `oriented_contrastive_denoising.py`, `tests/test_oriented_contrastive_denoising.py` | exports/config/head | complete | 2026-08-26 | `BoxOnlyOCDNoise`実装・registry export済み |

## 3. 既存worktree保護

開始時点で多数のtracked/untracked変更が存在する。これらは過去のユーザー作業と本プロジェクトの
成果であり、reset、checkout、cleanを行わない。各担当は許可されたファイルだけを変更し、統括が
`git diff -- <path>`でレビューする。commit/pushはユーザーが改めて求めるまで行わない。

## 4. 実装checklist

### Phase 0: 契約固定

- [x] 論文3本の主張、ablation、限界を一次資料で確認した。
- [x] 現行headがcompact Gaussian OBBを使うことを確認した。
- [x] Q=256がvalid最大205を収容することを確認した。
- [x] Stage 7/8の2D/3D bestを記録した。
- [x] claude-mem sessionを初期化し、最初の設計判断を記録した。
- [x] architecture documentを作成した。

### Phase 1: 独立部品

- [x] MALは`q=0` matched positiveをpositiveとして扱う。
- [x] MALはqualityとnegative focal weightをdetachする。
- [x] MALのnone/mean/sum、weight、avg_factorを試験する。
- [x] OBB Chamferはsquared/unsquaredを切り替えられる。
- [x] OBB Chamferは画像サイズ正規化と空集合を安全に扱う。
- [x] OBB Chamferは頂点順序・angle canonicalizationに不変である。
- [x] OCDはangle/pose属性を変更しない。
- [x] OCDはpositive:negative 1:1、clip、min-size、seed再現性を満たす。

### Phase 2: YOPO統合

- [x] `MatchabilityAwareLoss`、`MatchabilityQualityPolicy`、`OBBChamferCost`、`BoxOnlyOCDNoise`をregistry exportする。
- [x] headが一つのclassification adapterからFocal/QFL/MALを設定互換に選択できる。
- [x] MAL quality sourceを`hbb_iou`、`obb_gwd`、`blend`から選択できる。
- [x] quality計算をheadから独立componentへ分離し、decoder/encoder/DNの共通classification adapterから再利用する。
- [x] encoder/DN/decoder各lossでlabelからmatched-positive状態を保持し、`q=0`でもbackgroundへ変換しない。
- [x] Stage 3以降の主assignerからtranslation/rotation costを外したStage 10派生configを作る。
- [x] 新option未指定時のparameter shape、checkpoint key、既定Focal経路を変えない。

### Phase 3: 検証

- [x] 統合直後のfocused pytest 120件と、ScheduleFree/no-DN修正後の最終選定suite 113件が成功する。
- [x] configをregistry込みでbuildできる。
- [x] 1 batch forward/lossが全finiteである。
- [x] GPU 1–2 iteration smokeがOOMなしで完了する（peak約28.1 GiB）。
- [x] Stage 8 checkpointをFULL modelへ読み込み、想定外missing/unexpected/size mismatchがない。
- [x] control/MAL各330画像・256 queryのraw prediction dumpを保存できる。

### Phase 4: PDCA training

- [x] 同一checkpoint/seedの1-epoch controlを用意する。
- [x] 2D-anchored controlに対し、新規差分をMAL（`hbb_iou`）だけに固定したmatched probeを実行する。
- [ ] OCD追加probeは前段がgate通過した時だけ実行する。
- [x] AP50/AP75/AP50:95、3D、score calibrationを比較する（評価経路は下記証跡に明記）。
- [x] 採用gate通過後、fresh FULLを開始する（最大100 epochまたはearly stopping）。
- [ ] best checkpointでF1最適conf/IoUを再探索し、validation overlayを出力する。

## 5. 判断記録

### 2026-08-26: 第一実装の範囲

決定:

- 既存checkpoint互換なMALと2D-anchored assignmentを最優先にする。
- Chamferは明示5D OBB経路向けのoptional costとして独立実装する。
- OCDはangle/3D poseを摂動しないpure helperから始める。
- six-distribution refinerは第一probeの後に実装判断する。

理由:

- 現行3D headのOBBはcompact Gaussianであり、5D OBB Chamferを直接利用できない。
- Stage 7とStage 8のtrade-offは、2D assignmentを3D costで動かす危険性を示す。
- 本データは既に高密度で、Dense O2O Mosaicの利益よりquery/camera破壊リスクが大きい。
- 一度にparameter shapeを変えると、これまでの100 epoch級学習を有効活用できない。

### 2026-08-26: modularityとstorage

決定:

- loss、quality policy、matcher、DN noise strategyを独立componentにする。
- 新経路はregistry/configから選択し、既定値を変更しない。
- future runは`/home/kasm-user/Desktop/YOPO_clone_artifacts/work_dirs`へ出力する。
- temporary filesと新規uv cacheも同artifact rootへ分離する。

根拠:

- `/workspace`: 60 GiB中60 GiB、空き約245 MiB。
- Desktop filesystem: 初回監査で空き約123 GiB、最新確認で空き約114 GiB。
- 現行`.venv`と`work_dirs`は`/workspace/YOPO_clone`へのsymlinkである。
- claude-mem/chroma processが`/workspace/kasm-user/uv-cache`内interpreterを実行中のため、既存cacheを実行中に移動しない。

## 6. claude-mem記録方針

次の節目だけをmemoryへ記録し、逐次ログを大量保存しない。

1. architecture decisionと不変条件。
2. 独立部品のunit test結果。
3. 統合時に発見したデータ/shape/gradient契約。
4. GPU smokeのcheckpoint、config、peak memory、finite結果。
5. probe/FULL runの採否、best metrics、次の分岐。

memory記録は本書の証跡を置換せず、別agentが作業を再開するための索引として使う。

## 7. 作業ログ

### 2026-08-26

- `rgb-d` / HEAD `bfd1354`を確認した。
- dirty worktreeを記録し、既存変更を保護対象とした。
- 論文調査担当3名を独立実装へ再割当した。
- 現行datasetはraw 5D OBBをcompact Gaussianへ変換し、packerは`obb_gaussians`として渡すことを確認した。
- 現行Stage 3 matchingがFocal + HBB L1 + GIoU + Translation + Rotationであることを確認した。
- 設計上、初期FULL候補は2D anchorへ戻し、3D属性はその対応を共有する方針とした。
- user指示に基づきSOLID/KISS/DRY、config-driven portabilityを受入条件へ追加した。
- storage監査後、Desktop側に`YOPO_clone_artifacts/{work_dirs,tmp,uv-cache}`を作成した。
- 同じ判断をclaude-memへ`CodexModularityAndStorageDecision`として記録した。
- `MatchabilityAwareLoss`、`MatchabilityQualityPolicy`、`OBBChamferCost`、`BoxOnlyOCDNoise`を実装し、registry exportを確認した。
- DINO pose headのclassification adapterがFocal/QFL/MALを設定互換に選択できることを確認した。`tools/test.py`にはtrain-only configの評価時fallbackを追加した。
- focused test 120件、ruff、対象diff確認が成功した。GPU 1–2 iteration smokeはfinite、OOMなし、peak約28.1 GiBだった。
- NOCSMetricはraw HBB all-query（`score_thr=0`、NMSなし）とoperating 3D（`score_thr=0.2`、NMS IoU `0.3`）を分離して報告する。`AP50_95`はIoU `0.50:0.05:0.95`の10個のVOC-area AP平均であり、厳密なCOCO evaluatorではない。
- matched 1-epochではcontrolのraw HBB AP50/AP75/AP50_95が`0.3856/0.0588/0.1384`、3D IoU50が`0.6085`、pose 10deg/10cmが`0.2346`だった。MAL（`hbb_iou`）は順に`0.3875/0.0654/0.1428`、`0.6117`、`0.2421`で、採用gateを通過した。
- diagnosticはSpearman `0.56148 -> 0.63725`、AUC `0.71849 -> 0.72049`、no-NMS recall `0.71187 -> 0.71174`、NMS recall `0.66476 -> 0.66702`、NMS recall gap `0.04711 -> 0.04473`だった。
- 5-epoch MAL probeの旧filtered AP50はepoch 3でbest `0.5800`、best 3D IoUはepoch 2で`0.6204`だった。これはraw HBB評価と別設定のため直接比較しない。
- Stage 8 checkpointからfresh FULLを開始したが、独立監査でScheduleFree optimizerのvalidation mode切替が未実装と判明した。旧runのepoch 5/10値はtrain-mode weight上の暫定診断値であり、best選択・早期停止の正式記録から除外する。旧logは`/home/kasm-user/Desktop/YOPO_clone_artifacts/work_dirs/stage10_2d_anchor_mal_full_fresh/20260826_105203/20260826_105203.log`に保全した。
- `ScheduleFreeOptimizerModeHook`を追加し、`before_val`で`optimizer.eval()`、CheckpointHook/EarlyStoppingを含む全`after_val_epoch`処理後の`after_val`で`optimizer.train()`へ戻す。5件のhook testを含むfocused test 89件とruffが成功した。
- epoch 10 / iter 600のmodel・optimizer状態をそのまま保持し、誤った旧validationの`val/*` scalar、best score/path、periodic checkpoint bookkeepingだけを除いたresume artifactを作成した。これはmodel/optimizer/epochについてexactだが、評価・早期停止baselineは意図的にresetしている。
- 正式runは`/home/kasm-user/Desktop/YOPO_clone_artifacts/work_dirs/stage10_2d_anchor_mal_full_schedulefree_fixed_v2`へ`--resume`した。最大100 epoch、validation interval 5、EarlyStoppingは`AP50_95`の`min_delta=0.002`、patience 6である。最初のauthoritativeな平均化weight評価はepoch 15となる。FULLは2D-only assignment + MAL `hbb_iou`を新規差分とし、既存DINO DN、Gaussian GWD補助、pose/projection/CoPは有効のままである。新規のblend/custom OCD/Chamfer/6分布refinerは未採用とする。
- epoch 15のauthoritative評価はraw HBB `AP50=0.4089`、`AP75=0.0777`、`AP50_95=0.1547`、operating 3D `IoU@0.50=0.6069`、pose `10deg/10cm=0.2295`だった。`epoch_15.pth`内のoptimizerは`train_mode=False`で、直後のepoch 16 training stepも成功したため、averaged-weight保存とtrain-mode復帰の両方を実証した。
- 監査で見つかったactive非依存の`dn_cfg=None`経路も修正した。DNなしではmatching queryだけを使い、`dn_meta=None`を安全に分割する。設定済みDNのmodule/checkpoint keyは変えない。active FULLは`dn_cfg` configuredでlegacy DNを使うため、この互換修正による実行経路変更はない。修正直後のcombined focused suiteは99件、最終選定suiteは113件成功した。
- periodic `epoch_N.pth`はoptimizer stateを含む正確なresume用で、`best_*.pth`はaveraged model weightだけの評価/配布用である。`max_keep_ckpts=2`はperiodic fileだけを制限し、metric別bestは別に保存する。
