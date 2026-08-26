# 作業計画・記録: pretrained pyramid + dense proposal RT-DETR

日付: 2026-08-25–26  
repo: `/home/kasm-user/Desktop/YOPO_clone` (`rgb-d`, uv)  
重い出力: `/workspace/YOPO_clone/work_dirs`  
設計正本: `docs/rgbd_obb_modular_architecture_ablation_design.md`

## 1. 目的と判定基準

旧RGB-D query-only detectorのmAP50最大0.0527を、学習延長ではなく次の構造修正で上回る。

1. pretrained RGB featureとneckを連続して再利用する。
2. dense rotated O2M headで全cellを学習する。
3. dense top-kをRT-DETR型のquery content/referenceへ直結する。
4. finalは反復refineされたone-to-one OBB setとする。

成功はfull validation `rbbox_mAP_50 > 0.0527`で判定する。loss低下だけでは完了にしない。

## 2. 最短作業

- [x] `RGBDResidualBackbone`を実装し、depth gate 0でRGB featureとbit-exactにする。
- [x] sourceと同じ`ChannelMapper([384,768,1536] -> 256)`を接続する。
- [x] RGB backbone/neck/互換encoder-decoderを監査可能に転送し、非互換query回帰headを除外する。
- [x] `RotatedRTMDetSepBNHead`へltrb+angle、Dynamic Soft Label、QFL、RIoU/GWD、rotated NMSを実装する。
- [x] dense top-k cellのencoder memoryとxywh/angleをdecoder初期値にする。
- [x] rotated 5D regressionと4D deformable-attention referenceの反復refineを両立する。
- [x] focused test、model build、batch2 smoke、batch32 capacityを通す。
- [x] hybrid RIoUを5 epoch完走し、full validationを記録する（best 0.0919）。
- [x] RIoU bestからhybrid GWDを15 epoch完走する（best 0.1184）。
- [x] GWD bestから追加50 epochを完走し、全stageのbestを選ぶ（epoch 49、0.1955）。
- [x] 最良checkpointのconf 0.2 valid overlayを出力する。

## 3. 実行コマンド

```bash
cd /home/kasm-user/Desktop/YOPO_clone

uv run pytest -q \
  tests/test_rgbd_residual_backbone.py \
  tests/test_rgbd_stem_obb_config.py \
  tests/test_composable_pyramid_neck.py

uv run ruff check \
  yopo/models/backbones/dual_rgbd.py \
  yopo/models/dense_heads/rotated_rtmdet_head.py \
  yopo/models/detectors/rotated_rt_detr.py \
  yopo/models/layers/transformer/deformable_detr_layers.py \
  yopo/models/task_modules/assigners/iou2d_calculator.py \
  yopo/engine/hooks/rgbd_obb_transfer.py \
  tests/test_rgbd_residual_backbone.py \
  tests/test_rgbd_stem_obb_config.py

# RIoU 5 epoch
uv run python tools/train.py \
  configs/yopo/rotated_rt_detr_stem_rgbd_obb_hybrid_riou_full.py \
  --work-dir work_dirs/rtdetr_stem_rgbd_obb_hybrid_riou_full

# GWD 15 epoch。Nは実在するRIoU best epochへ置換する。
uv run python tools/train.py \
  configs/yopo/rotated_rt_detr_stem_rgbd_obb_hybrid_gwd_full.py \
  --work-dir work_dirs/rtdetr_stem_rgbd_obb_hybrid_gwd_full \
  --cfg-options \
  load_from=/workspace/YOPO_clone/work_dirs/rtdetr_stem_rgbd_obb_hybrid_riou_full/best_rbbox_mAP_50_epoch_N.pth

# GWD bestから追加50 epoch。Nは実在するGWD best epochへ置換する。
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python tools/train.py \
  configs/yopo/rotated_rt_detr_stem_rgbd_obb_hybrid_gwd_continue50.py \
  --work-dir work_dirs/rtdetr_stem_rgbd_obb_hybrid_angle_gwd_continue50_v2_retry \
  --cfg-options \
  load_from=/workspace/YOPO_clone/work_dirs/rtdetr_stem_rgbd_obb_hybrid_angle_gwd_full_v2/best_rbbox_mAP_50_epoch_15.pth
```

## 4. 実装契約

- source neckをreshapeしない。全tensorをexact shapeでのみ転送する。
- dense headは補助画像生成だけでなく、query選択へ実際に使う。
- pure denseとhybridをconfigで独立buildできるようにする。
- OOM時だけbatchを変更し、勝手なgradient accumulationへ切り替えない。
- RIoU bestが存在しない場合はGWDを開始しない。
- 既存dirty差分をresetしない。

## 5. 作業記録

| UTC | 作業 | 結果 |
|---|---|---|
| 22:21–22:24 | 旧pyramid refiner実装/smoke | focused 8 pass、batch2 smoke finite。source neck非互換を検出。 |
| 22:25–22:30 | 旧RIoU 5 epoch | best mAP50=0.0093、peak 26,596 MiB。 |
| 22:32–22:47 | 旧GWD 15 epoch | best epoch15 mAP50=0.0268。 |
| 22:48–23:08 | 旧GWD追加20 epoch | best epoch19 mAP50=0.0527。学習延長だけでは不十分と判定。 |
| 23:14–23:17 | pretrained経路修正 | RGB恒等backbone test 3 pass。実checkpointからRGB 360 + neck 12 tensors exact transfer。 |
| 23:15–23:17 | dense OBB head | GPUでRIoU+SmoothL1+QFL、O2M assignment、backward、rotated NMS成功。 |
| 23:18–23:22 | RT-DETR接続 | dense top-k→encoder memory/query xywh、iterative refineを実装。batch2×2 iterとvalidation成功。 |
| 23:22 | batch24 probe | peak 20,526 MiB、dense positives 3,256、finite。 |
| 23:23 | batch32 probe | peak 27,249 MiB、dense positives 3,624、finite、OOMなし。 |
| 23:24 | transfer拡張 | 初版557 tensors。後の診断でquery回帰head 32 tensorsを転送対象外へ変更。 |
| 23:25–23:32 | hybrid RIoU v1 | dense lossは低下したがquery IoU loss=2.0、mAP50=0。dense angle未接続と旧回帰head転送を原因特定。 |
| 23:33 | angle-aware smoke | 17 tests pass。転送525 tensors。query RIoU loss約1.67でfinite。 |
| 23:34–23:40 | hybrid RIoU v2 5 epoch | best epoch5 mAP50=0.0919、peak 27,379 MiB。旧bestを上回った。 |
| 23:41 | 枝別評価 | query 0.0919、dense 0.0051。finalはNMS-free query、denseはproposal用途と判定。 |
| 23:41–23:58 | hybrid GWD 15 epoch | best epoch15 mAP50=0.1184。旧best 0.0527の約2.25倍。 |
| 23:58–00:00 | 追加50 epoch開始 | 前runのGPU解放待ち競合で初回OOM。残骸終了後、clean retryを開始。 |
| 00:01 | 追加50 epoch 1 | mAP50=0.1192へ更新、peak 27,305 MiB。 |
| 00:01–00:54 | GWD追加50 epoch | 50 epoch完走。best epoch49 mAP50=0.1955、final epoch50=0.1937、peak 27,306 MiB。 |
| 00:54– | final replay/F1/可視化 | full validationを再現。最良conf=0.3355816305（F1=0.3248）。conf 0.2、NMSなし、6画像を描画。 |

## 6. 完了結果

- hybrid RIoU best: 0.0919 / `...hybrid_angle_riou_full_v2/best_rbbox_mAP_50_epoch_5.pth`
- hybrid GWD 15 best: 0.1184 / `...hybrid_angle_gwd_full_v2/best_rbbox_mAP_50_epoch_15.pth`
- hybrid GWD追加50 best: 0.1955 / epoch 49 /
  `work_dirs/rtdetr_stem_rgbd_obb_hybrid_angle_gwd_continue50_v2_retry/best_rbbox_mAP_50_epoch_49.pth`
- SHA256: `b4e6775474a6dbc1e24a9c70288c4777d93b0f0d857772e1caca7f85fd86d34c`
- 旧best 0.0527との差: +0.1428、約3.71倍。
- full validation（conf 0.05）: mAP50 0.1955、VOC07 0.2523、recall 0.6048、
  precision 0.1985、mean matched rIoU 0.6750、GT 27,723、prediction 84,478。
- F1最大点（rotated IoU 0.5、score降順greedy one-to-one）: conf 0.3355816305、
  F1 0.3248、precision 0.2565、recall 0.4425。
- threshold JSON: `work_dirs/rtdetr_hybrid_angle_gwd_continue50_best_eval/f1_threshold.json`
- conf 0.2 overlay:
  `work_dirs/rtdetr_hybrid_angle_gwd_continue50_best_eval/valid_overlays_conf0p2/`
- best-F1 conf overlay:
  `work_dirs/rtdetr_hybrid_angle_gwd_continue50_best_eval/valid_overlays_best_f1/`

## 7. Plateau確認run（2026-08-26追加）

- 入力: `...hybrid_angle_gwd_continue50_v2_retry/best_rbbox_mAP_50_epoch_49.pth`
- 上限: 追加50 epoch、validationは毎epoch。
- optimizer: checkpointがweights-onlyのため再構築。Muon `1e-5`、ScheduleFree `5e-7`。
- 飽和判定: mAP50の改善幅`0.0005`未満が12 validation連続で継続した時点。
- config: `configs/yopo/rotated_rt_detr_stem_rgbd_obb_hybrid_gwd_continue50_plateau.py`
- [x] 最大50 epochを完走。best更新幅が続いたためearly stopより上限を先に満たした。
- [x] 既存best 0.1955を含めてepoch 48（mAP50 0.2011801）を最終checkpointに選定。
- [ ] full validationと作業結果を追記。
