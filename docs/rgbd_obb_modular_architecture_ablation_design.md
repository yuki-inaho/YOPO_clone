# RGB-D 2D OBB: pretrained feature + dense proposal RT-DETR設計

最終更新: 2026-08-26
対象: `/home/kasm-user/Desktop/YOPO_clone` (`rgb-d`)
参照: RT-DETR (arXiv:2304.08069)、`lyuwenyu/RT-DETR`、local `rotated_rtmdet_jax`

## 1. 結論

旧構成のmAP50は追加学習後も最大0.0527だった。原因は学習時間ではなく、次の2つの構造差で
ある。

1. RGB backbone出力をランダムな256ch融合層へ通していたため、学習済みRGB neckを転送
   できず、pretrained featureの連続性がneck手前で切れていた。
2. 全予測を256 learned queryのone-to-one Hungarian matchingだけに任せ、dense detectorが持つ
   空間事前分布、one-to-many positives、良質なquery初期値がなかった。

採用構造は、dense headとquery headを競合させず役割分担する。

```text
RGB (HGNetV2-B2, pretrained) ───────────────┐
                                             ├─ RGB + beta[level] * depth
Depth (HGNetV2-B0) -> 1x1 adapter ──────────┘   beta初期値 = 0
  -> pretrained-compatible ChannelMapper (P3-P6)
  -> Rotated RTMDet dense head
       - 全pyramid cell: class + ltrb + angle
       - DynamicSoftLabelAssigner / rotated IoU / one-to-many
       - dense score top-k + decoded xywh + angle
  -> Deformable encoder memoryから同じtop-k cellをgather
  -> top-k xywh + angleをdecoder referenceに使用
  -> 4-layer iterative Deformable decoder（xywh/angleを各層でrefine）
  -> RotatedDeformableDETRHead / Hungarian one-to-one / NMS-free final
```

## 2. 不変条件

- `depth_beta == 0`では、backbone出力がRGB backbone出力とbit-exactに一致する。
- RGB feature channelは`[384, 768, 1536]`のままneckへ渡す。
- source checkpointのRGB backbone 360 tensorsとChannelMapper 12 tensorsをshape変換なしで
  転送する。
- hybridでは形状互換なencoder、decoder、level embeddingを転送する。旧query回帰headは
  spatial referenceの意味が異なるため転送しない。dense head、query head、depth adapter/gateを
  新規初期化する。
- dense pyramid flatten順とencoder memory flatten順は同一でなければ例外にする。
- dense headは全cellをO2Mで学習し、そのscore/boxがquery選択へ直接使われる。
- decoder attentionは4D `xywh` referenceを使う。5番目のangleはdense候補から初期化し、
  attentionとは分離して各decoder layerのOBB回帰branchで反復更新する。
- final predictionはquery headのNMS-free出力。pure dense configではrotated NMSを使う。

## 3. lossとcurriculum

### RIoU 5 epoch

- dense: Quality Focal + SmoothL1 + Rotated IoU
- query各層: Focal + SmoothL1 + Rotated IoU
- dense assignment: Dynamic Soft Label、top-k 13、rotated IoU
- query assignment: Hungarian、Focal/RBox L1/Gaussian cost

### GWD 15 epoch

RIoU bestをweights-onlyで読み、dense/queryのIoU lossをGWDへ交換する。optimizer/schedulerは
新規にする。RIoUは初期の重なり学習、GWDは角度周期性と細長いOBBの安定化を担う。

### GWD追加50 epoch

GWD 15 epochの実測bestをweights-onlyで読み、さらに50 epoch学習する。epoch 38までは
高LRを維持し、残り12 epochを1/10 LRで収束させる。validationは毎epoch実行し、長期化で
悪化してもbest checkpointを保持する。

## 4. 実装境界

| path | 責務 |
|---|---|
| `yopo/models/backbones/dual_rgbd.py` | `RGBDResidualBackbone`、RGB恒等経路とdepth残差 |
| `yopo/engine/hooks/rgbd_obb_transfer.py` | RGB/neckと互換detector tensorsの監査可能な転送 |
| `yopo/models/dense_heads/rotated_rtmdet_head.py` | dense OBB decode、O2M loss、rotated NMS |
| `yopo/models/detectors/rotated_rt_detr.py` | dense top-kからquery content/referenceを構築 |
| `yopo/models/layers/transformer/deformable_detr_layers.py` | 5D OBB回帰時も4D attention referenceを反復更新 |
| `configs/yopo/rotated_rtmdet_stem_rgbd_obb_dense_*.py` | pure dense比較/単独利用 |
| `configs/yopo/rotated_rt_detr_stem_rgbd_obb_hybrid_*.py` | 採用hybrid curriculum |

## 5. pure denseとhybridの意味

- pure denseはRTMDet型の空間学習だけを評価・利用する最短経路である。
- hybridはdense headを補助lossだけにせず、top-k候補がdecoder初期値を実際に決める。
- query数を増やすだけの変更ではない。query contentもreferenceも画像ごとのdense候補から得る。
- one-to-manyとone-to-oneは同じGTを別目的で学習する。前者はrecall/局所化、後者は重複除去済み
  final setを担当する。

## 6. 受入結果

- RGB恒等経路: CPU unit testでrtol=0/atol=0。
- 実checkpoint転送: RGB backbone 360 + neck 12 tensors exact。
- hybrid互換転送: 合計525 tensors（RGB 360、neck 12、decoder 88、encoder 64、level embed 1）。
- GPU smoke: batch2、2 iterations、dense/queryの全loss finite。angle修正後のquery RIoU lossは
  上限2.0固定から約1.67へ改善し、validation inference成功。
- dense positives: batch2で227–333、batch32で3624。one-to-manyが実際に成立。
- RTX 5090 batch32: peak 27,249 MiB、OOMなし。
- angle-aware hybrid RIoU 5 epoch: best mAP50 0.0919。
- hybrid GWD 15 epoch: best mAP50 0.1184（旧best 0.0527の約2.25倍）。
- hybrid GWD追加50 epoch: best epoch 49、mAP50 0.1955。旧bestの約3.71倍、
  GWD 15 epoch bestから+0.0771。epoch 50は0.1937のためepoch 49を採用する。
- 最終bestのfull validation（conf 0.05）: VOC07 mAP50 0.2523、recall 0.6048、
  precision 0.1985、mean matched rIoU 0.6750、GT 27,723、prediction 84,478。
- rotated IoU 0.5、score降順greedy one-to-one matchingでのF1最大点は
  `conf=0.3355816305`、F1 0.3248、precision 0.2565、recall 0.4425。
  可視確認用conf 0.2はrecall 0.6046を維持し、6画像へ各100 queryをNMSなしで描画した。
  同じ6画像についてbest-F1 conf版も出力した。
- 最終checkpoint: `work_dirs/rtdetr_stem_rgbd_obb_hybrid_angle_gwd_continue50_v2_retry/`
  `best_rbbox_mAP_50_epoch_49.pth`。SHA256:
  `b4e6775474a6dbc1e24a9c70288c4777d93b0f0d857772e1caca7f85fd86d34c`。
- 同一RIoU bestの枝別評価: final query 0.0919、dense単独 0.0051。denseはproposal生成、
  queryはNMS-free final setという役割分担を採用する。

## 7. 完了条件

1. [x] focused test/lint/config buildが成功する。
2. [x] hybrid RIoU 5 epochを完走し、旧best mAP50=0.0527と比較する。
3. [x] RIoU bestからhybrid GWD 15 epochを完走する。
4. [x] GWD bestから追加50 epochを完走し、長期runのbestを選ぶ。
5. [x] 最終bestをfull validationし、F1最大thresholdとconf 0.2 overlayを出力する。
6. [x] pure denseは同じpretrained pyramidを使う独立fallbackとして残す。
