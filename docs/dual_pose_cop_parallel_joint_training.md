# CoP / parallel 3D pose head 共同学習

## 1. 目的と結論

同じ DINO query、同じ 2D Hungarian 対応、同じ RGB-D feature から、次の二つの
3D pose 経路を同時に教師あり学習する。

- **parallel**: `z`、`size`、`rotation` を同じ decoder feature から独立予測する。
- **CoP chain**: `z -> size -> rotation` の順に、前段出力と depth query context を
  後段へ条件付けする。

採用 checkpoint は
`work_dirs/nocs_custom_fruit_rgbd_dual_pose_joint/best_3d_iou_0.50_epoch_10.pth`
である。同一重みを inference config だけ切り替えて両経路から読める。10 epoch
後の 3D IoU@0.50 は parallel `0.2947`、CoP `0.4878`。CoP 単独の開始値
`0.5143`に対して 94.85% を保持しつつ、CoP の pose 10 degree / 10 cm は
`0.3491 -> 0.3540`へ改善した。

## 2. well-defined な対応関係

画像 `i` の query `q` と GT `g` の対応は、3D予測経路に依存させず次だけで決める。

```text
C(i,q,g) = 2 C_focal + 5 C_L1(xywh) + 2 C_GIoU
```

Hungarian one-to-one matching の結果を `M_i` とする。parallel と CoP は同じ
`(q,g) in M_i` を使うため、二つの head が別の物体を説明して比較不能になることは
ない。3D translation / rotation cost を matching から外したのは、初期品質の低い
parallel 3D値が共通対応を壊さないためである。2D detector は引き続き DINO の
one-to-one query 構造であり、production graph は NMS-free のままである。

最終 decoder layerの pose lossは、正規化係数を省略して書くと次である。

```text
L_pose = L_parallel + L_CoP
L_parallel = 25 L_z + 25 L_size + 2.5 L_rotation
L_CoP      = 25 L_z_chain + 25 L_size_chain + 2.5 L_rotation_chain
```

実装上は `cop_prediction_mode="auxiliary"` が parallel を primary tensor、CoPを
`loss_*_chain` として返す。`cop_aux_loss_weights={z:1,size:1,rotation:1}` により
両経路を同じ基礎loss weightで監督する。既定値は過去config互換の
`{z:1,size:3,rotation:2}` のままであり、共同学習configだけが上書きする。

encoder proposalには decoder専用OBB予測がない。そのため encoder は
`obb_aux_supervision=False`を明示し、cls / bbox / IoU / center / parallel poseを
学習する。decoderでOBB lossが有効なのにOBB tensorが欠けた場合は従来どおり
fail-fastする。

## 3. 最適化とRTX 5090容量

開始点は検証済みCoP checkpoint
`nocs_custom_fruit_rgbd_nmsfree_control_continue5/best_3d_iou_0.50_epoch_5.pth`
である。CoPを忘却させず、古いparallel branchを追いつかせるため、基礎LR
`1e-6`に対してparallelの `reg_z/reg_size/reg_rotation` だけ `lr_mult=50` とした。
CoP、depth sampler、共有 detector は `lr_mult=1`。batch sizeは20、FP32、
10 epoch、validationは5 epochごとである。

1 epoch smokeは15 updateすべてfinite、NaN/OOMなし、peak memoryは
約`23,457 MiB`で、32 GiB上限に約8.5 GiBの余裕を残した。CoP LR multiplierを
`50 -> 5 -> 1`と一変数ずつ比較し、1だけが1 epoch後にCoP IoU@0.50を
`0.4989`（開始値の97.0%）保持したため採用した。

## 4. 実測結果

validation 50画像、GT 3,894個、query 150、同じNOCS evaluatorで測定した。

| checkpoint / path | AP50 | IoU@.50 | IoU@.75 | pose 10°/10cm | translation 10cm |
|---|---:|---:|---:|---:|---:|
| 開始CoP | 0.3520 | 0.5143 | 0.1653 | 0.3491 | 0.6064 |
| joint epoch 5 / parallel | 0.3460 | 0.2539 | 0.0515 | 0.0025 | 0.4865 |
| joint epoch 5 / CoP | 0.3460 | 0.4817 | 0.1553 | 0.3372 | 0.6329 |
| joint epoch 10 / parallel | 0.3410 | 0.2947 | 0.0703 | 0.0082 | 0.5370 |
| joint epoch 10 / CoP | 0.3410 | 0.4878 | 0.1495 | 0.3540 | 0.6512 |

epoch 10を採用する。理由はparallelの主gateであるIoU@.50がepoch 5から
`+0.0408`、CoPも`+0.0061`で、CoP pose / translationが開始点を超えたためである。
parallel rotationはまだCoPより大幅に弱く、現時点でproduction既定をparallelへ
切り替える根拠はない。同一checkpointに両経路を保持し、CoPをproduction候補、
parallelを独立比較可能な継続学習候補とする。

## 5. 再現手順

共同学習:

```bash
uv run python tools/train.py \
  configs/yopo/nocs_custom_fruit_rgbd_dual_pose_joint.py \
  --work-dir work_dirs/nocs_custom_fruit_rgbd_dual_pose_joint
```

同一checkpointからCoP prediction dump:

```bash
uv run python tools/analysis_tools/dump_rgbd_3dbbox_predictions.py \
  configs/yopo/nocs_custom_fruit_rgbd_dual_pose_joint_cop_inference.py \
  work_dirs/nocs_custom_fruit_rgbd_dual_pose_joint/best_3d_iou_0.50_epoch_10.pth \
  work_dirs/nocs_custom_fruit_rgbd_dual_pose_joint/predictions_cop_epoch10.pkl \
  --work-dir work_dirs/nocs_custom_fruit_rgbd_dual_pose_joint/eval_cop_epoch10
```

parallel側はconfigを
`nocs_custom_fruit_rgbd_dual_pose_joint_parallel_inference.py`へ、出力名を
`predictions_parallel_best.pkl`へ置き換える。

可視化はvalidationで探索した実用丸め値conf `0.35` / 2D NMS IoU `0.35`を使う。
これは表示・診断専用であり、model graphやNOCS AP計算にはNMSを入れない。
2D NMSで残ったquery IDだけを同じまま3D cuboidへ適用する。

```bash
uv run python tools/analysis_tools/render_rgbd_training_result_overlays.py \
  work_dirs/nocs_custom_fruit_rgbd_dual_pose_joint/predictions_cop_epoch10.pkl \
  configs/yopo/nocs_custom_fruit_rgbd_dual_pose_joint_cop_inference.py \
  work_dirs/visualizations/dual_pose_joint/cop_conf035_nms035 \
  --checkpoint work_dirs/nocs_custom_fruit_rgbd_dual_pose_joint/best_3d_iou_0.50_epoch_10.pth \
  --num-images 6 --all-predictions --score-threshold 0.35 \
  --nms-iou-threshold 0.35 --max-predictions 150
```

閾値は同じvalidation splitに合わせた値なので、汎化性能の最終主張には独立test
splitで再確認する。

## 6. 成果物整合性

| artifact | SHA-256 |
|---|---|
| epoch 10 best checkpoint | `71f94ecf51f7853ff8cdc632a7571dc485726756f51280b9da06723455254755` |
| CoP prediction dump | `d48baa31fae849e3d933dbd568fb42c5585f253ecc474e7c6f3c1ec98d9787a9` |
| parallel prediction dump | `86ab93d1681d7aa6af7b212c5dbc11738fc2d65f67d7c4a803b592090a5f9035` |
| CoP overlay manifest | `2b055bd68cfc56fb6c7c88d8f1e9dac7189085fe7ecc6d12a6fede5fd13fb11f` |
| parallel overlay manifest | `05dd2e24f5eca5ffef2540bf8c1022064eaf4f7100dcde30eea930b5063977cd` |

checkpoint、dump、PNGは大容量のためGitには含めず、`work_dirs` symlink先の
`/workspace/YOPO_clone/work_dirs`に保存する。config、実装、test、本書だけをGitで
追跡する。
