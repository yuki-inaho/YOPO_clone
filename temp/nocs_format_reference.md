# NOCS 合成データ / スモーク リファレンス（smoke-designer 解析 2026-06-19）

> 出典: smoke-designer エージェント（Claude/Sonnet）の最終レポート。`scripts/gen_synthetic_nocs.py` / `temp/smoke_nocs_r50_1iter.py` / `scripts/smoke_infer.py` の設計根拠。作業エージェント・監査エージェント共通の参照資料。コードで再確認した上で利用すること（出力はもう1段の主張であり ground truth ではない）。

## (a) NOCS on-disk フォーマット（`data_root/` 配下）
```
real/
  train_list.txt        # 各行: scene_1/0000  （拡張子・サフィックス無し。コードは basename stem を [:4] でスライス）
  test_list.txt
  scene_1/
    0000_color.png      # H=480 W=640 uint8 RGB
    0000_depth.png      # uint16 PNG, mm
    0000_label.pkl      # train ラベル
camera/
  train_list.txt
  val_list.txt
  scene_1/
    0000_color.png
    0000_label.pkl
camera_full_depths/
  scene_1/
    0000_composed.png   # uint16 PNG, camera split の depth（img_path の "/camera" を "/camera_full_depths" に replace して構築）
segmentation_results/
  REAL275/  results_test_scene_1_0000.pkl   # test ラベル
  CAMERA25/ results_val_scene_1_0000.pkl    # val ラベル
```

### train ラベル pkl キー（`*_label.pkl`）
- `class_ids`: (N,) int32 **1-indexed (1..6)**
- `instance_ids`: (N,) int32
- `bboxes`: (N,4) float32 **[y1, x1, y2, x2]**
- `translations`: (N,3) float32 カメラ系メートル
- `rotations`: (N,3,3) float32 回転行列
- `sizes`: (N,3) float32 正規化 NOCS モデルサイズ
- `scales`: (N,) float32 インスタンス毎スケール

### test ラベル pkl キー（`segmentation_results/*/results_*_*.pkl`）
- `gt_class_ids`: (N,) int32 1-indexed
- `gt_bboxes`: (N,4) float32 [y1, x1, y2, x2]
- `gt_RTs`: (N,4,4) float32 同次変換（最終行 [0,0,0,1]）
- `gt_scales`: (N,3) float32 メトリック 3D サイズ
- `gt_handle_visibility`: (N,) float32

### intrinsics（`SPLIT_INFO` ハードコード, `[fx, fy, cx, cy]`）
- `real`: `[591.0125, 590.16775, 322.525, 244.11084]`
- `camera`: `[577.5, 577.5, 319.5, 239.5]`

## (b) カテゴリ ID マッピング
| 1-indexed(pkl) | 0-indexed(model) | 名称 |
|---|---|---|
| 1 | 0 | bottle |
| 2 | 1 | bowl |
| 3 | 2 | camera |
| 4 | 3 | can |
| 5 | 4 | laptop |
| 6 | 5 | mug |
対称クラス(0-indexed): `sym_ids = [0,1,3]`（bottle, bowl, can）

## (c) 学習ループ / batch size
- base config: **EpochBasedTrainLoop, max_epochs=12**。
- smoke: EpochBased のまま `max_epochs=1`, `val_interval=9999`, **val を無効化するなら val_dataloader/val_cfg/val_evaluator を 3点とも None**（val_cfg=None だけだと `ValueError: ... should be either all None or not None`）。
- batch_size=**2**（collation を1サンプル超で検証しつつ OOM 回避）, `num_workers=0`。

## (d) 不確実だった点 / 仮定
- `models_info_path`（meta_keys）: dataset は未設定。`Pack9DPoseInputs` が `if key in results` でガード → 安全に省略。
- `frame_id`: 常に4桁ゼロ詰め（`{fi:04d}`）。
- `camera_full_depths` パス: `.replace("/camera", "/camera_full_depths")` 依存（split ディレクトリ名が `camera` なので成立）。
- `file_name = "/".join(img_file.split("/")[1:])`: 先頭パス要素を除去。data_root が1階層先頭を持つ前提で成立。

## (e) 初回失敗リスク（重要度順）
1. **torchvision resnet50 DL**: `backbone.init_cfg=Pretrained torchvision://resnet50` が初回 HTTP fetch。オフラインなら hang。回避: smoke config で `model = dict(backbone=dict(init_cfg=None))`。※本環境はネット有（checkpoint DL 済み）なので通常は問題なし。
2. **FilterAnnotations で全消し**: Resize 後 bbox が 1e-2×1e-2 未満だと frame=None → collate 崩壊。生成器は 50–150px bbox。発生時は `--n 8` 以上。
3. **YOLOXHSVRandomAug の scope 登録**: yopo scope 未登録だと config 解決時 KeyError。`yopo/datasets/transforms/__init__.py` を確認。
4. **test_step が `intrinsic` を要求**: head の `predict_single_img` が `img_meta['intrinsic']` で 3D 並進を復元。`smoke_infer.py` は test pipeline の meta_keys に `intrinsic` を明示。
5. **depth 欠如で silent None**: `parse_data_info` は depth 不在時に例外でなく None。全 frame None で空 dataset → 下流で難解クラッシュ。実行前に `ls data/nocs_smoke/real/scene_1/` で確認。
