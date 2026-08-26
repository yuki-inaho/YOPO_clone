# Compact YOPO weight transplant / native Group Fisher smoke (2026-08-26)

## 目的とスコープ

800x600 RGB-D YOPO を 32 GB VRAM で物理 batch 20--24 に近づけるため、
既存 teacher の重みを再利用しつつ、既知の小型構成へ移す検証用 worktree である。
デプロイ形式への変換は対象外で、architecture surgery、weight transplant、
Group Fisher による FFN channel 選択、fine-tuning 前の smoke test を対象にする。

- worktree: `/home/kasm-user/Desktop/YOPO_hgnetv2_b1_mmrazor_smoke`
- branch: `experiment/hgnetv2-b1-mmrazor-smoke-20260826`
- 元リポジトリ: `/home/kasm-user/Desktop/YOPO_clone`
- 元リポジトリの進行中ファイル、学習環境、work dir は変更しない。

## 目標構成

| 項目 | teacher | compact candidate |
| --- | ---: | ---: |
| RGB backbone | HGNetV2-B2 | HGNetV2-B1 |
| Depth backbone | HGNetV2-B0 | HGNetV2-B0 |
| encoder layers | 6 | 4 |
| decoder layers | 6 | 4 |
| FFN hidden width | 2048 | 1024 |
| `d_model` | 256 | 256 |
| queries | 256 | 256 |
| parameters | 43,700,041 | 24,589,045 |

parameter 数は 19,110,996（約 43.7%）減る。`d_model=256` と query 数は維持する。
`d_model` の channel pruning は neck、encoder/decoder attention、全 prediction branch、
depth query context へ依存が広がるため、この smoke では扱わない。

compact config:

`/home/kasm-user/Desktop/YOPO_hgnetv2_b1_mmrazor_smoke/configs/yopo/nocs_fruits_2025_2026_rgbd_3dbbox_b1b0_e4d4_ffn1024_native_800x600.py`

RGB B1 は公式 pretrained weight から初期化する。trained depth B0 と形状互換な
YOPO 部分は teacher から移し、B1 化で形状が変わる RGB backbone、depth adapters、
neck は明示的に再初期化する。

## 参照した pruning 実装

private repository は `gh repo clone` で次へ取得した。

- clone: `/home/kasm-user/Desktop/stem_semseg_rgbd_dual_path_stdc_ref_20260826`
- origin: `git@github.com:yuki-inaho/stem_semseg_rgbd_dual_path_stdc.git`
- branch: `feat/stdc-cpu-pruning`
- commit: `7684ad216ca6c0593d585e97364f82b45376e0ea`

主な参照箇所:

- `src/jax_channel_pruning/importance.py`: per-example Taylor / Fisher
- `src/jax_channel_pruning/graph.py`: channel dependency group
- `src/jax_channel_pruning/analyzer.py`: 対象軸の検証
- `src/jax_channel_pruning/mutator.py`: original index を保った物理 slice
- `src/jax_channel_pruning/core.py`: importance、selection、mutation の分離
- `src/jax_channel_pruning/cost.py`: importance の cost normalization
- `src/rgbd_cross_linear_jax/pruning_adapter.py`: RGB-D model adapter
- `tests_cross_linear/test_jax_channel_pruning.py`: mask と物理 slice の parity

YOPO への移植で保持した契約:

1. 各 example の virtual gate gradient `A * dL/dA` を計算する。
2. 同じ dependency group に複数 site がある場合は site 間を先に加算する。
3. Taylor は `mean(abs(g))`、Fisher は `0.5 * mean(g^2)` とする。
4. importance 収集、channel 選択、checkpoint の物理 slice を分離する。
5. 選択後も original channel index を保存し、曖昧な layout は fail closed にする。
6. dense FFN の channel mask と物理 slice の出力 parity をテストする。

YOPO 実装:

- `yopo/pruning/group_fisher.py`
- `yopo/pruning/compact_yopo.py`
- `tools/pruning/collect_yopo_ffn_importance.py`
- `tools/model_converters/transplant_compact_yopo.py`
- `tests/test_compact_yopo_pruning.py`

現段階の安全な物理 pruning 対象は、各 transformer FFN の
first Linear output / bias と second Linear input の coupled hidden axis である。
backbone/fusion の residual-add channel と `d_model` はまだ物理 pruning しない。

## MMRazor smoke の結論

隔離 `.venv` にだけ MMRazor 1.0.0 を導入した。環境は PyTorch 2.8.0+cu128、
MMCV 2.2.0、MMEngine 0.10.7 である。

MMRazor import は MMCV を `<=2.1.0` に制限して停止する。version assertion だけを
一時的に回避して `register_all_modules()` を試しても、MMRazor 1.0.0 が import する
`torch.ao.quantization.fuser_method_mappings.reverse2` 等が PyTorch 2.8 では存在せず、
quantization backend の import で停止する。

既存 YOPO 環境を MMCV / PyTorch ごと downgrade する影響に対して、今回必要なのは
明示的な FFN dependency group の重要度収集と物理 slice だけである。そのため、
MMRazor を runtime dependency にせず、上記 private repo の小さい pruning core を
PyTorch 用に移植する方針とした。

## Weight transplant のルール

`strict=False` をそのまま使うのではなく、converter が次を行う。

1. 同名かつ同 shape の tensor だけを候補にする。
2. teacher の decoder 6 層を compact の先頭4層へ移す。
3. teacher の indexed prediction branch 6 を compact branch 4 へ明示 remap する。
   これは compact model の encoder proposal 用最終 branch を誤って teacher branch 4
   から読む事故を防ぐためである。
4. FFN 2048→1024 は pruning plan の original index で coupled axis を物理 slice する。
5. plan がない場合の data-free baseline は group L2 とし、report に明記する。
6. RGB B1、depth adapters、neck は再初期化対象として移植から除外する。
7. loaded / remapped / sliced / reinitialized / mismatched key と SHA-256 を JSON に残す。
8. `purpose=smoke` の plan は `--allow-smoke-plan` なしでは拒否する。

teacher checkpoint:

`/home/kasm-user/Desktop/YOPO_clone_artifacts/work_dirs/stage10_2d_anchor_mal_full_schedulefree_fixed_v2/best_AP50_epoch_85.pth`

SHA-256:

`13721d66bbc72f38ac99a927f4c50a2105dc6f945a38d858aa58318d8a86ce9b`

## 実行済み smoke test

### 1. 単体テスト

```sh
cd /home/kasm-user/Desktop/YOPO_hgnetv2_b1_mmrazor_smoke
.venv/bin/pytest -q tests/test_compact_yopo_pruning.py
```

結果: `6 passed`。次を確認した。

- compact config が 24,589,045 parameters、4+4 FFN group を構築する。
- tied site を square 前に加算する Fisher / Taylor の解析値一致。
- FFN coupled-axis slice と encoder proposal branch remap。
- dense masked FFN と physically sliced FFN の出力 parity。
- smoke plan の明示 opt-in。
- calibrated plan に必要な target FFN group が欠けた場合の fail-closed。

### 2. synthetic wiring smoke

2 batches、batch 2、16 synthetic tokens で teacher の12 FFN groupを通し、
4 samples の score を保存した。これは hook 配線検証専用である。

### 3. 800x600 実データ smoke

data root:

`/home/kasm-user/Desktop/YOPO_clone/data/fruits_rgbd_2025_2026_800x600_preprocessed`

2025/2026 の dataloader を交互に読み、各1 sample を使った。入力は両方とも
`4x600x800` で、RGB、mapped depth、annotation を実際の YOPO loss 経路へ渡した。
BatchNorm running statistics は固定し、DINO の training-only loss inputs を有効にした。

出力:

- plan: `/home/kasm-user/Desktop/YOPO_hgnetv2_b1_mmrazor_smoke/work_dirs/compact_b1b0_e4d4_ffn1024_smoke/real_data_2batch_group_fisher_smoke_plan.json`
- scores: `/home/kasm-user/Desktop/YOPO_hgnetv2_b1_mmrazor_smoke/work_dirs/compact_b1b0_e4d4_ffn1024_smoke/real_data_2batch_group_fisher_smoke_plan.scores.npz`
- partial checkpoint: `/home/kasm-user/Desktop/YOPO_hgnetv2_b1_mmrazor_smoke/work_dirs/compact_b1b0_e4d4_ffn1024_smoke/teacher_epoch85_real_data_smoke_fisher_partial.pth`
- report: `/home/kasm-user/Desktop/YOPO_hgnetv2_b1_mmrazor_smoke/work_dirs/compact_b1b0_e4d4_ffn1024_smoke/teacher_epoch85_real_data_smoke_fisher_partial.pth.report.json`

実データ smoke の結果:

- teacher groups scored: 12
- compact FFN groups sliced: 8
- loaded keys: 766
- loaded tensor elements: 18,208,063
- target parameters: 24,589,045
- unexpected keys: 0
- B1+B0 feature forward: 全 tensor finite

2 samples の plan は `purpose=smoke` であり、精度判断または本学習には使用しない。

## Production calibration と fine-tuning の推奨手順

GPU が空いた時点で、固定 checkpoint、固定 seed、固定 batch size で 2025/2026 を
交互に最低 512 samples 程度収集する。最初は batch 1--2 とし、calibration は
学習ではないため optimizer state や activation を長期間保持しない。

```sh
cd /home/kasm-user/Desktop/YOPO_hgnetv2_b1_mmrazor_smoke

.venv/bin/python tools/pruning/collect_yopo_ffn_importance.py \
  configs/yopo/nocs_fruits_2025_2026_rgbd_3dbbox_stage10_mal_native_800x600.py \
  /home/kasm-user/Desktop/YOPO_clone_artifacts/work_dirs/stage10_2d_anchor_mal_full_schedulefree_fixed_v2/best_AP50_epoch_85.pth \
  work_dirs/compact_b1b0_e4d4_ffn1024/production_group_fisher_plan.json \
  --purpose selection \
  --estimator group_fisher \
  --batches 512 \
  --batch-size 1 \
  --remaining 1024 \
  --divisor 16 \
  --sampling balanced_years \
  --data-root /home/kasm-user/Desktop/YOPO_clone/data/fruits_rgbd_2025_2026_800x600_preprocessed \
  --device cuda

.venv/bin/python tools/model_converters/transplant_compact_yopo.py \
  /home/kasm-user/Desktop/YOPO_clone_artifacts/work_dirs/stage10_2d_anchor_mal_full_schedulefree_fixed_v2/best_AP50_epoch_85.pth \
  configs/yopo/nocs_fruits_2025_2026_rgbd_3dbbox_b1b0_e4d4_ffn1024_native_800x600.py \
  work_dirs/compact_b1b0_e4d4_ffn1024/teacher_epoch85_group_fisher_partial.pth \
  --pruning-plan work_dirs/compact_b1b0_e4d4_ffn1024/production_group_fisher_plan.json
```

生成 checkpoint を compact config の `load_from` として読む。初期段階は新規 B1、
depth adapters、neck と sliced FFN の回復を優先し、低 LR で全体 fine-tuning する。
学習前に次を別々に測る。

1. FP16 で physical batch 12, 16, 20, 24 の 800x600 one-step capacity。
2. teacher と compact の validation AP / pose metrics。
3. plan なし group L2、Group Fisher、必要なら Taylor の同一条件比較。
4. NaN/Inf、peak VRAM、step time、checkpoint resume。

現在 GPU は別プロセスが約 31.5 GiB 使用中だったため、800x600 の CUDA capacity
probe と 512-sample production calibration は未実施である。
