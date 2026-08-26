# RGB-D 3D OBBモデル構造・parameter・計算量解析

更新日: 2026-08-26

## 1. 目的と対象

現行YOPO RGB-D 3D OBBモデルについて、次を同じ契約で再現可能にする。

- モデル構造と主要component
- 総parameter数、学習対象／凍結parameter数、module別内訳
- parameter自体が占めるmemory
- 実データpipelineを通した入力解像度別FLOPs概算
- 未集計operatorを含むFLOPs値の限界
- 将来のlatency、throughput、peak VRAM実測方法

解析対象configは次の2つである。

1. 現行Stage 11 KFIoU probe:
   `configs/yopo/nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_stage11_kfiou_probe5.py`
2. 2025＋2026共同学習・事前変換済み800x600入力:
   `configs/yopo/nocs_fruits_2025_2026_rgbd_3dbbox_stage10_mal_native_800x600.py`

Stage 11のKFIoUはloss差し替えであり、学習可能parameterを追加しない。checkpointもweight値を
変更するだけでmodel structureとparameter数を変更しない。このため、以下のparameter結果は
両configで共通である。

## 2. 現行モデル構造

| 階層 | 実装 | 主な契約 |
|---|---|---|
| Detector | `DINO9DCenter2DPose` | RGB-Dから2D detection、center、z、rotation、size、projectionを予測 |
| Data preprocessor | `DetDataPreprocessor` | 4 channel、追加resizeなしのconfigでは固定canvasをそのまま使用 |
| Backbone | `RGBDResidualBackbone` | RGB backbone、depth backbone、depth adapterを統合 |
| Neck | `ChannelMapper` | multi-scale featureをencoder用channelへ変換 |
| Head | `DINO9DCenter2DPoseHead` | DINO detectionとCoP 3D pose branch |
| Encoder | `DeformableDetrTransformerEncoder` | 6 layers |
| Decoder | `DinoTransformerDecoder` | 6 layers |
| Query | `Embedding` | 256 queries |
| Denoising | `CdnQueryGenerator` | DINO contrastive denoising query生成 |

入力はRGB 3 channelとmapped depth 1 channelを結合した`(4, H, W)`である。

## 3. Parameter数

2026-08-26時点の現行configを`MODELS.build()`した直後の集計値である。

| Component | Parameters | 全体比 |
|---|---:|---:|
| Backbone | 9,940,979 | 22.75% |
| Neck | 4,229,120 | 9.68% |
| BBox＋3D pose head | 12,230,998 | 27.99% |
| Encoder | 7,693,056 | 17.60% |
| Decoder | 9,472,768 | 21.68% |
| Query/memory変換/DN/direct parameters | 133,120 | 0.30% |
| **合計** | **43,700,041** | **100%** |

- 学習対象parameter: `43,700,041`
- 凍結parameter: `0`
- FP32 parameter buffer: `174,800,164 bytes`、約`166.7 MiB`
- BF16/FP16換算: 約`83.4 MiB`
- registered buffer: `39,078` elements、`156,720 bytes`

これはparameter本体だけの値であり、gradient、optimizer state、activation、CUDA allocatorの
reserved memoryを含まない。学習時peak VRAMはparameter数から断定せず実測する。

### 3.1 再現コマンド

リポジトリrootから実行する。

```bash
.venv/bin/python - <<'PY'
from mmengine.config import Config
from yopo.registry import MODELS
from yopo.utils import register_all_modules

config_path = (
    "configs/yopo/"
    "nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_stage11_kfiou_probe5.py"
)
register_all_modules()
cfg = Config.fromfile(config_path)
model = MODELS.build(cfg.model)

total = sum(parameter.numel() for parameter in model.parameters())
trainable = sum(
    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
)
parameter_bytes = sum(
    parameter.numel() * parameter.element_size() for parameter in model.parameters()
)
print({
    "model": type(model).__name__,
    "total": total,
    "trainable": trainable,
    "frozen": total - trainable,
    "parameter_bytes": parameter_bytes,
})
PY
```

parameter数はforward traceや入力解像度に依存しない。freeze hookを追加したconfigでは、
hook適用前後のどちらを集計したか明記する。

## 4. FLOPs概算

repo内の`tools/analysis_tools/get_flops.py`は、MMDetection公式実装と同様に実dataloaderから
1画像を読み、data preprocessor適用後のtensorを`mmengine.analysis.get_model_complexity_info`
へ渡す。単純な`(C,H,W)` dummy tensorより、RGB-D detectorの実forward契約に近い。

| Config | Model入力 | MMEngine集計値 | Parameters |
|---|---:|---:|---:|
| 現行Stage 11 | 640x445 | 67.111G | 43.7M |
| 共同学習800x600 | 800x600 | 0.109T（約109G） | 43.7M |

800x600は640x445よりpixel数が約1.685倍で、今回の集計FLOPsは約1.624倍だった。
256 queryなど解像度に依存しない演算もあるため、全演算が面積比で増えるわけではない。

### 4.1 再現コマンド

GPU学習中でも干渉しないCPU trace例。実データを読むため、対応dataset rootが必要である。

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python \
  tools/analysis_tools/get_flops.py \
  configs/yopo/nocs_fruits_detection_Jun30_2025_rgbd_3dbbox_stage11_kfiou_probe5.py \
  --num-images 1

CUDA_VISIBLE_DEVICES='' .venv/bin/python \
  tools/analysis_tools/get_flops.py \
  configs/yopo/nocs_fruits_2025_2026_rgbd_3dbbox_stage10_mal_native_800x600.py \
  --num-images 1
```

### 4.2 FLOPs値の解釈

上表を論文用の完全な演算量と断定しない。今回のtraceでは少なくとも次のoperatorが
unsupportedとして報告された。

- `grid_sampler`
- `softmax`
- `batch_norm`、`group_norm`、`layer_norm`
- `pad`、`max_pool2d`
- 多数のelement-wise演算、三角関数、`topk`

またloss moduleやtraining-only branchはinference traceで呼ばれない。従って現状値は
**同じ解析器・同じforward modeでmodel variantや解像度を比較するための下限寄り概算**として使う。
技術報告や論文で絶対値を使う場合は、unsupported operator handlerを追加し、集計規約
（multiply-addを1 FLOPと数えるか2 FLOPsと数えるか）も明記する。

MMDetection公式もFLOPs toolをexperimentalとし、custom operatorが集計されない場合があるため
絶対値の再確認を求めている。

- MMEngine complexity API:
  <https://mmengine.readthedocs.io/en/stable/api/generated/mmengine.analysis.get_model_complexity_info.html>
- MMDetection公式`get_flops.py`:
  <https://github.com/open-mmlab/mmdetection/blob/main/tools/analysis_tools/get_flops.py>
- MMDetection useful tools:
  <https://mmdetection.readthedocs.io/en/stable/user_guides/useful_tools.html>

## 5. 推奨ツール構成

### 5.1 既定: repo内MMEngine解析

新しい依存を増やさず、次を正本にする。

- parameter確定値: `named_parameters()` / `numel()`
- FLOPs比較値: `tools/analysis_tools/get_flops.py`
- dataset/inference throughput: `tools/analysis_tools/benchmark.py`

### 5.2 実測: PyTorch ProfilerとCUDA memory stats

GPUが空いているときに、warmup後のlatency、operator別CUDA時間、tensor shape、peak allocated、
peak reservedを測る。`torch.cuda.reset_peak_memory_stats()`後の
`torch.cuda.max_memory_allocated()`と`max_memory_reserved()`を記録する。

- PyTorch Profiler:
  <https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>
- CUDA peak allocated memory:
  <https://docs.pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html>

GPU学習が動いている状態で別profileを開始しない。別GPUを明示するか、既存run終了後に測る。

### 5.3 精密FLOPsが必要な場合: custom operator handler

MMEngineの`FlopAnalyzer`または同系統のfvcore `FlopCountAnalysis`へYOPO固有handlerを追加する。
fvcoreはoperator別・module別集計とcustom handlerを提供する。

- fvcore FLOP count:
  <https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md>

`ptflops`、THOP、`torchinfo`は現環境に入っていない。OpenMMLabのdata sampleを伴うforwardと
deformable attentionを扱うためだけに依存を増やす必要はなく、まず既存MMEngine経路を使う。

## 6. 将来の統合解析ツールに必要な出力

専用CLIを追加する場合は、configと任意checkpointを入力し、JSONとMarkdownへ次を保存する。

1. config path、checkpoint path、各SHA-256
2. model class、module tree、encoder/decoder layer数、query数
3. 総／学習対象／凍結parameterとmodule別内訳
4. parameter/buffer dtype別memory
5. 入力shape、集計FLOPs、operator別／module別内訳
6. unsupported operator、uncalled module一覧
7. warmup回数、計測回数、batch size、precision
8. latency p50/p95、throughput、peak allocated/reserved VRAM
9. PyTorch、CUDA、MMCV、MMEngine、GPU情報

FLOPs概算とGPU実測値を同じ欄へ混在させず、`static_analysis`と`runtime_profile`に分ける。
