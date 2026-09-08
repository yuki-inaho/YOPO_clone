# YOPOセットアップ・RGB-D推論・ONNXデプロイ

## 対象

このリポジトリのリリース重みはNOCS向けのRGB-only YOPO R50です。したがって、`/home/inaho-omen/data/2026_tomato/standard` のRGB-Dペアを検証・列挙しますが、現行モデルへ深度画像を勝手に4チャネル目として連結しません。深度を実際に使うには、深度入力を含めて学習した別モデルと前処理・重みが必要です。

今回の実データはNOCSの評価ラベルではないため、10系列の試験は精度評価ではなく、ファイルペア、前処理、モデルロード、推論、出力保存のruntime smoke testです。

## セットアップ

Python 3.8を使うリポジトリのロックに合わせます。

```bash
uv python install 3.8
uv venv --python 3.8 .venv
uv sync --locked
uv pip install --python .venv/bin/python \
  -r requirements/runtime.txt plyfile pyyaml openmim 'setuptools<75' wheel
.venv/bin/mim install mmcv==2.2.0
uv pip install --python .venv/bin/python --no-build-isolation -e .
uv pip install --python .venv/bin/python -r requirements/deployment.txt
.venv/bin/python -m pip check
```

`pyproject.toml`には、upstreamの`setup.py`とuv用の最小PEP 621メタデータが混在するため、setup.py側の項目を`dynamic`として明示しています。これによりeditable installが再現可能になります。

## リリース重み

公式リリースページ:
`https://github.com/pitin-ev/YOPO/releases/tag/v1.0.0`

NOCS R50を取得します。

```bash
mkdir -p checkpoints
curl -fL -o checkpoints/nocs_yopo_real_camera_r50.pth \
  https://github.com/pitin-ev/YOPO/releases/download/v1.0.0/nocs_yopo_real_camera_r50.pth
sha256sum checkpoints/nocs_yopo_real_camera_r50.pth
```

期待するSHA-256は
`24e49cc17693a6b9eaff870688176d06df3b43458aeec93eb35de1b812193287`です。

## RGB-D 10系列のPyTorch推論

各sceneディレクトリからRGBとdepthのペアを1件選びます。実データには、同じstemで拡張子だけ異なるペア、`*_rgb`/`*_depth`ペアの両方があるため、列挙コードは両方に対応しています。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python tools/inference_tomato.py \
  --max-scenes 10 \
  --output work_dirs/tomato_inference.json
```

`tools/inference_tomato.py` はRGBのshape/dtype、depthのshape/`uint16`/有効画素率、RGBカメラ内部パラメータ、推論時間、上位候補をJSONへ保存します。

## ONNX exportとruntime

エクスポートはCPUで行います。MMCVのCUDA deformable-attention traceはこの環境でONNX展開結果と数値が一致しないため、CPU実装から標準ONNX演算へ展開し、ONNX Runtimeとの一致を必ず検証します。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
.venv/bin/python tools/deployment/export_yopo_onnx.py \
  --output work_dirs/yopo_onnx/yopo_nocs_r50.onnx
```

生成物は次の2つです。

- `work_dirs/yopo_onnx/yopo_nocs_r50.onnx`: 固定入力の推論グラフ
- `work_dirs/yopo_onnx/yopo_nocs_r50.json`: 入出力名、shape、クラス数、後処理、検証値

入力契約は`float32`のBGR、`[1,3,480,640]`、値域0..255です。グラフ内でBGR→RGBとYOPOのmean/std正規化を行います。出力は最終decoderのclass/bbox/center/z/rotation/sizeの6テンソルで、top-k、カメラ行列によるtranslation復元は`yopo.deployment.onnx.postprocess_pose_outputs`が行います。これはMMDeployの「backend graph + 明示的後処理」に相当する構成です。

```bash
OMP_NUM_THREADS=2 \
.venv/bin/python tools/deployment/run_yopo_onnx.py \
  --model work_dirs/yopo_onnx/yopo_nocs_r50.onnx \
  --max-scenes 10 \
  --output work_dirs/tomato_onnx_inference.json
```

現状のONNX artifactは、traceの安全性を優先してbatch=1、640x480固定です。異なる入力解像度やbatchを使う場合は、動的shapeを名乗らず、モデル前処理・proposal生成・後処理を含めて別途検証してください。

## 実行結果（2026-09-08）

- `torch==2.4.0+cu121`, `mmcv==2.2.0`, `mmengine==0.10.7`
- 公式checkpointのSHA-256一致
- PyTorch CUDA: 10系列すべて成功、初回約1459.5 ms、以後約143.3–203.8 ms/件
- ONNX checker/ONNX Runtime CPU: export検証 `max_abs=9.91821e-05`, `mean_abs=4.00964e-06`
- ONNX Runtime CPUの10系列: 全件成功、約10203.7–12333.8 ms/件
- 結果JSON: `work_dirs/tomato_inference.json`, `work_dirs/tomato_onnx_inference.json`

全候補の最大scoreは0.0158–0.0754程度で、NOCS学習済みモデルをトマト実データへ適用したout-of-domain smoke testとして解釈してください。これをトマト認識精度と解釈したり、深度を使った推定と解釈したりしないことが重要です。
