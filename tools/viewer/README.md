# GauCho-3D viewer

学習した GauCho-3D の予測を見るためのビューア。3D は、カメラ内部パラメータから
起こした RGB-D 点群と同じ座標系に予測楕円体を重ねる。2D は同じ楕円体を画像へ
投影したものを描く。

Open3D は既定では入らない（GUI/GL 一式を引くため、学習機や CI には不要）。

## 使い方

```bash
just viewer-sync                       # open3d を入れる（1 回だけ）
just viewer-export CONFIG CKPT         # 10 frame の bundle を作る
just viewer                            # 3D
just viewer-2d                         # 2D（confidence の数値つき）
just viewer-overlays                   # GUI 無しで PNG + contact sheet
just viewer-validate                   # 全 frame が読めるかだけ確認
```

`CONFIG` は **その run が実際に使った展開済み設定**（`work_dirs/<run>/<stamp>/
vis_data/config.py`）を渡すこと。`temp/` の学習設定を渡すと、深度アンカーが
無効な親から継承されて距離が 0 付近になる。例:

```bash
just viewer-export \
  work_dirs/centre5/20260902_041031/vis_data/config.py \
  work_dirs/centre5/best_gaucho3d_shared_AP_20_epoch_3.pth
```

bundle は `work_dirs/viewer_bundle/` に出る（生成物なので追跡しない）。

## 操作

| キー | 動作 |
|---|---|
| `→` / `N` | 次の frame |
| `←` / `P` | 前の frame |
| `↑` / `]` | score threshold +0.05 |
| `↓` / `[` | score threshold −0.05 |
| `S` | confidence の数値表示切替（2D のみ） |
| `D` | RGB-D 点群の表示切替（3D のみ） |
| `E` | 楕円体の表示切替（3D のみ） |
| `R` | view をリセット（3D のみ） |
| `Q` / `Esc` | 終了 |

## score threshold の既定 0.35 について

F1 を最大にする値。val 330 枚、rotated NMS 0.2 の後に閾値、対応 IoU 0.5:

| 閾値 | precision | recall | F1 | 枚あたり検出 |
|---:|---:|---:|---:|---:|
| 0.20 | 0.6422 | 0.7822 | 0.7053 | 91.5 |
| 0.30 | 0.7876 | 0.7382 | 0.7621 | 70.4 |
| **0.35** | **0.8430** | **0.6982** | **0.7637** | 62.2 |
| 0.40 | 0.8868 | 0.6425 | 0.7452 | 54.4 |
| 0.50 | 0.9465 | 0.4602 | 0.6192 | 36.5 |

0.30〜0.40 は平坦。取りこぼしを嫌うなら 0.25、誤検出を嫌うなら 0.45 前後。
`just viewer 0.5` のように引数で上書きできる。

## 3D の色

楕円体は 3 本の主断面リングで描く。色は confidence で、オレンジが低く緑が高い。
Open3D の LineSet に文字を載せられないため、3D 側に数値は出ない。数値が要る
ときは 2D を見ること。

## 構成

    tools/viewer/
      gaucho3d_viewer/app.py      3D ビューア（bundle 読み込み・点群・楕円体）
      gaucho3d_viewer/two_d.py    2D ビューアと静止画生成
      export_from_yopo.py         checkpoint から bundle を作る
