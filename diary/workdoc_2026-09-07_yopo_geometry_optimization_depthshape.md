# YOPO geometry optimization / Depth-shape connection 作業書

日付: 2026-09-07（実行記録はJST）。対象: `YOPO_clone`、branch `rgb-d`。
本書はRGB=s / depth=n raw-featureモデルに対する4改善の、公開用作業書兼設計書である。

## 1. 目的と完了条件

採用済みRGB-Dモデルを同一親checkpointとして、次の4仮説を単一変数に近い形で検証する。

1. bbox、pose、depth sampling、CoP、GauCho decodeをFP32 islandへ移す。
2. 幾何最終出力matrixだけをAMUSEの補助AdamWへ移し、中間matrixはMuonに保つ。
3. shape head以外を完全凍結し、KLDの中心項あり/なしを比較する。
4. Depth queryをzero-init residual adapter経由でGauCho shape headへ直接接続する。

既定flagは従来挙動を維持し、既存checkpointの暗黙fallbackを禁止する。全候補は同じ
B-best、同じ181枚validation、batch 25、最大15 epoch、5 epoch間隔で比較する。
リリース公開、データsplit変更、YOLO検出headの導入は本作業の非ゴールとした。

完了は、実装のfocused/full回帰、設定構築、checkpoint対応、独立評価、事前規則による
選抜、公開可能なsource/test/config/diaryの同期で判定する。性能改善自体は保証条件にせず、
棄却結果も仮説検証の成果として残す。

## 2. 設計

```text
BF16 decoder hidden ────────────────┬─> classification
                                    └─> FP32 geometry island
                                         ├─> box / center / z / pose / CoP
Depth pyramid -> query sampler ──────────┤
                                         └─> zero-init adapter
                                              + geometry hidden -> GauCho shape

AMUSE: geometry final matrices -> auxiliary AdamW
       other eligible matrices  -> Muon
```

Depth adapterは256→256のLinearをprediction layerごとに置き、weight/biasを0で初期化する。
したがって既存weightのロード直後はshape出力を変えず、学習によってのみDepth情報を加える。
モデルにはtwo-stage予約slotを含む5 adapter（328,960 parameter）があるが、decoderは4層のため
activeな4 adapter（263,168 parameter）だけが更新される。

shape比較では正規表現を`fullmatch`するfail-closed hookを用い、30 tensor / 686,430
parameterだけを学習可能にする。0件一致や過剰一致をprefix fallbackで救済しない。

## 3. 比較契約

| ID | 同一親から加える差分 | 推論parameter | 学習parameterの特記事項 |
|---|---|---:|---|
| B | 採用済み基準 | 36,523,656 | 全体 |
| D | geometry FP32 | 36,523,656 | 全体 |
| E | D + final matrix auxiliary AdamW | 36,523,656 | 全体 |
| F-on | E + shape-only、中心項あり | 36,523,656 | 686,430のみ |
| F-off | F-onから中心項だけ除外 | 36,523,656 | 686,430のみ |
| G | E + Depth-to-shape adapter | 36,852,616 | 全体、うちadapter 328,960 |

適格条件は同一epochでshared AP25 ≥ `.3108`、strict 3D AP25 ≥ `.1827`、
HBB/ellipse/projectionがBから各`.005`以内、invalid prediction=0。適格候補間は
sharedとstrict 3DのPareto非劣候補を残す。同率時だけray absolute median、depth extent
ratioの1からの距離、HBBの順で判定し、結果を見て閾値や重みを変更しない。

## 4. 実行チェックリスト

- [x] 変更前checkpoint、独立評価、dtype、optimizer、GPU上限を固定した。
- [x] 4契約の期待失敗テストを作成し、未実装理由だけでREDになることを確認した。
- [x] 後方互換flag付きのgeometry FP32 islandを実装・単体検証した。
- [x] AMUSE aux分類とexact freeze hookを実装・単体検証した。
- [x] zero-init Depth-to-shape adapterを実装し、初期等価性とgradientを確認した。
- [x] D/E/F-on/F-off/Gを同一親から学習し、各bestを181枚で独立評価した。
- [x] 事前規則でD15/G10をPareto集合に残し、ユーザー判断でG10を主採用、D15をshared向け代替とした。
- [x] focused/full pytest、ruff、diff/config/checkpoint監査を完了する。
- [x] private path/data/modelを除外し、公開変更をcommit/pushする。

## 5. 結果と選抜

| 候補 | HBB AP50 | ellipse | projection | shared AP25 | strict 3D AP25 | ray median mm | depth ratio | 適格 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B | .848343 | .848167 | .825071 | .315796 | .187744 | 8.4846 | 1.8235 | yes |
| **D15（shared代替）** | **.863438** | **.864990** | **.846640** | **.353354** | .220186 | **7.6803** | 1.6961 | yes |
| E15 | .881085 | .881639 | .859584 | .310539 | .241219 | 8.5413 | 1.4156 | no: shared |
| F-on15 | .848343 | .848167 | .828359 | .310222 | .187744 | 8.4836 | 1.4535 | no: shared |
| F-off5 | .848343 | .848167 | .821415 | .274820 | .187744 | 8.4518 | 1.2040 | no: shared |
| **G10（主採用）** | **.871735** | **.872072** | **.850226** | **.313780** | **.236604** | 8.5308 | **1.4567** | yes |
| G15 | .880279 | .880763 | .858270 | .308193 | .249705 | 8.5691 | 1.4109 | no: shared |

D15とG10は一方が他方を支配しない。事前規則でPareto集合を確定後、ユーザー判断により
strict 3Dが高いG10を現行の主採用、D15をshared重視の代替とした。G15のstrict値だけを
D/G10の2D値へ混ぜた成績は作らない。

実測driver peakはD/Eが29,192 MiB、Gが29,244 MiBで、固定上限29,346 MiB以内。
F-on/offは凍結により7,644 MiBだった。全runはexit 0、finite、invalid=0で完了した。

## 6. Findings / struggles / tips

| 観測 | 判断と次回への示唆 |
|---|---|
| FP32単独のDがshared、strict、rayを同時改善 | mm単位geometryにBF16量子化は無視できない。Dを既定の主候補にする |
| final matrixをAdamWへ移すと2Dとstrictは上がるがsharedは低下 | optimizer分類は有効なtrade-offだが、shared主目的では単独採用しない |
| 中心項なしはdepth ratioを1.04まで矯正したがsharedを大幅に損失 | 膨張は中心項由来の一面があるが、中心 supervisionを全除去してはいけない |
| Depth adapterはE比sharedを+.00324回復 | shape headへのDepth直結は働く。ただしDのsharedには届かずstrict向けPareto候補 |
| Gの5番目adapterがzeroのまま | defectではなく4-layer decoderに対するtwo-stage予約slot。監査ではactive層を区別する |
| 容量probeより本学習peakが高い | 32GB機でも短いprobeだけでbatchを決めず、driver上限を連続監視する |

## 7. 品質確認

- focused suite: `25 passed`。
- full suite: `669 passed / 24 failed`。24件は、現在配置されていない旧dataset、旧checkpoint、
  旧work_dirのlog/overlayを直接要求する15個のartifact結合test fileに限定された。
- 上記15 fileを明示除外したrepository code suite: `651 passed`。
- 変更Pythonのruff、`git diff --check`、全6 config buildはPASS。
- D15/G10を対応modelへloadし、両方ともmissing key 0、unexpected key 0。

履歴artifactを捏造したりテストを弱めたりせず、今回変更に由来するfailureが0であることを
分離して判定した。

## 8. 成果物対応

モデルと評価JSONはGitへ含めず、次のrepo相対ローカルartifactで対応を固定する。

| 用途 | config | checkpoint / SHA256 | 独立評価JSON / SHA256 |
|---|---|---|---|
| 親B | `configs/yopo/nocs_fruits_2026_rgbd_yolo26s_n_raw_features_calibrated_full.py` | `work_dirs/yopo_yolo26s_n_raw_features_calibrated_full_20260906/best_ellipsoid_shared_AP_25_epoch_20.pth` / `227da40d…510d` | `work_dirs/yopo_yolo26s_n_raw_features_b_best_eval_20260906/20260906_220020/20260906_220020.json` / `c4c04572…5f5b` |
| shared代替D15 | `configs/yopo/nocs_fruits_2026_rgbd_yolo26s_n_raw_features_geometry_fp32.py` | `work_dirs/yopo_geometry_fp32_20260906/best_ellipsoid_shared_AP_25_epoch_15.pth` / `e9851c71…bcd3` | `work_dirs/yopo_geometry_fp32_best_eval_20260907/20260907_000932/20260907_000932.json` / `a736713b…456c` |
| 主採用G10 | `configs/yopo/nocs_fruits_2026_rgbd_yolo26s_n_raw_features_geometry_depth_shape.py` | `work_dirs/yopo_geometry_depth_shape_20260906/best_ellipsoid_shared_AP_25_epoch_10.pth` / `dd01c325…3c64` | `work_dirs/yopo_geometry_depth_shape_best_eval_20260907/20260907_020053/20260907_020053.json` / `f902abca…1a07` |

lineageはいずれもB-bestからのweights-only loadで、D/G間のoptimizer stateや追加学習を
継承していない。完全なSHA256は結果要約に記載する。

## 9. 作業記録

| 日時（JST） | 内容 | 結果 |
|---|---|---|
| 2026-09-06 23:19 | 基準契約固定、期待失敗テスト | baseline回帰13 passed、新規契約5 expected failures |
| 2026-09-06 23:30–23:37 | 改善1〜4をflag付き実装 | 5 focused tests GREEN、既存flag false互換 |
| 2026-09-07 00:08–00:12 | D学習・独立評価 | D15が全gate通過、shared .353354 |
| 2026-09-07 00:42–00:47 | E学習・独立評価 | shared gateを.000261下回り棄却 |
| 2026-09-07 01:07–01:33 | F-on/off学習・独立評価 | 完全凍結成立。中心除去は膨張改善とshared低下を分離 |
| 2026-09-07 01:58–02:03 | G学習・独立評価 | G10が全gate通過、strict .236604 |
| 2026-09-07 02:06 | 事前規則による選抜・公開文書作成 | Pareto集合はD15/G10、主採用D15 |
| 2026-09-07 02:10–02:13 | 最終品質確認 | focused 25、code suite 651 passed。ruff/config/strict load PASS |
| 2026-09-07 02:13–02:16 | 公開安全監査・同期 | private artifact 0、commit `8de9577`を`origin/rgb-d`へpush |
| 2026-09-07 08:09 | ユーザー採用判断 | G10を現行主採用、D15をshared向けPareto代替へ更新 |
