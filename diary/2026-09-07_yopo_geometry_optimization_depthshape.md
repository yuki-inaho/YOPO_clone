# YOPO geometry improvements: result summary

RGB=s / depth=n raw-feature B-bestを共通の親に、geometry FP32、AMUSE出力層分離、
KLD中心項の完全凍結比較、Depth query→GauCho接続を順に検証した。

## 結論

- 主採用はgeometry FP32だけを加えた **D epoch 15**。181枚独立評価でHBB `.863438`、
  ellipse `.864990`、projection `.846640`、shared AP25 `.353354`、strict 3D AP25
  `.220186`、ray median `7.6803 mm`。B比shared `+.037558`、strict `+.032442`。
- strict 3D向けのPareto代替はDepth-to-shapeを含む **G epoch 10**。HBB `.871735`、
  ellipse `.872072`、projection `.850226`、shared `.313780`、strict `.236604`。
- D15はshared、G10はstrictが高く相互非支配である。sharedを主指標とするためD15を
  完成品の第一候補とし、用途でstrictを重視するときだけG10を選ぶ。

## 仮説ごとの判定

| 改善 | 結果 | 判定 |
|---|---|---|
| 1. geometry FP32 | shared/strict/ray/2Dを同時改善 | 採用 |
| 2. final matrixをaux AdamW | Eは2D/strict・shape比率改善、shared `.310539` | 単独棄却 |
| 3. KLD center off | depth ratio `1.2040`、volume ratio `1.2602`まで改善したがshared `.274820` | 中心全除去は棄却 |
| 4. Depth query→shape | E比shared `+.003241`、G10 strict `.236604` | strict向け採用 |

## Artifact identity

| 候補 | checkpoint SHA256 | independent metric JSON SHA256 |
|---|---|---|
| B parent | `227da40d4ec559f4792cac0dcde9cf3fea8afce5ded43a1853fcfc0bc460510d` | `c4c04572e6a151d477b703b846982beeb036ced1a4bed3bd49d344da3fa55b5f` |
| D15 primary | `e9851c71d2dea2dfebff4beb4d1a2392d88ba7f309b784f31b525b1b1aa8bcd3` | `a736713b79199d468e10c5d03935284a1496aff87078e6520e7db26c48a8456c` |
| G10 strict alternative | `dd01c325dba20dfa40787aa447d1cc08108785767ba1f39905904fa42f1f3c64` | `f902abca952f8b379fde2ccac003887e46cb40505cc4cc06773b5266cda71a07` |

対応configとrepo相対artifact path、制御条件、parameter/VRAM、全比較、再現上の注意は
[作業書](workdoc_2026-09-07_yopo_geometry_optimization_depthshape.md)にまとめた。
