# 作業計画書 兼 記録書: YOPO RGB-D デュアルパス化（RGB=HGNetV2-B2 / depth=HGNetV2-B0）+ depth MAE 事前学習

---

**日付：** `2026年08月24日`
**作業ディレクトリ・リポジトリ:** `/home/kasm-user/Desktop/YOPO_clone`（Git リポジトリ。リモート `https://github.com/yuki-inaho/YOPO_clone.git`、作業ブランチ `rgb-d`）
**作業者：** `opencode (DeepSeek V4 Flash)`（write-workdoc-uv スキルに基づき作成。他エージェントが実行可能な程度に詳細化）

---

## 1. 作業目的

本作業は、以下の目標を達成するために実施します。

*   **目標1:** 現行の「RGB+depth を 4ch に concat して 1 本の HGNetV2-B2(in_channels=4) に通す」単一パス構成から、**RGB と depth をモダリティ別のデュアルパス**に再設計し、各モダリティが独立に特徴を出すようにする。
*   **目標2:** depth 分岐は **HGNetV2-B0（軽量, pretrained あり）を `in_channels=1` で non-strict ロード**して使う（stem のみ 1ch ランダム初期化、stage1-4 は pretrained を完全利用）。
*   **目標3:** 各スケールで RGB/depth 特徴を**チャネル方向 stack（Proj → Concat → Conv1x1）**して共有の DeformableDETR encoder へ流す。encoder / decoder / head は無変更。
*   **目標4:** depth 分岐に対して **Masked Autoencoder（MAE）自己教師あり事前学習**（ConvNeXt V2 の FCMAE、MultiMAE 風）を導入し、深度の空間構造を事前に学習させる。

### 1.1 ゴール要求分析

*   **ユーザーの直観的・直截的な目的:** 「z（深さ）loss が 1.25 で頭打ちし、予測 z が GT から大きく乖離する」問題を解決したい。RGB も depth も互いに独立な特徴を encoder に出せるデュアルパスとし、depth は pretrained を活かした軽量モデル（HGNetV2-B0）で扱い、さらに depth の masked autoencoder 事前学習で深度表現を強化したい。
*   **明示要求:**
    1. depth は**3ch 複製しない**。1ch のまま HGNetV2-B0 で non-strict ロード（stem のみランダム、それ以外は pretrained）。
    2. depth 分岐は HGNetV2 の**軽量 variant**（B0 があればそれが望ましい）。
    3. 融合は「各モダリティが特徴を出した後に**stacking**」する方式。encoder は RGB/depth それぞれの特徴がちゃんと出るように。
    4. depth 側も masked autoencoder（MAE）で**自己教師あり事前学習**できるようにする。
*   **暗黙制約:**
    *   uv 環境（`.venv/`、uv コマンド）で作業。コマンドはリポジトリ root から実行。
    *   justfile を確認し、プロジェクト標準（`just env-doctor` 等）を活用。
    *   既存の DETR encoder / decoder / head（DINO9DCenter2DPose）はできるだけ無変更で、特徴抽出部（backbone+neck）を差し替える。
    *   暗黙の fallback 禁止: 依存・ファイル・前提が無い場合は明示的に記録し、代替を明記する。
    *   監査性: 全 Trace ID に証跡（コマンド出力・diff・ログ・checkpoint パス）を残す。
    *   大容量物（`.venv/`・`work_dirs/`・`data/`・checkpoint）はコミットしない。コミット・push はユーザー指示時のみ。
*   **非ゴール（今回やらないこと):**
    *   depth の encoder 直接注入（bypass は一旦保留。ユーザー指示）。
    *   マルチ GPU 分散学習、ONNX / mmdeploy エクスポート。
    *   RGB 分岐（HGNetV2-B2）自体の構造変更・置換。
    *   NOCS/HouseCat6D 実データでの論文精度再現。
*   **成功条件:**
    1. デュアルパスモデルが build でき、train/val の forward が NaN なしで完走する。
    2. 各スケール（i=1..3, stride 8,16,32）で RGB 特徴 [384,768,1536] と depth 特徴 [256,512,1024] が 256ch に投影され、concat→Conv1x1 で stack された F_i が ChannelMapper(4レベル) を経て encoder に入る。
    3. depth MAE 事前学習が単体で回り、masked 深度復元 loss が減少する（自己教師あり）。
    4. 再訓練後、従来（単一パス）より loss_z が改善する（1.25 → 明らかに下がる）。または少なくとも同等で NaN なし。
    5. 全 Trace ID に証跡が残る。
*   **リスクと前提:**
    *   HGNetV2-B0 の `in_channels=1` 化: stem のみ 1ch になり pretrained(3ch) は stem の形状不一致でスキップ、stage1-4 は一致するので non-strict ロードで保持される（RGB 4ch 化の時と同じ仕組み、検証済み）。
    *   `PPHGNetV2_B0_stage1.pth` が取得可能（既存 B2 と同じ storage リポジトリ URL）。ネット取得不可なら明示記録。
    *   depth 分岐 B0 の出力チャンネル [256,512,1024] は RGB B2 の [384,768,1536] と異なるが、各スケールの空間解像度は共通（stride 8,16,32）なのでスケール対応は一致する（Projection の in_channels が異なるだけ）。
    *   MAE はデータ量（train 300 枚）が少ないため、過学習リスク。masking ratio 高め（0.75）等で対処。loss 減少が見られない場合はユーザー確認。
    *   depth の正規化は `mean=153, std=76.5`（0.6m／0.3m 相当）に確定済み。デュアルパスでは depth 分岐への入力は 1ch の depth（メートル値 *255 相当の正規化前値）を渡す設計になるため、preprocessor の depth 正規化の扱いを再確認する。

### 1.2 サブゴール構造

| ID | サブゴール | 目的との対応 | 成果物 | 検証方法 |
| :--- | :--- | :--- | :--- | :--- |
| SG-1 | 現行の単一パス実装と HGNetV2/neck/encoder のデータフローを精査 | 目標1-3 / 設計基盤 | 影響範囲の整理・設計文書 | forward の shape 確認・既存 smoke 再現 |
| SG-2 | デュアルパス（RGB=HGNetV2-B2, depth=HGNetV2-B0）モジュール実装 | 目標1,2 | `RGBDDualBackbone` + 融合（stack）モジュール + `..._rgbd_dual.py` config | build 成功・forward で F_i shape 確認 |
| SG-3 | depth MAE 事前学習の実装と単体実行 | 目標4 / 明示要求4 | MAE 事前学習 config + スクリプト | masked 復元 loss が減少 |
| SG-4 | 統合 config と smoke 訓練（NaN なし, loss 収束） | 成功条件2,4 | 統合 config + smoke ログ | 1 epoch smoke 完走・loss 減少 |
| SG-5 | 再訓練（デュアルパス + MAE 事前学習済み depth） | 成功条件4 | 学習済み checkpoint | loss_z 改善・評価（IoU / z 誤差） |
| SG-6 | コミット・push、評価画像再生成、claude-mem/handover 更新 | 監査性 / ユーザー指示 | commit・push・画像 | git log / 画像ファイル / 記録 |

### 1.3 トレーサビリティ方針

| Trace ID | 要求・制約 | 対応する作業要素 | 証跡 |
| :--- | :--- | :--- | :--- |
| TR-1 | モダリティ別の独立特徴（目標1・暗黙制約） | SG-1, SG-2（フェーズ1-2） | 設計メモ、forward shape ログ |
| TR-2 | depth は 1ch non-strict ロード・軽量 variant（明示要求1,2） | SG-2（手順3-5） | build ログ、param 一致/不一致ログ |
| TR-3 | stacking 融合で共有 encoder（明示要求3） | SG-2（手順6） | F_i shape 確認ログ |
| TR-4 | depth MAE 事前学習（明示要求4） | SG-3（手順7-9） | MAE 事前学習 loss ログ、checkpoint パス |
| TR-5 | 統合 smoke・NaN なし・loss 改善（成功条件2,4） | SG-4, SG-5（手順10-13, 17） | smoke ログ、loss_z 比較 |
| TR-6 | push・可視化・記憶更新（監査性） | SG-6（手順14-16） | git log、vis 画像、claude-mem id |

---

## 2. 作業内容

### フェーズ 1: 現状把握と設計の確定（見積: 0.8h）

1.  **現状の単一パスを精査：**
    *   `yopo/models/backbones/hgnetv2.py` の `HGNetV2`（`in_channels` で stem を変える仕組み、B0-B6 config、pretrained URL）を確認。
    *   `yopo/models/detectors/base.py` の `extract_feat`、`base_detr.py` の extract_feat（backbone → neck）を確認。
    *   `configs/yopo/nocs_custom_real_hgnetv2_rgbd.py`（現 model config）と `nocs_custom_real_hgnetv2_rgbd_deim_cop.py`（CoP 統合 config）を確認。
    *   **目的:** 差し替え点（backbone+neck 部分）と、encoder への入力（multi-scale feats）の形を確定する。
2.  **深度入力の正規化の扱いを確認：**
    *   `yopo/datasets/transforms/loading.py` の `LoadDepthImageFromFile` / `ConcatDepthToImage`。
    *   `yopo/models/data_preprocessors/data_preprocessor.py` の 4ch 正規化。
    *   **目的:** デュアルパス化で depth を 1ch のままモデルに渡す際、pipeline から depth ch をどう取り出すか（`ConcatDepthToImage` のまま 4ch にして preprocessor 後 split するか、depth のみ保持する設計にするか）を決める。
3.  **設計文書化：**
    *   デュアルパス構成図（下記）とモジュール責務を本作業書に記載。
    *   **目的:** 実装開始前に他エージェントと設計が共有できるようにする。

**デュアルパス構成（確定案）— 実態の `return_idx` に整合する版**

```
RGB(3ch)  ─▶ HGNetV2-B2 (pretrained, in_channels=3, return_idx=[1,2,3])
               → C_i (i=1..3): [384, 768, 1536]（stride 8,16,32）
depth(1ch) ─▶ HGNetV2-B0 (pretrained non-strict, in_channels=1, return_idx=[1,2,3])
               → D_i (i=1..3): [256, 512, 1024]（stride 8,16,32）
                             │ (stem=stage0 のみ 1ch ランダム, stage1-3 は pretrained)
                             ▼
  各スケール i（stride 8,16,32 は共通, i=1..3）:
     C'_i = ProjRGB_i(C_i)     # 1x1 Conv(in=384/768/1536) → 256ch
     D'_i = ProjDepth_i(D_i)   # 1x1 Conv(in=256/512/1024) → 256ch
     F_i  = Conv1x1( Concat[C'_i, D'_i] )  # 512 → 256ch (stack 融合)
                             ▼
               F_1..F_3 → ChannelMapper(num_outs=4) → F_1..F_4 → DeformableDETR encoder
```

*   **スケール数の確定（実体照合済み）:** HGNetV2 は `return_idx=[1,2,3]` で **3 スケール**（stage1-3）を返す。stage0(stem) stride 4 は return されない。現 config も `return_idx=[1,2,3]`（出力 [384,768,1536]）を ChannelMapper `num_outs=4` で 4 レベルに拡張している。⇒ **dual 化でも同方針**（融合後 3 レベル → ChannelMapper で 4 レベル）が最小変更。
*   **融合モジュール:** `RGBDDualBackbone` が 2 つの HGNetV2 を内包し、各スケールで Proj→Concat→Conv1x1 融合して **3 レベルの F_1..F_3 を返す**。
*   **neck の扱い（確定）:** 融合は `RGBDDualBackbone` 内で行い、既存の `ChannelMapper(in_channels=[384,768,1536], num_outs=4)` は**そのまま維持**して F(3レベル,256ch) を 4 レベルに拡張する。ChannelMapper は入力スケール数(3)と入チャンネル(=256 共通)に合わせて config を更新（`in_channels=[256,256,256]` に変更）。ChannelMapper を Identity 化する案は不採用（num_outs=4 のレベル拡張が必要なため）。
*   **depth 入力:** 前処理済み 4ch のまま `extract_feat` に渡し、モジュール内で `x[:, :3]`（RGB）と `x[:, 3:4]`（depth）に split するのが最小変更。これなら pipeline / preprocessor は現状のまま（depth 正規化は preprocessor で適用済み）。

### フェーズ 2: デュアルパス実装（見積: 1.5h）

4.  **`RGBDDualBackbone` 実装：**
    *   `yopo/models/backbones/dual_rgbd.py`（新規）に `@MODELS.register_module()` な `RGBDDualBackbone(BaseModule)`。
    *   引数: `rgb_backbone=dict(...)`, `depth_backbone=dict(...)`, `num_fuse_outs=4`, `out_channels=256` など。
    *   内部: `self.rgb = MODELS.build(rgb_backbone)`（HGNetV2-B2）、`self.depth = MODELS.build(depth_backbone)`（HGNetV2-B0）。
    *   `forward(x)`: `x_rgb = x[:, :3]`, `x_d = x[:, 3:4]` → 両方 forward → 各スケールで Proj→Concat→Conv1x1 → リスト F を返す。
    *   **目的:** 2 つのモダリティ分岐を 1 モジュールに包み、既存 detector から呼び出せるようにする。
5.  **config 更新：**
    *   `configs/yopo/nocs_custom_real_hgnetv2_rgbd.py` の `backbone` を `RGBDDualBackbone`（rgb=HGNetV2-B2 in_channels=3 / depth=HGNetV2-B0 in_channels=1, 共に return_idx=[1,2,3]）に差し替え、`neck`(ChannelMapper) は維持するが `in_channels=[256,256,256]`（融合後 256ch×3 レベル）に更新。
    *   depth 側 HGNetV2-B0: `in_channels=1, return_idx=[1,2,3]`、`init_cfg` に `PPHGNetV2_B0_stage1.pth`。
    *   **目的:** デュアルパス構成を config で有効化し、既存 neck → encoder の経路を維持する。
6.  **build & forward shape 検証：**
    *   `MODELS.build` でモデル生成、ダミー 4ch 入力で forward → F_i（融合後 3 レベル 256ch, stride 8,16,32）と neck 出力（4 レベル）の shape を確認。
    *   **目的:** 融合が正しく機能し、encoder 以降（multi-scale feats, 4 レベル）に渡る形を確かめる。

### フェーズ 3: depth MAE 事前学習（見積: 1.5h）

7.  **depth MAE 実装方針の調査：**
    *   参考: ConvNeXt V2 (arXiv:2301.00808, FCMAE)、MultiMAE (arXiv:2204.01678)。
    *   **実装方針（既存フレームワークに載せる）:** MAE は既存の DETR モデルと別物なので、専用の軽量 model（`MAEDepth` クラス or `RGBDDepthMAE`）を新規登録し、`tools/train.py`（標準 mmengine Runner）で回す。構成は以下:
        - `backbone`: HGNetV2-B0（`in_channels=1`, `return_idx=[1,2,3]`, pretrained 初期化 or ランダム）
        - `neck/encoder`: 軽量（数層 Conv で B0 の 3 スケール出力 → 1 スケールに集約、または patch 化）
        - `head/decoder`: マスク位置の深度パッチを元解像度に復元する軽量 decoder（転置 Conv / アンサイン）
        - `loss`: masked 位置のみの L1（or L2）復元 loss
        - dataset: 現状 pipeline（depth まで）のまま、backbone に入れるのは depth 1ch（`x[:, 3:4]`）
    *   **目的:** 実装可能な最小構成（FCMAE 風）を確定する。
8.  **MAE ヘッド＆loss 実装：**
    *   `yopo/models/detectors/mae_depth.py`（新規, `MODELS` 登録）: `MAEDepth(data_preprocessor, backbone=dict(HGNetV2-B0 1ch), encoder/decoder, loss)`。
    *   `configs/yopo/mae_depth_hgnetv2_b0.py`（新規）: model=MAEDepth, data は depth 入力（現 pipeline）, optim, 数 epoch。
    *   **目的:** depth のみの自己教師あり事前学習を `tools/train.py` で回せるようにする。
9.  **MAE 事前学習の実行と検証：**
    *   `.venv/bin/python tools/train.py configs/yopo/mae_depth_hgnetv2_b0.py --work-dir work_dirs/mae_depth` で数 epoch 実行。
    *   masked 復元 loss（`loss_depth`）が初期 → 数 epoch 後で減少することを確認。
    *   checkpoint（B0 バックボーン + decoder）を保存。**既存 DETR への組み込みは、この checkpoint を `load_from`（strict=False）で depth 分岐にロードする**形（※整合するキーは B0 の stage のみ、decoder は捨てる）。

### フェーズ 4: 統合 smoke と再訓練（見積: 1.5h）

10. **統合 config 作成：**
    *   デュアルパス + CoP(use_cop_chain は任意) + depth MAE 事前学習済み checkpoint を組む。
    *   **MAE checkpoint のロード方法（具体）:** `load_from` はモデル全体に効くため使わず、`RGBDDualBackbone.depth_backbone.init_cfg = dict(type='Pretrained', checkpoint='work_dirs/mae_depth/best_loss_depth.pt')` のように **depth 分岐の Backbone init_cfg で指定**する（strict 挙動は Backbone の `init_cfg` 経由 = 非 strict、stage のみ一致、decoder キーは無視）。`load_from` を使う場合は prefix 対応（`depth.` が付く）が必要になるため注意。
    *   既存 `nocs_custom_real_hgnetv2_rgbd_deim_cop_retrain.py` を参考に、`..._deim_cop_dual.py` 等の新 config 化。
11. **smoke 訓練：**
    *   1 epoch smoke で NaN なし、loss 減少、memory が 20GB 内を確認。
12. **再訓練：**
    *   バックグラウンド（nohup）で 100 epoch cosine を実行。loss_z の推移を監視（従来 1.25 からの改善が目標）。
13. **評価：**
    *   `temp/dump_nocs_custom_infer.py` + `temp/draw_nocs_custom_results.py` で推論 & 結果画像を生成し、z 誤差・center 偏りが改善したか確認。

### フェーズ 5: 反映・記録（見積: 0.7h）

14. **commit & push：** デュアルパス実装・config・MAE 事前学習・作業書を `rgb-d` へ反映。
15. **claude-mem 更新：** デュアルパス完了・MAE・再訓練結果を記録。
16. **handover / 作業記録：** 本作業書の作業記録テーブルに全フェーズの開始・完了・結果を追記。

---

## 3. 作業チェックリスト

*作業を完了したら `[ ]` を `[x]` に変更します。*

### フェーズ 1: 現状把握と設計の確定

### 手順 1: 単一パス実装の精査
- [ ] 🖐 **操作**: `hgnetv2.py` / `base_detr.py` `/extract_feat` / 現 model config を読み、backbone・neck・encoder のデータフロー（shape）を記録する。
- [ ] 🔎 **確認**: 差し替え点（backbone+neck）と、encoder へ渡る multi-scale feats の形状が明記されている。
- [ ] 🧪 **テスト**: 既存の smoke（`just env-doctor` + 1 iter forward）が壊れていないことを確認する。
- [ ] 🛠 **エラー時対処**: ファイル欠落は `rg --files` で再探索し、前提不一致は作業記録に明記する。

### 手順 2: depth 入力の正規化フロー確認
- [ ] 🖐 **操作**: `loading.py` の `LoadDepthImageFromFile`/`ConcatDepthToImage` と `data_preprocessor.py` の 4ch 正規化を読む。
- [ ] 🔎 **確認**: depth が 4ch の一部として preprocessor を通過後、モデルに渡る経路が確定している（split 方式採用等）。
- [ ] 🧪 **テスト**: depth 1ch を取り出せること（ダミー forward で `x[:, 3:4]` shape = (B,1,H,W)）。
- [ ] 🛠 **エラー時対処**: 正規化が depth に効いていない場合は mean/std の適用順を確認。

### 手順 3: 設計文書化
- [ ] 🖐 **操作**: デュアルパス構成図とモジュール責務（3 スケール融合 + ChannelMapper 維持）を本作業書 §2 に反映（構成図は既記載。実装詳細を追記）。
- [ ] 🔎 **確認**: SG-2 の実装対象（モジュール・config・shape・neck の in_channels 変更）が他エージェントに伝わる。
- [ ] 🧪 **テスト**: 実装前に失敗させるテストケース名（forward shape テスト等）を明記。
- [ ] 🛠 **エラー時対処**: 設計の決定点は選択肢と採用理由を明記し、必要ならユーザー確認。

### フェーズ 2: デュアルパス実装

### 手順 4: `RGBDDualBackbone` 実装
- [ ] 🖐 **操作**: `yopo/models/backbones/dual_rgbd.py`（新規）に `RGBDDualBackbone` を実装（RGB HGNetV2-B2(3ch) + depth HGNetV2-B0(1ch) 内包、forward で `x[:, :3]`/`x[:, 3:4]` split → 各分岐 → スケール毎に Proj(in=384/768/1536 と 256/512/1024)→Concat→Conv1x1→256ch）。F_1..F_3（3 レベル）を返す。
- [ ] 🔎 **確認**: `MODELS.register_module()` 済み、引数で `rgb_backbone`/`depth_backbone`/`out_channels`（=256）を受け取れる。戻り値が 3 レベルの list[Tensor]。
- [ ] 🧪 **テスト**: build 後 `forward(x)` で F_1..F_3 の shape が (B,256,H/8,W/8)/(B,256,H/16,W/16)/(B,256,H/32,W/32)。
- [ ] 🛠 **エラー時対処**: スケール数不一致（3 vs 期待）は `return_idx` を確認、チャンネル不一致は Projection の `in_channels` を確認。

### 手順 5: config 更新（デュアルパス有効化）
- [ ] 🖐 **操作**: `nocs_custom_real_hgnetv2_rgbd.py` の `backbone` を `RGBDDualBackbone` に、`neck`(ChannelMapper) の `in_channels=[384,768,1536]` → `[256,256,256]`（融合後 256ch×3）に更新。
- [ ] 🔎 **確認**: depth 側 HGNetV2-B0 の `in_channels=1` / `return_idx=[1,2,3]` / `init_cfg(PPHGNetV2_B0_stage1.pth)` が設定されている。neck は `num_outs=4` のまま（4 レベル拡張維持）。
- [ ] 🧪 **テスト**: `Config.fromfile` + `MODELS.build` が通る。
- [ ] 🛠 **エラー時対処**: pretrained ロード差異（stem スキップ）は non-strict の警告として正常、ログに残す。ChannelMapper の in_channels 誤りは build 時の assert で検出。

### 手順 6: forward shape 検証
- [ ] 🖐 **操作**: ダミー 4ch 入力（B,4,H,W）でモデル forward → F_i（backbone 出力）と neck 出力（encoder へ渡る 4 レベル）の shape を print。
- [ ] 🔎 **確認**: backbone 出力が (B,256,stride 8/16/32)×3、neck 出力が 4 レベル（stride 8/16/32/64 相当）。
- [ ] 🧪 **テスト**: 期待 shape に一致しなければ失敗。一致で成功。
- [ ] 🛠 **エラー時対処**: shape 異常は各モジュールの出力チャンネル/ストライドを逐一確認。

### フェーズ 3: depth MAE 事前学習

### 手順 7: MAE 設計の調査
- [ ] 🖐 **操作**: ConvNeXt V2 (FCMAE, 2301.00808)・MultiMAE (2204.01678) の手法を確認し、depth へ適用する最小設計（マスク率 0.75、L1 復元 loss、軽量 decoder、`tools/train.py` で回す専用 `MAEDepth` model）を本作業書 §2 に記載。
- [ ] 🔎 **確認**: 実装方針（専用 MAEDepth モデル + standard Runner、mask ratio 0.75, L1）が確定。
- [ ] 🧪 **テスト**: 事前学習 loss（`loss_depth`）が減少する見込みの設計であること。
- [ ] 🛠 **エラー時対処**: 設計が定まらない場合は選択肢を並べユーザー確認。

### 手順 8: MAE ヘッド＆loss 実装
- [ ] 🖐 **操作**: `yopo/models/detectors/mae_depth.py` に `MAEDepth`（backbone=HGNetV2-B0 1ch, マスク位置深度復元 decoder, L1 loss, `MODELS` 登録）と `configs/yopo/mae_depth_hgnetv2_b0.py` を作成。
- [ ] 🔎 **確認**: 入力 depth(1ch)→B0→マスク埋め、出力が元 depth と同解像度になる。`MODELS.build` が通る。
- [ ] 🧪 **テスト**: 単体で forward → `loss_depth` が計算できる。
- [ ] 🛠 **エラー時対処**: 解像度不一致は decoder の upsample 段数を調整。

### 手順 9: MAE 事前学習の実行・検証
- [ ] 🖐 **操作**: `.venv/bin/python tools/train.py configs/yopo/mae_depth_hgnetv2_b0.py --work-dir work_dirs/mae_depth` を数 epoch 実行。
- [ ] 🔎 **確認**: `loss_depth` が初期 → 数 epoch 後で減少、checkpoint が保存される。
- [ ] 🧪 **テスト**: 初期 loss → 数 epoch 後 loss の減少（ログの loss_depth 推移）。
- [ ] 🛠 **エラー時対処**: loss 発散は lr 減 or mask 率調整。データ不足は epoch 増で様子見。対処後も停滞はユーザー確認。

### フェーズ 4: 統合 smoke と再訓練

### 手順 10: 統合 config 作成
- [ ] 🖐 **操作**: デュアルパス(`..._rgbd_dual.py`) + CoP(任意) + depth MAE checkpoint を組む。MAE checkpoint は `RGBDDualBackbone.depth_backbone.init_cfg=dict(type='Pretrained', checkpoint='work_dirs/mae_depth/...')` で depth 分岐にロード（`load_from` は使わない。prefix 問題回避）。
- [ ] 🔎 **確認**: `Config.fromfile` + `MODELS.build` が通る。
- [ ] 🧪 **テスト**: 既存 retrain config との差分が明示的。
- [ ] 🛠 **エラー時対処**: init_cfg ロード非互換は strict 扱い確認・キー名（`depth.` prefix, `backbone.` 等）を確認。decoder キーが余るのは非 strict で無視（ログに残す）。

### 手順 11: smoke 訓練（1 epoch）
- [ ] 🖐 **操作**: `tools/train.py <統合config> --cfg-options max_epochs=1` を実行。
- [ ] 🔎 **確認**: 150/150（または 38/38 + 蓄積）iter 完走、NaN なし、memory < 20GB。
- [ ] 🧪 **テスト**: 初期 1 epoch が失敗→修正後成功。
- [ ] 🛠 **エラー時対処**: NaN は loss 成分・入力正規化を確認。OOM は batch/accum 調整。

### 手順 12: 再訓練（100 epoch cosine）
- [ ] 🖐 **操作**: `nohup .venv/bin/python tools/train.py <統合config> --work-dir work_dirs/dual_retrain > /tmp/opencode/dual_train.log 2>&1 &`
- [ ] 🔎 **確認**: PID 起動・epoch 進行・loss_z 推移を監視。
- [ ] 🧪 **テスト**: loss_z が 1.25 から改善（例: 1.0 未満）するか、少なくとも単一パス同等で NaN なし。
- [ ] 🛠 **エラー時対処**: 即死はログの traceback から原因特定し修正後再起動。

### 手順 13: 評価（推論+描画）
- [ ] 🖐 **操作**: `temp/dump_nocs_custom_infer.py` で推論 dump → `temp/draw_nocs_custom_results.py` で結果画像生成。
- [ ] 🔎 **確認**: z 誤差・center 偏り・sizes が改善（従来 z=-0.09/size負値 から正常化）。
- [ ] 🧪 **テスト**: ダンプ統計（z mean, size 符号, center std）を記録し改善を確認。
- [ ] 🛠 **エラー時対処**: デコード異常は `_predict_by_feat_single` を再確認（既知の既存バグ）。

### フェーズ 5: 反映・記録

### 手順 14: commit & push
- [ ] 🖐 **操作**: デュアルパス実装・config・MAE・作業書を `git add` → commit → `git push origin rgb-d`。
- [ ] 🔎 **確認**: `git status` クリーン、push 成功。
- [ ] 🧪 **テスト**: `git log --oneline -1` でコミット確認。
- [ ] 🛠 **エラー時対処**: work_dirs/data が含まれないことを確認（.gitignore）。

### 手順 15: claude-mem 更新
- [ ] 🖐 **操作**: claude-mem にデュアルパス完了・MAE・再訓練結果の observation を追加。
- [ ] 🔎 **確認**: 新 id が追加されている。
- [ ] 🧪 **テスト**: `SELECT MAX(id)` で増分確認。
- [ ] 🛠 **エラー時対処**: DB 構成に合わせてスキーマを確認してから insert。

### 手順 16: 作業記録の締め
- [ ] 🖐 **操作**: 本書の作業記録テーブルに全フェーズの開始・完了・結果（コマンド・loss・画像パス）を追記。
- [ ] 🔎 **確認**: 注意事項（時刻記録・両端記録・結果備考）が守られている。
- [ ] 🧪 **テスト**: 全チェックリストが `[x]` か確認。
- [ ] 🛠 **エラー時対処**: 未記録項目は完了後すぐ補完。

---

## 4. 作業に使用するコマンド参考情報

### 基本的な開発ワークフロー

```bash
cd /home/kasm-user/Desktop/YOPO_clone
just env-doctor          # 環境確認（torch/cuda/GPU）
uv sync                  # 依存同期（必要時）

# モデルbuild検証
.venv/bin/python - <<'EOF'
from mmengine.registry import init_default_scope
init_default_scope('yopo'); import yopo
from mmengine.config import Config
from yopo.registry import MODELS
cfg = Config.fromfile('<config>')
print(MODELS.build(cfg.model))
EOF
```

### テストと品質管理（デュアルパス forward shape 検証）

```bash
.venv/bin/python - <<'EOF'
import torch
from mmengine.registry import init_default_scope
init_default_scope('yopo'); import yopo
from mmengine.config import Config
from yopo.registry import MODELS
# デュアルパス専用 config（フェーズ2で作成）
cfg = Config.fromfile('configs/yopo/nocs_custom_real_hgnetv2_rgbd_dual.py')
m = MODELS.build(cfg.model).cuda().eval()
x = torch.randn(2, 4, 480, 640).cuda()   # RGB3 + depth1
feats = m.backbone(x)                     # F_1..F_3 (256ch, stride 8/16/32)
for f in feats: print(tuple(f.shape))
EOF
```

### 特定機能の実行・デバッグ例

```bash
# MAE 事前学習（depth のみ, フェーズ3）
.venv/bin/python tools/train.py configs/yopo/mae_depth_hgnetv2_b0.py \
    --work-dir work_dirs/mae_depth

# デュアルパス smoke 訓練（1 epoch, NaN/OOM 確認）
.venv/bin/python tools/train.py configs/yopo/nocs_custom_real_hgnetv2_rgbd_dual.py \
    --work-dir work_dirs/dual_smoke --cfg-options max_epochs=1

# 再訓練（バックグラウンド）
nohup .venv/bin/python tools/train.py configs/yopo/nocs_custom_real_hgnetv2_rgbd_dual.py \
    --work-dir work_dirs/dual_retrain > /tmp/opencode/dual_train.log 2>&1 &

# 推定 & 描画（評価）
.venv/bin/python temp/dump_nocs_custom_infer.py --config configs/yopo/test_nocs_custom_rgbd_deim_cop.py \
    --checkpoint work_dirs/dual_retrain/epoch_50.pth --out work_dirs/dual_retrain/val.pkl
.venv/bin/python temp/draw_nocs_custom_results.py --input work_dirs/dual_retrain/val.pkl \
    --out-dir work_dirs/dual_retrain/vis

# claude-mem 確認
python3 -c "import sqlite3; c=sqlite3.connect('/home/kasm-user/.claude-mem/claude-mem.db'); print(c.execute('SELECT MAX(id) FROM observations').fetchone())"
```

---

## 6. 完了の定義

*作業が最後まで完了したら `[ ]` を `[x]` にしつつ、作業が本当に完了したかをチェックします*

- [ ] 観点1: デュアルパス（RGB=HGNetV2-B2 / depth=HGNetV2-B0 1ch non-strict, return_idx=[1,2,3]）が build・forward でき、各スケール stack 融合 → ChannelMapper(4レベル) → 共有 encoder に入る（成功条件1,2）。
- [ ] 観点2: depth MAE 事前学習が回り、masked 復元 loss が減少、checkpoint が保存される（成功条件3）。
- [ ] 観点3: 再訓練後に loss_z が改善（従来 1.25 → 明確に下がる）または NaN なしで同等。評価（z 誤差・center）が改善（成功条件4）。
- [ ] 観点4: すべての Trace ID（TR-1..6）に対応する証跡が作業記録（§7）に残っている。
- [ ] 観点5: 暗黙 fallback を使わず、例外・未対応事項（例: B0 pretrained 取得不可、MAE loss 停滞）は明示的に記録されている。

---

## 7. 作業記録

**重要な注意事項：**

*   作業開始前に必ず `date "+%Y-%m-%d %H:%M:%S %Z%z"` コマンドで現在時刻を確認し、正確な日時を記録します。
*   各作業項目を開始する際と完了する際の両方で記録を行うこと。
*   作業内容は具体的なコマンドや操作手順を詳細に記載すること。
*   結果・備考欄には成功／失敗、エラー内容、解決方法、重要な気づきを必ず記入すること。
*   複数のフェーズがある場合は、フェーズごとに開始・完了の記録を取ること。
*   コード変更を行った場合は、変更したファイル名と変更内容の概要を記録すること。
*   エラーが発生した場合は、エラーメッセージと解決策を詳細に記録すること。

| 日付 | 時刻 | 作業者 | 作業内容 | 結果・備考 |
| :--- | :--- | :--- | :--- | :--- |
| `2026-08-24` | `02:04:53 UTC` | opencode | 作業書作成（write-workdoc-uv） | デュアルパス（RGB-B2/depth-B0）＋MAE 事前学習の要件・設計・チェックリスト・DoD を整理。 |
| `2026-08-24` | `02:23:41 UTC` | opencode | フェーズ1開始: 単一パス実装の精査 | 作業書確認完了。DoD 完遂に向け実装フェーズ開始。`git status` は workdoc のみ untracked。 |
| `2026-08-24` | `02:26-02:31 UTC` | opencode | フェーズ2: `RGBDDualBackbone` 実装 + dual config + forward shape 検証 | `dual_rgbd.py` 新規実装（RGB=B2 3ch / depth=B0 1ch, return_idx=[1,2,3], Proj→Concat→Conv1x1）。`..._rgbd_dual.py` config 作成。forward shape: F(2,256,60,80)/(2,256,30,40)/(2,256,15,20), neck 4 レベル確認。`bbox_head.use_cop_chain=True` 必要（CoP無だと loss chain None で落ちる既存挙動）。 |
| `2026-08-24` | `02:31-02:42 UTC` | opencode | 手順11: dual smoke 訓練 | `tools/train.py ..._rgbd_dual.py` 15 epoch 実行（max_epochs 上書きが効かず 15 epoch まで回った）。NaN なし、loss 1079→354、memory ~9.4GB。単一パスと同様、loss_z ~1.27 で当初停滞傾向。 |
| `2026-08-24` | `02:42-02:50 UTC` | opencode | フェーズ3: `MAEDepth` 実装 + config + MAE 事前学習 | `mae_depth.py`（HGNetV2-B0 1ch encoder + マルチスケール decoder + masked L1）。`mae_depth_hgnetv2_b0.py` config。20 epoch 完了: loss_depth 4.03→0.64（明確に減少）。`enc_b0_mae.pth` 生成（encoder 生キー）。 |
| `2026-08-24` | `02:50-03:08 UTC` | opencode | フェーズ4: 統合 config + smoke | `..._deim_cop_dual.py`（RGBDDualBackbone + depth_backbone.init_cfg=enc_b0_mae.pth + CoP）。MAE 重み一致確認（MATCH: True）。smoke 1 epoch: NaN なし, memory 9.4GB, loss_z 1.41。 |
| `2026-08-24` | `03:08 UTC` | opencode | フェーズ4: 再訓練 100 epoch 開始 | `nohup ... --work-dir work_dirs/dual_retrain` 完全デタッチ起動（PID 1053875）。loss_z 推移を監視中。 |
| `YYYY-MM-DD` | `HH:MM:SS TZ` | `作業者名` | フェーズ1開始: `[タスク名]` | 作業計画書確認完了、`[タスク]`の要件を把握 |
| `YYYY-MM-DD` | `HH:MM:SS TZ` | `作業者名` | フェーズ2開始: `[タスク名]` | `[実行コマンド]` で build / forward 検証 |
| `YYYY-MM-DD` | `HH:MM:SS TZ` | `作業者名` | フェーズ3開始: `[MAE 事前学習]` | 事前学習 loss の推移 |
| `YYYY-MM-DD` | `HH:MM:SS TZ` | `作業者名` | フェーズ4開始: `[smoke / 再訓練]` | NaN 有無・loss_z 推移・memory |
| `YYYY-MM-DD` | `HH:MM:SS TZ` | `作業者名` | フェーズ5: `[commit・push・記録]` | コミット hash・claude-mem id・画像パス |

---

## 8. 補足（作業者が参照すべき前提メモ）

- **リポジトリ**: `/home/kasm-user/Desktop/YOPO_clone`（branch `rgb-d`）。uv 環境 `.venv/`。
- **GPU**: RTX 4000 Ada 20GB（sm_89）。実効バッチ 24 = batch 8 × accum 3 で fp32（~8.7GB）。AMP はこのモデルで NaN を誘発したため不使用（過去検証済み）。
- **モデル現状**: `DINO9DCenter2DPose`（num_queries=100, as_two_stage, DINO9DCenter2DPoseHead に CoP chain aux head 追加済み）。
- **現 config**: `nocs_custom_real_hgnetv2_rgbd.py`（単一パス 4ch）→ `..._deim.py` → `..._deim_cop.py` → `..._deim_cop_retrain.py`（正規化修正版ゼロショット再訓練, 50 epoch 完了, loss_z=1.25 頭打ち）。デュアルパスは **新規 `nocs_custom_real_hgnetv2_rgbd_dual.py`** として base を差し替える（既存 config を壊さない）。
- **depth 正規化**: preprocessor mean=[123.675,116.28,103.53,153.0], std=[58.395,57.12,57.375,76.5]（depth ch は 0.6m/0.3m 想定）。pipeline は `LoadDepthImageFromFile(norm_scale=1000)` → `ConcatDepthToImage(depth_scale=255)` → 4ch。
- **HGNetV2 variants**: B0(B0 url) / B1(S) / B2(M) / B3 / B4(L) / B5(X) / B6(H)。RGB 分岐 = **B2**（in_channels=3, return_idx=[1,2,3] → [384,768,1536]）。depth 分岐 = **B0**（in_channels=1, return_idx=[1,2,3] → [256,512,1024], `PPHGNetV2_B0_stage1.pth` を non-strict）。両者とも **3 スケール出力**、既存 ChannelMapper `num_outs=4` で 4 レベル拡張。
- **既知の既存バグ**: `_predict_by_feat_single` で予測 center が右下偏り・z 負値・sizes 負値（CoP 無 baseline でも発生）。デュアルパス化と別問題。評価時に再確認要。
- **評価用スクリプト**: `temp/dump_nocs_custom_infer.py`（推論→pkl）、`temp/draw_nocs_custom_results.py`（pkl→描画、軽量高速）。
- **類似手法（調査済み）**: MV-DETR（RGB pretrained + 軽量 geometry 分岐）、TANet（asymmetric encoder）、ConvNeXt V2(2301.00808)/MultiMAE(2204.01678)（MAE 事前学習）、LingBot-Depth(2601.17895, 別論文)。
- **git 最新**: rgb-d @ `8792f40`（CoP retrain config）。コミット禁止物: `.venv/` `work_dirs/` `data/` `*.log`。
- **作業書の注意事項**: 作業開始前に `date` で時刻確認、開始と完了の両方で記録、具体的なコマンド記録、エラー時は原因特定を記録。
