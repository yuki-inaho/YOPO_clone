# LLMオンボーディングサマリー — YOPO (uv / cu121)

> 新任 LLM エージェント（作業/監査/統括）が本プロジェクトに参加する際の初期資料。
> 唯一の真実源は作業書 `temp/workdoc_Jun19-2026_yopo_uv_cu121_gpu.md`。本書はその入口（索引）。

## 1. プロジェクト概要と目的
- **プロジェクト名称・領域:** **YOPO (You Only Pose Once)** — 単眼 RGB カテゴリレベル **9D 物体姿勢推定**（回転3 + 並進3 + サイズ3）。論文は ICRA 2026 採択。実体は **MMDetection 3.3 のフォーク**で、パッケージ名を `mmdet`→`yopo` にリネーム（mmcv / mmengine に依存、`mmdet` パッケージ非依存）。
- **最終成果物:** YOPO を **uv 管理の repo-local 仮想環境 (cu121 スタック) で GPU 学習・推論できる状態**。Docker 不要。
- **ビジネス背景・価値:** OpenMMLab 系（バージョン依存が厳しい）を、再現可能な uv 環境で手元 GPU（NVIDIA L4）上に立ち上げる。`inaho_repos/tomato_stem_mmsegmentation` で確立した uv×OpenMMLab パターンを YOPO に適用。
- **現時点の進捗サマリ:** `cu121` ブランチで **環境構築完了 + GPU 学習/推論スモーク緑化・Opus 監査承認済み（DoD-1〜7 全達成）**。実データ（NOCS REAL275 / CAMERA25 / HouseCat6D）の本番学習・論文精度再現は**未実施（非ゴール）**。GPU スモークは合成ミニデータで「学習・推論パイプラインが GPU 上で動く」ことを実証する方式。

## 2. クリティカルな要求・制約
> 「壊してはいけない」品質・仕様ライン。

- **依存スタックを固定する:** Python **3.10** / `torch==2.4.0+cu121` / `torchvision==0.19.0+cu121` / **`mmcv==2.2.0`（OpenMMLab 配布の prebuilt cu121 manylinux wheel。ソースビルド禁止）** / `mmengine 0.10.x` / **`numpy<2`**。これらを勝手に上げ下げしない（mmcv/mmengine と torch の ABI が崩れる）。
- **`pyproject.toml` は `[tool.uv] package = false`**。`yopo` パッケージは pip インストールせず、`just sync` が venv に `.pth`（リポジトリ root）を書いて develop 風に import 可能化する。
- **コマンドは必ずリポジトリ root から実行**（`import yopo` と config の `_base_` 相対解決のため）。
- **Runner を経由せずモデルを build する場合は `from mmengine.registry import init_default_scope; init_default_scope('yopo')` を `MODELS.build()` 前に呼ぶ**（さもないと `DetDataPreprocessor is not in the registry`）。
- **編集スコープ:** 変更は原則 `scripts/` `temp/` `data/`生成物 `docs/` `README.md` のみ。**`pyproject.toml` / `justfile` / `.venv/` / `yopo/` 本体は安易に変更しない**（`yopo/` は真のフレームワークバグ時のみ最小修正し、根拠と diff を作業書 §7 に記録）。
- **コミット禁止物:** `.venv/` `work_dirs/` `checkpoints/*.pth` `data/` `*.pkl` `*.log` は `.gitignore` 対象（検証ログのみ必要時に `git add -f`）。大容量 blob はコミットしない。
- **暗黙の fallback 禁止 / トレーサビリティ必須:** 依存・ファイル・前提が無い場合は明示記録し代替を明記。全 Trace ID（作業書 §1.3）に証跡を残す。

## 3. 参照すべき合意済み資料
| 種別 | ファイル/リンク | 概要・用途 |
|------|------------------|------------|
| 要求定義/要件定義/WBS/進捗 | `temp/workdoc_Jun19-2026_yopo_uv_cu121_gpu.md` | **唯一の真実源 (SSOT)**。ゴール要求分析・サブゴール(SG)・トレーサビリティ(TR)・フェーズ・原子的チェックリスト・**DoD(§6)**・作業記録(§7)・エージェント編成(§8) |
| セットアップ手順書 | `docs/CU121_TRAINING.md` | cu121 スタック・one-time setup・スモーク使用法・既知の gotcha・非ゴール・ファイル参照表 |
| データ仕様 | `temp/nocs_format_reference.md` | 合成 NOCS データの on-disk フォーマット・ラベル pkl キー・intrinsics・カテゴリ ID マッピング・初回失敗リスク |
| 役割定義 | `.agents/roles/{coordinator,worker,audit}.txt` | start-work-audit パターンの統括/作業/監査の責務・禁止事項・報告形式 |
| ビルド/タスク | `justfile`, `pyproject.toml` | `just` レシピ（sync/env-doctor/smoke-*/…）と uv 依存定義（cu121 index・mmcv wheel URL） |
| 既知課題/テスト資産 | `tests/`（上流 mmdet 由来）, 作業書 §7 | 上流テスト群。本作業のスモークは `just smoke-train`/`just smoke-infer` が回帰確認の役割 |
| 概要 | `README.md` | uv/cu121 Quickstart + モデルズー + 実データ手順 |

## 4. タスク境界（任せること / 任せないこと）
### 任せるタスク
- スモーク config（`temp/smoke_nocs_r50_1iter.py`）・推論スクリプト（`scripts/smoke_infer.py`）・合成データ生成器（`scripts/gen_synthetic_nocs.py`）の改善・デバッグ。
- 新しいデータセット loader / config の追加（スコープ内、`yopo/datasets/...` への追加は要相談）。
- `docs/` の整備、`README.md` の追記。
- スモークのエラー解析と修正（編集スコープ内）。

### 任せないタスク
- `pyproject.toml` / `justfile` / venv / 依存バージョンの独断変更（=環境の作り直しに直結）。
- `yopo/` 本体の大規模改変（フレームワーク改造）。
- `git commit` / `git push` / リモート操作の独断実行（ユーザー指示時のみ）。
- 破壊的コマンド（`rm -rf` 系で広範囲、`git reset --hard` など）。
- 実データの大規模ダウンロード・本番学習方針の決定（非ゴール、要ユーザー判断）。

## 5. インタラクション方針
- **回答スタイル:** 簡潔・技術的。結論先出し → 根拠（`file:line`・コマンド出力）。日本語。見出し＋箇条書き＋必要に応じ表。
- **回答手順:** 前提確認 → ローカル検証（主張は再実行/読み取りで裏取り）→ 提案・実行 → 証跡記録。
- **禁止事項・注意:** 未確定事項の断定をしない。ダミー/フォールバックで成功を偽装しない。他エージェント（DeepSeek/Claude）の出力は「主張」として扱い ground truth としない。
- **秘匿情報の扱い:** `OPENROUTER_API_KEY` 等の値は出力しない。`~/.bashrc` の中身を表示しない。秘密・鍵・広範な環境ダンプを他エージェントに渡さない。

## 6. 試行タスク（オンボーディング演習）
> いずれもリポジトリ root（`/home/kasm-user/Desktop/YOPO`）から。GPU 必須。
1. `just env-doctor` を実行し、`torch 2.4.0+cu121 / mmcv 2.2.0 / mmengine 0.10.7 / yopo 3.3.0 / cuda_available True / NVIDIA L4` が表示されることを確認する。
2. `just gen-synthetic && just smoke-train` を実行し、1 epoch（4 iter）が完走して実 loss がログに出力され、`work_dirs/smoke_train/epoch_1.pth` が保存されること（exit 0）を確認する。
3. `just download-ckpt && just smoke-infer` を実行し、公式 R50 checkpoint がロードされ GPU で forward → `SMOKE INFER OK`（exit 0、pred 9 キー）が出ることを確認する。

## 7. 運用ルール・変更管理
- **ドキュメント更新時の記載ルール:** 作業前に必ず `date "+%Y-%m-%d %H:%M:%S %Z%z"` で時刻確認し、作業書 §7 に「日付・時刻・作業者・作業内容・結果/備考」を開始時と完了時の両方で追記。コード変更時は変更ファイルと概要を記録。
- **TBD の扱い:** 「未確認 / 保留 / 未実施」と明示し、暗黙 fallback しない。判断が要る分岐は選択肢・採用基準・保留理由を書いてユーザー確認。
- **レビュー/承認フロー:** **start-work-audit パターン**。統括(coordinator) が作業書のチェックリストを最小ステップに分解 → 作業(worker) が実装 → 監査(auditor) が独立確認 → 統括が受理してから `[ ]`→`[x]`。**監査エージェントは Opus**、作業エージェントは **DeepSeek V4 Pro（OpenCode/tmux, persistent）優先**で Claude 枠を節約、フォールバックで Sonnet。
- **その他の運用ルール:** 大容量物はコミットしない。コミットはユーザー指示時のみ、メッセージ末尾に `Co-Authored-By` を付す。DoD（作業書 §6）を満たすまで作業を止めない。

---

### 付録: 参考情報
- **主要リポジトリ/ディレクトリ:**
  - リポジトリ: `/home/kasm-user/Desktop/YOPO`（branch `cu121`）、remote `git@github.com:yuki-inaho/YOPO_clone.git`
  - `yopo/`（フレームワーク本体 = リネーム済み mmdet）、`configs/yopo/`（モデル/データ config）、`scripts/`（合成データ・推論）、`temp/`（作業書・スモーク config・参照・ログ）、`docs/`（本書・CU121_TRAINING）、`.agents/roles/`（役割）
- **代表的なコマンド:**
  ```bash
  just --list            # レシピ一覧
  just setup             # uv sync + .pth（初回）
  just env-doctor        # バージョン + GPU トリアージ
  just gen-synthetic     # 合成 NOCS ミニデータ生成
  just download-ckpt     # 公式 R50 checkpoint (~196MB)
  just smoke-train       # GPU 1-epoch 学習スモーク
  just smoke-infer       # GPU 推論スモーク
  just train <config>    # 実データ学習（data/ が必要）
  just test <config> <ckpt>   # 実データ評価
  ```
- **依存ライブラリ:** Python 3.10 / `torch==2.4.0+cu121` / `torchvision==0.19.0+cu121` / `mmcv==2.2.0`(prebuilt cu121 wheel) / `mmengine 0.10.x` / `numpy<2` / pycocotools, scipy, shapely, plyfile, scikit-learn 等（`pyproject.toml` 参照）。
- **GPU/環境:** NVIDIA L4 (sm_89)、driver CUDA 12.7（cu121 ランタイムと後方互換）。
- **連絡先/責任者:** yoshikawa@inaho.co（yuki-inaho）。

> ※本書は必要に応じて拡張・縮退してよい。記入後はバージョン管理する。
