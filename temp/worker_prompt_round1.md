# 作業エージェント タスクパケット (Round 1) — YOPO uv/cu121 GPU smoke

Think internally in English. Do not reveal chain-of-thought. Provide your final report in Japanese.

You are the **作業エージェント (worker)** in a start-work-audit workflow. The coordinator (Claude Opus) assigns scope; an auditor (Claude Opus) reviews you afterward. Follow the worker role at `.agents/roles/worker.txt`.

## 唯一の真実源 (SSOT)
Read first and treat as the only source of truth:
`temp/workdoc_Jun19-2026_yopo_uv_cu121_gpu.md`
(Especially §1.1 制約・編集スコープ, §3 手順5–9 and 手順11, §6 DoD.)

## ワークスペース / 環境（既にセットアップ済み・検証済み）
- Repo root: `/home/kasm-user/Desktop/YOPO` (git branch `cu121`). Run ALL commands from this root.
- uv venv: `.venv` → Python `.venv/bin/python` (Python 3.10). Use `just` recipes where they exist.
- Verified working already (do NOT redo, do NOT change the env):
  - `just env-doctor` → torch 2.4.0+cu121 / torchvision 0.19.0+cu121 / mmcv 2.2.0 / mmengine 0.10.7 / yopo 3.3.0 / cuda_available True / NVIDIA L4.
  - mmcv CUDA ops (nms) run on the L4 GPU; the R50 model (DINO9DCenter2DPose, 51.2M) builds and its backbone runs a forward on GPU.
  - Synthetic NOCS mini-dataset already generated at `data/nocs_smoke/` (via `just gen-synthetic`).
  - Official R50 checkpoint already downloaded at `checkpoints/nocs_yopo_real_camera_r50.pth` (~196MB, valid mmengine format).
- Draft files already exist (you will fix/iterate them): `scripts/gen_synthetic_nocs.py`, `temp/smoke_nocs_r50_1iter.py`, `scripts/smoke_infer.py`.

## あなたのゴール (this round)
Make BOTH of these PASS on the GPU, then write the docs:
1. **DoD-2 (GPU 学習)**: `just smoke-train` runs ≥1 optimizer iteration on GPU (a loss value appears in the log), saves a readable checkpoint under `work_dirs/smoke_train/`, and exits 0.
2. **DoD-3 (GPU 推論)**: `just smoke-infer` loads `checkpoints/nocs_yopo_real_camera_r50.pth`, runs a forward on GPU on a synthetic frame, prints the prediction structure (type/shape/keys) and `SMOKE INFER OK`, exits 0.
3. **DoD-5 (文書)**: create `docs/CU121_TRAINING.md` (stack, setup, smoke usage, known gotchas, non-goals) and add a uv/cu121 quickstart section to `README.md`. Content must match the actual `justfile`/`pyproject.toml`/scripts.

## 既知の最初のエラー（出発点）
`just smoke-train` currently fails at Runner construction with:
`ValueError: val_dataloader, val_cfg, and val_evaluator should be either all None or not None`.
Fix in `temp/smoke_nocs_r50_1iter.py`: when disabling validation for the smoke, set **all three** (`val_dataloader`, `val_cfg`, `val_evaluator`) to `None` (use `_delete_` where the base sets them). Or alternatively keep validation but point it at the synthetic test split — your judgement; the smoke must not require data we don't have.

## 重要な技術メモ（ハマりどころ）
- Outside a Runner, you MUST call `from mmengine.registry import init_default_scope; init_default_scope('yopo')` before `MODELS.build(...)`, otherwise `DetDataPreprocessor is not in the registry`. `tools/train.py` / `tools/test.py` handle scope internally.
- Always run from the repo root so `import yopo` and config `_base_` relative paths resolve.
- Likely failure modes to expect and fix: missing keys in the synthetic `*_label.pkl` (KeyError from a transform — open the transform under `yopo/datasets/...` to find the exact key, then add it in `scripts/gen_synthetic_nocs.py` and re-run `just gen-synthetic`), intrinsics mismatch (match the hardcoded NOCS intrinsics in `yopo/datasets/pose_estimation/nocs_utils.py`), loss NaN / shape mismatch from implausible synthetic poses, and `batch_size=1` breaking heads/BN (use 2 if needed). For OOM, lower input resolution or batch size.

## 制約（厳守）
- **編集スコープ**: change ONLY `scripts/`, `temp/`, `data/` generators (and the generated `data/nocs_smoke/`), `docs/`, `README.md`. **DO NOT** modify `pyproject.toml`, `justfile`, `.venv/`, or `yopo/` core. If you believe a `yopo/` change is truly required (a real framework bug), make the **minimal** change, and clearly report the file, the diff, and why — do not silently patch.
- Do NOT change the env / re-run `uv sync` / change versions.
- Do NOT edit the workdoc checkboxes or work record — the coordinator owns the workdoc. Do NOT commit or push. No destructive commands.
- No implicit fallback: if something is missing, fix it explicitly or report the blocker.

## 進め方
1. Read the SSOT workdoc + `.agents/roles/worker.txt` + the relevant code (`temp/smoke_nocs_r50_1iter.py`, `scripts/smoke_infer.py`, `scripts/gen_synthetic_nocs.py`, `configs/yopo/nocs_yopo_real_camera_r50.py`, `configs/yopo/_base_/datasets/nocs_dataset.py`, `yopo/datasets/pose_estimation/nocs_dataset.py` + the transforms it uses, `yopo/models/detectors/sixd_pose/dino_9d_center2d_pose.py`).
2. Iterate `just smoke-train` until green. Re-run `just gen-synthetic` after generator changes.
3. Iterate `just smoke-infer` until green.
4. Write `docs/CU121_TRAINING.md` and the README quickstart.
5. Re-run both smokes once more from clean to confirm reproducibility; capture the final tails.

## 報告形式（最後に必ずこの形式で日本語で出力）
```markdown
## Worker Report
- 実施内容:
- 変更点:  (file ごとに、何をなぜ)
- 確認方法: (実行したコマンドと、smoke-train / smoke-infer の最終出力 tail を貼る。checkpoint パスも)
- 残る懸念点:
```
Keep going until smoke-train AND smoke-infer both pass and the docs are written. Do not claim completion before both smokes actually exit 0 with the required output.
