"""Fit the task boundary while preserving both raw backbone/encoder stacks."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from mmengine.config import Config

from tools.model_converters.calibrate_yolo26m_rgbd_frontend import (
    _deterministic_train_loader,
    _load_checkpoint,
    _load_component,
    _sample_aligned,
    _sha256,
)
from yopo.registry import MODELS
from yopo.utils import register_all_modules
from yopo.utils.yolo26_frontend_calibration import fit_ridge_projection_from_moments


def calibrate(
    student_config, student_checkpoint, teacher_config, teacher_checkpoint, output
):
    output = Path(output)
    if output.exists() or output.with_suffix(output.suffix + ".json").exists():
        raise FileExistsError(output)
    register_all_modules()
    torch.manual_seed(20260906)
    generator = torch.Generator().manual_seed(20260906)
    sc = Config.fromfile(str(student_config))
    tc = Config.fromfile(str(teacher_config))
    source_checkpoint = _load_checkpoint(student_checkpoint)
    state = source_checkpoint["state_dict"]
    teacher = _load_checkpoint(teacher_checkpoint)["state_dict"]
    sb = MODELS.build(sc.model.backbone).cuda().eval()
    tb = MODELS.build(tc.model.backbone).cuda().eval()
    tn = MODELS.build(tc.model.neck).cuda().eval()
    preprocessor = MODELS.build(sc.model.data_preprocessor).cuda().eval()
    _load_component(sb, state, "backbone.")
    _load_component(tb, teacher, "backbone.")
    _load_component(tn, teacher, "neck.")
    loader = _deterministic_train_loader(sc, batch_size=4, num_workers=4, seed=20260906)
    iterator = iter(loader)
    grams = [torch.zeros(256, 256, dtype=torch.float64) for _ in range(6)]
    crosses = [torch.zeros_like(value) for value in grams]

    def samples(batch):
        inputs = preprocessor(batch, training=False)["inputs"]
        sf, sd = sb.forward_with_depth_features(inputs)
        tf, td = tb.forward_with_depth_features(inputs)
        encoded = tn.forward_shared(tf)
        projected_depth = []
        for level, value in enumerate(td):
            key = f"bbox_head.depth_query_sampler.input_projections.{level}.weight"
            if key in teacher:
                value = F.conv2d(
                    value,
                    teacher[key].cuda(),
                    teacher[key.removesuffix("weight") + "bias"].cuda(),
                )
            projected_depth.append(value)
        return [
            _sample_aligned(x, y, maximum=2048, generator=generator)
            for x, y in zip((*sf, *sd), (*encoded, *projected_depth))
        ]

    with torch.inference_mode():
        for index in range(16):
            for level, (x, y) in enumerate(samples(next(iterator))):
                # Float64 CPU moments avoid accumulating float32 Gram errors.
                x, y = x.double().cpu(), y.double().cpu()
                grams[level] += x.T @ x
                crosses[level] += x.T @ y
            print(f"fit batch {index + 1}/16", flush=True)
        fits = [
            fit_ridge_projection_from_moments(g, c, ridge=1e-4).float()
            for g, c in zip(grams, crosses)
        ]
        errors = [dict(initial=0.0, calibrated=0.0, target=0.0) for _ in fits]
        for _ in range(4):
            for level, (x, y) in enumerate(samples(next(iterator))):
                pred = x @ fits[level].to(x).T
                if level < 3:
                    initial_matrix = (
                        state[f"neck.projections.{level}.weight"]
                        .squeeze(-1)
                        .squeeze(-1)
                        .to(x)
                    )
                    initial = x @ initial_matrix.T
                else:
                    initial = x
                errors[level]["initial"] += float((initial - y).square().sum())
                errors[level]["calibrated"] += float((pred - y).square().sum())
                errors[level]["target"] += float(y.square().sum())
    for error in errors:
        if not error["calibrated"] < error["initial"]:
            raise ValueError(f"train holdout did not improve: {errors}")
    calibrated = {key: value.clone() for key, value in state.items()}
    for level in range(3):
        calibrated[f"neck.projections.{level}.weight"] = fits[level][:, :, None, None]
    # Linear channel projection commutes with bilinear ROI sampling and pooling.
    # Fold the depth maps into the existing concatenated-context output linear.
    key = "bbox_head.depth_query_sampler.output_projection.0.weight"
    original = teacher[key]
    calibrated[key] = torch.cat(
        [
            original[:, level * 256 : (level + 1) * 256] @ fits[level + 3]
            for level in range(3)
        ],
        dim=1,
    )
    expected = {key, *(f"neck.projections.{level}.weight" for level in range(3))}
    changed = {name for name in state if not torch.equal(state[name], calibrated[name])}
    if changed != expected:
        raise ValueError(
            "calibration changed weights outside the four task boundary leaves"
        )
    if any(not torch.isfinite(value).all() for value in calibrated.values()):
        raise ValueError("calibrated state is not finite")
    report = dict(
        format="yolo26_raw_feature_boundary_v1",
        source_sha256=_sha256(student_checkpoint),
        teacher_sha256=_sha256(teacher_checkpoint),
        fit_images=64,
        holdout_images=16,
        split="train",
        labels_used=False,
        changed_keys=sorted(changed),
        errors=errors,
        source_transfer=source_checkpoint["meta"],
    )
    model = MODELS.build(sc.model)
    model.load_state_dict(calibrated, strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=calibrated, meta=report), output)
    report["output_sha256"] = _sha256(output)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(dict(output=str(output), changed=sorted(changed), errors=errors)),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    for name in (
        "student-config",
        "student-checkpoint",
        "teacher-config",
        "teacher-checkpoint",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    calibrate(**vars(parser.parse_args()))
