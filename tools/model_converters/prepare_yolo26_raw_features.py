"""Prepare YOPO RGB-s/depth-n from raw DEIM feature stacks and a YOPO task head."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config

from yopo.registry import MODELS
from yopo.utils import register_all_modules
from yopo.utils.yolo26_raw_feature_transfer import convert_raw_feature_branch


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source(directory, branch):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    checksum = sha256(directory / "arrays.npz")
    if checksum != manifest["array_sha256"]:
        raise ValueError("source SHA mismatch")
    cfg = manifest["metadata"]["config"]["model"]
    if cfg["backbone"]["variant"] != branch.scale:
        raise ValueError("source/target scale mismatch")
    expected = dict(
        name="hybrid",
        hidden_dim=256,
        num_heads=8,
        ffn_dim=1024,
        num_aifi_layers=1,
        feature_levels=3,
        dropout=0.0,
    )
    if any(cfg["encoder"].get(key) != value for key, value in expected.items()):
        raise ValueError("source encoder configuration mismatch")
    with np.load(directory / "arrays.npz", allow_pickle=False) as source:
        state, report = convert_raw_feature_branch(source, branch)
    report.update(source_sha256=checksum, source_step=manifest["metadata"]["step"])
    return state, report


def prepare(rgb, depth, task, config, output):
    output = Path(output)
    if output.exists() or output.with_suffix(output.suffix + ".json").exists():
        raise FileExistsError(output)
    torch.manual_seed(20260906)
    register_all_modules()
    cfg = Config.fromfile(str(config))
    model = MODELS.build(cfg.model)
    if (model.backbone.rgb_backbone.scale, model.backbone.depth_backbone.scale) != (
        "s",
        "n",
    ):
        raise ValueError("expected RGB=s/depth=n")
    state = model.state_dict()
    reports = {}
    selected = {}
    for name, source in (("rgb", rgb), ("depth", depth)):
        branch = getattr(model.backbone, name + "_backbone")
        mapped, report = read_source(source, branch)
        selected.update(
            {f"backbone.{name}_backbone." + key: value for key, value in mapped.items()}
        )
        reports[name] = report
    teacher = torch.load(task, map_location="cpu", weights_only=False)["state_dict"]

    # Reuse the task stack and P6; branch feature stacks come exclusively from raw DEIM.
    def reused(key):
        return (
            not key.startswith(("backbone.", "neck."))
            or key == "neck.derived_p6.weight"
        )

    task_keys = {key for key in state if reused(key)}
    ignored_task_inputs = "bbox_head.depth_query_sampler.input_projections."
    extra = {
        key
        for key in teacher
        if reused(key) and key not in state and not key.startswith(ignored_task_inputs)
    }
    if extra:
        raise ValueError(f"extra teacher task leaves: {sorted(extra)[:3]}")
    for key in task_keys:
        if key not in teacher or teacher[key].shape != state[key].shape:
            raise ValueError(f"missing or incompatible task leaf {key}")
        if not torch.isfinite(teacher[key]).all():
            raise ValueError(f"nonfinite task leaf {key}")
        selected[key] = teacher[key]
    fresh = set(state) - set(selected)
    allowed = {
        key
        for key in state
        if key.endswith("num_batches_tracked")
        or key.startswith(("backbone.depth_adapters.", "neck.projections."))
        or key == "backbone.depth_beta"
    }
    if fresh != allowed:
        raise ValueError(f"unexpected fresh leaves: {sorted(fresh ^ allowed)}")
    state.update(selected)
    model.load_state_dict(state, strict=True)
    report = dict(
        format="yopo_rgb_s_depth_n_raw_features_v1",
        sources=reports,
        teacher_sha256=sha256(task),
        reused_task_leaves=len(task_keys),
        fresh_leaves=sorted(fresh),
        trainable_parameters=sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "meta": report}, output)
    report["output_sha256"] = sha256(output)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": report["output_sha256"],
                "parameters": report["trainable_parameters"],
                "raw_backbone_source_leaves_per_branch": 200,
                "raw_encoder_source_leaves_per_branch": 23,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    for name in ("rgb", "depth", "task", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    prepare(**vars(parser.parse_args()))
