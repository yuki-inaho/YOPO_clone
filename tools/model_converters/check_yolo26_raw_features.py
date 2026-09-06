"""Check raw JAX feature fixtures against the initialized YOPO branches."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config

from yopo.registry import MODELS
from yopo.utils import register_all_modules


def check(config, checkpoint, fixtures, output):
    register_all_modules()
    cfg = Config.fromfile(str(config))
    model = MODELS.build(cfg.model).eval()
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"],
        strict=True,
    )
    reports = []
    with torch.no_grad():
        for name in ("rgb", "depth"):
            branch = getattr(model.backbone, name + "_backbone")
            with np.load(fixtures / (name + ".npz"), allow_pickle=False) as fixture:
                for index in range(2):
                    images = torch.from_numpy(
                        fixture[f"images_{index}"].transpose(0, 3, 1, 2).copy()
                    )
                    actual = branch(images)
                    for level, value in enumerate(actual):
                        expected = torch.from_numpy(
                            fixture[f"encoded_{index}_{level}"]
                            .transpose(0, 3, 1, 2)
                            .copy()
                        )
                        difference = (value - expected).abs()
                        result = dict(
                            branch=name,
                            fixture=index,
                            level=level,
                            max_abs=float(difference.max()),
                            mean_abs=float(difference.mean()),
                            reference_abs_max=float(expected.abs().max()),
                            passed=bool(
                                torch.allclose(value, expected, atol=1e-4, rtol=1e-3)
                            ),
                        )
                        reports.append(result)
    report = dict(
        passed=all(item["passed"] for item in reports),
        comparisons=reports,
        atol=1e-4,
        rtol=1e-3,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise ValueError("raw backbone-plus-encoder parity failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    for name in ("config", "checkpoint", "fixtures", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    check(**vars(parser.parse_args()))
