#!/usr/bin/env python3
"""Dump RGB-D 3D BBOX validation predictions through the configured val loop."""

from __future__ import annotations

import argparse
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner

from yopo.evaluation import DumpDetResults


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="RGB-D 3D BBOX config with val_dataloader")
    parser.add_argument("checkpoint", help="full model checkpoint to evaluate")
    parser.add_argument("output", help="prediction dump path ending in .pkl")
    parser.add_argument("--work-dir", help="optional val-loop log directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.suffix not in {".pkl", ".pickle"}:
        raise ValueError(f"output must be .pkl or .pickle, got {output}")
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {args.checkpoint}")

    output.parent.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint
    cfg.work_dir = args.work_dir or str(output.parent / "val_dump")
    # This tool runs ``Runner.val()`` without the train loop.  The training
    # checkpoint hook expects ``before_train`` to initialize its file backend,
    # so retaining it would fail after an otherwise complete validation.
    cfg.default_hooks.pop("checkpoint", None)
    runner = Runner.from_cfg(cfg)
    runner.val_evaluator.metrics.append(
        DumpDetResults(out_file_path=str(output)))
    runner.val()

    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"validation completed without a prediction dump: {output}")
    print(f"prediction dump: {output.resolve()} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
