#!/usr/bin/env python
"""Collect native Group Taylor/Fisher scores for YOPO transformer FFNs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import Runner

from yopo.pruning import (
    GroupGateImportanceCollector,
    discover_transformer_ffn_groups,
    select_top_channels,
)
from yopo.registry import MODELS
from yopo.utils import register_all_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Unpruned teacher config.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path, help="Output pruning-plan JSON.")
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--remaining", type=int, default=1024)
    parser.add_argument("--divisor", type=int, default=16)
    parser.add_argument(
        "--estimator",
        choices=("group_fisher", "group_taylor"),
        default="group_fisher",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Optional replacement root containing 2025/ and 2026/.",
    )
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help=(
            "Exercise hooks with synthetic tokens only. Scores from this mode "
            "must not be used as a training-selection claim."
        ),
    )
    parser.add_argument("--synthetic-tokens", type=int, default=32)
    parser.add_argument(
        "--purpose",
        choices=("smoke", "selection"),
        default="smoke",
        help=(
            "Declare whether the plan only validates wiring or is calibrated "
            "for checkpoint channel selection."
        ),
    )
    parser.add_argument(
        "--sampling",
        choices=("balanced_years", "concat"),
        default="balanced_years",
        help=(
            "Real-data sampling policy. balanced_years alternates the configured "
            "2025/2026 datasets; concat samples their union."
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_dataloader_cfg(cfg: Config, args: argparse.Namespace) -> None:
    loader = cfg.train_dataloader
    loader.batch_size = args.batch_size
    loader.num_workers = 0
    loader.persistent_workers = False
    loader.sampler.shuffle = True
    datasets = loader.dataset.datasets
    for dataset in datasets:
        if args.data_root is not None:
            year = Path(dataset.data_root.rstrip("/")).name
            dataset.data_root = str(args.data_root / year) + "/"
        dataset.pipeline = [
            transform
            for transform in dataset.pipeline
            if not str(transform.get("type", "")).startswith("Random")
        ]


def _build_real_data_iterators(
    cfg: Config,
    args: argparse.Namespace,
):
    loader_cfg = cfg.train_dataloader
    if args.sampling == "concat":
        loader = Runner.build_dataloader(loader_cfg, seed=args.seed)
        return (iter(loader),)

    iterators = []
    for index, dataset_cfg in enumerate(loader_cfg.dataset.datasets):
        year_loader_cfg = copy.deepcopy(loader_cfg)
        year_loader_cfg.dataset = copy.deepcopy(dataset_cfg)
        loader = Runner.build_dataloader(
            year_loader_cfg,
            seed=args.seed + index,
        )
        iterators.append(iter(loader))
    if len(iterators) < 2:
        raise ValueError(
            "balanced_years requires at least two configured train datasets"
        )
    return tuple(iterators)


def _synthetic_batch_loss(groups, args: argparse.Namespace) -> torch.Tensor:
    losses = []
    for group in groups:
        first_linear = next(
            module for module in group.sites[0].modules()
            if isinstance(module, torch.nn.Linear)
        )
        inputs = torch.randn(
            args.batch_size,
            args.synthetic_tokens,
            first_linear.in_features,
            device=args.device,
        )
        hidden = group.sites[0](inputs)
        losses.append(hidden.float().square().mean())
    return torch.stack(losses).mean()


def _real_batch_loss(model, data_batch) -> torch.Tensor:
    data = model.data_preprocessor(data_batch, training=True)
    losses = model(**data, mode="loss")
    parsed_loss, _ = model.parse_losses(losses)
    return parsed_loss


def main() -> None:
    args = parse_args()
    if args.batches <= 0 or args.batch_size <= 0:
        raise ValueError("batches and batch-size must be positive")
    if args.remaining <= 0 or args.remaining % args.divisor != 0:
        raise ValueError("remaining channels must be positive and divisor-aligned")
    if args.synthetic_smoke and args.purpose != "smoke":
        raise ValueError("synthetic calibration can only have purpose=smoke")
    if not args.config.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("config or checkpoint does not exist")

    torch.manual_seed(args.seed)
    register_all_modules()
    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" not in checkpoint:
        raise KeyError("checkpoint has no state_dict")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if args.synthetic_smoke:
        model.eval()
    else:
        # DINO only emits denoising and encoder-supervision inputs while its
        # training flag is set.  Keep that loss contract, but freeze BatchNorm
        # running statistics so calibration cannot mutate the teacher state.
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
    groups = discover_transformer_ffn_groups(model)
    for group in groups:
        if args.remaining > group.hidden_channels:
            raise ValueError(
                f"cannot keep {args.remaining} of {group.hidden_channels} in {group.name}"
            )

    iterators = None
    if not args.synthetic_smoke:
        _prepare_dataloader_cfg(cfg, args)
        iterators = _build_real_data_iterators(cfg, args)

    loss_values: list[float] = []
    with GroupGateImportanceCollector(groups) as collector:
        for batch_index in range(args.batches):
            model.zero_grad(set_to_none=True)
            collector.begin_batch()
            if args.synthetic_smoke:
                loss = _synthetic_batch_loss(groups, args)
            else:
                iterator_index = batch_index % len(iterators)
                try:
                    data_batch = next(iterators[iterator_index])
                except StopIteration as error:
                    raise RuntimeError(
                        f"calibration dataloader {iterator_index} exhausted"
                    ) from error
                loss = _real_batch_loss(model, data_batch)
            if not bool(torch.isfinite(loss)):
                raise ValueError("calibration loss is not finite")
            loss.backward()
            collector.end_batch()
            loss_values.append(float(loss.detach().cpu()))
        fisher = collector.scores("fisher")
        taylor = collector.scores("taylor")

    selected_scores = fisher if args.estimator == "group_fisher" else taylor
    choices = {
        name: list(select_top_channels(score, args.remaining))
        for name, score in selected_scores.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    score_path = args.output.with_suffix(".scores.npz")
    np.savez_compressed(
        score_path,
        **{
            **{f"fisher/{name}": value.numpy() for name, value in fisher.items()},
            **{f"taylor/{name}": value.numpy() for name, value in taylor.items()},
        },
    )
    payload = {
        "schema_version": 1,
        "choices": choices,
        "metadata": {
            "estimator": args.estimator,
            "purpose": args.purpose,
            "calibration": (
                "synthetic_smoke_only" if args.synthetic_smoke else "real_train_data"
            ),
            "config": str(args.config.resolve()),
            "config_sha256": sha256(args.config),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint),
            "batches": args.batches,
            "batch_size": args.batch_size,
            "sample_count": collector.sample_count,
            "remaining": args.remaining,
            "divisor": args.divisor,
            "seed": args.seed,
            "sampling": (
                "synthetic" if args.synthetic_smoke else args.sampling
            ),
            "model_mode": (
                "eval" if args.synthetic_smoke else "train_with_batchnorm_frozen"
            ),
            "loss_values": loss_values,
            "score_archive": str(score_path.resolve()),
            "score_archive_sha256": sha256(score_path),
        },
        "summary": {
            name: {
                "channels": value.numel(),
                "minimum": float(value.min()),
                "maximum": float(value.max()),
                "mean": float(value.mean()),
            }
            for name, value in selected_scores.items()
        },
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "calibration": payload["metadata"]["calibration"],
        "estimator": args.estimator,
        "groups": len(groups),
        "output": str(args.output.resolve()),
        "sample_count": collector.sample_count,
        "scores": str(score_path.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
