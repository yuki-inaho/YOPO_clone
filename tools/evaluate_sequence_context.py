#!/usr/bin/env python3
"""Reload and proxy-evaluate a raw YOPO sequence-context checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from mmengine.config import Config

from yopo.models.tracking.sequence_training import (
    CHECKPOINT_SCHEMA,
    atomic_write_json,
    build_pair_examples,
    evaluate_model,
    load_feature_cache,
    make_identity_head,
    sha256_file,
    uses_geometry_matching,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "smoke"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.fromfile(args.config)
    settings = config.sequence_context
    cache = load_feature_cache(args.feature_cache)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported sequence context checkpoint schema")
    expected = {
        "manifest_sha256": sha256_file(args.manifest),
        "config_sha256": sha256_file(args.config),
        "feature_cache_sha256": sha256_file(args.feature_cache),
    }
    for field, value in expected.items():
        if checkpoint["provenance"].get(field) != value:
            raise ValueError(f"checkpoint provenance mismatch: {field}")
    device = torch.device(args.device)
    model = make_identity_head(
        mode=checkpoint["mode"],
        appearance_dim=int(checkpoint["appearance_dim"]),
        head_config=checkpoint["head_config"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    examples = build_pair_examples(
        args.manifest,
        split=args.split,
        cache=cache,
        max_distance_m=float(settings.association.max_distance_m),
        ambiguity_margin_m=float(settings.association.ambiguity_margin_m),
        min_matches=int(settings.pairs.min_eval_matches),
        max_pairs=args.max_pairs,
    )
    metrics = evaluate_model(
        model,
        examples,
        cache,
        device=device,
        embedding_gate=float(settings.evaluation.embedding_gate),
        geometry_gate_m=float(settings.association.max_distance_m),
        use_geometry_matching=uses_geometry_matching(checkpoint["mode"]),
        loss_config=checkpoint.get("loss_config"),
    )
    report = {
        "status": "completed",
        "annotation_kind": "pseudo",
        "metric_prefix": "proxy_",
        "split": args.split,
        "mode": checkpoint["mode"],
        "checkpoint_kind": checkpoint["checkpoint_kind"],
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "loss_config": checkpoint.get("loss_config"),
        "metrics": metrics,
        "provenance": expected,
    }
    atomic_write_json(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
