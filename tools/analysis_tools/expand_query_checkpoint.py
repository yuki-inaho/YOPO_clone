#!/usr/bin/env python3
"""Expand a learned DETR query table without discarding existing queries."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch

from yopo.utils import register_mmengine_checkpoint_safe_globals


def expand_query_embedding(
    query_embedding: torch.Tensor,
    num_queries: int,
    *,
    seed: int = 3407,
    noise_scale: float = 0.01,
) -> torch.Tensor:
    """Copy learned queries and initialize extra rows from noisy learned rows."""
    if query_embedding.ndim != 2:
        raise ValueError(
            "query_embedding must be rank 2, got "
            f"shape={tuple(query_embedding.shape)}"
        )
    old_queries = query_embedding.shape[0]
    if num_queries < old_queries:
        raise ValueError(
            f"num_queries={num_queries} cannot shrink old_queries={old_queries}"
        )
    if num_queries == old_queries:
        return query_embedding.clone()

    generator = torch.Generator(device="cpu").manual_seed(seed)
    extra_count = num_queries - old_queries
    source_indices = torch.arange(extra_count) % old_queries
    extra = query_embedding.detach().cpu()[source_indices].clone()
    per_dimension_std = query_embedding.detach().float().cpu().std(dim=0)
    noise = torch.randn(extra.shape, generator=generator, dtype=torch.float32)
    extra = extra.float() + noise * per_dimension_std * noise_scale
    return torch.cat(
        [query_embedding.detach().cpu(), extra.to(query_embedding.dtype)], dim=0
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--num-queries", type=int, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--noise-scale", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.noise_scale < 0:
        raise ValueError("noise-scale must be non-negative")

    register_mmengine_checkpoint_safe_globals()
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError("checkpoint must contain a state_dict mapping")
    key = "query_embedding.weight"
    if key not in state_dict:
        raise KeyError(f"checkpoint state_dict lacks {key}")

    old_shape = tuple(state_dict[key].shape)
    state_dict[key] = expand_query_embedding(
        state_dict[key],
        args.num_queries,
        seed=args.seed,
        noise_scale=args.noise_scale,
    )
    new_shape = tuple(state_dict[key].shape)
    checkpoint.setdefault("meta", {})["query_expansion"] = {
        "source": str(args.input.resolve()),
        "old_shape": old_shape,
        "new_shape": new_shape,
        "seed": args.seed,
        "noise_scale": args.noise_scale,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, args.output)
    print(f"expanded: {old_shape} -> {new_shape}")
    print(f"output: {args.output.resolve()}")
    print(f"sha256: {_sha256(args.output)}")


if __name__ == "__main__":
    main()
