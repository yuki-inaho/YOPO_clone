"""Extract a reusable HGNet depth-encoder checkpoint from MAE training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in state_dict.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise ValueError(f"no encoder.* keys in {args.source}")
    nonfinite = [key for key, value in encoder_state.items() if not torch.isfinite(value).all()]
    if nonfinite:
        raise ValueError(f"nonfinite encoder tensors: {nonfinite}")

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": encoder_state}, args.destination)
    print(f"saved {len(encoder_state)} finite encoder tensors to {args.destination}")


if __name__ == "__main__":
    main()
