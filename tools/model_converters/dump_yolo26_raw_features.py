"""Run in the source DEIM JAX uv environment to export raw feature fixtures."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from deimv2_jax.config import EncoderConfig
from deimv2_jax.models.encoder import apply_encoder
from deimv2_jax.models.yolo26 import apply_yolo26_backbone


def dump(checkpoint, scale, depth, output):
    config = json.loads((checkpoint / "manifest.json").read_text())["metadata"][
        "config"
    ]["model"]
    if config["backbone"]["variant"] != scale:
        raise ValueError("fixture scale does not match source manifest")
    encoder_config = EncoderConfig(**config["encoder"])
    tree = {}
    with np.load(checkpoint / "arrays.npz", allow_pickle=False) as archive:
        for key in archive.files:
            if not key.startswith(("params::backbone/", "params::encoder/")):
                continue
            path = key.removeprefix("params::").split("/")
            node = tree
            for part in path[:-1]:
                node = node.setdefault(part, {})
            node[path[-1]] = jnp.asarray(archive[key])
    rng = np.random.default_rng(20260906)
    fixture = {}
    for index, (height, width) in enumerate(((96, 128), (160, 192))):
        images = rng.random((1, height, width, 1 if depth else 3), dtype=np.float32)
        source_images = np.repeat(images, 3, axis=-1) if depth else images
        features = apply_yolo26_backbone(
            jnp.asarray(source_images), tree["backbone"], scale, (4, 6, 10)
        )
        encoded, _, _ = apply_encoder(tree["encoder"], features, encoder_config)
        fixture[f"images_{index}"] = images
        for level, value in enumerate(encoded):
            fixture[f"encoded_{index}_{level}"] = np.asarray(value)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **fixture)
    print(
        f"exported {scale} depth={depth} raw features: {output}; backend={jax.default_backend()}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scale", choices=("n", "s"), required=True)
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    dump(**vars(parser.parse_args()))
