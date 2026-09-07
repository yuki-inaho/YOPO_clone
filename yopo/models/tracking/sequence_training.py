"""Reusable training primitives for YOPO sequence identity experiments."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from yopo.datasets.pose_estimation.yopo_sequence import SequenceIndex
from yopo.engine.optimizers.amuse import AmuseOptimizer

from .geometry_context import (
    GeometryContextIdentityHead,
    matched_identity_loss,
    mutual_nearest_association,
    stable_pairwise_distance,
)
from .online_tracker import partial_linear_assignment
from .prediction_cache import PREDICTION_FEATURE_CACHE_SCHEMA

FEATURE_CACHE_SCHEMA = "yopo_g10_sequence_features_v2"
CHECKPOINT_SCHEMA = "yopo_sequence_context_raw_v2"
EXPERIMENT_MODES = ("B0", "B1", "B2", "C0", "C1")


def uses_geometry_matching(mode: str) -> bool:
    if mode not in EXPERIMENT_MODES:
        raise ValueError(f"unsupported sequence experiment mode: {mode!r}")
    return mode in {"B1", "B2", "C1"}


@dataclass(frozen=True)
class PairExample:
    """One unique adjacent-frame pseudo-association training example."""

    source_key: str
    target_key: str
    matches: Tensor


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, document: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def atomic_torch_save(path: str | Path, document: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(dict(document), temporary)
    os.replace(temporary, destination)


def sample_pyramid_at_centers(
    features: Sequence[Tensor],
    *,
    batch_index: int,
    centers_px: Tensor,
    image_size: tuple[int, int],
) -> Tensor:
    """Bilinearly sample every G10 pyramid level at OBB pixel centres."""

    height, width = image_size
    if centers_px.ndim != 2 or centers_px.shape[1] != 2:
        raise ValueError("centers_px must have shape [N, 2]")
    if not features:
        raise ValueError("at least one feature level is required")
    if centers_px.numel() == 0:
        channels = sum(int(level.shape[1]) for level in features)
        return features[0].new_empty((0, channels))
    if not torch.isfinite(centers_px).all():
        raise ValueError("centers_px must be finite")
    if (
        (centers_px[:, 0] < 0).any()
        or (centers_px[:, 0] >= width).any()
        or (centers_px[:, 1] < 0).any()
        or (centers_px[:, 1] >= height).any()
    ):
        raise ValueError("centers_px must lie inside the source image")
    normalized = centers_px.clone()
    normalized[:, 0] = (normalized[:, 0] + 0.5) * (2.0 / width) - 1.0
    normalized[:, 1] = (normalized[:, 1] + 0.5) * (2.0 / height) - 1.0
    sampled = []
    for level in features:
        if level.ndim != 4 or batch_index >= level.shape[0]:
            raise ValueError("invalid pyramid feature shape or batch index")
        level_grid = normalized.clone()
        level_grid[:, 0].clamp_(
            min=-1.0 + 1.0 / level.shape[-1], max=1.0 - 1.0 / level.shape[-1]
        )
        level_grid[:, 1].clamp_(
            min=-1.0 + 1.0 / level.shape[-2], max=1.0 - 1.0 / level.shape[-2]
        )
        values = F.grid_sample(
            level[batch_index : batch_index + 1],
            level_grid.reshape(1, -1, 1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled.append(values[0, :, :, 0].transpose(0, 1))
    return torch.cat(sampled, dim=1)


def frame_key(scene: str, frame_id: int) -> str:
    return f"{scene}/{frame_id:06d}"


def load_feature_cache(path: str | Path) -> dict[str, Any]:
    document = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(document, dict) or document.get("schema") not in {
        FEATURE_CACHE_SCHEMA,
        PREDICTION_FEATURE_CACHE_SCHEMA,
    }:
        raise ValueError("unsupported sequence feature cache schema")
    if not isinstance(document.get("frames"), dict):
        raise ValueError("sequence feature cache has no frames")
    return document


def build_pair_examples(
    manifest_path: str | Path,
    *,
    split: str,
    cache: Mapping[str, Any],
    max_distance_m: float,
    ambiguity_margin_m: float,
    min_matches: int = 2,
    max_pairs: int | None = None,
) -> tuple[PairExample, ...]:
    """Build each source-authored adjacent pair once, without window weighting."""

    if min_matches < 0:
        raise ValueError("min_matches must be nonnegative")
    index = SequenceIndex(manifest_path, split=split)
    frames = cache.get("frames")
    if not isinstance(frames, Mapping):
        raise ValueError("feature cache frames must be a mapping")
    seen: set[tuple[str, int, int]] = set()
    examples: list[PairExample] = []
    for window in index.windows:
        for source_id, target_id in zip(window.frame_ids, window.frame_ids[1:]):
            identity = (window.scene, source_id, target_id)
            if identity in seen:
                continue
            seen.add(identity)
            source_key = frame_key(window.scene, source_id)
            target_key = frame_key(window.scene, target_id)
            if source_key not in frames or target_key not in frames:
                raise ValueError(f"feature cache misses source pair {identity}")
            source_frame = frames[source_key]
            target_frame = frames[target_key]
            source_centers = source_frame["centers_world"].float()
            target_centers = target_frame["centers_world"].float()
            source_rows = torch.arange(len(source_centers))
            target_rows = torch.arange(len(target_centers))
            if cache.get("schema") == PREDICTION_FEATURE_CACHE_SCHEMA:
                source_teacher_indices = source_frame["teacher_indices"].long()
                target_teacher_indices = target_frame["teacher_indices"].long()
                source_rows = (source_teacher_indices >= 0).nonzero().flatten()
                target_rows = (target_teacher_indices >= 0).nonzero().flatten()
                source_centers = source_frame["teacher_centers_world"][
                    source_rows
                ].float()
                target_centers = target_frame["teacher_centers_world"][
                    target_rows
                ].float()
            association = mutual_nearest_association(
                source_centers,
                target_centers,
                max_distance_m=max_distance_m,
                ambiguity_margin_m=ambiguity_margin_m,
            )
            if association.matches.shape[0] >= min_matches:
                matches = association.matches
                if cache.get("schema") == PREDICTION_FEATURE_CACHE_SCHEMA:
                    matches = torch.stack(
                        (
                            source_rows[matches[:, 0]],
                            target_rows[matches[:, 1]],
                        ),
                        dim=1,
                    )
                examples.append(PairExample(source_key, target_key, matches.cpu()))
            if max_pairs is not None and len(examples) >= max_pairs:
                return tuple(examples)
    if not examples:
        raise ValueError(
            f"split {split!r} has no pair with at least {min_matches} pseudo matches"
        )
    return tuple(examples)


def make_identity_head(
    *,
    mode: str,
    appearance_dim: int,
    head_config: Mapping[str, Any],
) -> GeometryContextIdentityHead:
    if mode == "B1":
        raise ValueError("B1 is the non-learned geometry-only baseline")
    return GeometryContextIdentityHead(
        appearance_dim=appearance_dim,
        embedding_dim=int(head_config["embedding_dim"]),
        mode=mode,
        hidden_dim=int(head_config["hidden_dim"]),
        context_radius_m=float(head_config["context_radius_m"]),
        fusion_strategy=str(head_config.get("fusion_strategy", "legacy_concat_v1")),
    )


def warm_start_residual_context(
    context_model: GeometryContextIdentityHead,
    parent_model: GeometryContextIdentityHead,
) -> dict[str, Any]:
    """Strictly initialize a residual context adapter from an appearance parent."""

    if parent_model.fusion_strategy != "residual_gated_v1":
        raise ValueError("appearance parent must use residual_gated_v1")
    if context_model.fusion_strategy != "residual_gated_v1":
        raise ValueError("context model must use residual_gated_v1")
    if parent_model.mode not in {"B0", "B2"}:
        raise ValueError("appearance parent mode must be B0 or B2")
    if context_model.mode not in {"C0", "C1"}:
        raise ValueError("context model mode must be C0 or C1")
    context_model.load_state_dict(parent_model.state_dict(), strict=True)
    if context_model.context_gate.detach().item() != 0.0:
        raise ValueError("residual context warm start requires a zero gate")
    frozen_names: list[str] = []
    trainable_names: list[str] = []
    for name, parameter in context_model.named_parameters():
        trainable = name.startswith(("context_encoder.", "context_projection.")) or (
            name == "context_gate"
        )
        parameter.requires_grad_(trainable)
        (trainable_names if trainable else frozen_names).append(name)
    if not frozen_names or not trainable_names:
        raise RuntimeError("residual context parameter partition is empty")
    return {
        "parent_parity_verified": True,
        "frozen_parameter_names": frozen_names,
        "trainable_parameter_names": trainable_names,
    }


def make_amuse_optimizer(
    model: nn.Module, optimizer_config: Mapping[str, Any]
) -> AmuseOptimizer:
    config = dict(optimizer_config)
    optimizer_type = config.pop("type", "AmuseOptimizer")
    if optimizer_type not in {
        "AmuseOptimizer",
        "yopo.engine.optimizers.amuse.AmuseOptimizer",
    }:
        raise ValueError(f"unsupported sequence optimizer: {optimizer_type}")
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    optimizer = AmuseOptimizer(trainable, **config)
    optimizer.train()
    return optimizer


def _frame_tensors(
    cache: Mapping[str, Any], key: str, device: torch.device
) -> tuple[Tensor, Tensor]:
    frame = cache["frames"][key]
    return (
        frame["appearance"].to(device=device, dtype=torch.float32),
        frame["centers_world"].to(device=device, dtype=torch.float32),
    )


def train_epoch(
    model: GeometryContextIdentityHead,
    optimizer: AmuseOptimizer,
    examples: Sequence[PairExample],
    cache: Mapping[str, Any],
    *,
    device: torch.device,
    pair_batch_size: int,
    seed: int,
    loss_config: Mapping[str, Any] | None = None,
) -> dict[str, float | int]:
    model.train()
    optimizer.train()
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    total_loss = 0.0
    update_count = 0
    example_count = 0
    learning_rate_start: float | None = None
    learning_rate_end: float | None = None
    gradient_norm_max = 0.0
    loss_settings = dict(loss_config or {})
    for offset in range(0, len(order), pair_batch_size):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index in order[offset : offset + pair_batch_size]:
            example = examples[index]
            source_appearance, source_centers = _frame_tensors(
                cache, example.source_key, device
            )
            target_appearance, target_centers = _frame_tensors(
                cache, example.target_key, device
            )
            matches = example.matches.to(device)
            losses.append(
                matched_identity_loss(
                    model(source_appearance, source_centers),
                    model(target_appearance, target_centers),
                    matches,
                    **loss_settings,
                )
            )
        if not losses:
            continue
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("sequence identity loss became nonfinite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("sequence identity gradient became nonfinite")
        gradient_norm_max = max(gradient_norm_max, float(gradient_norm))
        optimizer.step()
        current_lr = max(float(group["lr"]) for group in optimizer.param_groups)
        if learning_rate_start is None:
            learning_rate_start = current_lr
        learning_rate_end = current_lr
        total_loss += float(loss.detach()) * len(losses)
        example_count += len(losses)
        update_count += 1
    return {
        "loss": total_loss / max(example_count, 1),
        "examples": example_count,
        "updates": update_count,
        "learning_rate_start": learning_rate_start or 0.0,
        "learning_rate_end": learning_rate_end or 0.0,
        "gradient_norm_max": gradient_norm_max,
    }


def _embedding_mutual_pairs(
    source_embedding: Tensor,
    target_embedding: Tensor,
    source_centers: Tensor,
    target_centers: Tensor,
    *,
    embedding_gate: float,
    geometry_gate_m: float | None,
) -> set[tuple[int, int]]:
    if source_embedding.numel() == 0 or target_embedding.numel() == 0:
        return set()
    cosine_distance = 1.0 - source_embedding @ target_embedding.transpose(0, 1)
    cosine_distance = cosine_distance.clamp_min(0.0)
    cost = cosine_distance
    valid = cosine_distance <= embedding_gate
    if geometry_gate_m is not None:
        source_geometry_valid = torch.isfinite(source_centers).all(dim=1)
        target_geometry_valid = torch.isfinite(target_centers).all(dim=1)
        geometry_available = source_geometry_valid.unsqueeze(
            1
        ) & target_geometry_valid.unsqueeze(0)
        safe_source = torch.where(
            source_geometry_valid.unsqueeze(1),
            source_centers,
            torch.zeros_like(source_centers),
        )
        safe_target = torch.where(
            target_geometry_valid.unsqueeze(1),
            target_centers,
            torch.zeros_like(target_centers),
        )
        geometry_distance = stable_pairwise_distance(safe_source, safe_target)
        valid &= ~geometry_available | (geometry_distance <= geometry_gate_m)
        cost = 0.5 * cosine_distance + 0.5 * torch.where(
            geometry_available,
            geometry_distance / geometry_gate_m,
            cosine_distance,
        )
    return set(partial_linear_assignment(cost, valid, miss_cost=0.55, new_cost=0.55))


@torch.no_grad()
def evaluate_model(
    model: GeometryContextIdentityHead,
    examples: Sequence[PairExample],
    cache: Mapping[str, Any],
    *,
    device: torch.device,
    embedding_gate: float,
    geometry_gate_m: float,
    use_geometry_matching: bool = False,
    loss_config: Mapping[str, Any] | None = None,
) -> dict[str, float | int]:
    """Evaluate only against geometry-derived pseudo pairs (proxy metrics)."""

    model.eval()
    true_positive = 0
    predicted_count = 0
    pseudo_count = 0
    total_loss = 0.0
    loss_pair_count = 0
    zero_positive_pair_count = 0
    single_positive_pair_count = 0
    started = time.perf_counter()
    loss_settings = dict(loss_config or {})
    for example in examples:
        source_appearance, source_centers = _frame_tensors(
            cache, example.source_key, device
        )
        target_appearance, target_centers = _frame_tensors(
            cache, example.target_key, device
        )
        matches = example.matches.to(device)
        source_embedding = model(source_appearance, source_centers)
        target_embedding = model(target_appearance, target_centers)
        if matches.shape[0] >= 2:
            loss = matched_identity_loss(
                source_embedding, target_embedding, matches, **loss_settings
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("proxy evaluation loss became nonfinite")
            total_loss += float(loss)
            loss_pair_count += 1
        elif matches.shape[0] == 1:
            single_positive_pair_count += 1
        else:
            zero_positive_pair_count += 1
        predicted = _embedding_mutual_pairs(
            source_embedding,
            target_embedding,
            source_centers,
            target_centers,
            embedding_gate=embedding_gate,
            geometry_gate_m=geometry_gate_m if use_geometry_matching else None,
        )
        pseudo = {tuple(pair) for pair in example.matches.tolist()}
        true_positive += len(predicted & pseudo)
        predicted_count += len(predicted)
        pseudo_count += len(pseudo)
    elapsed = time.perf_counter() - started
    precision = true_positive / predicted_count if predicted_count else 0.0
    recall = true_positive / pseudo_count if pseudo_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "proxy_loss": total_loss / max(loss_pair_count, 1),
        "proxy_association_precision": precision,
        "proxy_association_recall": recall,
        "proxy_association_f1": f1,
        "proxy_true_positive": true_positive,
        "proxy_predicted_pairs": predicted_count,
        "proxy_pseudo_pairs": pseudo_count,
        "pair_count": len(examples),
        "loss_pair_count": loss_pair_count,
        "zero_positive_pair_count": zero_positive_pair_count,
        "single_positive_pair_count": single_positive_pair_count,
        "latency_ms_per_pair": elapsed * 1000.0 / len(examples),
    }


def _train_modes_independent(
    *,
    modes: Sequence[str],
    train_examples: Sequence[PairExample],
    validation_examples: Sequence[PairExample],
    cache: Mapping[str, Any],
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    work_dir: str | Path,
    device: torch.device,
    epochs: int,
    patience: int,
) -> dict[str, Any]:
    """Train comparable descriptors and report the fixed geometry-only baseline."""

    destination = Path(work_dir)
    destination.mkdir(parents=True, exist_ok=True)
    appearance_dim = int(cache["appearance_dim"])
    mode_reports: dict[str, Any] = {}
    loss_config = dict(
        config.get("loss", {"temperature": 0.07, "negative_scope": "matched_only"})
    )
    if set(modes) != set(EXPERIMENT_MODES):
        raise ValueError(
            f"fair comparison requires exactly these modes: {EXPERIMENT_MODES}"
        )
    for mode in modes:
        if mode == "B1":

            class ConstantGeometryBaseline(nn.Module):
                def forward(self, appearance: Tensor, positions: Tensor) -> Tensor:
                    embedding = appearance.new_zeros((appearance.shape[0], 1))
                    embedding[:, 0] = 1.0
                    return embedding

            baseline = ConstantGeometryBaseline().to(device)
            metrics = evaluate_model(
                baseline,
                validation_examples,
                cache,
                device=device,
                embedding_gate=float(config["embedding_gate"]),
                geometry_gate_m=float(config["association"]["max_distance_m"]),
                use_geometry_matching=True,
                loss_config=loss_config,
            )
            mode_reports[mode] = {
                "kind": "non_learned_geometry_only",
                "initialization_seed": int(config["seed"]),
                "best_epoch": 0,
                "best_checkpoint": None,
                "best_checkpoint_sha256": None,
                "early_stopped": False,
                "epochs_ran": 0,
                "epoch0_metrics": metrics,
                "history": [],
                "reload_metrics": metrics,
            }
            continue
        torch.manual_seed(int(config["seed"]))
        model = make_identity_head(
            mode=mode,
            appearance_dim=appearance_dim,
            head_config=config["head"],
        ).to(device)
        use_geometry = uses_geometry_matching(mode)
        epoch0_metrics = evaluate_model(
            model,
            validation_examples,
            cache,
            device=device,
            embedding_gate=float(config["embedding_gate"]),
            geometry_gate_m=float(config["association"]["max_distance_m"]),
            use_geometry_matching=use_geometry,
            loss_config=loss_config,
        )
        best_path = destination / f"best_raw_{mode.lower()}.pth"
        best_score = float(epoch0_metrics["proxy_association_f1"])
        best_epoch = 0
        stale_epochs = 0
        history = []
        atomic_torch_save(
            best_path,
            {
                "schema": CHECKPOINT_SCHEMA,
                "checkpoint_kind": "raw",
                "mode": mode,
                "uses_geometry_matching": use_geometry,
                "descriptor_contract": "appearance_plus_distance_context_v1",
                "appearance_dim": appearance_dim,
                "head_config": dict(config["head"]),
                "loss_config": loss_config,
                "model_state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "epoch": 0,
                "metrics": dict(epoch0_metrics),
                "provenance": dict(provenance),
            },
        )
        optimizer = make_amuse_optimizer(model, config["optimizer"])
        warmup_steps = int(config["optimizer"]["warmup_steps"])
        cumulative_updates = 0
        for epoch in range(1, epochs + 1):
            train_metrics = train_epoch(
                model,
                optimizer,
                train_examples,
                cache,
                device=device,
                pair_batch_size=int(config["pair_batch_size"]),
                seed=int(config["seed"]) + epoch,
                loss_config=loss_config,
            )
            cumulative_updates += int(train_metrics["updates"])
            optimizer.eval()
            validation_metrics = evaluate_model(
                model,
                validation_examples,
                cache,
                device=device,
                embedding_gate=float(config["embedding_gate"]),
                geometry_gate_m=float(config["association"]["max_distance_m"]),
                use_geometry_matching=use_geometry,
                loss_config=loss_config,
            )
            row = {
                "epoch": epoch,
                **train_metrics,
                "cumulative_updates": cumulative_updates,
                "warmup_steps": warmup_steps,
                "warmup_completed": cumulative_updates >= warmup_steps,
                **validation_metrics,
            }
            history.append(row)
            score = float(validation_metrics["proxy_association_f1"])
            if score > best_score:
                best_score = score
                best_epoch = epoch
                stale_epochs = 0
                raw_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                atomic_torch_save(
                    best_path,
                    {
                        "schema": CHECKPOINT_SCHEMA,
                        "checkpoint_kind": "raw",
                        "mode": mode,
                        "uses_geometry_matching": use_geometry,
                        "descriptor_contract": "appearance_plus_distance_context_v1",
                        "appearance_dim": appearance_dim,
                        "head_config": dict(config["head"]),
                        "loss_config": loss_config,
                        "model_state_dict": raw_state,
                        "epoch": epoch,
                        "metrics": dict(validation_metrics),
                        "provenance": dict(provenance),
                    },
                )
            else:
                stale_epochs += 1
            if stale_epochs >= patience and cumulative_updates >= warmup_steps:
                break
        reloaded = torch.load(best_path, map_location=device, weights_only=False)
        best_model = make_identity_head(
            mode=mode,
            appearance_dim=appearance_dim,
            head_config=config["head"],
        ).to(device)
        best_model.load_state_dict(reloaded["model_state_dict"], strict=True)
        reload_metrics = evaluate_model(
            best_model,
            validation_examples,
            cache,
            device=device,
            embedding_gate=float(config["embedding_gate"]),
            geometry_gate_m=float(config["association"]["max_distance_m"]),
            use_geometry_matching=use_geometry,
            loss_config=loss_config,
        )
        mode_reports[mode] = {
            "kind": "trained_descriptor",
            "initialization_seed": int(config["seed"]),
            "best_epoch": best_epoch,
            "best_checkpoint": str(best_path),
            "best_checkpoint_sha256": sha256_file(best_path),
            "early_stopped": len(history) < epochs,
            "epochs_ran": len(history),
            "actual_updates": cumulative_updates,
            "warmup_steps": warmup_steps,
            "warmup_completed": cumulative_updates >= warmup_steps,
            "epoch0_metrics": epoch0_metrics,
            "history": history,
            "reload_metrics": reload_metrics,
        }

    def score(mode: str) -> float:
        return float(
            mode_reports.get(mode, {})
            .get("reload_metrics", {})
            .get("proxy_association_f1", -1.0)
        )

    geometry_only = score("B1")
    candidate_mode = "C1" if score("C1") > score("B2") else "B2"
    if candidate_mode not in mode_reports:
        trainable = [mode for mode in modes if mode != "B1"]
        candidate_mode = max(trainable, key=score) if trainable else "B1"
    candidate_epoch0 = float(
        mode_reports[candidate_mode]
        .get("epoch0_metrics", {})
        .get("proxy_association_f1", -1.0)
    )
    candidate_is_trained = mode_reports[candidate_mode]["best_checkpoint"] is not None
    proxy_candidate_passes = (
        candidate_is_trained
        and score(candidate_mode) > geometry_only
        and score(candidate_mode) > candidate_epoch0
    )
    return {
        "modes": mode_reports,
        "selection": {
            "status": (
                "proxy_candidate_only"
                if proxy_candidate_passes
                else "no_proxy_candidate_passes_baselines"
            ),
            "candidate_mode": candidate_mode,
            "candidate_checkpoint": mode_reports[candidate_mode]["best_checkpoint"],
            "proxy_candidate_passes_baselines": proxy_candidate_passes,
            "candidate_improves_epoch0": score(candidate_mode) > candidate_epoch0,
            "candidate_improves_geometry_only": score(candidate_mode) > geometry_only,
            "c0_improves_b0": score("C0") > score("B0"),
            "c1_improves_b2": score("C1") > score("B2"),
            "adoption_requires_independently_checked_correspondence": True,
            "advance_to_c2_p1_f1": False,
        },
    }


def _metric_score(metrics: Mapping[str, Any]) -> float:
    return float(metrics["proxy_association_f1"])


def _select_context_candidate(
    context_reload: Mapping[str, Mapping[str, Any]],
    *,
    c0_improves: bool,
    c1_improves: bool,
) -> tuple[str, str]:
    improved_modes = [
        mode
        for mode, improved in (("C0", c0_improves), ("C1", c1_improves))
        if improved
    ]
    candidate_pool = improved_modes or ["C0", "C1"]
    candidate = max(
        candidate_pool, key=lambda mode: _metric_score(context_reload[mode])
    )
    basis = (
        "highest_absolute_proxy_f1_among_improved_context_modes"
        if improved_modes
        else "highest_absolute_proxy_f1_for_diagnostics_only"
    )
    return candidate, basis


def _checkpoint_payload(
    *,
    model: GeometryContextIdentityHead,
    mode: str,
    appearance_dim: int,
    head_config: Mapping[str, Any],
    loss_config: Mapping[str, Any],
    epoch: int,
    metrics: Mapping[str, Any],
    provenance: Mapping[str, Any],
    trajectory_id: str,
    parent_checkpoint_sha256: str | None = None,
    frozen_parameter_names: Sequence[str] = (),
    trainable_parameter_names: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "checkpoint_kind": "raw",
        "mode": mode,
        "uses_geometry_matching": uses_geometry_matching(mode),
        "descriptor_contract": "residual_gated_context_curriculum_v1",
        "training_strategy": "residual_context_curriculum_v1",
        "trajectory_id": trajectory_id,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "frozen_parameter_names": list(frozen_parameter_names),
        "trainable_parameter_names": list(trainable_parameter_names),
        "appearance_dim": appearance_dim,
        "head_config": dict(head_config),
        "loss_config": dict(loss_config),
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "epoch": epoch,
        "metrics": dict(metrics),
        "provenance": dict(provenance),
    }


def _evaluate_view(
    model: GeometryContextIdentityHead,
    mode: str,
    examples: Sequence[PairExample],
    cache: Mapping[str, Any],
    config: Mapping[str, Any],
    loss_config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, float | int]:
    return evaluate_model(
        model,
        examples,
        cache,
        device=device,
        embedding_gate=float(config["embedding_gate"]),
        geometry_gate_m=float(config["association"]["max_distance_m"]),
        use_geometry_matching=uses_geometry_matching(mode),
        loss_config=loss_config,
    )


def _reload_view(
    path: Path,
    *,
    mode: str,
    appearance_dim: int,
    head_config: Mapping[str, Any],
    examples: Sequence[PairExample],
    cache: Mapping[str, Any],
    config: Mapping[str, Any],
    loss_config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, float | int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("mode") != mode:
        raise ValueError("shared trajectory checkpoint mode mismatch")
    model = make_identity_head(
        mode=mode, appearance_dim=appearance_dim, head_config=head_config
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return _evaluate_view(model, mode, examples, cache, config, loss_config, device)


def _assert_metric_parity(
    actual: Mapping[str, float | int], expected: Mapping[str, float | int]
) -> None:
    for key, value in expected.items():
        if key == "latency_ms_per_pair":
            continue
        if actual[key] != value:
            raise RuntimeError(f"zero-gate parent metric mismatch: {key}")


def _assert_embedding_parity(
    parent: GeometryContextIdentityHead,
    context: GeometryContextIdentityHead,
    examples: Sequence[PairExample],
    cache: Mapping[str, Any],
    device: torch.device,
) -> None:
    parent.eval()
    context.eval()
    with torch.no_grad():
        for example in examples:
            for key in (example.source_key, example.target_key):
                appearance, centers = _frame_tensors(cache, key, device)
                if not torch.equal(
                    parent(appearance, centers), context(appearance, centers)
                ):
                    raise RuntimeError("zero-gate context does not preserve parent")


def _constant_geometry_report(
    *,
    examples: Sequence[PairExample],
    cache: Mapping[str, Any],
    config: Mapping[str, Any],
    loss_config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    class ConstantGeometryBaseline(nn.Module):
        def forward(self, appearance: Tensor, positions: Tensor) -> Tensor:
            embedding = appearance.new_zeros((appearance.shape[0], 1))
            embedding[:, 0] = 1.0
            return embedding

    metrics = _evaluate_view(
        ConstantGeometryBaseline().to(device),
        "B1",
        examples,
        cache,
        config,
        loss_config,
        device,
    )
    return {
        "kind": "non_learned_geometry_only",
        "trajectory_id": None,
        "initialization_seed": int(config["seed"]),
        "best_epoch": 0,
        "best_checkpoint": None,
        "best_checkpoint_sha256": None,
        "early_stopped": False,
        "epochs_ran": 0,
        "epoch0_metrics": metrics,
        "history": [],
        "reload_metrics": metrics,
    }


def _train_residual_context_curriculum(
    *,
    modes: Sequence[str],
    train_examples: Sequence[PairExample],
    validation_examples: Sequence[PairExample],
    cache: Mapping[str, Any],
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    work_dir: str | Path,
    device: torch.device,
    epochs: int,
    patience: int,
) -> dict[str, Any]:
    if set(modes) != set(EXPERIMENT_MODES):
        raise ValueError(
            f"fair comparison requires exactly these modes: {EXPERIMENT_MODES}"
        )
    head_config = dict(config["head"])
    if head_config.get("fusion_strategy") != "residual_gated_v1":
        raise ValueError("residual curriculum requires residual_gated_v1 fusion")
    destination = Path(work_dir)
    destination.mkdir(parents=True, exist_ok=True)
    appearance_dim = int(cache["appearance_dim"])
    seed = int(config["seed"])
    loss_config = dict(
        config.get("loss", {"temperature": 0.07, "negative_scope": "matched_only"})
    )
    warmup_steps = int(config["optimizer"]["warmup_steps"])

    torch.manual_seed(seed)
    appearance_model = make_identity_head(
        mode="B0", appearance_dim=appearance_dim, head_config=head_config
    ).to(device)
    appearance_epoch0 = {
        mode: _evaluate_view(
            appearance_model,
            mode,
            validation_examples,
            cache,
            config,
            loss_config,
            device,
        )
        for mode in ("B0", "B2")
    }
    appearance_paths = {
        mode: destination / f"best_raw_{mode.lower()}.pth" for mode in ("B0", "B2")
    }
    appearance_best_score = _metric_score(appearance_epoch0["B0"])
    appearance_best_epoch = 0
    for mode in ("B0", "B2"):
        atomic_torch_save(
            appearance_paths[mode],
            _checkpoint_payload(
                model=appearance_model,
                mode=mode,
                appearance_dim=appearance_dim,
                head_config=head_config,
                loss_config=loss_config,
                epoch=0,
                metrics=appearance_epoch0[mode],
                provenance=provenance,
                trajectory_id="appearance",
            ),
        )
    appearance_optimizer = make_amuse_optimizer(appearance_model, config["optimizer"])
    appearance_history = {"B0": [], "B2": []}
    appearance_updates = 0
    appearance_stale = 0
    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(
            appearance_model,
            appearance_optimizer,
            train_examples,
            cache,
            device=device,
            pair_batch_size=int(config["pair_batch_size"]),
            seed=seed + epoch,
            loss_config=loss_config,
        )
        appearance_updates += int(train_metrics["updates"])
        appearance_optimizer.eval()
        epoch_metrics = {
            mode: _evaluate_view(
                appearance_model,
                mode,
                validation_examples,
                cache,
                config,
                loss_config,
                device,
            )
            for mode in ("B0", "B2")
        }
        for mode in ("B0", "B2"):
            appearance_history[mode].append(
                {
                    "epoch": epoch,
                    **train_metrics,
                    "cumulative_updates": appearance_updates,
                    "warmup_steps": warmup_steps,
                    "warmup_completed": appearance_updates >= warmup_steps,
                    **epoch_metrics[mode],
                }
            )
        if _metric_score(epoch_metrics["B0"]) > appearance_best_score:
            appearance_best_score = _metric_score(epoch_metrics["B0"])
            appearance_best_epoch = epoch
            appearance_stale = 0
            for mode in ("B0", "B2"):
                atomic_torch_save(
                    appearance_paths[mode],
                    _checkpoint_payload(
                        model=appearance_model,
                        mode=mode,
                        appearance_dim=appearance_dim,
                        head_config=head_config,
                        loss_config=loss_config,
                        epoch=epoch,
                        metrics=epoch_metrics[mode],
                        provenance=provenance,
                        trajectory_id="appearance",
                    ),
                )
        else:
            appearance_stale += 1
        if appearance_stale >= patience and appearance_updates >= warmup_steps:
            break

    appearance_reload = {
        mode: _reload_view(
            appearance_paths[mode],
            mode=mode,
            appearance_dim=appearance_dim,
            head_config=head_config,
            examples=validation_examples,
            cache=cache,
            config=config,
            loss_config=loss_config,
            device=device,
        )
        for mode in ("B0", "B2")
    }
    parent_sha = sha256_file(appearance_paths["B0"])
    parent_checkpoint = torch.load(
        appearance_paths["B0"], map_location=device, weights_only=False
    )
    parent_model = make_identity_head(
        mode="B0", appearance_dim=appearance_dim, head_config=head_config
    ).to(device)
    parent_model.load_state_dict(parent_checkpoint["model_state_dict"], strict=True)
    torch.manual_seed(seed)
    context_model = make_identity_head(
        mode="C0", appearance_dim=appearance_dim, head_config=head_config
    ).to(device)
    partition = warm_start_residual_context(context_model, parent_model)
    _assert_embedding_parity(
        parent_model, context_model, validation_examples, cache, device
    )
    observed_c0 = _evaluate_view(
        context_model,
        "C0",
        validation_examples,
        cache,
        config,
        loss_config,
        device,
    )
    observed_c1 = _evaluate_view(
        context_model,
        "C1",
        validation_examples,
        cache,
        config,
        loss_config,
        device,
    )
    _assert_metric_parity(observed_c0, appearance_reload["B0"])
    _assert_metric_parity(observed_c1, appearance_reload["B2"])
    context_epoch0 = {
        "C0": dict(appearance_reload["B0"]),
        "C1": dict(appearance_reload["B2"]),
    }
    context_paths = {
        mode: destination / f"best_raw_{mode.lower()}.pth" for mode in ("C0", "C1")
    }
    context_best_score = {
        mode: _metric_score(context_epoch0[mode]) for mode in ("C0", "C1")
    }
    context_best_epoch = {"C0": 0, "C1": 0}
    for mode in ("C0", "C1"):
        atomic_torch_save(
            context_paths[mode],
            _checkpoint_payload(
                model=context_model,
                mode=mode,
                appearance_dim=appearance_dim,
                head_config=head_config,
                loss_config=loss_config,
                epoch=0,
                metrics=context_epoch0[mode],
                provenance=provenance,
                trajectory_id="context_adapter",
                parent_checkpoint_sha256=parent_sha,
                frozen_parameter_names=partition["frozen_parameter_names"],
                trainable_parameter_names=partition["trainable_parameter_names"],
            ),
        )
    context_optimizer = make_amuse_optimizer(context_model, config["optimizer"])
    context_history = {"C0": [], "C1": []}
    context_updates = 0
    context_stale = 0
    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(
            context_model,
            context_optimizer,
            train_examples,
            cache,
            device=device,
            pair_batch_size=int(config["pair_batch_size"]),
            seed=seed + epoch,
            loss_config=loss_config,
        )
        context_updates += int(train_metrics["updates"])
        context_optimizer.eval()
        epoch_metrics = {
            mode: _evaluate_view(
                context_model,
                mode,
                validation_examples,
                cache,
                config,
                loss_config,
                device,
            )
            for mode in ("C0", "C1")
        }
        improved = False
        for mode in ("C0", "C1"):
            context_history[mode].append(
                {
                    "epoch": epoch,
                    **train_metrics,
                    "cumulative_updates": context_updates,
                    "warmup_steps": warmup_steps,
                    "warmup_completed": context_updates >= warmup_steps,
                    "context_gate": float(
                        torch.tanh(context_model.context_gate).detach().item()
                    ),
                    **epoch_metrics[mode],
                }
            )
            if _metric_score(epoch_metrics[mode]) > context_best_score[mode]:
                improved = True
                context_best_score[mode] = _metric_score(epoch_metrics[mode])
                context_best_epoch[mode] = epoch
                atomic_torch_save(
                    context_paths[mode],
                    _checkpoint_payload(
                        model=context_model,
                        mode=mode,
                        appearance_dim=appearance_dim,
                        head_config=head_config,
                        loss_config=loss_config,
                        epoch=epoch,
                        metrics=epoch_metrics[mode],
                        provenance=provenance,
                        trajectory_id="context_adapter",
                        parent_checkpoint_sha256=parent_sha,
                        frozen_parameter_names=partition["frozen_parameter_names"],
                        trainable_parameter_names=partition[
                            "trainable_parameter_names"
                        ],
                    ),
                )
        context_stale = 0 if improved else context_stale + 1
        if context_stale >= patience and context_updates >= warmup_steps:
            break

    context_reload = {
        mode: _reload_view(
            context_paths[mode],
            mode=mode,
            appearance_dim=appearance_dim,
            head_config=head_config,
            examples=validation_examples,
            cache=cache,
            config=config,
            loss_config=loss_config,
            device=device,
        )
        for mode in ("C0", "C1")
    }
    mode_reports: dict[str, Any] = {
        "B1": _constant_geometry_report(
            examples=validation_examples,
            cache=cache,
            config=config,
            loss_config=loss_config,
            device=device,
        )
    }
    for mode in ("B0", "B2"):
        mode_reports[mode] = {
            "kind": "shared_appearance_trajectory",
            "trajectory_id": "appearance",
            "initialization_seed": seed,
            "best_epoch": appearance_best_epoch,
            "best_checkpoint": str(appearance_paths[mode]),
            "best_checkpoint_sha256": sha256_file(appearance_paths[mode]),
            "early_stopped": len(appearance_history[mode]) < epochs,
            "epochs_ran": len(appearance_history[mode]),
            "actual_updates": appearance_updates,
            "warmup_steps": warmup_steps,
            "warmup_completed": appearance_updates >= warmup_steps,
            "epoch0_metrics": appearance_epoch0[mode],
            "history": appearance_history[mode],
            "reload_metrics": appearance_reload[mode],
        }
    for mode in ("C0", "C1"):
        checkpoint = torch.load(
            context_paths[mode], map_location="cpu", weights_only=False
        )
        mode_reports[mode] = {
            "kind": "shared_context_adapter_trajectory",
            "trajectory_id": "context_adapter",
            "initialization_seed": seed,
            "parent_checkpoint_sha256": parent_sha,
            "parent_mode": "B0",
            "parent_parity_verified": True,
            "frozen_parameter_names": partition["frozen_parameter_names"],
            "trainable_parameter_names": partition["trainable_parameter_names"],
            "best_epoch": context_best_epoch[mode],
            "best_checkpoint": str(context_paths[mode]),
            "best_checkpoint_sha256": sha256_file(context_paths[mode]),
            "best_context_gate": float(
                torch.tanh(checkpoint["model_state_dict"]["context_gate"])
            ),
            "early_stopped": len(context_history[mode]) < epochs,
            "epochs_ran": len(context_history[mode]),
            "actual_updates": context_updates,
            "warmup_steps": warmup_steps,
            "warmup_completed": context_updates >= warmup_steps,
            "epoch0_metrics": context_epoch0[mode],
            "history": context_history[mode],
            "reload_metrics": context_reload[mode],
        }
    c0_improves = _metric_score(context_reload["C0"]) > _metric_score(
        appearance_reload["B0"]
    )
    c1_improves = _metric_score(context_reload["C1"]) > _metric_score(
        appearance_reload["B2"]
    )
    candidate_mode, candidate_selection_basis = _select_context_candidate(
        context_reload,
        c0_improves=c0_improves,
        c1_improves=c1_improves,
    )
    passed = c0_improves or c1_improves
    return {
        "training_strategy": "residual_context_curriculum_v1",
        "trajectories": {
            "appearance": {
                "actual_updates": appearance_updates,
                "epochs_ran": len(appearance_history["B0"]),
                "selected_parent_epoch": appearance_best_epoch,
                "selected_parent_checkpoint_sha256": parent_sha,
            },
            "context_adapter": {
                "actual_updates": context_updates,
                "epochs_ran": len(context_history["C0"]),
                "parent_checkpoint_sha256": parent_sha,
                **partition,
            },
        },
        "modes": mode_reports,
        "selection": {
            "status": "proxy_candidate_only" if passed else "no_context_gain",
            "candidate_mode": candidate_mode,
            "candidate_checkpoint": mode_reports[candidate_mode]["best_checkpoint"],
            "candidate_selection_basis": candidate_selection_basis,
            "proxy_candidate_passes_baselines": passed,
            "candidate_improves_epoch0": (
                _metric_score(mode_reports[candidate_mode]["reload_metrics"])
                > _metric_score(mode_reports[candidate_mode]["epoch0_metrics"])
            ),
            "candidate_improves_geometry_only": (
                _metric_score(mode_reports[candidate_mode]["reload_metrics"])
                > _metric_score(mode_reports["B1"]["reload_metrics"])
            ),
            "c0_improves_b0": c0_improves,
            "c1_improves_b2": c1_improves,
            "adoption_requires_independently_checked_correspondence": True,
            "advance_to_c2_p1_f1": False,
        },
    }


def train_modes(
    *,
    modes: Sequence[str],
    train_examples: Sequence[PairExample],
    validation_examples: Sequence[PairExample],
    cache: Mapping[str, Any],
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    work_dir: str | Path,
    device: torch.device,
    epochs: int,
    patience: int,
) -> dict[str, Any]:
    """Train sequence descriptors using the explicitly selected strategy."""

    strategy = str(config.get("training_strategy", "independent_v1"))
    arguments = {
        "modes": modes,
        "train_examples": train_examples,
        "validation_examples": validation_examples,
        "cache": cache,
        "config": config,
        "provenance": provenance,
        "work_dir": work_dir,
        "device": device,
        "epochs": epochs,
        "patience": patience,
    }
    if strategy == "independent_v1":
        return _train_modes_independent(**arguments)
    if strategy == "residual_context_curriculum_v1":
        return _train_residual_context_curriculum(**arguments)
    raise ValueError(f"unsupported sequence training strategy: {strategy!r}")


__all__ = [
    "CHECKPOINT_SCHEMA",
    "EXPERIMENT_MODES",
    "FEATURE_CACHE_SCHEMA",
    "PREDICTION_FEATURE_CACHE_SCHEMA",
    "PairExample",
    "atomic_torch_save",
    "atomic_write_json",
    "build_pair_examples",
    "evaluate_model",
    "frame_key",
    "load_feature_cache",
    "make_amuse_optimizer",
    "make_identity_head",
    "sample_pyramid_at_centers",
    "sha256_file",
    "train_epoch",
    "train_modes",
    "uses_geometry_matching",
    "warm_start_residual_context",
]
