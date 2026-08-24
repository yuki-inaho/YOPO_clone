"""Opt-in, schema-checked throughput measurement for RGB-D training."""

from __future__ import annotations

import json
import hashlib
import math
import time
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import torch
from mmengine.hooks import Hook

from yopo.registry import HOOKS


def _percentile(values: list[float], quantile: float) -> float:
    """Return the nearest-rank percentile for a finite, sorted sample."""
    rank = max(0, math.ceil(quantile * len(values)) - 1)
    return values[rank]


def _summary(records: list[dict[str, float | int]]) -> dict[str, Any]:
    compute = sorted(float(record["compute_seconds"]) for record in records)
    data = sorted(float(record["data_seconds"]) for record in records)
    memory = [int(record["cuda_max_memory_mib"]) for record in records]
    return {
        "measured_iterations": len(records),
        "compute_seconds": {
            "median": median(compute),
            "p95": _percentile(compute, 0.95),
            "min": compute[0],
            "max": compute[-1],
        },
        "data_seconds": {
            "median": median(data),
            "p95": _percentile(data, 0.95),
            "min": data[0],
            "max": data[-1],
        },
        "cuda_max_memory_mib": max(memory),
    }


def _validate_records(records: Iterable[dict[str, Any]], name: str) -> list[dict[str, float | int]]:
    validated: list[dict[str, float | int]] = []
    for expected_iteration, record in enumerate(records):
        required = {"iteration", "compute_seconds", "data_seconds", "cuda_max_memory_mib"}
        missing = required.difference(record)
        if missing:
            raise ValueError(f"{name} benchmark record missing fields: {sorted(missing)}")
        if record["iteration"] != expected_iteration:
            raise ValueError(
                f"{name} benchmark iteration must be contiguous from zero: "
                f"expected={expected_iteration}, actual={record['iteration']}"
            )
        compute = float(record["compute_seconds"])
        data = float(record["data_seconds"])
        memory = int(record["cuda_max_memory_mib"])
        if not math.isfinite(compute) or compute <= 0:
            raise ValueError(f"{name} benchmark compute_seconds must be finite and positive")
        if not math.isfinite(data) or data < 0:
            raise ValueError(f"{name} benchmark data_seconds must be finite and non-negative")
        if memory < 0:
            raise ValueError(f"{name} benchmark cuda_max_memory_mib must be non-negative")
        validated.append(
            {
                "iteration": expected_iteration,
                "compute_seconds": compute,
                "data_seconds": data,
                "cuda_max_memory_mib": memory,
            }
        )
    if not validated:
        raise ValueError(f"{name} benchmark has no records")
    return validated


def build_throughput_report(
    *,
    train_records: Iterable[dict[str, Any]],
    val_records: Iterable[dict[str, Any]],
    warmup_iters: int,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Validate timing samples and produce a portable benchmark report."""
    if warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative")
    config_hash = provenance.get("config_sha256")
    if (
        not provenance.get("config")
        or not isinstance(config_hash, str)
        or len(config_hash) != 64
        or not all(char in "0123456789abcdef" for char in config_hash)
        or not isinstance(provenance.get("seed"), int)
        or int(provenance.get("batch_size", 0)) <= 0
    ):
        raise ValueError(
            "benchmark provenance requires config, config_sha256, integer seed, "
            "and positive batch_size"
        )

    train = _validate_records(train_records, "train")
    val = _validate_records(val_records, "val")
    measured_train = train[warmup_iters:]
    if not measured_train:
        raise ValueError("benchmark has no train records after warmup")

    train_summary = _summary(measured_train)
    val_summary = _summary(val)
    return {
        "schema_version": 1,
        "provenance": provenance,
        "warmup_iterations": warmup_iters,
        "train": {"total_iterations": len(train), **train_summary},
        "val": val_summary,
        "cuda_max_memory_mib": max(
            train_summary["cuda_max_memory_mib"], val_summary["cuda_max_memory_mib"]
        ),
    }


@HOOKS.register_module()
class ThroughputBenchmarkHook(Hook):
    """Collect synchronized timing only when an explicit benchmark config enables it."""

    # Run after the NORMAL IterTimerHook (for data_time) but before the
    # BELOW_NORMAL LoggerHook, which consumes and resets CUDA peak stats.
    priority = "NORMAL"

    def __init__(self, output_filename: str, warmup_iters: int = 2) -> None:
        if warmup_iters < 0:
            raise ValueError("warmup_iters must be non-negative")
        self.output_filename = output_filename
        self.warmup_iters = warmup_iters
        self._train_records: list[dict[str, float | int]] = []
        self._val_records: list[dict[str, float | int]] = []
        self._train_start: float | None = None
        self._val_start: float | None = None
        self._train_data_seconds = 0.0
        self._val_data_seconds = 0.0

    @staticmethod
    def _cuda_synchronize() -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("ThroughputBenchmarkHook requires CUDA")
        torch.cuda.synchronize()

    @staticmethod
    def _message_hub_seconds(runner: Any, key: str) -> float:
        scalar = runner.message_hub.get_scalar(key)
        if scalar is None:
            raise RuntimeError(f"ThroughputBenchmarkHook requires MessageHub scalar {key}")
        value = float(scalar.current())
        if not math.isfinite(value) or value < 0:
            raise FloatingPointError(f"non-finite benchmark scalar {key}={value}")
        return value

    @staticmethod
    def _cuda_peak_mib() -> int:
        return int(math.ceil(torch.cuda.max_memory_allocated() / 1024**2))

    def before_train_epoch(self, runner: Any) -> None:
        self._train_records.clear()
        self._val_records.clear()
        self._cuda_synchronize()
        torch.cuda.reset_peak_memory_stats()

    def before_train_iter(self, runner: Any, batch_idx: int, **_: Any) -> None:
        self._train_data_seconds = self._message_hub_seconds(runner, "train/data_time")
        self._cuda_synchronize()
        self._train_start = time.perf_counter()

    def after_train_iter(self, runner: Any, batch_idx: int, **_: Any) -> None:
        if self._train_start is None:
            raise RuntimeError("ThroughputBenchmarkHook train start timestamp is missing")
        self._cuda_synchronize()
        self._train_records.append(
            {
                "iteration": batch_idx,
                "compute_seconds": time.perf_counter() - self._train_start,
                "data_seconds": self._train_data_seconds,
                "cuda_max_memory_mib": self._cuda_peak_mib(),
            }
        )
        self._train_start = None

    def before_val_iter(self, runner: Any, batch_idx: int, **_: Any) -> None:
        self._val_data_seconds = self._message_hub_seconds(runner, "val/data_time")
        self._cuda_synchronize()
        self._val_start = time.perf_counter()

    def after_val_iter(self, runner: Any, batch_idx: int, **_: Any) -> None:
        if self._val_start is None:
            raise RuntimeError("ThroughputBenchmarkHook validation start timestamp is missing")
        self._cuda_synchronize()
        self._val_records.append(
            {
                "iteration": batch_idx,
                "compute_seconds": time.perf_counter() - self._val_start,
                "data_seconds": self._val_data_seconds,
                "cuda_max_memory_mib": self._cuda_peak_mib(),
            }
        )
        self._val_start = None

    def after_val_epoch(self, runner: Any, **_: Any) -> None:
        config_name = str(getattr(runner.cfg, "filename", "<in-memory-config>"))
        randomness = runner.cfg.get("randomness")
        if not isinstance(randomness, dict) or not isinstance(randomness.get("seed"), int):
            raise RuntimeError("ThroughputBenchmarkHook requires an explicit integer randomness.seed")
        batch_size = int(runner.train_dataloader.batch_size)
        report = build_throughput_report(
            train_records=self._train_records,
            val_records=self._val_records,
            warmup_iters=self.warmup_iters,
            provenance={
                "config": config_name,
                "config_sha256": hashlib.sha256(
                    runner.cfg.pretty_text.encode("utf-8")
                ).hexdigest(),
                "seed": randomness["seed"],
                "batch_size": batch_size,
            },
        )
        output_path = Path(self.output_filename)
        if not output_path.is_absolute():
            output_path = Path(runner.work_dir) / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
