import pytest

from yopo.engine.hooks.throughput_benchmark_hook import (
    ThroughputBenchmarkHook,
    build_throughput_report,
)


def _record(index: int, *, compute: float, data: float, memory: int) -> dict:
    return {
        "iteration": index,
        "compute_seconds": compute,
        "data_seconds": data,
        "cuda_max_memory_mib": memory,
    }


def test_throughput_report_requires_finite_warm_records():
    records = [
        _record(0, compute=0.9, data=0.1, memory=1024),
        _record(1, compute=1.0, data=0.2, memory=1030),
        _record(2, compute=1.2, data=0.3, memory=1040),
    ]

    report = build_throughput_report(
        train_records=records,
        val_records=[_record(0, compute=0.2, data=0.02, memory=1040)],
        warmup_iters=1,
        provenance={
            "config": "benchmark.py",
            "config_sha256": "a" * 64,
            "seed": 3407,
            "batch_size": 26,
        },
    )

    assert report["schema_version"] == 1
    assert report["provenance"] == {
        "config": "benchmark.py",
        "config_sha256": "a" * 64,
        "seed": 3407,
        "batch_size": 26,
    }
    assert report["train"]["measured_iterations"] == 2
    assert report["train"]["compute_seconds"]["median"] == pytest.approx(1.1)
    assert report["train"]["data_seconds"]["p95"] == pytest.approx(0.3)
    assert report["val"]["measured_iterations"] == 1
    assert report["cuda_max_memory_mib"] == 1040


def test_throughput_hook_reads_peak_before_logger_resets_it():
    assert ThroughputBenchmarkHook.priority == "NORMAL"


def test_throughput_report_requires_reproducible_provenance():
    with pytest.raises(ValueError, match="config_sha256"):
        build_throughput_report(
            train_records=[_record(0, compute=1.0, data=0.1, memory=1)],
            val_records=[_record(0, compute=0.2, data=0.02, memory=1)],
            warmup_iters=0,
            provenance={"config": "benchmark.py", "batch_size": 26},
        )


@pytest.mark.parametrize(
    ("records", "warmup_iters"),
    [
        ([], 0),
        ([_record(0, compute=float("nan"), data=0.1, memory=1)], 0),
        ([_record(0, compute=1.0, data=0.1, memory=1)], 1),
        ([_record(1, compute=1.0, data=0.1, memory=1)], 0),
    ],
)
def test_throughput_report_rejects_empty_nonfinite_or_invalid_iteration_records(
    records, warmup_iters
):
    with pytest.raises(ValueError):
        build_throughput_report(
            train_records=records,
            val_records=[_record(0, compute=0.2, data=0.02, memory=1)],
            warmup_iters=warmup_iters,
            provenance={
                "config": "benchmark.py",
                "config_sha256": "a" * 64,
                "seed": 3407,
                "batch_size": 26,
            },
        )
