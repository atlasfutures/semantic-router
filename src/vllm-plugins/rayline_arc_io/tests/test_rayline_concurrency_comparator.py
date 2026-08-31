# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

comparator = importlib.import_module("rayline_concurrency_comparator")
parity = importlib.import_module("rayline_parity_comparator")


def _buckets(latency: float) -> dict[str, Any]:
    counts = {
        "lt_8192": 12,
        "8192_to_32767": 12,
        "32768_to_131071": 5,
        "gte_131072": 3,
    }
    return {
        name: {
            "scheduled": count,
            "completed": count,
            "failed": 0,
            "selection_latency_seconds": {
                "p50": latency,
                "p95": latency * 2,
                "p99": latency * 3,
            },
        }
        for name, count in counts.items()
    }


def _receipt(arm: str, concurrency: int, *, trace: str = "e" * 64) -> dict[str, Any]:
    throughput = float(concurrency)
    latency = 1.0 / concurrency
    return {
        "schema_version": parity.INPUT_SCHEMA,
        "arm": arm,
        "run_id": f"perf017-c{concurrency}-{arm}",
        "identity": {
            "measurement_scope": comparator.MEASUREMENT_SCOPE,
            "case_count": comparator.MEASURED_CASES,
            "corpus_sha256": "a" * 64,
            "workload_sha256": str(concurrency) * 64,
            "encoder_model": "Qwen/Qwen3.5-0.8B",
            "encoder_revision": "public-revision",
            "tokenizer_sha256": "b" * 64,
            "serializer_version": "mtrouter-token-blocks-v2",
            "policy_artifact_revision": "public-artifact-revision",
            "gpu_class": "NVIDIA H100 80GB",
            "worker_topology_sha256": "c" * 64,
            "placement_profile": "london-policy-us-east-encoder-public-https",
            "warm_state": "warm",
            "seed": 20260730,
        },
        "results": {
            "scheduled": comparator.MEASURED_CASES,
            "completed": comparator.MEASURED_CASES,
            "failed": 0,
            "duration_seconds": comparator.MEASURED_CASES / throughput,
            "throughput_rps": throughput,
            "selection_latency_seconds": {
                "p50": latency,
                "p95": latency * 2,
                "p99": latency * 3,
            },
            "selected_worker_trace_sha256": trace,
            "provider_calls": 0,
            "latency_by_input_tokens": _buckets(latency),
        },
    }


def _cells() -> dict[int, dict[str, dict[str, Any]]]:
    return {
        concurrency: {arm: _receipt(arm, concurrency) for arm in comparator.SWEEP_ARMS}
        for concurrency in comparator.CONCURRENCY_CELLS
    }


def test_identity_matched_sweep_passes_integrity_and_reports_scaling() -> None:
    report = comparator.compare_sweep(_cells())

    assert report["passed"] is True
    assert report["cross_cell_trace_match"] is True
    assert report["arc_vs_remote"]["c4"]["throughput_ratio"] == 1.0
    assert report["scaling"]["rayline_arc"]["c8_vs_c1"]["throughput_ratio"] == float(
        comparator.CONCURRENCY_CELLS[-1]
    )


def test_cross_cell_trace_mismatch_is_visible_failure() -> None:
    cells = _cells()
    cells[8]["rayline_arc"] = _receipt("rayline_arc", 8, trace="f" * 64)

    report = comparator.compare_sweep(cells)

    assert report["passed"] is False
    assert report["integrity"]["c8"]["trace_match"] is False
    assert report["cross_cell_trace_match"] is False
    assert report["selected_worker_trace_sha256"] is None
    assert (
        report["selected_worker_trace_sha256_by_cell"]["c8"]["rayline_arc"] == "f" * 64
    )


def test_fixed_identity_drift_fails_closed() -> None:
    cells = _cells()
    cells[4]["rayline_remote"]["identity"]["gpu_class"] = "NVIDIA L40S"
    cells[4]["rayline_arc"]["identity"]["gpu_class"] = "NVIDIA L40S"

    with pytest.raises(comparator.SweepComparisonError, match="fixed identities"):
        comparator.compare_sweep(cells)


def test_arm_identity_drift_fails_closed() -> None:
    cells = deepcopy(_cells())
    cells[1]["rayline_arc"]["identity"]["workload_sha256"] = "f" * 64

    with pytest.raises(comparator.SweepComparisonError, match="identities differ"):
        comparator.compare_sweep(cells)
