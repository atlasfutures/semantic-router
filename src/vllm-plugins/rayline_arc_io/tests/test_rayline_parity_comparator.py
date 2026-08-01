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

comparator = importlib.import_module("rayline_parity_comparator")

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
TRACE_HASH = "e" * 64


def _receipt(
    arm: str,
    *,
    throughput: float = 10.0,
    p50: float = 0.50,
    p95: float = 0.80,
    p99: float = 1.20,
    completed: int = 128,
    trace_hash: str = TRACE_HASH,
) -> dict[str, Any]:
    scheduled = 128
    return {
        "schema_version": comparator.INPUT_SCHEMA,
        "arm": arm,
        "run_id": f"public-{arm}",
        "identity": {
            "measurement_scope": "router_only",
            "case_count": scheduled,
            "corpus_sha256": HASH_A,
            "workload_sha256": HASH_B,
            "encoder_model": "Qwen/Qwen3.5-0.8B",
            "encoder_revision": "public-revision",
            "tokenizer_sha256": HASH_C,
            "serializer_version": "mtrouter-token-blocks-v2",
            "policy_artifact_revision": "public-artifact-revision",
            "gpu_class": "NVIDIA H100 80GB",
            "worker_topology_sha256": HASH_D,
            "placement_profile": "controlled-same-region-private-transport",
            "warm_state": "warm",
            "seed": 20260730,
        },
        "results": {
            "scheduled": scheduled,
            "completed": completed,
            "failed": scheduled - completed,
            "duration_seconds": completed / throughput,
            "throughput_rps": throughput,
            "selection_latency_seconds": {
                "p50": p50,
                "p95": p95,
                "p99": p99,
            },
            "selected_worker_trace_sha256": trace_hash,
            "provider_calls": 0,
        },
    }


def _passing_packet() -> list[dict[str, Any]]:
    return [_receipt(arm) for arm in comparator.ARMS]


def test_identity_matched_packet_passes_all_gates() -> None:
    report = comparator.compare_receipts(_passing_packet())

    assert report["schema_version"] == comparator.REPORT_SCHEMA
    assert report["passed"] is True
    assert report["selection_trace_match"] is True
    assert report["baseline_parity_passed"] is True
    assert all(result["passed"] for result in report["arm_gates"].values())


def test_identity_mismatch_is_not_a_comparison_result() -> None:
    packet = _passing_packet()
    packet[2]["identity"]["gpu_class"] = "NVIDIA L40S"

    with pytest.raises(comparator.ReceiptError, match="comparison identities differ"):
        comparator.compare_receipts(packet)


def test_perf014_shape_passes_capacity_but_fails_latency() -> None:
    packet = _passing_packet()
    packet[1] = _receipt(
        "rayline_remote",
        throughput=8.753049,
        p50=0.949589,
        p95=1.286,
        p99=1.308,
    )

    report = comparator.compare_receipts(packet)
    remote = report["arm_gates"]["rayline_remote"]

    assert remote["gates"]["throughput"] is True
    assert remote["gates"]["p95_latency"] is False
    assert remote["gates"]["p99_latency"] is True
    assert report["passed"] is False


def test_worker_trace_mismatch_fails_without_hiding_metrics() -> None:
    packet = _passing_packet()
    packet[2] = _receipt("rayline_arc", trace_hash="f" * 64)

    report = comparator.compare_receipts(packet)

    assert report["selection_trace_match"] is False
    assert report["arm_gates"]["rayline_arc"]["passed"] is True
    assert report["passed"] is False


def test_baseline_relative_gate_is_independent_from_absolute_slo() -> None:
    packet = _passing_packet()
    packet[1] = _receipt(
        "rayline_remote",
        throughput=8.5,
        p50=0.5,
        p95=0.9,
        p99=1.2,
    )

    report = comparator.compare_receipts(packet)
    relative = report["relative"]["rayline_remote_vs_modal"]

    assert report["arm_gates"]["rayline_remote"]["passed"] is True
    assert relative["gates"]["throughput_ratio"] is False
    assert relative["gates"]["p95_ratio"] is False
    assert report["baseline_parity_passed"] is False
    assert report["passed"] is False


def test_receipt_rejects_unregistered_fields() -> None:
    receipt = deepcopy(_receipt("rayline_arc"))
    receipt["results"]["prompt"] = "must never enter a receipt"

    with pytest.raises(comparator.ReceiptError, match="results keys differ"):
        comparator.validate_receipt(receipt)
