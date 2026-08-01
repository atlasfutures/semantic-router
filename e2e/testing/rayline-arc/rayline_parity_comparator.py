#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Compare identity-matched Modal, Remote, and ARC benchmark receipts."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

INPUT_SCHEMA = "rayline.vllm.three-arm-input.v1"
REPORT_SCHEMA = "rayline.vllm.three-arm-comparison.v1"
ARMS = ("modal_inprocess", "rayline_remote", "rayline_arc")

COMPLETION_FLOOR = 0.999
THROUGHPUT_FLOOR_RPS = 8.0
P95_CEILING_SECONDS = 1.0
P99_CEILING_SECONDS = 2.0
BASELINE_THROUGHPUT_RATIO_FLOOR = 0.90
BASELINE_P95_RATIO_CEILING = 1.10

IDENTITY_FIELDS = (
    "measurement_scope",
    "case_count",
    "corpus_sha256",
    "workload_sha256",
    "encoder_model",
    "encoder_revision",
    "tokenizer_sha256",
    "serializer_version",
    "policy_artifact_revision",
    "gpu_class",
    "worker_topology_sha256",
    "placement_profile",
    "warm_state",
    "seed",
)
RESULT_FIELDS = (
    "scheduled",
    "completed",
    "failed",
    "duration_seconds",
    "throughput_rps",
    "selection_latency_seconds",
    "selected_worker_trace_sha256",
    "provider_calls",
)
LATENCY_FIELDS = ("p50", "p95", "p99")
SHA256_LENGTH = 64


class ReceiptError(ValueError):
    """A receipt cannot participate in the frozen comparison."""


def _require_exact_keys(
    value: Mapping[str, Any], expected: tuple[str, ...], label: str
) -> None:
    missing = sorted(set(expected) - set(value))
    extra = sorted(set(value) - set(expected))
    if missing or extra:
        raise ReceiptError(f"{label} keys differ: missing={missing}, extra={extra}")


def _require_sha256(value: object, label: str) -> str:
    text = str(value)
    if len(text) != SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ReceiptError(f"{label} must be 64 lowercase hexadecimal characters")
    return text


def _finite_number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReceiptError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (positive and number <= 0):
        qualifier = "positive and finite" if positive else "finite and non-negative"
        raise ReceiptError(f"{label} must be {qualifier}")
    return number


def _non_negative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReceiptError(f"{label} must be a non-negative integer")
    return value


def validate_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_keys(
        receipt,
        ("schema_version", "arm", "run_id", "identity", "results"),
        "receipt",
    )
    if receipt["schema_version"] != INPUT_SCHEMA:
        raise ReceiptError("unsupported receipt schema")
    arm = str(receipt["arm"])
    if arm not in ARMS:
        raise ReceiptError(f"unsupported arm: {arm}")
    run_id = str(receipt["run_id"])
    if not run_id or len(run_id) > 128:
        raise ReceiptError("run_id must contain 1 to 128 characters")

    identity_raw = receipt["identity"]
    if not isinstance(identity_raw, Mapping):
        raise ReceiptError("identity must be an object")
    _require_exact_keys(identity_raw, IDENTITY_FIELDS, "identity")
    identity = dict(identity_raw)
    case_count = _non_negative_integer(identity["case_count"], "identity.case_count")
    if case_count == 0:
        raise ReceiptError("identity.case_count must be positive")
    seed = _non_negative_integer(identity["seed"], "identity.seed")
    identity["seed"] = seed
    for field in (
        "corpus_sha256",
        "workload_sha256",
        "tokenizer_sha256",
        "worker_topology_sha256",
    ):
        identity[field] = _require_sha256(identity[field], f"identity.{field}")
    for field in set(IDENTITY_FIELDS) - {
        "case_count",
        "seed",
        "corpus_sha256",
        "workload_sha256",
        "tokenizer_sha256",
        "worker_topology_sha256",
    }:
        text = str(identity[field])
        if not text:
            raise ReceiptError(f"identity.{field} must be non-empty")
        identity[field] = text

    results_raw = receipt["results"]
    if not isinstance(results_raw, Mapping):
        raise ReceiptError("results must be an object")
    _require_exact_keys(results_raw, RESULT_FIELDS, "results")
    results = dict(results_raw)
    for field in ("scheduled", "completed", "failed", "provider_calls"):
        results[field] = _non_negative_integer(results[field], f"results.{field}")
    if results["scheduled"] != case_count:
        raise ReceiptError("scheduled requests must equal identity.case_count")
    if results["completed"] + results["failed"] != results["scheduled"]:
        raise ReceiptError(
            "completed plus failed requests must equal scheduled requests"
        )
    results["duration_seconds"] = _finite_number(
        results["duration_seconds"], "results.duration_seconds", positive=True
    )
    results["throughput_rps"] = _finite_number(
        results["throughput_rps"], "results.throughput_rps", positive=True
    )
    observed_rps = results["completed"] / results["duration_seconds"]
    if not math.isclose(results["throughput_rps"], observed_rps, rel_tol=1e-6):
        raise ReceiptError("throughput_rps does not match completed / duration_seconds")

    latency_raw = results["selection_latency_seconds"]
    if not isinstance(latency_raw, Mapping):
        raise ReceiptError("results.selection_latency_seconds must be an object")
    _require_exact_keys(latency_raw, LATENCY_FIELDS, "selection_latency_seconds")
    latency = {
        field: _finite_number(latency_raw[field], f"selection_latency_seconds.{field}")
        for field in LATENCY_FIELDS
    }
    if not latency["p50"] <= latency["p95"] <= latency["p99"]:
        raise ReceiptError("selection latency percentiles must be monotonic")
    results["selection_latency_seconds"] = latency
    results["selected_worker_trace_sha256"] = _require_sha256(
        results["selected_worker_trace_sha256"],
        "results.selected_worker_trace_sha256",
    )
    return {
        "schema_version": INPUT_SCHEMA,
        "arm": arm,
        "run_id": run_id,
        "identity": identity,
        "results": results,
    }


def _matching_identity(receipts: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = receipts[ARMS[0]]["identity"]
    mismatches: dict[str, dict[str, object]] = {}
    for arm in ARMS[1:]:
        candidate = receipts[arm]["identity"]
        arm_mismatches = {
            field: {"baseline": baseline[field], "candidate": candidate[field]}
            for field in IDENTITY_FIELDS
            if candidate[field] != baseline[field]
        }
        if arm_mismatches:
            mismatches[arm] = arm_mismatches
    if mismatches:
        raise ReceiptError(f"comparison identities differ: {mismatches}")
    return dict(baseline)


def _arm_gates(receipt: Mapping[str, Any]) -> dict[str, Any]:
    results = receipt["results"]
    completion_ratio = results["completed"] / results["scheduled"]
    p95 = results["selection_latency_seconds"]["p95"]
    p99 = results["selection_latency_seconds"]["p99"]
    gates = {
        "completion": completion_ratio >= COMPLETION_FLOOR,
        "throughput": results["throughput_rps"] >= THROUGHPUT_FLOOR_RPS,
        "p95_latency": p95 <= P95_CEILING_SECONDS,
        "p99_latency": p99 <= P99_CEILING_SECONDS,
    }
    return {
        "completion_ratio": completion_ratio,
        "throughput_rps": results["throughput_rps"],
        "p95_seconds": p95,
        "p99_seconds": p99,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _pairwise(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    candidate_results = candidate["results"]
    baseline_results = baseline["results"]
    throughput_ratio = (
        candidate_results["throughput_rps"] / baseline_results["throughput_rps"]
    )
    baseline_p95 = baseline_results["selection_latency_seconds"]["p95"]
    candidate_p95 = candidate_results["selection_latency_seconds"]["p95"]
    if baseline_p95 == 0:
        p95_ratio = 1.0 if candidate_p95 == 0 else math.inf
    else:
        p95_ratio = candidate_p95 / baseline_p95
    gates = {
        "throughput_ratio": throughput_ratio >= BASELINE_THROUGHPUT_RATIO_FLOOR,
        "p95_ratio": p95_ratio <= BASELINE_P95_RATIO_CEILING,
    }
    return {
        "throughput_ratio": throughput_ratio,
        "p95_ratio": p95_ratio,
        "p95_delta_seconds": (candidate_p95 - baseline_p95),
        "gates": gates,
        "passed": all(gates.values()),
    }


def compare_receipts(raw_receipts: list[Mapping[str, Any]]) -> dict[str, Any]:
    receipts: dict[str, dict[str, Any]] = {}
    for raw_receipt in raw_receipts:
        receipt = validate_receipt(raw_receipt)
        arm = receipt["arm"]
        if arm in receipts:
            raise ReceiptError(f"duplicate receipt for arm: {arm}")
        receipts[arm] = receipt
    missing = [arm for arm in ARMS if arm not in receipts]
    if missing:
        raise ReceiptError(f"missing receipts for arms: {missing}")

    identity = _matching_identity(receipts)
    trace_digests = {
        arm: receipts[arm]["results"]["selected_worker_trace_sha256"] for arm in ARMS
    }
    selection_trace_match = len(set(trace_digests.values())) == 1
    arm_gates = {arm: _arm_gates(receipts[arm]) for arm in ARMS}
    modal = receipts["modal_inprocess"]
    relative = {
        "rayline_remote_vs_modal": _pairwise(receipts["rayline_remote"], modal),
        "rayline_arc_vs_modal": _pairwise(receipts["rayline_arc"], modal),
        "rayline_arc_vs_remote": _pairwise(
            receipts["rayline_arc"], receipts["rayline_remote"]
        ),
    }
    baseline_parity_passed = all(
        relative[name]["passed"]
        for name in ("rayline_remote_vs_modal", "rayline_arc_vs_modal")
    )
    passed = (
        selection_trace_match
        and all(result["passed"] for result in arm_gates.values())
        and baseline_parity_passed
    )
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "passed" if passed else "failed",
        "identity": identity,
        "thresholds": {
            "completion_floor": COMPLETION_FLOOR,
            "throughput_floor_rps": THROUGHPUT_FLOOR_RPS,
            "p95_ceiling_seconds": P95_CEILING_SECONDS,
            "p99_ceiling_seconds": P99_CEILING_SECONDS,
            "baseline_throughput_ratio_floor": BASELINE_THROUGHPUT_RATIO_FLOOR,
            "baseline_p95_ratio_ceiling": BASELINE_P95_RATIO_CEILING,
        },
        "run_ids": {arm: receipts[arm]["run_id"] for arm in ARMS},
        "selected_worker_trace_sha256": trace_digests,
        "selection_trace_match": selection_trace_match,
        "arm_gates": arm_gates,
        "relative": relative,
        "baseline_parity_passed": baseline_parity_passed,
        "passed": passed,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipts", nargs=3, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        raw_receipts = [json.loads(path.read_text()) for path in args.receipts]
        report = compare_receipts(raw_receipts)
    except (OSError, json.JSONDecodeError, ReceiptError) as error:
        raise SystemExit(f"invalid comparison packet: {error}") from error
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
