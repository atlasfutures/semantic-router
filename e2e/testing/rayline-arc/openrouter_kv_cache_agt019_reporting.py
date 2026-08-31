#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Aggregate-only v4 reporter for the AGT019 matched-pair cache comparison.

The workload, expected selection traces, and offline/parity encoder gates are
the AGT018 ones verbatim. The one behavioural difference is comparability:
AGT018's whole-set `matched_completion_policy` is replaced by the preregistered
matched-pair lanes in `openrouter_kv_cache_matched_pair`, so pinned
multi-provider fallthrough labels a worker inadmissible instead of failing the
run, while the per-request completion cap stays unconditional.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import openrouter_kv_cache_matched_pair as matched_pair
import openrouter_kv_cache_reporting as legacy
from openrouter_agentic_workload import PROVIDER_NAMES, WORKERS
from openrouter_kv_cache_agt019_contract import (
    ARTIFACT_REVISION,
    MAX_COMPLETION_TOKENS,
    MAXIMUM_EXTERNAL_ATTEMPTS,
    MAXIMUM_LOGICAL_PROVIDER_REQUESTS,
    NATIVE_APP_NAME,
    REMOTE_APP_NAME,
    REMOTE_ENGINE_BUILD_ID,
    REMOTE_GDN_PREFILL_BACKEND,
    REPORT_SCHEMA_VERSION,
    RUN_ID,
)
from openrouter_kv_cache_successor_workload import (
    ENCODER_ARTIFACT_ID,
    EPISODES,
    EXPECTED_REQUESTS_PER_DEPLOYMENT,
    EXPECTED_SELECTION_TRACES,
    MINIMUM_TOP_TWO_SCORE_GAP,
    MODES,
    OFFLINE_REPORT_SCHEMA_VERSION,
    SEQUENCE_IDS,
    STEPS,
)
from openrouter_kv_cache_successor_workload import (
    SCHEMA_VERSION as WORKLOAD_SCHEMA_VERSION,
)
from openrouter_modal_native_benchmark import _decision_cost
from openrouter_provider_preflight_contract import validate_report as validate_preflight

EXPECTED_ENCODER_STATES = len(SEQUENCE_IDS) * STEPS
EXPECTED_PROBE_SESSIONS = len(SEQUENCE_IDS)
MAXIMUM_ATTEMPTS_PER_MEASURED_REQUEST = 2
EXPECTED_MODES_PER_CELL = len(MODES)
LIMITATIONS = (
    "two episodes per history are a diagnostic, not an SLO",
    "requests are serial and do not establish concurrency saturation throughput",
    "provider latency remains externally variable despite paired ordering",
    "native observed first token follows buffered provider completion and is not provider TTFT",
    "cross-deployment E2E and first-token claims are restricted to the fully matched pair lane; coverage is reported per worker",
)


def _cell_key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row["sequence_id"]),
        str(row["mode"]),
        int(row["episode"]),
        int(row["step"]),
    )


def _trace(report: dict[str, Any]) -> dict[tuple[str, int], str]:
    return {
        (str(row["sequence_id"]), int(row["step"])): str(row["selected_worker"])
        for row in report["observations"]
    }


def _expected_trace() -> dict[tuple[str, int], str]:
    return {
        (sequence_id, step): worker
        for sequence_id, workers in EXPECTED_SELECTION_TRACES.items()
        for step, worker in enumerate(workers)
    }


def _validate_encoder_gates(offline: dict[str, Any], remote: dict[str, Any]) -> None:
    parity = remote.get("pre_provider_remote_encoder_parity")
    offline_observations = offline.get("observations")
    if (
        offline.get("schema_version") != OFFLINE_REPORT_SCHEMA_VERSION
        or offline.get("status") != "passed"
        or offline.get("workload_schema_version") != WORKLOAD_SCHEMA_VERSION
        or offline.get("encoder", {}).get("artifact_id") != ENCODER_ARTIFACT_ID
        or offline.get("all_workers_covered") is not True
        or not isinstance(offline_observations, list)
        or len(offline_observations) != EXPECTED_ENCODER_STATES
        or float(offline.get("minimum_top_two_score_gap") or 0)
        < MINIMUM_TOP_TWO_SCORE_GAP
        or offline.get("prompt_text_emitted") is not False
        or offline.get("raw_embeddings_emitted") is not False
        or offline.get("provider_calls") != 0
    ):
        raise RuntimeError("AGT019 offline encoder coverage evidence diverged")
    if (
        not isinstance(parity, dict)
        or parity.get("status") != "passed"
        or parity.get("states") != EXPECTED_ENCODER_STATES
        or len(parity.get("observations") or []) != EXPECTED_ENCODER_STATES
        or float(parity.get("minimum_top_two_score_gap") or 0)
        < MINIMUM_TOP_TWO_SCORE_GAP
        or parity.get("provider_calls") != 0
        or parity.get("sessions_closed") != EXPECTED_PROBE_SESSIONS
        or parity.get("prompt_text_emitted") is not False
        or parity.get("raw_embeddings_emitted") is not False
    ):
        raise RuntimeError("AGT019 remote encoder parity evidence diverged")
    expected = _expected_trace()
    if _trace(offline) != expected or _trace(parity) != expected:
        raise RuntimeError("AGT019 native and remote encoder traces diverged")


def _enrich_native(client: dict[str, Any], decisions: list[dict[str, Any]]) -> None:
    by_id = {str(row.get("request_id") or ""): row for row in decisions}
    if len(by_id) != EXPECTED_REQUESTS_PER_DEPLOYMENT or "" in by_id:
        raise RuntimeError("AGT019 native decision identity set diverged")
    for result in client["results"]:
        row = by_id.get(str(result.get("request_id") or ""))
        if row is None or row.get("error"):
            raise RuntimeError("AGT019 native client/decision join failed")
        worker = str(result["selected_worker"])
        if (
            row.get("selected_worker") != worker
            or row.get("served_model") != WORKERS[worker]
            or row.get("served_provider") not in PROVIDER_NAMES[worker]
        ):
            raise RuntimeError("AGT019 native serving identity diverged")
        features = row.get("features")
        if not isinstance(features, dict):
            raise RuntimeError("AGT019 native decision omitted encoder features")
        serialized = int(features.get("serialized_tokens") or 0)
        cached = int(features.get("cached_prefix_tokens") or 0)
        if serialized <= 0 or not 0 <= cached <= serialized:
            raise RuntimeError("AGT019 native cached-prefix accounting diverged")
        attempts = legacy._native_attempts(row)
        result.update(
            {
                "session_action": str(features.get("encode_mode") or ""),
                "encoder_serialized_tokens": serialized,
                "encoder_cached_prefix_tokens": cached,
                "encoder_token_work": serialized - cached,
                "router_seconds": float(row.get("decision_latency_ms") or 0) / 1000,
                "encoder_seconds": float(features.get("embedding_latency_ms") or 0)
                / 1000,
                "non_encoder_router_seconds": float(features.get("q_latency_ms") or 0)
                / 1000,
                "provider": str(row.get("served_provider") or ""),
                "cost_usd": _decision_cost(row),
                "external_attempts": attempts,
                "prompt_tokens": int(row.get("input_tokens") or 0),
                "completion_tokens": int(row.get("output_tokens") or 0),
            }
        )
        expected_action = (
            "delta"
            if result["mode"] == "retained" and int(result["step"]) > 0
            else "prefill"
        )
        if result["session_action"] != expected_action:
            raise RuntimeError("AGT019 native cache action contract diverged")


def _validate_client(client: dict[str, Any], deployment: str) -> None:
    if (
        client.get("run_id") != RUN_ID
        or client.get("deployment") != deployment
        or len(client.get("results", [])) != EXPECTED_REQUESTS_PER_DEPLOYMENT
        or client.get("workload", {}).get("requests")
        != EXPECTED_REQUESTS_PER_DEPLOYMENT
    ):
        raise RuntimeError("AGT019 client identity or request envelope diverged")
    seen: set[tuple[str, str, int, int]] = set()
    for result in client["results"]:
        key = _cell_key(result)
        if key in seen:
            raise RuntimeError("AGT019 client repeated a measurement cell")
        seen.add(key)
        expected = EXPECTED_SELECTION_TRACES[key[0]][key[3]]
        if (
            result.get("expected_worker") != expected
            or result.get("selected_worker") != expected
        ):
            raise RuntimeError("AGT019 natural selection trace diverged")
        attempts = int(result.get("external_attempts") or 0)
        if not 1 <= attempts <= MAXIMUM_ATTEMPTS_PER_MEASURED_REQUEST:
            raise RuntimeError("AGT019 measured request attempt envelope diverged")
    expected_cells = {
        (sequence_id, mode, episode, step)
        for sequence_id in SEQUENCE_IDS
        for mode in MODES
        for episode in range(EPISODES)
        for step in range(STEPS)
    }
    if seen != expected_cells:
        raise RuntimeError("AGT019 measured cell set diverged")


def _validate_pairing(client: dict[str, Any]) -> None:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for result in client["results"]:
        key = (
            str(result["sequence_id"]),
            int(result["episode"]),
            int(result["step"]),
        )
        grouped.setdefault(key, []).append(result)
    if len(grouped) != EXPECTED_REQUESTS_PER_DEPLOYMENT // 2:
        raise RuntimeError("AGT019 retained/replay cell envelope diverged")
    for pair in grouped.values():
        if (
            len(pair) != EXPECTED_MODES_PER_CELL
            or {row["mode"] for row in pair} != {"retained", "replay"}
            or len({int(row["completion_tokens"]) for row in pair}) != 1
        ):
            raise RuntimeError("AGT019 retained/replay completion parity diverged")


def _subset_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    return legacy._result_set_report(results)


def _deployment_report(client: dict[str, Any]) -> dict[str, Any]:
    report = legacy._deployment_report(client)
    results = client["results"]
    completion_tokens = sum(int(row["completion_tokens"]) for row in results)
    report.update(
        {
            "output_token_throughput_tps": completion_tokens
            / float(client["elapsed_seconds"]),
            "natural_worker_mix": dict(
                Counter(str(row["selected_worker"]) for row in results)
            ),
            "per_sequence": {
                sequence_id: _subset_report(
                    [row for row in results if row["sequence_id"] == sequence_id]
                )
                for sequence_id in SEQUENCE_IDS
            },
            "per_model": {
                WORKERS[worker]: _subset_report(
                    [row for row in results if row["selected_worker"] == worker]
                )
                for worker in WORKERS
            },
        }
    )
    return report


def _validate_deployments(
    native: dict[str, Any], remote: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    if {_cell_key(row): row["selected_worker"] for row in native["results"]} != {
        _cell_key(row): row["selected_worker"] for row in remote["results"]
    }:
        raise RuntimeError("AGT019 native and remote selections diverged")
    deployments = {
        "native_modal": _deployment_report(native),
        "remote_vllm": _deployment_report(remote),
    }
    for deployment in deployments.values():
        steady = deployment["steady_state"]
        if steady["comparison"]["retained_token_work_saved_fraction"] <= 0:
            raise RuntimeError("AGT019 retained session did not reduce token work")
    attempts = sum(
        int(row["external_attempts"])
        for client in (native, remote)
        for row in client["results"]
    )
    return deployments, attempts


def _validate_identity(
    native: dict[str, Any], remote: dict[str, Any]
) -> dict[str, bool]:
    expected_remote = {
        "encoder_app_name": REMOTE_APP_NAME,
        "encoder_gpu": "H100",
        "encoder_build_id": REMOTE_ENGINE_BUILD_ID,
        "encoder_gdn_prefill_backend": REMOTE_GDN_PREFILL_BACKEND,
        "encoder_ephemeral": True,
        "artifact_revision": ARTIFACT_REVISION,
    }
    if (
        native.get("app_name") != NATIVE_APP_NAME
        or native.get("gpu") != "H100"
        or native.get("checkpoint", {}).get("artifact_id") != ARTIFACT_REVISION
    ):
        raise RuntimeError("AGT019 native deployment identity diverged")
    for field, expected in expected_remote.items():
        if remote.get(field) != expected:
            raise RuntimeError(f"AGT019 remote deployment identity diverged: {field}")
    return {
        "native_h100": True,
        "remote_h100": True,
        "remote_flashinfer_runtime": True,
        "remote_ephemeral_app": True,
        "artifact_revision_matched": True,
    }


def _within_requested_cap(native: dict[str, Any], remote: dict[str, Any]) -> bool:
    """True when both arms asked for, and stayed under, the per-request cap."""

    requested = {
        int(native["workload"]["max_completion_tokens"]),
        int(remote["workload"]["max_completion_tokens"]),
    }
    observed = [
        int(row["completion_tokens"])
        for client in (native, remote)
        for row in client["results"]
    ]
    return (
        requested == {MAX_COMPLETION_TOKENS}
        and max(observed, default=0) <= MAX_COMPLETION_TOKENS
    )


def _matched_pair_section(
    native: dict[str, Any], remote: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Join both arms into lanes and decide the AGT019 comparability gate."""

    pairs = matched_pair.pair_cells(native["results"], remote["results"])
    comparability = matched_pair.comparability_report(pairs)
    policy_passed = matched_pair.policy_gate(
        comparability, EXPECTED_REQUESTS_PER_DEPLOYMENT
    )
    within_cap = _within_requested_cap(native, remote)
    section = {
        **comparability,
        "requested_max_completion_tokens": MAX_COMPLETION_TOKENS,
        "within_requested_cap": within_cap,
        "policy_gate_passed": policy_passed,
    }
    return section, policy_passed and within_cap


def _acceptance_gates(
    *,
    deployments: dict[str, Any],
    identity: dict[str, bool],
    matched_pair_passed: bool,
    attempts_passed: bool,
    budget_passed: bool,
    cleanup_passed: bool,
) -> dict[str, bool]:
    return {
        "exact_source_and_artifact_identity": all(identity.values()),
        "provider_preflight_all_three_models": True,
        "native_offline_three_worker_coverage": True,
        "remote_encoder_trace_matches_offline_trace": True,
        "native_remote_selected_worker_trace_parity": True,
        "native_retained_token_saving": (
            deployments["native_modal"]["steady_state"]["comparison"][
                "retained_token_work_saved_fraction"
            ]
            > 0
        ),
        "remote_retained_token_saving": (
            deployments["remote_vllm"]["steady_state"]["comparison"][
                "retained_token_work_saved_fraction"
            ]
            > 0
        ),
        "matched_pair_comparability_policy": matched_pair_passed,
        "request_attempt_and_cost_envelopes": attempts_passed and budget_passed,
        "privacy_and_cleanup": cleanup_passed,
    }


def build_report(
    *,
    native: dict[str, Any],
    decisions: list[dict[str, Any]],
    remote: dict[str, Any],
    native_deployment: dict[str, Any],
    remote_deployment: dict[str, Any],
    native_preflight: dict[str, Any],
    remote_preflight: dict[str, Any],
    offline_coverage: dict[str, Any],
    cleanup_receipt: dict[str, Any],
    native_key_usage: float,
    remote_key_usage: float,
    budget_receipt: dict[str, Any],
) -> dict[str, Any]:
    _validate_client(native, "native_modal")
    _validate_client(remote, "remote_vllm")
    _validate_encoder_gates(offline_coverage, remote)
    for preflight in (native_preflight, remote_preflight):
        validate_preflight(preflight)
        if preflight.get("run_id") != RUN_ID:
            raise RuntimeError("AGT019 provider preflight run identity diverged")
    _enrich_native(native, decisions)
    legacy._normalize_remote(remote)
    for client in (native, remote):
        _validate_pairing(client)
    comparability, matched_pair_passed = _matched_pair_section(native, remote)
    deployments, measured_attempts = _validate_deployments(native, remote)
    identity = _validate_identity(native_deployment, remote_deployment)
    preflight_attempts = sum(
        int(report["external_attempts"])
        for report in (native_preflight, remote_preflight)
    )
    actual_attempts = measured_attempts + preflight_attempts
    provider_spend = native_key_usage + remote_key_usage
    budget_passed = provider_spend <= float(
        budget_receipt["maximum_provider_spend_usd"]
    )
    attempts_passed = actual_attempts <= MAXIMUM_EXTERNAL_ATTEMPTS
    cleanup_passed = cleanup_receipt.get("passed") is True
    gates = _acceptance_gates(
        deployments=deployments,
        identity=identity,
        matched_pair_passed=matched_pair_passed,
        attempts_passed=attempts_passed,
        budget_passed=budget_passed,
        cleanup_passed=cleanup_passed,
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "status": "passed" if all(gates.values()) else "failed_acceptance",
        "workload": native["workload"],
        "models": WORKERS,
        "deployment_identity": {
            "native_modal": native_deployment,
            "remote_vllm": remote_deployment,
        },
        "deployment_attestation": identity,
        "encoder_coverage": {
            "native_offline": offline_coverage,
            "remote_vllm": remote["pre_provider_remote_encoder_parity"],
        },
        "provider_preflight": {
            "native_modal": native_preflight,
            "remote_vllm": remote_preflight,
        },
        "acceptance": {"passed": all(gates.values()), "gates": gates},
        "deployments": deployments,
        "cross_deployment": legacy._cross_deployment_report(deployments),
        "matched_pair_comparability": comparability,
        "actual_logical_provider_requests": MAXIMUM_LOGICAL_PROVIDER_REQUESTS,
        "actual_external_attempts": actual_attempts,
        "maximum_external_attempts": MAXIMUM_EXTERNAL_ATTEMPTS,
        "openrouter_key_usage_usd": {
            "native_modal": native_key_usage,
            "remote_vllm": remote_key_usage,
            "total": provider_spend,
        },
        "cost_accounting": {
            **budget_receipt,
            "actual_provider_spend_usd": provider_spend,
            "actual_provider_within_envelope": budget_passed,
        },
        "cleanup": cleanup_receipt,
        "automatic_prefix_cache_enabled": False,
        "cache_contracts": {
            "native_modal": "Pathfinder-owned chunk-grid KV session",
            "remote_vllm": (
                "explicit retained AsyncPoolingSession with causal-mean accumulator"
            ),
        },
        "release_qualification_1000_executed": False,
        "limitations": list(LIMITATIONS),
    }
