#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Aggregate-only v3 reporter for the AGT018 cache comparison."""

from __future__ import annotations

from collections import Counter
from typing import Any

import openrouter_kv_cache_reporting as legacy
from modal_session_canary import ENGINE_BUILD_ID
from openrouter_agentic_workload import PROVIDER_NAMES, WORKERS
from openrouter_kv_cache_successor_contract import (
    ARTIFACT_REVISION,
    MAXIMUM_EXTERNAL_ATTEMPTS,
    MAXIMUM_LOGICAL_PROVIDER_REQUESTS,
    NATIVE_APP_NAME,
    REMOTE_APP_NAME,
    REPORT_SCHEMA_VERSION,
    RUN_ID,
)
from openrouter_kv_cache_successor_workload import (
    ENCODER_ARTIFACT_ID,
    EPISODES,
    EXPECTED_REQUESTS_PER_DEPLOYMENT,
    EXPECTED_SELECTION_TRACES,
    MINIMUM_TOP_TWO_SCORE_GAP,
    OFFLINE_REPORT_SCHEMA_VERSION,
    SCHEMA_VERSION as WORKLOAD_SCHEMA_VERSION,
    SEQUENCE_IDS,
    MODES,
    STEPS,
)
from openrouter_modal_native_benchmark import _decision_cost
from openrouter_provider_preflight_contract import validate_report as validate_preflight


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
        or len(offline_observations) != 9
        or float(offline.get("minimum_top_two_score_gap") or 0)
        < MINIMUM_TOP_TWO_SCORE_GAP
        or offline.get("prompt_text_emitted") is not False
        or offline.get("raw_embeddings_emitted") is not False
        or offline.get("provider_calls") != 0
    ):
        raise RuntimeError("AGT018 offline encoder coverage evidence diverged")
    if (
        not isinstance(parity, dict)
        or parity.get("status") != "passed"
        or parity.get("states") != 9
        or len(parity.get("observations") or []) != 9
        or float(parity.get("minimum_top_two_score_gap") or 0)
        < MINIMUM_TOP_TWO_SCORE_GAP
        or parity.get("provider_calls") != 0
        or parity.get("sessions_closed") != 3
        or parity.get("prompt_text_emitted") is not False
        or parity.get("raw_embeddings_emitted") is not False
    ):
        raise RuntimeError("AGT018 remote encoder parity evidence diverged")
    expected = _expected_trace()
    if _trace(offline) != expected or _trace(parity) != expected:
        raise RuntimeError("AGT018 native and remote encoder traces diverged")


def _enrich_native(client: dict[str, Any], decisions: list[dict[str, Any]]) -> None:
    by_id = {str(row.get("request_id") or ""): row for row in decisions}
    if len(by_id) != EXPECTED_REQUESTS_PER_DEPLOYMENT or "" in by_id:
        raise RuntimeError("AGT018 native decision identity set diverged")
    for result in client["results"]:
        row = by_id.get(str(result.get("request_id") or ""))
        if row is None or row.get("error"):
            raise RuntimeError("AGT018 native client/decision join failed")
        worker = str(result["selected_worker"])
        if (
            row.get("selected_worker") != worker
            or row.get("served_model") != WORKERS[worker]
            or row.get("served_provider") not in PROVIDER_NAMES[worker]
        ):
            raise RuntimeError("AGT018 native serving identity diverged")
        features = row.get("features")
        if not isinstance(features, dict):
            raise RuntimeError("AGT018 native decision omitted encoder features")
        serialized = int(features.get("serialized_tokens") or 0)
        cached = int(features.get("cached_prefix_tokens") or 0)
        if serialized <= 0 or not 0 <= cached <= serialized:
            raise RuntimeError("AGT018 native cached-prefix accounting diverged")
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
            raise RuntimeError("AGT018 native cache action contract diverged")


def _validate_client(client: dict[str, Any], deployment: str) -> None:
    if (
        client.get("run_id") != RUN_ID
        or client.get("deployment") != deployment
        or len(client.get("results", [])) != EXPECTED_REQUESTS_PER_DEPLOYMENT
        or client.get("workload", {}).get("requests")
        != EXPECTED_REQUESTS_PER_DEPLOYMENT
    ):
        raise RuntimeError("AGT018 client identity or request envelope diverged")
    seen: set[tuple[str, str, int, int]] = set()
    for result in client["results"]:
        key = _cell_key(result)
        if key in seen:
            raise RuntimeError("AGT018 client repeated a measurement cell")
        seen.add(key)
        expected = EXPECTED_SELECTION_TRACES[key[0]][key[3]]
        if (
            result.get("expected_worker") != expected
            or result.get("selected_worker") != expected
        ):
            raise RuntimeError("AGT018 natural selection trace diverged")
        attempts = int(result.get("external_attempts") or 0)
        if not 1 <= attempts <= 2:
            raise RuntimeError("AGT018 measured request attempt envelope diverged")
    expected_cells = {
        (sequence_id, mode, episode, step)
        for sequence_id in SEQUENCE_IDS
        for mode in MODES
        for episode in range(EPISODES)
        for step in range(STEPS)
    }
    if seen != expected_cells:
        raise RuntimeError("AGT018 measured cell set diverged")


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
        raise RuntimeError("AGT018 retained/replay cell envelope diverged")
    for pair in grouped.values():
        if (
            len(pair) != 2
            or {row["mode"] for row in pair} != {"retained", "replay"}
            or len({int(row["completion_tokens"]) for row in pair}) != 1
        ):
            raise RuntimeError("AGT018 retained/replay completion parity diverged")


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
        raise RuntimeError("AGT018 native and remote selections diverged")
    deployments = {
        "native_modal": _deployment_report(native),
        "remote_vllm": _deployment_report(remote),
    }
    for deployment in deployments.values():
        steady = deployment["steady_state"]
        if steady["comparison"]["retained_token_work_saved_fraction"] <= 0:
            raise RuntimeError("AGT018 retained session did not reduce token work")
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
        "encoder_build_id": ENGINE_BUILD_ID,
        "encoder_gdn_prefill_backend": "flashinfer",
        "encoder_ephemeral": True,
        "artifact_revision": ARTIFACT_REVISION,
    }
    if (
        native.get("app_name") != NATIVE_APP_NAME
        or native.get("gpu") != "H100"
        or native.get("checkpoint", {}).get("artifact_id") != ARTIFACT_REVISION
    ):
        raise RuntimeError("AGT018 native deployment identity diverged")
    for field, expected in expected_remote.items():
        if remote.get(field) != expected:
            raise RuntimeError(f"AGT018 remote deployment identity diverged: {field}")
    return {
        "native_h100": True,
        "remote_h100": True,
        "remote_flashinfer_runtime": True,
        "remote_ephemeral_app": True,
        "artifact_revision_matched": True,
    }


def _completion_contract(
    native: dict[str, Any], remote: dict[str, Any]
) -> dict[str, Any]:
    values = {
        name: sorted({int(row["completion_tokens"]) for row in client["results"]})
        for name, client in (("native_modal", native), ("remote_vllm", remote))
    }
    requested = {
        int(native["workload"]["max_completion_tokens"]),
        int(remote["workload"]["max_completion_tokens"]),
    }
    matched = values["native_modal"] == values["remote_vllm"]
    within_cap = (
        requested == {24}
        and max((*values["native_modal"], *values["remote_vllm"]), default=0) <= 24
    )
    return {
        "requested_max_tokens": 24,
        "native_observed_completion_token_values": values["native_modal"],
        "remote_observed_completion_token_values": values["remote_vllm"],
        "within_deployment_retained_replay_parity": True,
        "cross_deployment_matched": matched,
        "within_requested_cap": within_cap,
        "cross_deployment_e2e_comparable": matched and within_cap,
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
            raise RuntimeError("AGT018 provider preflight run identity diverged")
    _enrich_native(native, decisions)
    legacy._normalize_remote(remote)
    for client in (native, remote):
        _validate_pairing(client)
    deployments, measured_attempts = _validate_deployments(native, remote)
    identity = _validate_identity(native_deployment, remote_deployment)
    completion = _completion_contract(native, remote)
    preflight_attempts = sum(
        int(report["external_attempts"])
        for report in (native_preflight, remote_preflight)
    )
    actual_attempts = measured_attempts + preflight_attempts
    provider_spend = native_key_usage + remote_key_usage
    budget_passed = provider_spend <= float(
        budget_receipt["maximum_provider_spend_usd"]
    )
    attempts_passed = (
        actual_attempts <= MAXIMUM_EXTERNAL_ATTEMPTS
        and MAXIMUM_LOGICAL_PROVIDER_REQUESTS == 78
    )
    cleanup_passed = cleanup_receipt.get("passed") is True
    gates = {
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
        "matched_completion_policy": completion["cross_deployment_e2e_comparable"],
        "request_attempt_and_cost_envelopes": attempts_passed and budget_passed,
        "privacy_and_cleanup": cleanup_passed,
    }
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
        "completion_contract": completion,
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
        "limitations": [
            "two episodes per history are a diagnostic, not an SLO",
            "requests are serial and do not establish concurrency saturation throughput",
            "provider latency remains externally variable despite paired ordering",
            "native observed first token follows buffered provider completion and is not provider TTFT",
        ],
    }
