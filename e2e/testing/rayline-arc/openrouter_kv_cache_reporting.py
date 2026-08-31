#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Aggregate-only reporter for the matched AGT017 cache comparison."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from modal_fullstack_canary import _summary
from openrouter_agentic_workload import PROVIDER_NAMES, WORKERS
from openrouter_kv_cache_matched_contract import (
    FLASHINFER_APP_NAME,
    FLASHINFER_BACKEND,
    FLASHINFER_ENGINE_BUILD_ID,
    MAX_COMPLETION_TOKENS,
    matched_budget_receipt,
)
from openrouter_modal_native_benchmark import _decision_cost, read_decisions

EXPECTED_REQUESTS_PER_DEPLOYMENT = 12
EXPECTED_MODES_PER_STATE = 2
AGT016_REFERENCE_STEADY_RETAINED_ROUTER_MEAN_SECONDS = 1.2457
MAX_FLASHINFER_TO_REFERENCE_ROUTER_RATIO = 0.80
MIN_NATIVE_STEADY_RETAINED_TOKEN_SAVING = 0.40
MIN_FLASHINFER_STEADY_RETAINED_TOKEN_SAVING = 0.55


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-client")
    parser.add_argument("--native-decisions")
    parser.add_argument("--native-deployment")
    parser.add_argument("--native-failure")
    parser.add_argument("--native-key-usage", required=True)
    parser.add_argument("--remote-client", required=True)
    parser.add_argument("--remote-deployment", required=True)
    parser.add_argument("--remote-key-usage", required=True)
    parser.add_argument("--cleanup-receipt")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _mean_summary(values: list[float]) -> dict[str, float]:
    return {**_summary(values), "mean_seconds": math.fsum(values) / len(values)}


def _native_attempts(row: dict[str, Any]) -> int:
    transport = row.get("transport")
    attempts = transport.get("attempts") if isinstance(transport, dict) else None
    return len(attempts) if isinstance(attempts, list) else 1


def enrich_native(client: dict[str, Any], decisions: list[dict[str, Any]]) -> None:
    by_id = {str(row.get("request_id") or ""): row for row in decisions}
    if len(by_id) != EXPECTED_REQUESTS_PER_DEPLOYMENT or "" in by_id:
        raise RuntimeError("native decision identity set diverged")
    for result in client["results"]:
        row = by_id.get(result["request_id"])
        if row is None or row.get("error"):
            raise RuntimeError("native client/decision join failed")
        selected_worker = str(result["selected_worker"])
        if row.get("selected_worker") != selected_worker:
            raise RuntimeError("native client/decision selection diverged")
        if str(row.get("served_model") or "") != WORKERS[selected_worker]:
            raise RuntimeError("native decision served the wrong OpenRouter model")
        if str(row.get("served_provider") or "") not in PROVIDER_NAMES[selected_worker]:
            raise RuntimeError(
                "native decision used a provider outside its frozen order"
            )
        features = row.get("features")
        if not isinstance(features, dict):
            raise RuntimeError("native decision omitted encoder features")
        serialized = int(features.get("serialized_tokens") or 0)
        cached = int(features.get("cached_prefix_tokens") or 0)
        if not 0 <= cached <= serialized:
            raise RuntimeError("native cached-prefix accounting diverged")
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
                "external_attempts": _native_attempts(row),
                "prompt_tokens": int(row.get("input_tokens") or 0),
                "completion_tokens": int(row.get("output_tokens") or 0),
            }
        )
    expected = {
        (mode, step): ("delta" if mode == "retained" and step > 0 else "prefill")
        for mode in ("retained", "replay")
        for step in range(3)
    }
    for result in client["results"]:
        if result["session_action"] != expected[(result["mode"], result["step"])]:
            raise RuntimeError("native cache action contract diverged")


def _normalize_remote(client: dict[str, Any]) -> None:
    for result in client["results"]:
        decomposition = result["router_stage"]["mean_decomposition"]
        coordinator = result["encoder_stage"]["coordinator"]
        result.update(
            {
                "encoder_token_work": int(coordinator["backend_appended_tokens"]),
                "router_seconds": float(decomposition["router_seconds"]),
                "encoder_seconds": float(decomposition["encoder_seconds"]),
                "non_encoder_router_seconds": max(
                    0.0,
                    float(decomposition["router_non_encoder_seconds"]),
                ),
            }
        )


def _mode_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = sum(int(result["external_attempts"]) for result in results)
    return {
        "requests": len(results),
        "end_to_end_latency": _mean_summary(
            [float(result["total_seconds"]) for result in results]
        ),
        "observed_first_token": _mean_summary(
            [float(result["first_token_seconds"]) for result in results]
        ),
        "router_latency": _mean_summary(
            [float(result["router_seconds"]) for result in results]
        ),
        "encoder_latency": _mean_summary(
            [float(result["encoder_seconds"]) for result in results]
        ),
        "encoder_token_work": sum(
            int(result["encoder_token_work"]) for result in results
        ),
        "session_actions": dict(
            Counter(result["session_action"] for result in results)
        ),
        "external_attempts": attempts,
        "retries": attempts - len(results),
        "provider_cost_usd": math.fsum(float(result["cost_usd"]) for result in results),
        "completion_tokens": sum(
            int(result["completion_tokens"]) for result in results
        ),
        "completion_token_values": sorted(
            {int(result["completion_tokens"]) for result in results}
        ),
        "selected_workers": dict(
            Counter(result["selected_worker"] for result in results)
        ),
        "providers": sorted({str(result.get("provider") or "") for result in results}),
    }


def _step_reports(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports = []
    for step in range(3):
        row: dict[str, Any] = {"step": step}
        for mode in ("retained", "replay"):
            cell = [
                result
                for result in results
                if result["step"] == step and result["mode"] == mode
            ]
            row[mode] = {
                "requests": len(cell),
                "encoder_token_work": sum(
                    int(result["encoder_token_work"]) for result in cell
                ),
                "router_mean_seconds": math.fsum(
                    float(result["router_seconds"]) for result in cell
                )
                / len(cell),
                "encoder_mean_seconds": math.fsum(
                    float(result["encoder_seconds"]) for result in cell
                )
                / len(cell),
                "e2e_mean_seconds": math.fsum(
                    float(result["total_seconds"]) for result in cell
                )
                / len(cell),
            }
        reports.append(row)
    return reports


def _result_set_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    retained = [result for result in results if result["mode"] == "retained"]
    replay = [result for result in results if result["mode"] == "replay"]
    retained_report = _mode_report(retained)
    replay_report = _mode_report(replay)
    return {
        "paths": {"retained": retained_report, "replay": replay_report},
        "comparison": {
            "retained_to_replay_token_work_ratio": (
                retained_report["encoder_token_work"]
                / replay_report["encoder_token_work"]
            ),
            "retained_token_work_saved_fraction": 1
            - retained_report["encoder_token_work"]
            / replay_report["encoder_token_work"],
            "retained_to_replay_router_mean_ratio": (
                retained_report["router_latency"]["mean_seconds"]
                / replay_report["router_latency"]["mean_seconds"]
            ),
            "retained_to_replay_encoder_mean_ratio": (
                retained_report["encoder_latency"]["mean_seconds"]
                / replay_report["encoder_latency"]["mean_seconds"]
            ),
            "retained_to_replay_e2e_mean_ratio": (
                retained_report["end_to_end_latency"]["mean_seconds"]
                / replay_report["end_to_end_latency"]["mean_seconds"]
            ),
        },
        "steps": _step_reports(results),
    }


def _deployment_report(client: dict[str, Any]) -> dict[str, Any]:
    results = client["results"]
    aggregate = _result_set_report(results)
    steady_state_episode = max(int(result["episode"]) for result in results)
    steady_state = _result_set_report(
        [result for result in results if int(result["episode"]) == steady_state_episode]
    )
    return {
        **aggregate,
        "serial_request_throughput_rps": len(results)
        / float(client["elapsed_seconds"]),
        "whole_run_elapsed_seconds": float(client["elapsed_seconds"]),
        "steady_state": {
            "episode": steady_state_episode,
            **steady_state,
        },
    }


def _selection_parity(native: dict[str, Any], remote: dict[str, Any]) -> None:
    def index(client: dict[str, Any]) -> dict[tuple[str, int, int], str]:
        return {
            (result["mode"], result["episode"], result["step"]): result[
                "selected_worker"
            ]
            for result in client["results"]
        }

    if index(native) != index(remote):
        raise RuntimeError("native Modal and remote vLLM selections diverged")


def _validate_within_deployment_completion_parity(client: dict[str, Any]) -> None:
    for episode in range(2):
        for step in range(3):
            pair = [
                int(result["completion_tokens"])
                for result in client["results"]
                if result["episode"] == episode and result["step"] == step
            ]
            if len(pair) != EXPECTED_MODES_PER_STATE or len(set(pair)) != 1:
                raise RuntimeError("retained/replay completion-token parity diverged")


def _validated_deployments(
    native: dict[str, Any], remote: dict[str, Any]
) -> tuple[dict[str, Any], int]:
    deployments = {
        "native_modal": _deployment_report(native),
        "remote_vllm": _deployment_report(remote),
    }
    for deployment in deployments.values():
        for scope in (deployment, deployment["steady_state"]):
            retained = scope["paths"]["retained"]
            replay = scope["paths"]["replay"]
            if retained["encoder_token_work"] >= replay["encoder_token_work"]:
                raise RuntimeError("retained session did not reduce encoder token work")
            if retained["retries"] or replay["retries"]:
                raise RuntimeError("KV comparison observed an external retry")
    actual_attempts = sum(
        int(result["external_attempts"])
        for client in (native, remote)
        for result in client["results"]
    )
    expected_attempts = EXPECTED_REQUESTS_PER_DEPLOYMENT * EXPECTED_MODES_PER_STATE
    if actual_attempts != expected_attempts:
        raise RuntimeError("KV comparison external attempt envelope diverged")
    return deployments, actual_attempts


def _completion_contract(
    native: dict[str, Any], remote: dict[str, Any]
) -> dict[str, Any]:
    native_values = sorted(
        {int(result["completion_tokens"]) for result in native["results"]}
    )
    remote_values = sorted(
        {int(result["completion_tokens"]) for result in remote["results"]}
    )
    matched = native_values == remote_values
    within_requested_cap = (
        int(native["workload"]["max_completion_tokens"])
        == int(remote["workload"]["max_completion_tokens"])
        == MAX_COMPLETION_TOKENS
        and max((*native_values, *remote_values), default=0) <= MAX_COMPLETION_TOKENS
    )
    return {
        "requested_max_tokens": int(native["workload"]["max_completion_tokens"]),
        "native_observed_completion_token_values": native_values,
        "remote_observed_completion_token_values": remote_values,
        "within_deployment_retained_replay_parity": True,
        "cross_deployment_matched": matched,
        "within_requested_cap": within_requested_cap,
        "cross_deployment_e2e_comparable": matched and within_requested_cap,
        "deviation": (
            ""
            if matched
            else ("native and remote observed completion-token policies diverged")
        ),
    }


def _cross_deployment_report(deployments: dict[str, Any]) -> dict[str, Any]:
    native = deployments["native_modal"]
    remote = deployments["remote_vllm"]
    native_steady = native["steady_state"]["paths"]["retained"]
    remote_steady = remote["steady_state"]["paths"]["retained"]
    return {
        "selection_parity": True,
        "primary_comparison_basis": "steady_state_episode_1",
        "native_to_vllm_retained_router_mean_ratio": (
            native["paths"]["retained"]["router_latency"]["mean_seconds"]
            / remote["paths"]["retained"]["router_latency"]["mean_seconds"]
        ),
        "native_to_vllm_steady_state_retained_router_mean_ratio": (
            native_steady["router_latency"]["mean_seconds"]
            / remote_steady["router_latency"]["mean_seconds"]
        ),
        "vllm_to_native_steady_state_retained_router_mean_ratio": (
            remote_steady["router_latency"]["mean_seconds"]
            / native_steady["router_latency"]["mean_seconds"]
        ),
        "vllm_to_native_steady_state_retained_e2e_mean_ratio": (
            remote_steady["end_to_end_latency"]["mean_seconds"]
            / native_steady["end_to_end_latency"]["mean_seconds"]
        ),
        "vllm_to_native_steady_state_retained_observed_first_token_mean_ratio": (
            remote_steady["observed_first_token"]["mean_seconds"]
            / native_steady["observed_first_token"]["mean_seconds"]
        ),
        "vllm_to_native_serial_request_throughput_ratio": (
            remote["serial_request_throughput_rps"]
            / native["serial_request_throughput_rps"]
        ),
    }


def _validate_deployment_identity(
    native_deployment: dict[str, Any], remote_deployment: dict[str, Any]
) -> dict[str, Any]:
    if native_deployment.get("gpu") != "H100":
        raise RuntimeError("native deployment did not attest H100")
    _validate_remote_deployment_identity(remote_deployment)
    return {
        "native_h100": True,
        "remote_h100": True,
        "remote_flashinfer_runtime": True,
        "remote_ephemeral_app": True,
    }


def _validate_remote_deployment_identity(
    remote_deployment: dict[str, Any],
) -> None:
    expected_remote = {
        "encoder_app_name": FLASHINFER_APP_NAME,
        "encoder_gpu": "H100",
        "encoder_build_id": FLASHINFER_ENGINE_BUILD_ID,
        "encoder_gdn_prefill_backend": FLASHINFER_BACKEND,
        "encoder_ephemeral": True,
    }
    for field, expected in expected_remote.items():
        if remote_deployment.get(field) != expected:
            raise RuntimeError(f"remote deployment identity diverged: {field}")


def _validate_single_deployment(deployment: dict[str, Any]) -> None:
    for scope in (deployment, deployment["steady_state"]):
        retained = scope["paths"]["retained"]
        replay = scope["paths"]["replay"]
        if retained["encoder_token_work"] >= replay["encoder_token_work"]:
            raise RuntimeError("retained session did not reduce encoder token work")
        if retained["retries"] or replay["retries"]:
            raise RuntimeError("KV comparison observed an external retry")


def _acceptance_report(
    deployments: dict[str, Any], completion_contract: dict[str, Any]
) -> dict[str, Any]:
    native = deployments["native_modal"]["steady_state"]
    remote = deployments["remote_vllm"]["steady_state"]
    remote_router_mean = remote["paths"]["retained"]["router_latency"]["mean_seconds"]
    remote_to_reference = (
        remote_router_mean / AGT016_REFERENCE_STEADY_RETAINED_ROUTER_MEAN_SECONDS
    )
    native_saving = native["comparison"]["retained_token_work_saved_fraction"]
    remote_saving = remote["comparison"]["retained_token_work_saved_fraction"]
    gates = {
        "completion_policy_matched": completion_contract[
            "cross_deployment_e2e_comparable"
        ],
        "native_retained_token_saving": (
            native_saving >= MIN_NATIVE_STEADY_RETAINED_TOKEN_SAVING
        ),
        "flashinfer_retained_token_saving": (
            remote_saving >= MIN_FLASHINFER_STEADY_RETAINED_TOKEN_SAVING
        ),
        "flashinfer_router_improves_on_agt016_reference": (
            remote_to_reference <= MAX_FLASHINFER_TO_REFERENCE_ROUTER_RATIO
        ),
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "thresholds": {
            "native_minimum_steady_retained_token_saving_fraction": (
                MIN_NATIVE_STEADY_RETAINED_TOKEN_SAVING
            ),
            "flashinfer_minimum_steady_retained_token_saving_fraction": (
                MIN_FLASHINFER_STEADY_RETAINED_TOKEN_SAVING
            ),
            "agt016_reference_steady_retained_router_mean_seconds": (
                AGT016_REFERENCE_STEADY_RETAINED_ROUTER_MEAN_SECONDS
            ),
            "maximum_flashinfer_to_reference_router_ratio": (
                MAX_FLASHINFER_TO_REFERENCE_ROUTER_RATIO
            ),
        },
        "observed": {
            "native_steady_retained_token_saving_fraction": native_saving,
            "flashinfer_steady_retained_token_saving_fraction": remote_saving,
            "flashinfer_steady_retained_router_mean_seconds": remote_router_mean,
            "flashinfer_to_agt016_reference_router_ratio": remote_to_reference,
        },
    }


def build_report(
    *,
    native: dict[str, Any],
    decisions: list[dict[str, Any]],
    remote: dict[str, Any],
    native_deployment: dict[str, Any],
    remote_deployment: dict[str, Any],
    native_key_usage: float,
    remote_key_usage: float,
) -> dict[str, Any]:
    if native["run_id"] != remote["run_id"]:
        raise RuntimeError("KV comparison run IDs diverged")
    if any(
        len(client["results"]) != EXPECTED_REQUESTS_PER_DEPLOYMENT
        for client in (native, remote)
    ):
        raise RuntimeError("KV comparison request count diverged")
    enrich_native(native, decisions)
    _normalize_remote(remote)
    _selection_parity(native, remote)
    for client in (native, remote):
        _validate_within_deployment_completion_parity(client)
    deployments, actual_attempts = _validated_deployments(native, remote)
    completion_contract = _completion_contract(native, remote)
    deployment_attestation = _validate_deployment_identity(
        native_deployment, remote_deployment
    )
    acceptance = _acceptance_report(deployments, completion_contract)
    budget = matched_budget_receipt()
    provider_usage = native_key_usage + remote_key_usage
    return {
        "schema_version": "rayline.openrouter-kv-cache-comparison.v2",
        "run_id": native["run_id"],
        "status": "passed" if acceptance["passed"] else "failed_acceptance",
        "workload": native["workload"],
        "models": WORKERS,
        "deployment_identity": {
            "native_modal": native_deployment,
            "remote_vllm": remote_deployment,
        },
        "deployment_attestation": deployment_attestation,
        "acceptance": acceptance,
        "deployments": deployments,
        "cross_deployment": _cross_deployment_report(deployments),
        "actual_provider_requests": EXPECTED_REQUESTS_PER_DEPLOYMENT * 2,
        "actual_external_attempts": actual_attempts,
        "completion_contract": completion_contract,
        "openrouter_key_usage_usd": {
            "native_modal": native_key_usage,
            "remote_vllm": remote_key_usage,
            "total": native_key_usage + remote_key_usage,
        },
        "cost_accounting": {
            **budget,
            "actual_provider_spend_usd": provider_usage,
            "actual_provider_within_envelope": (
                provider_usage <= budget["maximum_provider_spend_usd"]
            ),
        },
        "automatic_prefix_cache_enabled": False,
        "cache_contracts": {
            "native_modal": "Pathfinder-owned chunk-grid KV session",
            "remote_vllm": "explicit retained AsyncPoolingSession with causal-mean accumulator",
        },
        "release_qualification_1000_executed": False,
        "limitations": [
            "two episodes and three history states are a diagnostic, not an SLO",
            "native Modal buffers provider completion, so its observed first token is not provider TTFT",
            "native and vLLM cache effects are normalized against replay within each deployment",
            "episode 0 includes first-shape compilation; episode 1 is the steady-state comparison",
            "the serial request rate includes provider generation and is not a concurrency throughput SLO",
            "provider latency remains externally variable despite interleaved retained/replay requests",
            "native observed first token is emitted after Pathfinder buffers the provider response and is not provider TTFT",
        ],
    }


def _incomplete_remote_analysis(
    remote: dict[str, Any], remote_deployment: dict[str, Any]
) -> tuple[dict[str, Any], int, list[int], bool]:
    if len(remote.get("results", [])) != EXPECTED_REQUESTS_PER_DEPLOYMENT:
        raise RuntimeError("remote KV comparison request count diverged")
    _normalize_remote(remote)
    _validate_within_deployment_completion_parity(remote)
    remote_report = _deployment_report(remote)
    _validate_single_deployment(remote_report)
    remote_attempts = sum(
        int(result["external_attempts"]) for result in remote["results"]
    )
    if remote_attempts != EXPECTED_REQUESTS_PER_DEPLOYMENT:
        raise RuntimeError("remote KV comparison external attempt envelope diverged")
    _validate_remote_deployment_identity(remote_deployment)

    remote_completion_values = sorted(
        {int(result["completion_tokens"]) for result in remote["results"]}
    )
    within_requested_cap = (
        int(remote["workload"]["max_completion_tokens"]) == MAX_COMPLETION_TOKENS
        and max(remote_completion_values, default=0) <= MAX_COMPLETION_TOKENS
    )
    return (
        remote_report,
        remote_attempts,
        remote_completion_values,
        within_requested_cap,
    )


def _incomplete_acceptance(remote_report: dict[str, Any]) -> dict[str, Any]:
    remote_steady = remote_report["steady_state"]
    remote_saving = remote_steady["comparison"]["retained_token_work_saved_fraction"]
    remote_router_mean = remote_steady["paths"]["retained"]["router_latency"][
        "mean_seconds"
    ]
    remote_to_reference = (
        remote_router_mean / AGT016_REFERENCE_STEADY_RETAINED_ROUTER_MEAN_SECONDS
    )
    return {
        "passed": False,
        "blocker": "native arm returned HTTP 429 before durable client evidence",
        "gates": {
            "native_arm_complete": False,
            "completion_policy_matched": False,
            "selection_parity": False,
            "native_retained_token_saving": False,
            "flashinfer_retained_token_saving": (
                remote_saving >= MIN_FLASHINFER_STEADY_RETAINED_TOKEN_SAVING
            ),
            "flashinfer_router_improves_on_agt016_reference": (
                remote_to_reference <= MAX_FLASHINFER_TO_REFERENCE_ROUTER_RATIO
            ),
        },
        "not_evaluable": [
            "native retained token saving",
            "cross-deployment selection parity",
            "cross-deployment completion parity",
            "native-to-vLLM latency and throughput ratios",
        ],
        "observed": {
            "flashinfer_steady_retained_token_saving_fraction": remote_saving,
            "flashinfer_steady_retained_router_mean_seconds": remote_router_mean,
            "flashinfer_to_agt016_reference_router_ratio": remote_to_reference,
        },
    }


def _incomplete_counts(
    remote_attempts: int,
    remote_completion_values: list[int],
    within_requested_cap: bool,
) -> dict[str, Any]:
    return {
        "completed_provider_requests": {
            "native_modal": None,
            "remote_vllm": EXPECTED_REQUESTS_PER_DEPLOYMENT,
            "confirmed_minimum": EXPECTED_REQUESTS_PER_DEPLOYMENT,
        },
        "actual_external_attempts": {
            "native_modal": None,
            "remote_vllm": remote_attempts,
            "confirmed_minimum": remote_attempts,
        },
        "completion_contract": {
            "requested_max_tokens": MAX_COMPLETION_TOKENS,
            "native_observed_completion_token_values": None,
            "remote_observed_completion_token_values": remote_completion_values,
            "remote_within_requested_cap": within_requested_cap,
            "cross_deployment_matched": False,
            "cross_deployment_e2e_comparable": False,
        },
    }


def _incomplete_cost(
    native_key_usage: float, remote_key_usage: float
) -> dict[str, Any]:
    budget = matched_budget_receipt()
    provider_usage = native_key_usage + remote_key_usage
    return {
        "openrouter_key_usage_usd": {
            "native_modal": native_key_usage,
            "remote_vllm": remote_key_usage,
            "total": provider_usage,
        },
        "cost_accounting": {
            **budget,
            "actual_provider_spend_usd": provider_usage,
            "actual_provider_within_envelope": (
                provider_usage <= budget["maximum_provider_spend_usd"]
            ),
        },
    }


def _observed_remote_models(remote: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(
                result.get("response_model")
                or WORKERS.get(str(result["selected_worker"]), "")
            )
            for result in remote["results"]
        }
    )


def build_incomplete_report(
    *,
    native_failure: dict[str, Any],
    remote: dict[str, Any],
    remote_deployment: dict[str, Any],
    native_key_usage: float,
    remote_key_usage: float,
    cleanup_receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Preserve a failed paired run when the native arm has no client artifact."""

    if native_failure.get("run_id") != remote.get("run_id"):
        raise RuntimeError("incomplete KV comparison run IDs diverged")
    remote_report, remote_attempts, completion_values, within_cap = (
        _incomplete_remote_analysis(remote, remote_deployment)
    )
    return {
        "schema_version": "rayline.openrouter-kv-cache-comparison.v2",
        "run_id": remote["run_id"],
        "status": "failed_incomplete",
        "failure": native_failure,
        "workload": remote["workload"],
        "models": WORKERS,
        "observed_models": {
            "native_modal": [],
            "remote_vllm": _observed_remote_models(remote),
        },
        "deployment_identity": {
            "native_modal": None,
            "remote_vllm": remote_deployment,
        },
        "deployment_attestation": {
            "native_h100": None,
            "remote_h100": True,
            "remote_flashinfer_runtime": True,
            "remote_ephemeral_app": True,
        },
        "acceptance": _incomplete_acceptance(remote_report),
        "deployments": {"remote_vllm": remote_report},
        "cross_deployment": {
            "comparable": False,
            "reason": "native arm did not produce a complete client/decision pair",
        },
        **_incomplete_counts(remote_attempts, completion_values, within_cap),
        **_incomplete_cost(native_key_usage, remote_key_usage),
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
            "the native arm is incomplete and has no latency or cache-effect result",
            "remote routing selected only DeepSeek V4 Flash for this workload",
            "the serial request rate includes provider generation and is not a concurrency throughput SLO",
            "native Modal buffers provider completion, so any future observed first token is not provider TTFT",
            "provider usage is the final management-key observation before key deletion",
        ],
    }


def main() -> None:
    args = _args()

    def load(value: str) -> dict[str, Any]:
        return json.loads(Path(value).read_text())

    if args.native_failure:
        report = build_incomplete_report(
            native_failure=load(args.native_failure),
            remote=load(args.remote_client),
            remote_deployment=load(args.remote_deployment),
            native_key_usage=float(load(args.native_key_usage)["usage_usd"]),
            remote_key_usage=float(load(args.remote_key_usage)["usage_usd"]),
            cleanup_receipt=(
                load(args.cleanup_receipt) if args.cleanup_receipt else None
            ),
        )
    else:
        missing = [
            name
            for name, value in (
                ("--native-client", args.native_client),
                ("--native-decisions", args.native_decisions),
                ("--native-deployment", args.native_deployment),
            )
            if not value
        ]
        if missing:
            raise SystemExit("complete report requires " + ", ".join(missing))
        report = build_report(
            native=load(args.native_client),
            decisions=read_decisions(Path(args.native_decisions)),
            remote=load(args.remote_client),
            native_deployment=load(args.native_deployment),
            remote_deployment=load(args.remote_deployment),
            native_key_usage=float(load(args.native_key_usage)["usage_usd"]),
            remote_key_usage=float(load(args.remote_key_usage)["usage_usd"]),
        )
    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
