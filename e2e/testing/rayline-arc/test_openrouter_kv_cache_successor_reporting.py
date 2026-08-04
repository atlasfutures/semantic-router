# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import itertools

import openrouter_kv_cache_successor_reporting as reporting
import pytest
from openrouter_agentic_workload import PROVIDER_NAMES, WORKERS
from openrouter_kv_cache_successor_contract import (
    ARTIFACT_REVISION,
    NATIVE_APP_NAME,
    REMOTE_APP_NAME,
    RUN_ID,
)
from openrouter_kv_cache_successor_workload import (
    ENCODER_ARTIFACT_ID,
    EXPECTED_SELECTION_TRACES,
    OFFLINE_REPORT_SCHEMA_VERSION,
    SEQUENCE_IDS,
)
from openrouter_kv_cache_successor_workload import (
    SCHEMA_VERSION as WORKLOAD_SCHEMA_VERSION,
)
from openrouter_provider_preflight_contract import REPORT_SCHEMA

EXPECTED_LOGICAL_PROVIDER_REQUESTS = 78
EXPECTED_NO_RETRY_EXTERNAL_ATTEMPTS = 78
EXPECTED_ONE_RETRY_EXTERNAL_ATTEMPTS = 79
EXPECTED_REMOTE_TO_NATIVE_THROUGHPUT_RATIO = 2.0


def _client(deployment: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    results: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    serialized = [100, 150, 200]
    cells = itertools.product(
        SEQUENCE_IDS,
        range(2),
        range(3),
        ("retained", "replay"),
    )
    for sequence_id, episode, step, mode in cells:
        worker = EXPECTED_SELECTION_TRACES[sequence_id][step]
        request_id = f"{deployment}-{sequence_id}-{episode}-{step}-{mode}"
        common = {
            "sequence_id": sequence_id,
            "mode": mode,
            "episode": episode,
            "step": step,
            "request_id": request_id,
            "selected_worker": worker,
            "expected_worker": worker,
            "total_seconds": 1.0,
            "first_token_seconds": 0.8,
            "external_attempts": 1,
            "cost_usd": 0.001,
            "completion_tokens": 24,
        }
        if deployment == "remote_vllm":
            work = 50 if mode == "retained" and step > 0 else serialized[step]
            common.update(
                {
                    "session_action": (
                        "appended" if mode == "retained" and step > 0 else "created"
                    ),
                    "provider": PROVIDER_NAMES[worker][0],
                    "prompt_tokens": serialized[step],
                    "router_stage": {
                        "mean_decomposition": {
                            "router_seconds": 0.2,
                            "encoder_seconds": 0.19,
                            "router_non_encoder_seconds": 0.01,
                        }
                    },
                    "encoder_stage": {"coordinator": {"backend_appended_tokens": work}},
                }
            )
        results.append(common)
        if deployment == "native_modal":
            cached = serialized[step] - 50 if mode == "retained" and step > 0 else 0
            decisions.append(
                {
                    "request_id": request_id,
                    "error": "",
                    "selected_worker": worker,
                    "served_model": WORKERS[worker],
                    "served_provider": PROVIDER_NAMES[worker][0],
                    "decision_latency_ms": 100,
                    "features": {
                        "serialized_tokens": serialized[step],
                        "cached_prefix_tokens": cached,
                        "encode_mode": (
                            "delta" if mode == "retained" and step > 0 else "prefill"
                        ),
                        "embedding_latency_ms": 90,
                        "q_latency_ms": 10,
                    },
                    "transport": {"attempts": [{"status": "success"}]},
                    "settlement_cost_basis": {"provider_charged_usd": 0.001},
                    "input_tokens": serialized[step],
                    "output_tokens": 24,
                }
            )
    return (
        {
            "schema_version": "rayline.openrouter-kv-cache-client.v2",
            "run_id": RUN_ID,
            "deployment": deployment,
            "workload": {
                "requests": 36,
                "max_completion_tokens": 24,
            },
            "elapsed_seconds": 72.0 if deployment == "native_modal" else 36.0,
            "results": results,
        },
        decisions,
    )


def _observations() -> list[dict[str, object]]:
    return [
        {
            "sequence_id": sequence_id,
            "step": step,
            "selected_worker": worker,
            "expected_worker": worker,
            "top_two_score_gap": 0.002,
        }
        for sequence_id, trace in EXPECTED_SELECTION_TRACES.items()
        for step, worker in enumerate(trace)
    ]


def _preflight() -> dict[str, object]:
    return {
        "schema_version": REPORT_SCHEMA,
        "run_id": RUN_ID,
        "status": "passed",
        "provider_requests": 3,
        "maximum_provider_requests": 3,
        "external_attempts": 3,
        "maximum_external_attempts": 6,
        "retries": 0,
        "cost_usd": 0.001,
        "maximum_completion_tokens": 1,
        "workers": {
            worker: {
                "model": model,
                "provider": PROVIDER_NAMES[worker][0],
                "completion_tokens": 1,
                "client_attempts": 1,
                "external_attempts": 1,
            }
            for worker, model in WORKERS.items()
        },
        "provider_fallbacks": False,
        "reasoning_enabled": False,
        "performance_inference_admissible": False,
    }


def _inputs() -> dict[str, object]:
    native, decisions = _client("native_modal")
    remote, _unused = _client("remote_vllm")
    remote["pre_provider_remote_encoder_parity"] = {
        "status": "passed",
        "states": 9,
        "minimum_top_two_score_gap": 0.002,
        "provider_calls": 0,
        "sessions_closed": 3,
        "prompt_text_emitted": False,
        "raw_embeddings_emitted": False,
        "observations": _observations(),
    }
    return {
        "native": native,
        "decisions": decisions,
        "remote": remote,
        "native_deployment": {
            "app_name": NATIVE_APP_NAME,
            "gpu": "H100",
            "checkpoint": {"artifact_id": ARTIFACT_REVISION},
        },
        "remote_deployment": {
            "encoder_app_name": REMOTE_APP_NAME,
            "encoder_gpu": "H100",
            "encoder_build_id": (
                "vllm@9f5ea81ca0aa570aea46baf82311a1139c1267ca" "+gdn-flashinfer-eager"
            ),
            "encoder_gdn_prefill_backend": "flashinfer",
            "encoder_ephemeral": True,
            "artifact_revision": ARTIFACT_REVISION,
        },
        "native_preflight": _preflight(),
        "remote_preflight": _preflight(),
        "offline_coverage": {
            "schema_version": OFFLINE_REPORT_SCHEMA_VERSION,
            "status": "passed",
            "workload_schema_version": WORKLOAD_SCHEMA_VERSION,
            "encoder": {"artifact_id": ENCODER_ARTIFACT_ID},
            "all_workers_covered": True,
            "minimum_top_two_score_gap": 0.002,
            "observations": _observations(),
            "prompt_text_emitted": False,
            "raw_embeddings_emitted": False,
            "provider_calls": 0,
        },
        "cleanup_receipt": {"passed": True},
        "native_key_usage": 0.02,
        "remote_key_usage": 0.02,
        "budget_receipt": {"maximum_provider_spend_usd": 0.10},
    }


def test_v3_report_aggregates_each_natural_model_and_architecture() -> None:
    report = reporting.build_report(**copy.deepcopy(_inputs()))

    assert report["schema_version"] == "rayline.openrouter-kv-cache-comparison.v3"
    assert report["status"] == "passed"
    assert (
        report["actual_logical_provider_requests"] == EXPECTED_LOGICAL_PROVIDER_REQUESTS
    )
    assert report["actual_external_attempts"] == EXPECTED_NO_RETRY_EXTERNAL_ATTEMPTS
    assert report["acceptance"]["passed"] is True
    assert report["cross_deployment"]["selection_parity"] is True
    assert (
        report["cross_deployment"]["vllm_to_native_serial_request_throughput_ratio"]
        == EXPECTED_REMOTE_TO_NATIVE_THROUGHPUT_RATIO
    )
    for deployment in report["deployments"].values():
        assert deployment["natural_worker_mix"] == {
            "worker-a": 12,
            "worker-b": 12,
            "worker-c": 12,
        }
        assert set(deployment["per_model"]) == set(WORKERS.values())
        assert set(deployment["per_sequence"]) == set(SEQUENCE_IDS)
        assert deployment["output_token_throughput_tps"] > 0


def test_v3_report_rejects_remote_encoder_trace_divergence() -> None:
    inputs = _inputs()
    inputs["remote"]["pre_provider_remote_encoder_parity"]["observations"][0][
        "selected_worker"
    ] = "worker-a"
    with pytest.raises(RuntimeError, match="encoder traces diverged"):
        reporting.build_report(**inputs)


def test_v3_report_keeps_one_server_retry_inside_the_attempt_envelope() -> None:
    inputs = _inputs()
    inputs["remote"]["results"][0]["external_attempts"] = 2
    report = reporting.build_report(**inputs)
    assert report["status"] == "passed"
    assert report["actual_external_attempts"] == EXPECTED_ONE_RETRY_EXTERNAL_ATTEMPTS
    assert report["deployments"]["remote_vllm"]["paths"]["retained"]["retries"] == 1
