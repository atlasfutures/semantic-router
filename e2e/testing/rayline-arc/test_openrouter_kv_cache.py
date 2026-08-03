# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import itertools

import openrouter_kv_cache_benchmark as benchmark
import openrouter_kv_cache_reporting as reporting
import openrouter_launch_authority as authority
import yaml
from openrouter_agentic_workload import PROVIDER_NAMES, WORKERS
from openrouter_modal_native_fixture import router_config_text

EXPECTED_REQUESTS_PER_DEPLOYMENT = 12
EXPECTED_TOTAL_REQUESTS = 24
EXPECTED_COMPLETION_LIMIT = 24
EXPECTED_STEPS = 3


def test_history_states_are_strict_growing_prefixes() -> None:
    states = benchmark.history_states()
    assert len(states) == benchmark.STEPS == EXPECTED_STEPS
    assert benchmark.EXPECTED_REQUESTS == EXPECTED_REQUESTS_PER_DEPLOYMENT
    for previous, current in itertools.pairwise(states):
        previous_messages = previous["messages"]
        current_messages = current["messages"]
        assert current_messages[: len(previous_messages)] == previous_messages
        assert len(current_messages) == len(previous_messages) + 2


def test_native_fixture_accepts_the_cache_completion_cap() -> None:
    config = yaml.safe_load(
        router_config_text(
            training_stage="openrouter_kv_cache_agt016",
            max_completion_tokens=EXPECTED_COMPLETION_LIMIT,
            app_title="Rayline AGT016",
        )
    )
    assert config["router"]["training_stage"] == "openrouter_kv_cache_agt016"
    assert config["router"]["openrouter_app_title"] == "Rayline AGT016"
    assert all(
        worker["max_completion_tokens"] == EXPECTED_COMPLETION_LIMIT
        for worker in config["workers"]
    )
    assert all(
        worker["minimum_completion_tokens"] == EXPECTED_COMPLETION_LIMIT
        for worker in config["workers"]
    )


def _native_client() -> tuple[dict[str, object], list[dict[str, object]]]:
    results: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    serialized = [100, 150, 200]
    retained_cached = [0, 100, 150]
    for episode in range(2):
        for step in range(3):
            for mode in ("retained", "replay"):
                request_id = f"native-{episode}-{step}-{mode}"
                results.append(
                    {
                        "mode": mode,
                        "episode": episode,
                        "step": step,
                        "request_id": request_id,
                        "selected_worker": "worker-a",
                        "total_seconds": 1.0,
                        "first_token_seconds": 0.9,
                        "external_attempts": 1,
                        "cost_usd": 0.001,
                    }
                )
                cached = retained_cached[step] if mode == "retained" else 0
                decisions.append(
                    {
                        "request_id": request_id,
                        "error": "",
                        "selected_worker": "worker-a",
                        "served_model": WORKERS["worker-a"],
                        "decision_latency_ms": 101,
                        "served_provider": PROVIDER_NAMES["worker-a"][0],
                        "features": {
                            "serialized_tokens": serialized[step],
                            "cached_prefix_tokens": cached,
                            "encode_mode": (
                                "delta"
                                if mode == "retained" and step > 0
                                else "prefill"
                            ),
                            "embedding_latency_ms": 100,
                            "q_latency_ms": 1,
                        },
                        "transport": {"attempts": [{"status": "success"}]},
                        "settlement_cost_basis": {"provider_charged_usd": 0.001},
                    }
                )
    return (
        {
            "run_id": "agt015-test",
            "workload": {"requests": EXPECTED_REQUESTS_PER_DEPLOYMENT},
            "results": results,
        },
        decisions,
    )


def _remote_client() -> dict[str, object]:
    results: list[dict[str, object]] = []
    serialized = [100, 150, 200]
    retained_work = [100, 50, 50]
    for episode in range(2):
        for step in range(3):
            for mode in ("retained", "replay"):
                work = retained_work[step] if mode == "retained" else serialized[step]
                results.append(
                    {
                        "mode": mode,
                        "episode": episode,
                        "step": step,
                        "selected_worker": "worker-a",
                        "session_action": (
                            "appended" if mode == "retained" and step > 0 else "created"
                        ),
                        "total_seconds": 1.1,
                        "first_token_seconds": 1.0,
                        "external_attempts": 1,
                        "cost_usd": 0.001,
                        "provider": PROVIDER_NAMES["worker-a"][0],
                        "router_stage": {
                            "mean_decomposition": {
                                "router_seconds": 0.2,
                                "encoder_seconds": 0.19,
                                "router_non_encoder_seconds": 0.01,
                            }
                        },
                        "encoder_stage": {
                            "coordinator": {"backend_appended_tokens": work}
                        },
                    }
                )
    return {
        "run_id": "agt015-test",
        "workload": {"requests": EXPECTED_REQUESTS_PER_DEPLOYMENT},
        "results": results,
    }


def test_report_enforces_a_smaller_retained_token_work_envelope() -> None:
    native, decisions = _native_client()
    remote = _remote_client()
    report = reporting.build_report(
        native=copy.deepcopy(native),
        decisions=copy.deepcopy(decisions),
        remote=copy.deepcopy(remote),
        native_deployment={"gpu": "H100"},
        remote_deployment={"gpu": "H100"},
        native_key_usage=0.01,
        remote_key_usage=0.01,
    )
    assert report["status"] == "passed"
    assert report["actual_provider_requests"] == EXPECTED_TOTAL_REQUESTS
    assert report["actual_external_attempts"] == EXPECTED_TOTAL_REQUESTS
    assert report["cross_deployment"]["selection_parity"] is True
    for deployment in report["deployments"].values():
        assert deployment["comparison"]["retained_token_work_saved_fraction"] > 0
        assert deployment["paths"]["retained"]["retries"] == 0
        assert deployment["paths"]["replay"]["retries"] == 0


def test_cross_deployment_selection_divergence_fails() -> None:
    native, decisions = _native_client()
    remote = _remote_client()
    remote["results"][0]["selected_worker"] = "worker-b"
    try:
        reporting.build_report(
            native=copy.deepcopy(native),
            decisions=copy.deepcopy(decisions),
            remote=copy.deepcopy(remote),
            native_deployment={"gpu": "H100"},
            remote_deployment={"gpu": "H100"},
            native_key_usage=0.01,
            remote_key_usage=0.01,
        )
    except RuntimeError as error:
        assert "selections diverged" in str(error)
    else:
        raise AssertionError("selection divergence did not fail")


def test_worker_set_remains_the_three_model_openrouter_pool() -> None:
    assert set(WORKERS) == {"worker-a", "worker-b", "worker-c"}


def test_paid_remote_launch_starts_source_closed() -> None:
    preregistration, authorization = authority.AUTHORITY_PINS["kv-cache"]
    assert preregistration == "02102e02a6da8090d272ece8c18ce7bc32f7e8d9"
    assert authorization == "4742e176cde87e8f0da365f3c81261858a4980ab"


def test_native_request_uses_session_identity_for_kv_isolation(monkeypatch) -> None:
    monkeypatch.setenv("RAYLINE_MODAL_NATIVE_ROUTER_TOKEN", "test-token")
    headers = benchmark._request_headers("native_modal", "episode-1")
    assert headers["x-rayline-episode-id"] == "episode-1"
    assert headers["x-rayline-session"] == "episode-1"
