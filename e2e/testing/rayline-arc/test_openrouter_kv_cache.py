# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib.machinery
import importlib.util
import itertools
import json
import sys
import types
from argparse import Namespace
from pathlib import Path
from typing import Any

if "modal" not in sys.modules and importlib.util.find_spec("modal") is None:
    modal_stub = types.ModuleType("modal")
    modal_stub.__spec__ = importlib.machinery.ModuleSpec("modal", loader=None)
    sys.modules["modal"] = modal_stub

import openrouter_encoder_runtime as encoder_runtime
import openrouter_kv_cache_artifact_fixture as artifact_fixture
import openrouter_kv_cache_benchmark as benchmark
import openrouter_kv_cache_journal as journal
import openrouter_kv_cache_reporting as reporting
import openrouter_kv_cache_workload_contract as workload_contract
import openrouter_launch_authority as authority
import openrouter_provider_preflight as provider_preflight
import pytest
import run_openrouter_kv_cache_native as native_launcher
import run_openrouter_modal_native as native_support
import yaml
from openrouter_agentic_workload import PROVIDER_NAMES, WORKERS
from openrouter_fullstack_state import EncoderDeployment, RunPacket, RuntimeState
from openrouter_kv_cache_matched_contract import (
    ARTIFACT_REVISION,
    FLASHINFER_APP_NAME,
    FLASHINFER_BACKEND,
    FLASHINFER_ENGINE_BUILD_ID,
    REQUIRED_FINAL_RESERVE_USD,
    RUN_ID,
    matched_budget_receipt,
)
from openrouter_modal_native_fixture import router_config_text

EXPECTED_REQUESTS_PER_DEPLOYMENT = 12
EXPECTED_TOTAL_REQUESTS = 24
EXPECTED_COMPLETION_LIMIT = 24
EXPECTED_STEPS = 3
EXPECTED_THROUGHPUT_RATIO = 0.5
MAXIMUM_EXPECTED_PACKET_USD = 9.14
HTTP_TOO_MANY_REQUESTS = 429
PARTIAL_FAILURE_ORDINAL = 3
PROVIDER_FAILURE_ORDINAL = 2
EXPECTED_PREFLIGHT_FAILURE_ATTEMPTS = 3


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
            training_stage="openrouter_kv_cache_agt017",
            max_completion_tokens=EXPECTED_COMPLETION_LIMIT,
            app_title="Rayline AGT017",
        )
    )
    assert config["router"]["training_stage"] == "openrouter_kv_cache_agt017"
    assert config["router"]["openrouter_app_title"] == "Rayline AGT017"
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
                        "input_tokens": serialized[step],
                        "output_tokens": EXPECTED_COMPLETION_LIMIT,
                    }
                )
    return (
        {
            "run_id": RUN_ID,
            "elapsed_seconds": 12.0,
            "workload": {
                "requests": EXPECTED_REQUESTS_PER_DEPLOYMENT,
                "max_completion_tokens": EXPECTED_COMPLETION_LIMIT,
            },
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
                        "prompt_tokens": serialized[step],
                        "completion_tokens": EXPECTED_COMPLETION_LIMIT,
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
        "run_id": RUN_ID,
        "elapsed_seconds": 24.0,
        "workload": {
            "requests": EXPECTED_REQUESTS_PER_DEPLOYMENT,
            "max_completion_tokens": EXPECTED_COMPLETION_LIMIT,
        },
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
        remote_deployment=_remote_deployment(),
        native_key_usage=0.01,
        remote_key_usage=0.01,
    )
    assert report["status"] == "passed"
    assert report["actual_provider_requests"] == EXPECTED_TOTAL_REQUESTS
    assert report["actual_external_attempts"] == EXPECTED_TOTAL_REQUESTS
    assert report["cross_deployment"]["selection_parity"] is True
    assert report["completion_contract"]["cross_deployment_matched"] is True
    assert report["completion_contract"]["cross_deployment_e2e_comparable"] is True
    assert report["deployment_attestation"]["remote_flashinfer_runtime"] is True
    assert report["acceptance"]["passed"] is True
    assert (
        report["cross_deployment"]["vllm_to_native_serial_request_throughput_ratio"]
        == EXPECTED_THROUGHPUT_RATIO
    )
    assert report["deployments"]["native_modal"]["steady_state"]["episode"] == 1
    for deployment in report["deployments"].values():
        assert deployment["comparison"]["retained_token_work_saved_fraction"] > 0
        assert deployment["paths"]["retained"]["retries"] == 0
        assert deployment["paths"]["replay"]["retries"] == 0


def test_incomplete_report_preserves_remote_metrics_after_native_429() -> None:
    remote = _remote_client()
    report = reporting.build_incomplete_report(
        native_failure={
            "run_id": RUN_ID,
            "deployment": "native_modal",
            "http_status": 429,
        },
        remote=copy.deepcopy(remote),
        remote_deployment=_remote_deployment(),
        native_key_usage=0.002,
        remote_key_usage=0.005,
        cleanup_receipt={"passed": True},
    )
    assert report["status"] == "failed_incomplete"
    assert report["acceptance"]["passed"] is False
    assert report["acceptance"]["gates"]["native_arm_complete"] is False
    assert report["acceptance"]["gates"]["flashinfer_retained_token_saving"] is True
    assert report["deployments"]["remote_vllm"]["paths"]["retained"]["retries"] == 0
    assert (
        report["completed_provider_requests"]["remote_vllm"]
        == EXPECTED_REQUESTS_PER_DEPLOYMENT
    )
    assert report["cleanup"] == {"passed": True}


def test_report_marks_a_completion_policy_deviation() -> None:
    native, decisions = _native_client()
    remote = _remote_client()
    for result in remote["results"]:
        result["completion_tokens"] = 96 if result["step"] < EXPECTED_STEPS - 1 else 18
    final_serialized_tokens = max(
        decision["features"]["serialized_tokens"] for decision in decisions
    )
    for decision in decisions:
        decision["output_tokens"] = (
            EXPECTED_COMPLETION_LIMIT
            if decision["features"]["serialized_tokens"] < final_serialized_tokens
            else 18
        )
    report = reporting.build_report(
        native=copy.deepcopy(native),
        decisions=copy.deepcopy(decisions),
        remote=copy.deepcopy(remote),
        native_deployment={"gpu": "H100"},
        remote_deployment=_remote_deployment(),
        native_key_usage=0.01,
        remote_key_usage=0.01,
    )
    assert report["status"] == "failed_acceptance"
    assert report["acceptance"]["passed"] is False
    assert report["completion_contract"]["cross_deployment_matched"] is False
    assert report["completion_contract"]["cross_deployment_e2e_comparable"] is False


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
            remote_deployment=_remote_deployment(),
            native_key_usage=0.01,
            remote_key_usage=0.01,
        )
    except RuntimeError as error:
        assert "selections diverged" in str(error)
    else:
        raise AssertionError("selection divergence did not fail")


def test_worker_set_remains_the_three_model_openrouter_pool() -> None:
    assert set(WORKERS) == {"worker-a", "worker-b", "worker-c"}


def _remote_deployment() -> dict[str, object]:
    return {
        "encoder_app_name": FLASHINFER_APP_NAME,
        "encoder_gpu": "H100",
        "encoder_build_id": FLASHINFER_ENGINE_BUILD_ID,
        "encoder_gdn_prefill_backend": FLASHINFER_BACKEND,
        "encoder_ephemeral": True,
    }


def test_paid_remote_launch_is_permanently_source_closed() -> None:
    preregistration, authorization = authority.AUTHORITY_PINS["kv-cache-flashinfer"]
    assert preregistration == ""
    assert authorization == ""


def test_matched_artifact_freezes_the_24_token_worker_contract(tmp_path) -> None:
    artifact_fixture.generate(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["artifact_id"] == ARTIFACT_REVISION
    assert all(
        worker["minimum_completion_tokens"] == EXPECTED_COMPLETION_LIMIT
        and worker["max_completion_tokens"] == EXPECTED_COMPLETION_LIMIT
        for worker in manifest["workers"]
    )


def test_matched_budget_preserves_the_frozen_reserve() -> None:
    receipt = matched_budget_receipt()
    assert receipt["maximum_complete_packet_usd"] < MAXIMUM_EXPECTED_PACKET_USD
    assert receipt["reserve_after_complete_envelope_usd"] >= REQUIRED_FINAL_RESERVE_USD


def test_native_request_uses_session_identity_for_kv_isolation(monkeypatch) -> None:
    monkeypatch.setenv("RAYLINE_MODAL_NATIVE_ROUTER_TOKEN", "test-token")
    headers = benchmark._request_headers("native_modal", "episode-1")
    assert headers["x-rayline-episode-id"] == "episode-1"
    assert headers["x-rayline-session"] == "episode-1"


def test_kv_journal_survives_a_partial_failed_run(tmp_path, monkeypatch) -> None:
    calls = 0

    def fake_request(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == PARTIAL_FAILURE_ORDINAL:
            raise benchmark.OpenRouterHTTPError(
                endpoint="test",
                status_code=HTTP_TOO_MANY_REQUESTS,
                retry_after_seconds=1,
                error_type="rate_limit",
                provider_code="429",
                external_attempts=2,
            )
        return {
            "selected_worker": "worker-a",
            "client_attempts": 1,
            "external_attempts": 1,
        }

    journal_path = tmp_path / "native-journal.jsonl"
    monkeypatch.setattr(benchmark, "_request", fake_request)
    args = Namespace(
        deployment="native_modal",
        base_url="http://native.invalid",
        metrics_url="",
        run_id="public-journal-test",
        timeout_seconds=1,
        journal=str(journal_path),
    )

    with pytest.raises(benchmark.OpenRouterHTTPError, match="HTTP 429"):
        benchmark.run(args)

    events = journal.read(journal_path)
    assert [event["event"] for event in events] == [
        "request_succeeded",
        "request_succeeded",
        "request_failed",
    ]
    assert events[-1]["ordinal"] == PARTIAL_FAILURE_ORDINAL
    assert events[-1]["error"] == {
        "client_attempts": 1,
        "error_category": "",
        "error_class": "OpenRouterHTTPError",
        "error_type": "rate_limit",
        "external_attempts": 2,
        "http_status": HTTP_TOO_MANY_REQUESTS,
        "provider_code": "429",
        "retry_statuses": [],
    }
    assert "message" not in json.dumps(events)


def test_kv_journal_rejects_request_content(tmp_path: Path) -> None:
    journal_path = tmp_path / "private.jsonl"
    journal.initialize(journal_path)
    with pytest.raises(RuntimeError, match="request content"):
        journal.append(
            journal_path,
            {"event": "request_failed", "messages": [{"content": "private"}]},
        )


def test_kv_journal_recovers_fsynced_prefix_from_a_truncated_tail(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "partial.jsonl"
    journal.initialize(journal_path)
    journal.append(journal_path, {"event": "request_succeeded", "ordinal": 1})
    with journal_path.open("ab") as output:
        output.write(b'{"event":"request_failed"')

    assert journal.read(journal_path) == [
        {
            "schema_version": journal.SCHEMA_VERSION,
            "event": "request_succeeded",
            "ordinal": 1,
        }
    ]


def test_kv_journal_rejects_a_complete_corrupt_tail(tmp_path: Path) -> None:
    journal_path = tmp_path / "corrupt.jsonl"
    journal.initialize(journal_path)
    with journal_path.open("ab") as output:
        output.write(b'{"event":"request_failed"\n')

    with pytest.raises(RuntimeError, match="corrupted before its tail"):
        journal.read(journal_path)


def test_native_failure_still_flushes_and_downloads_decisions(
    tmp_path: Path, monkeypatch
) -> None:
    context = native_support.LaunchContext(
        semantic_root=tmp_path,
        pathfinder_root=tmp_path,
        pathfinder_python=tmp_path / "python",
        output_dir=tmp_path,
        environment={},
        semantic_head="semantic",
        pathfinder_head="pathfinder",
        timeout_seconds=1,
    )
    calls: list[str] = []

    monkeypatch.setattr(native_launcher, "_register_context", lambda **_kwargs: None)
    monkeypatch.setattr(native_support, "_wait_ready", lambda *_args: {"ready": True})
    monkeypatch.setattr(native_support, "_request_json", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        native_support,
        "_flush",
        lambda *_args: calls.append("flush") or {"flushed": True},
    )

    def fake_run(command: list[str], **_kwargs: Any) -> None:
        if str(native_launcher.BENCHMARK) in command:
            calls.append("benchmark")
            raise RuntimeError("workload failed")
        if "volume" in command and "get" in command:
            calls.append("decisions")
            return
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(native_support, "_run", fake_run)
    with pytest.raises(RuntimeError, match="workload failed"):
        native_launcher._measure(
            context,
            ephemeral_key="ephemeral-key",
            router_token="router-token",
        )

    assert calls == ["benchmark", "flush", "decisions"]


def test_provider_preflight_covers_every_worker_without_gpu(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_stream(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        worker = kwargs["expected_worker"]
        return {
            "response_model": WORKERS[worker],
            "provider": PROVIDER_NAMES[worker][0],
            "completion_tokens": 1,
            "client_attempts": 1,
            "external_attempts": 1,
            "cost_usd": 0.00001,
        }

    monkeypatch.setattr(provider_preflight, "_stream_request", fake_stream)
    report = provider_preflight.run_preflight(
        openrouter_key="test-key",
        run_id="public-provider-preflight",
        timeout_seconds=1,
    )

    assert report["status"] == "passed"
    assert list(report["workers"]) == list(WORKERS)
    assert [call["expected_worker"] for call in calls] == list(WORKERS)
    assert all(call["path"] == "direct" for call in calls)
    assert all(call["max_completion_tokens"] == 1 for call in calls)


def test_provider_preflight_preserves_sanitized_429_evidence(monkeypatch) -> None:
    calls = 0

    def fake_stream(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        worker = kwargs["expected_worker"]
        if calls == PROVIDER_FAILURE_ORDINAL:
            raise benchmark.OpenRouterHTTPError(
                endpoint="test",
                status_code=HTTP_TOO_MANY_REQUESTS,
                retry_after_seconds=1,
                error_type="rate_limit",
                provider_code="429",
                error_category="rate_limited",
                external_attempts=2,
            )
        return {
            "response_model": WORKERS[worker],
            "provider": PROVIDER_NAMES[worker][0],
            "completion_tokens": 1,
            "client_attempts": 1,
            "external_attempts": 1,
            "cost_usd": 0.00001,
        }

    monkeypatch.setattr(provider_preflight, "_stream_request", fake_stream)
    report = provider_preflight.run_preflight(
        openrouter_key="test-key",
        run_id="public-provider-preflight",
        timeout_seconds=1,
    )

    assert report["status"] == "failed"
    assert report["failed_worker"] == "worker-b"
    assert report["http_status"] == HTTP_TOO_MANY_REQUESTS
    assert report["external_attempts"] == EXPECTED_PREFLIGHT_FAILURE_ATTEMPTS
    assert "message" not in report


def test_workload_contract_separates_semantic_and_static_coverage() -> None:
    contract = workload_contract.validate()
    semantic = contract["semantic_cache_lane"]
    static = contract["stratified_serving_lane"]

    assert semantic["routing"] == "natural_rayline_selection"
    assert semantic["three_worker_coverage_required"] is False
    assert {cell["worker"] for cell in static["cells"]} == set(WORKERS)
    assert static["semantic_selection_claim_admissible"] is False
    assert contract["retry_policy"] == {
        "owner": {
            "native_modal": "Pathfinder OpenRouter worker transport",
            "remote_vllm": "Envoy OpenRouter route",
        },
        "retryable_statuses": [429, 503],
        "maximum_retries": 1,
        "benchmark_client_retries": 0,
        "reason": "keep retries below one semantic selection transaction",
    }


def _ephemeral_packet(tmp_path) -> RunPacket:
    return RunPacket(
        compose_override=tmp_path / "compose.yaml",
        config=tmp_path / "config.yaml",
        driver=tmp_path / "driver.py",
        project_name="test",
        key_limit_usd=0.05,
        maximum_seconds=1,
        protected_encoder=True,
        encoder=EncoderDeployment(
            app_name=FLASHINFER_APP_NAME,
            class_name="SessionEncoder",
            build_id=FLASHINFER_ENGINE_BUILD_ID,
            deployment_source_commit="test",
            plugin_source_digest="test",
            deploy_service_path=tmp_path / "service.py",
            ephemeral=True,
        ),
    )


def test_cleanup_does_not_touch_an_unowned_ephemeral_encoder(
    tmp_path, monkeypatch
) -> None:
    packet = _ephemeral_packet(tmp_path)
    state = RuntimeState(environment={}, encoder_deployed=False, encoder_owned=False)
    monkeypatch.setattr(
        encoder_runtime,
        "_restore_encoder_scale_to_zero",
        lambda _state: (_ for _ in ()).throw(AssertionError("unexpected cleanup")),
    )
    encoder_runtime.cleanup_encoder(
        packet,
        state,
        ["modal"],
        {},
        cwd=tmp_path,
    )


def test_cleanup_stops_an_owned_ephemeral_encoder(tmp_path, monkeypatch) -> None:
    packet = _ephemeral_packet(tmp_path)
    state = RuntimeState(
        environment={},
        encoder_app_id="ap-test",
        encoder_deployed=True,
        encoder_owned=True,
    )
    calls = []
    monkeypatch.setattr(
        encoder_runtime,
        "_restore_encoder_scale_to_zero",
        lambda _state: calls.append("restore"),
    )
    monkeypatch.setattr(
        encoder_runtime,
        "_stop_encoder_containers",
        lambda *_args, **_kwargs: calls.append("containers"),
    )
    monkeypatch.setattr(
        encoder_runtime,
        "_run",
        lambda *_args, **_kwargs: calls.append("app") or None,
    )
    monkeypatch.setattr(
        encoder_runtime,
        "_wait_ephemeral_encoder_cleanup",
        lambda *_args, **_kwargs: calls.append("verify"),
    )
    encoder_runtime.cleanup_encoder(
        packet,
        state,
        ["modal"],
        {},
        cwd=tmp_path,
    )
    assert calls == ["restore", "containers", "app", "verify"]
