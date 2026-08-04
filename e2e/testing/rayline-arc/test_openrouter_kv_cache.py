# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import itertools
import json

import openrouter_encoder_runtime as encoder_runtime
import openrouter_kv_cache_artifact_fixture as artifact_fixture
import openrouter_kv_cache_benchmark as benchmark
import openrouter_kv_cache_reporting as reporting
import openrouter_launch_authority as authority
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


def test_paid_remote_launch_starts_source_closed() -> None:
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
