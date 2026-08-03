# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.machinery
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
DEPLOY_DIR = REPO_ROOT / "deploy/compose/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

artifact = importlib.import_module("openrouter_agentic_artifact_fixture")
benchmark = importlib.import_module("openrouter_agentic_benchmark")
preflight = importlib.import_module("openrouter_agentic_preflight")
gateway_diagnostic = importlib.import_module("openrouter_gateway_shape_diagnostic")
prime_diagnostic = importlib.import_module("openrouter_gateway_prime_diagnostic")
reporting = importlib.import_module("openrouter_agentic_reporting")
workload = importlib.import_module("openrouter_agentic_workload")
if "modal" not in sys.modules and importlib.util.find_spec("modal") is None:
    modal_stub = types.ModuleType("modal")
    modal_stub.__spec__ = importlib.machinery.ModuleSpec("modal", loader=None)
    sys.modules["modal"] = modal_stub
launcher = importlib.import_module("run_openrouter_fullstack")
EXPECTED_MAX_COMPLETION_TOKENS = 96
EXPECTED_MAX_MEASURED_REQUESTS = 72
EXPECTED_BENCHMARK_PROVIDER_REQUESTS = 100
EXPECTED_BENCHMARK_EXTERNAL_ATTEMPTS = 203
EXPECTED_MAX_PROVIDER_REQUESTS = 104
EXPECTED_MAX_EXTERNAL_ATTEMPTS = 214
EXPECTED_PROVIDER_COST_LIMIT_USD = 0.50
EXPECTED_MIN_TOOL_RESULT_CHARACTERS = 1_000
EXPECTED_SELECTED_CASES = 6
EXPECTED_SELECTED_CASES_PER_ACTIVE_WORKER = 3
EXPECTED_EPHEMERAL_KEY_LIMIT_USD = 0.75
EXPECTED_DIAGNOSTIC_KEY_LIMIT_USD = 0.05
EXPECTED_DIAGNOSTIC_REQUESTS = 6
EXPECTED_DIAGNOSTIC_SUCCESSES = 4
EXPECTED_DIAGNOSTIC_FAILURES = 2


def test_agentic_artifact_uses_the_requested_low_cost_pool(tmp_path: Path) -> None:
    artifact.generate(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())

    assert manifest["artifact_id"] == "public-rayline-arc-openrouter-agentic-v1"
    assert [worker["model"] for worker in manifest["workers"]] == [
        "deepseek/deepseek-v4-flash",
        "xiaomi/mimo-v2.5",
        "tencent/hy3",
    ]
    assert [worker["openrouter_provider_slug"] for worker in manifest["workers"]] == [
        "baidu",
        "xiaomi",
        "tencent",
    ]
    assert [worker["openrouter_provider_name"] for worker in manifest["workers"]] == [
        "Baidu",
        "Xiaomi",
        "Tencent",
    ]
    for worker in manifest["workers"]:
        assert worker["capability_tags"] == ["public-openrouter-agentic-benchmark"]
        assert worker["max_completion_tokens"] == EXPECTED_MAX_COMPLETION_TOKENS
        assert worker["openrouter_allow_fallbacks"] is False
        assert worker["openrouter_max_retries"] == 1
        assert worker["thinking_mode"] == "off"


def test_agentic_workload_is_bounded_realistic_and_balanced() -> None:
    assert benchmark.PATHS == ("direct", "gateway_static", "arc")
    assert benchmark.CONCURRENCY_LEVELS == (1, 4)
    assert benchmark.MAX_COMPLETION_TOKENS == EXPECTED_MAX_COMPLETION_TOKENS
    assert benchmark.MAX_MEASURED_REQUESTS == EXPECTED_MAX_MEASURED_REQUESTS
    assert benchmark.MAX_PROVIDER_REQUESTS == EXPECTED_MAX_PROVIDER_REQUESTS
    assert benchmark.MAX_EXTERNAL_ATTEMPTS == EXPECTED_MAX_EXTERNAL_ATTEMPTS
    assert (
        benchmark.MAX_BENCHMARK_PROVIDER_REQUESTS
        == EXPECTED_BENCHMARK_PROVIDER_REQUESTS
    )
    assert (
        benchmark.MAX_BENCHMARK_EXTERNAL_ATTEMPTS
        == EXPECTED_BENCHMARK_EXTERNAL_ATTEMPTS
    )
    assert benchmark.MAX_REPORTED_PROVIDER_COST_USD == EXPECTED_PROVIDER_COST_LIMIT_USD

    cases = [benchmark._candidate_case(index) for index in range(3)]
    assert {case["scenario"] for case in cases} == set(benchmark.SCENARIOS)
    for case in cases:
        roles = [message["role"] for message in case["messages"]]
        assert roles == ["system", "user", "assistant", "tool", "user"]
        assert case["tools"] == workload.TOOLS
        tool_result = next(
            message["content"]
            for message in case["messages"]
            if message["role"] == "tool"
        )
        assert len(tool_result) > EXPECTED_MIN_TOOL_RESULT_CHARACTERS


def test_agentic_case_selection_balances_three_active_workers_and_all_shapes() -> None:
    candidates = {
        "worker-a": [benchmark._candidate_case(0), benchmark._candidate_case(3)],
        "worker-b": [benchmark._candidate_case(1), benchmark._candidate_case(4)],
        "worker-c": [benchmark._candidate_case(2), benchmark._candidate_case(5)],
    }

    selected = benchmark._choose_cases(candidates)

    assert selected is not None
    assert len(selected) == EXPECTED_SELECTED_CASES
    assert {case["scenario"] for case in selected} == set(benchmark.SCENARIOS)


def test_agentic_case_selection_treats_zero_share_as_a_result() -> None:
    candidates = {
        "worker-a": [benchmark._candidate_case(index) for index in (1, 2, 4, 5)],
        "worker-b": [],
        "worker-c": [benchmark._candidate_case(index) for index in (0, 3, 6, 9)],
    }

    selected = benchmark._choose_cases(candidates)

    assert selected is not None
    assert len(selected) == EXPECTED_SELECTED_CASES
    assert {case["scenario"] for case in selected} == set(benchmark.SCENARIOS)
    assert (
        sum(case in candidates["worker-a"] for case in selected)
        == EXPECTED_SELECTED_CASES_PER_ACTIVE_WORKER
    )
    assert sum(case in candidates["worker-b"] for case in selected) == 0
    assert (
        sum(case in candidates["worker-c"] for case in selected)
        == EXPECTED_SELECTED_CASES_PER_ACTIVE_WORKER
    )


def test_agentic_endpoint_probes_reach_every_pinned_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_stream(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "selected_worker": kwargs["expected_worker"],
            "external_attempts": 1,
            "cost_usd": 0.001,
        }

    monkeypatch.setattr(benchmark, "_stream_request", fake_stream)
    results = benchmark._probe_endpoints(
        gateway_url="http://gateway.invalid",
        openrouter_key="test-key",
        run_id="test-run",
        timeout_seconds=1.0,
    )

    assert len(results) == len(benchmark.WORKERS)
    assert [call["path"] for call in calls] == ["gateway_static"] * 3
    assert [call["expected_worker"] for call in calls] == list(benchmark.WORKERS)
    assert all(
        call["maximum_attempts"] == benchmark.MAX_DATA_PLANE_ATTEMPTS for call in calls
    )
    assert all(
        call["retryable_status_codes"]
        == benchmark.ENDPOINT_REACHABILITY_RETRYABLE_STATUS_CODES
        for call in calls
    )


def test_agentic_key_readiness_is_a_bounded_direct_ds4_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_stream(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "selected_worker": kwargs["expected_worker"],
            "external_attempts": 1,
            "cost_usd": 0.0,
        }

    monkeypatch.setattr(benchmark, "_stream_request", fake_stream)
    result = benchmark._probe_key_readiness(
        gateway_url="http://gateway.invalid",
        openrouter_key="test-key",
        run_id="test-run",
        timeout_seconds=1.0,
    )

    assert result["selected_worker"] == "worker-a"
    assert len(calls) == benchmark.KEY_READINESS_REQUESTS
    assert calls[0]["path"] == "direct"
    assert calls[0]["expected_worker"] == "worker-a"
    assert calls[0]["max_completion_tokens"] == 1
    assert calls[0]["maximum_attempts"] == benchmark.MAX_DATA_PLANE_ATTEMPTS
    assert calls[0]["retryable_status_codes"] == frozenset({404, 429, 503})


def test_agentic_preflight_proves_all_endpoints_without_persisting_request_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preflight,
        "_probe_key_readiness",
        lambda **_kwargs: {
            "response_model": benchmark.WORKERS["worker-a"],
            "provider": benchmark.PROVIDER_NAMES["worker-a"],
            "completion_tokens": 1,
            "external_attempts": 1,
            "cost_usd": 0.001,
        },
    )
    monkeypatch.setattr(
        preflight,
        "_probe_endpoints",
        lambda **_kwargs: [
            {
                "response_model": model,
                "provider": benchmark.PROVIDER_NAMES[worker],
                "completion_tokens": EXPECTED_MAX_COMPLETION_TOKENS,
                "external_attempts": 1,
                "cost_usd": 0.001,
            }
            for worker, model in benchmark.WORKERS.items()
        ],
    )

    report = preflight.run_preflight(
        gateway_url="http://gateway.invalid",
        openrouter_key="private-key",
        run_id="public-preflight",
        timeout_seconds=1.0,
    )

    assert report["schema_version"] == preflight.REPORT_SCHEMA
    assert report["provider_requests"] == preflight.MAX_PROVIDER_REQUESTS
    assert report["external_attempts"] == preflight.MAX_PROVIDER_REQUESTS
    assert set(report["workers"]) == set(benchmark.WORKERS)
    assert report["performance_inference_admissible"] is False
    assert "private-key" not in preflight.encode_report(report, "private-key")


def test_agentic_benchmark_requires_reused_preflight_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {
        "schema_version": preflight.REPORT_SCHEMA,
        "status": "passed",
        "provider_requests": 4,
        "maximum_provider_requests": 4,
        "external_attempts": 4,
        "maximum_external_attempts": 11,
        "cost_usd": 0.004,
        "envoy_container_reused": True,
        "ephemeral_key_reused": True,
        "workers": {
            worker: {
                "model": model,
                "provider": benchmark.PROVIDER_NAMES[worker],
            }
            for worker, model in benchmark.WORKERS.items()
        },
    }
    monkeypatch.setenv(benchmark.TRANSPORT_PREFLIGHT_ENV, json.dumps(report))

    assert benchmark._transport_preflight_from_environment() == report

    report["envoy_container_reused"] = False
    monkeypatch.setenv(benchmark.TRANSPORT_PREFLIGHT_ENV, json.dumps(report))
    with pytest.raises(RuntimeError, match="contract diverged"):
        benchmark._transport_preflight_from_environment()


def test_agentic_discovery_reports_the_full_natural_mix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_coverage_request(**kwargs: Any) -> dict[str, Any]:
        index = int(kwargs["case"]["case_id"].rsplit("-", maxsplit=1)[1])
        return {
            "case_id": kwargs["case"]["case_id"],
            "selected_worker": "worker-c" if index % 3 == 0 else "worker-a",
        }

    monkeypatch.setattr(benchmark, "_coverage_request", fake_coverage_request)
    selected, coverage = benchmark._discover_cases(
        gateway_url="http://gateway.invalid",
        run_id="test-run",
        timeout_seconds=1.0,
    )

    assert len(coverage) == benchmark.MAX_COVERAGE_REQUESTS
    assert len(selected) == EXPECTED_SELECTED_CASES
    benchmark._bind_expected_workers(selected, coverage)
    assert (
        sum(case["expected_worker"] == "worker-a" for case in selected)
        == EXPECTED_SELECTED_CASES_PER_ACTIVE_WORKER
    )
    assert sum(case["expected_worker"] == "worker-b" for case in selected) == 0
    assert (
        sum(case["expected_worker"] == "worker-c" for case in selected)
        == EXPECTED_SELECTED_CASES_PER_ACTIVE_WORKER
    )
    selected_ids = {case["case_id"] for case in selected}
    assert len(selected_ids) == EXPECTED_SELECTED_CASES


def test_agentic_per_model_report_represents_zero_share() -> None:
    assert reporting._per_model_report([]) == {
        "requests": 0,
        "time_to_first_token": None,
        "end_to_end_latency": None,
        "envoy_upstream_service_time": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "external_attempts": 0,
        "retries": 0,
        "cost_usd": 0.0,
        "providers": [],
        "models": [],
    }


def test_agentic_paths_preserve_one_payload_and_change_only_routing() -> None:
    case = benchmark._candidate_case(0)
    direct = benchmark._request_payload(
        path="direct", case=case, expected_worker="worker-b"
    )
    static = benchmark._request_payload(
        path="gateway_static", case=case, expected_worker="worker-b"
    )
    arc = benchmark._request_payload(path="arc", case=case, expected_worker="worker-b")

    assert direct["model"] == "xiaomi/mimo-v2.5"
    assert static["model"] == "worker-b"
    assert arc["model"] == "auto"
    assert direct["messages"] == static["messages"] == arc["messages"]
    assert direct["tools"] == static["tools"] == arc["tools"]
    assert (
        direct["provider"]
        == static["provider"]
        == {
            "order": ["xiaomi"],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    )
    assert "provider" not in arc
    assert all(payload["stream"] is True for payload in (direct, static, arc))


def test_gateway_paths_never_forward_a_caller_authorization_value() -> None:
    direct = benchmark._request_headers(
        path="direct", openrouter_key="ephemeral-key", episode_id="direct"
    )
    static = benchmark._request_headers(
        path="gateway_static", openrouter_key="ignored", episode_id="static"
    )
    arc = benchmark._request_headers(
        path="arc", openrouter_key="ignored", episode_id="arc-episode"
    )

    assert direct["authorization"] == "Bearer ephemeral-key"
    assert "authorization" not in static
    assert "authorization" not in arc
    assert "x-rayline-episode-id" not in static
    assert arc["x-rayline-episode-id"] == "arc-episode"


def test_direct_path_retries_one_pre_response_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[dict[str, Any]] = []
    sleeps: list[float] = []

    def fake_once(**kwargs: Any) -> dict[str, Any]:
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise benchmark.OpenRouterHTTPError(
                endpoint="direct test",
                status_code=429,
                retry_after_seconds=0.25,
                error_type="rate_limit",
                provider_code="429",
            )
        return {"external_attempts": 1}

    monkeypatch.setattr(benchmark, "_stream_request_once", fake_once)
    monkeypatch.setattr(benchmark.time, "sleep", sleeps.append)
    result = benchmark._stream_request(
        path="direct",
        case={},
        expected_worker="worker-a",
        gateway_url="http://gateway.invalid",
        openrouter_key="test-key",
        episode_id="test-episode",
        timeout_seconds=1.0,
    )

    assert len(attempts) == benchmark.MAX_DATA_PLANE_ATTEMPTS
    assert result["external_attempts"] == benchmark.MAX_DATA_PLANE_ATTEMPTS
    assert sleeps == [0.25]


def test_key_readiness_can_retry_one_pre_response_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[dict[str, Any]] = []
    sleeps: list[float] = []

    def fake_once(**kwargs: Any) -> dict[str, Any]:
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise benchmark.OpenRouterHTTPError(
                endpoint="readiness test",
                status_code=404,
                retry_after_seconds=0.5,
                error_type="not_found",
                provider_code="404",
            )
        return {"external_attempts": 1}

    monkeypatch.setattr(benchmark, "_stream_request_once", fake_once)
    monkeypatch.setattr(benchmark.time, "sleep", sleeps.append)
    result = benchmark._stream_request(
        path="direct",
        case={},
        expected_worker="worker-a",
        gateway_url="http://gateway.invalid",
        openrouter_key="test-key",
        episode_id="test-readiness",
        timeout_seconds=1.0,
        max_completion_tokens=1,
        maximum_attempts=benchmark.MAX_DATA_PLANE_ATTEMPTS,
        retryable_status_codes=benchmark.KEY_READINESS_RETRYABLE_STATUS_CODES,
    )

    assert len(attempts) == benchmark.MAX_DATA_PLANE_ATTEMPTS
    assert result["external_attempts"] == benchmark.MAX_DATA_PLANE_ATTEMPTS
    assert sleeps == [0.5]


def test_gateway_readiness_retries_one_404_and_counts_both_wire_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[dict[str, Any]] = []
    sleeps: list[float] = []

    def fake_once(**kwargs: Any) -> dict[str, Any]:
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise benchmark.OpenRouterHTTPError(
                endpoint="gateway readiness",
                status_code=404,
                retry_after_seconds=0.5,
                error_type="",
                provider_code="404",
                error_category="no_endpoints",
                external_attempts=1,
            )
        return {"external_attempts": 1}

    monkeypatch.setattr(benchmark, "_stream_request_once", fake_once)
    monkeypatch.setattr(benchmark.time, "sleep", sleeps.append)
    result = benchmark._stream_request(
        path="gateway_static",
        case={},
        expected_worker="worker-a",
        gateway_url="http://gateway.invalid",
        openrouter_key="ignored",
        episode_id="gateway-readiness",
        timeout_seconds=1.0,
        maximum_attempts=benchmark.MAX_DATA_PLANE_ATTEMPTS,
        retryable_status_codes=(benchmark.ENDPOINT_REACHABILITY_RETRYABLE_STATUS_CODES),
    )

    assert len(attempts) == benchmark.MAX_DATA_PLANE_ATTEMPTS
    assert result["external_attempts"] == benchmark.MAX_DATA_PLANE_ATTEMPTS
    assert result["client_attempts"] == benchmark.MAX_DATA_PLANE_ATTEMPTS
    assert sleeps == [0.5]


def test_gateway_shape_diagnostic_interleaves_exact_bounded_probe_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_stream_request(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if (
            kwargs["path"] == "gateway_static"
            and kwargs["max_completion_tokens"] == EXPECTED_MAX_COMPLETION_TOKENS
        ):
            raise benchmark.OpenRouterHTTPError(
                endpoint="gateway diagnostic",
                status_code=404,
                retry_after_seconds=1.0,
                error_type="",
                provider_code="404",
                error_category="no_endpoints",
                external_attempts=1,
            )
        return {
            "response_model": "deepseek/deepseek-v4-flash",
            "provider": "Baidu",
            "completion_tokens": kwargs["max_completion_tokens"],
            "external_attempts": 1,
        }

    monkeypatch.setattr(gateway_diagnostic, "_stream_request", fake_stream_request)
    report = gateway_diagnostic.run_diagnostic(
        gateway_url="http://gateway.invalid",
        openrouter_key="private-key",
        run_id="public-diagnostic",
        timeout_seconds=1.0,
    )

    assert [(call["path"], call["max_completion_tokens"]) for call in calls] == [
        ("direct", 1),
        ("gateway_static", 1),
        ("direct", 96),
        ("gateway_static", 96),
        ("direct", 96),
        ("gateway_static", 96),
    ]
    assert all(call["maximum_attempts"] == 1 for call in calls)
    assert report["provider_requests"] == EXPECTED_DIAGNOSTIC_REQUESTS
    assert report["successful_requests"] == EXPECTED_DIAGNOSTIC_SUCCESSES
    assert report["failed_requests"] == EXPECTED_DIAGNOSTIC_FAILURES
    assert {result.get("error_category") for result in report["results"]} == {
        None,
        "no_endpoints",
    }
    assert "private-key" not in json.dumps(report)


def test_gateway_prime_diagnostic_exercises_first_request_and_prime_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_stream_request(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "response_model": "deepseek/deepseek-v4-flash",
            "provider": "Baidu",
            "completion_tokens": kwargs["max_completion_tokens"],
            "external_attempts": 1,
        }

    monkeypatch.setattr(prime_diagnostic, "_stream_request", fake_stream_request)
    report = prime_diagnostic.run_diagnostic(
        gateway_url="http://gateway.invalid",
        openrouter_key="private-key",
        run_id="public-prime-diagnostic",
        timeout_seconds=1.0,
    )

    assert [(call["path"], call["max_completion_tokens"]) for call in calls] == [
        ("gateway_static", EXPECTED_MAX_COMPLETION_TOKENS),
        ("gateway_static", 1),
        ("gateway_static", EXPECTED_MAX_COMPLETION_TOKENS),
        ("direct", EXPECTED_MAX_COMPLETION_TOKENS),
    ]
    assert all(call["maximum_attempts"] == 1 for call in calls)
    assert report["provider_requests"] == len(prime_diagnostic.PROBES)
    assert report["successful_requests"] == len(prime_diagnostic.PROBES)
    assert report["failed_requests"] == 0
    assert "private-key" not in json.dumps(report)


def test_agentic_launcher_pins_and_restores_one_encoder_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    updates: list[dict[str, int]] = []

    class FakeInstance:
        def update_autoscaler(self, **kwargs: int) -> None:
            updates.append(kwargs)

    class FakeClass:
        def __call__(self) -> FakeInstance:
            return FakeInstance()

    class FakeClsAPI:
        @staticmethod
        def from_name(app_name: str, class_name: str) -> FakeClass:
            assert app_name == launcher.ENCODER_APP_NAME
            assert class_name == launcher.ENCODER_CLASS_NAME
            return FakeClass()

    monkeypatch.setattr(launcher.modal, "Cls", FakeClsAPI, raising=False)
    state = launcher.RuntimeState(environment={})

    launcher._pin_encoder_singleton(state)
    assert state.encoder_autoscaler_pinned is True
    launcher._restore_encoder_scale_to_zero(state)
    assert state.encoder_autoscaler_pinned is False
    assert updates == [
        {
            "min_containers": 1,
            "max_containers": 1,
            "buffer_containers": 0,
            "scaledown_window": 300,
        },
        {
            "min_containers": 0,
            "max_containers": 1,
            "buffer_containers": 0,
            "scaledown_window": 300,
        },
    ]


def test_agentic_launcher_reuses_envoy_and_key_across_encoder_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    class FakeToken:
        token_id = "modal-token-id"
        token_secret = "modal-token-secret"

    class FakeManager:
        @staticmethod
        def create() -> FakeToken:
            return FakeToken()

    state = launcher.RuntimeState(
        environment={"phase": "preflight"},
        ephemeral_key="ephemeral-key",
        transport_preflight={
            "schema_version": preflight.REPORT_SCHEMA,
            "status": "passed",
            "provider_requests": 4,
            "maximum_provider_requests": 4,
            "external_attempts": 4,
            "maximum_external_attempts": 11,
            "cost_usd": 0.0,
        },
    )
    monkeypatch.setattr(
        launcher,
        "_compose_service_container_id",
        lambda _packet, _environment, service: (
            "stable-envoy-container" if service == "envoy" else ""
        ),
    )
    monkeypatch.setattr(
        launcher,
        "_pin_encoder_singleton",
        lambda runtime_state: setattr(runtime_state, "encoder_autoscaler_pinned", True),
    )
    monkeypatch.setattr(launcher, "_wait_protected_encoder", lambda _token: None)
    monkeypatch.setattr(launcher, "_wait_http", lambda _url: None)
    monkeypatch.setattr(launcher, "_wait_arc_component_ready", lambda _url: None)

    def fake_run(command: list[str], **_kwargs: Any) -> Any:
        commands.append(command)
        return types.SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(launcher, "_run", fake_run)
    packet = launcher.PACKETS["agentic"]

    launcher._activate_protected_encoder(
        packet=packet,
        manager=FakeManager(),
        state=state,
    )

    assert any(
        command[-5:]
        == [
            "up",
            "--detach",
            "--no-deps",
            "--force-recreate",
            "router",
        ]
        for command in commands
    )
    assert state.environment["OPENROUTER_EPHEMERAL_API_KEY"] == "ephemeral-key"
    assert state.environment["RAYLINE_ARC_E2E_ENCODER_BASE_URL"] == (
        f"https://{launcher.ENCODER_HOST}"
    )
    persisted = json.loads(state.environment[launcher.TRANSPORT_PREFLIGHT_ENV])
    assert persisted["envoy_container_reused"] is True
    assert persisted["ephemeral_key_reused"] is True


def test_agentic_compose_config_and_launcher_are_source_bounded() -> None:
    config = (DEPLOY_DIR / "config-openrouter-agentic.yaml").read_text()
    override = (DEPLOY_DIR / "compose-openrouter-agentic.yaml").read_text()
    dockerfile = (SCRIPT_DIR / "Dockerfile").read_text()

    for worker, model in benchmark.WORKERS.items():
        assert f"name: {worker}" in config
        assert f"provider_model_id: {model}" in config
    assert "public-rayline-arc-openrouter-agentic-v1" in config
    assert "openrouter_agentic_artifact_fixture.py" in override
    assert "openrouter_agentic_artifact_fixture.py" in dockerfile
    assert "moonshotai/kimi" not in config
    assert "z-ai/glm" not in config
    assert "fireworks/fast" not in config
    assert launcher.PACKETS["agentic"].key_limit_usd == EXPECTED_EPHEMERAL_KEY_LIMIT_USD
    assert launcher.PACKETS["agentic"].maximum_seconds == 30 * 60
    assert launcher.AGT008_PREREGISTRATION_COMMIT == ""
    assert launcher.AGT008_AUTHORIZATION_COMMIT == ""
    assert launcher.DGN003_PREREGISTRATION_COMMIT == ""
    assert launcher.DGN003_AUTHORIZATION_COMMIT == ""
    assert launcher.DGN004_PREREGISTRATION_COMMIT == ""
    assert launcher.DGN004_AUTHORIZATION_COMMIT == ""
    gateway_packet = launcher.PACKETS["gateway-shape"]
    assert gateway_packet.key_limit_usd == EXPECTED_DIAGNOSTIC_KEY_LIMIT_USD
    assert gateway_packet.maximum_seconds == 5 * 60
    assert gateway_packet.protected_encoder is False
    gateway_environment = launcher._runtime_environment(
        openrouter_key="ephemeral-key",
        modal_key="",
        modal_secret="",
        packet=gateway_packet,
    )
    assert gateway_environment["RAYLINE_ARC_E2E_ENCODER_BASE_URL"] == (
        "http://fake-encoder:8080"
    )
    assert gateway_environment["RAYLINE_ARC_E2E_ENCODER_BUILD_ID"] == (
        "vllm@public-rayline-e2e-build"
    )
    agentic_packet = launcher.PACKETS["agentic"]
    assert agentic_packet.preflight_driver == (
        SCRIPT_DIR / "openrouter_agentic_preflight.py"
    )
    preflight_environment = launcher._runtime_environment(
        openrouter_key="ephemeral-key",
        modal_key="",
        modal_secret="",
        packet=agentic_packet,
        protected_encoder=False,
    )
    assert preflight_environment["RAYLINE_ARC_E2E_ENCODER_BASE_URL"] == (
        "http://fake-encoder:8080"
    )
    prime_packet = launcher.PACKETS["gateway-prime"]
    assert prime_packet.key_limit_usd == EXPECTED_DIAGNOSTIC_KEY_LIMIT_USD
    assert prime_packet.maximum_seconds == 5 * 60
    assert prime_packet.protected_encoder is False
    assert "source=public-synthetic" in launcher.PUBLIC_REQUEST_LOG_MARKERS
    benchmark_source = (SCRIPT_DIR / "openrouter_agentic_benchmark.py").read_text()
    reporting_source = (SCRIPT_DIR / "openrouter_agentic_reporting.py").read_text()
    assert '"rayline.arc.openrouter-agentic-benchmark.v4"' in reporting_source
    assert '"openrouter_key_readiness"' in reporting_source
    assert '"selected_case_counts_by_worker"' in reporting_source
    assert '"selected_cases": [' not in benchmark_source + reporting_source
    assert (
        "execute-paid-1000"
        not in (SCRIPT_DIR / "run_openrouter_fullstack.py").read_text()
    )
