# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
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

artifact = importlib.import_module("openrouter_artifact_fixture")
canary = importlib.import_module("openrouter_fullstack_canary")
if importlib.util.find_spec("modal") is None:
    sys.modules["modal"] = types.ModuleType("modal")
launcher = importlib.import_module("run_openrouter_fullstack")

EXPECTED_MAX_TOKENS = 8
EXPECTED_MAX_COVERAGE_REQUESTS = 24
EXPECTED_MAX_PROVIDER_REQUESTS = 31
EXPECTED_MAX_EXTERNAL_ATTEMPTS = 62
EXPECTED_RETRY_BASE_SECONDS = 2.0
EXPECTED_MAX_RETRY_DELAY_SECONDS = 30.0
EXPECTED_MAX_PROVIDER_COST_USD = 0.10
EXPECTED_EPHEMERAL_USAGE_USD = 0.01
EXPECTED_WORKER_COUNT = 3
EXPECTED_ENVOY_EXTERNAL_ATTEMPTS = 2
EXPECTED_READINESS_REQUEST_TIMEOUT_SECONDS = 2


def test_openrouter_artifact_is_three_arm_pinned_and_bounded(tmp_path: Path) -> None:
    artifact.generate(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    golden = json.loads((tmp_path / "head_golden.json").read_text())

    assert manifest["artifact_id"] == "public-rayline-arc-openrouter-luna-v2"
    assert manifest["architecture"]["pool"] == [
        "worker-a",
        "worker-b",
        "worker-c",
    ]
    assert [worker["model"] for worker in manifest["workers"]] == [
        "deepseek/deepseek-v4-flash",
        "openai/gpt-5.6-luna",
        "z-ai/glm-5.2",
    ]
    expected_providers = {
        "worker-a": ("fireworks", "Fireworks"),
        "worker-b": ("openai", "OpenAI"),
        "worker-c": ("fireworks", "Fireworks"),
    }
    for worker in manifest["workers"]:
        provider_slug, provider_name = expected_providers[worker["id"]]
        assert worker["openrouter_provider_slug"] == provider_slug
        assert worker["openrouter_provider_name"] == provider_name
        assert worker["openrouter_provider_order"] == [provider_slug]
        assert worker["openrouter_allow_fallbacks"] is False
        assert worker["openrouter_require_parameters"] is True
        assert worker["openrouter_max_retries"] == 1
        assert worker["openrouter_retry_base_seconds"] == EXPECTED_RETRY_BASE_SECONDS
        assert (
            worker["openrouter_retry_cap_seconds"] == EXPECTED_MAX_RETRY_DELAY_SECONDS
        )
        assert worker["max_completion_tokens"] == EXPECTED_MAX_TOKENS
        assert worker["reasoning_budget_tokens"] == 0
        assert worker["extra_body"] == {
            "reasoning": {"enabled": False, "effort": "none"}
        }
    assert "temperature" not in manifest["workers"][1]
    assert manifest["workers"][0]["temperature"] == 0
    assert manifest["workers"][2]["temperature"] == 0
    assert [case["selected_index"] for case in golden["cases"]] == [0, 1, 2]
    assert (tmp_path / "head.safetensors").stat().st_size > 0


def test_openrouter_canary_request_and_cost_bounds_are_small() -> None:
    assert canary.MAX_TOKENS == EXPECTED_MAX_TOKENS
    assert canary.MAX_COVERAGE_REQUESTS == EXPECTED_MAX_COVERAGE_REQUESTS
    assert canary.MAX_PROVIDER_REQUESTS == EXPECTED_MAX_PROVIDER_REQUESTS
    assert canary.MAX_DATA_PLANE_RETRIES_PER_ROUTED_REQUEST == 1
    assert canary.MAX_EXTERNAL_ATTEMPTS == EXPECTED_MAX_EXTERNAL_ATTEMPTS
    assert canary.REQUEST_PACING_SECONDS == 1.0
    assert canary.MAX_RETRY_DELAY_SECONDS == EXPECTED_MAX_RETRY_DELAY_SECONDS
    assert canary.MAX_REPORTED_PROVIDER_COST_USD == EXPECTED_MAX_PROVIDER_COST_USD
    assert canary.PROVIDER_SLUGS == {
        "worker-a": "fireworks",
        "worker-b": "openai",
        "worker-c": "fireworks",
    }
    assert canary.PROVIDER_NAMES == {
        "worker-a": "Fireworks",
        "worker-b": "OpenAI",
        "worker-c": "Fireworks",
    }
    assert canary.TEMPERATURES == {
        "worker-a": 0,
        "worker-b": None,
        "worker-c": 0,
    }


def test_luna_direct_request_uses_openai_pin_and_omits_temperature() -> None:
    request = canary._chat_request(
        model=canary.WORKERS["worker-b"],
        prompt="bounded public prompt",
        direct_openrouter=True,
        provider_slug=canary.PROVIDER_SLUGS["worker-b"],
        temperature=canary.TEMPERATURES["worker-b"],
    )

    assert request["model"] == "openai/gpt-5.6-luna"
    assert request["provider"] == {
        "order": ["openai"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    assert request["reasoning"] == {"enabled": False, "effort": "none"}
    assert "temperature" not in request


def test_chat_does_not_supply_a_client_owned_429_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    sleeps: list[float] = []

    def fake_chat_once(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        raise canary.OpenRouterHTTPError(
            endpoint="chat endpoint",
            status_code=429,
            retry_after_seconds=3.0,
            error_type="rate_limit_exceeded",
            provider_code="429",
        )

    monkeypatch.setattr(canary, "_chat_once", fake_chat_once)
    monkeypatch.setattr(canary.time, "sleep", sleeps.append)
    with pytest.raises(canary.OpenRouterHTTPError, match="HTTP 429"):
        canary._chat(
            base_url="http://gateway/v1",
            model="auto",
            prompt="private prompt",
            authorization="Bearer private-key",
            timeout_seconds=1,
            episode_id="stable-episode",
        )

    assert len(calls) == 1
    assert calls[0]["episode_id"] == "stable-episode"
    assert sleeps == []


def test_http_error_exposes_only_bounded_metadata_and_retry_delay() -> None:
    prompt = "private prompt that must not be emitted"
    credential = "Bearer private-key"
    error = canary._http_error(
        endpoint="chat endpoint",
        status_code=429,
        body=json.dumps(
            {
                "error": {
                    "message": f"{prompt} {credential}",
                    "metadata": {
                        "error_type": "rate_limit_exceeded",
                        "provider_code": "unsafe provider details",
                    },
                }
            }
        ).encode(),
        retry_after="999",
    )

    assert error.retriable is True
    assert error.error_type == "rate_limit_exceeded"
    assert error.provider_code == ""
    assert error.error_category == "other_json_error"
    assert error.external_attempts == 1
    assert error.retry_after_seconds == canary.MAX_RETRY_DELAY_SECONDS
    assert prompt not in str(error)
    assert credential not in str(error)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("No endpoints found for this request", "no_endpoints"),
        ("This endpoint does not support tools", "unsupported_parameters"),
        ("Model not available", "unavailable"),
        ("Route not found", "not_found"),
        ("Provider rate limit reached", "rate_limited"),
        ("Invalid API key", "authentication"),
        ("opaque provider failure", "other_json_error"),
    ],
)
def test_http_error_classifies_messages_without_retaining_them(
    message: str, expected: str
) -> None:
    error = canary._http_error(
        endpoint="chat endpoint",
        status_code=404,
        body=json.dumps({"error": {"message": message, "code": 404}}).encode(),
        retry_after=None,
        external_attempts=2,
    )

    assert error.error_category == expected
    assert error.provider_code == "404"
    assert error.external_attempts == EXPECTED_ENVOY_EXTERNAL_ATTEMPTS
    assert message not in str(error)


def test_chat_does_not_retry_non_transient_http_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def fake_chat_once(**_kwargs: Any) -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        raise canary.OpenRouterHTTPError(
            endpoint="chat endpoint",
            status_code=401,
            retry_after_seconds=1.0,
            error_type="",
            provider_code="",
        )

    monkeypatch.setattr(canary, "_chat_once", fake_chat_once)
    with pytest.raises(canary.OpenRouterHTTPError, match="HTTP 401"):
        canary._chat(
            base_url="http://gateway/v1",
            model="auto",
            prompt="private prompt",
            authorization="Bearer private-key",
            timeout_seconds=1,
        )

    assert attempts == 1


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, 1), ("", 1), ("2", 2), ("0", 1), ("invalid", 1)],
)
def test_attempt_count_is_read_from_envoy_response(
    value: str | None,
    expected: int,
) -> None:
    assert canary._attempt_count(value) == expected


def test_coverage_discovers_all_three_models_without_emitting_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = list(canary.CANDIDATE_PROMPTS[:3])
    workers = iter(canary.WORKERS)

    def fake_chat(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["prompt"] in prompts
        worker = next(workers)
        return {
            "selected_worker": worker,
            "response_model": canary.WORKERS[worker],
            "provider": "Fireworks",
            "latency_seconds": 1,
            "prompt_tokens": 4,
            "completion_tokens": 2,
            "cost_usd": 0.0001,
        }

    monkeypatch.setattr(canary, "_chat", fake_chat)
    selected, results = canary._cover_models(
        gateway_url="http://gateway",
        run_id="public-unit",
        timeout_seconds=1,
    )

    assert set(selected) == set(canary.WORKERS)
    assert len(results) == EXPECTED_WORKER_COUNT
    assert not any(prompt in json.dumps(results) for prompt in prompts)


def test_openrouter_compose_contract_routes_only_known_workers() -> None:
    config = (DEPLOY_DIR / "config-openrouter.yaml").read_text()
    envoy = (DEPLOY_DIR / "envoy-openrouter.yaml").read_text()
    override = (DEPLOY_DIR / "compose-openrouter.yaml").read_text()
    dockerfile = (SCRIPT_DIR / "Dockerfile").read_text()

    for worker, model in canary.WORKERS.items():
        assert f"name: {worker}" in config
        assert f"provider_model_id: {model}" in config
        assert f"exact: {worker}" in envoy
    # Semantic Router resolves the OpenAI-compatible base URL into this full
    # upstream path before Envoy clears and rematches the route cache.
    assert envoy.count("prefix: /api/v1/") == EXPECTED_WORKER_COUNT
    assert "prefix_rewrite:" not in envoy
    assert "host_rewrite_literal: openrouter.ai" in envoy
    assert config.count("provider: openai") == EXPECTED_WORKER_COUNT
    assert "provider: openrouter" not in config
    assert "fireworks/fast" not in config
    assert "moonshotai/kimi" not in config
    assert "artifact_revision: public-rayline-arc-openrouter-luna-v2" in config
    assert "prompt_per_1m: 0.1" in config
    assert "cached_input_per_1m: 0.01" in config
    assert "cache_write_per_1m: 0.125" in config
    assert "completion_per_1m: 0.6" in config
    assert "envoy.transport_sockets.tls" in envoy
    assert "retry_on: retriable-status-codes" in envoy
    assert "retriable_status_codes: [429, 503]" in envoy
    assert "retriable_request_headers:" not in envoy
    assert "include_attempt_count_in_response: true" in envoy
    assert "rate_limited_retry_back_off:" in envoy
    assert "openrouter_artifact_fixture.py" in override
    assert "openrouter_artifact_fixture.py" in dockerfile


def test_self_hosted_worker_routes_do_not_retry_backpressure() -> None:
    envoy = (DEPLOY_DIR / "envoy-real-workers.yaml").read_text()

    assert "retry_policy:" not in envoy
    assert "retry_on:" not in envoy


def test_openrouter_launcher_uses_ephemeral_limited_key_and_exact_cleanup() -> None:
    launcher_source = (SCRIPT_DIR / "run_openrouter_fullstack.py").read_text()
    encoder_runtime_source = (SCRIPT_DIR / "openrouter_encoder_runtime.py").read_text()
    key_source = (SCRIPT_DIR / "openrouter_key_management.py").read_text()

    assert "OPENROUTER_KEY_LIMIT_USD = 0.25" in launcher_source
    assert (
        'management_key = os.environ.get("OPENROUTER_MANAGEMENT_KEY", "")'
        in launcher_source
    )
    assert '"limit": key_limit_usd' in key_source
    assert "_delete_ephemeral_key(management_key, key_hash)" in launcher_source
    assert 'ENCODER_APP_ID = "ap-XtsWCBEWdw1ncu9Kv12Chj"' in launcher_source
    assert (
        'ENCODER_BUILD_ID = "vllm@9f5ea81ca0aa570aea46baf82311a1139c1267ca"'
        in launcher_source
    )
    assert (
        'ENCODER_DEPLOYMENT_SOURCE_COMMIT = "0e07fa25410adf2ec2fc8e087dd951436c6b6e0d"'
        in launcher_source
    )
    assert "1ff4ee4d7a22cc1d74c0cdb0352d3f76f5081b7201fa63e7f8f3dd10af246afd" in (
        launcher_source
    )
    assert '"container", "stop", container_id, "--yes"' in encoder_runtime_source
    assert "cleanup_encoder(" in launcher_source
    assert "manager.delete(proxy_token.token_id)" in launcher_source
    assert "_wait_arc_component_ready(METRICS_URL)" in launcher_source
    execution_source = launcher_source.split("def _execute_runtime(", maxsplit=1)[
        1
    ].split("def main() -> None:", maxsplit=1)[0]
    assert execution_source.index("_create_ephemeral_key(") < execution_source.index(
        "_run_transport_preflight("
    )
    assert execution_source.index("_run_transport_preflight(") < execution_source.index(
        "_activate_protected_encoder("
    )
    prepare_source = launcher_source.split("def _prepare_encoder_runtime(", maxsplit=1)[
        1
    ].split("def _collect_evidence_safely(", maxsplit=1)[0]
    assert prepare_source.index("verify_encoder_deployment(") < prepare_source.index(
        "encoder_containers("
    )
    launch_source = launcher_source.split("def _launch_packet(", maxsplit=1)[1].split(
        "def _raise_outcome(", maxsplit=1
    )[0]
    assert launch_source.index("_collect_evidence_safely(") < launch_source.index(
        "_cleanup_safely("
    )
    assert "execute-paid-1000" not in launcher_source


def test_openrouter_post_run_evidence_scans_logs_before_reading_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []
    packet = launcher.PACKETS["canary"]

    def fake_scan(
        received_packet: launcher.RunPacket,
        environment: dict[str, str],
        protected_values: tuple[str, ...],
    ) -> None:
        calls.append(("scan", (received_packet, environment, protected_values)))

    def fake_usage(management_key: str, key_hash: str) -> float:
        calls.append(("usage", (management_key, key_hash)))
        return EXPECTED_EPHEMERAL_USAGE_USD

    monkeypatch.setattr(launcher, "_scan_logs", fake_scan)
    monkeypatch.setattr(launcher, "_ephemeral_key_usage", fake_usage)
    usage = launcher._collect_post_run_evidence(
        environment={"public": "value"},
        protected_values=("protected",),
        management_key="management-key",
        key_hash="key-hash",
        packet=packet,
    )

    assert usage == EXPECTED_EPHEMERAL_USAGE_USD
    assert calls == [
        ("scan", (packet, {"public": "value"}, ("protected",))),
        ("usage", ("management-key", "key-hash")),
    ]


def test_openrouter_post_run_evidence_allows_no_key_before_encoder_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_scan(*_args: Any) -> None:
        calls.append("scan")

    monkeypatch.setattr(launcher, "_scan_logs", fake_scan)
    usage = launcher._collect_post_run_evidence(
        environment={},
        protected_values=(),
        management_key="management-key",
        key_hash="",
        packet=launcher.PACKETS["agentic"],
    )

    assert usage == 0.0
    assert calls == ["scan"]


def test_openrouter_launcher_parses_arc_component_readiness() -> None:
    metric = launcher.ARC_READY_METRIC

    assert launcher._arc_component_ready(f"{metric} 1\n") is True
    assert launcher._arc_component_ready(f"{metric} 0\n") is False
    assert launcher._arc_component_ready("unrelated_metric 1\n") is None


def test_arc_component_readiness_waits_through_transient_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric = launcher.ARC_READY_METRIC
    bodies = iter((f"{metric} 0\n", f"{metric} 1\n"))
    calls: list[str] = []

    class FakeResponse:
        status = launcher.HTTP_OK

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return next(bodies).encode()

    def fake_urlopen(url: str, timeout: float) -> FakeResponse:
        assert timeout == EXPECTED_READINESS_REQUEST_TIMEOUT_SECONDS
        calls.append(url)
        return FakeResponse()

    monkeypatch.setattr(launcher.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(launcher.time, "sleep", lambda _seconds: None)

    launcher._wait_arc_component_ready("http://metrics.invalid")

    assert calls == ["http://metrics.invalid", "http://metrics.invalid"]
