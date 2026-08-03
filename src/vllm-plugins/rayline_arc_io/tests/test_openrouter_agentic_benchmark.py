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
workload = importlib.import_module("openrouter_agentic_workload")
if importlib.util.find_spec("modal") is None:
    modal_stub = types.ModuleType("modal")
    modal_stub.__spec__ = importlib.machinery.ModuleSpec("modal", loader=None)
    sys.modules["modal"] = modal_stub
launcher = importlib.import_module("run_openrouter_fullstack")
EXPECTED_MAX_COMPLETION_TOKENS = 96
EXPECTED_MAX_MEASURED_REQUESTS = 72
EXPECTED_MAX_PROVIDER_REQUESTS = 96
EXPECTED_PROVIDER_COST_LIMIT_USD = 0.50
EXPECTED_MIN_TOOL_RESULT_CHARACTERS = 1_000
EXPECTED_SELECTED_CASES = 6
EXPECTED_EPHEMERAL_KEY_LIMIT_USD = 0.75


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


def test_agentic_case_selection_requires_two_per_worker_and_all_shapes() -> None:
    candidates = {
        "worker-a": [benchmark._candidate_case(0), benchmark._candidate_case(3)],
        "worker-b": [benchmark._candidate_case(1), benchmark._candidate_case(4)],
        "worker-c": [benchmark._candidate_case(2), benchmark._candidate_case(5)],
    }

    selected = benchmark._choose_cases(candidates)

    assert selected is not None
    assert len(selected) == EXPECTED_SELECTED_CASES
    assert {case["scenario"] for case in selected} == set(benchmark.SCENARIOS)


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
    assert (
        launcher.AGENTIC_PREREGISTRATION_COMMIT
        == "59334dbd7b92e39df440ad581f049ceb87323153"
    )
    assert launcher.AGENTIC_AUTHORIZATION_COMMIT == ""
    assert "source=public-synthetic" in launcher.PUBLIC_REQUEST_LOG_MARKERS
    benchmark_source = (SCRIPT_DIR / "openrouter_agentic_benchmark.py").read_text()
    assert '"selected_case_counts_by_worker"' in benchmark_source
    assert '"selected_cases": [' not in benchmark_source
    assert (
        "execute-paid-1000"
        not in (SCRIPT_DIR / "run_openrouter_fullstack.py").read_text()
    )
