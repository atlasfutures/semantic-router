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
EXPECTED_MAX_PROVIDER_COST_USD = 0.10
EXPECTED_WORKER_COUNT = 3


def test_openrouter_artifact_is_three_arm_pinned_and_bounded(tmp_path: Path) -> None:
    artifact.generate(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    golden = json.loads((tmp_path / "head_golden.json").read_text())

    assert manifest["artifact_id"] == "public-rayline-arc-openrouter-v1"
    assert manifest["architecture"]["pool"] == [
        "worker-a",
        "worker-b",
        "worker-c",
    ]
    assert [worker["model"] for worker in manifest["workers"]] == [
        "deepseek/deepseek-v4-flash",
        "moonshotai/kimi-k3",
        "z-ai/glm-5.2",
    ]
    for worker in manifest["workers"]:
        assert worker["openrouter_provider_order"] == ["fireworks"]
        assert worker["openrouter_allow_fallbacks"] is False
        assert worker["openrouter_require_parameters"] is True
        assert worker["max_completion_tokens"] == EXPECTED_MAX_TOKENS
        assert worker["reasoning_budget_tokens"] == 0
        assert worker["extra_body"] == {
            "reasoning": {"enabled": False, "effort": "none"}
        }
    assert [case["selected_index"] for case in golden["cases"]] == [0, 1, 2]
    assert (tmp_path / "head.safetensors").stat().st_size > 0


def test_openrouter_canary_request_and_cost_bounds_are_small() -> None:
    assert canary.MAX_TOKENS == EXPECTED_MAX_TOKENS
    assert canary.MAX_COVERAGE_REQUESTS == EXPECTED_MAX_COVERAGE_REQUESTS
    assert canary.MAX_PROVIDER_REQUESTS == EXPECTED_MAX_PROVIDER_REQUESTS
    assert canary.MAX_REPORTED_PROVIDER_COST_USD == EXPECTED_MAX_PROVIDER_COST_USD
    assert canary.PROVIDER_SLUG == "fireworks"
    assert canary.PROVIDER_NAME == "Fireworks"


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
    assert "envoy.transport_sockets.tls" in envoy
    assert "openrouter_artifact_fixture.py" in override
    assert "openrouter_artifact_fixture.py" in dockerfile


def test_openrouter_launcher_uses_ephemeral_limited_key_and_exact_cleanup() -> None:
    launcher_source = (SCRIPT_DIR / "run_openrouter_fullstack.py").read_text()

    assert "OPENROUTER_KEY_LIMIT_USD = 0.25" in launcher_source
    assert (
        'management_key = os.environ.get("OPENROUTER_MANAGEMENT_KEY", "")'
        in launcher_source
    )
    assert '"limit": OPENROUTER_KEY_LIMIT_USD' in launcher_source
    assert "_delete_ephemeral_key(management_key, key_hash)" in launcher_source
    assert 'ENCODER_APP_ID = "ap-rs3UkEn5XUnWjrZOXYbkuB"' in launcher_source
    assert '"container", "stop", container_id, "--yes"' in launcher_source
    assert "manager.delete(proxy_token.token_id)" in launcher_source
    assert "_wait_arc_component_ready(METRICS_URL)" in launcher_source
    assert "execute-paid-1000" not in launcher_source


def test_openrouter_launcher_rejects_failed_arc_component_readiness() -> None:
    metric = launcher.ARC_READY_METRIC

    assert launcher._arc_component_ready(f"{metric} 1\n") is True
    assert launcher._arc_component_ready(f"{metric} 0\n") is False
    assert launcher._arc_component_ready("unrelated_metric 1\n") is None
