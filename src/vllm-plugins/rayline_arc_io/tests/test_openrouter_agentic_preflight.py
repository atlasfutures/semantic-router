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
sys.path.insert(0, str(SCRIPT_DIR))

benchmark = importlib.import_module("openrouter_agentic_benchmark")
preflight = importlib.import_module("openrouter_agentic_preflight")
if "modal" not in sys.modules and importlib.util.find_spec("modal") is None:
    modal_stub = types.ModuleType("modal")
    modal_stub.__spec__ = importlib.machinery.ModuleSpec("modal", loader=None)
    sys.modules["modal"] = modal_stub
launcher = importlib.import_module("run_openrouter_fullstack")

EXPECTED_MAX_COMPLETION_TOKENS = 96
EXPECTED_HTTP_NOT_FOUND = 404
EXPECTED_COMPLETED_BEFORE_FAILURE = 2
EXPECTED_FAILURE_ATTEMPTS = 4


def test_agentic_preflight_proves_all_endpoints_without_request_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preflight,
        "_probe_key_readiness",
        lambda **_kwargs: {
            "response_model": benchmark.WORKERS["worker-a"],
            "provider": benchmark.PROVIDER_NAMES["worker-a"][0],
            "completion_tokens": 1,
            "external_attempts": 1,
            "cost_usd": 0.001,
        },
    )
    monkeypatch.setattr(
        preflight,
        "_probe_endpoint",
        lambda **kwargs: {
            "response_model": benchmark.WORKERS[kwargs["worker"]],
            "provider": benchmark.PROVIDER_NAMES[kwargs["worker"]][0],
            "completion_tokens": EXPECTED_MAX_COMPLETION_TOKENS,
            "external_attempts": 1,
            "cost_usd": 0.001,
        },
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


def test_agentic_preflight_returns_bounded_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        preflight,
        "_probe_key_readiness",
        lambda **_kwargs: {
            "response_model": benchmark.WORKERS["worker-a"],
            "provider": benchmark.PROVIDER_NAMES["worker-a"][0],
            "completion_tokens": 1,
            "external_attempts": 1,
            "cost_usd": 0.001,
        },
    )

    def fake_endpoint(**kwargs: Any) -> dict[str, Any]:
        if kwargs["worker"] == "worker-b":
            raise benchmark.OpenRouterHTTPError(
                endpoint="static endpoint",
                status_code=EXPECTED_HTTP_NOT_FOUND,
                retry_after_seconds=1.0,
                error_type="not_found",
                provider_code="404",
                error_category="no_endpoints",
                external_attempts=2,
            )
        return {
            "response_model": benchmark.WORKERS[kwargs["worker"]],
            "provider": benchmark.PROVIDER_NAMES[kwargs["worker"]][0],
            "completion_tokens": EXPECTED_MAX_COMPLETION_TOKENS,
            "external_attempts": 1,
            "cost_usd": 0.001,
        }

    monkeypatch.setattr(preflight, "_probe_endpoint", fake_endpoint)
    report = preflight.run_preflight(
        gateway_url="http://gateway.invalid",
        openrouter_key="private-key",
        run_id="public-preflight-failure",
        timeout_seconds=1.0,
    )

    assert report["status"] == "failed"
    assert report["failed_stage"] == "static_endpoint_reachability"
    assert report["failed_worker"] == "worker-b"
    assert report["http_status"] == EXPECTED_HTTP_NOT_FOUND
    assert report["error_category"] == "no_endpoints"
    assert report["provider_code"] == "404"
    assert report["completed_provider_requests"] == EXPECTED_COMPLETED_BEFORE_FAILURE
    assert report["external_attempts"] == EXPECTED_FAILURE_ATTEMPTS
    assert launcher._validate_transport_preflight(report) == report
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
                "provider": benchmark.PROVIDER_NAMES[worker][0],
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
