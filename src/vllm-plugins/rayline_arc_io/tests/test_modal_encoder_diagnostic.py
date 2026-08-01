# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

diagnostic = importlib.import_module("modal_encoder_diagnostic")
reporting = importlib.import_module("modal_encoder_diagnostic_report")

EXPECTED_MEASURED_REQUESTS = 60
EXPECTED_SETUP_REQUESTS = 30
EXPECTED_MAX_REQUESTS = 92
EXPECTED_CUMULATIVE_ENVELOPE_USD = 26.73054802
EXPECTED_BUDGET_CAP_USD = 40.0
EXPECTED_WAVES_PER_LEVEL = 2


def test_encoder_diagnostic_packet_is_fixed_and_h100_only() -> None:
    assert diagnostic.CONCURRENCY_LEVELS == (1, 2, 4, 8)
    assert diagnostic.WAVES_PER_LEVEL == EXPECTED_WAVES_PER_LEVEL
    assert diagnostic.PHASES == ("create", "append")
    assert diagnostic.MEASURED_REQUESTS == EXPECTED_MEASURED_REQUESTS
    assert diagnostic.APPEND_SETUP_REQUESTS == EXPECTED_SETUP_REQUESTS
    assert diagnostic.MAX_POOLING_REQUESTS == EXPECTED_MAX_REQUESTS
    assert pytest.approx(EXPECTED_CUMULATIVE_ENVELOPE_USD) == (
        diagnostic.CUMULATIVE_BEFORE_USD + diagnostic.MAX_RESOURCE_ENVELOPE_USD
    )
    assert EXPECTED_CUMULATIVE_ENVELOPE_USD < EXPECTED_BUDGET_CAP_USD
    assert "case" not in (SCRIPT_DIR / "modal_encoder_diagnostic.py").read_text()
    assert (
        "SAMPLER_INTERVAL_SECONDS"
        not in (SCRIPT_DIR / "modal_encoder_diagnostic.py").read_text()
    )


def test_metric_delta_rejects_missing_or_decreasing_values() -> None:
    before = {"coordinator": {"requests_started_total": 2}}
    after = {"coordinator": {"requests_started_total": 4}}

    assert diagnostic._metric_delta(
        before,
        after,
        "coordinator",
        ("requests_started_total",),
    ) == {"requests_started_total": 2.0}

    after["coordinator"]["requests_started_total"] = 1
    with pytest.raises(RuntimeError, match="moved backwards"):
        diagnostic._metric_delta(
            before,
            after,
            "coordinator",
            ("requests_started_total",),
        )


def test_append_metrics_require_exact_synchronous_observations(monkeypatch) -> None:
    before = {
        "engine": dict.fromkeys(diagnostic.ENGINE_CUMULATIVE_FIELDS, 0.0),
    }
    complete = {
        "engine": dict.fromkeys(diagnostic.ENGINE_CUMULATIVE_FIELDS, 0.0),
    }
    for field in (
        "queue_time_observations",
        "inference_time_observations",
        "e2e_time_observations",
        "prompt_token_observations",
    ):
        complete["engine"][field] = 1.0
    monkeypatch.setattr(diagnostic, "_read_metrics", lambda _client: complete)

    assert diagnostic._read_append_metrics(object(), before, 1) is complete


def test_cross_episode_gate_requires_exact_counts_and_zero_lock_contention() -> None:
    requests = 4
    coordinator = dict.fromkeys(diagnostic.COORDINATOR_CUMULATIVE_FIELDS, 0.0)
    coordinator.update(
        {
            "tokenization_calls_total": requests,
            "requests_started_total": requests,
            "requests_succeeded_total": requests,
            "backend_calls_started_total": requests,
            "backend_calls_succeeded_total": requests,
            "backend_appended_tokens_total": 128,
        }
    )
    engine = dict.fromkeys(diagnostic.ENGINE_CUMULATIVE_FIELDS, 0.0)
    for field in (
        "queue_time_observations",
        "inference_time_observations",
        "e2e_time_observations",
        "prompt_token_observations",
    ):
        engine[field] = requests

    diagnostic._validate_phase_deltas(
        coordinator=coordinator,
        engine=engine,
        requests=requests,
    )

    coordinator["session_lock_contentions_total"] = 1
    with pytest.raises(RuntimeError, match="same-session serialization"):
        diagnostic._validate_phase_deltas(
            coordinator=coordinator,
            engine=engine,
            requests=requests,
        )


def test_capacity_gate_uses_server_retained_scheduler_peak() -> None:
    coordinator = {"backend_inflight_max": max(diagnostic.CONCURRENCY_LEVELS)}
    engine = {
        "requests_scheduled_max": diagnostic.MIN_ENGINE_CONCURRENT_REQUESTS,
        "scheduler_updates_total": 1,
    }

    reporting.validate_capacity_peaks(
        coordinator,
        engine,
        minimum_backend_inflight=max(diagnostic.CONCURRENCY_LEVELS),
        minimum_scheduled_requests=diagnostic.MIN_ENGINE_CONCURRENT_REQUESTS,
    )

    engine["requests_scheduled_max"] = diagnostic.MIN_ENGINE_CONCURRENT_REQUESTS - 1
    with pytest.raises(RuntimeError, match="never scheduled concurrent"):
        reporting.validate_capacity_peaks(
            coordinator,
            engine,
            minimum_backend_inflight=max(diagnostic.CONCURRENCY_LEVELS),
            minimum_scheduled_requests=diagnostic.MIN_ENGINE_CONCURRENT_REQUESTS,
        )


def test_final_capacity_failure_emits_partial_aggregate_report(capsys) -> None:
    report = {"status": "passed", "results": [{"requests": 60}]}
    coordinator = {"backend_inflight_max": max(diagnostic.CONCURRENCY_LEVELS)}
    engine = {"requests_scheduled_max": 1, "scheduler_updates_total": 1}

    with pytest.raises(RuntimeError, match="never scheduled concurrent"):
        reporting.validate_or_emit_failure(
            report,
            coordinator,
            engine,
            minimum_backend_inflight=max(diagnostic.CONCURRENCY_LEVELS),
            minimum_scheduled_requests=diagnostic.MIN_ENGINE_CONCURRENT_REQUESTS,
        )

    emitted = json.loads(capsys.readouterr().out)
    assert emitted["status"] == "failed"
    assert emitted["results"] == [{"requests": 60}]
    assert emitted["failure"]["phase"] == "final_capacity_peak_validation"


def test_launcher_deploys_exact_service_and_cleans_only_encoder_containers() -> None:
    launcher = (SCRIPT_DIR / "run_modal_encoder_diagnostic.py").read_text()

    assert 'REQUIRED_MODAL_VERSION = "1.5.1"' in launcher
    assert "MAX_DIAGNOSTIC_SECONDS = 15 * 60" in launcher
    assert 'ENCODER_APP_ID = "ap-rs3UkEn5XUnWjrZOXYbkuB"' in launcher
    assert 'modal_command, "deploy", str(SERVICE)' in launcher
    assert "check=False" in launcher
    assert "_emit_sanitized_result" in launcher
    assert "raise SystemExit(result.returncode)" in launcher
    assert "BUDGET_CAP_USD = 40.0" in launcher
    assert "encoder diagnostic requires renewed budget authority" in launcher
    assert '"container", "stop", container_id, "--yes"' in launcher
    assert "manager.delete(proxy_token.token_id)" in launcher
    assert "execute-paid-1000" not in launcher
