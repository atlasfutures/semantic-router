# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

probe = importlib.import_module("modal_encoder_batch_probe")

EXPECTED_CONCURRENCY = 8
EXPECTED_MAX_REQUESTS = 8
EXPECTED_MIN_SCHEDULED_REQUESTS = 2
EXPECTED_CUMULATIVE_ENVELOPE_USD = 29.23016482
EXPECTED_BUDGET_CAP_USD = 40.0


def test_batch_probe_is_a_fixed_minimal_packet() -> None:
    assert probe.PROBE_CONCURRENCY == EXPECTED_CONCURRENCY
    assert probe.PROBE_WAVES == 1
    assert probe.MAX_POOLING_REQUESTS == EXPECTED_MAX_REQUESTS
    assert probe.MIN_SCHEDULED_REQUESTS == EXPECTED_MIN_SCHEDULED_REQUESTS
    assert pytest.approx(EXPECTED_CUMULATIVE_ENVELOPE_USD) == (
        probe.CUMULATIVE_BEFORE_USD + probe.MAX_RESOURCE_ENVELOPE_USD
    )
    assert EXPECTED_CUMULATIVE_ENVELOPE_USD < EXPECTED_BUDGET_CAP_USD


def test_batch_probe_rejects_non_fresh_metrics() -> None:
    metrics = {
        "coordinator": {"requests_started_total": 0},
        "engine": {
            "e2e_time_observations": 0,
            "requests_running": 0,
            "requests_waiting": 0,
            "requests_running_max": 0,
            "requests_waiting_max": 0,
            "requests_scheduled_max": 0,
            "scheduler_updates_total": 0,
        },
    }
    probe._validate_fresh_start(metrics)
    metrics["engine"]["requests_scheduled_max"] = 1
    with pytest.raises(RuntimeError, match="fresh scheduler peak"):
        probe._validate_fresh_start(metrics)


def test_batch_probe_launcher_is_bounded_and_fail_closed() -> None:
    launcher = (SCRIPT_DIR / "run_modal_encoder_batch_probe.py").read_text()
    source = (SCRIPT_DIR / "modal_encoder_batch_probe.py").read_text()

    assert "MAX_PROBE_SECONDS = 5 * 60" in launcher
    assert "REQUEST_TIMEOUT_SECONDS = 300.0" in launcher
    assert "BUDGET_CAP_USD = 40.0" in launcher
    assert "encoder batch probe requires renewed budget authority" in launcher
    assert "_emit_sanitized_result" in launcher
    assert "_stop_encoder_containers" in launcher
    assert "manager.delete(proxy_token.token_id)" in launcher
    assert "retry" not in source.lower()
    assert "execute-paid-1000" not in launcher
    assert "case" not in source
