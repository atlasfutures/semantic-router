# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

telemetry = importlib.import_module("rayline_three_arm_telemetry")


def _metrics(*, appended: int = 6) -> str:
    lines = [
        'llm_rayline_arc_component_ready{component="artifact_head_encoder"} 1',
        'llm_rayline_arc_session_actions_total{action="created"} 2',
        f'llm_rayline_arc_session_actions_total{{action="appended"}} {appended}',
        "llm_rayline_arc_cache_miss_tokens_sum 0",
        "llm_rayline_arc_cache_miss_tokens_count 8",
    ]
    for kind, total in {
        "full": 2000,
        "serialized": 2000,
        "retained": 800,
        "appended": 1200,
        "cached": 800,
        "truncated": 0,
    }.items():
        lines.extend(
            [
                f'llm_rayline_arc_tokens_sum{{kind="{kind}"}} {total}',
                f'llm_rayline_arc_tokens_count{{kind="{kind}"}} 8',
            ]
        )
    return "\n".join(lines) + "\n"


def test_metrics_snapshot_is_aggregate_and_reconciled() -> None:
    receipt = telemetry.parse_arc_metrics(_metrics())

    assert receipt["schema_version"] == telemetry.TELEMETRY_SCHEMA
    assert receipt["component_ready"] == 1
    assert receipt["session_actions"] == {
        "created": 2,
        "appended": 6,
        "rebuilt": 0,
        "reused": 0,
    }
    assert receipt["tokens"]["retained"] == {"sum": 800, "count": 8}
    assert receipt["cache_miss_tokens"] == {"sum": 0, "count": 8}


def test_action_and_request_count_mismatch_fails_closed() -> None:
    with pytest.raises(telemetry.TelemetryError, match="do not reconcile"):
        telemetry.parse_arc_metrics(_metrics(appended=5))
