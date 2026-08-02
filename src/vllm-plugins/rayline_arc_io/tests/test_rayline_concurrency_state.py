# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

state = importlib.import_module("rayline_concurrency_state")


class FakeRequester:
    def __init__(self, *, missing: set[str] | None = None) -> None:
        self.missing = missing or set()
        self.deleted: list[str] = []

    def __call__(self, method: str, path: str) -> dict[str, Any]:
        if method == "GET" and path == "/health":
            return {
                "status": "ok",
                "resident_sessions": 0,
                "resident_tokens": 0,
            }
        assert method == "DELETE"
        self.deleted.append(path)
        return {"closed": path not in self.missing}


def _path(run_id: str, episode_id: str) -> str:
    digest = state.hash_episode_id(f"{run_id}:{episode_id}")
    return f"/v1/rayline/arc/session/{digest}"


def test_cell_cleanup_namespaces_and_closes_exact_sessions() -> None:
    requester = FakeRequester(missing={_path("perf017:c1:arc", "warmup")})

    receipt = state.close_cell_sessions(
        requester=requester,
        probe_run_id="perf017:c1:arc",
        measured_episode_ids=("measured-a", "measured-b"),
        warmup_episode_ids=("warmup",),
        require_measured_present=True,
    )

    assert requester.deleted == [
        _path("perf017:c1:arc", "measured-a"),
        _path("perf017:c1:arc", "measured-b"),
        _path("perf017:c1:arc", "warmup"),
    ]
    assert receipt["measured_sessions_closed"] == len(("measured-a", "measured-b"))
    assert receipt["warmup_sessions_missing"] == 1
    assert receipt["resident_sessions_after_cleanup"] == 0


def test_completed_measured_episode_must_be_resident() -> None:
    missing = {_path("perf017:c4:arc", "measured-a")}

    with pytest.raises(state.StateResetError, match="measured episode"):
        state.close_cell_sessions(
            requester=FakeRequester(missing=missing),
            probe_run_id="perf017:c4:arc",
            measured_episode_ids=("measured-a",),
            warmup_episode_ids=("warmup",),
            require_measured_present=True,
        )


def test_failed_arm_cleanup_allows_absent_measured_sessions() -> None:
    missing = {_path("perf017:c8:arc", "measured-a")}

    receipt = state.close_cell_sessions(
        requester=FakeRequester(missing=missing),
        probe_run_id="perf017:c8:arc",
        measured_episode_ids=("measured-a",),
        warmup_episode_ids=("warmup",),
        require_measured_present=False,
    )

    assert receipt["measured_sessions_missing"] == 1


def test_non_empty_health_fails_closed() -> None:
    def requester(_method: str, _path: str) -> dict[str, Any]:
        return {"status": "ok", "resident_sessions": 1, "resident_tokens": 10}

    with pytest.raises(state.StateResetError, match="retained state"):
        state.assert_encoder_empty(requester)


def test_protected_client_repr_redacts_credentials() -> None:
    client = state.ProtectedEncoderClient(
        "https://public.example",
        "private-key",
        "private-secret",
    )

    assert "private-key" not in repr(client)
    assert "private-secret" not in repr(client)
