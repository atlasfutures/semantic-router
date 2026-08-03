# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

contract = importlib.import_module("rayline_replica_stop_contract")
launcher = importlib.import_module("rayline_replica_stop_launcher")


def test_unregistered_replica_stop_stops_before_side_effects(tmp_path: Path) -> None:
    args = launcher.argparse.Namespace(
        run_id="unregistered-run",
        pathfinder_root=tmp_path,
        packet_dir=tmp_path / "packet",
        runtime_dir=tmp_path / "runtime",
        router_image="unused",
    )
    with pytest.raises(ValueError, match="no Rayline replica-stop experiment"):
        launcher._preflight(args)
    assert list(tmp_path.iterdir()) == []


def test_exact_stop_boundary_requires_zero_outage_and_one_survivor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher.scaleout, "_stop_app", lambda *_args: True)
    monkeypatch.setattr(
        launcher.scaleout,
        "_named_encoder_apps",
        lambda _context: [
            {
                "description": contract.UNAVAILABLE_APP_NAME,
                "state": "stopped",
                "tasks": "0",
            },
            {
                "description": launcher.scaleout.ENCODER_APP_NAMES[1],
                "state": "deployed",
                "tasks": "1",
            },
        ],
    )
    monkeypatch.setattr(
        launcher.scaleout,
        "_named_encoder_containers",
        lambda _context: [{"app_name": launcher.scaleout.ENCODER_APP_NAMES[1]}],
    )
    receipt = launcher._stop_boundary(
        SimpleNamespace(), SimpleNamespace(service_environment={})
    )
    assert receipt["action"] == "stop_exact_app"
    assert receipt["unavailable_containers_remaining"] == 0
    assert receipt["survivor_containers_running"] == 1


def test_launcher_has_no_provider_or_qualification_path() -> None:
    source = (SCRIPT_DIR / "rayline_replica_stop_launcher.py").read_text()
    assert "openrouter" not in source.lower()
    assert "execute-paid-1000" not in source
    assert '"release_qualification_1000_executed": False' in source
    assert contract.LAUNCHABLE_CONTRACT is None
