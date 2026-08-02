# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

launcher = importlib.import_module("rayline_scaleout_launcher")


def test_unregistered_scaleout_stops_before_side_effects(tmp_path: Path) -> None:
    args = launcher.argparse.Namespace(
        run_id="unregistered-run",
        pathfinder_root=tmp_path,
        packet_dir=tmp_path / "packet",
        runtime_dir=tmp_path / "runtime",
        router_image="unused",
    )

    with pytest.raises(ValueError, match="launcher only permits preregistered"):
        launcher._preflight(args)

    assert list(tmp_path.iterdir()) == []


def test_launcher_has_frozen_arms_apps_and_no_provider_or_qualification_path() -> None:
    source = (SCRIPT_DIR / "rayline_scaleout_launcher.py").read_text()

    assert launcher.SCALEOUT_ARMS == ("arc_single", "arc_dual_affinity")
    assert launcher.ENCODER_APP_NAMES == (
        "rayline-arc-session-encoder-a",
        "rayline-arc-session-encoder-b",
    )
    assert "openrouter" not in source.lower()
    assert "execute-paid-1000" not in source
    assert '"release_qualification_1000_executed": False' in source
