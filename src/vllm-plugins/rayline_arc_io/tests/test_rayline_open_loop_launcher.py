# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

launcher = importlib.import_module("rayline_open_loop_launcher")


def test_unregistered_open_loop_stops_before_side_effects(tmp_path: Path) -> None:
    args = launcher.argparse.Namespace(
        run_id="unregistered-run",
        pathfinder_root=tmp_path,
        packet_dir=tmp_path / "packet",
        runtime_dir=tmp_path / "runtime",
        router_image="unused",
    )

    with pytest.raises(ValueError, match="only permits preregistered run id"):
        launcher._preflight(args)

    assert list(tmp_path.iterdir()) == []


def test_launcher_has_no_provider_or_qualification_path() -> None:
    source = (SCRIPT_DIR / "rayline_open_loop_launcher.py").read_text()

    assert launcher.OPEN_LOOP_ARMS == ("rayline_remote", "rayline_arc")
    assert "modal_inprocess" not in source
    assert "openrouter" not in source.lower()
    assert "execute-paid-1000" not in source
    assert '"release_qualification_1000_executed": False' in source
