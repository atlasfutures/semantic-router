# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

contract = importlib.import_module("rayline_concurrency_contract")
launcher = importlib.import_module("rayline_concurrency_launcher")


def test_closed_contract_stops_before_preflight_side_effects(tmp_path: Path) -> None:
    args = launcher.argparse.Namespace(
        run_id=contract.PERF017_RUN_ID,
        pathfinder_root=tmp_path,
        packet_dir=tmp_path / "packet",
        runtime_dir=tmp_path / "runtime",
        router_image="unused",
    )

    with pytest.raises(ValueError, match="currently launchable"):
        launcher._preflight(args)

    assert list(tmp_path.iterdir()) == []


def test_launcher_has_only_registered_sweep_arms_and_no_qualification() -> None:
    source = (SCRIPT_DIR / "rayline_concurrency_launcher.py").read_text()

    assert launcher.SWEEP_ARMS == ("rayline_remote", "rayline_arc")
    assert "modal_inprocess" not in source
    assert "execute-paid-1000" not in source
    assert '"release_qualification_1000_executed": False' in source
