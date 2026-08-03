# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

contract = importlib.import_module("rayline_failover_contract")
launcher = importlib.import_module("rayline_failover_launcher")


def test_unregistered_failover_stops_before_side_effects(tmp_path: Path) -> None:
    args = launcher.argparse.Namespace(
        run_id="unregistered-run",
        pathfinder_root=tmp_path,
        packet_dir=tmp_path / "packet",
        runtime_dir=tmp_path / "runtime",
        router_image="unused",
    )

    with pytest.raises(ValueError, match="no Rayline failover experiment"):
        launcher._preflight(args)

    assert list(tmp_path.iterdir()) == []


def test_launcher_freezes_forced_remap_without_provider_or_qualification() -> None:
    source = (SCRIPT_DIR / "rayline_failover_launcher.py").read_text()

    assert launcher.FAILOVER_ARMS == (
        "arc_dual_sticky",
        "arc_dual_forced_failover",
    )
    assert launcher.FAILOVER_AFTER_POOLING == contract.TURNS_PER_EPISODE // 2
    assert "openrouter" not in source.lower()
    assert "execute-paid-1000" not in source
    assert '"release_qualification_1000_executed": False' in source
    assert launcher.SHARED_SESSION_NAMESPACE == "shared-affinity"
    assert "session_namespace=SHARED_SESSION_NAMESPACE" in source
    assert contract.LAUNCHABLE_CONTRACT is None
