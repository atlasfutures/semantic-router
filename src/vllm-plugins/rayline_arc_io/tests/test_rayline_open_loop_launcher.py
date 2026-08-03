# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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

    with pytest.raises(ValueError, match="no Rayline open-loop sweep"):
        launcher._preflight(args)

    assert list(tmp_path.iterdir()) == []


def test_launcher_has_no_provider_or_qualification_path() -> None:
    source = (SCRIPT_DIR / "rayline_open_loop_launcher.py").read_text()

    assert launcher.OPEN_LOOP_ARMS == ("rayline_remote", "rayline_arc")
    assert "modal_inprocess" not in source
    assert "openrouter" not in source.lower()
    assert "execute-paid-1000" not in source
    assert '"release_qualification_1000_executed": False' in source


def test_probe_cell_can_share_session_namespace_without_changing_receipt_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    packet_dir = tmp_path / "packet"
    (packet_dir / "cells/r030").mkdir(parents=True)
    context = SimpleNamespace(
        contract=SimpleNamespace(run_id="run"),
        packet_dir=packet_dir,
        semantic_root=REPO_ROOT,
    )
    cell = SimpleNamespace(label="r030")
    seen: list[str] = []

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        seen.extend(command)
        output = Path(command[command.index("--output") + 1])
        output.write_text('{"status":"ok"}\n')
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(launcher, "_run", fake_run)
    receipt = launcher._probe_cell(
        context,
        cell,
        "rayline_arc",
        "http://example.test",
        output_dir,
        10.0,
        logical_arm="treatment",
        session_namespace="shared-affinity",
    )

    assert seen[seen.index("--run-id") + 1] == "run:r030:shared-affinity"
    assert (output_dir / "treatment.json").exists()
    assert receipt == {"status": "ok"}
