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


def test_unregistered_run_stops_before_preflight_side_effects(tmp_path: Path) -> None:
    args = launcher.argparse.Namespace(
        run_id="unregistered-run",
        pathfinder_root=tmp_path,
        packet_dir=tmp_path / "packet",
        runtime_dir=tmp_path / "runtime",
        router_image="unused",
    )

    with pytest.raises(ValueError, match="no Rayline concurrency sweep"):
        launcher._preflight(args)

    assert list(tmp_path.iterdir()) == []


def test_launcher_has_only_registered_sweep_arms_and_no_qualification() -> None:
    source = (SCRIPT_DIR / "rayline_concurrency_launcher.py").read_text()

    assert launcher.SWEEP_ARMS == ("rayline_remote", "rayline_arc")
    assert "modal_inprocess" not in source
    assert "execute-paid-1000" not in source
    assert '"release_qualification_1000_executed": False' in source


def test_encoder_cleanup_addresses_the_app_that_was_actually_deployed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped: list[str] = []
    listed: list[str] = []
    monkeypatch.setattr(
        launcher,
        "_stop_modal_encoder",
        lambda _python, _root, _env, _run_id, app_name: stopped.append(app_name),
    )
    monkeypatch.setattr(
        launcher,
        "_modal_containers",
        lambda _python, _root, _env, app_name: listed.append(app_name) or [],
    )
    ownership = launcher.EncoderOwnership(
        manager=type("M", (), {"delete": staticmethod(lambda _token: None)})(),
        proxy=type("P", (), {"token_id": "token"})(),
        service_environment={},
        client=None,
        app_name="rayline-arc-session-encoder-flashinfer-perf031",
    )
    context = launcher.SweepContext(
        contract=type("C", (), {"run_id": "run"})(),
        semantic_root=Path("/unused"),
        pathfinder_root=Path("/unused"),
        pathfinder_python=Path("/unused"),
        packet_dir=Path("/unused"),
        runtime_dir=Path("/unused"),
        router_image="unused",
        semantic_head="",
        pathfinder_head="",
        worker_ids=[],
        checkpoint=Path("/unused"),
        output_dir=Path("/unused"),
        base_environment={},
        modal=None,
        yaml=None,
    )

    launcher._cleanup_encoder(context, ownership)

    # Stopping the default app while a profiled app is live would leak an H100
    # and still report clean.
    assert stopped == ["rayline-arc-session-encoder-flashinfer-perf031"]
    assert listed == ["rayline-arc-session-encoder-flashinfer-perf031"]
    assert ownership.cleanup["encoder_containers_remaining"] == 0


def test_default_encoder_ownership_stays_the_frozen_perf021_identity() -> None:
    ownership = launcher.EncoderOwnership(
        manager=None,
        proxy=None,
        service_environment={},
        client=None,
    )

    assert ownership.app_name == "rayline-arc-session-encoder"
    assert ownership.base_url == launcher.IDENTITY.encoder_url
