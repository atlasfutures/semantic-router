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


def _prepare_cell_context(tmp_path: Path) -> SimpleNamespace:
    configs = tmp_path / "pathfinder/configs"
    configs.mkdir(parents=True)
    (configs / "live_gap_c82_coldswitch.yaml").write_text("router: {}\n")
    return SimpleNamespace(
        contract=SimpleNamespace(run_id="run"),
        semantic_root=REPO_ROOT,
        pathfinder_root=tmp_path / "pathfinder",
        pathfinder_python=tmp_path / "python",
        runtime_dir=tmp_path / "runtime",
        checkpoint=tmp_path / "checkpoint.pt",
        worker_ids=["worker-a"],
        yaml=SimpleNamespace(
            safe_load=lambda _text: {"router": {}},
            safe_dump=lambda _config, **_kwargs: "router: {}\n",
        ),
    )


def test_prepare_cell_gives_both_routers_the_same_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A run whose encoder differs from the frozen default is a split brain if
    # only the ARC config is told: the `pathfinder_transaction` arms reach the
    # encoder through the Pathfinder config, which has no environment
    # override. PERF031B crashed on exactly this divergence.
    flashinfer_url = (
        "https://atlasfutures-dev--rayline-arc-session-encoder-flashinfer-"
        "bc2d60.modal.run"
    )
    flashinfer_build_id = f"{launcher.IDENTITY.engine_build_id}+gdn-flashinfer-eager"
    derived: dict[str, object] = {}
    commands: list[list[str]] = []
    monkeypatch.setattr(
        launcher,
        "derive_pathfinder_config",
        lambda _base, **kwargs: derived.update(kwargs) or {"router": {}},
    )
    monkeypatch.setattr(
        launcher,
        "_run",
        lambda command, **_kwargs: commands.append(command),
    )

    launcher._prepare_cell(
        _prepare_cell_context(tmp_path),
        SimpleNamespace(concurrency=8),
        tmp_path / "work",
        encoder_base_url=flashinfer_url,
        encoder_build_id=flashinfer_build_id,
    )

    assert derived["encoder_base_url"] == flashinfer_url
    assert derived["encoder_build_id"] == flashinfer_build_id
    arc_command = commands[-1]
    assert arc_command[arc_command.index("--encoder-base-url") + 1] == flashinfer_url
    assert (
        arc_command[arc_command.index("--encoder-build-id") + 1] == flashinfer_build_id
    )


def test_prepare_cell_defaults_stay_the_frozen_perf020_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    derived: dict[str, object] = {}
    monkeypatch.setattr(
        launcher,
        "derive_pathfinder_config",
        lambda _base, **kwargs: derived.update(kwargs) or {"router": {}},
    )
    monkeypatch.setattr(launcher, "_run", lambda _command, **_kwargs: None)

    launcher._prepare_cell(
        _prepare_cell_context(tmp_path),
        SimpleNamespace(concurrency=8),
        tmp_path / "work",
    )

    assert derived["encoder_base_url"] == launcher.IDENTITY.encoder_url
    assert derived["encoder_build_id"] == launcher.IDENTITY.engine_build_id
