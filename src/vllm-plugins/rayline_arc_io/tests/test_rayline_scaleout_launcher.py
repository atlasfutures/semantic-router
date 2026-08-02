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

launcher = importlib.import_module("rayline_scaleout_launcher")


def test_unregistered_scaleout_stops_before_side_effects(tmp_path: Path) -> None:
    args = launcher.argparse.Namespace(
        run_id="unregistered-run",
        pathfinder_root=tmp_path,
        packet_dir=tmp_path / "packet",
        runtime_dir=tmp_path / "runtime",
        router_image="unused",
    )

    with pytest.raises(ValueError, match="no Rayline scale-out experiment"):
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


def test_encoder_url_uses_cls_instance_web_method() -> None:
    calls: list[tuple[str, str, str]] = []

    class FakeWeb:
        def get_web_url(self) -> str:
            return "https://replica.test/"

    class FakeCls:
        def __call__(self) -> object:
            return SimpleNamespace(web=FakeWeb())

    class FakeClsAPI:
        @staticmethod
        def from_name(app: str, cls: str, *, environment_name: str) -> FakeCls:
            calls.append((app, cls, environment_name))
            return FakeCls()

    context = SimpleNamespace(modal=SimpleNamespace(Cls=FakeClsAPI))

    assert launcher._deployed_encoder_url(context, "encoder-a") == (
        "https://replica.test"
    )
    assert calls == [("encoder-a", "SessionEncoder", "dev")]


def test_cleanup_stop_is_noninteractive() -> None:
    source = (SCRIPT_DIR / "rayline_scaleout_launcher.py").read_text()

    assert '"stop",\n            "-y",' in source
    assert "Function.from_name" not in source


def test_cleanup_deletes_token_and_waits_for_stable_zero(monkeypatch) -> None:
    deleted: list[str] = []
    calls: list[str] = []

    class FakeManager:
        def delete(self, token: str) -> None:
            deleted.append(token)

    ownership = launcher.EncoderPairOwnership(
        manager=FakeManager(),
        proxy=SimpleNamespace(token_id="token"),
        service_environment={},
        base_urls=("", ""),
    )
    context = SimpleNamespace()

    def stop(_context, _environment, app: str) -> bool:
        calls.append(app)
        if len(calls) == 1:
            raise TimeoutError("transient stop timeout")
        return True

    monkeypatch.setattr(launcher, "_stop_app", stop)
    app_states = iter(
        (
            [{"state": "deployed", "tasks": "0"}],
            [{"state": "stopped", "tasks": "0"}],
        )
    )
    monkeypatch.setattr(
        launcher,
        "_named_encoder_apps",
        lambda _context: next(app_states),
    )
    monkeypatch.setattr(launcher, "_named_encoder_containers", lambda _context: [])
    monkeypatch.setattr(launcher.time, "sleep", lambda _seconds: None)

    launcher._cleanup_encoder_pair(context, ownership)

    assert calls == list(launcher.ENCODER_APP_NAMES)
    assert deleted == ["token"]
    assert ownership.cleanup == {
        "proxy_token_deleted": True,
        "encoder_apps_stopped": True,
        "encoder_containers_remaining": 0,
    }
