#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Exact-name Modal encoder lifecycle helpers for OpenRouter packets."""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from openrouter_fullstack_state import EncoderDeployment, RunPacket, RuntimeState

HTTP_OK = 200
HTTP_UNAUTHORIZED = 401
MAX_STARTUP_SECONDS = 240
MAX_CLEANUP_SECONDS = 60


def _run(
    command: list[str],
    *,
    environment: dict[str, str],
    cwd: Path,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=check,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def packet_encoder(packet: RunPacket) -> EncoderDeployment:
    if not packet.protected_encoder or packet.encoder is None:
        raise RuntimeError("protected packet omitted its encoder deployment contract")
    return packet.encoder


def encoder_containers(
    packet: RunPacket,
    modal_command: list[str],
    environment: dict[str, str],
    *,
    cwd: Path,
    app_id: str = "",
) -> list[str]:
    encoder = packet_encoder(packet)
    expected_app_id = app_id or encoder.app_id
    result = _run(
        [*modal_command, "container", "list", "--json"],
        environment=environment,
        cwd=cwd,
    )
    containers = json.loads(result.stdout)
    return sorted(
        str(container["container_id"])
        for container in containers
        if (
            (expected_app_id and container.get("app_id") == expected_app_id)
            or (not expected_app_id and container.get("app_name") == encoder.app_name)
        )
    )


def verify_encoder_deployment(
    packet: RunPacket,
    modal_command: list[str],
    environment: dict[str, str],
    *,
    cwd: Path,
) -> tuple[dict[str, Any], str, Any]:
    encoder = packet_encoder(packet)
    result = _run(
        [*modal_command, "app", "list", "--json"],
        environment=environment,
        cwd=cwd,
    )
    apps = json.loads(result.stdout)
    matching = [
        app
        for app in apps
        if app.get("description") == encoder.app_name and app.get("state") == "deployed"
    ]
    if len(matching) != 1:
        # Stopped apps from prior stopped attempts stay in Modal's listing
        # until they age out, so identity requires exactly one DEPLOYED app.
        raise RuntimeError("protected encoder deployment identity is ambiguous")
    app = matching[0]
    if (encoder.app_id and app.get("app_id") != encoder.app_id) or str(
        app.get("tasks")
    ) != "0":
        raise RuntimeError("protected encoder deployment is not the frozen idle app")
    modal = importlib.import_module("modal")
    instance = modal.Cls.from_name(
        encoder.app_name,
        encoder.class_name,
        environment_name=environment.get("MODAL_ENVIRONMENT", "dev"),
    )()
    base_url = instance.web.get_web_url()
    if not base_url:
        raise RuntimeError("protected encoder web URL is unavailable")
    if encoder.expected_host and base_url.rstrip("/") != (
        f"https://{encoder.expected_host}"
    ):
        raise RuntimeError("protected encoder web URL identity diverged")
    request = urllib.request.Request(f"{base_url.rstrip('/')}/health")
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            status = response.status
    except urllib.error.HTTPError as error:
        error.read()
        status = error.code
    if status != HTTP_UNAUTHORIZED:
        raise RuntimeError("protected encoder route is not registered behind auth")
    return app, base_url.rstrip("/"), instance


def assert_encoder_inactive(
    packet: RunPacket,
    modal_command: list[str],
    environment: dict[str, str],
    *,
    cwd: Path,
) -> None:
    encoder = packet_encoder(packet)
    result = _run(
        [*modal_command, "app", "list", "--json"],
        environment=environment,
        cwd=cwd,
    )
    matching = [
        app
        for app in json.loads(result.stdout)
        if app.get("description") == encoder.app_name
    ]
    active = [
        app
        for app in matching
        if app.get("state") != "stopped" or str(app.get("tasks")) != "0"
    ]
    if active or encoder_containers(packet, modal_command, environment, cwd=cwd):
        raise RuntimeError("ephemeral protected encoder is already active")


def deploy_encoder(
    packet: RunPacket,
    modal_command: list[str],
    environment: dict[str, str],
    state: RuntimeState,
    *,
    cwd: Path,
) -> None:
    encoder = packet_encoder(packet)
    if not encoder.ephemeral or encoder.deploy_service_path is None:
        return
    state.encoder_deployed = True
    state.encoder_owned = True
    result = _run(
        [*modal_command, "deploy", str(encoder.deploy_service_path)],
        environment={
            **environment,
            "RAYLINE_ARC_SESSION_APP_NAME": encoder.app_name,
        },
        cwd=cwd,
        timeout=15 * 60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("ephemeral FlashInfer encoder deployment failed")


def wait_protected_encoder(state: RuntimeState, proxy_token: Any) -> None:
    request = urllib.request.Request(
        f"{state.encoder_base_url}/health",
        headers={
            "Modal-Key": proxy_token.token_id,
            "Modal-Secret": proxy_token.token_secret,
        },
    )
    deadline = time.monotonic() + MAX_STARTUP_SECONDS
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                body = json.loads(response.read())
                if (
                    response.status == HTTP_OK
                    and body.get("status") == "ok"
                    and body.get("resident_sessions") == 0
                    and body.get("resident_tokens") == 0
                ):
                    return
        except urllib.error.HTTPError as error:
            error.read()
            if error.code == HTTP_UNAUTHORIZED:
                raise RuntimeError(
                    "protected encoder rejected its proxy token"
                ) from error
        except (json.JSONDecodeError, OSError, TypeError):
            pass
        time.sleep(1)
    raise RuntimeError("timed out waiting for the protected encoder deployment")


def pin_encoder_singleton(packet: RunPacket, state: RuntimeState) -> None:
    encoder = packet_encoder(packet)
    if state.encoder_instance is None:
        modal = importlib.import_module("modal")
        state.encoder_instance = modal.Cls.from_name(
            encoder.app_name,
            encoder.class_name,
            environment_name=state.environment.get("MODAL_ENVIRONMENT", "dev"),
        )()
    state.encoder_owned = True
    state.encoder_instance.update_autoscaler(
        min_containers=1,
        max_containers=1,
        buffer_containers=0,
        scaledown_window=300,
    )
    state.encoder_autoscaler_pinned = True


def _restore_encoder_scale_to_zero(state: RuntimeState) -> None:
    if not state.encoder_autoscaler_pinned or state.encoder_instance is None:
        return
    state.encoder_instance.update_autoscaler(
        min_containers=0,
        max_containers=1,
        buffer_containers=0,
        scaledown_window=300,
    )
    state.encoder_autoscaler_pinned = False


def _stop_encoder_containers(
    packet: RunPacket,
    modal_command: list[str],
    environment: dict[str, str],
    *,
    cwd: Path,
    app_id: str,
) -> None:
    for container_id in encoder_containers(
        packet, modal_command, environment, cwd=cwd, app_id=app_id
    ):
        _run(
            [*modal_command, "container", "stop", container_id, "--yes"],
            environment=environment,
            cwd=cwd,
            check=False,
        )
    deadline = time.monotonic() + MAX_CLEANUP_SECONDS
    while time.monotonic() < deadline:
        if not encoder_containers(
            packet, modal_command, environment, cwd=cwd, app_id=app_id
        ):
            return
        time.sleep(1)
    raise RuntimeError("protected encoder container remained after cleanup")


def _wait_ephemeral_encoder_cleanup(
    packet: RunPacket,
    modal_command: list[str],
    environment: dict[str, str],
    *,
    cwd: Path,
    app_id: str,
) -> None:
    encoder = packet_encoder(packet)
    deadline = time.monotonic() + MAX_CLEANUP_SECONDS
    while time.monotonic() < deadline:
        app_result = _run(
            [*modal_command, "app", "list", "--json"],
            environment=environment,
            cwd=cwd,
        )
        matching = [
            app
            for app in json.loads(app_result.stdout)
            if app.get("description") == encoder.app_name
        ]
        stopped = all(
            app.get("state") == "stopped" and str(app.get("tasks")) == "0"
            for app in matching
        )
        if stopped and not encoder_containers(
            packet, modal_command, environment, cwd=cwd, app_id=app_id
        ):
            return
        time.sleep(1)
    raise RuntimeError("ephemeral protected encoder remained after cleanup")


def cleanup_encoder(
    packet: RunPacket,
    state: RuntimeState,
    modal_command: list[str],
    environment: dict[str, str],
    *,
    cwd: Path,
) -> None:
    encoder = packet_encoder(packet)
    if not state.encoder_owned:
        return
    try:
        _restore_encoder_scale_to_zero(state)
    finally:
        try:
            _stop_encoder_containers(
                packet,
                modal_command,
                environment,
                cwd=cwd,
                app_id=state.encoder_app_id,
            )
        finally:
            if encoder.ephemeral and state.encoder_deployed:
                _run(
                    [*modal_command, "app", "stop", "-y", encoder.app_name],
                    environment=environment,
                    cwd=cwd,
                    check=False,
                    timeout=120,
                )
                _wait_ephemeral_encoder_cleanup(
                    packet,
                    modal_command,
                    environment,
                    cwd=cwd,
                    app_id=state.encoder_app_id,
                )


def plugin_source_digest(repo_root: Path) -> str:
    package_root = repo_root / "src/vllm-plugins/rayline_arc_io/src/rayline_arc_io"
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        digest.update(path.relative_to(package_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
