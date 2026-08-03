#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Launch and clean up the bounded three-model OpenRouter ARC canary."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal
from modal_fullstack_inputs import CANDIDATE_PROMPTS

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILE = REPO_ROOT / "deploy/compose/rayline-arc/compose.yaml"
COMPOSE_OVERRIDE_FILE = REPO_ROOT / "deploy/compose/rayline-arc/compose-openrouter.yaml"
OPENROUTER_CONFIG_FILE = REPO_ROOT / "deploy/compose/rayline-arc/config-openrouter.yaml"
OPENROUTER_ENVOY_FILE = REPO_ROOT / "deploy/compose/rayline-arc/envoy-openrouter.yaml"
DRIVER = Path(__file__).with_name("openrouter_fullstack_canary.py")
PROJECT_NAME = "rayline-arc-openrouter"
ENCODER_APP_ID = "ap-XtsWCBEWdw1ncu9Kv12Chj"
ENCODER_APP_NAME = "rayline-arc-session-encoder"
ENCODER_CLASS_NAME = "SessionEncoder"
ENCODER_HOST = (
    "atlasfutures-dev--rayline-arc-session-encoder-sessionenc-2d82ac.modal.run"
)
ENCODER_BUILD_ID = "vllm@9f5ea81ca0aa570aea46baf82311a1139c1267ca"
ENCODER_DEPLOYMENT_SOURCE_COMMIT = "0e07fa25410adf2ec2fc8e087dd951436c6b6e0d"
ENCODER_PLUGIN_SOURCE_DIGEST = (
    "1ff4ee4d7a22cc1d74c0cdb0352d3f76f5081b7201fa63e7f8f3dd10af246afd"
)
OPENROUTER_MANAGEMENT_URL = "https://openrouter.ai/api/v1/keys"
OPENROUTER_KEY_LIMIT_USD = 0.25
GATEWAY_URL = "http://127.0.0.1:18888"
ROUTER_HEALTH_URL = "http://127.0.0.1:18082/health"
METRICS_URL = "http://127.0.0.1:19190/metrics"
ARC_READY_METRIC = 'llm_rayline_arc_component_ready{component="artifact_head_encoder"}'
MAX_STARTUP_SECONDS = 240
MAX_CANARY_SECONDS = 15 * 60
MAX_CLEANUP_SECONDS = 60
HTTP_OK = 200
HTTP_CREATED = 201
HTTP_UNAUTHORIZED = 401
GIT_SHA1_HEX_LENGTH = 40
REQUIRED_MODAL_VERSION = "1.5.1"
AGENTIC_PREREGISTRATION_COMMIT = "25ef39dac03015934bde87b6739505dfac2e5210"
AGENTIC_AUTHORIZATION_COMMIT = ""
DGN003_PREREGISTRATION_COMMIT = "d40ffb7421025bc0a779ea63e85a5a91341be3a7"
DGN003_AUTHORIZATION_COMMIT = "b87a602f9ad0a666e838a54cc15c02e67b584d1d"
AGENTIC_SOURCE_REMOTE_REF = "atlasfutures/codex/rayline-remote-mvp"
PUBLIC_REQUEST_LOG_MARKERS = (
    *CANDIDATE_PROMPTS,
    "await state.commit",
    "source=public-synthetic",
    "event=retry",
)


@dataclass(frozen=True)
class RunPacket:
    compose_override: Path
    config: Path
    driver: Path
    project_name: str
    key_limit_usd: float
    maximum_seconds: int
    protected_encoder: bool


@dataclass
class RuntimeState:
    environment: dict[str, str]
    proxy_token: Any = None
    ephemeral_key: str = ""
    key_hash: str = ""
    encoder_instance: Any = None
    encoder_autoscaler_pinned: bool = False


PACKETS = {
    "canary": RunPacket(
        compose_override=COMPOSE_OVERRIDE_FILE,
        config=OPENROUTER_CONFIG_FILE,
        driver=DRIVER,
        project_name=PROJECT_NAME,
        key_limit_usd=OPENROUTER_KEY_LIMIT_USD,
        maximum_seconds=MAX_CANARY_SECONDS,
        protected_encoder=True,
    ),
    "agentic": RunPacket(
        compose_override=(
            REPO_ROOT / "deploy/compose/rayline-arc/compose-openrouter-agentic.yaml"
        ),
        config=(
            REPO_ROOT / "deploy/compose/rayline-arc/config-openrouter-agentic.yaml"
        ),
        driver=Path(__file__).with_name("openrouter_agentic_benchmark.py"),
        project_name="rayline-arc-openrouter-agentic",
        key_limit_usd=0.75,
        maximum_seconds=30 * 60,
        protected_encoder=True,
    ),
    "gateway-shape": RunPacket(
        compose_override=(
            REPO_ROOT / "deploy/compose/rayline-arc/compose-openrouter-agentic.yaml"
        ),
        config=(
            REPO_ROOT / "deploy/compose/rayline-arc/config-openrouter-agentic.yaml"
        ),
        driver=Path(__file__).with_name("openrouter_gateway_shape_diagnostic.py"),
        project_name="rayline-arc-openrouter-gateway-shape",
        key_limit_usd=0.05,
        maximum_seconds=5 * 60,
        protected_encoder=False,
    ),
}


def _run(
    command: list[str],
    *,
    environment: dict[str, str],
    check: bool = True,
    capture_output: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        check=check,
        text=True,
        capture_output=capture_output,
        timeout=timeout,
    )


def _compose_command(packet: RunPacket, *arguments: str) -> list[str]:
    return [
        "docker",
        "compose",
        "--project-name",
        packet.project_name,
        "--file",
        str(COMPOSE_FILE),
        "--file",
        str(packet.compose_override),
        *arguments,
    ]


def _wait_http(url: str) -> None:
    deadline = time.monotonic() + MAX_STARTUP_SECONDS
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == HTTP_OK:
                    return
        except OSError:
            pass
        time.sleep(1)
    raise RuntimeError(f"timed out waiting for {url}")


def _arc_component_ready(metrics: str) -> bool | None:
    prefix = f"{ARC_READY_METRIC} "
    for line in metrics.splitlines():
        if line.startswith(prefix):
            return float(line.removeprefix(prefix)) == 1.0
    return None


def _wait_arc_component_ready(url: str) -> None:
    deadline = time.monotonic() + MAX_STARTUP_SECONDS
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status != HTTP_OK:
                    continue
                ready = _arc_component_ready(response.read().decode())
                if ready is True:
                    return
                if ready is False:
                    raise RuntimeError("Rayline ARC component failed startup readiness")
        except RuntimeError:
            raise
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise RuntimeError("timed out waiting for Rayline ARC component readiness")


def _management_request(
    *,
    method: str,
    management_key: str,
    path: str = "",
    payload: dict[str, Any] | None = None,
    expected_status: int = HTTP_OK,
) -> dict[str, Any]:
    body = None
    headers = {"authorization": f"Bearer {management_key}"}
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        headers["content-type"] = "application/json"
    request = urllib.request.Request(
        f"{OPENROUTER_MANAGEMENT_URL}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            if response.status != expected_status:
                raise RuntimeError(
                    "OpenRouter management request returned wrong status"
                )
            response_body = response.read()
    except urllib.error.HTTPError as error:
        error.read()
        raise RuntimeError("OpenRouter management request failed") from error
    if not response_body:
        return {}
    decoded = json.loads(response_body)
    if not isinstance(decoded, dict):
        raise TypeError("OpenRouter management response was malformed")
    return decoded


def _create_ephemeral_key(
    management_key: str, run_id: str, key_limit_usd: float
) -> tuple[str, str]:
    response = _management_request(
        method="POST",
        management_key=management_key,
        payload={
            "name": f"rayline-arc-{run_id}",
            "limit": key_limit_usd,
            "include_byok_in_limit": True,
        },
        expected_status=HTTP_CREATED,
    )
    key = response.get("key")
    data = response.get("data")
    key_hash = data.get("hash") if isinstance(data, dict) else None
    if (
        not isinstance(key, str)
        or not key
        or not isinstance(key_hash, str)
        or not key_hash
    ):
        raise RuntimeError("OpenRouter did not return an ephemeral key contract")
    return key, key_hash


def _ephemeral_key_usage(management_key: str, key_hash: str) -> float:
    response = _management_request(
        method="GET", management_key=management_key, path=f"/{key_hash}"
    )
    data = response.get("data")
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, (int, float)):
        raise TypeError("OpenRouter key usage was unavailable")
    return float(usage)


def _delete_ephemeral_key(management_key: str, key_hash: str) -> None:
    _management_request(
        method="DELETE",
        management_key=management_key,
        path=f"/{key_hash}",
        expected_status=HTTP_OK,
    )


def _encoder_containers(
    modal_command: list[str], environment: dict[str, str]
) -> list[str]:
    result = _run(
        [*modal_command, "container", "list", "--json"],
        environment=environment,
        capture_output=True,
    )
    containers = json.loads(result.stdout)
    return sorted(
        str(container["container_id"])
        for container in containers
        if container.get("app_id") == ENCODER_APP_ID
    )


def _verify_encoder_deployment(
    modal_command: list[str], environment: dict[str, str]
) -> None:
    result = _run(
        [*modal_command, "app", "list", "--json"],
        environment=environment,
        capture_output=True,
    )
    apps = json.loads(result.stdout)
    matching = [app for app in apps if app.get("description") == ENCODER_APP_NAME]
    if len(matching) != 1:
        raise SystemExit("protected encoder deployment identity is ambiguous")
    app = matching[0]
    if (
        app.get("app_id") != ENCODER_APP_ID
        or app.get("state") != "deployed"
        or str(app.get("tasks")) != "0"
    ):
        raise SystemExit("protected encoder deployment is not the frozen idle app")
    request = urllib.request.Request(f"https://{ENCODER_HOST}/health")
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            status = response.status
    except urllib.error.HTTPError as error:
        error.read()
        status = error.code
    if status != HTTP_UNAUTHORIZED:
        raise SystemExit("protected encoder route is not registered behind auth")


def _wait_protected_encoder(proxy_token: Any) -> None:
    request = urllib.request.Request(
        f"https://{ENCODER_HOST}/health",
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


def _stop_encoder_containers(
    modal_command: list[str], environment: dict[str, str]
) -> None:
    for container_id in _encoder_containers(modal_command, environment):
        _run(
            [*modal_command, "container", "stop", container_id, "--yes"],
            environment=environment,
            check=False,
            capture_output=True,
        )
    deadline = time.monotonic() + MAX_CLEANUP_SECONDS
    while time.monotonic() < deadline:
        if not _encoder_containers(modal_command, environment):
            return
        time.sleep(1)
    raise RuntimeError("protected encoder container remained after cleanup")


def _pin_encoder_singleton(state: RuntimeState) -> None:
    state.encoder_instance = modal.Cls.from_name(
        ENCODER_APP_NAME,
        ENCODER_CLASS_NAME,
    )()
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


def _scan_logs(
    packet: RunPacket,
    environment: dict[str, str],
    protected_values: tuple[str, ...],
) -> None:
    result = _run(
        _compose_command(packet, "logs", "--no-color"),
        environment=environment,
        check=False,
        capture_output=True,
    )
    logs = result.stdout + result.stderr
    for value in protected_values:
        if value and value in logs:
            raise RuntimeError("protected credential appeared in compose logs")


def _collect_post_run_evidence(
    *,
    environment: dict[str, str],
    protected_values: tuple[str, ...],
    management_key: str,
    key_hash: str,
    packet: RunPacket,
) -> float:
    _scan_logs(packet, environment, protected_values)
    if not key_hash:
        print(
            "OpenRouter ephemeral key was not created",
            file=sys.stderr,
            flush=True,
        )
        return 0.0
    usage = _ephemeral_key_usage(management_key, key_hash)
    print(
        f"OpenRouter ephemeral key usage: ${usage:.8f}",
        file=sys.stderr,
        flush=True,
    )
    if usage > packet.key_limit_usd:
        raise RuntimeError("OpenRouter key usage exceeded its hard limit")
    return usage


def _cleanup_runtime(
    *,
    environment: dict[str, str],
    manager: Any,
    proxy_token: Any,
    management_key: str,
    key_hash: str,
    modal_command: list[str],
    packet: RunPacket,
    state: RuntimeState,
) -> None:
    print("OpenRouter cleanup: starting", file=sys.stderr, flush=True)
    _run(
        _compose_command(packet, "down", "--volumes", "--remove-orphans"),
        environment=environment,
        check=False,
        capture_output=True,
    )
    try:
        if proxy_token is not None:
            manager.delete(proxy_token.token_id)
    finally:
        try:
            if key_hash:
                _delete_ephemeral_key(management_key, key_hash)
        finally:
            if packet.protected_encoder:
                try:
                    _restore_encoder_scale_to_zero(state)
                finally:
                    _stop_encoder_containers(modal_command, environment)
    print(
        "OpenRouter cleanup: compose down, keys deleted, encoder stopped",
        file=sys.stderr,
        flush=True,
    )


def _runtime_environment(
    *,
    openrouter_key: str,
    modal_key: str,
    modal_secret: str,
    packet: RunPacket,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "OPENROUTER_EPHEMERAL_API_KEY": openrouter_key,
            "RAYLINE_ARC_E2E_PROVIDER_KEY": openrouter_key,
            "RAYLINE_ARC_E2E_CONFIG_PATH": str(packet.config),
            "RAYLINE_ARC_E2E_ENVOY_CONFIG_PATH": str(OPENROUTER_ENVOY_FILE),
        }
    )
    if packet.protected_encoder:
        environment.update(
            {
                "RAYLINE_ARC_E2E_MODAL_KEY": modal_key,
                "RAYLINE_ARC_E2E_MODAL_SECRET": modal_secret,
                "RAYLINE_ARC_E2E_ENCODER_BASE_URL": f"https://{ENCODER_HOST}",
                "RAYLINE_ARC_E2E_ENCODER_BUILD_ID": ENCODER_BUILD_ID,
            }
        )
    else:
        environment.update(
            {
                "RAYLINE_ARC_E2E_MODAL_KEY": "public-e2e-modal-key",
                "RAYLINE_ARC_E2E_MODAL_SECRET": "public-e2e-modal-secret",
                "RAYLINE_ARC_E2E_ENCODER_BASE_URL": "http://fake-encoder:8080",
                "RAYLINE_ARC_E2E_ENCODER_BUILD_ID": ("vllm@public-rayline-e2e-build"),
            }
        )
    return environment


def _verify_source_authority(mode: str, environment: dict[str, str]) -> None:
    authorities = {
        "agentic": (AGENTIC_PREREGISTRATION_COMMIT, AGENTIC_AUTHORIZATION_COMMIT),
        "gateway-shape": (
            DGN003_PREREGISTRATION_COMMIT,
            DGN003_AUTHORIZATION_COMMIT,
        ),
    }
    pins = authorities.get(mode)
    if pins is None:
        return
    if any(len(pin) != GIT_SHA1_HEX_LENGTH for pin in pins):
        raise SystemExit(f"{mode} launch authority is source-closed")
    status = _run(
        ["git", "status", "--porcelain"],
        environment=environment,
        capture_output=True,
    )
    if status.stdout:
        raise SystemExit(f"{mode} launch requires a clean source checkpoint")
    head = _run(
        ["git", "rev-parse", "HEAD"],
        environment=environment,
        capture_output=True,
    ).stdout.strip()
    remote = _run(
        ["git", "rev-parse", AGENTIC_SOURCE_REMOTE_REF],
        environment=environment,
        capture_output=True,
    ).stdout.strip()
    pushed = _run(
        ["git", "merge-base", "--is-ancestor", head, remote],
        environment=environment,
        check=False,
        capture_output=True,
    )
    if pushed.returncode != 0:
        raise SystemExit(f"{mode} launch source is not remote-visible")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=sorted(PACKETS), default="canary")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser.parse_args()


def _execute_runtime(
    *,
    args: argparse.Namespace,
    packet: RunPacket,
    management_key: str,
    manager: Any,
    state: RuntimeState,
) -> None:
    if packet.protected_encoder:
        _pin_encoder_singleton(state)
        state.proxy_token = manager.create()
        _wait_protected_encoder(state.proxy_token)
    state.ephemeral_key, state.key_hash = _create_ephemeral_key(
        management_key, args.run_id, packet.key_limit_usd
    )
    state.environment = _runtime_environment(
        openrouter_key=state.ephemeral_key,
        modal_key=(state.proxy_token.token_id if state.proxy_token is not None else ""),
        modal_secret=(
            state.proxy_token.token_secret if state.proxy_token is not None else ""
        ),
        packet=packet,
    )
    _run(
        _compose_command(packet, "down", "--volumes", "--remove-orphans"),
        environment=state.environment,
        check=False,
        capture_output=True,
    )
    _run(
        _compose_command(packet, "up", "--build", "--detach"),
        environment=state.environment,
    )
    _wait_http(ROUTER_HEALTH_URL)
    _wait_arc_component_ready(METRICS_URL)
    _run(
        [
            sys.executable,
            str(packet.driver),
            "--gateway-url",
            GATEWAY_URL,
            "--metrics-url",
            METRICS_URL,
            "--run-id",
            args.run_id,
            "--timeout-seconds",
            str(args.timeout_seconds),
        ],
        environment=state.environment,
        timeout=packet.maximum_seconds,
    )


def main() -> None:
    args = _parse_args()

    packet = PACKETS[args.mode]

    if packet.protected_encoder and modal.__version__ != REQUIRED_MODAL_VERSION:
        raise SystemExit(
            f"Modal SDK {REQUIRED_MODAL_VERSION} is required; found {modal.__version__}"
        )
    management_key = os.environ.get("OPENROUTER_MANAGEMENT_KEY", "")
    if not management_key:
        raise SystemExit("OPENROUTER_MANAGEMENT_KEY is required")
    modal_command = [sys.executable, "-m", "modal"]
    base_environment = os.environ.copy()
    _verify_source_authority(args.mode, base_environment)
    if packet.protected_encoder:
        _verify_encoder_deployment(modal_command, base_environment)
        if _encoder_containers(modal_command, base_environment):
            raise SystemExit("protected encoder already has a running container")

    manager = (
        modal.Workspace.from_context().proxy_tokens
        if packet.protected_encoder
        else None
    )
    state = RuntimeState(environment=base_environment)
    run_failure: Exception | None = None
    evidence_failure: Exception | None = None
    try:
        _execute_runtime(
            args=args,
            packet=packet,
            management_key=management_key,
            manager=manager,
            state=state,
        )
    except Exception as error:  # preserve the primary failure through evidence cleanup
        run_failure = error
    finally:
        try:
            _collect_post_run_evidence(
                environment=state.environment,
                protected_values=(
                    management_key,
                    state.ephemeral_key,
                    state.proxy_token.token_id if state.proxy_token is not None else "",
                    (
                        state.proxy_token.token_secret
                        if state.proxy_token is not None
                        else ""
                    ),
                    *PUBLIC_REQUEST_LOG_MARKERS,
                ),
                management_key=management_key,
                key_hash=state.key_hash,
                packet=packet,
            )
        except Exception as error:
            evidence_failure = error
            print(
                "OpenRouter post-run evidence collection failed",
                file=sys.stderr,
                flush=True,
            )
        _cleanup_runtime(
            environment=state.environment,
            manager=manager,
            proxy_token=state.proxy_token,
            management_key=management_key,
            key_hash=state.key_hash,
            modal_command=modal_command,
            packet=packet,
            state=state,
        )
    if run_failure is not None:
        if evidence_failure is not None:
            run_failure.add_note("post-run evidence collection also failed")
        raise run_failure
    if evidence_failure is not None:
        raise RuntimeError("OpenRouter post-run evidence collection failed") from (
            evidence_failure
        )


if __name__ == "__main__":
    main()
