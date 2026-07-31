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
from pathlib import Path
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILE = REPO_ROOT / "deploy/compose/rayline-arc/compose.yaml"
COMPOSE_OVERRIDE_FILE = REPO_ROOT / "deploy/compose/rayline-arc/compose-openrouter.yaml"
OPENROUTER_CONFIG_FILE = REPO_ROOT / "deploy/compose/rayline-arc/config-openrouter.yaml"
OPENROUTER_ENVOY_FILE = REPO_ROOT / "deploy/compose/rayline-arc/envoy-openrouter.yaml"
DRIVER = Path(__file__).with_name("openrouter_fullstack_canary.py")
PROJECT_NAME = "rayline-arc-openrouter"
ENCODER_APP_ID = "ap-rs3UkEn5XUnWjrZOXYbkuB"
ENCODER_HOST = (
    "atlasfutures-dev--rayline-arc-session-encoder-sessionenc-2d82ac.modal.run"
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
REQUIRED_MODAL_VERSION = "1.5.1"


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


def _compose_command(*arguments: str) -> list[str]:
    return [
        "docker",
        "compose",
        "--project-name",
        PROJECT_NAME,
        "--file",
        str(COMPOSE_FILE),
        "--file",
        str(COMPOSE_OVERRIDE_FILE),
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


def _create_ephemeral_key(management_key: str, run_id: str) -> tuple[str, str]:
    response = _management_request(
        method="POST",
        management_key=management_key,
        payload={
            "name": f"rayline-arc-{run_id}",
            "limit": OPENROUTER_KEY_LIMIT_USD,
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


def _scan_logs(environment: dict[str, str], protected_values: tuple[str, ...]) -> None:
    result = _run(
        _compose_command("logs", "--no-color"),
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
) -> float:
    _scan_logs(environment, protected_values)
    usage = _ephemeral_key_usage(management_key, key_hash)
    print(
        f"OpenRouter ephemeral key usage: ${usage:.8f}",
        file=sys.stderr,
        flush=True,
    )
    if usage > OPENROUTER_KEY_LIMIT_USD:
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
) -> None:
    print("OpenRouter cleanup: starting", file=sys.stderr, flush=True)
    _run(
        _compose_command("down", "--volumes", "--remove-orphans"),
        environment=environment,
        check=False,
        capture_output=True,
    )
    try:
        if proxy_token is not None:
            manager.delete(proxy_token.token_id)
    finally:
        try:
            _delete_ephemeral_key(management_key, key_hash)
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
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "OPENROUTER_EPHEMERAL_API_KEY": openrouter_key,
            "RAYLINE_ARC_E2E_PROVIDER_KEY": openrouter_key,
            "RAYLINE_ARC_E2E_MODAL_KEY": modal_key,
            "RAYLINE_ARC_E2E_MODAL_SECRET": modal_secret,
            "RAYLINE_ARC_E2E_ENCODER_BASE_URL": f"https://{ENCODER_HOST}",
            "RAYLINE_ARC_E2E_ENCODER_BUILD_ID": (
                "vllm@b1049f6dd95c27d2e1b052eebc3b1a7f9f41195f"
            ),
            "RAYLINE_ARC_E2E_CONFIG_PATH": str(OPENROUTER_CONFIG_FILE),
            "RAYLINE_ARC_E2E_ENVOY_CONFIG_PATH": str(OPENROUTER_ENVOY_FILE),
        }
    )
    return environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()

    if modal.__version__ != REQUIRED_MODAL_VERSION:
        raise SystemExit(
            f"Modal SDK {REQUIRED_MODAL_VERSION} is required; found {modal.__version__}"
        )
    management_key = os.environ.get("OPENROUTER_MANAGEMENT_KEY", "")
    if not management_key:
        raise SystemExit("OPENROUTER_MANAGEMENT_KEY is required")
    modal_command = [sys.executable, "-m", "modal"]
    base_environment = os.environ.copy()
    if _encoder_containers(modal_command, base_environment):
        raise SystemExit("protected encoder already has a running container")

    ephemeral_key, key_hash = _create_ephemeral_key(management_key, args.run_id)
    proxy_token = None
    manager = modal.Workspace.from_context().proxy_tokens
    environment = base_environment
    run_failure: Exception | None = None
    evidence_failure: Exception | None = None
    try:
        proxy_token = manager.create()
        environment = _runtime_environment(
            openrouter_key=ephemeral_key,
            modal_key=proxy_token.token_id,
            modal_secret=proxy_token.token_secret,
        )
        _run(
            _compose_command("down", "--volumes", "--remove-orphans"),
            environment=environment,
            check=False,
            capture_output=True,
        )
        _run(
            _compose_command("up", "--build", "--detach"),
            environment=environment,
        )
        _wait_http(ROUTER_HEALTH_URL)
        _wait_arc_component_ready(METRICS_URL)
        _run(
            [
                sys.executable,
                str(DRIVER),
                "--gateway-url",
                GATEWAY_URL,
                "--metrics-url",
                METRICS_URL,
                "--run-id",
                args.run_id,
                "--timeout-seconds",
                str(args.timeout_seconds),
            ],
            environment=environment,
            timeout=MAX_CANARY_SECONDS,
        )
    except Exception as error:  # preserve the primary failure through evidence cleanup
        run_failure = error
    finally:
        try:
            _collect_post_run_evidence(
                environment=environment,
                protected_values=(
                    management_key,
                    ephemeral_key,
                    proxy_token.token_id if proxy_token is not None else "",
                    proxy_token.token_secret if proxy_token is not None else "",
                ),
                management_key=management_key,
                key_hash=key_hash,
            )
        except Exception as error:
            evidence_failure = error
            print(
                "OpenRouter post-run evidence collection failed",
                file=sys.stderr,
                flush=True,
            )
        _cleanup_runtime(
            environment=environment,
            manager=manager,
            proxy_token=proxy_token,
            management_key=management_key,
            key_hash=key_hash,
            modal_command=modal_command,
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
