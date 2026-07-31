#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Launch and clean up a bounded Modal real-worker full-stack run."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILE = REPO_ROOT / "deploy/compose/rayline-arc/compose.yaml"
REAL_WORKER_ENVOY_FILE = (
    REPO_ROOT / "deploy/compose/rayline-arc/envoy-real-workers.yaml"
)
WORKER_SERVICE = (
    REPO_ROOT / "src/vllm-plugins/rayline_arc_io/modal_generation_workers.py"
)
DRIVERS = {
    "canary": Path(__file__).with_name("modal_fullstack_canary.py"),
    "benchmark": Path(__file__).with_name("modal_fullstack_benchmark.py"),
}
PROJECT_NAME = "rayline-arc-real-workers"
WORKER_APP_NAME = "rayline-arc-generation-workers"
ENCODER_APP_ID = "ap-rs3UkEn5XUnWjrZOXYbkuB"
WORKER_A_HOST = "atlasfutures-dev--rayline-arc-generation-workers-worker-a.modal.run"
WORKER_B_HOST = "atlasfutures-dev--rayline-arc-generation-workers-worker-b.modal.run"
ENCODER_HOST = (
    "atlasfutures-dev--rayline-arc-session-encoder-sessionenc-2d82ac.modal.run"
)
GATEWAY_URL = "http://127.0.0.1:18888"
ROUTER_HEALTH_URL = "http://127.0.0.1:18082/health"
METRICS_URL = "http://127.0.0.1:19190/metrics"
MAX_STARTUP_SECONDS = 240
MAX_CANARY_SECONDS = 15 * 60
MAX_CLEANUP_SECONDS = 60
HTTP_OK = 200
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


def _runtime_environment(
    *, worker_api_key: str, modal_key: str, modal_secret: str
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "RAYLINE_ARC_WORKER_API_KEY": worker_api_key,
            "RAYLINE_ARC_E2E_PROVIDER_KEY": worker_api_key,
            "RAYLINE_ARC_E2E_DISPATCH_BACKEND": "openai_compatible",
            "RAYLINE_ARC_E2E_MODAL_KEY": modal_key,
            "RAYLINE_ARC_E2E_MODAL_SECRET": modal_secret,
            "RAYLINE_ARC_E2E_ENCODER_BASE_URL": f"https://{ENCODER_HOST}",
            "RAYLINE_ARC_E2E_ENCODER_BUILD_ID": (
                "vllm@b1049f6dd95c27d2e1b052eebc3b1a7f9f41195f"
            ),
            "RAYLINE_ARC_E2E_ENVOY_CONFIG_PATH": str(REAL_WORKER_ENVOY_FILE),
            "RAYLINE_ARC_E2E_WORKER_A_ENDPOINT": f"{WORKER_A_HOST}:443",
            "RAYLINE_ARC_E2E_WORKER_A_PROTOCOL": "https",
            "RAYLINE_ARC_E2E_WORKER_A_BASE_URL": f"https://{WORKER_A_HOST}/v1",
            "RAYLINE_ARC_E2E_WORKER_B_ENDPOINT": f"{WORKER_B_HOST}:443",
            "RAYLINE_ARC_E2E_WORKER_B_PROTOCOL": "https",
            "RAYLINE_ARC_E2E_WORKER_B_BASE_URL": f"https://{WORKER_B_HOST}/v1",
        }
    )
    return environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=sorted(DRIVERS), default="canary")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()

    if modal.__version__ != REQUIRED_MODAL_VERSION:
        raise SystemExit(
            f"Modal SDK {REQUIRED_MODAL_VERSION} is required; found {modal.__version__}"
        )
    modal_command = [sys.executable, "-m", "modal"]
    base_environment = os.environ.copy()
    if _encoder_containers(modal_command, base_environment):
        raise SystemExit("protected encoder already has a running container")
    worker_api_key = secrets.token_urlsafe(32)
    manager = modal.Workspace.from_context().proxy_tokens
    proxy_token = manager.create()
    environment = _runtime_environment(
        worker_api_key=worker_api_key,
        modal_key=proxy_token.token_id,
        modal_secret=proxy_token.token_secret,
    )
    try:
        print("real-worker deploy: starting", file=sys.stderr, flush=True)
        _run(
            [*modal_command, "deploy", str(WORKER_SERVICE)],
            environment=environment,
        )
        print("real-worker compose: starting", file=sys.stderr, flush=True)
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
        print(f"real-worker {args.mode}: starting", file=sys.stderr, flush=True)
        _run(
            [
                sys.executable,
                str(DRIVERS[args.mode]),
                "--gateway-url",
                GATEWAY_URL,
                "--metrics-url",
                METRICS_URL,
                "--worker-a-url",
                f"https://{WORKER_A_HOST}",
                "--worker-b-url",
                f"https://{WORKER_B_HOST}",
                "--run-id",
                args.run_id,
                "--timeout-seconds",
                str(args.timeout_seconds),
            ],
            environment=environment,
            timeout=MAX_CANARY_SECONDS,
        )
        _scan_logs(
            environment,
            (
                worker_api_key,
                proxy_token.token_id,
                proxy_token.token_secret,
            ),
        )
    finally:
        print("real-worker cleanup: starting", file=sys.stderr, flush=True)
        _run(
            _compose_command("down", "--volumes", "--remove-orphans"),
            environment=environment,
            check=False,
            capture_output=True,
        )
        try:
            manager.delete(proxy_token.token_id)
        finally:
            _run(
                [*modal_command, "app", "stop", WORKER_APP_NAME, "--yes"],
                environment=environment,
                check=False,
                capture_output=True,
            )
            _stop_encoder_containers(modal_command, environment)
        print(
            "real-worker cleanup: compose down, proxy token deleted, apps stopped",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()
