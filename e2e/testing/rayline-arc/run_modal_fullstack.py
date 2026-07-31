#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Launch and clean up the bounded Modal real-worker full-stack canary."""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILE = REPO_ROOT / "deploy/compose/rayline-arc/compose.yaml"
WORKER_SERVICE = (
    REPO_ROOT / "src/vllm-plugins/rayline_arc_io/modal_generation_workers.py"
)
DRIVER = Path(__file__).with_name("modal_fullstack_canary.py")
PROJECT_NAME = "rayline-arc-real-workers"
WORKER_APP_NAME = "rayline-arc-generation-workers"
WORKER_A_HOST = "atlasfutures-dev--rayline-arc-generation-workers-worker-a.modal.run"
WORKER_B_HOST = "atlasfutures-dev--rayline-arc-generation-workers-worker-b.modal.run"
GATEWAY_URL = "http://127.0.0.1:18888"
METRICS_URL = "http://127.0.0.1:19190/metrics"
MAX_STARTUP_SECONDS = 180
MAX_CANARY_SECONDS = 15 * 60
HTTP_OK = 200


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()

    modal_command = shutil.which("modal")
    if not modal_command:
        raise SystemExit("modal CLI is required")
    worker_api_key = secrets.token_urlsafe(32)
    manager = modal.Workspace.from_context().proxy_tokens
    proxy_token = manager.create()
    environment = os.environ.copy()
    environment.update(
        {
            "RAYLINE_ARC_WORKER_API_KEY": worker_api_key,
            "RAYLINE_ARC_E2E_PROVIDER_KEY": worker_api_key,
            "RAYLINE_ARC_E2E_DISPATCH_BACKEND": "openai_compatible",
            "RAYLINE_ARC_E2E_MODAL_KEY": proxy_token.token_id,
            "RAYLINE_ARC_E2E_MODAL_SECRET": proxy_token.token_secret,
            "RAYLINE_ARC_E2E_WORKER_A_ENDPOINT": f"{WORKER_A_HOST}:443",
            "RAYLINE_ARC_E2E_WORKER_A_PROTOCOL": "https",
            "RAYLINE_ARC_E2E_WORKER_A_BASE_URL": f"https://{WORKER_A_HOST}/v1",
            "RAYLINE_ARC_E2E_WORKER_B_ENDPOINT": f"{WORKER_B_HOST}:443",
            "RAYLINE_ARC_E2E_WORKER_B_PROTOCOL": "https",
            "RAYLINE_ARC_E2E_WORKER_B_BASE_URL": f"https://{WORKER_B_HOST}/v1",
        }
    )
    try:
        print("real-worker deploy: starting", file=sys.stderr, flush=True)
        _run(
            [modal_command, "deploy", str(WORKER_SERVICE)],
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
        _wait_http(f"{GATEWAY_URL}/health")
        print("real-worker canary: starting", file=sys.stderr, flush=True)
        _run(
            [
                sys.executable,
                str(DRIVER),
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
                [modal_command, "app", "stop", WORKER_APP_NAME, "--yes"],
                environment=environment,
                check=False,
                capture_output=True,
            )
        print(
            "real-worker cleanup: compose down, proxy token deleted, app stopped",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()
