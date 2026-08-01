#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Deploy, run, and clean up the bounded protected-encoder diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[3]
DRIVER = Path(__file__).with_name("modal_encoder_diagnostic.py")
SERVICE = REPO_ROOT / "src/vllm-plugins/rayline_arc_io/modal_session_service.py"
ENCODER_APP_ID = "ap-rs3UkEn5XUnWjrZOXYbkuB"
ENCODER_URL = (
    "https://atlasfutures-dev--rayline-arc-session-encoder-sessionenc-2d82ac."
    "modal.run"
)
REQUIRED_MODAL_VERSION = "1.5.1"
MAX_DIAGNOSTIC_SECONDS = 15 * 60
MAX_CLEANUP_SECONDS = 60


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


def _emit_sanitized_result(
    result: subprocess.CompletedProcess[str],
    protected_values: tuple[str, ...],
) -> None:
    combined = result.stdout + result.stderr
    if any(value and value in combined for value in protected_values):
        raise RuntimeError("protected credential appeared in diagnostic output")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    print(result.stdout, end="")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()

    if modal.__version__ != REQUIRED_MODAL_VERSION:
        raise SystemExit(
            f"Modal SDK {REQUIRED_MODAL_VERSION} is required; "
            f"found {modal.__version__}"
        )
    modal_command = [sys.executable, "-m", "modal"]
    environment = os.environ.copy()
    if _encoder_containers(modal_command, environment):
        raise SystemExit("protected encoder already has a running container")

    manager = modal.Workspace.from_context().proxy_tokens
    proxy_token = manager.create()
    environment.update(
        {
            "RAYLINE_ARC_MODAL_KEY": proxy_token.token_id,
            "RAYLINE_ARC_MODAL_SECRET": proxy_token.token_secret,
        }
    )
    try:
        print("encoder diagnostic deploy: starting", file=sys.stderr, flush=True)
        _run(
            [*modal_command, "deploy", str(SERVICE)],
            environment=environment,
        )
        print("encoder diagnostic packet: starting", file=sys.stderr, flush=True)
        result = _run(
            [
                sys.executable,
                str(DRIVER),
                "--base-url",
                ENCODER_URL,
                "--run-id",
                args.run_id,
                "--timeout-seconds",
                str(args.timeout_seconds),
            ],
            environment=environment,
            check=False,
            capture_output=True,
            timeout=MAX_DIAGNOSTIC_SECONDS,
        )
        _emit_sanitized_result(
            result,
            (proxy_token.token_id, proxy_token.token_secret),
        )
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    finally:
        print("encoder diagnostic cleanup: starting", file=sys.stderr, flush=True)
        try:
            manager.delete(proxy_token.token_id)
        finally:
            _stop_encoder_containers(modal_command, environment)
        print(
            "encoder diagnostic cleanup: proxy token deleted, encoder stopped",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()
