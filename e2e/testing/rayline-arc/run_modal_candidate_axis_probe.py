#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Launch and clean up the bounded protected-H100 candidate-axis probe."""

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
DRIVER = Path(__file__).with_name("modal_candidate_axis_probe.py")
ENCODER_APP_ID = "ap-rs3UkEn5XUnWjrZOXYbkuB"
ENCODER_URL = (
    "https://atlasfutures-dev--rayline-arc-session-encoder-sessionenc-2d82ac."
    "modal.run"
)
REQUIRED_MODAL_VERSION = "1.5.1"
MAX_PROBE_SECONDS = 12 * 60
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
        _run(
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
            timeout=MAX_PROBE_SECONDS,
        )
    finally:
        try:
            manager.delete(proxy_token.token_id)
        finally:
            _stop_encoder_containers(modal_command, environment)


if __name__ == "__main__":
    main()
