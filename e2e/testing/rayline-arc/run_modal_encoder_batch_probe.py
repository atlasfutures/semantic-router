#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Deploy, run once, and clean up the protected-encoder batch proof."""

from __future__ import annotations

import argparse
import os
import sys

import modal
from modal_encoder_batch_probe import (
    CUMULATIVE_BEFORE_USD,
    MAX_RESOURCE_ENVELOPE_USD,
)
from run_modal_encoder_diagnostic import (
    ENCODER_URL,
    REQUIRED_MODAL_VERSION,
    SERVICE,
    _emit_sanitized_result,
    _encoder_containers,
    _run,
    _stop_encoder_containers,
)

DRIVER = os.path.join(
    os.path.dirname(__file__),
    "modal_encoder_batch_probe.py",
)
MAX_PROBE_SECONDS = 5 * 60
REQUEST_TIMEOUT_SECONDS = 300.0
BUDGET_CAP_USD = 40.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    if CUMULATIVE_BEFORE_USD + MAX_RESOURCE_ENVELOPE_USD > BUDGET_CAP_USD:
        raise SystemExit("encoder batch probe requires renewed budget authority")
    if modal.__version__ != REQUIRED_MODAL_VERSION:
        raise SystemExit(
            f"Modal SDK {REQUIRED_MODAL_VERSION} is required; found {modal.__version__}"
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
        print("encoder batch probe deploy: starting", file=sys.stderr, flush=True)
        _run(
            [*modal_command, "deploy", str(SERVICE)],
            environment=environment,
        )
        print("encoder batch probe packet: starting", file=sys.stderr, flush=True)
        result = _run(
            [
                sys.executable,
                DRIVER,
                "--base-url",
                ENCODER_URL,
                "--run-id",
                args.run_id,
                "--timeout-seconds",
                str(REQUEST_TIMEOUT_SECONDS),
            ],
            environment=environment,
            check=False,
            capture_output=True,
            timeout=MAX_PROBE_SECONDS,
        )
        _emit_sanitized_result(
            result,
            (proxy_token.token_id, proxy_token.token_secret),
        )
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    finally:
        print("encoder batch probe cleanup: starting", file=sys.stderr, flush=True)
        try:
            manager.delete(proxy_token.token_id)
        finally:
            _stop_encoder_containers(modal_command, environment)
        print(
            "encoder batch probe cleanup: proxy token deleted, encoder stopped",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()
