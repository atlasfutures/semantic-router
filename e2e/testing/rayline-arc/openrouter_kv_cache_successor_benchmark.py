#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""AGT018 three-worker retained-versus-replay benchmark driver."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import openrouter_kv_cache_benchmark as base
from modal_fullstack_canary import _episode_id
from openrouter_agentic_stage_metrics import encoder_client_from_environment
from openrouter_kv_cache_journal import append as append_journal
from openrouter_kv_cache_journal import initialize as initialize_journal
from openrouter_kv_cache_successor_contract import MAX_COMPLETION_TOKENS, RUN_ID
from openrouter_kv_cache_successor_remote_gate import verify_remote_encoder
from openrouter_kv_cache_successor_workload import (
    EPISODES,
    EXPECTED_REQUESTS_PER_DEPLOYMENT,
    MODES,
    SEQUENCE_IDS,
    STEPS,
    history_sequences,
)

SCHEMA_VERSION = "rayline.openrouter-kv-cache-client.v2"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deployment", choices=("native_modal", "remote_vllm"), required=True
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--metrics-url", default="")
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--output", required=True)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser.parse_args()


def _journal_path(args: argparse.Namespace) -> Path | None:
    raw = getattr(args, "journal", "")
    return Path(raw) if raw else None


def _run_recorded_cell(
    *,
    args: argparse.Namespace,
    encoder_client: Any,
    journal_path: Path | None,
    sequence_id: str,
    case: dict[str, Any],
    mode: str,
    episode: int,
    step: int,
    ordinal: int,
) -> dict[str, Any]:
    label = f"agt018:{sequence_id}:{mode}:episode-{episode}"
    if mode == "replay":
        label += f":step-{step}"
    cell = {
        "ordinal": ordinal,
        "deployment": args.deployment,
        "run_id": args.run_id,
        "sequence_id": sequence_id,
        "mode": mode,
        "episode": episode,
        "step": step,
        "case_id": case["case_id"],
        "expected_worker": case["expected_worker"],
    }
    try:
        result = base._run_cell(
            args=args,
            encoder_client=encoder_client,
            case=case,
            episode_id=_episode_id(args.run_id, label),
            mode=mode,
            episode=episode,
            step=step,
        )
        if result["selected_worker"] != case["expected_worker"]:
            raise RuntimeError("AGT018 natural selection trace diverged")
        result.update(
            {
                "sequence_id": sequence_id,
                "expected_worker": case["expected_worker"],
            }
        )
    except Exception as error:
        if journal_path is not None:
            append_journal(
                journal_path,
                {
                    **cell,
                    "event": "request_failed",
                    "error": base._journal_failure(error),
                },
            )
        raise
    if journal_path is not None:
        append_journal(
            journal_path,
            {**cell, "event": "request_succeeded", "result": result},
        )
    return result


def _run_workload(
    *,
    args: argparse.Namespace,
    encoder_client: Any,
    journal_path: Path | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for sequence_index, sequence in enumerate(history_sequences()):
        sequence_id = str(sequence["sequence_id"])
        for episode in range(EPISODES):
            for step, case in enumerate(sequence["states"]):
                mode_order = (
                    MODES
                    if (sequence_index + episode + step) % 2 == 0
                    else tuple(reversed(MODES))
                )
                for mode in mode_order:
                    results.append(
                        _run_recorded_cell(
                            args=args,
                            encoder_client=encoder_client,
                            journal_path=journal_path,
                            sequence_id=sequence_id,
                            case=case,
                            mode=mode,
                            episode=episode,
                            step=step,
                            ordinal=len(results) + 1,
                        )
                    )
    return results


def _validate_results(results: list[dict[str, Any]]) -> None:
    if len(results) != EXPECTED_REQUESTS_PER_DEPLOYMENT:
        raise RuntimeError("AGT018 request envelope diverged")
    for sequence_id in SEQUENCE_IDS:
        for episode in range(EPISODES):
            for step in range(STEPS):
                pair = [
                    result
                    for result in results
                    if result["sequence_id"] == sequence_id
                    and result["episode"] == episode
                    and result["step"] == step
                ]
                if (
                    len(pair) != len(MODES)
                    or len({row["selected_worker"] for row in pair}) != 1
                    or any(
                        row["selected_worker"] != row["expected_worker"] for row in pair
                    )
                ):
                    raise RuntimeError("AGT018 retained/replay selection parity failed")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.run_id != RUN_ID:
        raise RuntimeError("AGT018 run identity diverged")
    if args.deployment == "remote_vllm" and not args.metrics_url:
        raise RuntimeError("remote vLLM comparison requires the router metrics URL")
    encoder_client = (
        encoder_client_from_environment(args.timeout_seconds)
        if args.deployment == "remote_vllm"
        else None
    )
    journal_path = _journal_path(args)
    if journal_path is not None:
        initialize_journal(journal_path)

    remote_parity = None
    if encoder_client is not None:
        remote_parity = verify_remote_encoder(encoder_client, args.run_id)

    started = time.perf_counter()
    results = _run_workload(
        args=args,
        encoder_client=encoder_client,
        journal_path=journal_path,
    )
    _validate_results(results)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "deployment": args.deployment,
        "status": "client_passed",
        "pre_provider_remote_encoder_parity": remote_parity,
        "workload": {
            "sequence_ids": list(SEQUENCE_IDS),
            "episodes_per_sequence": EPISODES,
            "steps": STEPS,
            "modes": list(MODES),
            "requests": EXPECTED_REQUESTS_PER_DEPLOYMENT,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "routing": "natural_rayline_selection",
        },
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
    }


def main() -> None:
    args = _args()
    report = run(args)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    for name in (
        "OPENROUTER_EPHEMERAL_API_KEY",
        "RAYLINE_MODAL_NATIVE_ROUTER_TOKEN",
        "RAYLINE_ARC_E2E_MODAL_KEY",
        "RAYLINE_ARC_E2E_MODAL_SECRET",
    ):
        value = os.environ.get(name, "")
        if value and value in encoded:
            raise RuntimeError("credential entered AGT018 client report")
    Path(args.output).write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
