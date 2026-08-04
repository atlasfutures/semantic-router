#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Pre-provider remote-encoder parity gate for the AGT018 workload."""

from __future__ import annotations

import math
from typing import Any

from modal_fullstack_canary import _episode_id
from openrouter_kv_cache_successor_workload import (
    EXPECTED_SELECTION_TRACES,
    SCHEMA_VERSION as WORKLOAD_SCHEMA_VERSION,
    evaluate_embeddings,
    history_sequences,
    normalized_turns,
)

SCHEMA_VERSION = "rayline.openrouter-kv-cache-remote-parity.v1"


def _validate_transition(
    summary: dict[str, Any],
    step: int,
    previous_serialized_tokens: int | None,
) -> None:
    expected_action = "created" if step == 0 else "appended"
    if summary.get("action") != expected_action:
        raise RuntimeError("remote encoder parity session action diverged")
    if summary.get("revision") != step + 1:
        raise RuntimeError("remote encoder parity session revision diverged")
    serialized = summary.get("serialized_tokens")
    retained = summary.get("retained_prefix_tokens")
    appended = summary.get("appended_tokens")
    if not all(isinstance(value, int) for value in (serialized, retained, appended)):
        raise RuntimeError("remote encoder parity token telemetry is incomplete")
    if retained + appended != serialized:
        raise RuntimeError("remote encoder parity token accounting diverged")
    if step == 0 and retained != 0:
        raise RuntimeError("remote encoder parity create unexpectedly retained tokens")
    if step > 0 and retained != previous_serialized_tokens:
        raise RuntimeError(
            "remote encoder parity append did not retain the exact prefix"
        )


def verify_remote_encoder(client: Any, run_id: str) -> dict[str, Any]:
    """Verify all nine frozen states before any routed provider request."""

    embeddings: dict[tuple[str, int], tuple[float, ...]] = {}
    telemetry: dict[tuple[str, int], dict[str, Any]] = {}
    session_ids: list[str] = []
    failure: Exception | None = None
    cleanup_failures: list[Exception] = []
    try:
        for sequence in history_sequences():
            sequence_id = str(sequence["sequence_id"])
            session_id = _episode_id(run_id, f"agt018-parity:{sequence_id}")
            session_ids.append(session_id)
            previous_serialized_tokens = None
            for step, case in enumerate(sequence["states"]):
                summary, embedding = client.encode_with_embedding(
                    session_id,
                    normalized_turns(case),
                )
                _validate_transition(summary, step, previous_serialized_tokens)
                previous_serialized_tokens = int(summary["serialized_tokens"])
                embeddings[(sequence_id, step)] = embedding
                telemetry[(sequence_id, step)] = {
                    "action": summary["action"],
                    "revision": summary["revision"],
                    "serialized_tokens": summary["serialized_tokens"],
                    "retained_prefix_tokens": summary["retained_prefix_tokens"],
                    "appended_tokens": summary["appended_tokens"],
                    "latency_seconds": summary["latency_seconds"],
                }
    except Exception as error:
        failure = error
    finally:
        for session_id in session_ids:
            try:
                client.close_if_present(session_id)
            except Exception as error:
                cleanup_failures.append(error)
    if cleanup_failures:
        raise RuntimeError(
            "remote encoder parity sessions did not close cleanly"
        ) from cleanup_failures[0]
    if failure is not None:
        raise failure

    evaluated = evaluate_embeddings(embeddings)
    observations = []
    for row in evaluated["observations"]:
        key = (str(row["sequence_id"]), int(row["step"]))
        observations.append({**row, **telemetry[key]})
    latency_seconds = math.fsum(float(row["latency_seconds"]) for row in observations)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "workload_schema_version": WORKLOAD_SCHEMA_VERSION,
        "states": len(observations),
        "expected_selection_traces": {
            key: list(value) for key, value in EXPECTED_SELECTION_TRACES.items()
        },
        "observations": observations,
        "minimum_top_two_score_gap": evaluated["minimum_top_two_score_gap"],
        "selected_workers": evaluated["selected_workers"],
        "total_latency_seconds": latency_seconds,
        "sessions_closed": len(session_ids),
        "prompt_text_emitted": False,
        "raw_embeddings_emitted": False,
        "provider_calls": 0,
        "release_qualification_1000_executed": False,
    }
