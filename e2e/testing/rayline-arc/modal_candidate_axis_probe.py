#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Find one robust real-embedding axis without emitting prompt/vector data."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections.abc import Sequence
from typing import Any

from modal_fullstack_inputs import CANDIDATE_PROMPTS
from modal_session_canary import EMBEDDING_DIMENSION, CanaryClient, _episode_hash

MIN_SIGN_GROUP_COUNT = 6
MIN_ABSOLUTE_MARGIN = 0.0001


def _normalize(embedding: Sequence[float]) -> tuple[float, ...]:
    norm = math.sqrt(math.fsum(float(value) ** 2 for value in embedding))
    if not math.isfinite(norm) or norm == 0:
        raise RuntimeError("candidate embedding has invalid norm")
    return tuple(float(value) / norm for value in embedding)


def select_axis(embeddings: Sequence[Sequence[float]]) -> dict[str, Any]:
    if len(embeddings) != len(CANDIDATE_PROMPTS):
        raise RuntimeError("candidate embedding count does not match frozen inputs")
    normalized = [_normalize(embedding) for embedding in embeddings]
    if any(len(embedding) != EMBEDDING_DIMENSION for embedding in normalized):
        raise RuntimeError("candidate embedding dimension mismatch")

    ranked: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for axis in range(EMBEDDING_DIMENSION):
        values = [embedding[axis] for embedding in normalized]
        positive_count = sum(value > 0 for value in values)
        negative_count = sum(value < 0 for value in values)
        zero_count = len(values) - positive_count - negative_count
        if not positive_count or not negative_count or zero_count:
            continue
        absolute = sorted(abs(value) for value in values)
        summary = {
            "axis_index": axis,
            "sign_counts": {
                "negative": negative_count,
                "positive": positive_count,
                "zero": zero_count,
            },
            "absolute_margin": {
                "minimum": absolute[0],
                "p50": statistics.median(absolute),
                "maximum": absolute[-1],
            },
        }
        rank = (
            float(min(positive_count, negative_count)),
            absolute[0],
            float(statistics.median(absolute)),
            float(-axis),
        )
        ranked.append((rank, summary))

    if not ranked:
        raise RuntimeError("no candidate axis contains both signs without zeros")
    _rank, selected = max(ranked, key=lambda item: item[0])
    minority_count = min(
        selected["sign_counts"]["negative"],
        selected["sign_counts"]["positive"],
    )
    if minority_count < MIN_SIGN_GROUP_COUNT:
        raise RuntimeError("best candidate axis does not meet sign-balance gate")
    if selected["absolute_margin"]["minimum"] < MIN_ABSOLUTE_MARGIN:
        raise RuntimeError("best candidate axis does not meet robustness gate")
    return selected


def run_probe(client: CanaryClient, run_id: str) -> dict[str, Any]:
    embeddings: list[tuple[float, ...]] = []
    latencies: list[float] = []
    for index, prompt in enumerate(CANDIDATE_PROMPTS):
        episode_id = _episode_hash(f"{run_id}:candidate-axis:{index}")
        summary, embedding = client.encode_with_embedding(
            episode_id,
            [{"role": "user", "text": prompt}],
        )
        client.close(episode_id)
        latencies.append(float(summary["latency_seconds"]))
        embeddings.append(embedding)
    selected = select_axis(embeddings)

    health, _latency = client.request("GET", "/health")
    if health.get("resident_sessions") != 0:
        raise RuntimeError("candidate-axis sessions leaked after explicit close")
    return {
        "schema_version": "rayline.arc.candidate-axis-probe.v1",
        "run_id": run_id,
        "status": "passed",
        "candidate_count": len(CANDIDATE_PROMPTS),
        "embedding_dimension": EMBEDDING_DIMENSION,
        "selection_order": [
            "maximum_minority_sign_count",
            "maximum_minimum_absolute_margin",
            "maximum_median_absolute_margin",
            "lowest_axis_index",
        ],
        "acceptance": {
            "minimum_sign_group_count": MIN_SIGN_GROUP_COUNT,
            "minimum_absolute_margin": MIN_ABSOLUTE_MARGIN,
        },
        "selected": selected,
        "latency_seconds": {
            "minimum": min(latencies),
            "p50": statistics.median(latencies),
            "maximum": max(latencies),
        },
        "resident_sessions_after_close": health["resident_sessions"],
        "raw_embeddings_emitted": False,
        "prompt_text_emitted": False,
        "generation_requests": 0,
        "provider_calls": 0,
        "release_qualification_1000_executed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()

    modal_key = os.environ.get("RAYLINE_ARC_MODAL_KEY", "")
    modal_secret = os.environ.get("RAYLINE_ARC_MODAL_SECRET", "")
    if not modal_key or not modal_secret:
        raise SystemExit(
            "RAYLINE_ARC_MODAL_KEY and RAYLINE_ARC_MODAL_SECRET are required"
        )
    report = run_probe(
        CanaryClient(
            base_url=args.base_url,
            modal_key=modal_key,
            modal_secret=modal_secret,
            timeout_seconds=args.timeout_seconds,
        ),
        args.run_id,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
