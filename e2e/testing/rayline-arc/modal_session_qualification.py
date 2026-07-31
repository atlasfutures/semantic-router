#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Frozen 128-case retained-session development qualification.

This is deliberately smaller than the held 1,000-case release gate. Inputs are
public and synthetic; output contains only aggregate metrics and never emits
credentials, turn text, or embedding values.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import sys
import time
from typing import Any

from modal_session_canary import CanaryClient, _episode_hash

CASE_COUNT = 128
EPISODES_PER_SHAPE = 8
TURNS_PER_EPISODE = 4
CONCURRENT_EPISODES = 8
MAX_ALLOWED_CASES = 200
MIN_COSINE_SIMILARITY = 0.9999
MAX_ABSOLUTE_DRIFT = 0.01
MAX_SYNTHETIC_SCORE_DRIFT = 0.005
MAX_RETAINED_TO_REPLAY_TOKEN_RATIO = 0.75
MAX_CONCURRENCY_WALL_TO_SUM_RATIO = 0.85
SHAPES = (
    ("short", 1),
    ("medium", 8),
    ("tool_dump", 32),
    ("long", 96),
)


def _vector_metrics(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> dict[str, float]:
    if len(left) != len(right) or not left:
        raise RuntimeError("qualification embeddings are not shape-aligned")
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    differences = [a - b for a, b in zip(left, right, strict=True)]
    return {
        "cosine_similarity": dot / (left_norm * right_norm),
        "max_absolute_drift": max(abs(value) for value in differences),
        "l2_drift": math.sqrt(math.fsum(value * value for value in differences)),
    }


def _synthetic_scores(vector: tuple[float, ...]) -> tuple[float, ...]:
    scale = 1.0 / math.sqrt(len(vector))
    return tuple(
        math.fsum(
            value * math.sin((arm + 1) * (index + 1) * 0.017)
            for index, value in enumerate(vector)
        )
        * scale
        for arm in range(4)
    )


def _synthetic_metrics(
    retained: tuple[float, ...],
    replay: tuple[float, ...],
) -> tuple[float, bool]:
    left = _synthetic_scores(retained)
    right = _synthetic_scores(replay)
    score_drift = max(abs(a - b) for a, b in zip(left, right, strict=True))
    return score_drift, left.index(max(left)) != right.index(max(right))


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[rank]


def _latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "p50_seconds": _percentile(values, 0.50),
        "p95_seconds": _percentile(values, 0.95),
        "p99_seconds": _percentile(values, 0.99),
        "max_seconds": max(values),
    }


def _turn(shape: str, repetitions: int, episode: int, step: int) -> dict[str, str]:
    unit = f"Public synthetic {shape} routing evidence {episode}-{step}. "
    return {
        "role": "user" if step % 2 == 0 else "assistant",
        "text": unit * repetitions,
    }


def _expect_action(summary: dict[str, Any], expected: str) -> None:
    if summary["action"] != expected:
        raise RuntimeError(
            f"expected retained-session action {expected}, got {summary['action']}"
        )


def _close_if_present(client: CanaryClient, episode_id: str) -> None:
    response, _elapsed = client.request(
        "DELETE",
        f"/v1/rayline/arc/session/{episode_id}",
    )
    if not isinstance(response.get("closed"), bool):
        raise RuntimeError("session close response omitted boolean status")


def _run_parity_cases(client: CanaryClient, run_id: str) -> dict[str, Any]:
    retained_latencies: list[float] = []
    replay_latencies: list[float] = []
    cosine_values: list[float] = []
    absolute_drifts: list[float] = []
    l2_drifts: list[float] = []
    score_drifts: list[float] = []
    selection_flips = 0
    retained_appended_tokens = 0
    replay_serialized_tokens = 0
    cases = 0

    for shape, repetitions in SHAPES:
        for episode in range(EPISODES_PER_SHAPE):
            retained_id = _episode_hash(f"{run_id}:retained:{shape}:{episode}")
            turns: list[dict[str, str]] = []
            for step in range(TURNS_PER_EPISODE):
                turns.append(_turn(shape, repetitions, episode, step))
                retained_summary, retained_vector = client.encode_with_embedding(
                    retained_id,
                    turns,
                )
                replay_id = _episode_hash(f"{run_id}:replay:{shape}:{episode}:{step}")
                replay_summary, replay_vector = client.encode_with_embedding(
                    replay_id,
                    turns,
                )
                client.close(replay_id)

                _expect_action(
                    retained_summary,
                    "created" if step == 0 else "appended",
                )
                _expect_action(replay_summary, "created")
                if (
                    retained_summary["serialized_tokens"]
                    != replay_summary["serialized_tokens"]
                ):
                    raise RuntimeError("retained and replay token counts diverged")

                vector_metrics = _vector_metrics(retained_vector, replay_vector)
                score_drift, selection_flipped = _synthetic_metrics(
                    retained_vector,
                    replay_vector,
                )
                cosine_values.append(vector_metrics["cosine_similarity"])
                absolute_drifts.append(vector_metrics["max_absolute_drift"])
                l2_drifts.append(vector_metrics["l2_drift"])
                score_drifts.append(score_drift)
                selection_flips += int(selection_flipped)
                retained_latencies.append(retained_summary["latency_seconds"])
                replay_latencies.append(replay_summary["latency_seconds"])
                retained_appended_tokens += retained_summary["appended_tokens"]
                replay_serialized_tokens += replay_summary["serialized_tokens"]
                cases += 1
            client.close(retained_id)
        print(
            f"qualification parity: {cases}/{CASE_COUNT} history states complete",
            file=sys.stderr,
            flush=True,
        )

    if cases != CASE_COUNT:
        raise RuntimeError(
            f"qualification produced {cases} cases, expected {CASE_COUNT}"
        )
    token_ratio = retained_appended_tokens / replay_serialized_tokens
    if min(cosine_values) < MIN_COSINE_SIMILARITY:
        raise RuntimeError("retained/replay cosine similarity gate failed")
    if max(absolute_drifts) > MAX_ABSOLUTE_DRIFT:
        raise RuntimeError("retained/replay absolute drift gate failed")
    if max(score_drifts) > MAX_SYNTHETIC_SCORE_DRIFT or selection_flips:
        raise RuntimeError("retained/replay synthetic selection gate failed")
    if token_ratio > MAX_RETAINED_TO_REPLAY_TOKEN_RATIO:
        raise RuntimeError("retained token-efficiency gate failed")

    return {
        "cases": cases,
        "parity": {
            "minimum_cosine_similarity": min(cosine_values),
            "maximum_absolute_drift": max(absolute_drifts),
            "maximum_l2_drift": max(l2_drifts),
            "maximum_synthetic_score_drift": max(score_drifts),
            "synthetic_selection_flips": selection_flips,
        },
        "tokens": {
            "retained_appended": retained_appended_tokens,
            "full_replay_serialized": replay_serialized_tokens,
            "retained_to_replay_ratio": token_ratio,
        },
        "latency": {
            "retained": _latency_summary(retained_latencies),
            "full_replay": _latency_summary(replay_latencies),
        },
    }


def _run_concurrency(client: CanaryClient, run_id: str) -> dict[str, Any]:
    episode_ids = [
        _episode_hash(f"{run_id}:concurrent:{index}")
        for index in range(CONCURRENT_EPISODES)
    ]
    first_turn = [_turn("concurrent", 16, 0, 0)]
    second_turns = [*first_turn, _turn("concurrent", 16, 0, 1)]

    def concurrent_encode(
        turns: list[dict[str, str]],
    ) -> tuple[list[dict[str, Any]], float]:
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=CONCURRENT_EPISODES
        ) as executor:
            futures = [
                executor.submit(client.encode, episode_id, turns)
                for episode_id in episode_ids
            ]
            results = [future.result() for future in futures]
        return results, time.perf_counter() - started

    created, created_wall = concurrent_encode(first_turn)
    appended, appended_wall = concurrent_encode(second_turns)
    if any(result["action"] != "created" for result in created):
        raise RuntimeError("cross-episode concurrent create action diverged")
    if any(result["action"] != "appended" for result in appended):
        raise RuntimeError("cross-episode concurrent append action diverged")
    wall_to_sum_ratios = [
        created_wall / sum(result["latency_seconds"] for result in created),
        appended_wall / sum(result["latency_seconds"] for result in appended),
    ]
    if max(wall_to_sum_ratios) > MAX_CONCURRENCY_WALL_TO_SUM_RATIO:
        raise RuntimeError("cross-episode concurrency overlap gate failed")
    for episode_id in episode_ids:
        client.close(episode_id)

    same_id = _episode_hash(f"{run_id}:same-episode")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        same_results = [
            future.result()
            for future in (
                executor.submit(client.encode, same_id, first_turn),
                executor.submit(client.encode, same_id, first_turn),
            )
        ]
    if sorted(result["action"] for result in same_results) != ["created", "reused"]:
        raise RuntimeError("same-episode serialization gate failed")
    client.close(same_id)

    return {
        "episodes": CONCURRENT_EPISODES,
        "create_wall_seconds": created_wall,
        "append_wall_seconds": appended_wall,
        "wall_to_sum_latency_ratio": wall_to_sum_ratios,
        "same_episode_actions": sorted(result["action"] for result in same_results),
    }


def _run_rebuilds(client: CanaryClient, run_id: str) -> dict[str, Any]:
    turns = [_turn("rebuild", 8, 0, 0)]
    eviction_ids = [_episode_hash(f"{run_id}:eviction:{index}") for index in range(9)]
    first_summary, first_vector = client.encode_with_embedding(eviction_ids[0], turns)
    _expect_action(first_summary, "created")
    for episode_id in eviction_ids[1:]:
        _expect_action(client.encode(episode_id, turns), "created")
    rebuilt_summary, rebuilt_vector = client.encode_with_embedding(
        eviction_ids[0],
        turns,
    )
    _expect_action(rebuilt_summary, "created")
    eviction_metrics = _vector_metrics(first_vector, rebuilt_vector)
    if eviction_metrics["cosine_similarity"] < MIN_COSINE_SIMILARITY:
        raise RuntimeError("LRU eviction rebuild parity gate failed")
    for episode_id in eviction_ids:
        _close_if_present(client, episode_id)

    affinity_id = _episode_hash(f"{run_id}:affinity-loss")
    affinity_turns = [turns[0], _turn("rebuild", 8, 0, 1)]
    client.encode(affinity_id, turns)
    before_summary, before_vector = client.encode_with_embedding(
        affinity_id,
        affinity_turns,
    )
    _expect_action(before_summary, "appended")
    client.close(affinity_id)
    after_summary, after_vector = client.encode_with_embedding(
        affinity_id,
        affinity_turns,
    )
    _expect_action(after_summary, "created")
    affinity_metrics = _vector_metrics(before_vector, after_vector)
    if affinity_metrics["cosine_similarity"] < MIN_COSINE_SIMILARITY:
        raise RuntimeError("affinity-loss rebuild parity gate failed")
    client.close(affinity_id)

    health, _elapsed = client.request("GET", "/health")
    if health.get("resident_sessions") != 0 or health.get("resident_tokens") != 0:
        raise RuntimeError("qualification leaked retained sessions")
    return {
        "lru_eviction_action": rebuilt_summary["action"],
        "lru_eviction_cosine_similarity": eviction_metrics["cosine_similarity"],
        "affinity_loss_action": after_summary["action"],
        "affinity_loss_cosine_similarity": affinity_metrics["cosine_similarity"],
        "resident_sessions_after_cleanup": health["resident_sessions"],
        "resident_tokens_after_cleanup": health["resident_tokens"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cases", type=int, default=CASE_COUNT)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()
    if args.cases != CASE_COUNT or args.cases > MAX_ALLOWED_CASES:
        raise SystemExit(
            f"this frozen qualification requires exactly {CASE_COUNT} cases"
        )

    modal_key = os.environ.get("RAYLINE_ARC_MODAL_KEY", "")
    modal_secret = os.environ.get("RAYLINE_ARC_MODAL_SECRET", "")
    if not modal_key or not modal_secret:
        raise SystemExit(
            "RAYLINE_ARC_MODAL_KEY and RAYLINE_ARC_MODAL_SECRET are required"
        )
    client = CanaryClient(
        base_url=args.base_url,
        modal_key=modal_key,
        modal_secret=modal_secret,
        timeout_seconds=args.timeout_seconds,
    )

    started = time.perf_counter()
    parity = _run_parity_cases(client, args.run_id)
    print("qualification concurrency: starting", file=sys.stderr, flush=True)
    concurrency = _run_concurrency(client, args.run_id)
    print("qualification rebuilds: starting", file=sys.stderr, flush=True)
    rebuilds = _run_rebuilds(client, args.run_id)
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": "rayline.arc.modal-session-qualification.v1",
        "run_id": args.run_id,
        "status": "passed",
        "case_count": CASE_COUNT,
        "workload": {
            "shapes": [shape for shape, _repetitions in SHAPES],
            "episodes_per_shape": EPISODES_PER_SHAPE,
            "turns_per_episode": TURNS_PER_EPISODE,
        },
        "parity": parity,
        "concurrency": concurrency,
        "rebuilds": rebuilds,
        "elapsed_seconds": elapsed,
        "history_states_per_second": CASE_COUNT / elapsed,
        "provider_calls": 0,
        "automatic_prefix_cache_enabled": False,
        "release_qualification_1000_executed": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
