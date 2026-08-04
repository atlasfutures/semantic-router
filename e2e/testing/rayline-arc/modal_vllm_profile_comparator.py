# SPDX-License-Identifier: Apache-2.0

"""Compare vLLM GDN backends on identical retained-session histories.

Inputs are public and synthetic. The result contains aggregate timings and
drift metrics only; prompt text and embedding values are never emitted.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

from modal_session_canary import CanaryClient, _episode_hash
from modal_session_qualification import (
    _latency_summary,
    _synthetic_metrics,
    _vector_metrics,
)
from modal_vllm_profile_contract import (
    BOOTSTRAP_REQUESTS_PER_PROFILE,
    CANDIDATE_LABEL,
    EPISODES,
    MAX_ABSOLUTE_DRIFT,
    MAX_CANDIDATE_TO_REFERENCE_ENGINE_RATIO,
    MAX_SELECTION_FLIPS,
    MAX_SYNTHETIC_SCORE_DRIFT,
    MEASURED_REQUESTS_PER_PROFILE,
    MIN_COSINE_SIMILARITY,
    MODES,
    PROFILE_LABELS,
    PROFILES,
    REFERENCE_LABEL,
    STEPS,
    WARMUP_REQUESTS_PER_PROFILE,
)
from openrouter_agentic_stage_metrics import (
    encoder_stage_delta,
    read_encoder_snapshot,
)
from openrouter_agentic_workload import candidate_case
from openrouter_kv_cache_benchmark import history_states


def _stable_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def benchmark_turn_states() -> list[list[dict[str, str]]]:
    """Project the realistic OpenAI-shaped fixture into ARC's turn schema."""
    cases = history_states()
    tools = _stable_json(candidate_case(2)["tools"])
    result: list[list[dict[str, str]]] = []
    for case in cases:
        turns: list[dict[str, str]] = []
        for index, message in enumerate(case["messages"]):
            source_role = str(message["role"])
            role = "assistant" if source_role == "assistant" else "user"
            text = str(message.get("content") or "")
            if index == 0:
                text = f"Available tools (public JSON): {tools}\n{text}"
            if message.get("tool_calls"):
                text += f"\nTool calls: {_stable_json(message['tool_calls'])}"
            if source_role == "tool":
                text = f"Tool result {message.get('tool_call_id', '')}:\n{text}"
            elif source_role == "system":
                text = f"System instruction:\n{text}"
            turns.append({"role": role, "text": text})
        result.append(turns)
    if len(result) != STEPS:
        raise RuntimeError("PERF029 history-state count diverged")
    for left, right in pairwise(result):
        if right[: len(left)] != left:
            raise RuntimeError("PERF029 histories are not strict turn prefixes")
    return result


def bootstrap_turn_states() -> list[list[dict[str, str]]]:
    """Prime cold JIT paths before presenting the first realistic history."""
    states: list[list[dict[str, str]]] = []
    turns: list[dict[str, str]] = []
    for step, repetitions in enumerate((1, 64, 256)):
        turns.append(
            {
                "role": "user" if step % 2 == 0 else "assistant",
                "text": (f"Public staged kernel bootstrap {step}. " * repetitions),
            }
        )
        states.append(list(turns))
    if len(states) != BOOTSTRAP_REQUESTS_PER_PROFILE:
        raise RuntimeError("PERF029 bootstrap-state count diverged")
    return states


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise RuntimeError("PERF029 cannot summarize an empty sample")
    return math.fsum(values) / len(values)


def _timing_summary(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    client = [float(sample["summary"]["latency_seconds"]) for sample in samples]
    stage_fields = {
        "tokenization_mean_seconds": ("coordinator", "tokenization_mean_seconds"),
        "coordinator_mean_seconds": ("coordinator", "coordinator_mean_seconds"),
        "backend_mean_seconds": ("coordinator", "backend_mean_seconds"),
        "engine_queue_mean_seconds": ("engine", "queue_time_mean_seconds"),
        "engine_inference_mean_seconds": ("engine", "inference_time_mean_seconds"),
        "engine_e2e_mean_seconds": ("engine", "e2e_time_mean_seconds"),
    }
    stage_means = {
        output: _mean([float(sample["stage"][section][field]) for sample in samples])
        for output, (section, field) in stage_fields.items()
    }
    return {
        "requests": len(samples),
        "client_latency": {
            **_latency_summary(client),
            "mean_seconds": _mean(client),
        },
        "stage_means": stage_means,
        "serialized_tokens": {
            "minimum": min(
                int(sample["summary"]["serialized_tokens"]) for sample in samples
            ),
            "maximum": max(
                int(sample["summary"]["serialized_tokens"]) for sample in samples
            ),
            "mean": _mean(
                [float(sample["summary"]["serialized_tokens"]) for sample in samples]
            ),
        },
        "appended_tokens_total": sum(
            int(sample["summary"]["appended_tokens"]) for sample in samples
        ),
    }


def _parity_summary(
    pairs: Sequence[tuple[tuple[float, ...], tuple[float, ...]]],
) -> dict[str, Any]:
    vector_metrics = [_vector_metrics(left, right) for left, right in pairs]
    synthetic = [_synthetic_metrics(left, right) for left, right in pairs]
    result = {
        "comparisons": len(pairs),
        "minimum_cosine_similarity": min(
            metric["cosine_similarity"] for metric in vector_metrics
        ),
        "maximum_absolute_drift": max(
            metric["max_absolute_drift"] for metric in vector_metrics
        ),
        "maximum_l2_drift": max(metric["l2_drift"] for metric in vector_metrics),
        "maximum_synthetic_score_drift": max(score for score, _flip in synthetic),
        "synthetic_selection_flips": sum(int(flip) for _score, flip in synthetic),
    }
    result["passed"] = (
        result["minimum_cosine_similarity"] >= MIN_COSINE_SIMILARITY
        and result["maximum_absolute_drift"] <= MAX_ABSOLUTE_DRIFT
        and result["maximum_synthetic_score_drift"] <= MAX_SYNTHETIC_SCORE_DRIFT
        and result["synthetic_selection_flips"] <= MAX_SELECTION_FLIPS
    )
    return result


def _expect_action(summary: Mapping[str, Any], expected: str) -> None:
    if summary.get("action") != expected:
        raise RuntimeError(
            f"PERF029 expected session action {expected}, got {summary.get('action')}"
        )


def _health_is_empty(client: CanaryClient) -> None:
    health, _elapsed = client.request("GET", "/health")
    if health.get("resident_sessions") != 0 or health.get("resident_tokens") != 0:
        raise RuntimeError("PERF029 retained-session cleanup is not empty")


def _close_if_present(client: CanaryClient, episode_id: str) -> None:
    response, _elapsed = client.request(
        "DELETE", f"/v1/rayline/arc/session/{episode_id}"
    )
    if not isinstance(response.get("closed"), bool):
        raise TypeError("PERF029 session close response omitted boolean status")


def _close_after_phase(
    client: CanaryClient,
    episode_id: str,
    *,
    phase_completed: bool,
) -> None:
    if phase_completed:
        _close_if_present(client, episode_id)
        return
    with contextlib.suppress(BaseException):
        _close_if_present(client, episode_id)


def _warm_profile(
    label: str,
    client: CanaryClient,
    states: Sequence[list[dict[str, str]]],
    run_id: str,
    phase: str,
) -> None:
    episode_id = _episode_hash(f"{run_id}:warmup:{phase}:{label}")
    phase_completed = False
    try:
        for step, turns in enumerate(states):
            summary = client.encode(episode_id, turns)
            _expect_action(summary, "created" if step == 0 else "appended")
        phase_completed = True
    finally:
        _close_after_phase(client, episode_id, phase_completed=phase_completed)
    _health_is_empty(client)


def _measure_one(
    client: CanaryClient,
    episode_id: str,
    turns: list[dict[str, str]],
) -> tuple[dict[str, Any], tuple[float, ...]]:
    before = read_encoder_snapshot(client)
    summary, embedding = client.encode_with_embedding(episode_id, turns)
    after = read_encoder_snapshot(client)
    return {
        "summary": summary,
        "stage": encoder_stage_delta(before=before, after=after, requests=1),
    }, embedding


def _measure_cell(
    *,
    clients: Mapping[str, CanaryClient],
    retained_ids: Mapping[str, str],
    turns: list[dict[str, str]],
    run_id: str,
    episode: int,
    step: int,
    mode: str,
    profile_order: Sequence[str],
    samples: dict[str, list[dict[str, Any]]],
    vectors: dict[tuple[str, int, int, str], tuple[float, ...]],
) -> None:
    for label in profile_order:
        episode_id = (
            retained_ids[label]
            if mode == "retained"
            else _episode_hash(f"{run_id}:{label}:replay:{episode}:{step}")
        )
        sample, vector = _measure_one(clients[label], episode_id, turns)
        expected_action = "created" if mode == "replay" or step == 0 else "appended"
        _expect_action(sample["summary"], expected_action)
        sample.update({"episode": episode, "step": step, "mode": mode})
        samples[label].append(sample)
        vectors[(label, episode, step, mode)] = vector
        if mode == "replay":
            clients[label].close(episode_id)


def _measure_profiles(
    clients: Mapping[str, CanaryClient],
    states: Sequence[list[dict[str, str]]],
    run_id: str,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[tuple[str, int, int, str], tuple[float, ...]],
]:
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    vectors: dict[tuple[str, int, int, str], tuple[float, ...]] = {}
    cell = 0
    for episode in range(EPISODES):
        phase_completed = False
        retained_ids = {
            label: _episode_hash(f"{run_id}:{label}:retained:{episode}")
            for label in PROFILE_LABELS
        }
        try:
            for step, turns in enumerate(states):
                for mode in MODES:
                    profile_order = (
                        PROFILE_LABELS
                        if cell % 2 == 0
                        else tuple(reversed(PROFILE_LABELS))
                    )
                    _measure_cell(
                        clients=clients,
                        retained_ids=retained_ids,
                        turns=turns,
                        run_id=run_id,
                        episode=episode,
                        step=step,
                        mode=mode,
                        profile_order=profile_order,
                        samples=samples,
                        vectors=vectors,
                    )
                    cell += 1
            phase_completed = True
        finally:
            for label in PROFILE_LABELS:
                _close_after_phase(
                    clients[label],
                    retained_ids[label],
                    phase_completed=phase_completed,
                )
    return samples, vectors


def _parity_reports(
    vectors: Mapping[tuple[str, int, int, str], tuple[float, ...]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    within_profile: dict[str, Any] = {}
    for label in PROFILE_LABELS:
        pairs = [
            (
                vectors[(label, episode, step, "retained")],
                vectors[(label, episode, step, "replay")],
            )
            for episode in range(EPISODES)
            for step in range(STEPS)
        ]
        within_profile[label] = _parity_summary(pairs)
    cross_profile = _parity_summary(
        [
            (
                vectors[(REFERENCE_LABEL, episode, step, mode)],
                vectors[(CANDIDATE_LABEL, episode, step, mode)],
            )
            for episode in range(EPISODES)
            for step in range(STEPS)
            for mode in MODES
        ]
    )
    return within_profile, cross_profile


def _timing_reports(
    samples: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        label: {
            "all": _timing_summary(samples[label]),
            **{
                mode: _timing_summary(
                    [sample for sample in samples[label] if sample["mode"] == mode]
                )
                for mode in MODES
            },
        }
        for label in PROFILE_LABELS
    }


def _build_report(
    *,
    run_id: str,
    bootstrap_states: Sequence[list[dict[str, str]]],
    states: Sequence[list[dict[str, str]]],
    samples: Mapping[str, list[dict[str, Any]]],
    vectors: Mapping[tuple[str, int, int, str], tuple[float, ...]],
) -> dict[str, Any]:
    within_profile, cross_profile = _parity_reports(vectors)
    timing = _timing_reports(samples)
    reference_engine = timing[REFERENCE_LABEL]["all"]["stage_means"][
        "engine_inference_mean_seconds"
    ]
    candidate_engine = timing[CANDIDATE_LABEL]["all"]["stage_means"][
        "engine_inference_mean_seconds"
    ]
    if reference_engine <= 0:
        raise RuntimeError("PERF029 reference engine timing is not positive")
    engine_ratio = candidate_engine / reference_engine
    correctness_passed = (
        all(report["passed"] for report in within_profile.values())
        and cross_profile["passed"]
    )
    performance_passed = engine_ratio <= MAX_CANDIDATE_TO_REFERENCE_ENGINE_RATIO
    candidate_accepted = correctness_passed and performance_passed

    workload_digest = hashlib.sha256(_stable_json(states).encode()).hexdigest()
    return {
        "schema_version": "rayline.arc.modal-vllm-profile-comparison.v1",
        "run_id": run_id,
        "status": "passed",
        "candidate_decision": "accepted" if candidate_accepted else "rejected",
        "profiles": {
            label: {
                "app_name": PROFILES[label].app_name,
                "gdn_prefill_backend": PROFILES[label].gdn_prefill_backend,
                "engine_build_id": PROFILES[label].engine_build_id,
                "enforce_eager": True,
            }
            for label in PROFILE_LABELS
        },
        "workload": {
            "sha256": workload_digest,
            "bootstrap_sha256": hashlib.sha256(
                _stable_json(bootstrap_states).encode()
            ).hexdigest(),
            "bootstrap_history_states": len(bootstrap_states),
            "episodes": EPISODES,
            "history_states_per_episode": STEPS,
            "modes": list(MODES),
            "warmup_requests_per_profile": WARMUP_REQUESTS_PER_PROFILE,
            "measured_requests_per_profile": MEASURED_REQUESTS_PER_PROFILE,
            "public_synthetic": True,
        },
        "correctness": {
            "within_profile_retained_vs_replay": within_profile,
            "cross_profile_reference_vs_candidate": cross_profile,
            "passed": correctness_passed,
        },
        "performance": {
            "timing": timing,
            "candidate_to_reference_engine_inference_ratio": engine_ratio,
            "maximum_accepted_engine_inference_ratio": (
                MAX_CANDIDATE_TO_REFERENCE_ENGINE_RATIO
            ),
            "passed": performance_passed,
            "primary_gate_uses_engine_internal_time": True,
            "client_latency_is_diagnostic_only": True,
        },
        "provider_calls": 0,
        "raw_embeddings_emitted": False,
        "prompt_text_emitted": False,
        "release_qualification_1000_executed": False,
    }


def run_comparison(
    clients: Mapping[str, CanaryClient],
    run_id: str,
) -> dict[str, Any]:
    if tuple(clients) != PROFILE_LABELS:
        raise RuntimeError("PERF029 clients are not in the frozen profile order")
    bootstrap_states = bootstrap_turn_states()
    states = benchmark_turn_states()
    for label in PROFILE_LABELS:
        _warm_profile(label, clients[label], bootstrap_states, run_id, "bootstrap")
        _warm_profile(label, clients[label], states, run_id, "exact-shape")

    samples, vectors = _measure_profiles(clients, states, run_id)
    for label in PROFILE_LABELS:
        if len(samples[label]) != MEASURED_REQUESTS_PER_PROFILE:
            raise RuntimeError("PERF029 measured request count diverged")
        _health_is_empty(clients[label])
    return _build_report(
        run_id=run_id,
        bootstrap_states=bootstrap_states,
        states=states,
        samples=samples,
        vectors=vectors,
    )
