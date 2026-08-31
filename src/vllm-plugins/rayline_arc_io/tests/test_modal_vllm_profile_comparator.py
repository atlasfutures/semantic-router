# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

E2E_DIR = Path(__file__).resolve().parents[4] / "e2e/testing/rayline-arc"
sys.path.insert(0, str(E2E_DIR))

import modal_vllm_profile_comparator as comparator  # noqa: E402
from modal_vllm_profile_comparator import (  # noqa: E402
    benchmark_turn_states,
    bootstrap_turn_states,
)
from modal_vllm_profile_contract import (  # noqa: E402
    BOOTSTRAP_REQUESTS_PER_PROFILE,
    MAXIMUM_POOLING_REQUESTS,
    MEASURED_REQUESTS_PER_PROFILE,
    PERF030_BUDGET,
    PROFILE_LABELS,
    PROFILES,
    STEPS,
    WARMUP_REQUESTS_PER_PROFILE,
)
from rayline_three_arm_budget import budget_receipt  # noqa: E402

EXPECTED_CANDIDATE_RATIO = 0.5


def test_perf030_history_states_are_strict_public_turn_prefixes() -> None:
    states = benchmark_turn_states()
    bootstrap = bootstrap_turn_states()

    assert len(states) == STEPS
    assert all(
        turn["role"] in {"user", "assistant"} for turns in states for turn in turns
    )
    assert states[1][: len(states[0])] == states[0]
    assert states[2][: len(states[1])] == states[1]
    assert all("text" in turn for turns in states for turn in turns)
    assert len(bootstrap) == BOOTSTRAP_REQUESTS_PER_PROFILE
    assert bootstrap[1][: len(bootstrap[0])] == bootstrap[0]
    assert bootstrap[2][: len(bootstrap[1])] == bootstrap[1]


def test_perf030_contract_is_bounded_and_profiles_only_one_backend_axis() -> None:
    receipt = budget_receipt(PERF030_BUDGET)

    assert (
        len(PROFILE_LABELS)
        * (WARMUP_REQUESTS_PER_PROFILE + MEASURED_REQUESTS_PER_PROFILE)
        == MAXIMUM_POOLING_REQUESTS
    )
    assert receipt["maximum_resource_envelope_usd"] < PERF030_BUDGET.packet_ceiling_usd
    assert receipt["provider_spend_usd"] == 0.0
    assert {profile.gdn_prefill_backend for profile in PROFILES.values()} == {
        "torch_reference",
        "flashinfer",
    }
    assert all("eager" in profile.engine_build_id for profile in PROFILES.values())


class FakeClient:
    def __init__(self, label: str) -> None:
        self.label = label
        self.sessions: dict[str, list[dict[str, str]]] = {}
        self.pooling_requests = 0

    def encode(self, episode_id: str, turns: list[dict[str, str]]) -> dict[str, Any]:
        summary, _embedding = self.encode_with_embedding(episode_id, turns)
        return summary

    def encode_with_embedding(
        self, episode_id: str, turns: list[dict[str, str]]
    ) -> tuple[dict[str, Any], tuple[float, ...]]:
        previous = self.sessions.get(episode_id)
        action = "created" if previous is None else "appended"
        self.sessions[episode_id] = list(turns)
        self.pooling_requests += 1
        return (
            {
                "action": action,
                "latency_seconds": 1.0,
                "serialized_tokens": len(turns) * 100,
                "appended_tokens": (len(turns) - len(previous or [])) * 100,
            },
            (1.0, 0.0),
        )

    def close(self, episode_id: str) -> None:
        self.sessions.pop(episode_id, None)

    def request(self, method: str, path: str) -> tuple[dict[str, Any], float]:
        if method == "DELETE":
            episode_id = path.rsplit("/", 1)[-1]
            return {"closed": self.sessions.pop(episode_id, None) is not None}, 0.0
        assert method == "GET" and path == "/health"
        return {
            "status": "ok",
            "resident_sessions": len(self.sessions),
            "resident_tokens": 0,
        }, 0.0


def test_perf030_full_comparator_accepts_a_faster_parity_candidate(
    monkeypatch: Any,
) -> None:
    clients = {label: FakeClient(label) for label in PROFILE_LABELS}

    monkeypatch.setattr(
        comparator,
        "read_encoder_snapshot",
        lambda client: {"label": client.label},
    )

    def stage_delta(**kwargs: Any) -> dict[str, Any]:
        inference = (
            EXPECTED_CANDIDATE_RATIO
            if kwargs["before"]["label"] == "flashinfer"
            else 1.0
        )
        return {
            "coordinator": {
                "tokenization_mean_seconds": 0.01,
                "coordinator_mean_seconds": inference + 0.02,
                "backend_mean_seconds": inference + 0.01,
            },
            "engine": {
                "queue_time_mean_seconds": 0.0,
                "inference_time_mean_seconds": inference,
                "e2e_time_mean_seconds": inference,
            },
        }

    monkeypatch.setattr(comparator, "encoder_stage_delta", stage_delta)
    report = comparator.run_comparison(clients, "test-perf030")

    assert report["status"] == "passed"
    assert report["candidate_decision"] == "accepted"
    assert report["correctness"]["passed"] is True
    assert (
        report["performance"]["candidate_to_reference_engine_inference_ratio"]
        == EXPECTED_CANDIDATE_RATIO
    )
    expected_requests = WARMUP_REQUESTS_PER_PROFILE + MEASURED_REQUESTS_PER_PROFILE
    assert all(
        client.pooling_requests == expected_requests for client in clients.values()
    )
    assert all(not client.sessions for client in clients.values())
