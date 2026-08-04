# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import json

import openrouter_kv_cache_successor_contract as successor_contract
import openrouter_kv_cache_successor_workload as workload
from openrouter_agentic_workload import WORKERS

EXPECTED_REQUESTS_PER_DEPLOYMENT = 36
EXPECTED_MAXIMUM_EXTERNAL_ATTEMPTS = 156


def test_successor_histories_are_public_strict_prefixes() -> None:
    sequences = workload.history_sequences()
    assert [sequence["sequence_id"] for sequence in sequences] == list(
        workload.SEQUENCE_IDS
    )
    assert workload.EXPECTED_REQUESTS_PER_DEPLOYMENT == EXPECTED_REQUESTS_PER_DEPLOYMENT
    for sequence in sequences:
        states = sequence["states"]
        assert len(states) == workload.STEPS
        assert [state["expected_worker"] for state in states] == sequence[
            "expected_selection_trace"
        ]
        for previous, current in itertools.pairwise(states):
            prior_messages = previous["messages"]
            current_messages = current["messages"]
            assert current_messages[: len(prior_messages)] == prior_messages
            assert len(current_messages) == len(prior_messages) + 2


def test_successor_prompts_have_no_model_specific_routing_anchors() -> None:
    encoded = json.dumps(
        [
            {
                "messages": state["messages"],
                "tools": state["tools"],
            }
            for sequence in workload.history_sequences()
            for state in sequence["states"]
        ]
    ).lower()
    forbidden = [
        *[worker.lower() for worker in WORKERS],
        *[model.lower() for model in WORKERS.values()],
        "deepseek-v4",
        "mimo-v2.5",
        "tencent/hy3",
    ]
    assert not any(value in encoded for value in forbidden)


def test_normalized_turns_preserve_the_agentic_tool_history() -> None:
    case = workload.history_sequences()[0]["states"][0]
    turns = workload.normalized_turns(case)
    assert [turn["role"] for turn in turns] == [
        "user",
        "assistant",
        "user",
        "user",
    ]
    assert "[tool_call search_workspace]" in turns[1]["text"]
    assert turns[2]["text"].startswith("[tool_result search_workspace]\n")
    assert not any(turn["role"] == "system" for turn in turns)


def test_axis_classifier_covers_all_workers_with_margin() -> None:
    observations = {
        "worker-a": workload.classify_axis(0.005),
        "worker-b": workload.classify_axis(0.0),
        "worker-c": workload.classify_axis(-0.005),
    }
    assert {selected for selected, _gap in observations.values()} == set(WORKERS)
    assert all(
        gap > workload.MINIMUM_TOP_TWO_SCORE_GAP
        for _selected, gap in observations.values()
    )


def test_successor_contract_is_source_closed_and_exactly_bounded() -> None:
    contract = successor_contract.validate()
    assert contract["source_closed"] is True
    assert contract["launch_authorized"] is False
    assert contract["requires_new_budget_authority"] is True
    assert contract["logical_provider_requests"] == {
        "provider_preflight": 6,
        "semantic_cache_measurement": 72,
        "maximum_total": 78,
    }
    assert contract["maximum_external_attempts"] == EXPECTED_MAXIMUM_EXTERNAL_ATTEMPTS
    assert contract["report_schema_version"] == (
        "rayline.openrouter-kv-cache-comparison.v3"
    )
    assert "remote_encoder_trace_matches_offline_trace" in contract["acceptance_gates"]
