#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Frozen model pool and public agentic inputs for the OpenRouter benchmark."""

from __future__ import annotations

from typing import Any

from modal_fullstack_inputs import CANDIDATE_PROMPTS

WORKERS = {
    "worker-a": "deepseek/deepseek-v4-flash",
    "worker-b": "openai/gpt-5.6-luna",
    "worker-c": "tencent/hy3",
}
# Re-vetted 2026-08-05 for AGT018d: StreamLake and DeepInfra no longer pass
# the frozen require_parameters filter for DS4 Flash (OpenRouter returns 404
# "No endpoints found"), leaving Baidu as the only effective provider, and
# Baidu's shared upstream pool intermittently returns tpm_rate_limit_exceeded.
# GMICloud and SiliconFlow (both fp8, like the Baidu endpoint) passed the
# byte-exact frozen payload, and OpenRouter falls through a rate-limited
# provider to the next entry within the pinned order even with fallbacks
# disabled. All launchable legacy packets are pin-closed, so this table only
# governs AGT018 runtime and validation.
#
# Re-vetted again 2026-08-05 for AGT019: worker-b's frozen order collapsed to
# Venice alone under the same require_parameters filter (Xiaomi, Parasail,
# GMICloud and Novita were all filtered out at re-vet time), and Venice's
# shared pool flaps sub-minute — three availability preflights aborted at $0.
# The original four flap in and out of eligibility rather than disappearing
# (Xiaomi and Venice each served full runs the same morning), so none are
# removed and the order is left at its frozen four.
#
# DeepInfra was briefly appended as a fifth last-resort entry and is now
# retracted (2026-08-05c). Streaming diagnosis showed its MiMo endpoint emits
# only empty-content deltas and closes with finish_reason=length under the
# frozen payload — the model spends the whole completion budget on a hidden
# reasoning phase there — so it can never satisfy the benchmark's
# content-token requirement at the 24-token cap. The earlier probe accepted it
# on a non-streaming HTTP 200, which is too shallow: provider vetting now
# requires a streaming probe that observes at least one content token, not an
# endpoint status code.
#
# Worker-b was replaced wholesale on 2026-08-07 (AGT019 luna amendment). MiMo's
# entire four-provider pool stayed unusable for more than 21 hours: Xiaomi,
# Parasail and Novita were dropped from OpenRouter's routing pool altogether
# (single-provider pins return 404 "No endpoints found" rather than 429) while
# Venice returned upstream 429s, so the two-of-four depth gate this benchmark
# requires was never satisfiable. MiMo's registered endpoints never delisted —
# they simply never became healthy again — and there was nothing left to widen
# to, DeepInfra having already been retracted above. Worker-b is therefore now
# `openai/gpt-5.6-luna` pinned to the single OpenAI provider.
#
# gpt-5.6-luna does not advertise `temperature`, so the frozen payload must not
# send it: under `require_parameters` OpenRouter filters out every OpenAI
# endpoint and returns the same 404 "No endpoints found that can handle the
# requested parameters" (verified live 2026-08-07 against all three OpenAI
# tags). Suppression is per worker and data-driven, never global — workers a
# and c keep `temperature: 0`:
#   * on the measured ARC path both routers drive temperature from the worker
#     manifest, so the artifact simply declares worker-b's temperature as None
#     (`_worker_contract` omits the key; the Go dispatch deletes the client's
#     value when the manifest carries none, and the native router's
#     `apply_worker_temperature` treats None as a no-op);
#   * on the direct path the benchmark payload drops the parameters listed in
#     UNSUPPORTED_REQUEST_PARAMETERS below.
# `require_parameters` stays True for every worker on every path — nothing is
# weakened to accommodate the new lane.
PROVIDER_SLUGS = {
    "worker-a": ("baidu", "gmicloud", "siliconflow"),
    "worker-b": ("openai",),
    "worker-c": ("tencent", "deepinfra", "novita"),
}
PROVIDER_NAMES = {
    "worker-a": ("Baidu", "GMICloud", "SiliconFlow"),
    "worker-b": ("OpenAI",),
    "worker-c": ("Tencent", "DeepInfra", "Novita"),
}
# Request parameters the pinned model does not advertise, dropped per worker
# from the direct-path payload so the frozen `require_parameters` filter can
# still be enforced. Empty/absent means "send the payload unchanged".
UNSUPPORTED_REQUEST_PARAMETERS = {
    "worker-b": ("temperature",),
}


def pinned_provider_payload(
    payload: dict[str, Any], expected_worker: str
) -> dict[str, Any]:
    """Pin a non-ARC payload to its worker's frozen provider order.

    `require_parameters` makes OpenRouter reject the whole request when the
    pinned provider does not advertise a supplied parameter, so a worker whose
    model lacks one must not send it at all. Dropping the parameter keeps the
    filter strict rather than relaxing it to accommodate the worker. The ARC
    path carries no provider block: both arms' routers pin those requests from
    the worker manifest instead, and drive temperature from it too.
    """

    payload["provider"] = {
        "order": list(PROVIDER_SLUGS[expected_worker]),
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    payload["reasoning"] = {"enabled": False, "effort": "none"}
    for parameter in UNSUPPORTED_REQUEST_PARAMETERS.get(expected_worker, ()):
        payload.pop(parameter, None)
    return payload


MODAL_REFERENCE = {
    "description": (
        "Qwen3.5-0.8B generation on two Modal L4 workers with the same "
        "Modal H100 retained encoder"
    ),
    "concurrency_1": {
        "arc_to_static_throughput_ratio": 0.748,
        "arc_minus_static_p95_seconds": 0.351,
    },
    "concurrency_4": {
        "arc_to_static_throughput_ratio": 0.755,
        "arc_minus_static_p95_seconds": 0.596,
    },
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 repository file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_workspace",
            "description": "Search repository text and return bounded matches.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
]


def _code_result() -> str:
    return "\n".join(
        f"src/router/session_{index % 7}.py:{20 + index}: "
        f"await state.commit(turn={index}, fenced=True)"
        for index in range(80)
    )


def _research_result() -> str:
    return "\n".join(
        f"document-{index:03d}: queueing observation {index % 11}; "
        f"sample={1000 + index}; source=public-synthetic"
        for index in range(220)
    )


def _incident_result() -> str:
    return "\n".join(
        f"2026-08-03T12:{index // 60:02d}:{index % 60:02d}Z "
        f"worker={index % 3} queue={index % 9} retained={index * 17} "
        f"event={'retry' if index % 37 == 0 else 'complete'}"
        for index in range(420)
    )


SCENARIOS = {
    "code_patch": {
        "system": "You are a coding agent. Produce a concise implementation plan.",
        "user": "Trace a transactional state bug before proposing the smallest patch.",
        "tool": "search_workspace",
        "arguments": '{"query":"state.commit"}',
        "result": _code_result(),
        "final": "Summarize the likely race and give a four-step patch plan.",
    },
    "research_synthesis": {
        "system": "You are a research agent. Synthesize evidence without quoting it.",
        "user": "Compare queueing behavior across the supplied observations.",
        "tool": "read_file",
        "arguments": '{"path":"public-observations.txt"}',
        "result": _research_result(),
        "final": "State the dominant pattern, two caveats, and the next measurement.",
    },
    "incident_triage": {
        "system": "You are an operations agent. Triage the bounded service trace.",
        "user": "Identify whether retries or queueing dominate this incident.",
        "tool": "read_file",
        "arguments": '{"path":"public-service.log"}',
        "result": _incident_result(),
        "final": "Give the diagnosis, confidence, and three immediate checks.",
    },
}


def candidate_case(index: int) -> dict[str, Any]:
    scenario_name = tuple(SCENARIOS)[index % len(SCENARIOS)]
    scenario = SCENARIOS[scenario_name]
    anchor = CANDIDATE_PROMPTS[index % len(CANDIDATE_PROMPTS)]
    call_id = f"public_call_{index:02d}"
    return {
        "case_id": f"agentic-{index:02d}",
        "scenario": scenario_name,
        "messages": [
            {"role": "system", "content": scenario["system"]},
            {"role": "user", "content": f"{scenario['user']} Anchor: {anchor}"},
            {
                "role": "assistant",
                "content": "I will inspect the bounded public evidence.",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": scenario["tool"],
                            "arguments": scenario["arguments"],
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": scenario["result"],
            },
            {
                "role": "user",
                "content": f"{scenario['final']} Routing anchor: {anchor}",
            },
        ],
        "tools": TOOLS,
    }
