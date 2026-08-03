#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Frozen model pool and public agentic inputs for the OpenRouter benchmark."""

from __future__ import annotations

from typing import Any

from modal_fullstack_inputs import CANDIDATE_PROMPTS

WORKERS = {
    "worker-a": "deepseek/deepseek-v4-flash",
    "worker-b": "xiaomi/mimo-v2.5",
    "worker-c": "tencent/hy3",
}
PROVIDER_SLUGS = {
    "worker-a": ("baidu", "streamlake", "deepinfra"),
    "worker-b": ("xiaomi", "parasail", "venice", "novita"),
    "worker-c": ("tencent", "deepinfra", "novita"),
}
PROVIDER_NAMES = {
    "worker-a": ("Baidu", "StreamLake", "DeepInfra"),
    "worker-b": ("Xiaomi", "Parasail", "Venice", "Novita"),
    "worker-c": ("Tencent", "DeepInfra", "Novita"),
}

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
