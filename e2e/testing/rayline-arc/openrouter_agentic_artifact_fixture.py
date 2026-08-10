#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Generate the bounded three-arm ARC artifact for the agentic packet.

The worker pool here is the same frozen pool `openrouter_agentic_workload`
governs and `config-openrouter-agentic.yaml` prices; the router refuses to
serve when the two disagree, so the amendment dates recorded in the workload
module apply verbatim to this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

from openrouter_artifact_fixture import generate_contract

ARTIFACT_ID = "public-rayline-arc-openrouter-agentic-v2"
EXPECTED_ARGUMENT_COUNT = 2
# The frozen benchmark pins every reply to 96 tokens on purpose. This packet is
# also the one an interactive agent drives, and 96 tokens is below one tool call.
# 4096 is the ceiling the surrounding contract already implies: workers carry
# `attempt_deadline_seconds: 120`, and 4096 tokens at the 40-80 tok/s these
# pinned endpoints sustain already consumes 50-100s of that budget.
MAX_COMPLETION_TOKENS = 4096
# Only a floor, never a target: it exists so `max_tokens` is always present on
# the wire (an absent cap is uncapped provider spend), not to pad short replies.
MINIMUM_COMPLETION_TOKENS = 16
WORKERS = (
    {
        "id": "worker-a",
        "model": "deepseek/deepseek-v4-flash",
        "provider_slug": "baidu",
        "provider_name": "Baidu",
        # 2026-08-05 AGT018d re-vet: StreamLake and DeepInfra stopped passing
        # the frozen require_parameters filter for DS4 Flash.
        "provider_order": ["baidu", "gmicloud", "siliconflow"],
        "allow_fallbacks": False,
        "pricing_source": "openrouter-bounded-provider-order-2026-08-03",
        "prompt_cost": 0.00000009,
        "cache_read_cost": 0.00000009,
        "cache_write_cost": 0.00000009,
        "completion_cost": 0.00000018,
        "temperature": 0,
    },
    {
        # 2026-08-07 AGT019 luna amendment: MiMo's whole provider pool went
        # unserveable. Priced at the per-field maxima across OpenAI's three
        # gpt-5.6-luna endpoint tags, matching config-openrouter-agentic.yaml.
        "id": "worker-b",
        "model": "openai/gpt-5.6-luna",
        "provider_slug": "openai",
        "provider_name": "OpenAI",
        "provider_order": ["openai"],
        "allow_fallbacks": False,
        "pricing_source": "openrouter-openai-endpoint-maxima-2026-08-07",
        "prompt_cost": 0.0000002,
        "cache_read_cost": 0.00000002,
        "cache_write_cost": 0.00000025,
        "completion_cost": 0.0000012,
        # gpt-5.6-luna does not advertise `temperature`; a None temperature
        # makes `_worker_contract` omit the key, which the Go dispatch reads as
        # "delete the client's value" rather than sending an unadvertised one.
        "temperature": None,
    },
    {
        "id": "worker-c",
        "model": "tencent/hy3",
        "provider_slug": "tencent",
        "provider_name": "Tencent",
        "provider_order": ["tencent", "deepinfra", "novita"],
        "allow_fallbacks": False,
        "pricing_source": "openrouter-bounded-provider-order-2026-08-03",
        "prompt_cost": 0.00000014,
        "cache_read_cost": 0.00000014,
        "cache_write_cost": 0.00000014,
        "completion_cost": 0.00000058,
        "temperature": 0,
    },
)


def generate(output_dir: Path) -> None:
    generate_contract(
        output_dir,
        artifact_id=ARTIFACT_ID,
        workers=WORKERS,
        capability_tag="public-openrouter-agentic-benchmark",
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        minimum_completion_tokens=MINIMUM_COMPLETION_TOKENS,
        created_at="2026-08-07T00:00:00Z",
        exporter_commit="public-openrouter-agentic-exporter-v2",
        pricing_snapshot="openrouter-bounded-provider-orders-2026-08-07",
    )


if __name__ == "__main__":
    if len(sys.argv) != EXPECTED_ARGUMENT_COUNT:
        raise SystemExit("usage: openrouter_agentic_artifact_fixture.py OUTPUT_DIR")
    generate(Path(sys.argv[1]))
