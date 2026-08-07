#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Generate the matched 24-token AGT017 ARC artifact."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from openrouter_agentic_artifact_fixture import WORKERS
from openrouter_artifact_fixture import generate_contract
from openrouter_kv_cache_matched_contract import (
    ARTIFACT_REVISION,
    MAX_COMPLETION_TOKENS,
)

EXPECTED_ARGUMENT_COUNT = 2
# v3 was retired unpublished: its worker-a provider order collapsed to a
# single saturated provider before any deployment used it. v4 re-vets
# worker-a against the 2026-08-05 OpenRouter landscape with conservative
# maximum-rate pricing across the amended order; workers b and c are
# unchanged.
SUCCESSOR_ARTIFACT_REVISION = "public-rayline-arc-openrouter-kv-cache-v4"
# v4 carries the AGT018 worker-a re-vetting and is retained verbatim as the
# historical (regenerable) AGT018 artifact. v5 appended DeepInfra to worker-b
# and was retired unpublished, exactly as v3 was: streaming diagnosis showed
# DeepInfra's MiMo endpoint emits only empty-content deltas and closes with
# finish_reason=length under the frozen payload, so it can never satisfy the
# benchmark's content-token requirement at the 24-token cap. v6 restores
# worker-b's frozen four-provider order and its v4 Novita-maxima pricing;
# worker-a and worker-c are unchanged from v4. No deployment consumed v5.
AGT019_ARTIFACT_REVISION = "public-rayline-arc-openrouter-kv-cache-v6"
# v7 replaces worker-b's lane outright (AGT019 luna amendment, 2026-08-07).
# MiMo's whole four-provider pool stayed unhealthy for more than 21 hours —
# three providers dropped out of OpenRouter's routing pool entirely and Venice
# returned upstream 429s — so the benchmark's two-of-four depth gate was never
# satisfiable and no provider remained to widen to. worker-b is now
# `openai/gpt-5.6-luna` pinned to the single OpenAI provider, priced at the
# per-field maxima across OpenAI's three endpoint tags (openai, openai/flex,
# openai/priority) so the rate can only be over-stated, never under-stated.
# Unlike the other workers OpenAI prices cache reads and writes separately, so
# the three input rates are no longer flat. worker-b declares no temperature:
# gpt-5.6-luna does not advertise the parameter and `require_parameters` would
# otherwise filter out every OpenAI endpoint. v6 is untouched and remains the
# historical, regenerable revision consumed by the banked native arm.
AGT019_LUNA_ARTIFACT_REVISION = "public-rayline-arc-openrouter-kv-cache-v7"
ALLOWED_ARTIFACT_REVISIONS = frozenset(
    {
        ARTIFACT_REVISION,
        SUCCESSOR_ARTIFACT_REVISION,
        AGT019_ARTIFACT_REVISION,
        AGT019_LUNA_ARTIFACT_REVISION,
    }
)
SUCCESSOR_WORKERS = tuple(
    (
        {
            **worker,
            "provider_order": ["baidu", "gmicloud", "siliconflow"],
            "pricing_source": "openrouter-bounded-provider-order-2026-08-05",
            "prompt_cost": 0.00000013,
            "cache_read_cost": 0.00000013,
            "cache_write_cost": 0.00000013,
            "completion_cost": 0.00000028,
        }
        if worker["id"] == "worker-a"
        else worker
    )
    for worker in WORKERS
)
AGT019_WORKERS = tuple(
    (
        {
            **worker,
            "provider_order": ["xiaomi", "parasail", "venice", "novita"],
            "pricing_source": "openrouter-bounded-provider-order-2026-08-05c",
            "prompt_cost": 0.000000168,
            "cache_read_cost": 0.000000168,
            "cache_write_cost": 0.000000168,
            "completion_cost": 0.000000336,
        }
        if worker["id"] == "worker-b"
        else worker
    )
    for worker in SUCCESSOR_WORKERS
)
AGT019_LUNA_WORKERS = tuple(
    (
        {
            **worker,
            "model": "openai/gpt-5.6-luna",
            "provider_slug": "openai",
            "provider_name": "OpenAI",
            "provider_order": ["openai"],
            "pricing_source": "openrouter-openai-endpoint-maxima-2026-08-07",
            "prompt_cost": 0.0000002,
            "cache_read_cost": 0.00000002,
            "cache_write_cost": 0.00000025,
            "completion_cost": 0.0000012,
            # gpt-5.6-luna does not advertise `temperature`; a None temperature
            # makes `_worker_contract` omit the key entirely, which both arms'
            # routers read as "do not send one".
            "temperature": None,
        }
        if worker["id"] == "worker-b"
        else worker
    )
    for worker in AGT019_WORKERS
)


def generate(output_dir: Path, artifact_revision: str = ARTIFACT_REVISION) -> None:
    if artifact_revision not in ALLOWED_ARTIFACT_REVISIONS:
        raise RuntimeError("KV cache artifact revision is not source-registered")
    if artifact_revision == AGT019_LUNA_ARTIFACT_REVISION:
        workers = AGT019_LUNA_WORKERS
        pricing_snapshot = "openrouter-bounded-provider-orders-2026-08-07"
    elif artifact_revision == AGT019_ARTIFACT_REVISION:
        workers = AGT019_WORKERS
        pricing_snapshot = "openrouter-bounded-provider-orders-2026-08-05c"
    elif artifact_revision == SUCCESSOR_ARTIFACT_REVISION:
        workers = SUCCESSOR_WORKERS
        pricing_snapshot = "openrouter-bounded-provider-orders-2026-08-05"
    else:
        workers = WORKERS
        pricing_snapshot = "openrouter-bounded-provider-orders-2026-08-03"
    generate_contract(
        output_dir,
        artifact_id=artifact_revision,
        workers=workers,
        capability_tag="public-openrouter-kv-cache-benchmark",
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        created_at="2026-08-04T00:00:00Z",
        exporter_commit="public-openrouter-kv-cache-exporter-v2",
        pricing_snapshot=pricing_snapshot,
    )


if __name__ == "__main__":
    if len(sys.argv) != EXPECTED_ARGUMENT_COUNT:
        raise SystemExit("usage: openrouter_kv_cache_artifact_fixture.py OUTPUT_DIR")
    generate(
        Path(sys.argv[1]),
        os.environ.get("RAYLINE_ARC_ARTIFACT_REVISION", ARTIFACT_REVISION),
    )
