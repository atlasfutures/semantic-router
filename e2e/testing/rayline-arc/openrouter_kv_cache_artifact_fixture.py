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
ALLOWED_ARTIFACT_REVISIONS = frozenset({ARTIFACT_REVISION, SUCCESSOR_ARTIFACT_REVISION})
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


def generate(output_dir: Path, artifact_revision: str = ARTIFACT_REVISION) -> None:
    if artifact_revision not in ALLOWED_ARTIFACT_REVISIONS:
        raise RuntimeError("KV cache artifact revision is not source-registered")
    successor = artifact_revision == SUCCESSOR_ARTIFACT_REVISION
    generate_contract(
        output_dir,
        artifact_id=artifact_revision,
        workers=SUCCESSOR_WORKERS if successor else WORKERS,
        capability_tag="public-openrouter-kv-cache-benchmark",
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        created_at="2026-08-04T00:00:00Z",
        exporter_commit="public-openrouter-kv-cache-exporter-v2",
        pricing_snapshot=(
            "openrouter-bounded-provider-orders-2026-08-05"
            if successor
            else "openrouter-bounded-provider-orders-2026-08-03"
        ),
    )


if __name__ == "__main__":
    if len(sys.argv) != EXPECTED_ARGUMENT_COUNT:
        raise SystemExit("usage: openrouter_kv_cache_artifact_fixture.py OUTPUT_DIR")
    generate(
        Path(sys.argv[1]),
        os.environ.get("RAYLINE_ARC_ARTIFACT_REVISION", ARTIFACT_REVISION),
    )
