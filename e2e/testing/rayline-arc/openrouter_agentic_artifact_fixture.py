#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Generate the bounded three-arm ARC artifact for the agentic benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

from openrouter_artifact_fixture import generate_contract

ARTIFACT_ID = "public-rayline-arc-openrouter-agentic-v1"
EXPECTED_ARGUMENT_COUNT = 2
MAX_COMPLETION_TOKENS = 96
WORKERS = (
    {
        "id": "worker-a",
        "model": "deepseek/deepseek-v4-flash",
        "provider_slug": "baidu",
        "provider_name": "Baidu",
        "pricing_source": "openrouter-baidu-2026-08-03",
        "prompt_cost": 0.0000000882,
        "cache_read_cost": 0.00000001764,
        "cache_write_cost": 0.0000000882,
        "completion_cost": 0.0000001764,
        "temperature": 0,
    },
    {
        "id": "worker-b",
        "model": "xiaomi/mimo-v2.5",
        "provider_slug": "xiaomi",
        "provider_name": "Xiaomi",
        "pricing_source": "openrouter-xiaomi-2026-08-03",
        "prompt_cost": 0.00000014,
        "cache_read_cost": 0.0000000028,
        "cache_write_cost": 0.00000014,
        "completion_cost": 0.00000028,
        "temperature": 0,
    },
    {
        "id": "worker-c",
        "model": "tencent/hy3",
        "provider_slug": "tencent",
        "provider_name": "Tencent",
        "pricing_source": "openrouter-tencent-2026-08-03",
        "prompt_cost": 0.000000132,
        "cache_read_cost": 0.000000033,
        "cache_write_cost": 0.000000132,
        "completion_cost": 0.000000528,
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
        created_at="2026-08-03T00:00:00Z",
        exporter_commit="public-openrouter-agentic-exporter-v1",
        pricing_snapshot="openrouter-pinned-endpoints-2026-08-03",
    )


if __name__ == "__main__":
    if len(sys.argv) != EXPECTED_ARGUMENT_COUNT:
        raise SystemExit("usage: openrouter_agentic_artifact_fixture.py OUTPUT_DIR")
    generate(Path(sys.argv[1]))
