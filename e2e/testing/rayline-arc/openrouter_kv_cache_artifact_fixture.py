#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Generate the matched 24-token AGT017 ARC artifact."""

from __future__ import annotations

import sys
from pathlib import Path

from openrouter_agentic_artifact_fixture import WORKERS
from openrouter_artifact_fixture import generate_contract
from openrouter_kv_cache_matched_contract import (
    ARTIFACT_REVISION,
    MAX_COMPLETION_TOKENS,
)

EXPECTED_ARGUMENT_COUNT = 2


def generate(output_dir: Path) -> None:
    generate_contract(
        output_dir,
        artifact_id=ARTIFACT_REVISION,
        workers=WORKERS,
        capability_tag="public-openrouter-kv-cache-benchmark",
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        created_at="2026-08-04T00:00:00Z",
        exporter_commit="public-openrouter-kv-cache-exporter-v2",
        pricing_snapshot="openrouter-bounded-provider-orders-2026-08-03",
    )


if __name__ == "__main__":
    if len(sys.argv) != EXPECTED_ARGUMENT_COUNT:
        raise SystemExit("usage: openrouter_kv_cache_artifact_fixture.py OUTPUT_DIR")
    generate(Path(sys.argv[1]))
