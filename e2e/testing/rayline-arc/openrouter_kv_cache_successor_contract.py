#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Source-closed identity and request envelope for the AGT018 successor."""

from __future__ import annotations

from typing import Any

from openrouter_kv_cache_successor_workload import (
    EXPECTED_REQUESTS_PER_DEPLOYMENT,
)
from openrouter_kv_cache_successor_workload import (
    SCHEMA_VERSION as WORKLOAD_SCHEMA_VERSION,
)

SCHEMA_VERSION = "rayline.openrouter-kv-cache-successor-contract.v1"
REPORT_SCHEMA_VERSION = "rayline.openrouter-kv-cache-comparison.v3"
RUN_ID = "rayline-openrouter-kv-cache-agt018-20260804"
SEMANTIC_BRANCH = "codex/rayline-remote-mvp"
SEMANTIC_REMOTE_REF = f"atlasfutures/{SEMANTIC_BRANCH}"
PATHFINDER_BRANCH = "codex/rayline-vsr-mvp"

PREREGISTRATION_COMMIT = ""
AUTHORIZATION_COMMIT = ""
NATIVE_APP_NAME = "rayline-router-openrouter-agt018"
NATIVE_WEBHOOK_LABEL = "router-openrouter-agt018"
REMOTE_APP_NAME = "rayline-arc-session-encoder-flashinfer-agt018"
REMOTE_CLASS_NAME = "SessionEncoder"
ARTIFACT_REVISION = "public-rayline-arc-openrouter-kv-cache-v3"
MAX_COMPLETION_TOKENS = 24
# Fail-closed packet placeholders. A reviewed budget contract must replace both
# values in the same checkpoint that opens the two source-authority pins.
SOURCE_CLOSED_KEY_LIMIT_USD_PER_ARM = 0.0
SOURCE_CLOSED_MAXIMUM_PAID_WALL_SECONDS = 0

PROVIDER_PREFLIGHT_REQUESTS_PER_DEPLOYMENT = 3
DEPLOYMENTS = 2
MAXIMUM_RETRIES_PER_REQUEST = 1
MAXIMUM_LOGICAL_PROVIDER_REQUESTS = DEPLOYMENTS * (
    PROVIDER_PREFLIGHT_REQUESTS_PER_DEPLOYMENT + EXPECTED_REQUESTS_PER_DEPLOYMENT
)
MAXIMUM_EXTERNAL_ATTEMPTS = MAXIMUM_LOGICAL_PROVIDER_REQUESTS * (
    MAXIMUM_RETRIES_PER_REQUEST + 1
)
EXPECTED_SEMANTIC_REQUESTS_PER_DEPLOYMENT = 36
EXPECTED_MAXIMUM_LOGICAL_PROVIDER_REQUESTS = 78
EXPECTED_MAXIMUM_EXTERNAL_ATTEMPTS = 156
ACCEPTANCE_GATES = (
    "exact_source_and_artifact_identity",
    "provider_preflight_all_three_models",
    "native_offline_three_worker_coverage",
    "remote_encoder_trace_matches_offline_trace",
    "native_remote_selected_worker_trace_parity",
    "native_retained_token_saving",
    "remote_retained_token_saving",
    "matched_completion_policy",
    "request_attempt_and_cost_envelopes",
    "privacy_and_cleanup",
)


def validate() -> dict[str, Any]:
    if PREREGISTRATION_COMMIT or AUTHORIZATION_COMMIT:
        raise RuntimeError(
            "AGT018 source preparation unexpectedly has launch authority"
        )
    if EXPECTED_REQUESTS_PER_DEPLOYMENT != EXPECTED_SEMANTIC_REQUESTS_PER_DEPLOYMENT:
        raise RuntimeError("AGT018 semantic workload request envelope diverged")
    if (
        MAXIMUM_LOGICAL_PROVIDER_REQUESTS != EXPECTED_MAXIMUM_LOGICAL_PROVIDER_REQUESTS
        or MAXIMUM_EXTERNAL_ATTEMPTS != EXPECTED_MAXIMUM_EXTERNAL_ATTEMPTS
    ):
        raise RuntimeError("AGT018 provider request or attempt envelope diverged")
    if (
        SOURCE_CLOSED_KEY_LIMIT_USD_PER_ARM != 0
        or SOURCE_CLOSED_MAXIMUM_PAID_WALL_SECONDS != 0
    ):
        raise RuntimeError("AGT018 source-closed resource envelope diverged")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "source_closed": True,
        "launch_authorized": False,
        "requires_new_budget_authority": True,
        "workload_schema_version": WORKLOAD_SCHEMA_VERSION,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "artifact_revision": ARTIFACT_REVISION,
        "logical_provider_requests": {
            "provider_preflight": (
                DEPLOYMENTS * PROVIDER_PREFLIGHT_REQUESTS_PER_DEPLOYMENT
            ),
            "semantic_cache_measurement": (
                DEPLOYMENTS * EXPECTED_REQUESTS_PER_DEPLOYMENT
            ),
            "maximum_total": MAXIMUM_LOGICAL_PROVIDER_REQUESTS,
        },
        "maximum_external_attempts": MAXIMUM_EXTERNAL_ATTEMPTS,
        "maximum_completion_tokens": MAX_COMPLETION_TOKENS,
        "acceptance_gates": list(ACCEPTANCE_GATES),
        "release_qualification_1000_executed": False,
    }
