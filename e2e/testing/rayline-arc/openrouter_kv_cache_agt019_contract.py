#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Bound identity and request envelope for AGT019, source-closed.

AGT019 succeeds AGT018d, whose only failing gate was whole-set completion
matching: pinned multi-provider fallthrough legally served worker-b from
different providers on the two arms. AGT019 replaces that gate with the
matched-pair comparability policy in `openrouter_kv_cache_matched_pair`, and
keeps every other acceptance gate unconditional.
"""

from __future__ import annotations

from typing import Any

from openrouter_kv_cache_successor_workload import (
    EXPECTED_REQUESTS_PER_DEPLOYMENT,
)
from openrouter_kv_cache_successor_workload import (
    SCHEMA_VERSION as WORKLOAD_SCHEMA_VERSION,
)

SCHEMA_VERSION = "rayline.openrouter-kv-cache-agt019-contract.v1"
REPORT_SCHEMA_VERSION = "rayline.openrouter-kv-cache-comparison.v4"
RUN_ID = "rayline-openrouter-kv-cache-agt019-20260805"
SEMANTIC_BRANCH = "codex/rayline-remote-mvp"
SEMANTIC_REMOTE_REF = f"atlasfutures/{SEMANTIC_BRANCH}"
PATHFINDER_BRANCH = "codex/rayline-vsr-mvp"

# AGT019 is staged source-closed: the policy engine, contract, and tests land
# before any launch authority exists. Both pins stay empty until a checkpoint
# binds a fresh preregistration and authorization commit.
PREREGISTRATION_COMMIT = ""
AUTHORIZATION_COMMIT = ""
GIT_SHA1_HEX_LENGTH = 40
NATIVE_APP_NAME = "rayline-router-openrouter-agt019"
NATIVE_WEBHOOK_LABEL = "router-openrouter-agt019"
REMOTE_APP_NAME = "rayline-arc-session-encoder-flashinfer-agt019"
REMOTE_CLASS_NAME = "SessionEncoder"
VLLM_COMMIT = "9f5ea81ca0aa570aea46baf82311a1139c1267ca"
REMOTE_ENGINE_BUILD_ID = f"vllm@{VLLM_COMMIT}+gdn-flashinfer-eager"
REMOTE_GDN_PREFILL_BACKEND = "flashinfer"
# The v4 artifact content is unchanged and reused verbatim from AGT018.
ARTIFACT_REVISION = "public-rayline-arc-openrouter-kv-cache-v4"
MAX_COMPLETION_TOKENS = 24

# No BudgetContract is declared here on purpose. AGT018's conservative
# accounting closed at $142.418831066383 against the $144.31282402 authority,
# leaving $1.893992953617 — only about $0.69 of packet headroom above the
# $1.20 required final reserve, which cannot fund two H100 arms plus two
# provider keys. A reviewed budget contract plus fresh user authority must be
# added in the same checkpoint that opens the pins below.
REQUIRED_FINAL_RESERVE_USD = 1.20
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
    "matched_pair_comparability_policy",
    "request_attempt_and_cost_envelopes",
    "privacy_and_cleanup",
)


def validate() -> dict[str, Any]:
    if PREREGISTRATION_COMMIT or AUTHORIZATION_COMMIT:
        raise RuntimeError("AGT019 launch authority is not yet bound")
    if EXPECTED_REQUESTS_PER_DEPLOYMENT != EXPECTED_SEMANTIC_REQUESTS_PER_DEPLOYMENT:
        raise RuntimeError("AGT019 semantic workload request envelope diverged")
    if (
        MAXIMUM_LOGICAL_PROVIDER_REQUESTS != EXPECTED_MAXIMUM_LOGICAL_PROVIDER_REQUESTS
        or MAXIMUM_EXTERNAL_ATTEMPTS != EXPECTED_MAXIMUM_EXTERNAL_ATTEMPTS
    ):
        raise RuntimeError("AGT019 provider request or attempt envelope diverged")
    if (
        SOURCE_CLOSED_KEY_LIMIT_USD_PER_ARM != 0
        or SOURCE_CLOSED_MAXIMUM_PAID_WALL_SECONDS != 0
    ):
        raise RuntimeError("AGT019 source-closed resource envelope diverged")
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
