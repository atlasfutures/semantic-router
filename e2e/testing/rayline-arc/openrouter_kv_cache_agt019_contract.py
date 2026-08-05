#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Identity, budget authority, and request envelope for AGT019, source-closed.

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
from rayline_three_arm_budget import BudgetContract, budget_receipt

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

# AGT018's conservative accounting closed at $142.418831066383 against the
# $144.31282402 authority, leaving $1.893992953617 — only about $0.69 of packet
# headroom above the $1.20 required final reserve, which cannot fund two H100
# arms plus two provider keys. The user approved fresh $10 authority on
# 2026-08-05, raising cumulative authority to $154.31282402. The $0.15 per-arm
# key limit is the AGT018-proven value: $0.05 hit OpenRouter's HTTP 402 at
# request 35 of 36 because its limit check counts in-flight pre-authorization
# holds, while $0.15 gives the 36-request workload roughly 3x settled-cost
# headroom.
AUTHORIZED_KEY_LIMIT_USD_PER_ARM = 0.15
AUTHORIZED_MAXIMUM_PAID_WALL_SECONDS = 20 * 60
MAXIMUM_PROVIDER_SPEND_USD = 2 * AUTHORIZED_KEY_LIMIT_USD_PER_ARM
REQUIRED_FINAL_RESERVE_USD = 1.20

# Fail-closed packet placeholders. The authorized values above replace both
# zeros in the same checkpoint that opens the two source-authority pins.
SOURCE_CLOSED_KEY_LIMIT_USD_PER_ARM = 0.0
SOURCE_CLOSED_MAXIMUM_PAID_WALL_SECONDS = 0

AGT019_RESOURCE_BUDGET = BudgetContract(
    run_id=RUN_ID,
    previous_conservative_usd=142.418831066383,
    authorized_cumulative_usd=154.31282402,
    packet_ceiling_usd=9.1,
    # The provider limits are accounted separately below. Reserving $1.50 at
    # this layer guarantees at least $1.20 after both $0.15 keys are exhausted.
    required_reserve_usd=REQUIRED_FINAL_RESERVE_USD + MAXIMUM_PROVIDER_SPEND_USD,
    maximum_paid_wall_seconds=AUTHORIZED_MAXIMUM_PAID_WALL_SECONDS,
    encoder_replicas=2,
)


def agt019_budget_receipt() -> dict[str, Any]:
    """Return the complete two-H100 plus two-provider-key envelope."""

    receipt = budget_receipt(AGT019_RESOURCE_BUDGET)
    cumulative = receipt["cumulative_if_full_envelope_usd"] + MAXIMUM_PROVIDER_SPEND_USD
    reserve = AGT019_RESOURCE_BUDGET.authorized_cumulative_usd - cumulative
    if reserve < REQUIRED_FINAL_RESERVE_USD:
        raise RuntimeError("AGT019 complete envelope exceeds user authority")
    return {
        **receipt,
        "maximum_provider_spend_usd": MAXIMUM_PROVIDER_SPEND_USD,
        "maximum_complete_packet_usd": (
            receipt["maximum_resource_envelope_usd"] + MAXIMUM_PROVIDER_SPEND_USD
        ),
        "cumulative_if_complete_envelope_usd": cumulative,
        "reserve_after_complete_envelope_usd": reserve,
    }


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
    receipt = agt019_budget_receipt()
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
        "budget": {
            "authorized_cumulative_usd": (
                AGT019_RESOURCE_BUDGET.authorized_cumulative_usd
            ),
            "maximum_complete_packet_usd": receipt["maximum_complete_packet_usd"],
            "reserve_after_complete_envelope_usd": (
                receipt["reserve_after_complete_envelope_usd"]
            ),
        },
    }
