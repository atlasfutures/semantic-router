#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Bound identity, budget authority, and request envelope for AGT018."""

from __future__ import annotations

from typing import Any

from openrouter_kv_cache_successor_workload import (
    EXPECTED_REQUESTS_PER_DEPLOYMENT,
)
from openrouter_kv_cache_successor_workload import (
    SCHEMA_VERSION as WORKLOAD_SCHEMA_VERSION,
)
from rayline_three_arm_budget import BudgetContract, budget_receipt

SCHEMA_VERSION = "rayline.openrouter-kv-cache-successor-contract.v1"
REPORT_SCHEMA_VERSION = "rayline.openrouter-kv-cache-comparison.v3"
RUN_ID = "rayline-openrouter-kv-cache-agt018-20260804"
SEMANTIC_BRANCH = "codex/rayline-remote-mvp"
SEMANTIC_REMOTE_REF = f"atlasfutures/{SEMANTIC_BRANCH}"
PATHFINDER_BRANCH = "codex/rayline-vsr-mvp"

PREREGISTRATION_COMMIT = "5319d7d7ba7780a53cda8f7631e8abeeb802e95e"
AUTHORIZATION_COMMIT = "cb4c6993f514f1fbd804e4ac03488994df049020"
GIT_SHA1_HEX_LENGTH = 40
NATIVE_APP_NAME = "rayline-router-openrouter-agt018"
NATIVE_WEBHOOK_LABEL = "router-openrouter-agt018"
REMOTE_APP_NAME = "rayline-arc-session-encoder-flashinfer-agt018"
REMOTE_CLASS_NAME = "SessionEncoder"
VLLM_COMMIT = "9f5ea81ca0aa570aea46baf82311a1139c1267ca"
REMOTE_ENGINE_BUILD_ID = f"vllm@{VLLM_COMMIT}+gdn-flashinfer-eager"
REMOTE_GDN_PREFILL_BACKEND = "flashinfer"
ARTIFACT_REVISION = "public-rayline-arc-openrouter-kv-cache-v4"
MAX_COMPLETION_TOKENS = 24

# The user approved fresh $10 authority on 2026-08-05, raising cumulative
# authority from the AGT017-era $134.31282402 to $144.31282402. AGT017's
# complete envelope was already charged in full at $133.087957466383.
AUTHORIZED_KEY_LIMIT_USD_PER_ARM = 0.05
AUTHORIZED_MAXIMUM_PAID_WALL_SECONDS = 20 * 60
MAXIMUM_PROVIDER_SPEND_USD = 2 * AUTHORIZED_KEY_LIMIT_USD_PER_ARM
REQUIRED_FINAL_RESERVE_USD = 1.20

# Authority bound 2026-08-05: the authorized values replaced the fail-closed
# zero placeholders in the same checkpoint that opened both authority pins.
SOURCE_CLOSED_KEY_LIMIT_USD_PER_ARM = AUTHORIZED_KEY_LIMIT_USD_PER_ARM
SOURCE_CLOSED_MAXIMUM_PAID_WALL_SECONDS = AUTHORIZED_MAXIMUM_PAID_WALL_SECONDS

AGT018_RESOURCE_BUDGET = BudgetContract(
    run_id=RUN_ID,
    previous_conservative_usd=133.087957466383,
    authorized_cumulative_usd=144.31282402,
    packet_ceiling_usd=9.1,
    # The provider limits are accounted separately below. Reserving $1.30 at
    # this layer guarantees at least $1.20 after both $0.05 keys are exhausted.
    required_reserve_usd=REQUIRED_FINAL_RESERVE_USD + MAXIMUM_PROVIDER_SPEND_USD,
    maximum_paid_wall_seconds=AUTHORIZED_MAXIMUM_PAID_WALL_SECONDS,
    encoder_replicas=2,
)


def successor_budget_receipt() -> dict[str, Any]:
    """Return the complete two-H100 plus two-provider-key envelope."""

    receipt = budget_receipt(AGT018_RESOURCE_BUDGET)
    cumulative = receipt["cumulative_if_full_envelope_usd"] + MAXIMUM_PROVIDER_SPEND_USD
    reserve = AGT018_RESOURCE_BUDGET.authorized_cumulative_usd - cumulative
    if reserve < REQUIRED_FINAL_RESERVE_USD:
        raise RuntimeError("AGT018 complete envelope exceeds user authority")
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
    "matched_completion_policy",
    "request_attempt_and_cost_envelopes",
    "privacy_and_cleanup",
)


def validate() -> dict[str, Any]:
    bound_pins = (PREREGISTRATION_COMMIT, AUTHORIZATION_COMMIT)
    if any(len(pin) != GIT_SHA1_HEX_LENGTH for pin in bound_pins):
        raise RuntimeError("AGT018 launch authority pins are not bound")
    if EXPECTED_REQUESTS_PER_DEPLOYMENT != EXPECTED_SEMANTIC_REQUESTS_PER_DEPLOYMENT:
        raise RuntimeError("AGT018 semantic workload request envelope diverged")
    if (
        MAXIMUM_LOGICAL_PROVIDER_REQUESTS != EXPECTED_MAXIMUM_LOGICAL_PROVIDER_REQUESTS
        or MAXIMUM_EXTERNAL_ATTEMPTS != EXPECTED_MAXIMUM_EXTERNAL_ATTEMPTS
    ):
        raise RuntimeError("AGT018 provider request or attempt envelope diverged")
    if (
        SOURCE_CLOSED_KEY_LIMIT_USD_PER_ARM != AUTHORIZED_KEY_LIMIT_USD_PER_ARM
        or SOURCE_CLOSED_MAXIMUM_PAID_WALL_SECONDS
        != AUTHORIZED_MAXIMUM_PAID_WALL_SECONDS
    ):
        raise RuntimeError("AGT018 authorized resource envelope diverged")
    receipt = successor_budget_receipt()
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "source_closed": False,
        "launch_authorized": True,
        "requires_new_budget_authority": False,
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
                AGT018_RESOURCE_BUDGET.authorized_cumulative_usd
            ),
            "maximum_complete_packet_usd": receipt["maximum_complete_packet_usd"],
            "reserve_after_complete_envelope_usd": (
                receipt["reserve_after_complete_envelope_usd"]
            ),
        },
    }
