# SPDX-License-Identifier: Apache-2.0

"""Frozen authority, identity, and cost contract for matched AGT017 E2E."""

from __future__ import annotations

from typing import Any

from rayline_three_arm_budget import BudgetContract, budget_receipt

RUN_ID = "rayline-openrouter-kv-cache-agt017-20260804"
SEMANTIC_BRANCH = "codex/rayline-remote-mvp"
SEMANTIC_REMOTE_REF = f"atlasfutures/{SEMANTIC_BRANCH}"
PATHFINDER_BRANCH = "codex/rayline-vsr-mvp"
PREREGISTRATION_COMMIT = "b827fdafeae14ee0699107e74ac3c870d33f3388"
AUTHORIZATION_COMMIT = "eee540c80a3650e805a8c68c1576376739898913"

NATIVE_APP_NAME = "rayline-router-openrouter-agt017"
NATIVE_WEBHOOK_LABEL = "router-openrouter-agt017"
FLASHINFER_APP_NAME = "rayline-arc-session-encoder-flashinfer-agt017"
FLASHINFER_CLASS_NAME = "SessionEncoder"
VLLM_COMMIT = "9f5ea81ca0aa570aea46baf82311a1139c1267ca"
FLASHINFER_ENGINE_BUILD_ID = f"vllm@{VLLM_COMMIT}+gdn-flashinfer-eager"
FLASHINFER_BACKEND = "flashinfer"
ARTIFACT_REVISION = "public-rayline-arc-openrouter-kv-cache-v2"

MAX_COMPLETION_TOKENS = 24
OPENROUTER_KEY_LIMIT_USD_PER_ARM = 0.05
MAXIMUM_PROVIDER_SPEND_USD = 2 * OPENROUTER_KEY_LIMIT_USD_PER_ARM
REQUIRED_FINAL_RESERVE_USD = 1.20

AGT017_RESOURCE_BUDGET = BudgetContract(
    run_id=RUN_ID,
    previous_conservative_usd=123.957083866383,
    authorized_cumulative_usd=134.31282402,
    packet_ceiling_usd=9.1,
    # The provider limits are accounted separately below. Reserving $1.30 at
    # this layer guarantees at least $1.20 after both $0.05 keys are exhausted.
    required_reserve_usd=REQUIRED_FINAL_RESERVE_USD + MAXIMUM_PROVIDER_SPEND_USD,
    maximum_paid_wall_seconds=20 * 60,
    encoder_replicas=2,
)


def matched_budget_receipt() -> dict[str, Any]:
    """Return the complete two-H100 plus two-provider-key envelope."""

    receipt = budget_receipt(AGT017_RESOURCE_BUDGET)
    cumulative = receipt["cumulative_if_full_envelope_usd"] + MAXIMUM_PROVIDER_SPEND_USD
    reserve = AGT017_RESOURCE_BUDGET.authorized_cumulative_usd - cumulative
    if reserve < REQUIRED_FINAL_RESERVE_USD:
        raise RuntimeError("AGT017 complete envelope exceeds user authority")
    return {
        **receipt,
        "maximum_provider_spend_usd": MAXIMUM_PROVIDER_SPEND_USD,
        "maximum_complete_packet_usd": (
            receipt["maximum_resource_envelope_usd"] + MAXIMUM_PROVIDER_SPEND_USD
        ),
        "cumulative_if_complete_envelope_usd": cumulative,
        "reserve_after_complete_envelope_usd": reserve,
    }
