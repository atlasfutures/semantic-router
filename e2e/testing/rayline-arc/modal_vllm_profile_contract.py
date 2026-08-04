# SPDX-License-Identifier: Apache-2.0

"""Frozen authority and acceptance contract for the PERF029 GDN A/B."""

from __future__ import annotations

from dataclasses import dataclass

from rayline_three_arm_budget import BudgetContract

RUN_ID = "rayline-vllm-gdn-perf029-20260804"
SEMANTIC_BRANCH = "codex/rayline-remote-mvp"
SEMANTIC_REMOTE_REF = f"atlasfutures/{SEMANTIC_BRANCH}"
PREREGISTRATION_COMMIT = "dbcaf022ab0ce8c66029414d63da88423ed7bc00"
AUTHORIZATION_COMMIT = ""
REQUIRED_MODAL_VERSION = "1.5.1"
MODAL_ENVIRONMENT = "dev"
VLLM_COMMIT = "9f5ea81ca0aa570aea46baf82311a1139c1267ca"
REFERENCE_LABEL = "torch_reference"
CANDIDATE_LABEL = "flashinfer"
PROFILE_LABELS = (REFERENCE_LABEL, CANDIDATE_LABEL)

MIN_COSINE_SIMILARITY = 0.9999
MAX_ABSOLUTE_DRIFT = 0.01
MAX_SYNTHETIC_SCORE_DRIFT = 0.005
MAX_SELECTION_FLIPS = 0
MAX_CANDIDATE_TO_REFERENCE_ENGINE_RATIO = 0.80

EPISODES = 2
STEPS = 3
MODES = ("retained", "replay")
BOOTSTRAP_REQUESTS_PER_PROFILE = 3
WARMUP_REQUESTS_PER_PROFILE = BOOTSTRAP_REQUESTS_PER_PROFILE + STEPS
MEASURED_REQUESTS_PER_PROFILE = EPISODES * STEPS * len(MODES)
MAXIMUM_POOLING_REQUESTS = len(PROFILE_LABELS) * (
    WARMUP_REQUESTS_PER_PROFILE + MEASURED_REQUESTS_PER_PROFILE
)


@dataclass(frozen=True)
class Profile:
    label: str
    app_name: str
    gdn_prefill_backend: str
    engine_build_id: str


PROFILES = {
    REFERENCE_LABEL: Profile(
        label=REFERENCE_LABEL,
        app_name="rayline-arc-session-encoder-reference-perf029",
        gdn_prefill_backend="torch_reference",
        engine_build_id=f"vllm@{VLLM_COMMIT}+gdn-torch-reference-eager",
    ),
    CANDIDATE_LABEL: Profile(
        label=CANDIDATE_LABEL,
        app_name="rayline-arc-session-encoder-flashinfer-perf029",
        gdn_prefill_backend="flashinfer",
        engine_build_id=f"vllm@{VLLM_COMMIT}+gdn-flashinfer-eager",
    ),
}

PERF029_BUDGET = BudgetContract(
    run_id=RUN_ID,
    previous_conservative_usd=105.895336666383,
    authorized_cumulative_usd=134.31282402,
    packet_ceiling_usd=10.0,
    required_reserve_usd=15.0,
    maximum_paid_wall_seconds=20 * 60,
    encoder_replicas=2,
)
