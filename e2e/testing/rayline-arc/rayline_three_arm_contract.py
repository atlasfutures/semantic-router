#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Immutable identities and launch authority for three-arm measurements."""

from __future__ import annotations

from dataclasses import dataclass

from rayline_three_arm_budget import BudgetContract

PERF015_RUN_ID = "rayline-three-arm-directional-perf015-20260802"
PERF016_RUN_ID = "rayline-three-arm-repeat-perf016-20260802"


@dataclass(frozen=True)
class RunContract:
    """Exact mutable-resource ownership for one preregistered run."""

    run_id: str
    compose_project: str
    temporary_prefix: str
    budget: BudgetContract


@dataclass(frozen=True)
class FrozenIdentity:
    """Shared source and service identity for every three-arm run."""

    encoder_app_name: str
    encoder_url: str
    engine_build_id: str
    plugin_version: str
    checkpoint_repo: str
    checkpoint_revision: str
    checkpoint_path: str
    checkpoint_sha256: str
    pathfinder_branch: str
    semantic_branch: str
    modal_environment: str
    required_modal_version: str


IDENTITY = FrozenIdentity(
    encoder_app_name="rayline-arc-session-encoder",
    encoder_url=(
        "https://atlasfutures-dev--rayline-arc-session-encoder-sessionenc-"
        "2d82ac.modal.run"
    ),
    engine_build_id="vllm@9f5ea81ca0aa570aea46baf82311a1139c1267ca",
    plugin_version="rayline-arc-io@0.1.0",
    checkpoint_repo="rayline-ai/mtrouter-c82",
    checkpoint_revision="a06a4cc194761cfb39f92549ba305b0a8173a3d4",
    checkpoint_path="provenance/source/mtrouter_estimator.pt",
    checkpoint_sha256=(
        "c2b0e63216c11f1496b47b22dff9f6c83baa6ef065e205a34897deff7493920f"
    ),
    pathfinder_branch="codex/rayline-vsr-mvp",
    semantic_branch="codex/rayline-remote-mvp",
    modal_environment="dev",
    required_modal_version="1.5.1",
)

NON_RUNTIME_SECRET_NAMES = (
    "ANTHROPIC_API_KEY",
    "DEEPSEEK_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "TOGETHER_API_KEY",
    "HF_TOKEN",
    "HF_API_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
    "OP_SERVICE_ACCOUNT_TOKEN",
    "GITHUB_TOKEN",
    "GH_TOKEN",
)


PERF015 = RunContract(
    run_id=PERF015_RUN_ID,
    compose_project="rayline-three-arm-perf015",
    temporary_prefix="rayline-perf015-",
    budget=BudgetContract(
        run_id=PERF015_RUN_ID,
        previous_conservative_usd=39.31282402,
        authorized_cumulative_usd=59.31282402,
        packet_ceiling_usd=15.0,
        required_reserve_usd=5.0,
        maximum_paid_wall_seconds=90 * 60,
    ),
)

PERF016 = RunContract(
    run_id=PERF016_RUN_ID,
    compose_project="rayline-three-arm-perf016",
    temporary_prefix="rayline-perf016-",
    budget=BudgetContract(
        run_id=PERF016_RUN_ID,
        previous_conservative_usd=49.47255682,
        authorized_cumulative_usd=59.31282402,
        packet_ceiling_usd=7.0,
        required_reserve_usd=3.0,
        maximum_paid_wall_seconds=40 * 60,
    ),
)

# Historical contracts remain calculable, but closed experiment IDs are not
# launchable. Changing this value requires a new preregistered experiment ID.
LAUNCHABLE_CONTRACT = PERF016


def resolve_launch_contract(run_id: str) -> RunContract:
    if run_id != LAUNCHABLE_CONTRACT.run_id:
        raise ValueError(
            f"launcher only permits preregistered run id "
            f"{LAUNCHABLE_CONTRACT.run_id}"
        )
    return LAUNCHABLE_CONTRACT
