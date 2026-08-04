#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Bounded packet catalog for the OpenRouter full-stack launcher."""

from __future__ import annotations

from pathlib import Path

from openrouter_fullstack_state import EncoderDeployment, RunPacket
from openrouter_kv_cache_matched_contract import (
    AGT017_RESOURCE_BUDGET,
    OPENROUTER_KEY_LIMIT_USD_PER_ARM,
)


def _agentic_packet(
    repo_root: Path,
    default_encoder: EncoderDeployment,
    driver_name: str,
    project_name: str,
) -> RunPacket:
    return RunPacket(
        compose_override=(
            repo_root / "deploy/compose/rayline-arc/compose-openrouter-agentic.yaml"
        ),
        config=repo_root / "deploy/compose/rayline-arc/config-openrouter-agentic.yaml",
        driver=Path(__file__).with_name(driver_name),
        project_name=project_name,
        key_limit_usd=0.75,
        maximum_seconds=30 * 60,
        protected_encoder=True,
        preflight_driver=Path(__file__).with_name("openrouter_agentic_preflight.py"),
        encoder=default_encoder,
    )


def packet_catalog(
    repo_root: Path,
    default_encoder: EncoderDeployment,
    flashinfer_encoder: EncoderDeployment,
    *,
    canary_key_limit_usd: float,
    maximum_canary_seconds: int,
) -> dict[str, RunPacket]:
    deploy_root = repo_root / "deploy/compose/rayline-arc"
    script_root = Path(__file__).resolve().parent
    agentic_override = deploy_root / "compose-openrouter-agentic.yaml"
    agentic_config = deploy_root / "config-openrouter-agentic.yaml"
    return {
        "canary": RunPacket(
            compose_override=deploy_root / "compose-openrouter.yaml",
            config=deploy_root / "config-openrouter.yaml",
            driver=script_root / "openrouter_fullstack_canary.py",
            project_name="rayline-arc-openrouter",
            key_limit_usd=canary_key_limit_usd,
            maximum_seconds=maximum_canary_seconds,
            protected_encoder=True,
            encoder=default_encoder,
        ),
        "agentic": _agentic_packet(
            repo_root,
            default_encoder,
            "openrouter_agentic_benchmark.py",
            "rayline-arc-openrouter-agentic",
        ),
        "agentic-stage": _agentic_packet(
            repo_root,
            default_encoder,
            "openrouter_agentic_stage_benchmark.py",
            "rayline-arc-openrouter-agentic-stage",
        ),
        "kv-cache": RunPacket(
            compose_override=agentic_override,
            config=agentic_config,
            driver=script_root / "openrouter_kv_cache_remote.py",
            project_name="rayline-arc-openrouter-kv-cache",
            key_limit_usd=0.50,
            maximum_seconds=20 * 60,
            protected_encoder=True,
            encoder=default_encoder,
        ),
        "kv-cache-flashinfer": RunPacket(
            compose_override=deploy_root / "compose-openrouter-kv-cache.yaml",
            config=deploy_root / "config-openrouter-kv-cache.yaml",
            driver=script_root / "openrouter_kv_cache_remote.py",
            project_name="rayline-arc-openrouter-kv-cache-flashinfer-agt017",
            key_limit_usd=OPENROUTER_KEY_LIMIT_USD_PER_ARM,
            maximum_seconds=AGT017_RESOURCE_BUDGET.maximum_paid_wall_seconds,
            protected_encoder=True,
            encoder=flashinfer_encoder,
        ),
        "gateway-shape": RunPacket(
            compose_override=agentic_override,
            config=agentic_config,
            driver=script_root / "openrouter_gateway_shape_diagnostic.py",
            project_name="rayline-arc-openrouter-gateway-shape",
            key_limit_usd=0.05,
            maximum_seconds=5 * 60,
            protected_encoder=False,
        ),
        "gateway-prime": RunPacket(
            compose_override=agentic_override,
            config=agentic_config,
            driver=script_root / "openrouter_gateway_prime_diagnostic.py",
            project_name="rayline-arc-openrouter-gateway-prime",
            key_limit_usd=0.05,
            maximum_seconds=5 * 60,
            protected_encoder=False,
        ),
    }
