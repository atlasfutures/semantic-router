#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Bounded packet catalog for the OpenRouter full-stack launcher."""

from __future__ import annotations

from pathlib import Path

from openrouter_fullstack_state import EncoderDeployment, RunPacket
from openrouter_kv_cache_matched_contract import (
    AGT017_RESOURCE_BUDGET,
    FLASHINFER_APP_NAME,
    FLASHINFER_CLASS_NAME,
    FLASHINFER_ENGINE_BUILD_ID,
    OPENROUTER_KEY_LIMIT_USD_PER_ARM,
)
from openrouter_kv_cache_matched_contract import (
    RUN_ID as AGT017_RUN_ID,
)
from openrouter_kv_cache_successor_contract import (
    ARTIFACT_REVISION as AGT018_ARTIFACT_REVISION,
)
from openrouter_kv_cache_successor_contract import (
    REMOTE_APP_NAME as AGT018_REMOTE_APP_NAME,
)
from openrouter_kv_cache_successor_contract import (
    REMOTE_CLASS_NAME as AGT018_REMOTE_CLASS_NAME,
)
from openrouter_kv_cache_successor_contract import (
    REMOTE_ENGINE_BUILD_ID as AGT018_REMOTE_ENGINE_BUILD_ID,
)
from openrouter_kv_cache_successor_contract import (
    RUN_ID as AGT018_RUN_ID,
)
from openrouter_kv_cache_successor_contract import (
    SOURCE_CLOSED_KEY_LIMIT_USD_PER_ARM,
    SOURCE_CLOSED_MAXIMUM_PAID_WALL_SECONDS,
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


def _ephemeral_encoder(
    *,
    app_name: str,
    class_name: str,
    build_id: str,
    service_path: Path,
) -> EncoderDeployment:
    return EncoderDeployment(
        app_name=app_name,
        class_name=class_name,
        build_id=build_id,
        deployment_source_commit="runtime-attested-launch-source",
        plugin_source_digest="runtime-attested-launch-source",
        deploy_service_path=service_path,
        ephemeral=True,
    )


def _successor_packet(
    deploy_root: Path,
    script_root: Path,
    encoder: EncoderDeployment,
) -> RunPacket:
    return RunPacket(
        compose_override=deploy_root / "compose-openrouter-kv-cache.yaml",
        config=deploy_root / "config-openrouter-kv-cache.yaml",
        driver=script_root / "openrouter_kv_cache_successor_remote.py",
        project_name="rayline-arc-openrouter-kv-cache-flashinfer-agt018",
        key_limit_usd=SOURCE_CLOSED_KEY_LIMIT_USD_PER_ARM,
        maximum_seconds=SOURCE_CLOSED_MAXIMUM_PAID_WALL_SECONDS,
        protected_encoder=True,
        provider_preflight=True,
        encoder=encoder,
        expected_run_id=AGT018_RUN_ID,
        modal_environment="dev",
        artifact_revision=AGT018_ARTIFACT_REVISION,
    )


def packet_catalog(
    repo_root: Path,
    default_encoder: EncoderDeployment,
    session_service_path: Path,
    *,
    canary_key_limit_usd: float,
    maximum_canary_seconds: int,
) -> dict[str, RunPacket]:
    deploy_root = repo_root / "deploy/compose/rayline-arc"
    script_root = Path(__file__).resolve().parent
    agentic_override = deploy_root / "compose-openrouter-agentic.yaml"
    agentic_config = deploy_root / "config-openrouter-agentic.yaml"
    flashinfer_encoder = _ephemeral_encoder(
        app_name=FLASHINFER_APP_NAME,
        class_name=FLASHINFER_CLASS_NAME,
        build_id=FLASHINFER_ENGINE_BUILD_ID,
        service_path=session_service_path,
    )
    successor_encoder = _ephemeral_encoder(
        app_name=AGT018_REMOTE_APP_NAME,
        class_name=AGT018_REMOTE_CLASS_NAME,
        build_id=AGT018_REMOTE_ENGINE_BUILD_ID,
        service_path=session_service_path,
    )
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
            provider_preflight=True,
            encoder=flashinfer_encoder,
            expected_run_id=AGT017_RUN_ID,
            modal_environment="dev",
            artifact_revision="public-rayline-arc-openrouter-kv-cache-v2",
        ),
        "kv-cache-flashinfer-agt018": _successor_packet(
            deploy_root,
            script_root,
            successor_encoder,
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
