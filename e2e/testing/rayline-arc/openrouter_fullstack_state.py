#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Runtime state contracts for the bounded OpenRouter launcher."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EncoderDeployment:
    app_name: str
    class_name: str
    build_id: str
    deployment_source_commit: str
    plugin_source_digest: str
    app_id: str = ""
    expected_host: str = ""
    deploy_service_path: Path | None = None
    ephemeral: bool = False


@dataclass(frozen=True)
class RunPacket:
    compose_override: Path
    config: Path
    driver: Path
    project_name: str
    key_limit_usd: float
    maximum_seconds: int
    protected_encoder: bool
    preflight_driver: Path | None = None
    encoder: EncoderDeployment | None = None


@dataclass
class RuntimeState:
    environment: dict[str, str]
    proxy_token: Any = None
    ephemeral_key: str = ""
    key_hash: str = ""
    encoder_instance: Any = None
    encoder_autoscaler_pinned: bool = False
    transport_preflight: dict[str, Any] | None = None
    encoder_app_id: str = ""
    encoder_base_url: str = ""
    encoder_deployed: bool = False
    encoder_owned: bool = False


@dataclass
class LaunchOutcome:
    run_failure: Exception | None = None
    evidence_failure: Exception | None = None
    cleanup_failure: Exception | None = None
    paid_elapsed: float | None = None
