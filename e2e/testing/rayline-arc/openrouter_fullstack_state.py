#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Runtime state contracts for the bounded OpenRouter launcher."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


@dataclass
class RuntimeState:
    environment: dict[str, str]
    proxy_token: Any = None
    ephemeral_key: str = ""
    key_hash: str = ""
    encoder_instance: Any = None
    encoder_autoscaler_pinned: bool = False
    transport_preflight: dict[str, Any] | None = None
