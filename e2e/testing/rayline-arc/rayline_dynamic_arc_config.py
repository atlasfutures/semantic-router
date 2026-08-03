#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Render the production dynamic-membership ARC benchmark configuration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rayline_parity_arc_config import build_config

DYNAMIC_DECISION_NAME = "rayline-three-arm-parity"
DYNAMIC_KEY_PREFIX = "vsr:rayline-dyn006:"
DYNAMIC_IDLE_TTL_SECONDS = 5 * 60


class DynamicConfigError(ValueError):
    """The generated parity configuration does not expose the ARC seam."""


def build_dynamic_config(
    template: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    artifact_mount_path: str,
    encoder_build_id: str,
    encoder_plugin_version: str,
    worker_endpoint: str,
) -> dict[str, Any]:
    config = build_config(
        template,
        manifest,
        artifact_mount_path=artifact_mount_path,
        encoder_base_url="https://dynamic-membership.invalid",
        encoder_build_id=encoder_build_id,
        encoder_plugin_version=encoder_plugin_version,
        worker_endpoint=worker_endpoint,
    )
    decisions = config.get("routing", {}).get("decisions")
    if not isinstance(decisions, list) or len(decisions) != 1:
        raise DynamicConfigError("generated config must contain one decision")
    arc = decisions[0].get("algorithm", {}).get("rayline_arc")
    if not isinstance(arc, dict):
        raise DynamicConfigError("generated config omits Rayline ARC")
    encoder = arc.get("encoder")
    episode = arc.get("episode")
    if not isinstance(encoder, dict) or not isinstance(episode, dict):
        raise DynamicConfigError("generated config omits encoder or episode")
    encoder.pop("base_url", None)
    encoder["membership"] = {
        "schema_version": "rayline.arc.encoder-membership.v1",
        "source": "redis",
        "refresh_seconds": 1,
    }
    encoder["failover"] = {
        "schema_version": "rayline.arc.encoder-failover.v1",
        "unavailable_status_codes": [404, 410, 502, 503, 504],
        "unavailable_cooldown_seconds": 1,
        "max_remaps": 1,
    }
    episode["close_header"] = "x-rayline-episode-close"
    episode["key_prefix"] = DYNAMIC_KEY_PREFIX
    episode["idle_ttl_seconds"] = DYNAMIC_IDLE_TTL_SECONDS
    return config
