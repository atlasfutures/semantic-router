#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Capture production replica-route and close aggregates for DYN006."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

from rayline_three_arm_telemetry import (
    HTTP_OK,
    TelemetryError,
    _parse_metrics,
    _value,
    parse_arc_metrics,
)

DYNAMIC_TELEMETRY_SCHEMA = "rayline.vllm.dynamic-arc-telemetry.v1"


def parse_dynamic_arc_metrics(text: str) -> dict[str, Any]:
    metrics = _parse_metrics(text)
    base = parse_arc_metrics(text)
    return {
        **base,
        "schema_version": DYNAMIC_TELEMETRY_SCHEMA,
        "replica_routes": {
            outcome: _value(
                metrics,
                "llm_rayline_arc_encoder_replica_routes_total",
                {"outcome": outcome},
                optional=outcome == "failover",
            )
            for outcome in ("direct", "failover")
        },
        "session_closes": {
            outcome: _value(
                metrics,
                "llm_rayline_arc_encoder_session_closes_total",
                {"outcome": outcome},
                optional=outcome in {"unavailable", "failed"},
            )
            for outcome in ("closed", "unavailable", "failed")
        },
    }


def capture_dynamic_arc_telemetry(url: str, output: Path) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=5) as response:
        if response.status != HTTP_OK:
            raise TelemetryError(
                f"ARC metrics endpoint returned HTTP {response.status}"
            )
        receipt = parse_dynamic_arc_metrics(response.read().decode(errors="strict"))
    if receipt["component_ready"] != 1:
        raise TelemetryError("ARC component was not ready at telemetry capture")
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
