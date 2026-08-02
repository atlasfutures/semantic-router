#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Capture a bounded aggregate ARC metrics receipt before Compose teardown."""

from __future__ import annotations

import json
import math
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

TELEMETRY_SCHEMA = "rayline.vllm.arc-telemetry.v1"
HTTP_OK = 200
MIN_QUOTED_VALUE_LENGTH = 2
TOKEN_KINDS = ("full", "serialized", "retained", "appended", "cached", "truncated")
SESSION_ACTIONS = ("created", "appended", "rebuilt", "reused")


class TelemetryError(ValueError):
    """The ARC metric snapshot is missing, malformed, or internally inconsistent."""


def _metric_key(name: str, labels: Mapping[str, str] | None = None) -> str:
    if not labels:
        return name
    rendered = ",".join(f'{key}="{value}"' for key, value in sorted(labels.items()))
    return f"{name}{{{rendered}}}"


def _parse_labels(raw: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    if not raw:
        return labels
    for item in raw.split(","):
        key, separator, value = item.partition("=")
        if (
            not separator
            or len(value) < MIN_QUOTED_VALUE_LENGTH
            or value[0] != '"'
            or value[-1] != '"'
        ):
            raise TelemetryError("ARC metric labels are malformed")
        labels[key] = value[1:-1]
    return labels


def _parse_metrics(text: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        metric, separator, raw_value = line.rpartition(" ")
        if not separator:
            continue
        if "{" in metric:
            name, raw_labels = metric.split("{", 1)
            if not raw_labels.endswith("}"):
                raise TelemetryError("ARC metric selector is malformed")
            key = _metric_key(name, _parse_labels(raw_labels[:-1]))
        else:
            key = metric
        try:
            value = float(raw_value)
        except ValueError as error:
            raise TelemetryError(f"ARC metric {key} is not numeric") from error
        if not math.isfinite(value) or value < 0:
            raise TelemetryError(f"ARC metric {key} is not finite and non-negative")
        parsed[key] = value
    return parsed


def _value(
    metrics: Mapping[str, float],
    name: str,
    labels: Mapping[str, str] | None = None,
    *,
    optional: bool = False,
) -> int | float:
    key = _metric_key(name, labels)
    if key not in metrics:
        if optional:
            return 0
        raise TelemetryError(f"ARC metric is missing: {key}")
    value = metrics[key]
    return int(value) if value.is_integer() else value


def parse_arc_metrics(text: str) -> dict[str, Any]:
    metrics = _parse_metrics(text)
    actions = {
        action: _value(
            metrics,
            "llm_rayline_arc_session_actions_total",
            {"action": action},
            optional=action in {"rebuilt", "reused"},
        )
        for action in SESSION_ACTIONS
    }
    tokens = {
        kind: {
            "sum": _value(metrics, "llm_rayline_arc_tokens_sum", {"kind": kind}),
            "count": _value(metrics, "llm_rayline_arc_tokens_count", {"kind": kind}),
        }
        for kind in TOKEN_KINDS
    }
    if actions["created"] + actions["appended"] != tokens["full"]["count"]:
        raise TelemetryError("ARC session actions do not reconcile with request count")
    return {
        "schema_version": TELEMETRY_SCHEMA,
        "component_ready": _value(
            metrics,
            "llm_rayline_arc_component_ready",
            {"component": "artifact_head_encoder"},
        ),
        "session_actions": actions,
        "tokens": tokens,
        "cache_miss_tokens": {
            "sum": _value(metrics, "llm_rayline_arc_cache_miss_tokens_sum"),
            "count": _value(metrics, "llm_rayline_arc_cache_miss_tokens_count"),
        },
    }


def capture_arc_telemetry(url: str, output: Path) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=5) as response:
        if response.status != HTTP_OK:
            raise TelemetryError(
                f"ARC metrics endpoint returned HTTP {response.status}"
            )
        receipt = parse_arc_metrics(response.read().decode(errors="strict"))
    if receipt["component_ready"] != 1:
        raise TelemetryError("ARC component was not ready at telemetry capture")
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt
