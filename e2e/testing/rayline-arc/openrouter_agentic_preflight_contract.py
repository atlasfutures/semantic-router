#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Privacy-safe wire contract for the pre-encoder OpenRouter preflight."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from modal_fullstack_inputs import CANDIDATE_PROMPTS
from openrouter_agentic_workload import PROVIDER_NAMES, WORKERS

REPORT_SCHEMA = "rayline.arc.openrouter-agentic-preflight.v1"
ENVIRONMENT_KEY = "RAYLINE_ARC_E2E_TRANSPORT_PREFLIGHT"
MAX_PROVIDER_REQUESTS = 1 + len(WORKERS)
MAX_EXTERNAL_ATTEMPTS = MAX_PROVIDER_REQUESTS * 2 + len(WORKERS)


def _response_model_matches(response_model: str, expected_model: str) -> bool:
    return response_model == expected_model or response_model.startswith(
        f"{expected_model}-"
    )


def validate_report(report: Any, *, require_reuse: bool) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise TypeError("agentic transport preflight report was malformed")
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or report.get("maximum_provider_requests") != MAX_PROVIDER_REQUESTS
        or report.get("maximum_external_attempts") != MAX_EXTERNAL_ATTEMPTS
    ):
        raise RuntimeError("agentic transport preflight contract diverged")
    status = report.get("status")
    if status == "failed":
        if require_reuse:
            raise RuntimeError("agentic transport preflight did not pass")
        completed = report.get("completed_provider_requests")
        provider_requests = report.get("provider_requests")
        attempts = report.get("external_attempts")
        cost = report.get("cost_usd")
        if (
            not isinstance(completed, int)
            or completed < 0
            or not isinstance(provider_requests, int)
            or provider_requests != completed + 1
            or provider_requests > MAX_PROVIDER_REQUESTS
            or not isinstance(attempts, int)
            or attempts < provider_requests
            or attempts > MAX_EXTERNAL_ATTEMPTS
            or not isinstance(cost, (int, float))
            or cost < 0
        ):
            raise RuntimeError("agentic transport preflight failure was malformed")
        return report
    if status != "passed" or report.get("provider_requests") != MAX_PROVIDER_REQUESTS:
        raise RuntimeError("agentic transport preflight did not pass all probes")
    if require_reuse and (
        report.get("envoy_container_reused") is not True
        or report.get("ephemeral_key_reused") is not True
    ):
        raise RuntimeError("agentic transport preflight reuse contract diverged")
    attempts = report.get("external_attempts")
    cost = report.get("cost_usd")
    workers = report.get("workers")
    if (
        not isinstance(attempts, int)
        or attempts < MAX_PROVIDER_REQUESTS
        or attempts > MAX_EXTERNAL_ATTEMPTS
        or not isinstance(cost, (int, float))
        or cost < 0
        or not isinstance(workers, dict)
    ):
        raise RuntimeError("agentic transport preflight totals were malformed")
    for worker, model in WORKERS.items():
        identity = workers.get(worker)
        if not isinstance(identity, dict) or not _response_model_matches(
            str(identity.get("model", "")), model
        ):
            raise RuntimeError("agentic preflight model identity diverged")
        if identity.get("provider") != PROVIDER_NAMES[worker]:
            raise RuntimeError("agentic preflight provider identity diverged")
    return report


def from_environment(environment: Mapping[str, str]) -> dict[str, Any]:
    try:
        report = json.loads(environment.get(ENVIRONMENT_KEY, ""))
    except json.JSONDecodeError as error:
        raise RuntimeError("agentic transport preflight was unavailable") from error
    return validate_report(report, require_reuse=True)


def encode_report(report: dict[str, Any], openrouter_key: str) -> str:
    encoded = json.dumps(report, separators=(",", ":"), sort_keys=True)
    if openrouter_key in encoded:
        raise RuntimeError("OpenRouter credential entered the preflight report")
    if any(anchor in encoded for anchor in CANDIDATE_PROMPTS):
        raise RuntimeError("agentic preflight report included a routing anchor")
    return encoded
