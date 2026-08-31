#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Privacy-safe contract for the shared no-GPU provider availability gate."""

from __future__ import annotations

import json
from typing import Any

from modal_fullstack_inputs import CANDIDATE_PROMPTS
from openrouter_agentic_workload import PROVIDER_NAMES, WORKERS

REPORT_SCHEMA = "rayline.openrouter-provider-availability.v1"
MAX_PROVIDER_REQUESTS = len(WORKERS)
MAX_EXTERNAL_ATTEMPTS = MAX_PROVIDER_REQUESTS * 2


def _model_matches(actual: str, expected: str) -> bool:
    return actual == expected or actual.startswith(f"{expected}-")


def validate_report(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise TypeError("provider availability report was malformed")
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or report.get("maximum_provider_requests") != MAX_PROVIDER_REQUESTS
        or report.get("maximum_external_attempts") != MAX_EXTERNAL_ATTEMPTS
        or report.get("performance_inference_admissible") is not False
    ):
        raise RuntimeError("provider availability contract diverged")
    requests = report.get("provider_requests")
    attempts = report.get("external_attempts")
    cost = report.get("cost_usd")
    if (
        not isinstance(requests, int)
        or requests < 1
        or requests > MAX_PROVIDER_REQUESTS
        or not isinstance(attempts, int)
        or attempts < requests
        or attempts > MAX_EXTERNAL_ATTEMPTS
        or not isinstance(cost, (int, float))
        or cost < 0
    ):
        raise RuntimeError("provider availability totals were malformed")
    if report.get("status") == "failed":
        if report.get("completed_provider_requests") != requests - 1:
            raise RuntimeError("provider availability failure count diverged")
        if report.get("failed_worker") not in WORKERS:
            raise RuntimeError("provider availability failed worker was unknown")
        return report
    if report.get("status") != "passed" or requests != MAX_PROVIDER_REQUESTS:
        raise RuntimeError("provider availability gate did not pass every worker")
    workers = report.get("workers")
    if not isinstance(workers, dict) or set(workers) != set(WORKERS):
        raise RuntimeError("provider availability worker coverage diverged")
    for worker, expected_model in WORKERS.items():
        identity = workers[worker]
        if not isinstance(identity, dict) or not _model_matches(
            str(identity.get("model", "")), expected_model
        ):
            raise RuntimeError("provider availability model identity diverged")
        if identity.get("provider") not in PROVIDER_NAMES[worker]:
            raise RuntimeError("provider availability provider identity diverged")
    return report


def encode_report(report: dict[str, Any], openrouter_key: str) -> str:
    validated = validate_report(report)
    encoded = json.dumps(validated, separators=(",", ":"), sort_keys=True)
    if openrouter_key and openrouter_key in encoded:
        raise RuntimeError("OpenRouter credential entered provider availability report")
    if any(anchor in encoded for anchor in CANDIDATE_PROMPTS):
        raise RuntimeError("routing anchor entered provider availability report")
    return encoded
