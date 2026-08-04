#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Check all three OpenRouter workers before any paid GPU is launched."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from modal_fullstack_canary import _episode_id
from openrouter_agentic_benchmark import (
    MAX_DATA_PLANE_ATTEMPTS,
    OpenRouterHTTPError,
    _stream_request,
)
from openrouter_agentic_workload import WORKERS
from openrouter_agentic_workload import candidate_case as _candidate_case
from openrouter_provider_preflight_contract import (
    MAX_EXTERNAL_ATTEMPTS,
    MAX_PROVIDER_REQUESTS,
    REPORT_SCHEMA,
    encode_report,
    validate_report,
)

MAX_COMPLETION_TOKENS = 1


def _failure(
    *,
    run_id: str,
    worker: str,
    results: list[dict[str, Any]],
    error: Exception,
) -> dict[str, Any]:
    external_attempts = sum(int(result["external_attempts"]) for result in results)
    if isinstance(error, OpenRouterHTTPError):
        external_attempts += error.external_attempts
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "run_id": run_id,
        "status": "failed",
        "failed_worker": worker,
        "failure_class": type(error).__name__,
        "provider_requests": len(results) + 1,
        "completed_provider_requests": len(results),
        "maximum_provider_requests": MAX_PROVIDER_REQUESTS,
        "external_attempts": max(len(results) + 1, external_attempts),
        "maximum_external_attempts": MAX_EXTERNAL_ATTEMPTS,
        "cost_usd": sum(float(result["cost_usd"]) for result in results),
        "performance_inference_admissible": False,
    }
    if isinstance(error, OpenRouterHTTPError):
        report.update(
            {
                "http_status": error.status_code,
                "error_category": error.error_category,
                "error_type": error.error_type,
                "provider_code": error.provider_code,
            }
        )
    return validate_report(report)


def run_preflight(
    *, openrouter_key: str, run_id: str, timeout_seconds: float
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for index, worker in enumerate(WORKERS):
        try:
            result = _stream_request(
                path="direct",
                case=_candidate_case(index),
                expected_worker=worker,
                gateway_url="",
                openrouter_key=openrouter_key,
                episode_id=_episode_id(run_id, f"provider-preflight-{worker}"),
                timeout_seconds=timeout_seconds,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                maximum_attempts=MAX_DATA_PLANE_ATTEMPTS,
            )
        except Exception as error:
            return _failure(
                run_id=run_id,
                worker=worker,
                results=results,
                error=error,
            )
        results.append(result)
    attempts = sum(int(result["external_attempts"]) for result in results)
    report = {
        "schema_version": REPORT_SCHEMA,
        "run_id": run_id,
        "status": "passed",
        "provider_requests": len(results),
        "maximum_provider_requests": MAX_PROVIDER_REQUESTS,
        "external_attempts": attempts,
        "maximum_external_attempts": MAX_EXTERNAL_ATTEMPTS,
        "retries": attempts - len(results),
        "cost_usd": sum(float(result["cost_usd"]) for result in results),
        "maximum_completion_tokens": MAX_COMPLETION_TOKENS,
        "workers": {
            worker: {
                "model": result["response_model"],
                "provider": result["provider"],
                "completion_tokens": int(result["completion_tokens"]),
                "client_attempts": int(result["client_attempts"]),
                "external_attempts": int(result["external_attempts"]),
            }
            for worker, result in zip(WORKERS, results, strict=True)
        },
        "provider_fallbacks": False,
        "reasoning_enabled": False,
        "performance_inference_admissible": False,
    }
    return validate_report(report)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    openrouter_key = os.environ.get("OPENROUTER_EPHEMERAL_API_KEY", "")
    if not openrouter_key:
        raise SystemExit("OPENROUTER_EPHEMERAL_API_KEY is required")
    report = run_preflight(
        openrouter_key=openrouter_key,
        run_id=args.run_id,
        timeout_seconds=args.timeout_seconds,
    )
    encoded = encode_report(report, openrouter_key)
    if args.output:
        with open(args.output, "x", encoding="utf-8") as output:
            output.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
