#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Bounded direct/static OpenRouter request-shape diagnostic."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from modal_fullstack_canary import _episode_id
from openrouter_agentic_benchmark import (
    OpenRouterHTTPError,
    _candidate_case,
    _stream_request,
)

OPENROUTER_KEY_ENV = "OPENROUTER_EPHEMERAL_API_KEY"
PROBES = (
    ("direct_1", "direct", 1),
    ("static_1", "gateway_static", 1),
    ("direct_96_a", "direct", 96),
    ("static_96_a", "gateway_static", 96),
    ("direct_96_b", "direct", 96),
    ("static_96_b", "gateway_static", 96),
)
MAX_PROVIDER_REQUESTS = len(PROBES)
MAX_EXTERNAL_ATTEMPTS = len(PROBES) * 2


def _result_summary(
    *, label: str, path: str, max_tokens: int, result: dict[str, Any]
) -> dict[str, Any]:
    return {
        "label": label,
        "path": path,
        "max_tokens": max_tokens,
        "status": "complete",
        "http_status": 200,
        "response_model": result["response_model"],
        "provider": result["provider"],
        "completion_tokens": result["completion_tokens"],
        "external_attempts": result["external_attempts"],
    }


def _error_summary(
    *, label: str, path: str, max_tokens: int, error: OpenRouterHTTPError
) -> dict[str, Any]:
    return {
        "label": label,
        "path": path,
        "max_tokens": max_tokens,
        "status": "http_error",
        "http_status": error.status_code,
        "error_category": error.error_category,
        "error_type": error.error_type,
        "provider_code": error.provider_code,
        "external_attempts": error.external_attempts,
    }


def run_diagnostic(
    *, gateway_url: str, openrouter_key: str, run_id: str, timeout_seconds: float
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    case = _candidate_case(0)
    for label, path, max_tokens in PROBES:
        try:
            result = _stream_request(
                path=path,
                case=case,
                expected_worker="worker-a",
                gateway_url=gateway_url,
                openrouter_key=openrouter_key,
                episode_id=_episode_id(run_id, label),
                timeout_seconds=timeout_seconds,
                max_completion_tokens=max_tokens,
                maximum_attempts=1,
                retryable_status_codes=frozenset(),
            )
        except OpenRouterHTTPError as error:
            results.append(
                _error_summary(
                    label=label,
                    path=path,
                    max_tokens=max_tokens,
                    error=error,
                )
            )
        else:
            results.append(
                _result_summary(
                    label=label,
                    path=path,
                    max_tokens=max_tokens,
                    result=result,
                )
            )
    attempts = sum(int(result["external_attempts"]) for result in results)
    if attempts > MAX_EXTERNAL_ATTEMPTS:
        raise RuntimeError("gateway-shape diagnostic exceeded its attempt bound")
    return {
        "schema_version": "rayline.arc.openrouter-gateway-shape.v1",
        "aggregate_only": True,
        "run_id": run_id,
        "model": "deepseek/deepseek-v4-flash",
        "provider": "Baidu",
        "provider_fallbacks": False,
        "results": results,
        "provider_requests": len(results),
        "external_attempts": attempts,
        "successful_requests": sum(
            result["status"] == "complete" for result in results
        ),
        "failed_requests": sum(result["status"] != "complete" for result in results),
        "performance_inference_admissible": False,
        "release_qualification_1000_executed": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--metrics-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    openrouter_key = os.environ.get(OPENROUTER_KEY_ENV, "")
    if not openrouter_key:
        raise SystemExit(f"{OPENROUTER_KEY_ENV} is required")
    report = run_diagnostic(
        gateway_url=args.gateway_url,
        openrouter_key=openrouter_key,
        run_id=args.run_id,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
