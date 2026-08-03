#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Prove the frozen OpenRouter gateway endpoints before paid encoder startup."""

from __future__ import annotations

import argparse
import os
from typing import Any

from openrouter_agentic_benchmark import (
    MAX_COMPLETION_TOKENS,
    WORKERS,
    _probe_endpoints,
    _probe_key_readiness,
)
from openrouter_agentic_preflight_contract import (
    MAX_EXTERNAL_ATTEMPTS,
    MAX_PROVIDER_REQUESTS,
    REPORT_SCHEMA,
    encode_report,
)


def run_preflight(
    *, gateway_url: str, openrouter_key: str, run_id: str, timeout_seconds: float
) -> dict[str, Any]:
    key_readiness = _probe_key_readiness(
        gateway_url=gateway_url,
        openrouter_key=openrouter_key,
        run_id=run_id,
        timeout_seconds=timeout_seconds,
    )
    endpoint_probes = _probe_endpoints(
        gateway_url=gateway_url,
        openrouter_key=openrouter_key,
        run_id=run_id,
        timeout_seconds=timeout_seconds,
    )
    results = [key_readiness, *endpoint_probes]
    attempts = sum(int(result["external_attempts"]) for result in results)
    if len(results) != MAX_PROVIDER_REQUESTS or attempts > MAX_EXTERNAL_ATTEMPTS:
        raise RuntimeError("agentic transport preflight exceeded its frozen bounds")
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "passed",
        "provider_requests": len(results),
        "maximum_provider_requests": MAX_PROVIDER_REQUESTS,
        "external_attempts": attempts,
        "maximum_external_attempts": MAX_EXTERNAL_ATTEMPTS,
        "retries": attempts - len(results),
        "cost_usd": sum(float(result["cost_usd"]) for result in results),
        "maximum_completion_tokens": MAX_COMPLETION_TOKENS,
        "key_readiness": {
            "model": key_readiness["response_model"],
            "provider": key_readiness["provider"],
            "completion_tokens": int(key_readiness["completion_tokens"]),
        },
        "workers": {
            worker: {
                "model": result["response_model"],
                "provider": result["provider"],
                "completion_tokens": int(result["completion_tokens"]),
            }
            for worker, result in zip(WORKERS, endpoint_probes, strict=True)
        },
        "provider_fallbacks": False,
        "reasoning_enabled": False,
        "performance_inference_admissible": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    openrouter_key = os.environ.get("OPENROUTER_EPHEMERAL_API_KEY", "")
    if not openrouter_key:
        raise SystemExit("OPENROUTER_EPHEMERAL_API_KEY is required")
    report = run_preflight(
        gateway_url=args.gateway_url,
        openrouter_key=openrouter_key,
        run_id=args.run_id,
        timeout_seconds=args.timeout_seconds,
    )
    print(encode_report(report, openrouter_key))


if __name__ == "__main__":
    main()
