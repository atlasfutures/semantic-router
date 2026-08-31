# SPDX-License-Identifier: Apache-2.0

"""Post-baseline residency warmup for the protected Rayline ARC encoder."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from typing import Any

from modal_http import request_following_result_redirects

HTTP_OK = 200
REQUIRED_CAPABILITIES = {"chunked_causal_mean", "resumable_causal_mean"}


def warm_encoder_from_environment(
    *,
    timeout_seconds: float,
    connection_factory: Callable[[str, float], tuple[Any, str]],
) -> dict[str, Any]:
    """Resolve the launcher-owned protected endpoint and credentials."""
    return warm_encoder(
        base_url=os.environ.get("RAYLINE_ARC_E2E_ENCODER_BASE_URL", ""),
        modal_key=os.environ.get("RAYLINE_ARC_E2E_MODAL_KEY", ""),
        modal_secret=os.environ.get("RAYLINE_ARC_E2E_MODAL_SECRET", ""),
        timeout_seconds=timeout_seconds,
        connection_factory=connection_factory,
    )


def warm_encoder(
    *,
    base_url: str,
    modal_key: str,
    modal_secret: str,
    timeout_seconds: float,
    connection_factory: Callable[[str, float], tuple[Any, str]],
) -> dict[str, Any]:
    """Warm the exact protected encoder without allocating a session."""
    if not base_url or not modal_key or not modal_secret:
        raise ValueError("protected encoder warmup configuration is incomplete")
    started = time.perf_counter()
    connection, response = request_following_result_redirects(
        connection_factory=connection_factory,
        method="GET",
        url=f"{base_url.rstrip('/')}/health",
        body=None,
        headers={"Modal-Key": modal_key, "Modal-Secret": modal_secret},
        timeout_seconds=timeout_seconds,
    )
    body = response.read()
    elapsed = time.perf_counter() - started
    connection.close()
    if response.status != HTTP_OK:
        raise RuntimeError(f"encoder warmup returned HTTP {response.status}")
    try:
        decoded = json.loads(body)
    except (TypeError, ValueError) as error:
        raise RuntimeError("encoder warmup returned invalid JSON") from error
    if not isinstance(decoded, dict) or decoded.get("status") != "ok":
        raise RuntimeError("encoder warmup health contract failed")
    capabilities = decoded.get("pooling_capabilities")
    if not isinstance(capabilities, list) or not REQUIRED_CAPABILITIES.issubset(
        set(capabilities)
    ):
        raise RuntimeError("encoder warmup omitted required pooling capabilities")
    return {
        "latency_seconds": elapsed,
        "pooling_capabilities": sorted(REQUIRED_CAPABILITIES),
        "status": "ok",
    }
