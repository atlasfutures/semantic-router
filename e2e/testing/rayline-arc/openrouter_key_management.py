#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Bounded OpenRouter ephemeral-key lifecycle for live E2E packets."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

OPENROUTER_MANAGEMENT_URL = "https://openrouter.ai/api/v1/keys"
HTTP_OK = 200
HTTP_CREATED = 201


def _management_request(
    *,
    method: str,
    management_key: str,
    path: str = "",
    payload: dict[str, Any] | None = None,
    expected_status: int = HTTP_OK,
) -> dict[str, Any]:
    body = None
    headers = {"authorization": f"Bearer {management_key}"}
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        headers["content-type"] = "application/json"
    request = urllib.request.Request(
        f"{OPENROUTER_MANAGEMENT_URL}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            if response.status != expected_status:
                raise RuntimeError(
                    "OpenRouter management request returned wrong status"
                )
            response_body = response.read()
    except urllib.error.HTTPError as error:
        error.read()
        raise RuntimeError("OpenRouter management request failed") from error
    if not response_body:
        return {}
    decoded = json.loads(response_body)
    if not isinstance(decoded, dict):
        raise TypeError("OpenRouter management response was malformed")
    return decoded


def create_ephemeral_key(
    management_key: str, run_id: str, key_limit_usd: float
) -> tuple[str, str]:
    response = _management_request(
        method="POST",
        management_key=management_key,
        payload={
            "name": f"rayline-arc-{run_id}",
            "limit": key_limit_usd,
            "include_byok_in_limit": True,
        },
        expected_status=HTTP_CREATED,
    )
    key = response.get("key")
    data = response.get("data")
    key_hash = data.get("hash") if isinstance(data, dict) else None
    if (
        not isinstance(key, str)
        or not key
        or not isinstance(key_hash, str)
        or not key_hash
    ):
        raise RuntimeError("OpenRouter did not return an ephemeral key contract")
    return key, key_hash


def ephemeral_key_usage(management_key: str, key_hash: str) -> float:
    response = _management_request(
        method="GET", management_key=management_key, path=f"/{key_hash}"
    )
    data = response.get("data")
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, (int, float)):
        raise TypeError("OpenRouter key usage was unavailable")
    return float(usage)


def delete_ephemeral_key(management_key: str, key_hash: str) -> None:
    _management_request(
        method="DELETE",
        management_key=management_key,
        path=f"/{key_hash}",
        expected_status=HTTP_OK,
    )
