#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Bounded Semantic Router -> protected Modal session endpoint canary."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import re
import time
import urllib.request
from typing import Any
from urllib.parse import urlparse

SESSION_METRIC = "llm_rayline_arc_session_actions_total"
FAILURE_METRIC = "llm_rayline_arc_selection_failures_total"
SELECTED_MODELS = {"worker-a", "worker-b"}
HTTP_OK = 200
MIN_CREATED_ACTIONS = 1.0


def _chat(
    base_url: str,
    episode_id: str,
    messages: list[dict[str, str]],
    timeout_seconds: float,
) -> dict[str, Any]:
    parsed = urlparse(base_url)
    connection = http.client.HTTPConnection(
        parsed.hostname,
        parsed.port,
        timeout=timeout_seconds,
    )
    payload = json.dumps(
        {
            "model": "auto",
            "messages": messages,
            "max_tokens": 1,
            "stream": False,
        },
        separators=(",", ":"),
    ).encode()
    started = time.perf_counter()
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=payload,
        headers={
            "content-type": "application/json",
            "x-rayline-episode-id": episode_id,
            "authorization": "Bearer public-modal-gateway-canary",
        },
    )
    response = connection.getresponse()
    response_body = response.read()
    elapsed = time.perf_counter() - started
    selected_model = response.getheader("x-vsr-selected-model", "")
    connection.close()
    if response.status != HTTP_OK:
        raise RuntimeError(f"gateway returned HTTP {response.status}")
    if selected_model not in SELECTED_MODELS:
        raise RuntimeError("gateway omitted a valid selected-model header")
    decoded = json.loads(response_body)
    if not isinstance(decoded, dict) or not decoded.get("choices"):
        raise RuntimeError("gateway response omitted provider choices")
    return {
        "http_status": response.status,
        "selected_model": selected_model,
        "latency_seconds": elapsed,
    }


def _metric_values(metrics: str, name: str) -> dict[str, float]:
    values: dict[str, float] = {}
    pattern = re.compile(
        rf'^{re.escape(name)}\{{[^}}]*action="([^"]+)"[^}}]*\}}\s+([0-9.eE+-]+)$'
    )
    for line in metrics.splitlines():
        if match := pattern.match(line):
            values[match.group(1)] = float(match.group(2))
    return values


def _failure_total(metrics: str) -> float:
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in metrics.splitlines()
        if line.startswith(FAILURE_METRIC + "{")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--metrics-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()

    episode_id = (
        "modal-gateway-" + hashlib.sha256(args.run_id.encode()).hexdigest()[:24]
    )
    first_turn = [{"role": "user", "content": "Choose a concise public answer."}]
    extended_turns = [
        *first_turn,
        {"role": "assistant", "content": "A short public response."},
        {"role": "user", "content": "Now prefer the lower-latency option."},
    ]
    requests = [
        _chat(args.gateway_url, episode_id, first_turn, args.timeout_seconds),
        _chat(args.gateway_url, episode_id, extended_turns, args.timeout_seconds),
    ]

    with urllib.request.urlopen(
        args.metrics_url,
        timeout=args.timeout_seconds,
    ) as response:
        metrics = response.read().decode()
    session_actions = _metric_values(metrics, SESSION_METRIC)
    if session_actions.get("created", 0.0) < MIN_CREATED_ACTIONS:
        raise RuntimeError("gateway did not observe unique request session creation")
    if session_actions.get("appended", 0.0) < 1.0:
        raise RuntimeError("gateway did not observe retained-session append")
    if _failure_total(metrics) != 0.0:
        raise RuntimeError("gateway recorded a Rayline ARC selection failure")

    report = {
        "schema_version": "rayline.arc.modal-gateway-canary.v1",
        "run_id": args.run_id,
        "status": "passed",
        "requests": requests,
        "session_actions": {
            "created": session_actions["created"],
            "appended": session_actions["appended"],
        },
        "selection_failures": 0,
        "automatic_prefix_cache_enabled": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
