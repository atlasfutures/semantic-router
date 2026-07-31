#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Bounded real-HTTP canary for the protected retained-session service.

The driver uses public synthetic turns and emits only aggregate timings,
token counts, actions, and embedding shape/norm. It never emits credentials,
turn text, or embedding values.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

REQUEST_SCHEMA = "rayline.arc.session-pooling-request.v1"
RESPONSE_SCHEMA = "rayline.arc.session-pooling-response.v1"
SERIALIZER_VERSION = "mtrouter-token-blocks-v2"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
ENGINE_BUILD_ID = "vllm@b1049f6dd95c27d2e1b052eebc3b1a7f9f41195f"
PLUGIN_VERSION = "rayline-arc-io@0.1.0"
CAPABILITIES = ["chunked_causal_mean", "resumable_causal_mean"]
EMBEDDING_DIMENSION = 1024
MIN_EMBEDDING_NORM = 0.9
MAX_EMBEDDING_NORM = 1.1
EXPECTED_RESIDENT_SESSIONS = 3


def _episode_hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@dataclass(frozen=True)
class CanaryClient:
    base_url: str
    modal_key: str
    modal_secret: str
    timeout_seconds: float

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], float]:
        body = None
        headers = {
            "Accept": "application/json",
            "Modal-Key": self.modal_key,
            "Modal-Secret": self.modal_secret,
        }
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url.rstrip("/") + path,
            data=body,
            headers=headers,
            method=method,
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                response_body = response.read()
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"retained-session HTTP status {error.code}") from error
        elapsed = time.perf_counter() - started
        decoded = json.loads(response_body)
        if not isinstance(decoded, dict):
            raise RuntimeError("retained-session response must be an object")
        return decoded, elapsed

    def encode(
        self,
        episode_id: str,
        turns: list[dict[str, str]],
    ) -> dict[str, Any]:
        response, elapsed = self.request(
            "POST",
            "/v1/rayline/arc/session/pooling",
            {
                "schema_version": REQUEST_SCHEMA,
                "serializer_version": SERIALIZER_VERSION,
                "serving_rung": "B",
                "episode_id_hash": episode_id,
                "turns": turns,
            },
        )
        return _validate_response(response, elapsed)

    def close(self, episode_id: str) -> None:
        response, _elapsed = self.request(
            "DELETE",
            f"/v1/rayline/arc/session/{episode_id}",
        )
        if response != {"closed": True}:
            raise RuntimeError("retained session did not close cleanly")


def _validate_response(response: dict[str, Any], elapsed: float) -> dict[str, Any]:
    expected = {
        "schema_version": RESPONSE_SCHEMA,
        "serializer_version": SERIALIZER_VERSION,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "engine_build_id": ENGINE_BUILD_ID,
        "io_plugin_version": PLUGIN_VERSION,
        "pooling_capabilities": CAPABILITIES,
    }
    for field, wanted in expected.items():
        if response.get(field) != wanted:
            raise RuntimeError(f"retained-session metadata mismatch: {field}")

    embedding = response.pop("embedding", None)
    if not isinstance(embedding, list) or len(embedding) != EMBEDDING_DIMENSION:
        raise RuntimeError("retained-session embedding shape mismatch")
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in embedding
    ):
        raise RuntimeError("retained-session embedding contains a non-finite value")
    norm = math.sqrt(math.fsum(float(value) ** 2 for value in embedding))
    if not MIN_EMBEDDING_NORM <= norm <= MAX_EMBEDDING_NORM:
        raise RuntimeError("retained-session embedding is not normalized")

    serialized = response.get("serialized_tokens")
    retained = response.get("retained_prefix_tokens")
    appended = response.get("appended_tokens")
    if not all(isinstance(value, int) for value in (serialized, retained, appended)):
        raise RuntimeError("retained-session token counts must be integers")
    if retained + appended != serialized:
        raise RuntimeError("retained-session token accounting mismatch")

    return {
        "action": response["session_action"],
        "revision": response["session_revision"],
        "serialized_tokens": serialized,
        "retained_prefix_tokens": retained,
        "appended_tokens": appended,
        "embedding_dimension": len(embedding),
        "embedding_norm": norm,
        "latency_seconds": elapsed,
    }


def _expect(
    result: dict[str, Any],
    *,
    action: str,
    revision: int,
) -> None:
    if result["action"] != action or result["revision"] != revision:
        raise RuntimeError(
            f"expected session action/revision {action}/{revision}, got "
            f"{result['action']}/{result['revision']}"
        )


def run_canary(client: CanaryClient, run_id: str) -> dict[str, Any]:
    primary_id = _episode_hash(f"{run_id}:primary")
    concurrent_ids = [
        _episode_hash(f"{run_id}:concurrent:{index}") for index in range(2)
    ]
    first_turn = [{"role": "user", "text": "Choose a concise public answer."}]
    extended_turns = [
        *first_turn,
        {"role": "assistant", "text": "A short public response."},
        {"role": "user", "text": "Now prefer the lower-latency option."},
    ]
    replacement_turn = [
        {"role": "user", "text": "Rebuild this public synthetic episode."}
    ]

    created = client.encode(primary_id, first_turn)
    appended = client.encode(primary_id, extended_turns)
    reused = client.encode(primary_id, extended_turns)
    rebuilt = client.encode(primary_id, replacement_turn)
    _expect(created, action="created", revision=1)
    _expect(appended, action="appended", revision=2)
    _expect(reused, action="reused", revision=2)
    _expect(rebuilt, action="rebuilt", revision=3)
    if appended["retained_prefix_tokens"] != created["serialized_tokens"]:
        raise RuntimeError("append did not retain the exact serialized prefix")
    if reused["retained_prefix_tokens"] != appended["serialized_tokens"]:
        raise RuntimeError("retry did not retain the complete serialized history")

    concurrent_started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(client.encode, episode_id, first_turn)
            for episode_id in concurrent_ids
        ]
        concurrent_results = [future.result() for future in futures]
    concurrent_elapsed = time.perf_counter() - concurrent_started
    for result in concurrent_results:
        _expect(result, action="created", revision=1)

    health, health_latency = client.request("GET", "/health")
    if (
        health.get("status") != "ok"
        or health.get("resident_sessions") != EXPECTED_RESIDENT_SESSIONS
    ):
        raise RuntimeError("retained-session health counts diverged")

    for episode_id in [primary_id, *concurrent_ids]:
        client.close(episode_id)
    final_health, final_health_latency = client.request("GET", "/health")
    if final_health.get("resident_sessions") != 0:
        raise RuntimeError("retained sessions leaked after explicit close")

    return {
        "schema_version": "rayline.arc.modal-session-http-canary.v1",
        "run_id": run_id,
        "status": "passed",
        "state_transitions": [created, appended, reused, rebuilt],
        "concurrency": {
            "episodes": len(concurrent_results),
            "wall_seconds": concurrent_elapsed,
            "request_latency_seconds": [
                result["latency_seconds"] for result in concurrent_results
            ],
        },
        "resident_sessions_before_close": health["resident_sessions"],
        "resident_tokens_before_close": health["resident_tokens"],
        "resident_sessions_after_close": final_health["resident_sessions"],
        "health_latency_seconds": health_latency,
        "final_health_latency_seconds": final_health_latency,
        "automatic_prefix_cache_enabled": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()

    modal_key = os.environ.get("RAYLINE_ARC_MODAL_KEY", "")
    modal_secret = os.environ.get("RAYLINE_ARC_MODAL_SECRET", "")
    if not modal_key or not modal_secret:
        raise SystemExit(
            "RAYLINE_ARC_MODAL_KEY and RAYLINE_ARC_MODAL_SECRET are required"
        )

    report = run_canary(
        CanaryClient(
            base_url=args.base_url,
            modal_key=modal_key,
            modal_secret=modal_secret,
            timeout_seconds=args.timeout_seconds,
        ),
        args.run_id,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
