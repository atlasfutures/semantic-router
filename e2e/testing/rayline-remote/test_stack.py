#!/usr/bin/env python3
"""Host-side acceptance for authoritative remote Rayline routing."""

from __future__ import annotations

import concurrent.futures
import hashlib
import hmac
import http.client
import json
import os
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from typing import Any

ENVOY_PORT = int(os.getenv("RAYLINE_REMOTE_E2E_ENVOY_PORT", "18889"))
POLICY_PORT = int(os.getenv("RAYLINE_REMOTE_E2E_POLICY_PORT", "18090"))
PROVIDER_A_PORT = int(os.getenv("RAYLINE_REMOTE_E2E_PROVIDER_A_PORT", "18091"))
PROVIDER_B_PORT = int(os.getenv("RAYLINE_REMOTE_E2E_PROVIDER_B_PORT", "18092"))
METRICS_PORT = int(os.getenv("RAYLINE_REMOTE_E2E_METRICS_PORT", "19191"))

PROMPT_CANARY = "rayline-remote-private-prompt-canary"
TOOL_CANARY = "rayline-remote-private-tool-canary"
EPISODE_CANARY = "rayline-remote-private-episode-canary"
CLIENT_KEY_CANARY = "rayline-remote-private-client-key-canary"
RECEIPT_CANARY = "rayline-remote-private-receipt-canary"
HMAC_KEY = b"public-e2e-hmac-key-32-bytes-long"
ROUTER_KEY = "public-e2e-rayline-key"
SCHEMA = "rayline-router.selection-transaction.v1"
BUNDLE = "rayline-remote-e2e-bundle-v1"
INPUT_TOKENS = 11
OUTPUT_TOKENS = 7
DIRECT_INPUT_TOKENS = 3
EXPECTED_TURN_COUNT = 2
COST_TOLERANCE = 1e-12


def url(port: int, path: str) -> str:
    return f"http://127.0.0.1:{port}{path}"


def json_request(
    port: int,
    path: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 10,
) -> tuple[int, Any]:
    payload = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url(port, path),
        data=payload,
        method=method,
        headers={
            "content-type": "application/json",
            **(headers or {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return response.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as error:
        raw = error.read()
        return error.code, json.loads(raw) if raw else {}


def chat(
    episode: str,
    marker: str,
    *,
    stream: bool = False,
    timeout: float = 10,
) -> tuple[int, bytes, dict[str, str]]:
    body = json.dumps(
        {
            "model": "auto",
            "messages": [
                {
                    "role": "user",
                    "content": f"{PROMPT_CANARY} {marker}",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "private_tool",
                        "description": TOOL_CANARY,
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "stream": stream,
        }
    ).encode()
    connection = http.client.HTTPConnection(
        "127.0.0.1",
        ENVOY_PORT,
        timeout=timeout,
    )
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=body,
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {CLIENT_KEY_CANARY}",
            "x-rayline-episode-id": episode,
            # An untrusted client value must not become the transaction key.
            "x-rayline-route-id": RECEIPT_CANARY,
        },
    )
    response = connection.getresponse()
    raw = response.read()
    headers = {key.lower(): value for key, value in response.getheaders()}
    status = response.status
    connection.close()
    return status, raw, headers


def opaque_episode(raw_episode: str) -> str:
    return (
        "hmac-sha256:"
        + hmac.new(
            HMAC_KEY,
            raw_episode.encode(),
            hashlib.sha256,
        ).hexdigest()
    )


def observed(port: int) -> dict[str, Any]:
    status, body = json_request(port, "/observed")
    assert status == HTTPStatus.OK
    assert isinstance(body, dict)
    return body


def reset() -> None:
    for port in (POLICY_PORT, PROVIDER_A_PORT, PROVIDER_B_PORT):
        status, _ = json_request(port, "/reset", method="POST", body={})
        assert status == HTTPStatus.OK


def set_mode(mode: str) -> None:
    status, _ = json_request(
        POLICY_PORT,
        "/mode",
        method="POST",
        body={"mode": mode},
    )
    assert status == HTTPStatus.OK


def wait_for_transaction(
    raw_episode: str,
    state: str,
    *,
    timeout: float = 3,
) -> dict[str, Any]:
    episode_key = opaque_episode(raw_episode)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = observed(POLICY_PORT)
        prepare = next(
            (
                item
                for item in reversed(snapshot["prepares"])
                if item["episode_key"] == episode_key
            ),
            None,
        )
        if prepare is not None:
            transaction = snapshot["transactions"][prepare["decision_id"]]
            if transaction["state"] == state:
                return transaction
        time.sleep(0.05)
    raise AssertionError(f"transaction for {raw_episode!r} did not reach {state!r}")


def provider_counts() -> tuple[int, int]:
    return (
        int(observed(PROVIDER_A_PORT)["count"]),
        int(observed(PROVIDER_B_PORT)["count"]),
    )


def assert_happy_path_and_candidate_mask() -> None:
    reset()
    episode = EPISODE_CANARY + "-happy"
    status, raw, headers = chat(episode, "HAPPY")
    assert status == HTTPStatus.OK, (status, raw, headers)
    body = json.loads(raw)
    assert body["choices"][0]["message"]["content"] == "provider-b", (
        body,
        headers,
        provider_counts(),
    )
    assert headers["x-vsr-selected-model"] == "worker-b"
    assert provider_counts() == (0, 1)
    transaction = wait_for_transaction(episode, "settled")
    assert transaction["status_code"] == HTTPStatus.OK
    assert transaction["outcome"]["input_tokens"] == INPUT_TOKENS
    assert transaction["outcome"]["output_tokens"] == OUTPUT_TOKENS
    assert abs(transaction["outcome"]["cost_usd"] - 0.00005) < COST_TOLERANCE

    snapshot = observed(POLICY_PORT)
    prepare = snapshot["prepares"][-1]
    assert prepare["candidates"] == [
        "remote-worker-a",
        "remote-worker-b",
    ]
    assert prepare["selected_worker"] == "remote-worker-b"
    assert "outside-worker" not in prepare["candidates"]
    episode_state = snapshot["episodes"][opaque_episode(episode)]
    assert episode_state["route_call_index"] == 1
    assert episode_state["previous_worker"] == "remote-worker-b"

    status, _, _ = chat(episode, "NEXT")
    assert status == HTTPStatus.OK
    wait_for_transaction(episode, "settled")
    snapshot = observed(POLICY_PORT)
    assert snapshot["prepares"][-1]["route_call_index"] == 1
    assert snapshot["prepares"][-1]["previous_worker"] == ("remote-worker-b")
    assert (
        snapshot["episodes"][opaque_episode(episode)]["route_call_index"]
        == EXPECTED_TURN_COUNT
    )


def assert_pre_dispatch_failures() -> None:
    for mode in (
        "timeout_prepare",
        "wrong_worker",
        "wrong_bundle",
        "wrong_decision",
        "malformed_receipt",
        "renew_fail",
    ):
        reset()
        set_mode(mode)
        before = provider_counts()
        episode = f"{EPISODE_CANARY}-{mode}"
        status, raw, _ = chat(episode, mode, timeout=6)
        assert status == HTTPStatus.SERVICE_UNAVAILABLE, (
            mode,
            status,
            raw,
        )
        assert provider_counts() == before
        snapshot = observed(POLICY_PORT)
        episode_state = snapshot["episodes"].get(
            opaque_episode(episode),
            {},
        )
        assert episode_state.get("route_call_index", 0) == 0


def assert_provider_failure_aborts_and_retries_same_turn() -> None:
    reset()
    episode = EPISODE_CANARY + "-provider-failure"
    status, _, _ = chat(episode, "REMOTE_PROVIDER_5XX")
    assert status in (502, 503, 504)
    transaction = wait_for_transaction(episode, "aborted")
    assert transaction["route_call_index"] == 0
    snapshot = observed(POLICY_PORT)
    assert snapshot["episodes"][opaque_episode(episode)]["route_call_index"] == 0
    set_mode("normal")
    status, _, _ = chat(episode, "RETRY")
    assert status == HTTPStatus.OK
    wait_for_transaction(episode, "settled")
    snapshot = observed(POLICY_PORT)
    assert snapshot["prepares"][-1]["route_call_index"] == 0
    assert snapshot["episodes"][opaque_episode(episode)]["route_call_index"] == 1


def assert_streaming_commit_and_broken_body() -> None:
    reset()
    success_episode = EPISODE_CANARY + "-stream-success"
    status, raw, _ = chat(
        success_episode,
        "REMOTE_STREAM",
        stream=True,
    )
    assert status == HTTPStatus.OK and b"data: [DONE]" in raw
    wait_for_transaction(success_episode, "settled")

    broken_episode = EPISODE_CANARY + "-stream-broken"
    status, raw, _ = chat(
        broken_episode,
        "REMOTE_STREAM_BREAK",
        stream=True,
    )
    assert status in (HTTPStatus.OK, HTTPStatus.BAD_GATEWAY), (
        status,
        raw,
    )
    if status == HTTPStatus.OK:
        assert b"provider-b" in raw
    transaction = wait_for_transaction(broken_episode, "settled")
    assert transaction["outcome"]["outcome_class"] in (
        "stream_error",
        "unknown",
    )
    snapshot = observed(POLICY_PORT)
    assert snapshot["episodes"][opaque_episode(broken_episode)]["route_call_index"] == 1


def assert_unknown_usage_stays_unknown() -> None:
    reset()
    episode = EPISODE_CANARY + "-no-usage"
    status, _, _ = chat(episode, "REMOTE_NO_USAGE")
    assert status == HTTPStatus.OK
    transaction = wait_for_transaction(episode, "settled")
    outcome = transaction["outcome"]
    assert "input_tokens" not in outcome
    assert "output_tokens" not in outcome
    assert "cost_usd" not in outcome


def direct_transaction_payload(decision_id: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "decision_id": decision_id,
        "episode_key": opaque_episode(EPISODE_CANARY + "-direct-idempotency"),
        "bundle_version": BUNDLE,
        "candidates": ["remote-worker-a", "remote-worker-b"],
        "request": {
            "protocol": "openai.chat.completions",
            "messages": [{"role": "user", "content": "bounded"}],
        },
    }


def assert_protocol_idempotency() -> None:
    reset()
    auth = {"authorization": f"Bearer {ROUTER_KEY}"}
    payload = direct_transaction_payload("rt_direct-idempotency")
    first_status, first = json_request(
        POLICY_PORT,
        "/v1/route/prepare",
        method="POST",
        body=payload,
        headers=auth,
    )
    second_status, second = json_request(
        POLICY_PORT,
        "/v1/route/prepare",
        method="POST",
        body=payload,
        headers=auth,
    )
    assert first_status == second_status == HTTPStatus.OK
    assert first == second
    lifecycle = {
        "schema_version": SCHEMA,
        "decision_id": payload["decision_id"],
        "receipt": first["receipt"],
    }
    for _ in range(2):
        status, body = json_request(
            POLICY_PORT,
            "/v1/route/commit",
            method="POST",
            body={**lifecycle, "status_code": 200},
            headers=auth,
        )
        assert status == HTTPStatus.OK and body["state"] == "committed"
    outcome = {
        "outcome_class": "success",
        "status_code": 200,
        "input_tokens": 3,
    }
    for _ in range(2):
        status, body = json_request(
            POLICY_PORT,
            "/v1/route/settle",
            method="POST",
            body={**lifecycle, "outcome": outcome},
            headers=auth,
        )
        assert status == HTTPStatus.OK and body["state"] == "settled"
    snapshot = observed(POLICY_PORT)
    episode = snapshot["episodes"][payload["episode_key"]]
    assert episode["route_call_index"] == 1
    assert episode["previous_input_tokens"] == DIRECT_INPUT_TOKENS


def assert_concurrency_boundary() -> None:
    reset()
    same_episode = EPISODE_CANARY + "-same-concurrency"
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda suffix: chat(
                    same_episode,
                    f"REMOTE_HOLD {suffix}",
                ),
                (1, 2),
            )
        )
    statuses = sorted(result[0] for result in results)
    assert statuses in ([200, 503], [200, 200]), statuses
    snapshot = observed(POLICY_PORT)
    state = snapshot["episodes"][opaque_episode(same_episode)]
    assert state["route_call_index"] == statuses.count(200)

    reset()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda suffix: chat(
                    f"{EPISODE_CANARY}-parallel-{suffix}",
                    f"REMOTE_HOLD {suffix}",
                ),
                (1, 2),
            )
        )
    assert [result[0] for result in results] == [200, 200]


def assert_privacy_surfaces() -> None:
    snapshot = json.dumps(observed(POLICY_PORT), sort_keys=True)
    for canary in (
        PROMPT_CANARY,
        TOOL_CANARY,
        EPISODE_CANARY,
        CLIENT_KEY_CANARY,
        RECEIPT_CANARY,
    ):
        assert canary not in snapshot
    with urllib.request.urlopen(
        url(METRICS_PORT, "/metrics"),
        timeout=5,
    ) as response:
        metrics = response.read().decode(errors="replace")
    for canary in (
        PROMPT_CANARY,
        TOOL_CANARY,
        EPISODE_CANARY,
        CLIENT_KEY_CANARY,
        RECEIPT_CANARY,
        ROUTER_KEY,
    ):
        assert canary not in metrics


def main() -> None:
    assert_happy_path_and_candidate_mask()
    assert_pre_dispatch_failures()
    assert_provider_failure_aborts_and_retries_same_turn()
    assert_streaming_commit_and_broken_body()
    assert_unknown_usage_stays_unknown()
    assert_protocol_idempotency()
    assert_concurrency_boundary()
    assert_privacy_surfaces()
    print("Rayline remote hermetic full-stack acceptance: PASS")


if __name__ == "__main__":
    main()
