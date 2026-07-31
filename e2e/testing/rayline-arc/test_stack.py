#!/usr/bin/env python3
"""Host-side acceptance checks for the hermetic Rayline ARC stack."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ENVOY_PORT = int(os.getenv("RAYLINE_ARC_E2E_ENVOY_PORT", "18888"))
ENCODER_PORT = int(os.getenv("RAYLINE_ARC_E2E_ENCODER_PORT", "18080"))
PROVIDER_PORT = int(os.getenv("RAYLINE_ARC_E2E_PROVIDER_PORT", "18081"))
REDIS_PORT = int(os.getenv("RAYLINE_ARC_E2E_REDIS_PORT", "16379"))
REDIS_PASSWORD = os.getenv(
    "RAYLINE_ARC_E2E_REDIS_PASSWORD",
    "public-e2e-redis-secret",
)

PROMPT_CANARY = "rayline-arc-private-prompt-canary"
TOOL_CANARY = "rayline-arc-private-tool-canary"
EPISODE_CANARY = "rayline-arc-private-episode-canary"
KEY_CANARY = "rayline-arc-private-key-canary"
STATE_KEY_PREFIX = "vsr:rayline-arc:"
HTTP_OK = 200
HTTP_BAD_GATEWAY = 502
HTTP_UNAVAILABLE = 503
REASONING_BUDGET_TOKENS = 64
CONCURRENT_REQUESTS = 2


def _url(port: int, path: str) -> str:
    return f"http://127.0.0.1:{port}{path}"


def _json_request(
    port: int,
    path: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    timeout: float = 10,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    payload = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        _url(port, path),
        data=payload,
        method=method,
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return (
                response.status,
                json.loads(raw) if raw else {},
                {key.lower(): value for key, value in response.headers.items()},
            )
    except urllib.error.HTTPError as error:
        raw = error.read()
        return (
            error.code,
            json.loads(raw) if raw else {},
            {key.lower(): value for key, value in error.headers.items()},
        )


def _chat(
    episode: str,
    marker: str,
    *,
    messages: list[dict[str, str]] | None = None,
    stream: bool = False,
    timeout: float = 15,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    body: dict[str, Any] = {
        "model": "auto",
        "messages": messages
        or [
            {
                "role": "user",
                "content": f"{PROMPT_CANARY} {marker}",
            }
        ],
        "max_tokens": 1,
        "max_completion_tokens": 999,
        "temperature": 1.9,
        "provider": {
            "order": ["client-provider"],
            "allow_fallbacks": True,
        },
        "reasoning": {"enabled": False, "max_tokens": 999},
        "reasoning_effort": "low",
        "context_management": {"edits": ["client-owned"]},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "private_tool",
                    "description": TOOL_CANARY,
                    "parameters": {"type": "object"},
                },
                "defer_loading": True,
                "allowed_callers": [TOOL_CANARY],
                "eager_input_streaming": True,
            }
        ],
        "stream": stream,
    }
    payload = json.dumps(body).encode()
    connection = http.client.HTTPConnection(
        "127.0.0.1",
        ENVOY_PORT,
        timeout=timeout,
    )
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=payload,
        headers={
            "content-type": "application/json",
            "x-rayline-episode-id": episode,
            "authorization": f"Bearer {KEY_CANARY}",
        },
    )
    response = connection.getresponse()
    headers = {key.lower(): value for key, value in response.getheaders()}
    raw = response.read()
    connection.close()
    parsed: dict[str, Any] = {}
    if raw and not stream:
        try:
            decoded = json.loads(raw)
            if isinstance(decoded, dict):
                parsed = decoded
        except json.JSONDecodeError:
            parsed = {"non_json_error_bytes": len(raw)}
    return response.status, parsed, headers


def _cancel_chat(episode: str) -> None:
    body = json.dumps(
        {
            "model": "auto",
            "messages": [
                {
                    "role": "user",
                    "content": f"{PROMPT_CANARY} ARC_PROVIDER_DELAY",
                }
            ],
        }
    ).encode()
    connection = http.client.HTTPConnection(
        "127.0.0.1",
        ENVOY_PORT,
        timeout=5,
    )
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=body,
        headers={
            "content-type": "application/json",
            "x-rayline-episode-id": episode,
        },
    )
    connection.close()


def _redis_command(*parts: str) -> bytes | None:
    with socket.create_connection(("127.0.0.1", REDIS_PORT), timeout=3) as sock:
        file = sock.makefile("rwb", buffering=0)
        for command in (("AUTH", REDIS_PASSWORD), parts):
            encoded = [
                f"*{len(command)}\r\n".encode(),
                *[
                    f"${len(value.encode())}\r\n".encode() + value.encode() + b"\r\n"
                    for value in command
                ],
            ]
            file.write(b"".join(encoded))
            response = _read_redis(file)
            if command[0] == "AUTH" and response != b"OK":
                raise AssertionError("Redis authentication failed")
        return response


def _read_redis(file: Any) -> bytes | None:
    line = file.readline()
    if not line:
        raise AssertionError("Redis closed the connection")
    kind, payload = line[:1], line[1:-2]
    if kind == b"+":
        return payload
    if kind == b"$":
        length = int(payload)
        if length == -1:
            return None
        value = file.read(length)
        if file.read(2) != b"\r\n":
            raise AssertionError("invalid Redis bulk response")
        return value
    if kind == b"-":
        raise AssertionError("Redis command failed")
    if kind == b":":
        return payload
    raise AssertionError("unsupported Redis response")


def _state_key(episode: str) -> str:
    digest = hashlib.sha256(episode.encode()).hexdigest()
    return f"{STATE_KEY_PREFIX}{digest}:state"


def _state(episode: str) -> dict[str, Any] | None:
    raw = _redis_command("GET", _state_key(episode))
    return None if raw is None else json.loads(raw)


def _wait_for_state(
    episode: str,
    expected_version: int | None,
    *,
    timeout: float = 4,
) -> dict[str, Any] | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = _state(episode)
        if expected_version is None:
            if value is None:
                return None
        elif value is not None and value["version"] == expected_version:
            return value
        time.sleep(0.05)
    return _state(episode)


def _provider_requests() -> list[dict[str, Any]]:
    status, body, _ = _json_request(PROVIDER_PORT, "/observed")
    assert status == HTTP_OK
    return body["requests"]


def _reset_service(port: int) -> None:
    status, _, _ = _json_request(port, "/reset", method="POST", body={})
    assert status == HTTP_OK


def _assert_dispatch(
    body: dict[str, Any],
    *,
    model: str,
    provider: str,
    reasoning: bool,
    max_tokens: int,
    temperature: float,
) -> None:
    assert body["model"] == model, body
    assert body["provider"] == {
        "order": [provider],
        "allow_fallbacks": False,
        "require_parameters": True,
    }, body
    assert body["reasoning"]["enabled"] is reasoning, body
    if reasoning:
        assert body["reasoning"]["max_tokens"] == REASONING_BUDGET_TOKENS, body
    else:
        assert body["reasoning"].get("effort") == "none", body
    assert body["max_tokens"] == max_tokens, body
    assert body["temperature"] == temperature, body
    for removed in (
        "max_completion_tokens",
        "reasoning_effort",
        "context_management",
    ):
        assert removed not in body, body
    tool = body["tools"][0]
    assert "defer_loading" not in tool
    assert "allowed_callers" not in tool
    assert "eager_input_streaming" not in tool
    assert TOOL_CANARY in json.dumps(tool)


def _assert_route(
    episode: str,
    marker: str,
    *,
    worker: str,
    reasoning_header: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    before_count = len(_provider_requests())
    status, response_body, headers = _chat(episode, marker)
    assert status == HTTP_OK, (status, response_body, headers)
    assert headers["x-vsr-selected-model"] == worker, headers
    if "x-vsr-selected-reasoning" in headers:
        assert headers["x-vsr-selected-reasoning"] == reasoning_header, headers
    requests = _provider_requests()
    assert len(requests) == before_count + 1
    state = _state(episode)
    assert state is not None
    return requests[-1], state


def _assert_failure_transactions() -> None:
    provider_episode = f"{EPISODE_CANARY}-provider-5xx"
    before_count = len(_provider_requests())
    status, _, _ = _chat(provider_episode, "ARC_PROVIDER_5XX")
    assert status == HTTP_UNAVAILABLE
    assert len(_provider_requests()) == before_count + 1
    assert _wait_for_state(provider_episode, None) is None

    transport_episode = f"{EPISODE_CANARY}-provider-transport"
    before_count = len(_provider_requests())
    status, _, _ = _chat(transport_episode, "ARC_PROVIDER_TRANSPORT")
    assert status in (HTTP_BAD_GATEWAY, HTTP_UNAVAILABLE)
    assert len(_provider_requests()) == before_count + 1
    assert _wait_for_state(transport_episode, None) is None

    encoder_episode = f"{EPISODE_CANARY}-encoder-5xx"
    before_count = len(_provider_requests())
    status, _, _ = _chat(encoder_episode, "ARC_ENCODER_FAIL")
    assert status == HTTP_UNAVAILABLE
    assert len(_provider_requests()) == before_count
    assert _wait_for_state(encoder_episode, None) is None


def _assert_concurrency() -> None:
    _reset_service(ENCODER_PORT)
    same_episode = f"{EPISODE_CANARY}-same-concurrency"
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda suffix: _chat(
                    same_episode,
                    f"ARC_DELAY same-{suffix}",
                    timeout=20,
                ),
                (1, 2),
            )
        )
    assert all(result[0] == HTTP_OK for result in results), results
    state = _state(same_episode)
    assert state is not None and state["version"] == CONCURRENT_REQUESTS
    status, stats, _ = _json_request(ENCODER_PORT, "/stats")
    assert status == HTTP_OK and stats["max_same_episode"] == 1, stats

    _reset_service(ENCODER_PORT)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda suffix: _chat(
                    f"{EPISODE_CANARY}-parallel-{suffix}",
                    f"ARC_DELAY parallel-{suffix}",
                    timeout=20,
                ),
                (1, 2),
            )
        )
    assert all(result[0] == HTTP_OK for result in results), results
    status, stats, _ = _json_request(ENCODER_PORT, "/stats")
    assert status == HTTP_OK and stats["max_global"] >= CONCURRENT_REQUESTS, stats


def _assert_retained_session_extension() -> None:
    _reset_service(ENCODER_PORT)
    episode = f"{EPISODE_CANARY}-retained-extension"
    first_message = {
        "role": "user",
        "content": f"{PROMPT_CANARY} ARC_ROUTE_A retained-first",
    }
    status, _, _ = _chat(
        episode,
        "ARC_ROUTE_A retained-first",
        messages=[first_message],
    )
    assert status == HTTP_OK
    status, _, _ = _chat(
        episode,
        "ARC_ROUTE_A retained-second",
        messages=[
            first_message,
            {"role": "assistant", "content": "public synthetic prior answer"},
            {
                "role": "user",
                "content": f"{PROMPT_CANARY} ARC_ROUTE_A retained-second",
            },
        ],
    )
    assert status == HTTP_OK
    status, stats, _ = _json_request(ENCODER_PORT, "/stats")
    assert status == HTTP_OK
    assert stats["session_actions"] == {"created": 1, "appended": 1}, stats


def _assert_response_boundaries() -> None:
    stream_episode = f"{EPISODE_CANARY}-stream-abort"
    try:
        status, _, _ = _chat(
            stream_episode,
            "ARC_STREAM_ABORT",
            stream=True,
        )
        # Envoy may surface the incomplete buffered body as 503 even though
        # the upstream 2xx headers already crossed the ARC commit boundary.
        assert status in (HTTP_OK, HTTP_BAD_GATEWAY, HTTP_UNAVAILABLE), status
    except http.client.IncompleteRead:
        pass
    state = _wait_for_state(stream_episode, 1)
    assert state is not None and state["turn_index"] == 1, state

    cancel_episode = f"{EPISODE_CANARY}-client-cancel"
    _cancel_chat(cancel_episode)
    time.sleep(1.5)
    assert _state(cancel_episode) is None


def _initial(receipt: Path) -> None:
    _reset_service(PROVIDER_PORT)
    body_a, state_a = _assert_route(
        f"{EPISODE_CANARY}-route-a",
        "ARC_ROUTE_A",
        worker="worker-a",
        reasoning_header="off",
    )
    _assert_dispatch(
        body_a,
        model="synthetic/provider-a",
        provider="synthetic-a",
        reasoning=False,
        max_tokens=128,
        temperature=0.2,
    )
    assert state_a["version"] == 1 and state_a["turn_index"] == 1, state_a

    body_b, state_b = _assert_route(
        f"{EPISODE_CANARY}-route-b",
        "ARC_ROUTE_B",
        worker="worker-b",
        reasoning_header="on",
    )
    _assert_dispatch(
        body_b,
        model="synthetic/provider-b",
        provider="synthetic-b",
        reasoning=True,
        max_tokens=128,
        temperature=0.3,
    )
    assert state_b["version"] == 1 and state_b["turn_index"] == 1, state_b

    persistent_episode = f"{EPISODE_CANARY}-restart"
    _, persistent_state = _assert_route(
        persistent_episode,
        "before restart",
        worker="worker-a",
        reasoning_header="off",
    )
    receipt.write_text(
        json.dumps(
            {
                "episode": persistent_episode,
                "version": persistent_state["version"],
                "turn_index": persistent_state["turn_index"],
            },
            sort_keys=True,
        )
    )
    _assert_failure_transactions()
    _assert_retained_session_extension()
    _assert_concurrency()
    _assert_response_boundaries()


def _resume(receipt: Path) -> None:
    expected = json.loads(receipt.read_text())
    before = _state(expected["episode"])
    assert before is not None
    assert before["version"] == expected["version"], before
    assert before["turn_index"] == expected["turn_index"], before
    _, after = _assert_route(
        expected["episode"],
        "after router restart",
        worker="worker-a",
        reasoning_header="off",
    )
    assert after["version"] == before["version"] + 1, after
    assert after["turn_index"] == before["turn_index"] + 1, after


def _redis_loss() -> None:
    before_count = len(_provider_requests())
    status, _, _ = _chat(
        f"{EPISODE_CANARY}-redis-loss",
        "Redis is intentionally unavailable",
    )
    assert status == HTTP_UNAVAILABLE, status
    assert len(_provider_requests()) == before_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("initial", "resume", "redis-loss"),
        required=True,
    )
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "initial":
        _initial(args.receipt)
    elif args.phase == "resume":
        _resume(args.receipt)
    else:
        _redis_loss()
    print(f"Rayline ARC stack phase {args.phase}: PASS")


if __name__ == "__main__":
    main()
