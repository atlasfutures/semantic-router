# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
import sys
from http import HTTPStatus
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

proxy = importlib.import_module("rayline_affinity_proxy")


def _state(
    *, failover_after_pooling: int | None = None
) -> tuple[object, list[tuple[str, str]]]:
    calls: list[tuple[str, str]] = []
    pooling_by_host: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.url.host or "", request.url.path))
        if request.url.path == "/health":
            resident = 1 if request.url.host == "replica-a.test" else 2
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "resident_sessions": resident,
                    "resident_tokens": resident * 10,
                    "max_sessions": 8,
                    "max_resident_tokens": 100,
                    "pooling_capabilities": [
                        "chunked_causal_mean",
                        "resumable_causal_mean",
                    ],
                },
            )
        if request.method == "DELETE":
            return httpx.Response(200, json={"closed": True})
        host = request.url.host or ""
        pooling_by_host[host] = pooling_by_host.get(host, 0) + 1
        return httpx.Response(
            200,
            json={
                "session_action": (
                    "created" if pooling_by_host[host] == 1 else "appended"
                )
            },
        )

    state = proxy.AffinityState(
        ["https://replica-a.test", "https://replica-b.test"],
        modal_key="key",
        modal_secret="secret",
        timeout_seconds=10.0,
        failover_after_pooling=failover_after_pooling,
    )
    state._client.close()
    state._client = httpx.Client(transport=httpx.MockTransport(handler))
    return state, calls


def test_episode_hash_has_stable_replica_for_pooling_and_close() -> None:
    state, calls = _state()
    episode_a = "0" * 64
    episode_b = "0" * 63 + "1"
    # The shard uses the first 64 bits, so choose a first-word value of one.
    episode_b = "0000000000000001" + episode_b[16:]
    try:
        for episode in (episode_a, episode_b):
            body = json.dumps({"episode_id_hash": episode, "turns": [{}]}).encode()
            assert state.forward_session_pooling(body).status_code == HTTPStatus.OK
            assert state.forward_session_pooling(body).status_code == HTTPStatus.OK
            assert state.forward_close(episode).status_code == HTTPStatus.OK
        stats = json.loads(state.stats().body)
    finally:
        state.close()

    assert calls[:3] == [
        ("replica-a.test", proxy.SESSION_POOLING_PATH),
        ("replica-a.test", proxy.SESSION_POOLING_PATH),
        ("replica-a.test", proxy.SESSION_CLOSE_PREFIX + episode_a),
    ]
    assert calls[3:] == [
        ("replica-b.test", proxy.SESSION_POOLING_PATH),
        ("replica-b.test", proxy.SESSION_POOLING_PATH),
        ("replica-b.test", proxy.SESSION_CLOSE_PREFIX + episode_b),
    ]
    assert stats == {
        "schema_version": proxy.STATS_SCHEMA,
        "replicas": 2,
        "requests_by_replica": [3, 3],
        "session_pooling_requests_by_replica": [2, 2],
        "session_deletes_by_replica": [1, 1],
        "unique_sessions_by_replica": [1, 1],
        "affinity_mismatches": 0,
    }

    state, _calls = _state()
    try:
        body = json.dumps({"episode_id_hash": episode_a, "turns": [{}]}).encode()
        state.forward_session_pooling(body)
        assert json.loads(state.reset_stats().body) == {"status": "reset"}
        assert json.loads(state.stats().body)["requests_by_replica"] == [0, 0]
    finally:
        state.close()


def test_health_aggregates_state_without_hiding_replica_capacity() -> None:
    state, _calls = _state()
    try:
        health = json.loads(state.aggregate_health().body)
    finally:
        state.close()

    assert health == {
        "status": "ok",
        "resident_sessions": 3,
        "resident_tokens": 30,
        "max_sessions": 16,
        "max_resident_tokens": 200,
        "pooling_capabilities": [
            "chunked_causal_mean",
            "resumable_causal_mean",
        ],
    }


def test_forced_failover_rebuilds_on_peer_and_fans_out_close() -> None:
    state, calls = _state(failover_after_pooling=2)
    episode = "0" * 64
    body = json.dumps({"episode_id_hash": episode, "turns": [{}]}).encode()
    try:
        for _ in range(4):
            assert state.forward_session_pooling(body).status_code == HTTPStatus.OK
        close = json.loads(state.forward_close(episode).body)
        stats = json.loads(state.stats().body)
    finally:
        state.close()

    assert calls == [
        ("replica-a.test", proxy.SESSION_POOLING_PATH),
        ("replica-a.test", proxy.SESSION_POOLING_PATH),
        ("replica-b.test", proxy.SESSION_POOLING_PATH),
        ("replica-b.test", proxy.SESSION_POOLING_PATH),
        ("replica-a.test", proxy.SESSION_CLOSE_PREFIX + episode),
        ("replica-b.test", proxy.SESSION_CLOSE_PREFIX + episode),
    ]
    assert close == {"closed": True}
    assert stats == {
        "schema_version": proxy.FAILOVER_STATS_SCHEMA,
        "replicas": 2,
        "requests_by_replica": [3, 3],
        "session_pooling_requests_by_replica": [2, 2],
        "session_deletes_by_replica": [1, 1],
        "unique_sessions_by_replica": [1, 1],
        "primary_sessions_by_replica": [1, 0],
        "affinity_mismatches": 0,
        "failover_after_pooling": 2,
        "failover_pooling_requests": 2,
        "failover_sessions": 1,
        "close_fanout_requests": 2,
        "session_rebuild_responses": 1,
    }


def test_proxy_rejects_duplicate_upstreams_and_invalid_hashes() -> None:
    try:
        proxy.AffinityState(
            ["https://replica.test", "https://replica.test"],
            modal_key="key",
            modal_secret="secret",
            timeout_seconds=10.0,
        )
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate affinity upstreams were accepted")

    state, _calls = _state()
    try:
        try:
            state.replica_for_hash("not-a-hash")
        except proxy.AffinityProxyError:
            pass
        else:
            raise AssertionError("invalid episode hash was accepted")
    finally:
        state.close()
