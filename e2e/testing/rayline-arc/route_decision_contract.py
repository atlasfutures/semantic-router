#!/usr/bin/env python3
"""Hermetic acceptance checks for the decision-only POST /v1/route endpoint."""

from __future__ import annotations

from typing import Any

from test_stack import (
    EPISODE_CANARY,
    HTTP_OK,
    PROMPT_CANARY,
    ROUTER_API_PORT,
    _json_request,
    _provider_requests,
    _state,
    _wait_for_state,
)

HTTP_BAD_REQUEST = 400
SECOND_TURN = 2

# The router has no source for these, so it must omit them. An invented value
# reads as measured to whoever joins these rows offline.
UNSOURCED_FIELDS = (
    "bundle_version",
    "pricing_snapshot_version",
    "catalog_provider",
    "policy",
    "route_call_index",
)


def _route_decision(
    session: str,
    marker: str,
    *,
    route_id: str | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    """One decision-only consult against the management listener."""
    headers = {"x-rayline-session": session}
    if route_id is not None:
        headers["x-rayline-route-id"] = route_id
    if messages is None:
        messages = [{"role": "user", "content": f"{PROMPT_CANARY} {marker}"}]
    return _json_request(
        ROUTER_API_PORT,
        "/v1/route",
        method="POST",
        body={"model": "claude-sonnet-4", "messages": messages, "max_tokens": 1},
        headers=headers,
    )


def _assert_names_a_worker_without_dispatching() -> None:
    """The endpoint answers with a worker and calls nobody.

    The caller executes the chosen worker itself, so "the provider saw no
    request" is the load-bearing assertion here, not a detail.
    """
    session = f"{EPISODE_CANARY}-decision-only-b"
    dispatched = len(_provider_requests())
    route_id = "rt_0a1b2c3d"

    status, body, _ = _route_decision(session, "ARC_ROUTE_B", route_id=route_id)

    assert status == HTTP_OK, (status, body)
    assert body["decision_id"] == route_id, body
    assert body["selected_worker"] == "worker-b", body
    assert body["worker_model"] == "synthetic/provider-b", body
    assert body["provider"] == "synthetic-b", body
    assert isinstance(body["decision_latency_ms"], (int, float)), body
    for unsourced in UNSOURCED_FIELDS:
        assert unsourced not in body, (unsourced, body)
    assert len(_provider_requests()) == dispatched, _provider_requests()

    # The encoder's routing axis is deterministic, so the other marker must
    # reach the other worker. Without this, the assertion above would also
    # pass on a router that always answered "worker-b".
    status, body, _ = _route_decision(
        f"{EPISODE_CANARY}-decision-only-a",
        "ARC_ROUTE_A",
        route_id="rt_0c2d",
    )
    assert status == HTTP_OK, (status, body)
    assert body["selected_worker"] == "worker-a", body
    assert body["worker_model"] == "synthetic/provider-a", body
    assert len(_provider_requests()) == dispatched, _provider_requests()


def _assert_commits_the_episode_at_decision_time() -> None:
    """The episode advances without a dispatch phase to commit against."""
    session = f"{EPISODE_CANARY}-decision-only-turns"

    status, body, _ = _route_decision(session, "ARC_ROUTE_B", route_id="rt_0d1e")
    assert status == HTTP_OK, (status, body)
    state = _wait_for_state(session, 1)
    assert state is not None and state["turn_index"] == 1, state

    status, body, _ = _route_decision(session, "ARC_ROUTE_B", route_id="rt_ml_0d1e")
    assert status == HTTP_OK, (status, body)
    assert body["decision_id"] == "rt_ml_0d1e", body
    state = _wait_for_state(session, SECOND_TURN)
    assert state is not None and state["turn_index"] == SECOND_TURN, state


def _assert_mints_a_decision_id_when_absent() -> None:
    """No route id is not an error: the router mints one and echoes it."""
    status, body, _ = _route_decision(
        f"{EPISODE_CANARY}-decision-only-minted",
        "ARC_ROUTE_A",
    )
    assert status == HTTP_OK, (status, body)
    assert isinstance(body["decision_id"], str) and body["decision_id"], body


def _assert_refuses_malformed_consults() -> None:
    """Malformed consults are refused before any episode turn is reserved."""
    dispatched = len(_provider_requests())
    for name, session, kwargs in (
        (
            "malformed route id",
            f"{EPISODE_CANARY}-decision-only-bad-id",
            {"route_id": "route-42"},
        ),
        (
            "empty messages",
            f"{EPISODE_CANARY}-decision-only-empty",
            {"route_id": "rt_0e3f", "messages": []},
        ),
    ):
        status, body, _ = _route_decision(session, "ARC_ROUTE_A", **kwargs)
        assert status == HTTP_BAD_REQUEST, (name, status, body)
        assert isinstance(body.get("detail"), str) and body["detail"], (name, body)
        # A refused consult must not shift the caller's route index.
        assert _state(session) is None, (name, _state(session))
    assert len(_provider_requests()) == dispatched, _provider_requests()


def main() -> None:
    _assert_names_a_worker_without_dispatching()
    _assert_commits_the_episode_at_decision_time()
    _assert_mints_a_decision_id_when_absent()
    _assert_refuses_malformed_consults()
    print("Rayline ARC decision-only route contract passed", flush=True)


if __name__ == "__main__":
    main()
