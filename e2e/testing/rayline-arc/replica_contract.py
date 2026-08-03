#!/usr/bin/env python3
"""Hermetic retained-encoder membership, failover, recovery, and close checks."""

from __future__ import annotations

import hashlib
import time

from test_stack import (
    EPISODE_CANARY,
    HTTP_OK,
    _chat,
    _encoder_stats,
    _metric_value,
    _reset_encoders,
    _state,
)

REPLICA_IDS = ("encoder-a", "encoder-b")
EXPECTED_SURVIVOR_REQUESTS = 2
MIN_CLOSED_REPLICAS = len(REPLICA_IDS)


def _rendezvous_owner(episode: str) -> str:
    episode_hash = hashlib.sha256(episode.encode()).hexdigest()
    scores = {
        replica_id: hashlib.sha256(
            episode_hash.encode() + b"\x00" + replica_id.encode()
        ).digest()
        for replica_id in REPLICA_IDS
    }
    return max(scores, key=scores.__getitem__)


def _episode_for_owner(prefix: str, owner: str) -> str:
    for index in range(10_000):
        episode = f"{prefix}-{index}"
        if _rendezvous_owner(episode) == owner:
            return episode
    raise AssertionError(f"could not find episode for {owner}")


def assert_replica_contract() -> None:
    _reset_encoders()
    episode = _episode_for_owner(
        f"{EPISODE_CANARY}-replica-failover",
        "encoder-a",
    )
    before_failovers = _metric_value(
        "llm_rayline_arc_encoder_replica_routes_total",
        outcome="failover",
    )
    status, _, _ = _chat(episode, "ARC_ENCODER_A_UNAVAILABLE first")
    assert status == HTTP_OK
    first_state = _state(episode)
    assert first_state is not None
    assert first_state["schema_version"] == "rayline.arc.episode-state.v2"
    assert first_state["encoder_owner"] == "encoder-b", first_state
    assert first_state["encoder_visited_owners"] == [
        "encoder-a",
        "encoder-b",
    ], first_state

    status, _, _ = _chat(episode, "sticky survivor")
    assert status == HTTP_OK
    second_state = _state(episode)
    assert second_state is not None
    assert second_state["encoder_owner"] == "encoder-b", second_state
    stats = _encoder_stats()
    assert stats["encoder-a"]["pooling_requests"] == 1, stats
    assert stats["encoder-b"]["pooling_requests"] == EXPECTED_SURVIVOR_REQUESTS, stats
    assert (
        _metric_value(
            "llm_rayline_arc_encoder_replica_routes_total",
            outcome="failover",
        )
        == before_failovers + 1
    )

    status, _, _ = _chat(episode, "explicit close", close=True)
    assert status == HTTP_OK
    closed_state = _state(episode)
    assert closed_state is not None
    assert closed_state["encoder_owner"] == "", closed_state
    assert closed_state["encoder_visited_owners"] == [], closed_state
    stats = _encoder_stats()
    assert all(value["close_calls"] == 1 for value in stats.values()), stats
    assert all(value["resident_sessions"] == 0 for value in stats.values()), stats
    assert (
        _metric_value(
            "llm_rayline_arc_encoder_session_closes_total",
            outcome="closed",
        )
        >= MIN_CLOSED_REPLICAS
    )

    time.sleep(1.1)
    recovered_episode = _episode_for_owner(
        f"{EPISODE_CANARY}-replica-recovered",
        "encoder-a",
    )
    status, _, _ = _chat(recovered_episode, "recovered assignment")
    assert status == HTTP_OK
    recovered_state = _state(recovered_episode)
    assert recovered_state is not None
    assert recovered_state["encoder_owner"] == "encoder-a", recovered_state


if __name__ == "__main__":
    assert_replica_contract()
    print("Rayline ARC replica contract: PASS", flush=True)
