#!/usr/bin/env python3
"""Hermetic dynamic ARC membership and controller drain acceptance checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time
from collections.abc import Callable
from typing import Any

from test_stack import (
    EPISODE_CANARY,
    HTTP_OK,
    STATE_KEY_PREFIX,
    _chat,
    _redis_command,
    _state,
)

MEMBERSHIP_KEY = f"{STATE_KEY_PREFIX}encoder-membership"
REFRESH_WAIT_SECONDS = 1.2
SCALE_OUT_REVISION = 2
DRAIN_REVISION = 3
REMOVAL_REVISION = 4
DYNAMIC_MEMBER_COUNT = 3


def _owner(episode: str, replicas: tuple[str, ...]) -> str:
    episode_hash = hashlib.sha256(episode.encode()).hexdigest()
    scores = {
        replica: hashlib.sha256(
            episode_hash.encode() + b"\x00" + replica.encode()
        ).digest()
        for replica in replicas
    }
    return max(scores, key=scores.__getitem__)


def _episode_for_owner(prefix: str, replicas: tuple[str, ...], owner: str) -> str:
    for index in range(10_000):
        episode = f"{prefix}-{index}"
        if _owner(episode, replicas) == owner:
            return episode
    raise AssertionError(f"could not find dynamic episode for {owner}")


def _membership() -> dict[str, Any]:
    payload = _redis_command("GET", MEMBERSHIP_KEY)
    assert isinstance(payload, bytes), payload
    document = json.loads(payload)
    assert isinstance(document, dict), document
    return document


def _wait_for_membership(
    predicate: Callable[[dict[str, Any]], bool],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last = _membership()
    while time.monotonic() < deadline:
        if predicate(last):
            return last
        time.sleep(0.25)
        last = _membership()
    raise AssertionError(f"membership transition timed out: {last}")


def _write_receipt(path: pathlib.Path, episode_a: str) -> None:
    path.write_text(json.dumps({"episode_a": episode_a}), encoding="utf-8")


def _read_episode(path: pathlib.Path) -> str:
    document = json.loads(path.read_text(encoding="utf-8"))
    episode = document.get("episode_a")
    assert isinstance(episode, str) and episode, document
    return episode


def prepare_dynamic_membership(receipt: pathlib.Path) -> None:
    members = ("encoder-a", "encoder-b", "encoder-c")

    membership = _wait_for_membership(
        lambda current: current.get("revision") == SCALE_OUT_REVISION
        and {member.get("id") for member in current.get("replicas", [])} == set(members)
        and all(
            member.get("state") == "active" for member in current.get("replicas", [])
        ),
        timeout_seconds=5,
    )
    assert len(membership["replicas"]) == DYNAMIC_MEMBER_COUNT, membership
    time.sleep(REFRESH_WAIT_SECONDS)
    episode_a = _episode_for_owner(
        f"{EPISODE_CANARY}-dynamic-drain",
        members,
        "encoder-a",
    )
    status, _, _ = _chat(episode_a, "dynamic active owner")
    assert status == HTTP_OK
    state = _state(episode_a)
    assert state is not None and state["encoder_owner"] == "encoder-a", state
    _write_receipt(receipt, episode_a)


def assert_controller_drain(receipt: pathlib.Path) -> None:
    membership = _wait_for_membership(
        lambda current: current.get("revision") == DRAIN_REVISION
        and any(
            member.get("id") == "encoder-a"
            and member.get("state") == "draining"
            and member.get("drain_started_at")
            for member in current.get("replicas", [])
        ),
        timeout_seconds=5,
    )
    assert len(membership["replicas"]) == DYNAMIC_MEMBER_COUNT, membership
    time.sleep(REFRESH_WAIT_SECONDS)

    episode_a = _read_episode(receipt)
    status, _, _ = _chat(episode_a, "dynamic draining owner", close=True)
    assert status == HTTP_OK
    state = _state(episode_a)
    assert state is not None and state["encoder_owner"] == "", state
    assert state["encoder_visited_owners"] == [], state


def assert_controller_removal() -> None:
    membership = _wait_for_membership(
        lambda current: current.get("revision") == REMOVAL_REVISION
        and {member.get("id") for member in current.get("replicas", [])}
        == {"encoder-b", "encoder-c"},
        timeout_seconds=20,
    )
    assert all(member.get("state") == "active" for member in membership["replicas"])
    time.sleep(REFRESH_WAIT_SECONDS)

    episode_c = _episode_for_owner(
        f"{EPISODE_CANARY}-dynamic-removed",
        ("encoder-b", "encoder-c"),
        "encoder-c",
    )
    status, _, _ = _chat(episode_c, "dynamic replacement owner")
    assert status == HTTP_OK
    state = _state(episode_c)
    assert state is not None and state["encoder_owner"] == "encoder-c", state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=("prepare", "drain", "removed"),
    )
    parser.add_argument("--receipt", type=pathlib.Path)
    args = parser.parse_args()

    if args.phase in {"prepare", "drain"} and args.receipt is None:
        parser.error("--receipt is required for prepare and drain")
    if args.phase == "prepare":
        prepare_dynamic_membership(args.receipt)
    elif args.phase == "drain":
        assert_controller_drain(args.receipt)
    else:
        assert_controller_removal()
    print(f"Rayline ARC dynamic membership {args.phase}: PASS", flush=True)


if __name__ == "__main__":
    main()
