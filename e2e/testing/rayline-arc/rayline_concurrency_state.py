#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed retained-session cleanup between PERF017 sweep cells."""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

STATE_RECEIPT_SCHEMA = "rayline.vllm.concurrency-cell-state-reset.v1"
HTTP_OK = 200


class StateResetError(RuntimeError):
    """A sweep cell did not return the protected encoder to empty state."""


JSONRequester = Callable[[str, str], Mapping[str, Any]]


def hash_episode_id(raw_episode_id: str) -> str:
    """Match semantic-router's raylinearc.HashEpisodeID exactly."""

    return hashlib.sha256(raw_episode_id.encode()).hexdigest()


@dataclass(frozen=True)
class ProtectedEncoderClient:
    """Minimal authenticated client that never records credentials or payloads."""

    base_url: str
    modal_key: str = field(repr=False)
    modal_secret: str = field(repr=False)
    timeout_seconds: float = 30.0

    def request(self, method: str, path: str) -> Mapping[str, Any]:
        request = urllib.request.Request(
            self.base_url.rstrip("/") + path,
            headers={
                "Accept": "application/json",
                "Modal-Key": self.modal_key,
                "Modal-Secret": self.modal_secret,
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                status = response.status
                body = response.read()
        except urllib.error.HTTPError as error:
            raise StateResetError(
                f"protected encoder returned HTTP {error.code}"
            ) from error
        if status != HTTP_OK:
            raise StateResetError(f"protected encoder returned HTTP {status}")
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as error:
            raise StateResetError("protected encoder returned invalid JSON") from error
        if not isinstance(decoded, dict):
            raise StateResetError("protected encoder response must be an object")
        return decoded


def assert_encoder_empty(requester: JSONRequester) -> dict[str, int]:
    health = requester("GET", "/health")
    if health.get("status") != "ok":
        raise StateResetError("protected encoder health is not ok")
    resident_sessions = health.get("resident_sessions")
    resident_tokens = health.get("resident_tokens")
    if (
        isinstance(resident_sessions, bool)
        or not isinstance(resident_sessions, int)
        or isinstance(resident_tokens, bool)
        or not isinstance(resident_tokens, int)
    ):
        raise StateResetError("protected encoder health counts are invalid")
    if resident_sessions < 0 or resident_tokens < 0:
        raise StateResetError("protected encoder health counts are invalid")
    if resident_sessions != 0 or resident_tokens != 0:
        raise StateResetError("protected encoder retained state between sweep cells")
    return {
        "resident_sessions": resident_sessions,
        "resident_tokens": resident_tokens,
    }


def close_cell_sessions(
    *,
    requester: JSONRequester,
    probe_run_id: str,
    measured_episode_ids: Iterable[str],
    warmup_episode_ids: Iterable[str],
    require_measured_present: bool,
) -> dict[str, Any]:
    """Close one cell's namespaced ARC sessions and prove empty residency.

    The encoder has eight slots while the packet has one warmup plus eight
    measured episodes. The warmup is therefore allowed to have been evicted;
    every measured episode must be present after a successful measured arm.
    """

    measured = tuple(dict.fromkeys(map(str, measured_episode_ids)))
    warmup = tuple(dict.fromkeys(map(str, warmup_episode_ids)))
    if not measured or set(measured) & set(warmup):
        raise StateResetError("cell episode sets are empty or overlap")

    measured_closed = 0
    measured_missing = 0
    warmup_closed = 0
    warmup_missing = 0
    for episode_id, required in (
        *((episode_id, require_measured_present) for episode_id in measured),
        *((episode_id, False) for episode_id in warmup),
    ):
        episode_hash = hash_episode_id(f"{probe_run_id}:{episode_id}")
        response = requester(
            "DELETE",
            f"/v1/rayline/arc/session/{episode_hash}",
        )
        closed = response.get("closed")
        if not isinstance(closed, bool):
            raise StateResetError("session close response omitted boolean status")
        if required and not closed:
            raise StateResetError("a completed measured episode was not resident")
        if episode_id in measured:
            measured_closed += int(closed)
            measured_missing += int(not closed)
        else:
            warmup_closed += int(closed)
            warmup_missing += int(not closed)

    empty = assert_encoder_empty(requester)
    return {
        "schema_version": STATE_RECEIPT_SCHEMA,
        "measured_episode_count": len(measured),
        "measured_sessions_closed": measured_closed,
        "measured_sessions_missing": measured_missing,
        "warmup_episode_count": len(warmup),
        "warmup_sessions_closed": warmup_closed,
        "warmup_sessions_missing": warmup_missing,
        "resident_sessions_after_cleanup": empty["resident_sessions"],
        "resident_tokens_after_cleanup": empty["resident_tokens"],
    }
