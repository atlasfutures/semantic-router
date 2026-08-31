#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Crash-durable, privacy-safe evidence for each KV benchmark request."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from modal_fullstack_inputs import CANDIDATE_PROMPTS

SCHEMA_VERSION = "rayline.openrouter-kv-cache-journal.v1"
FORBIDDEN_KEYS = frozenset(
    {
        "authorization",
        "messages",
        "prompt",
        "request_body",
        "response_body",
        "tools",
    }
)
PROTECTED_ENVIRONMENT_NAMES = (
    "OPENROUTER_EPHEMERAL_API_KEY",
    "RAYLINE_MODAL_NATIVE_ROUTER_TOKEN",
    "RAYLINE_ARC_E2E_MODAL_KEY",
    "RAYLINE_ARC_E2E_MODAL_SECRET",
)


def _validate_privacy(value: Any) -> None:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True)
    if any(anchor in encoded for anchor in CANDIDATE_PROMPTS):
        raise RuntimeError("routing anchor entered the KV request journal")
    for name in PROTECTED_ENVIRONMENT_NAMES:
        protected = os.environ.get(name, "")
        if protected and protected in encoded:
            raise RuntimeError("credential entered the KV request journal")

    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            if any(str(key).casefold() in FORBIDDEN_KEYS for key in current):
                raise RuntimeError("request content entered the KV request journal")
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)


def initialize(path: Path) -> None:
    """Create a new journal and durably publish its directory entry."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def append(path: Path, event: dict[str, Any]) -> None:
    """Append and fsync one complete JSONL event."""

    record = {"schema_version": SCHEMA_VERSION, **event}
    _validate_privacy(record)
    encoded = (
        json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    descriptor = os.open(path, os.O_APPEND | os.O_WRONLY)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError("short write while appending KV request journal")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read(path: Path) -> list[dict[str, Any]]:
    """Read a complete journal for recovery and focused validation."""

    raw = path.read_text()
    lines = [line for line in raw.splitlines() if line]
    events: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as error:
            if index != len(lines) - 1 or raw.endswith("\n"):
                raise RuntimeError(
                    "KV request journal was corrupted before its tail"
                ) from error
            break
    if any(event.get("schema_version") != SCHEMA_VERSION for event in events):
        raise RuntimeError("KV request journal schema diverged")
    return events
