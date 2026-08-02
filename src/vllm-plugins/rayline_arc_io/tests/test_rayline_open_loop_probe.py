# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import http.client
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

parity = importlib.import_module("rayline_parity_http_probe")
probe = importlib.import_module("rayline_open_loop_probe")
receipt_schema = importlib.import_module("rayline_parity_comparator")


def _case(episode: str, turn: int) -> parity.Case:
    return parity.Case(
        case_id=f"{episode}-{turn}",
        episode_id=episode,
        input_tokens=100,
        messages=({"role": "user", "content": "private"},),
    )


def test_poisson_schedule_is_seeded_and_round_robins_episode_turns() -> None:
    cases = [_case(episode, turn) for episode in ("a", "b") for turn in range(2)]

    first = probe.poisson_schedule(cases, 0.3, 20260730)
    second = probe.poisson_schedule(cases, 0.3, 20260730)

    assert first == second
    assert sorted(first, key=first.get) == ["a-0", "b-0", "a-1", "b-1"]
    assert first["a-0"] == 0.0
    assert all(value >= 0 for value in first.values())


def test_json_client_closes_the_calling_threads_connection() -> None:
    client = parity.JSONClient("http://127.0.0.1:1", timeout_seconds=1.0)
    client._local.connection = http.client.HTTPConnection("127.0.0.1", 1)

    client.close_thread_connection()

    assert client._local.connection is None


def test_open_loop_receipt_schema_is_aggregate_only() -> None:
    source = (SCRIPT_DIR / "rayline_open_loop_probe.py").read_text()

    assert "scheduled_latency_seconds" in probe.RESULT_FIELDS
    assert "start_lag_seconds" in probe.RESULT_FIELDS
    assert "realized_arrival_rate_rps" in probe.RESULT_FIELDS
    assert "backlog_at_final_arrival" in probe.RESULT_FIELDS
    assert '"messages"' not in probe.RESULT_FIELDS
    assert "raw_cases" not in probe.RESULT_FIELDS
    assert "provider_calls" in source


def test_open_loop_probe_executes_all_episode_lanes(monkeypatch) -> None:
    measured = [
        _case(f"episode-{episode}", turn) for episode in range(8) for turn in range(4)
    ]
    digest = hashlib.sha256(b"fixture").hexdigest()
    identity = {
        field: (
            probe.MEASUREMENT_SCOPE
            if field == "measurement_scope"
            else (
                32
                if field == "case_count"
                else (
                    20260730
                    if field == "seed"
                    else digest if field.endswith("sha256") else field
                )
            )
        )
        for field in receipt_schema.IDENTITY_FIELDS
    }
    monkeypatch.setattr(
        probe,
        "_selector",
        lambda **_kwargs: lambda _case_value: ("worker", 0.0001),
    )

    class Client:
        closes = 0

        def close_thread_connection(self) -> None:
            self.closes += 1

    client = Client()
    receipt = probe.run_open_loop_probe(
        arm="rayline_arc",
        protocol=parity.PROTOCOL_BY_ARM["rayline_arc"],
        client=client,
        warmup=[],
        measured=measured,
        identity=identity,
        worker_map={"worker": "canonical"},
        run_id="test-open-loop",
        offered_rate_rps=10000.0,
    )

    assert receipt["results"]["completed"] == len(measured)
    assert receipt["results"]["failed"] == 0
    assert receipt["results"]["provider_calls"] == 0
    assert client.closes == len(measured)
    assert receipt["results"]["realized_arrival_rate_rps"] > 0

    legacy = copy.deepcopy(receipt)
    legacy["schema_version"] = probe.LEGACY_INPUT_SCHEMA
    legacy["results"].pop("realized_arrival_rate_rps")
    normalized = probe.validate_receipt(legacy)
    assert normalized["schema_version"] == probe.LEGACY_INPUT_SCHEMA
    assert normalized["results"]["realized_arrival_rate_rps"] == (
        (len(measured) - 1) / normalized["results"]["scheduled_span_seconds"]
    )
