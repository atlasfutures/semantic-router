# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

open_loop = importlib.import_module("rayline_open_loop_probe")
probe = importlib.import_module("rayline_replica_stop_probe")


class FakeClient:
    def close_thread_connection(self) -> None:
        pass


def test_staged_probe_preloads_before_boundary_and_measures_only_post(
    monkeypatch,
) -> None:
    packet = REPO_ROOT / ".agent-harness/rayline-parity/packet-perf020"
    warmup, measured, identity, worker_map, workload = open_loop.load_open_loop_packet(
        arm="rayline_arc",
        corpus_path=packet / "corpus.json",
        workload_path=packet / "cells/r030/workload.json",
        topology_path=packet / "topology.json",
        identity_path=packet / "cells/r030/identity.json",
    )
    selected: list[str] = []
    boundary_seen: list[int] = []

    def selector(case):
        selected.append(case.case_id)
        return next(iter(worker_map)), 0.5

    monkeypatch.setattr(probe, "_warm_selector", lambda **_kwargs: selector)

    def execute(*, lanes, schedule, **_kwargs):
        outcomes = {}
        for lane in lanes.values():
            for index, case in enumerate(lane, start=1):
                scheduled = schedule[case.case_id]
                outcomes[case.case_id] = {
                    "case_id": case.case_id,
                    "canonical": next(iter(worker_map.values())),
                    "success": True,
                    "service": 0.5,
                    "start": scheduled + 0.01,
                    "completion": scheduled + 0.5 + index * 0.01,
                    "scheduled": scheduled,
                }
        return outcomes

    monkeypatch.setattr(probe, "_execute_open_loop", execute)

    def boundary():
        boundary_seen.append(len(selected))
        return {"action": "control"}

    receipt = probe.run_replica_stop_probe(
        client=FakeClient(),
        warmup=warmup,
        measured=measured,
        identity=identity,
        worker_map=worker_map,
        run_id="run:r030:shared",
        offered_rate_rps=float(workload["offered_rate_rps"]),
        boundary_callback=boundary,
    )

    assert boundary_seen == [probe.EXPECTED_PRELOAD_TURNS]
    assert receipt["preload"]["completed"] == probe.EXPECTED_PRELOAD_TURNS
    assert receipt["results"]["completed"] == probe.EXPECTED_POST_BOUNDARY_TURNS
    assert receipt["results"]["failed"] == 0
    assert receipt["stage"]["boundary_excluded_from_latency"] is True
    assert receipt["boundary"] == {"action": "control"}
