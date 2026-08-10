# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

comparator = importlib.import_module("rayline_open_loop_comparator")
contract = importlib.import_module("rayline_open_loop_contract")
packet = importlib.import_module("rayline_open_loop_packet")
parity = importlib.import_module("rayline_parity_comparator")
probe = importlib.import_module("rayline_open_loop_probe")


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _identity(label: str) -> dict[str, object]:
    identity: dict[str, object] = {}
    digest_fields = {
        "corpus_sha256",
        "workload_sha256",
        "tokenizer_sha256",
        "worker_topology_sha256",
    }
    for field in parity.IDENTITY_FIELDS:
        if field == "measurement_scope":
            identity[field] = packet.MEASUREMENT_SCOPE
        elif field == "case_count":
            identity[field] = 32
        elif field == "seed":
            identity[field] = 20260730
        elif field in digest_fields:
            identity[field] = _digest(label if field == "workload_sha256" else field)
        else:
            identity[field] = field
    return identity


def _receipt(label: str, rate: float, arm: str) -> dict[str, object]:
    latency = {"p50": 0.1, "p95": 0.2, "p99": 0.3}
    return {
        "schema_version": probe.INPUT_SCHEMA,
        "arm": arm,
        "run_id": f"run:{label}:{arm}",
        "identity": _identity(label),
        "results": {
            "scheduled": 32,
            "completed": 32,
            "failed": 0,
            "offered_rate_rps": rate,
            "realized_arrival_rate_rps": rate,
            "scheduled_span_seconds": 31 / rate,
            "duration_seconds": 101.0,
            "completion_throughput_rps": 32 / 101,
            "achieved_start_rate_rps": rate,
            "service_latency_seconds": latency,
            "scheduled_latency_seconds": latency,
            "start_lag_seconds": {"p50": 0.0, "p95": 0.0, "p99": 0.0},
            "max_client_backlog": 2,
            "backlog_at_final_arrival": 1,
            "drain_seconds_after_final_arrival": 1.0,
            "selected_worker_trace_sha256": _digest("trace"),
            "provider_calls": 0,
        },
    }


def _cells(
    rates: tuple[float, ...] = packet.OFFERED_RATES,
) -> dict[str, dict[str, dict[str, object]]]:
    return {
        packet.rate_label(rate): {
            arm: _receipt(packet.rate_label(rate), rate, arm)
            for arm in comparator.OPEN_LOOP_ARMS
        }
        for rate in rates
    }


def test_open_loop_comparison_passes_integrity_and_reports_knee() -> None:
    cells = _cells()
    cells["r045"]["rayline_arc"]["results"]["achieved_start_rate_rps"] = 0.3
    cells["r045"]["rayline_arc"]["results"]["backlog_at_final_arrival"] = 8
    cells["r045"]["rayline_arc"]["results"]["max_client_backlog"] = 9

    result = comparator.compare_open_loop(cells)

    assert result["status"] == "passed"
    assert result["cross_cell_trace_match"] is True
    assert result["first_overloaded_cell"]["rayline_remote"] is None
    assert result["first_overloaded_cell"]["rayline_arc"] == "r045"


def test_open_loop_comparison_fails_trace_integrity() -> None:
    cells = _cells()
    cells["r030"]["rayline_arc"]["results"]["selected_worker_trace_sha256"] = _digest(
        "different"
    )

    result = comparator.compare_open_loop(cells)

    assert result["status"] == "failed"
    assert result["cross_cell_trace_match"] is False


def test_open_loop_comparison_rejects_cross_arm_identity_mismatch() -> None:
    cells = _cells()
    broken = copy.deepcopy(cells)
    broken["r015"]["rayline_arc"]["identity"]["warm_state"] = "cold"

    with pytest.raises(comparator.OpenLoopComparisonError, match="identities differ"):
        comparator.compare_open_loop(broken)


# PERF032 ran a legitimately parameterised four-rung ladder and the comparator
# rejected it, because it validated against the packet module's frozen default
# instead of the rungs the run contracted for. The rung set is a property of the
# run; the default is only a convenience for the two closed runs that predate it.
PERF032_RATES = (0.45, 0.60, 0.90, 1.20)


def test_open_loop_comparison_accepts_a_contracted_four_rung_ladder() -> None:
    cells = _cells(PERF032_RATES)

    result = comparator.compare_open_loop(cells, PERF032_RATES)

    assert result["status"] == "passed"
    assert tuple(result["arc_vs_remote"]) == ("r045", "r060", "r090", "r120")
    assert tuple(result["load_diagnostics"]["rayline_arc"]) == (
        "r045",
        "r060",
        "r090",
        "r120",
    )


def test_open_loop_comparison_defaults_to_the_frozen_three_rung_ladder() -> None:
    result = comparator.compare_open_loop(_cells())

    assert result["status"] == "passed"
    assert tuple(result["arc_vs_remote"]) == ("r015", "r030", "r045")
    assert comparator.compare_open_loop(_cells(), packet.OFFERED_RATES) == result


def test_open_loop_comparison_rejects_cells_that_are_not_the_contracted_rungs() -> None:
    with pytest.raises(
        comparator.OpenLoopComparisonError, match="differ from the contracted rates"
    ):
        comparator.compare_open_loop(_cells(PERF032_RATES))

    with pytest.raises(
        comparator.OpenLoopComparisonError, match="differ from the contracted rates"
    ):
        comparator.compare_open_loop(_cells(), PERF032_RATES)


def test_open_loop_comparison_rejects_a_malformed_rung_set() -> None:
    with pytest.raises(comparator.OpenLoopComparisonError, match="must ascend"):
        comparator.compare_open_loop(_cells(), (0.45, 0.30))


# The closed runs' receipts live under the gitignored `.agent-harness/`, so the
# tests that replay real measurements skip wherever they are absent. They are
# the only evidence that separates a criterion that measures saturation from
# one that measures the arithmetic of a finite corpus, so they are worth the
# conditional.
PARITY_ROOT = REPO_ROOT / ".agent-harness/rayline-parity"
PERF021_RUN = PARITY_ROOT / "rayline-open-loop-sweep-perf021-20260802"
PERF020_RUN = PARITY_ROOT / "rayline-open-loop-sweep-perf020-20260802"
PERF032_RUN = PARITY_ROOT / "rayline-saturation-knee-perf032-20260810"
PERF032_RATES = (0.45, 0.60, 0.90, 1.20)
EIGHT_LANES = contract.SaturationCriterion(episode_lanes=8, occupancy_ratio=1.0)


def _on_disk_cells(
    run_dir: Path, rates: tuple[float, ...]
) -> dict[str, dict[str, dict[str, object]]]:
    if not run_dir.is_dir():
        pytest.skip(f"{run_dir.name} receipts are not present")
    return {
        packet.rate_label(rate): {
            arm: json.loads((run_dir / packet.rate_label(rate) / f"{arm}.json").read_text())
            for arm in comparator.OPEN_LOOP_ARMS
        }
        for rate in rates
    }


def test_closed_runs_replay_into_byte_identical_comparisons() -> None:
    """PERF021's own receipts must still produce PERF021's own report.

    The saturation criterion is opt-in precisely so this holds. A closed run
    whose recorded verdict silently changes shape is a closed run whose
    evidence can no longer be checked.
    """

    cells = _on_disk_cells(PERF021_RUN, packet.OFFERED_RATES)
    recorded = (PERF021_RUN / "comparison.json").read_text()

    replayed = comparator.compare_open_loop(cells)

    assert replayed["schema_version"] == comparator.REPORT_SCHEMA
    assert json.dumps(replayed, indent=2, sort_keys=True) + "\n" == recorded


def test_saturation_criterion_fires_on_the_known_saturated_control() -> None:
    """PERF021 saturated at `r030` and the criterion has to say so unprompted.

    This is the positive control. A criterion that cannot reproduce a knee the
    repo already measured is not a criterion, whatever it does elsewhere.
    """

    cells = _on_disk_cells(PERF021_RUN, packet.OFFERED_RATES)

    report = comparator.compare_open_loop(
        cells, packet.OFFERED_RATES, 32, EIGHT_LANES
    )

    assert report["schema_version"] == comparator.SATURATION_REPORT_SCHEMA
    assert report["first_saturated_cell"] == {
        "rayline_arc": "r030",
        "rayline_remote": "r030",
    }
    assert report["saturation"]["rayline_arc"]["r015"]["peak_lane_occupancy"] == 0.75
    assert report["saturation"]["rayline_arc"]["r030"]["peak_lane_occupancy"] == 1.0


def test_saturation_criterion_does_not_fire_on_perf032() -> None:
    """PERF032 never saturated, and the successor criterion must not pretend it did.

    Every request in every PERF032 cell started within six milliseconds of its
    scheduled arrival and the drain never exceeded one service-p95 request, so
    nothing queued at any rung. What the criterion adds over the predicate that
    ran is resolution: `r120` is recorded as `0.875` of the lane ceiling
    instead of as an undifferentiated `false`.
    """

    cells = _on_disk_cells(PERF032_RUN, PERF032_RATES)

    report = comparator.compare_open_loop(cells, PERF032_RATES, 32, EIGHT_LANES)

    assert report["first_saturated_cell"] == {
        "rayline_arc": None,
        "rayline_remote": None,
    }
    occupancy = [
        report["saturation"]["rayline_arc"][label]["peak_lane_occupancy"]
        for label in ("r045", "r060", "r090", "r120")
    ]
    assert occupancy == [0.375, 0.375, 0.625, 0.875]
    for label in ("r045", "r060", "r090", "r120"):
        cell = report["saturation"]["rayline_arc"][label]
        # The drain never exceeded a single service-p95 request, which is what
        # an unqueued cell looks like however far its completion ratio has
        # fallen.
        assert cell["drain_service_tail_multiple"] <= 1.01


def test_completion_ratio_alone_would_misread_perf032() -> None:
    """Why the completion ratio is recorded but does not vote.

    PERF032's `r120` completed `1.15475` decisions per second against
    `1.48960` realized arrivals, which reads as a system 22.5% underwater and
    is what a `0.90` or `0.95` completion floor would have fired on. It is not
    a service deficit. `completion_throughput_rps` is `32 / duration` and
    `duration` is `span + drain`, so a fixed service tail takes a larger share
    of the run as the arrival span shrinks. Reconstructing the same cell with
    zero queueing and its own measured tail reproduces the observed ratio to
    within a fraction of a percent at every rung: there is nothing left for
    saturation to explain.
    """

    cells = _on_disk_cells(PERF032_RUN, PERF032_RATES)

    report = comparator.compare_open_loop(cells, PERF032_RATES, 32, EIGHT_LANES)

    top = report["saturation"]["rayline_arc"]["r120"]
    assert top["saturated"] is False
    assert top["completion_ratio"] < 0.78
    # `unqueued_completion_ratio` charges the cell exactly one service-p95
    # request of drain. Where the last request to finish really was a p95 one
    # -- `r090` and `r120`, the two rungs read as saturated -- it reproduces
    # the observed ratio outright. `r045` and `r060` ended on a faster request
    # than p95, so the model over-charges them and reads low; that is the
    # model being conservative, not the cell being loaded.
    for label in ("r090", "r120"):
        cell = report["saturation"]["rayline_arc"][label]
        assert cell["unqueued_completion_ratio"] == pytest.approx(
            cell["completion_ratio"], rel=1e-3
        )
    for label in ("r045", "r060"):
        cell = report["saturation"]["rayline_arc"][label]
        assert cell["unqueued_completion_ratio"] < cell["completion_ratio"]


def test_saturation_block_is_absent_without_a_contracted_criterion() -> None:
    report = comparator.compare_open_loop(_cells())

    assert report["schema_version"] == comparator.REPORT_SCHEMA
    assert "saturation" not in report
    assert "first_saturated_cell" not in report


def test_comparison_case_count_comes_from_the_caller_not_the_module() -> None:
    """A comparator constant must not decide whether a run passed.

    `MEASURED_CASES = 32` was read here and written by the packet builder, so
    a sweep over any other corpus size would have been recorded as
    `passed: False` having completed every case it had. The count now arrives
    from the contract the packet was built and validated against.
    """

    cells = _cells()

    with pytest.raises(comparator.OpenLoopComparisonError):
        comparator.compare_open_loop(cells, None, 31)

    assert comparator.compare_open_loop(cells, None, 32)["status"] == "passed"
