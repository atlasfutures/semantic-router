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
PERF033_RUN = PARITY_ROOT / "rayline-saturation-knee-perf033-20260810"
PERF033_RATES = (1.20, 1.60, 2.20, 3.20)
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


def test_plateau_block_is_absent_without_a_contracted_gain() -> None:
    """An occupancy-only criterion must keep producing the exact v3 report.

    PERF033 closed under v3, so the plateau firing point has to be armed by
    the contract, never implied by the criterion's existence.
    """

    report = comparator.compare_open_loop(_cells(), None, 32, EIGHT_LANES)

    assert report["schema_version"] == comparator.SATURATION_REPORT_SCHEMA
    assert "throughput_plateau" not in report
    assert "first_throughput_plateau_cell" not in report
    assert "throughput_plateau_gain" not in report["saturation_criterion"]


def test_perf033s_recorded_v3_report_replays_byte_identically() -> None:
    """PERF033's own receipts must still produce PERF033's own report."""

    cells = _on_disk_cells(PERF033_RUN, PERF033_RATES)
    recorded = (PERF033_RUN / "comparison.json").read_text()

    replayed = comparator.compare_open_loop(
        cells, PERF033_RATES, 32, EIGHT_LANES
    )

    assert replayed["schema_version"] == comparator.SATURATION_REPORT_SCHEMA
    assert json.dumps(replayed, indent=2, sort_keys=True) + "\n" == recorded


PERF034_RATES = (1.20, 2.40, 4.00, 6.45)
THIRTY_TWO_LANES = contract.SaturationCriterion(
    episode_lanes=32, occupancy_ratio=1.0, throughput_plateau_gain=1 / 3
)


def _capacity_cells(
    throughputs: tuple[float, ...],
) -> dict[str, dict[str, dict[str, object]]]:
    """Cells whose throughput series is chosen while occupancy stays low."""

    cells = _cells(PERF034_RATES)
    for rate, throughput in zip(PERF034_RATES, throughputs, strict=True):
        for arm in comparator.OPEN_LOOP_ARMS:
            results = cells[packet.rate_label(rate)][arm]["results"]
            results["completion_throughput_rps"] = throughput
            results["max_client_backlog"] = 6
    return cells


def test_throughput_plateau_fires_where_added_load_stops_completing() -> None:
    """The plateau must fire while the occupancy criterion correctly stays silent.

    This is the PERF034 scenario the second firing point exists for: the
    encoder binds at six of 32 lanes, so occupancy never approaches its
    ceiling and `first_saturated_cell` stays `None` -- and that silence is
    exactly why the plateau has to be an independent criterion.
    """

    # Throughput tracks realized arrivals to `r240`, then adds 0.2 against
    # 1.6 additional arrivals per second: a marginal gain of 0.125.
    cells = _capacity_cells((1.2, 2.4, 2.6, 2.65))

    report = comparator.compare_open_loop(
        cells, PERF034_RATES, 32, THIRTY_TWO_LANES
    )

    assert report["schema_version"] == comparator.PLATEAU_REPORT_SCHEMA
    assert report["saturation_criterion"]["throughput_plateau_gain"] == 1 / 3
    assert report["first_saturated_cell"] == {
        "rayline_arc": None,
        "rayline_remote": None,
    }
    assert report["first_throughput_plateau_cell"] == {
        "rayline_arc": "r400",
        "rayline_remote": "r400",
    }
    arc = report["throughput_plateau"]["rayline_arc"]
    # The anchor is the baseline: a slope needs two points.
    assert arc["r120"]["marginal_throughput_gain"] is None
    assert arc["r120"]["implied_residence_delta_seconds"] is None
    assert arc["r120"]["plateaued"] is False
    assert arc["r240"]["marginal_throughput_gain"] == pytest.approx(1.0)
    assert arc["r240"]["plateaued"] is False
    assert arc["r400"]["marginal_throughput_gain"] == pytest.approx(0.125)
    assert arc["r400"]["plateaued"] is True
    assert arc["r645"]["plateaued"] is True


def test_plateau_floor_spares_perf032s_unqueued_ladder() -> None:
    """PERF032 never queued, and the plateau must not read its drain arithmetic.

    This is the calibration that moved the floor off the intuitive `0.5`:
    PERF032's provably unqueued top rung converts only `0.51`/`0.46` of its
    additional arrivals because the shrinking span raises a fixed service
    tail's share of the run. The recorded, non-voting drain-corrected gain
    says so -- it stays near `1.0` at every rung.
    """

    cells = _on_disk_cells(PERF032_RUN, PERF032_RATES)
    criterion = contract.SaturationCriterion(
        episode_lanes=8, occupancy_ratio=1.0, throughput_plateau_gain=1 / 3
    )

    report = comparator.compare_open_loop(cells, PERF032_RATES, 32, criterion)

    assert report["first_throughput_plateau_cell"] == {
        "rayline_arc": None,
        "rayline_remote": None,
    }
    for arm in comparator.OPEN_LOOP_ARMS:
        for label in ("r060", "r090", "r120"):
            cell = report["throughput_plateau"][arm][label]
            assert cell["marginal_throughput_gain"] > 1 / 3
            assert cell["drain_corrected_marginal_gain"] == pytest.approx(
                1.0, abs=0.07
            )


def test_plateau_fires_alongside_occupancy_on_the_saturated_control() -> None:
    """PERF021's rig-bound knee trips both firing points at the same rung.

    The two criteria disagree only when the encoder binds below the rig's
    ceiling; when the rig itself binds, added arrivals stop completing at the
    same rung the lanes pin, and both must say `r030`.
    """

    cells = _on_disk_cells(PERF021_RUN, packet.OFFERED_RATES)
    criterion = contract.SaturationCriterion(
        episode_lanes=8, occupancy_ratio=1.0, throughput_plateau_gain=1 / 3
    )

    report = comparator.compare_open_loop(
        cells, packet.OFFERED_RATES, 32, criterion
    )

    assert report["first_saturated_cell"] == {
        "rayline_arc": "r030",
        "rayline_remote": "r030",
    }
    assert report["first_throughput_plateau_cell"] == {
        "rayline_arc": "r030",
        "rayline_remote": "r030",
    }


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
