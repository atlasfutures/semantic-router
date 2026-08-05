# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from typing import Any

if "modal" not in sys.modules and importlib.util.find_spec("modal") is None:
    modal_stub = types.ModuleType("modal")
    modal_stub.__spec__ = importlib.machinery.ModuleSpec("modal", loader=None)
    sys.modules["modal"] = modal_stub

import openrouter_kv_cache_agt019_contract as agt019_contract
import openrouter_kv_cache_matched_pair as matched_pair
import pytest

EXPECTED_REQUESTS_PER_DEPLOYMENT = 36
EXPECTED_PREFLIGHT_REQUESTS = 6
EXPECTED_MEASUREMENT_REQUESTS = 72
EXPECTED_MAXIMUM_LOGICAL_PROVIDER_REQUESTS = 78
EXPECTED_MAXIMUM_EXTERNAL_ATTEMPTS = 156
EXPECTED_MAXIMUM_COMPLETION_TOKENS = 24

SEQUENCE_WORKERS = (
    ("code_patch", "worker-a"),
    ("doc_edit", "worker-b"),
    ("data_query", "worker-c"),
)
STEPS_PER_SEQUENCE = 2
MODE = "retained"
EPISODE = 0

DIVERGED_WORKER = "worker-b"
BASE_PROVIDER = "GMICloud"
DIVERGED_REMOTE_PROVIDER = "Venice"
BASE_COMPLETION_TOKENS = 18
DIVERGED_REMOTE_COMPLETION_TOKENS = 11
NATIVE_SECONDS = 2.0
REMOTE_SECONDS = 1.0

EXPECTED_TOTAL_PAIRS = len(SEQUENCE_WORKERS) * STEPS_PER_SEQUENCE
PAIRS_PER_WORKER = STEPS_PER_SEQUENCE
EXPECTED_MATCHED_PAIRS_WITH_ONE_DIVERGED_WORKER = (
    EXPECTED_TOTAL_PAIRS - PAIRS_PER_WORKER
)
NO_PAIRS = 0
FULL_COVERAGE = 1.0
EXPECTED_DIVERGED_COVERAGE = (
    EXPECTED_MATCHED_PAIRS_WITH_ONE_DIVERGED_WORKER / EXPECTED_TOTAL_PAIRS
)
EXPECTED_REMOTE_TO_NATIVE_RATIO = REMOTE_SECONDS / NATIVE_SECONDS


def _arm_rows(
    *,
    seconds: float,
    diverged_worker: str = "",
    drop_field: str = "",
) -> list[dict[str, Any]]:
    """Build one arm's measurement cells for the three-worker workload."""

    rows: list[dict[str, Any]] = []
    for sequence_id, worker in SEQUENCE_WORKERS:
        diverged = worker == diverged_worker
        for step in range(STEPS_PER_SEQUENCE):
            row: dict[str, Any] = {
                "sequence_id": sequence_id,
                "mode": MODE,
                "episode": EPISODE,
                "step": step,
                "selected_worker": worker,
                "provider": (DIVERGED_REMOTE_PROVIDER if diverged else BASE_PROVIDER),
                "completion_tokens": (
                    DIVERGED_REMOTE_COMPLETION_TOKENS
                    if diverged
                    else BASE_COMPLETION_TOKENS
                ),
                "total_seconds": seconds,
            }
            if drop_field:
                row.pop(drop_field)
            rows.append(row)
    return rows


def test_agt019_contract_is_source_closed_with_the_matched_pair_gate() -> None:
    contract = agt019_contract.validate()

    assert agt019_contract.PREREGISTRATION_COMMIT == ""
    assert agt019_contract.AUTHORIZATION_COMMIT == ""
    assert agt019_contract.SOURCE_CLOSED_KEY_LIMIT_USD_PER_ARM == 0
    assert agt019_contract.SOURCE_CLOSED_MAXIMUM_PAID_WALL_SECONDS == 0
    assert contract["source_closed"] is True
    assert contract["launch_authorized"] is False
    assert contract["requires_new_budget_authority"] is True
    assert contract["run_id"] == "rayline-openrouter-kv-cache-agt019-20260805"
    assert contract["report_schema_version"] == (
        "rayline.openrouter-kv-cache-comparison.v4"
    )
    assert "matched_pair_comparability_policy" in contract["acceptance_gates"]
    assert "matched_completion_policy" not in contract["acceptance_gates"]
    assert contract["logical_provider_requests"] == {
        "provider_preflight": EXPECTED_PREFLIGHT_REQUESTS,
        "semantic_cache_measurement": EXPECTED_MEASUREMENT_REQUESTS,
        "maximum_total": EXPECTED_MAXIMUM_LOGICAL_PROVIDER_REQUESTS,
    }
    assert contract["maximum_external_attempts"] == EXPECTED_MAXIMUM_EXTERNAL_ATTEMPTS
    assert contract["maximum_completion_tokens"] == EXPECTED_MAXIMUM_COMPLETION_TOKENS
    assert (
        agt019_contract.EXPECTED_SEMANTIC_REQUESTS_PER_DEPLOYMENT
        == EXPECTED_REQUESTS_PER_DEPLOYMENT
    )
    assert "budget" not in contract


def test_identical_arms_are_fully_matched_everywhere() -> None:
    pairs = matched_pair.pair_cells(
        _arm_rows(seconds=NATIVE_SECONDS),
        _arm_rows(seconds=REMOTE_SECONDS),
    )
    report = matched_pair.comparability_report(pairs)

    assert report["total_pairs"] == EXPECTED_TOTAL_PAIRS
    assert report["fully_matched_pairs"] == EXPECTED_TOTAL_PAIRS
    assert report["fully_matched_coverage_fraction"] == FULL_COVERAGE
    assert report["inadmissible_workers"] == []
    assert report["admissible_lane"]["pairs"] == EXPECTED_TOTAL_PAIRS
    assert report["admissible_lane"]["remote_to_native_e2e_mean_ratio"] == (
        EXPECTED_REMOTE_TO_NATIVE_RATIO
    )
    assert all(
        lane["cross_deployment_e2e_admissible"] is True
        for lane in report["per_worker"].values()
    )
    assert matched_pair.policy_gate(report, EXPECTED_TOTAL_PAIRS) is True


def test_provider_fallthrough_labels_one_worker_inadmissible_without_failing() -> None:
    pairs = matched_pair.pair_cells(
        _arm_rows(seconds=NATIVE_SECONDS),
        _arm_rows(seconds=REMOTE_SECONDS, diverged_worker=DIVERGED_WORKER),
    )
    report = matched_pair.comparability_report(pairs)
    diverged_lane = report["per_worker"][DIVERGED_WORKER]

    assert report["inadmissible_workers"] == [DIVERGED_WORKER]
    assert diverged_lane["cross_deployment_e2e_admissible"] is False
    assert diverged_lane["total_pairs"] == PAIRS_PER_WORKER
    assert diverged_lane["fully_matched"]["pairs"] == NO_PAIRS
    assert diverged_lane["completion_matched"]["pairs"] == NO_PAIRS
    assert diverged_lane["native_providers"] == [BASE_PROVIDER]
    assert diverged_lane["remote_providers"] == [DIVERGED_REMOTE_PROVIDER]
    assert report["fully_matched_pairs"] == (
        EXPECTED_MATCHED_PAIRS_WITH_ONE_DIVERGED_WORKER
    )
    assert report["fully_matched_coverage_fraction"] == EXPECTED_DIVERGED_COVERAGE
    assert report["admissible_lane"]["pairs"] == (
        EXPECTED_MATCHED_PAIRS_WITH_ONE_DIVERGED_WORKER
    )
    assert matched_pair.policy_gate(report, EXPECTED_TOTAL_PAIRS) is True


@pytest.mark.parametrize("field", ["provider", "completion_tokens"])
def test_missing_evidence_fields_are_defects_not_outcomes(field: str) -> None:
    with pytest.raises(RuntimeError, match="AGT019 matched-pair join"):
        matched_pair.pair_cells(
            _arm_rows(seconds=NATIVE_SECONDS),
            _arm_rows(seconds=REMOTE_SECONDS, drop_field=field),
        )


def test_unpaired_cells_are_rejected() -> None:
    remote = _arm_rows(seconds=REMOTE_SECONDS)[:-1]
    with pytest.raises(RuntimeError, match="unpaired cells"):
        matched_pair.pair_cells(_arm_rows(seconds=NATIVE_SECONDS), remote)


def test_duplicated_cells_are_rejected() -> None:
    native = _arm_rows(seconds=NATIVE_SECONDS)
    native.append(dict(native[0]))
    with pytest.raises(RuntimeError, match="duplicate cells"):
        matched_pair.pair_cells(native, _arm_rows(seconds=REMOTE_SECONDS))


def test_selection_divergence_is_rejected() -> None:
    remote = _arm_rows(seconds=REMOTE_SECONDS)
    remote[0] = {**remote[0], "selected_worker": "worker-c"}
    with pytest.raises(RuntimeError, match="selection divergence"):
        matched_pair.pair_cells(_arm_rows(seconds=NATIVE_SECONDS), remote)


def test_dishonest_labeling_fails_the_policy_gate() -> None:
    pairs = matched_pair.pair_cells(
        _arm_rows(seconds=NATIVE_SECONDS),
        _arm_rows(seconds=REMOTE_SECONDS, diverged_worker=DIVERGED_WORKER),
    )
    mislabeled = matched_pair.comparability_report(pairs)
    mislabeled["inadmissible_workers"] = sorted(mislabeled["per_worker"])

    assert matched_pair.policy_gate(mislabeled, EXPECTED_TOTAL_PAIRS) is False

    undercounted = matched_pair.comparability_report(pairs)
    undercounted["total_pairs"] = EXPECTED_TOTAL_PAIRS - 1

    assert matched_pair.policy_gate(undercounted, EXPECTED_TOTAL_PAIRS) is False
