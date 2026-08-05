#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Matched-pair cross-deployment comparability lanes for AGT019.

AGT018d proved that pinned multi-provider fallthrough — required to survive
upstream saturation — lets the two architecture arms land on different
serving providers for the same worker, which breaks whole-set completion
matching and with it any unconditioned cross-deployment E2E claim. This
module implements the AGT019 policy: join the arms' identical measurement
cells, partition the pairs into explicit comparability lanes, and label each
worker's cross-deployment E2E evidence admissible only when at least one
fully matched pair (same served provider and same completion token count)
exists. The policy gate verifies the partition is complete and honestly
labeled; it never depends on providers happening to agree.
"""

from __future__ import annotations

import math
from typing import Any

MINIMUM_FULLY_MATCHED_PAIRS_PER_WORKER = 1


def _cell_key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row["sequence_id"]),
        str(row["mode"]),
        int(row["episode"]),
        int(row["step"]),
    )


def pair_cells(
    native_results: list[dict[str, Any]],
    remote_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join both arms' measurement cells one-to-one.

    Every cell must exist exactly once on each arm with a recorded selected
    worker, served provider, and completion token count; anything less is an
    evidence defect, not a comparability outcome.
    """

    native_by_key = {_cell_key(row): row for row in native_results}
    remote_by_key = {_cell_key(row): row for row in remote_results}
    if len(native_by_key) != len(native_results) or len(remote_by_key) != len(
        remote_results
    ):
        raise RuntimeError("AGT019 matched-pair join found duplicate cells")
    if native_by_key.keys() != remote_by_key.keys():
        raise RuntimeError("AGT019 matched-pair join found unpaired cells")
    pairs = []
    for key in sorted(native_by_key):
        native_row = native_by_key[key]
        remote_row = remote_by_key[key]
        worker = str(native_row.get("selected_worker") or "")
        if not worker or worker != str(remote_row.get("selected_worker") or ""):
            raise RuntimeError("AGT019 matched-pair join found selection divergence")
        native_provider = str(native_row.get("provider") or "")
        remote_provider = str(remote_row.get("provider") or "")
        if not native_provider or not remote_provider:
            raise RuntimeError("AGT019 matched-pair join missing served provider")
        native_completion = int(native_row.get("completion_tokens") or -1)
        remote_completion = int(remote_row.get("completion_tokens") or -1)
        if native_completion < 0 or remote_completion < 0:
            raise RuntimeError("AGT019 matched-pair join missing completion tokens")
        completion_matched = native_completion == remote_completion
        provider_matched = native_provider == remote_provider
        pairs.append(
            {
                "sequence_id": key[0],
                "mode": key[1],
                "episode": key[2],
                "step": key[3],
                "worker": worker,
                "native_provider": native_provider,
                "remote_provider": remote_provider,
                "native_completion_tokens": native_completion,
                "remote_completion_tokens": remote_completion,
                "completion_matched": completion_matched,
                "provider_matched": provider_matched,
                "fully_matched": completion_matched and provider_matched,
                "native_total_seconds": float(native_row["total_seconds"]),
                "remote_total_seconds": float(remote_row["total_seconds"]),
            }
        )
    return pairs


def _lane_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if not pairs:
        return {"pairs": 0}
    native_mean = math.fsum(pair["native_total_seconds"] for pair in pairs) / len(pairs)
    remote_mean = math.fsum(pair["remote_total_seconds"] for pair in pairs) / len(pairs)
    return {
        "pairs": len(pairs),
        "native_e2e_mean_seconds": native_mean,
        "remote_e2e_mean_seconds": remote_mean,
        "remote_to_native_e2e_mean_ratio": (
            remote_mean / native_mean if native_mean > 0 else None
        ),
    }


def comparability_report(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Partition pairs into lanes and label per-worker E2E admissibility."""

    workers = sorted({pair["worker"] for pair in pairs})
    per_worker: dict[str, Any] = {}
    for worker in workers:
        worker_pairs = [pair for pair in pairs if pair["worker"] == worker]
        fully = [pair for pair in worker_pairs if pair["fully_matched"]]
        completion_only = [pair for pair in worker_pairs if pair["completion_matched"]]
        per_worker[worker] = {
            "total_pairs": len(worker_pairs),
            "fully_matched": _lane_summary(fully),
            "completion_matched": _lane_summary(completion_only),
            "native_providers": sorted(
                {pair["native_provider"] for pair in worker_pairs}
            ),
            "remote_providers": sorted(
                {pair["remote_provider"] for pair in worker_pairs}
            ),
            "cross_deployment_e2e_admissible": (
                len(fully) >= MINIMUM_FULLY_MATCHED_PAIRS_PER_WORKER
            ),
        }
    fully_all = [pair for pair in pairs if pair["fully_matched"]]
    return {
        "policy": (
            "cross-deployment E2E and first-token claims are admissible only "
            "over fully matched pairs (same served provider and completion "
            "token count); unmatched workers are labeled inadmissible rather "
            "than failing the run"
        ),
        "minimum_fully_matched_pairs_per_worker": (
            MINIMUM_FULLY_MATCHED_PAIRS_PER_WORKER
        ),
        "total_pairs": len(pairs),
        "fully_matched_pairs": len(fully_all),
        "fully_matched_coverage_fraction": (
            len(fully_all) / len(pairs) if pairs else 0.0
        ),
        "admissible_lane": _lane_summary(fully_all),
        "per_worker": per_worker,
        "inadmissible_workers": [
            worker
            for worker in workers
            if not per_worker[worker]["cross_deployment_e2e_admissible"]
        ],
    }


def policy_gate(report: dict[str, Any], expected_total_pairs: int) -> bool:
    """True when every cell was paired, recorded, and honestly labeled.

    The gate never requires provider agreement: a worker with zero fully
    matched pairs passes as long as it is listed as inadmissible.
    """

    per_worker = report.get("per_worker") or {}
    labeled_inadmissible = set(report.get("inadmissible_workers") or [])
    if report.get("total_pairs") != expected_total_pairs or not per_worker:
        return False
    for worker, lane in per_worker.items():
        admissible = lane.get("cross_deployment_e2e_admissible") is True
        if admissible == (worker in labeled_inadmissible):
            return False
        fully = lane.get("fully_matched") or {}
        if admissible and fully.get("pairs", 0) < (
            MINIMUM_FULLY_MATCHED_PAIRS_PER_WORKER
        ):
            return False
    return True
