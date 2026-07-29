# SPDX-License-Identifier: Apache-2.0

"""Numerical comparisons and frozen budgets for the Rung B CUDA canary."""

from __future__ import annotations

import math
from typing import Any

from modal_canary_runtime import EMBEDDING_DIMENSION, SHAPES

RUNG_B_NUMERIC_BUDGETS = {
    # Frozen from rayline-arc-rung-b-raw-20260728-attempt3 on one H100 BF16,
    # with 20% explicit headroom rounded away from the observed maxima.
    "raw_mean_max_abs": 0.77,
    "raw_mean_l2": 6.35,
    "raw_mean_cosine_distance": 0.00325,
    "embedding_max_abs": 0.0105,
    "embedding_l2": 0.089,
    "cosine_distance": 0.00325,
    "synthetic_score_max_abs": 0.00175,
    "synthetic_top_two_gap_drift": 0.005,
}


def _synthetic_scores(embedding: list[float]) -> list[float]:
    scale = 1.0 / math.sqrt(EMBEDDING_DIMENSION)
    return [
        math.fsum(
            value * math.sin((arm + 1) * (index + 1) * 0.017)
            for index, value in enumerate(embedding)
        )
        * scale
        for arm in range(4)
    ]


def _synthetic_score_delta(
    reference: list[float],
    candidate: list[float],
) -> dict[str, Any]:
    left = _synthetic_scores(reference)
    right = _synthetic_scores(candidate)
    ordered = sorted(left, reverse=True)
    candidate_ordered = sorted(right, reverse=True)
    return {
        "max_abs": max(abs(a - b) for a, b in zip(left, right, strict=True)),
        "reference_top_two_gap": ordered[0] - ordered[1],
        "candidate_top_two_gap": candidate_ordered[0] - candidate_ordered[1],
        "top_two_gap_drift": abs(
            (ordered[0] - ordered[1]) - (candidate_ordered[0] - candidate_ordered[1])
        ),
        "selected_arm_reference": max(range(len(left)), key=left.__getitem__),
        "selected_arm_candidate": max(range(len(right)), key=right.__getitem__),
    }


def _numeric_delta(reference: list[float], candidate: list[float]) -> dict[str, Any]:
    differences = [
        right - left for left, right in zip(reference, candidate, strict=True)
    ]
    dot = math.fsum(
        left * right for left, right in zip(reference, candidate, strict=True)
    )
    reference_norm = math.sqrt(math.fsum(value * value for value in reference))
    candidate_norm = math.sqrt(math.fsum(value * value for value in candidate))
    return {
        "max_abs": max(abs(value) for value in differences),
        "l2": math.sqrt(math.fsum(value * value for value in differences)),
        "cosine_distance": abs(1.0 - dot / (reference_norm * candidate_norm)),
        "synthetic_scores": _synthetic_score_delta(reference, candidate),
    }


def summarize_numeric(
    full_vectors: dict[str, list[list[float]]],
    chunked_vectors: dict[str, list[list[float]]],
    full_raw_means: dict[str, list[float]] | None = None,
    chunked_raw_means: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    repeat_deltas: list[dict[str, Any]] = []
    for mode, vectors_by_shape in (
        ("single_schedule", full_vectors),
        ("chunked_8192", chunked_vectors),
    ):
        for shape, vectors in vectors_by_shape.items():
            for repetition, candidate in enumerate(vectors[1:], start=1):
                repeat_deltas.append(
                    {
                        "mode": mode,
                        "shape": shape,
                        "reference_repetition": 0,
                        "candidate_repetition": repetition,
                        **_numeric_delta(vectors[0], candidate),
                    }
                )

    cross_mode = [
        {
            "shape": shape,
            **_numeric_delta(full_vectors[shape][0], chunked_vectors[shape][0]),
        }
        for shape in SHAPES
        if shape in full_vectors and shape in chunked_vectors
    ]
    all_deltas = [*repeat_deltas, *cross_mode]
    if any(
        row["synthetic_scores"]["selected_arm_reference"]
        != row["synthetic_scores"]["selected_arm_candidate"]
        for row in all_deltas
    ):
        raise RuntimeError("pooling canary changed the synthetic selected arm")

    summary = {
        "repeat_deltas": repeat_deltas,
        "cross_mode_deltas": cross_mode,
        "observed_maxima": {
            "embedding_max_abs": max(row["max_abs"] for row in all_deltas),
            "embedding_l2": max(row["l2"] for row in all_deltas),
            "cosine_distance": max(row["cosine_distance"] for row in all_deltas),
            "synthetic_score_max_abs": max(
                row["synthetic_scores"]["max_abs"] for row in all_deltas
            ),
            "synthetic_top_two_gap_drift": max(
                row["synthetic_scores"]["top_two_gap_drift"] for row in all_deltas
            ),
            "synthetic_min_top_two_gap": min(
                row["synthetic_scores"]["reference_top_two_gap"] for row in all_deltas
            ),
            "selected_arm_parity": 1.0,
        },
    }
    if full_raw_means is not None and chunked_raw_means is not None:
        raw_mean_cross_mode = []
        for shape in SHAPES:
            if shape not in full_raw_means or shape not in chunked_raw_means:
                continue
            delta = _numeric_delta(
                full_raw_means[shape],
                chunked_raw_means[shape],
            )
            delta.pop("synthetic_scores")
            raw_mean_cross_mode.append({"shape": shape, **delta})
        summary["raw_mean_cross_mode_deltas"] = raw_mean_cross_mode
        summary["observed_maxima"].update(
            {
                "raw_mean_max_abs": max(row["max_abs"] for row in raw_mean_cross_mode),
                "raw_mean_l2": max(row["l2"] for row in raw_mean_cross_mode),
                "raw_mean_cosine_distance": max(
                    row["cosine_distance"] for row in raw_mean_cross_mode
                ),
            }
        )
    return summary


def enforce_rung_b_numeric_budgets(summary: dict[str, Any]) -> None:
    observed = summary["observed_maxima"]
    violations = [
        f"{metric}={observed[metric]:.9g} > {budget:.9g}"
        for metric, budget in RUNG_B_NUMERIC_BUDGETS.items()
        if observed[metric] > budget
    ]
    if violations:
        raise RuntimeError("Rung B numerical budget exceeded: " + "; ".join(violations))
