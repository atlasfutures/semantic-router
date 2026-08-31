#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Separate AGT018 semantic coverage from static three-model serving proof."""

from __future__ import annotations

from typing import Any

from openrouter_agentic_workload import PROVIDER_NAMES, WORKERS
from openrouter_kv_cache_successor_workload import (
    EXPECTED_SELECTION_TRACES,
)
from openrouter_kv_cache_successor_workload import (
    SCHEMA_VERSION as SUCCESSOR_WORKLOAD_SCHEMA_VERSION,
)

SCHEMA_VERSION = "rayline.openrouter-kv-cache-workload.v2"
STATIC_SERVING_CELLS = tuple(
    {
        "worker": worker,
        "model": model,
        "allowed_providers": tuple(PROVIDER_NAMES[worker]),
    }
    for worker, model in WORKERS.items()
)
SERVER_RETRYABLE_STATUSES = (429, 503)
SERVER_MAX_RETRIES = 1


def contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "applies_to": "AGT018 successor only; AGT017 remains historical",
        "semantic_cache_lane": {
            "routing": "natural_rayline_selection",
            "workload_schema_version": SUCCESSOR_WORKLOAD_SCHEMA_VERSION,
            "expected_selection_traces": {
                sequence: list(trace)
                for sequence, trace in EXPECTED_SELECTION_TRACES.items()
            },
            "claims": [
                "retained_vs_replay_cache_effect",
                "cross_architecture_selection_parity",
                "three_worker_semantic_coverage",
            ],
            "three_worker_coverage_required": True,
            "offline_native_coverage_required": True,
            "remote_encoder_parity_required_before_provider_measurement": True,
        },
        "stratified_serving_lane": {
            "routing": "explicit_static_worker",
            "cells": [
                {
                    **cell,
                    "allowed_providers": list(cell["allowed_providers"]),
                }
                for cell in STATIC_SERVING_CELLS
            ],
            "claims": ["provider_availability", "three_model_serving_coverage"],
            "semantic_selection_claim_admissible": False,
        },
        "retry_policy": {
            "owner": {
                "native_modal": "Pathfinder OpenRouter worker transport",
                "remote_vllm": "Envoy OpenRouter route",
            },
            "retryable_statuses": list(SERVER_RETRYABLE_STATUSES),
            "maximum_retries": SERVER_MAX_RETRIES,
            "benchmark_client_retries": 0,
            "reason": "keep retries below one semantic selection transaction",
        },
    }


def validate() -> dict[str, Any]:
    value = contract()
    cells = value["stratified_serving_lane"]["cells"]
    if (
        len(cells) != len(WORKERS)
        or {cell["worker"] for cell in cells} != set(WORKERS)
        or any(cell["model"] != WORKERS[cell["worker"]] for cell in cells)
        or any(
            tuple(cell["allowed_providers"]) != tuple(PROVIDER_NAMES[cell["worker"]])
            for cell in cells
        )
    ):
        raise RuntimeError("KV workload three-model serving coverage diverged")
    if value["stratified_serving_lane"]["semantic_selection_claim_admissible"]:
        raise RuntimeError("static serving coverage was mislabeled as semantic")
    semantic = value["semantic_cache_lane"]
    observed = {
        worker
        for trace in semantic["expected_selection_traces"].values()
        for worker in trace
    }
    if not semantic["three_worker_coverage_required"] or observed != set(WORKERS):
        raise RuntimeError("AGT018 natural semantic coverage diverged")
    retry = value["retry_policy"]
    if (
        retry["retryable_statuses"] != list(SERVER_RETRYABLE_STATUSES)
        or retry["maximum_retries"] != SERVER_MAX_RETRIES
        or retry["benchmark_client_retries"] != 0
    ):
        raise RuntimeError("KV workload retry ownership diverged")
    return value
