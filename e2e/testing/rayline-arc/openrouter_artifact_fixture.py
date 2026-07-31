#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Generate the public three-arm ARC artifact for the OpenRouter canary."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from artifact_fixture import (
    ARM_DIMENSION,
    HIDDEN_DIMENSION,
    HISTORY_DIMENSION,
    JOINT_DIMENSION,
    MODEL_REVISION,
    SERIALIZER,
    Tensor,
    _arm_axis,
    _encode_safetensors,
    _sha256,
)
from modal_fullstack_inputs import ROUTING_AXIS_INDEX

ARTIFACT_ID = "public-rayline-arc-openrouter-v1"
EXPECTED_ARGUMENT_COUNT = 2
TARGET_AXIS_VALUE = 0.006
PROVIDER_SLUG = "fireworks"
PROVIDER_NAME = "Fireworks"
MAX_COMPLETION_TOKENS = 8
WORKERS = (
    {
        "id": "worker-a",
        "model": "deepseek/deepseek-v4-flash",
        "prompt_cost": 0.00000014,
        "cache_read_cost": 0.000000028,
        "completion_cost": 0.00000028,
    },
    {
        "id": "worker-b",
        "model": "moonshotai/kimi-k3",
        "prompt_cost": 0.000003,
        "cache_read_cost": 0.0000003,
        "completion_cost": 0.000015,
    },
    {
        "id": "worker-c",
        "model": "z-ai/glm-5.2",
        "prompt_cost": 0.0000014,
        "cache_read_cost": 0.00000014,
        "completion_cost": 0.0000044,
    },
)


def _tensors() -> dict[str, Tensor]:
    candidate_offset = HISTORY_DIMENSION
    residual_offset = 32
    arm_scale = TARGET_AXIS_VALUE / _arm_axis()
    return {
        "model_encoder.meta_mean": Tensor((8,)),
        "model_encoder.meta_std": Tensor((8,), fill=1),
        "model_encoder.all_metas": Tensor((len(WORKERS), 8)),
        "model_encoder.meta_mlp.0.weight": Tensor((32, 8)),
        "model_encoder.meta_mlp.0.bias": Tensor((32,)),
        "model_encoder.meta_mlp.2.weight": Tensor((32, 32)),
        "model_encoder.meta_mlp.2.bias": Tensor((32,)),
        "model_encoder.residual_embed.weight": Tensor(
            (len(WORKERS), 16),
            sparse={0: 1, 32: -1},
        ),
        "model_encoder.output_proj.0.weight": Tensor(
            (ARM_DIMENSION, 48),
            sparse={residual_offset: 1},
        ),
        "model_encoder.output_proj.0.bias": Tensor((ARM_DIMENSION,)),
        "model_encoder.output_proj.1.weight": Tensor(
            (ARM_DIMENSION,),
            fill=1,
        ),
        "model_encoder.output_proj.1.bias": Tensor((ARM_DIMENSION,)),
        "q_network.backbone.0.weight": Tensor(
            (HIDDEN_DIMENSION, JOINT_DIMENSION),
            sparse={
                ROUTING_AXIS_INDEX: 1,
                candidate_offset: -arm_scale,
                JOINT_DIMENSION + ROUTING_AXIS_INDEX: -1,
                JOINT_DIMENSION + candidate_offset: arm_scale,
            },
        ),
        "q_network.backbone.0.bias": Tensor((HIDDEN_DIMENSION,)),
        "q_network.backbone.3.weight": Tensor(
            (HIDDEN_DIMENSION, HIDDEN_DIMENSION),
            sparse={0: 1, HIDDEN_DIMENSION + 1: 1},
        ),
        "q_network.backbone.3.bias": Tensor((HIDDEN_DIMENSION,)),
        "q_network.head.weight": Tensor(
            (1, HIDDEN_DIMENSION),
            sparse={0: -1, 1: -1},
        ),
        "q_network.head.bias": Tensor((1,)),
    }


def _embedding(axis_value: float) -> list[float]:
    embedding = [0.0] * HISTORY_DIMENSION
    embedding[ROUTING_AXIS_INDEX] = axis_value
    if axis_value == 0:
        embedding[0] = 1
    return embedding


def _scores(axis_value: float) -> list[float]:
    return [
        -abs(axis_value - TARGET_AXIS_VALUE),
        -abs(axis_value),
        -abs(axis_value + TARGET_AXIS_VALUE),
    ]


def _golden(checkpoint_sha256: str) -> dict[str, object]:
    return {
        "schema_version": "rayline.mtrouter-head-golden.v1",
        "checkpoint_sha256": checkpoint_sha256,
        "score_tolerance": 0.001,
        "pool": [worker["id"] for worker in WORKERS],
        "cases": [
            {
                "id": "route-positive",
                "embedding": _embedding(1),
                "previous_model_index": -1,
                "route_call_index": 0,
                "scores": _scores(1),
                "selected_index": 0,
            },
            {
                "id": "route-middle",
                "embedding": _embedding(0),
                "previous_model_index": -1,
                "route_call_index": 0,
                "scores": _scores(0),
                "selected_index": 1,
            },
            {
                "id": "route-negative",
                "embedding": _embedding(-1),
                "previous_model_index": -1,
                "route_call_index": 0,
                "scores": _scores(-1),
                "selected_index": 2,
            },
        ],
    }


def _worker_contract(worker: dict[str, object]) -> dict[str, object]:
    return {
        "id": worker["id"],
        "model": worker["model"],
        "api_key_env": "SYNTHETIC_API_KEY",
        "estimated_input_cost_per_token": worker["prompt_cost"],
        "estimated_cache_read_cost_per_token": worker["cache_read_cost"],
        "estimated_cache_write_cost_per_token": worker["prompt_cost"],
        "estimated_output_cost_per_token": worker["completion_cost"],
        "latency_ms": 1000,
        "capability_tags": ["public-openrouter-canary"],
        "openrouter_provider_slug": PROVIDER_SLUG,
        "openrouter_provider_name": PROVIDER_NAME,
        "openrouter_provider_order": [PROVIDER_SLUG],
        "openrouter_allow_fallbacks": False,
        "openrouter_require_parameters": True,
        "openrouter_pricing_source": "openrouter-fireworks-2026-07-31",
        "thinking_mode": "off",
        "reasoning_budget_tokens": 0,
        "minimum_completion_tokens": MAX_COMPLETION_TOKENS,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "temperature": 0,
        "supports_output_effort": False,
        "extra_body": {"reasoning": {"enabled": False, "effort": "none"}},
        "openrouter_max_retries": 1,
        "openrouter_retry_base_seconds": 0.1,
        "openrouter_retry_cap_seconds": 0.2,
        "attempt_deadline_seconds": 120,
    }


def _manifest(weights: bytes, golden: bytes) -> dict[str, object]:
    checkpoint_sha256 = hashlib.sha256(
        b"public synthetic ARC OpenRouter checkpoint"
    ).hexdigest()
    return {
        "schema_version": "rayline.mtrouter-runtime.v3",
        "artifact_id": ARTIFACT_ID,
        "created_at": "2026-07-31T00:00:00Z",
        "exporter_commit": "public-openrouter-canary-exporter-v1",
        "weights": {
            "file": "head.safetensors",
            "sha256": _sha256(weights),
            "dtype": "F32",
        },
        "source": {
            "checkpoint": {
                "file": "synthetic-openrouter-checkpoint.pt",
                "sha256": checkpoint_sha256,
                "modal_volume": "public-synthetic",
                "modal_path": "/public/synthetic-openrouter-checkpoint.pt",
            },
            "model_meta": {"sha256": "4" * 64},
            "reliability_table": {"sha256": "5" * 64},
            "train_report": {"sha256": "6" * 64},
        },
        "encoder": {
            "model": "Qwen/Qwen3.5-0.8B",
            "revision": MODEL_REVISION,
            "dimension": HISTORY_DIMENSION,
            "max_tokens": 262144,
            "min_recent_turns": 1,
            "min_recent_tokens": 64,
            "serialization": SERIALIZER,
            "pooling": "masked_mean",
            "normalize_embeddings": True,
            "attention_implementation": "sdpa",
            "dtype": "BF16",
            "incremental_default": True,
            "kv_chunk_tokens": 8192,
            "kv_session_budget_tokens": 262144,
            "kv_process_budget_tokens": 524288,
            "kv_idle_ttl_seconds": 900,
            "native": {"runtime": "public-openrouter-canary"},
        },
        "architecture": {
            "name": "switch_aware",
            "history_dimension": HISTORY_DIMENSION,
            "arm_embedding_dimension": ARM_DIMENSION,
            "joint_input_dimension": JOINT_DIMENSION,
            "hidden_dimensions": [HIDDEN_DIMENSION, HIDDEN_DIMENSION],
            "dropout": 0.1,
            "pool": [worker["id"] for worker in WORKERS],
        },
        "policy": {
            "previous_worker_stay_margin": 0,
            "cold_switch_margin_per_usd": 0,
            "cold_switch_upgrade_exempt": False,
            "stay_margin_upgrade_exempt": False,
            "reference_worker": "worker-a",
            "reference_margin": 0,
        },
        "workers": [_worker_contract(worker) for worker in WORKERS],
        "pricing_snapshot": {
            "config_path": "openrouter-fireworks-endpoints-2026-07-31",
            "config_commit": "public-openrouter-pricing-v1",
            "mutable_live_prices_affect_decisions": False,
        },
        "golden": {
            "head": {
                "file": "head_golden.json",
                "sha256": _sha256(golden),
                "score_tolerance": 0.001,
            },
            "adjusted_top_two_gap_tolerance": 0.001,
            "required_selection_parity": 1,
        },
    }


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = _encode_safetensors(_tensors())
    checkpoint_sha256 = hashlib.sha256(
        b"public synthetic ARC OpenRouter checkpoint"
    ).hexdigest()
    golden = json.dumps(_golden(checkpoint_sha256), indent=2, sort_keys=True).encode()
    manifest = json.dumps(_manifest(weights, golden), indent=2, sort_keys=True).encode()
    (output_dir / "head.safetensors").write_bytes(weights)
    (output_dir / "head_golden.json").write_bytes(golden)
    (output_dir / "manifest.json").write_bytes(manifest)


if __name__ == "__main__":
    if len(sys.argv) != EXPECTED_ARGUMENT_COUNT:
        raise SystemExit("usage: openrouter_artifact_fixture.py OUTPUT_DIR")
    generate(Path(sys.argv[1]))
