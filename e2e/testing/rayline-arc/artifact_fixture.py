#!/usr/bin/env python3
"""Generate a public synthetic ARC artifact for hermetic stack tests."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import NamedTuple

ARTIFACT_ID = "public-rayline-arc-e2e-v1"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
SERIALIZER = "mtrouter-token-blocks-v2"
HISTORY_DIMENSION = 1024
ARM_DIMENSION = 64
JOINT_DIMENSION = 1154
HIDDEN_DIMENSION = 256
EXPECTED_ARGUMENT_COUNT = 2


class Tensor(NamedTuple):
    shape: tuple[int, ...]
    fill: float = 0.0
    sparse: dict[int, float] | None = None


def _product(shape: tuple[int, ...]) -> int:
    result = 1
    for value in shape:
        result *= value
    return result


def _tensor_bytes(tensor: Tensor) -> bytes:
    count = _product(tensor.shape)
    if tensor.fill == 0:
        result = bytearray(count * 4)
    else:
        encoded = struct.pack("<f", tensor.fill)
        result = bytearray(encoded * count)
    for index, value in (tensor.sparse or {}).items():
        if index < 0 or index >= count:
            raise ValueError(f"sparse tensor index {index} is out of bounds")
        struct.pack_into("<f", result, index * 4, value)
    return bytes(result)


def _tensors() -> dict[str, Tensor]:
    candidate_offset = HISTORY_DIMENSION
    residual_offset = 32
    return {
        "model_encoder.meta_mean": Tensor((8,)),
        "model_encoder.meta_std": Tensor((8,), fill=1),
        "model_encoder.all_metas": Tensor((2, 8)),
        "model_encoder.meta_mlp.0.weight": Tensor((32, 8)),
        "model_encoder.meta_mlp.0.bias": Tensor((32,)),
        "model_encoder.meta_mlp.2.weight": Tensor((32, 32)),
        "model_encoder.meta_mlp.2.bias": Tensor((32,)),
        "model_encoder.residual_embed.weight": Tensor(
            (2, 16),
            sparse={0: 1, 16: -1},
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
                0: 1,
                candidate_offset: 1,
                JOINT_DIMENSION: -1,
                JOINT_DIMENSION + candidate_offset: -1,
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
            sparse={0: 1, 1: 1},
        ),
        "q_network.head.bias": Tensor((1,)),
    }


def _encode_safetensors(tensors: dict[str, Tensor]) -> bytes:
    raw = bytearray()
    descriptors: dict[str, object] = {}
    for name in sorted(tensors):
        encoded = _tensor_bytes(tensors[name])
        start = len(raw)
        raw.extend(encoded)
        descriptors[name] = {
            "dtype": "F32",
            "shape": list(tensors[name].shape),
            "data_offsets": [start, len(raw)],
        }
    header = json.dumps(
        descriptors,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return struct.pack("<Q", len(header)) + header + raw


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _embedding(sign: int) -> list[float]:
    return [float(sign)] + [0.0] * (HISTORY_DIMENSION - 1)


def _arm_axis() -> float:
    mean = 1 / ARM_DIMENSION
    variance = ((1 - mean) ** 2 + (ARM_DIMENSION - 1) * mean**2) / ARM_DIMENSION
    return (1 - mean) / math.sqrt(variance + 1e-5)


def _golden(checkpoint_sha256: str) -> dict[str, object]:
    arm = _arm_axis()
    return {
        "schema_version": "rayline.mtrouter-head-golden.v1",
        "checkpoint_sha256": checkpoint_sha256,
        "score_tolerance": 0.001,
        "pool": ["worker-a", "worker-b"],
        "cases": [
            {
                "id": "route-a",
                "embedding": _embedding(1),
                "previous_model_index": -1,
                "route_call_index": 0,
                "scores": [arm + 1, arm - 1],
                "selected_index": 0,
            },
            {
                "id": "route-b",
                "embedding": _embedding(-1),
                "previous_model_index": -1,
                "route_call_index": 0,
                "scores": [arm - 1, arm + 1],
                "selected_index": 1,
            },
        ],
    }


def _worker(
    worker_id: str,
    model: str,
    provider: str,
    *,
    input_cost: float,
    cache_read_cost: float,
    cache_write_cost: float,
    output_cost: float,
    thinking: bool,
) -> dict[str, object]:
    reasoning: dict[str, object] = {"enabled": thinking}
    if thinking:
        reasoning["max_tokens"] = 64
    else:
        reasoning["effort"] = "none"
    return {
        "id": worker_id,
        "model": model,
        "api_key_env": "SYNTHETIC_API_KEY",
        "estimated_input_cost_per_token": input_cost,
        "estimated_cache_read_cost_per_token": cache_read_cost,
        "estimated_cache_write_cost_per_token": cache_write_cost,
        "estimated_output_cost_per_token": output_cost,
        "latency_ms": 10,
        "capability_tags": ["public-synthetic"],
        "openrouter_provider_slug": provider,
        "openrouter_provider_name": provider,
        "openrouter_provider_order": [provider],
        "openrouter_allow_fallbacks": False,
        "openrouter_require_parameters": True,
        "openrouter_pricing_source": "public-synthetic-e2e",
        "thinking_mode": "on" if thinking else "off",
        "reasoning_budget_tokens": 64 if thinking else 0,
        "minimum_completion_tokens": 32,
        "max_completion_tokens": 128,
        "temperature": 0.3 if thinking else 0.2,
        "supports_output_effort": False,
        "extra_body": {"reasoning": reasoning},
        "openrouter_max_retries": 1,
        "openrouter_retry_base_seconds": 0.1,
        "openrouter_retry_cap_seconds": 0.2,
        "attempt_deadline_seconds": 30,
    }


def _manifest(weights: bytes, golden: bytes) -> dict[str, object]:
    checkpoint_sha256 = _sha256(b"public synthetic ARC E2E checkpoint")
    return {
        "schema_version": "rayline.mtrouter-runtime.v3",
        "artifact_id": ARTIFACT_ID,
        "created_at": "2026-01-01T00:00:00Z",
        "exporter_commit": "public-synthetic-e2e-exporter-v1",
        "weights": {
            "file": "head.safetensors",
            "sha256": _sha256(weights),
            "dtype": "F32",
        },
        "source": {
            "checkpoint": {
                "file": "synthetic-checkpoint.pt",
                "sha256": checkpoint_sha256,
                "modal_volume": "public-synthetic",
                "modal_path": "/public/synthetic-checkpoint.pt",
            },
            "model_meta": {"sha256": "1" * 64},
            "reliability_table": {"sha256": "2" * 64},
            "train_report": {"sha256": "3" * 64},
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
            "native": {"runtime": "public-synthetic-e2e"},
        },
        "architecture": {
            "name": "switch_aware",
            "history_dimension": HISTORY_DIMENSION,
            "arm_embedding_dimension": ARM_DIMENSION,
            "joint_input_dimension": JOINT_DIMENSION,
            "hidden_dimensions": [HIDDEN_DIMENSION, HIDDEN_DIMENSION],
            "dropout": 0.1,
            "pool": ["worker-a", "worker-b"],
        },
        "policy": {
            "previous_worker_stay_margin": 0.05,
            "cold_switch_margin_per_usd": 1,
            "cold_switch_upgrade_exempt": True,
            "stay_margin_upgrade_exempt": False,
            "reference_worker": "worker-a",
            "reference_margin": 0,
        },
        "workers": [
            _worker(
                "worker-a",
                "synthetic/provider-a",
                "synthetic-a",
                input_cost=0.000001,
                cache_read_cost=0.0000005,
                cache_write_cost=0.0000015,
                output_cost=0.000002,
                thinking=False,
            ),
            _worker(
                "worker-b",
                "synthetic/provider-b",
                "synthetic-b",
                input_cost=0.000002,
                cache_read_cost=0.000001,
                cache_write_cost=0.0000025,
                output_cost=0.000004,
                thinking=True,
            ),
        ],
        "pricing_snapshot": {
            "config_path": "public/synthetic-pricing.yaml",
            "config_commit": "public-synthetic-pricing-v1",
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
    checkpoint_sha256 = _sha256(b"public synthetic ARC E2E checkpoint")
    golden = json.dumps(
        _golden(checkpoint_sha256),
        indent=2,
        sort_keys=True,
    ).encode()
    manifest = json.dumps(
        _manifest(weights, golden),
        indent=2,
        sort_keys=True,
    ).encode()
    (output_dir / "head.safetensors").write_bytes(weights)
    (output_dir / "head_golden.json").write_bytes(golden)
    (output_dir / "manifest.json").write_bytes(manifest)


if __name__ == "__main__":
    if len(sys.argv) != EXPECTED_ARGUMENT_COUNT:
        raise SystemExit("usage: artifact_fixture.py OUTPUT_DIR")
    generate(Path(sys.argv[1]))
