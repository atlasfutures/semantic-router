#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Build the native-Rayline form of the AGT013 synthetic policy artifact."""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from typing import Any

import yaml
from artifact_fixture import MODEL_REVISION, SERIALIZER
from openrouter_agentic_artifact_fixture import (
    MAX_COMPLETION_TOKENS,
    WORKERS,
)
from openrouter_artifact_fixture import _golden, _tensors

ENCODER_MODEL = "Qwen/Qwen3.5-0.8B"
CHECKPOINT_REMOTE_PATH = "agt014/native-openrouter-agentic.pt"
DECISION_LOG_REMOTE_PATH = "agt014/native-openrouter-decisions.jsonl"
PRICING_SNAPSHOT = "openrouter-bounded-provider-orders-2026-08-03"
GOLDEN_TOLERANCE = 0.001
CLI_ARGUMENT_COUNT = 3


def _load_pathfinder(pathfinder_root: Path) -> tuple[Any, Any, Any, Any]:
    sys.path.insert(0, str(pathfinder_root / "src"))
    torch = importlib.import_module("torch")
    model = importlib.import_module("rayline_router.policy.mtrouter_model")
    return (
        torch,
        model.MTRouterConfig,
        model.MTRouterEstimator,
        model.MTRouterModelMeta,
    )


def _model_meta(model_meta_type: Any) -> list[Any]:
    return [
        model_meta_type(
            model_id=str(worker["id"]),
            max_input_tokens=262_144,
            max_output_tokens=MAX_COMPLETION_TOKENS,
            input_cost_per_million=float(worker["prompt_cost"]) * 1_000_000,
            output_cost_per_million=float(worker["completion_cost"]) * 1_000_000,
            release_date="2026-08-03",
            base_model_id=str(worker["model"]),
            thinking_mode="disabled",
            reasoning_budget_tokens=0,
            cache_read_cost_per_million=(float(worker["cache_read_cost"]) * 1_000_000),
            cache_write_cost_per_million=(
                float(worker["cache_write_cost"]) * 1_000_000
            ),
        )
        for worker in WORKERS
    ]


def build_checkpoint(
    pathfinder_root: Path,
    output: Path,
    *,
    artifact_id: str = "",
) -> dict[str, Any]:
    torch, config_type, estimator_type, model_meta_type = _load_pathfinder(
        pathfinder_root
    )
    config = config_type(
        encoder_model=ENCODER_MODEL,
        encoder_revision=MODEL_REVISION,
        encoder_dim=1024,
        max_tokens=262_144,
        min_recent_turns=1,
        min_recent_tokens=64,
        serialization_version=SERIALIZER,
        pooling_mode="masked_mean",
        normalize_embeddings=True,
        attention_implementation="sdpa",
        hidden_dims=(256, 256),
        dropout=0.1,
        pool=[str(worker["id"]) for worker in WORKERS],
        architecture="switch_aware",
    )
    estimator = estimator_type(_model_meta(model_meta_type), config)
    state = estimator.state_dict()
    specs = _tensors()
    if set(state) != set(specs):
        raise RuntimeError("native and ARC policy tensor names diverged")
    for name, spec in specs.items():
        value = torch.full(spec.shape, float(spec.fill), dtype=torch.float32)
        flattened = value.reshape(-1)
        for index, element in (spec.sparse or {}).items():
            flattened[index] = float(element)
        state[name] = value
    estimator.load_state_dict(state, strict=True)
    estimator.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    estimator.save(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    golden = _golden(digest)
    for case in golden["cases"]:
        embedding = torch.tensor(case["embedding"], dtype=torch.float32)
        actual = estimator.q_values_all_models(
            embedding,
            previous_model_index=int(case["previous_model_index"]),
            route_call_index=int(case["route_call_index"]),
        ).tolist()
        expected = list(map(float, case["scores"]))
        if (
            max(abs(left - right) for left, right in zip(actual, expected, strict=True))
            > GOLDEN_TOLERANCE
        ):
            raise RuntimeError("native checkpoint did not reproduce the ARC golden")
        if max(range(len(actual)), key=actual.__getitem__) != int(
            case["selected_index"]
        ):
            raise RuntimeError("native checkpoint changed the ARC selection")
    receipt = {
        "checkpoint_sha256": digest,
        "golden_cases": len(golden["cases"]),
        "worker_pool": [str(worker["id"]) for worker in WORKERS],
        "encoder_model": ENCODER_MODEL,
        "encoder_revision": MODEL_REVISION,
        "serialization": SERIALIZER,
        "pooling": "masked_mean",
    }
    if artifact_id:
        receipt["artifact_id"] = artifact_id
    return receipt


def router_config_text(
    *,
    training_stage: str = "openrouter_modal_native_agt014",
    max_completion_tokens: int = MAX_COMPLETION_TOKENS,
    app_title: str = "Rayline AGT014",
) -> str:
    workers = []
    for worker in WORKERS:
        workers.append(
            {
                "id": worker["id"],
                "backend": "openrouter",
                "model": worker["model"],
                "api_key_env": "OPENROUTER_API_KEY",
                "estimated_input_cost_per_token": worker["prompt_cost"],
                "estimated_cache_read_cost_per_token": worker["cache_read_cost"],
                "estimated_cache_write_cost_per_token": worker["cache_write_cost"],
                "estimated_output_cost_per_token": worker["completion_cost"],
                "latency_ms": 1000,
                "capability_tags": ["public-openrouter-agentic-benchmark"],
                "openrouter_provider_slug": worker["provider_slug"],
                "openrouter_provider_name": worker["provider_name"],
                "openrouter_provider_order": worker["provider_order"],
                "openrouter_allow_fallbacks": False,
                "openrouter_require_parameters": True,
                "openrouter_pricing_source": worker["pricing_source"],
                "thinking_mode": "disabled",
                "reasoning_budget_tokens": 0,
                "minimum_completion_tokens": max_completion_tokens,
                "max_completion_tokens": max_completion_tokens,
                "temperature": 0,
                "extra_body": {"reasoning": {"enabled": False, "effort": "none"}},
                "openrouter_max_retries": 1,
                "openrouter_retry_base_seconds": 2.0,
                "openrouter_retry_cap_seconds": 30.0,
                "attempt_deadline_seconds": 120,
            }
        )
    config = {
        "router": {
            "policy": "mtrouter",
            "checkpoint_path": f"/artifacts/{CHECKPOINT_REMOTE_PATH}",
            "log_path": f"/artifacts/{DECISION_LOG_REMOTE_PATH}",
            "event_sink": "file",
            "pricing_snapshot_version": PRICING_SNAPSHOT,
            "encoder_dim": 1024,
            "seed": 20260803,
            "training_stage": training_stage,
            "trace_store": "memory",
            "mtrouter_device": "cuda",
            "mtrouter_incremental_encode": True,
            "mtrouter_previous_worker_stay_margin": 0,
            "mtrouter_cold_switch_margin_per_usd": 0,
            "mtrouter_upgrade_margin_exempt": False,
            "mtrouter_stay_margin_upgrade_exempt": False,
            "mtrouter_kv_session_budget_tokens": 262_144,
            "mtrouter_kv_process_budget_tokens": 524_288,
            "mtrouter_kv_session_idle_ttl_s": 900,
            "openrouter_app_title": app_title,
            "openrouter_app_url": "https://rayline.ai",
            "openrouter_app_categories": "benchmark",
        },
        "workers": workers,
    }
    return yaml.safe_dump(config, sort_keys=False)


if __name__ == "__main__":
    if len(sys.argv) != CLI_ARGUMENT_COUNT:
        raise SystemExit(
            "usage: openrouter_modal_native_fixture.py PATHFINDER_ROOT OUTPUT"
        )
    print(build_checkpoint(Path(sys.argv[1]), Path(sys.argv[2])))
