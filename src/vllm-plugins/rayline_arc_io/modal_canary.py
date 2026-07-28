# SPDX-License-Identifier: Apache-2.0

"""Ephemeral Modal CUDA correctness canaries for Rayline ARC pooling.

The canaries use public synthetic inputs, loopback-only vLLM servers, disabled
request logging and APC, and return only bounded metrics—never prompts or
embeddings.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import modal

rt = importlib.import_module("modal_canary_runtime")
numeric_helpers = importlib.import_module("modal_canary_numeric")
RUNG_B_NUMERIC_BUDGETS = numeric_helpers.RUNG_B_NUMERIC_BUDGETS
enforce_rung_b_numeric_budgets = numeric_helpers.enforce_rung_b_numeric_budgets
summarize_numeric = numeric_helpers.summarize_numeric

_THIS_DIR = Path(__file__).resolve().parent
_REMOTE_PLUGIN_DIR = "/opt/rayline_arc_io"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        f"vllm=={rt.VLLM_VERSION}",
        extra_index_url=rt.VLLM_WHEEL_INDEX,
        extra_options="--index-strategy unsafe-best-match",
    )
    .add_local_dir(_THIS_DIR, _REMOTE_PLUGIN_DIR, copy=True)
    .run_commands(f"uv pip install --system {_REMOTE_PLUGIN_DIR}")
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "HF_HUB_CACHE": "/root/.cache/huggingface/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_CACHE_ROOT": "/root/.cache/vllm",
            "VLLM_LOGGING_LEVEL": "WARNING",
            "RAYLINE_ARC_ENGINE_BUILD_ID": rt.ENGINE_BUILD_ID,
        }
    )
)

app = modal.App(rt.APP_NAME)
hf_cache = modal.Volume.from_name("rayline-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("rayline-vllm-cache", create_if_missing=True)

_rung_b_copy_commands = " && ".join(
    command
    for path in rt.RUNG_B_RUNTIME_FILES
    for command in (
        (f"cp /opt/vllm-rung-b/{path} /usr/local/lib/python3.12/site-packages/{path}"),
        (
            f"cmp -s /opt/vllm-rung-b/{path} "
            f"/usr/local/lib/python3.12/site-packages/{path}"
        ),
    )
)
rung_b_image = (
    image.apt_install("git")
    .run_commands(
        "git clone --depth 1 "
        f"--branch {rt.VLLM_RUNG_B_BRANCH} {rt.VLLM_RUNG_B_REPOSITORY} "
        "/opt/vllm-rung-b",
        f'test "$(git -C /opt/vllm-rung-b rev-parse HEAD)" = "{rt.VLLM_RUNG_B_COMMIT}"',
        _rung_b_copy_commands,
    )
    .env({"RAYLINE_ARC_ENGINE_BUILD_ID": rt.RUNG_B_ENGINE_BUILD_ID})
)


def _cost_record(elapsed_seconds: float) -> dict[str, Any]:
    estimated_cost = rt._estimated_cost(elapsed_seconds)
    if estimated_cost > rt.COST_CEILING_USD:
        raise RuntimeError(
            f"estimated Modal cost ${estimated_cost:.4f} exceeded "
            f"${rt.COST_CEILING_USD:.2f} ceiling"
        )
    return {
        "pricing_snapshot_date": rt.PRICING_SNAPSHOT_DATE,
        "pricing_source": "https://modal.com/pricing",
        "ceiling_usd": rt.COST_CEILING_USD,
        "estimated_usd": estimated_cost,
        "provider_spend_usd": 0.0,
        "h100_usd_per_second": rt.H100_USD_PER_SECOND,
        "cpu_core_usd_per_second": rt.CPU_CORE_USD_PER_SECOND,
        "memory_gib_usd_per_second": rt.MEMORY_GIB_USD_PER_SECOND,
    }


def _gpu_record(gpu: rt._GpuMonitor) -> dict[str, Any]:
    return {
        "requested_type": rt.GPU_TYPE,
        "observed_name": gpu.gpu_name,
        "total_memory_mib": gpu.total_memory_mib,
        "max_memory_used_mib": gpu.max_memory_used_mib,
        "max_utilization_percent": gpu.max_utilization_percent,
        "samples": gpu.samples,
    }


def _verify_rung_b_overlay() -> dict[str, str]:
    source_root = Path("/opt/vllm-rung-b")
    installed_root = Path("/usr/local/lib/python3.12/site-packages")
    installed_hashes: dict[str, str] = {}
    for relative_path in rt.RUNG_B_RUNTIME_FILES:
        source_bytes = (source_root / relative_path).read_bytes()
        installed_bytes = (installed_root / relative_path).read_bytes()
        if installed_bytes != source_bytes:
            raise RuntimeError(
                f"installed Rung B runtime file does not match {relative_path}"
            )
        installed_hashes[relative_path] = hashlib.sha256(installed_bytes).hexdigest()
    actual_commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if actual_commit != rt.VLLM_RUNG_B_COMMIT:
        raise RuntimeError(
            f"Rung B source commit is {actual_commit}, expected {rt.VLLM_RUNG_B_COMMIT}"
        )
    return installed_hashes


@app.function(
    gpu=rt.GPU_TYPE,
    image=image,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    cpu=rt.CPU_CORES,
    memory=rt.MEMORY_MIB,
    timeout=rt.FUNCTION_TIMEOUT_SECONDS,
)
def canary(run_id: str) -> dict[str, Any]:
    started = time.monotonic()
    with rt._GpuMonitor() as gpu:
        full_report, full_vectors = rt._run_server_mode(
            run_id=run_id,
            mode="single-schedule",
            port=8001,
            schedule_tokens=rt.FULL_SCHEDULE_TOKENS,
            shapes=("short", "multi_chunk"),
        )
        chunked_report, chunked_vectors = rt._run_server_mode(
            run_id=run_id,
            mode="chunked-8192",
            port=8002,
            schedule_tokens=rt.CHUNK_SCHEDULE_TOKENS,
            shapes=("short", "multi_chunk", "maximum_contract"),
        )
    elapsed_seconds = time.monotonic() - started
    cost = _cost_record(elapsed_seconds)
    hf_cache.commit()
    vllm_cache.commit()
    return {
        "schema_version": "rayline.arc.rung-a-modal-canary.v1",
        "run_id": run_id,
        "vllm_commit": rt.VLLM_COMMIT,
        "vllm_version": rt.VLLM_VERSION,
        "model": rt.MODEL_ID,
        "model_revision": rt.MODEL_REVISION,
        "plugin_version": rt.PLUGIN_VERSION,
        "serializer_version": rt.SERIALIZER_VERSION,
        "dtype": "bfloat16",
        "apc_enabled": False,
        "request_logging_enabled": False,
        "gpu": _gpu_record(gpu),
        "servers": [full_report, chunked_report],
        "numeric": summarize_numeric(full_vectors, chunked_vectors),
        "elapsed_seconds": elapsed_seconds,
        "cost": cost,
    }


@app.function(
    gpu=rt.GPU_TYPE,
    image=rung_b_image,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    cpu=rt.CPU_CORES,
    memory=rt.MEMORY_MIB,
    timeout=rt.FUNCTION_TIMEOUT_SECONDS,
)
def rung_b_canary(run_id: str) -> dict[str, Any]:
    started = time.monotonic()
    installed_hashes = _verify_rung_b_overlay()
    with rt._GpuMonitor() as gpu:
        full_report, full_vectors, full_raw_means = rt._run_rung_b_server_mode(
            mode="rung-b-single-schedule",
            port=8001,
            schedule_tokens=rt.FULL_SCHEDULE_TOKENS,
            shapes=("short", "multi_chunk", "maximum_contract"),
        )
        chunked_report, chunked_vectors, chunked_raw_means = rt._run_rung_b_server_mode(
            mode="rung-b-chunked-8192",
            port=8002,
            schedule_tokens=rt.CHUNK_SCHEDULE_TOKENS,
            shapes=("short", "multi_chunk", "maximum_contract"),
        )
    elapsed_seconds = time.monotonic() - started
    cost = _cost_record(elapsed_seconds)
    numeric = summarize_numeric(
        full_vectors,
        chunked_vectors,
        full_raw_means,
        chunked_raw_means,
    )
    enforce_rung_b_numeric_budgets(numeric)
    hf_cache.commit()
    vllm_cache.commit()
    return {
        "schema_version": "rayline.arc.rung-b-modal-canary.v1",
        "run_id": run_id,
        "vllm_base_wheel_commit": rt.VLLM_COMMIT,
        "vllm_commit": rt.VLLM_RUNG_B_COMMIT,
        "vllm_version": rt.VLLM_VERSION,
        "vllm_runtime_file_sha256": installed_hashes,
        "model": rt.MODEL_ID,
        "model_revision": rt.MODEL_REVISION,
        "serializer_version": rt.SERIALIZER_VERSION,
        "dtype": "bfloat16",
        "apc_enabled": False,
        "request_logging_enabled": False,
        "pooling": {
            "task": "embed",
            "pooling_type": "MEAN",
            "use_activation": True,
            "accumulator_dtype": "float32",
        },
        "gpu": _gpu_record(gpu),
        "servers": [full_report, chunked_report],
        "numeric": numeric,
        "numeric_budgets": {
            "source_run_id": "rayline-arc-rung-b-raw-20260728-attempt3",
            "hardware": "NVIDIA H100 80GB HBM3",
            "dtype": "bfloat16",
            "headroom_fraction": 0.2,
            "limits": RUNG_B_NUMERIC_BUDGETS,
        },
        "elapsed_seconds": elapsed_seconds,
        "cost": cost,
    }


@app.function(
    gpu=rt.GPU_TYPE,
    image=rung_b_image,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    cpu=rt.CPU_CORES,
    memory=rt.MEMORY_MIB,
    timeout=rt.FUNCTION_TIMEOUT_SECONDS,
)
def rung_b_plugin_canary(run_id: str) -> dict[str, Any]:
    started = time.monotonic()
    installed_hashes = _verify_rung_b_overlay()
    with rt._GpuMonitor() as gpu:
        plugin_report, _ = rt._run_server_mode(
            run_id=run_id,
            mode="rung-b-plugin-chunked-8192",
            port=8004,
            schedule_tokens=rt.CHUNK_SCHEDULE_TOKENS,
            shapes=("short", "maximum_contract"),
            serving_rung="B",
        )
    elapsed_seconds = time.monotonic() - started
    cost = _cost_record(elapsed_seconds)
    hf_cache.commit()
    vllm_cache.commit()
    return {
        "schema_version": "rayline.arc.rung-b-plugin-modal-canary.v1",
        "run_id": run_id,
        "vllm_base_wheel_commit": rt.VLLM_COMMIT,
        "vllm_commit": rt.VLLM_RUNG_B_COMMIT,
        "vllm_version": rt.VLLM_VERSION,
        "vllm_runtime_file_sha256": installed_hashes,
        "model": rt.MODEL_ID,
        "model_revision": rt.MODEL_REVISION,
        "plugin_version": rt.PLUGIN_VERSION,
        "serializer_version": rt.SERIALIZER_VERSION,
        "serving_rung": "B",
        "pooling_capabilities": ["chunked_causal_mean"],
        "dtype": "bfloat16",
        "apc_enabled": False,
        "request_logging_enabled": False,
        "gpu": _gpu_record(gpu),
        "server": plugin_report,
        "elapsed_seconds": elapsed_seconds,
        "cost": cost,
    }


@app.function(
    gpu=rt.GPU_TYPE,
    image=rung_b_image,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    cpu=rt.CPU_CORES,
    memory=rt.MEMORY_MIB,
    timeout=rt.FUNCTION_TIMEOUT_SECONDS,
)
def rung_b_boundary_canary(run_id: str) -> dict[str, Any]:
    started = time.monotonic()
    installed_hashes = _verify_rung_b_overlay()
    with rt._GpuMonitor() as gpu:
        boundary = rt._run_rung_b_boundary_probe()
    elapsed_seconds = time.monotonic() - started
    cost = _cost_record(elapsed_seconds)
    hf_cache.commit()
    vllm_cache.commit()
    return {
        "schema_version": "rayline.arc.rung-b-boundary-modal-canary.v1",
        "run_id": run_id,
        "vllm_base_wheel_commit": rt.VLLM_COMMIT,
        "vllm_commit": rt.VLLM_RUNG_B_COMMIT,
        "vllm_version": rt.VLLM_VERSION,
        "vllm_runtime_file_sha256": installed_hashes,
        "model": rt.MODEL_ID,
        "model_revision": rt.MODEL_REVISION,
        "gpu": _gpu_record(gpu),
        "boundary": boundary,
        "elapsed_seconds": elapsed_seconds,
        "cost": cost,
    }


@app.local_entrypoint()
def main(run_id: str, output: str, rung: str = "a") -> None:
    if rung == "a":
        report = canary.remote(run_id=run_id)
    elif rung == "b":
        report = rung_b_canary.remote(run_id=run_id)
    elif rung == "b-plugin":
        report = rung_b_plugin_canary.remote(run_id=run_id)
    elif rung == "b-boundary":
        report = rung_b_boundary_canary.remote(run_id=run_id)
    else:
        raise ValueError("rung must be 'a', 'b', 'b-plugin', or 'b-boundary'")
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = {
        "run_id": report["run_id"],
        "output": str(output_path),
        "gpu": report["gpu"],
        "elapsed_seconds": report["elapsed_seconds"],
        "estimated_cost_usd": report["cost"]["estimated_usd"],
    }
    if "numeric" in report:
        summary["observed_maxima"] = report["numeric"]["observed_maxima"]
    if "boundary" in report:
        summary["boundary"] = report["boundary"]
    print(json.dumps(summary, indent=2, sort_keys=True))
