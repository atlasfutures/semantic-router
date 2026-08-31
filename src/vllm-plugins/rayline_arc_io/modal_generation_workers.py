# SPDX-License-Identifier: Apache-2.0

"""Ephemeral real-vLLM workers for the Rayline full-stack E2E rung.

Deploy with ``RAYLINE_ARC_WORKER_API_KEY`` set. The value is injected into the
two Modal containers as an ephemeral secret and is required again by the local
Semantic Router stack. Stop this app after the bounded canary.
"""

from __future__ import annotations

import os
import subprocess

import modal

APP_NAME = "rayline-arc-generation-workers"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
CUDA_BASE_IMAGE = (
    "nvidia/cuda:13.0.1-devel-ubuntu22.04"
    "@sha256:93a8d207db5aaa6384f834a6bf70d417433f709e61b57a91e7cc99c16172f49c"
)
VLLM_BASE_WHEEL_COMMIT = "98e91a9600eb75b2de14ef27f13b10088d1a1279"
VLLM_VERSION = "0.26.1rc1.dev36+g98e91a960"
VLLM_WHEEL_INDEX = f"https://wheels.vllm.ai/{VLLM_BASE_WHEEL_COMMIT}/cu130"

GPU_TYPE = "L4"
PORT = 8000
MAX_CONTAINERS = 1
MAX_CONCURRENT_INPUTS = 16
MAX_MODEL_LEN = 4_096
SCALEDOWN_WINDOW_SECONDS = 60
FUNCTION_TIMEOUT_SECONDS = 15 * 60

_deployment_api_key = os.environ.get("RAYLINE_ARC_WORKER_API_KEY", "")
_worker_secret = modal.Secret.from_dict(
    {"VLLM_API_KEY": _deployment_api_key},
)

image = (
    modal.Image.from_registry(CUDA_BASE_IMAGE, add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        f"vllm=={VLLM_VERSION}",
        extra_index_url=VLLM_WHEEL_INDEX,
        extra_options="--index-strategy unsafe-best-match",
    )
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "HF_HUB_CACHE": "/root/.cache/huggingface/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_CACHE_ROOT": "/root/.cache/vllm",
            "VLLM_LOGGING_LEVEL": "WARNING",
        }
    )
)

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("rayline-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("rayline-vllm-cache", create_if_missing=True)


def server_command(served_model_name: str) -> list[str]:
    """Build a fail-closed, revision-pinned OpenAI-compatible vLLM command."""
    api_key = os.environ.get("VLLM_API_KEY", "")
    if not api_key:
        raise RuntimeError("VLLM_API_KEY is required")
    return [
        "vllm",
        "serve",
        MODEL_ID,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--served-model-name",
        served_model_name,
        "--revision",
        MODEL_REVISION,
        "--tokenizer",
        MODEL_ID,
        "--tokenizer-revision",
        MODEL_REVISION,
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-batched-tokens",
        str(MAX_MODEL_LEN),
        "--max-num-seqs",
        str(MAX_CONCURRENT_INPUTS),
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization",
        "0.90",
        "--generation-config",
        "vllm",
        "--api-key",
        api_key,
        "--enforce-eager",
    ]


def _launch(served_model_name: str) -> None:
    subprocess.Popen(server_command(served_model_name), start_new_session=True)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=4.0,
    memory=16_384,
    timeout=FUNCTION_TIMEOUT_SECONDS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    max_containers=MAX_CONTAINERS,
    secrets=[_worker_secret],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
@modal.web_server(PORT, startup_timeout=10 * 60)
def worker_a() -> None:
    _launch("synthetic/provider-a")


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=4.0,
    memory=16_384,
    timeout=FUNCTION_TIMEOUT_SECONDS,
    scaledown_window=SCALEDOWN_WINDOW_SECONDS,
    max_containers=MAX_CONTAINERS,
    secrets=[_worker_secret],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
@modal.web_server(PORT, startup_timeout=10 * 60)
def worker_b() -> None:
    _launch("synthetic/provider-b")
