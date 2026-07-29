# SPDX-License-Identifier: Apache-2.0

"""Protected Modal deployment for the frozen Rayline ARC Rung B encoder.

Deploy explicitly after the Rung B CUDA gate passes:

    modal deploy modal_service.py

The web endpoint is protected by Modal proxy authentication. Semantic Router
must send the corresponding ``Modal-Key`` and ``Modal-Secret`` headers via its
environment-referenced ARC encoder configuration.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rayline_arc_io.integrity import compute_source_digest

APP_NAME = "rayline-arc-encoder"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
# Digest-pinned so rebuilding this deployment later cannot silently resolve a
# different CUDA base while keeping the same engine/plugin identity.
CUDA_BASE_IMAGE = (
    "nvidia/cuda:13.0.1-devel-ubuntu22.04"
    "@sha256:93a8d207db5aaa6384f834a6bf70d417433f709e61b57a91e7cc99c16172f49c"
)
VLLM_BASE_WHEEL_COMMIT = "98e91a9600eb75b2de14ef27f13b10088d1a1279"
VLLM_COMMIT = "918a2d159b718ab7f50d3fba87578e310034593d"
VLLM_VERSION = "0.26.1rc1.dev36+g98e91a960"
VLLM_WHEEL_INDEX = f"https://wheels.vllm.ai/{VLLM_BASE_WHEEL_COMMIT}/cu130"
VLLM_REPOSITORY = "https://github.com/davidvgilmore/vllm.git"
VLLM_BRANCH = "rayline/pl-0039-causal-mean"
ENGINE_BUILD_ID = f"vllm@{VLLM_COMMIT}"
MAX_SERIALIZED_TOKENS = 262_144
CHUNK_SCHEDULE_TOKENS = 8_192
PORT = 8000

_THIS_DIR = Path(__file__).resolve().parent
_REMOTE_PLUGIN_DIR = "/opt/rayline_arc_io"
# Bind the deployed plugin identity to the exact shipped source: the plugin
# recomputes this digest over its installed files at startup and fails closed
# on mismatch.
PLUGIN_SOURCE_DIGEST = compute_source_digest(_THIS_DIR / "src" / "rayline_arc_io")
_POOLER_CONFIG = {
    "task": "embed",
    "pooling_type": "MEAN",
    "use_activation": True,
    "enable_chunked_processing": False,
}
_RUNTIME_FILES = (
    "vllm/config/model.py",
    "vllm/model_executor/layers/pooler/seqwise/heads.py",
    "vllm/model_executor/layers/pooler/seqwise/methods.py",
    "vllm/model_executor/layers/pooler/seqwise/poolers.py",
    "vllm/model_executor/models/gritlm.py",
    "vllm/pooling_params.py",
    "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/pool/metadata.py",
    "vllm/v1/worker/gpu_input_batch.py",
    "vllm/v1/worker/gpu_model_runner.py",
)

image = (
    modal.Image.from_registry(
        CUDA_BASE_IMAGE,
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        f"vllm=={VLLM_VERSION}",
        extra_index_url=VLLM_WHEEL_INDEX,
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
            "RAYLINE_ARC_ENGINE_BUILD_ID": ENGINE_BUILD_ID,
            "RAYLINE_ARC_PLUGIN_SOURCE_DIGEST": PLUGIN_SOURCE_DIGEST,
        }
    )
)

_runtime_copy_commands = " && ".join(
    command
    for path in _RUNTIME_FILES
    for command in (
        (f"cp /opt/vllm-rung-b/{path} /usr/local/lib/python3.12/site-packages/{path}"),
        (
            f"cmp -s /opt/vllm-rung-b/{path} "
            f"/usr/local/lib/python3.12/site-packages/{path}"
        ),
    )
)
_expected_runtime_diff = "\\n".join(_RUNTIME_FILES)
image = image.apt_install("git").run_commands(
    # Fetch the immutable commit rather than the branch tip so this pinned
    # deployment stays rebuildable after the branch advances.
    "git init /opt/vllm-rung-b",
    f"git -C /opt/vllm-rung-b remote add origin {VLLM_REPOSITORY}",
    f"git -C /opt/vllm-rung-b fetch --depth 1 origin {VLLM_COMMIT}",
    "git -C /opt/vllm-rung-b checkout --detach FETCH_HEAD",
    f'test "$(git -C /opt/vllm-rung-b rev-parse HEAD)" = "{VLLM_COMMIT}"',
    # The overlay list must cover exactly the fork's runtime diff; a commit
    # that changes an uncopied file must fail the build, not deploy a
    # silently divergent engine.
    f"git -C /opt/vllm-rung-b fetch --depth 1 origin {VLLM_BASE_WHEEL_COMMIT}",
    (
        'test "$(git -C /opt/vllm-rung-b diff --name-only '
        f"{VLLM_BASE_WHEEL_COMMIT}..{VLLM_COMMIT} -- 'vllm/')\" = "
        f"\"$(printf '{_expected_runtime_diff}')\""
    ),
    _runtime_copy_commands,
)

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("rayline-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("rayline-vllm-cache", create_if_missing=True)


def server_command() -> list[str]:
    """Return the immutable production Rung B vLLM command."""
    return [
        "vllm",
        "serve",
        MODEL_ID,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--runner",
        "pooling",
        "--revision",
        MODEL_REVISION,
        "--tokenizer",
        MODEL_ID,
        "--tokenizer-revision",
        MODEL_REVISION,
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(MAX_SERIALIZED_TOKENS),
        "--max-num-batched-tokens",
        str(CHUNK_SCHEDULE_TOKENS),
        "--max-num-seqs",
        "1",
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization",
        "0.92",
        "--io-processor-plugin",
        "rayline_arc_io",
        "--pooler-config",
        json.dumps(_POOLER_CONFIG, separators=(",", ":")),
        "--no-enable-log-requests",
        "--disable-log-stats",
    ]


@app.function(
    image=image,
    gpu="H100",
    cpu=8.0,
    memory=65_536,
    timeout=31 * 60,
    scaledown_window=300,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
@modal.concurrent(max_inputs=1)
@modal.web_server(
    PORT,
    startup_timeout=10 * 60,
    requires_proxy_auth=True,
)
def serve() -> None:
    subprocess.Popen(server_command(), start_new_session=True)
