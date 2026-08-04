# SPDX-License-Identifier: Apache-2.0

"""Protected Modal deployment for explicit retained Rayline ARC sessions."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rayline_arc_io.constants import MAX_SERIALIZED_TOKENS
from rayline_arc_io.integrity import compute_source_digest, installed_source_digest

DEFAULT_APP_NAME = "rayline-arc-session-encoder"
SCALEOUT_APP_NAMES = (
    "rayline-arc-session-encoder-a",
    "rayline-arc-session-encoder-b",
    "rayline-arc-session-encoder-c",
)
PERF030_APP_PROFILES = {
    "rayline-arc-session-encoder-reference-perf030": "torch_reference",
    "rayline-arc-session-encoder-flashinfer-perf030": "flashinfer",
}
AGT017_APP_PROFILES = {
    "rayline-arc-session-encoder-flashinfer-agt017": "flashinfer",
}
EXPERIMENT_APP_PROFILES = {**PERF030_APP_PROFILES, **AGT017_APP_PROFILES}
ALLOWED_APP_NAMES = (DEFAULT_APP_NAME, *SCALEOUT_APP_NAMES, *EXPERIMENT_APP_PROFILES)
APP_NAME = os.environ.get("RAYLINE_ARC_SESSION_APP_NAME", DEFAULT_APP_NAME)
if APP_NAME not in ALLOWED_APP_NAMES:
    raise RuntimeError("unsupported Rayline ARC session app name")
MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
CUDA_BASE_IMAGE = (
    "nvidia/cuda:13.0.1-devel-ubuntu22.04"
    "@sha256:93a8d207db5aaa6384f834a6bf70d417433f709e61b57a91e7cc99c16172f49c"
)
VLLM_BASE_WHEEL_COMMIT = "98e91a9600eb75b2de14ef27f13b10088d1a1279"
VLLM_COMMIT = "9f5ea81ca0aa570aea46baf82311a1139c1267ca"
VLLM_VERSION = "0.26.1rc1.dev36+g98e91a960"
VLLM_WHEEL_INDEX = f"https://wheels.vllm.ai/{VLLM_BASE_WHEEL_COMMIT}/cu130"
VLLM_REPOSITORY = "https://github.com/atlasfutures/vllm.git"
BASE_ENGINE_BUILD_ID = f"vllm@{VLLM_COMMIT}"
GDN_PREFILL_BACKEND = EXPERIMENT_APP_PROFILES.get(APP_NAME, "torch_reference")
ENGINE_BUILD_ID = (
    f"{BASE_ENGINE_BUILD_ID}+gdn-{GDN_PREFILL_BACKEND.replace('_', '-')}-eager"
    if APP_NAME in EXPERIMENT_APP_PROFILES
    else BASE_ENGINE_BUILD_ID
)


def _runtime_profile() -> tuple[str, str]:
    runtime_app_name = os.environ.get("RAYLINE_ARC_SESSION_APP_NAME", "")
    if runtime_app_name not in ALLOWED_APP_NAMES or runtime_app_name != APP_NAME:
        raise RuntimeError("Rayline ARC runtime app identity diverged")
    backend = EXPERIMENT_APP_PROFILES.get(runtime_app_name, "torch_reference")
    build_id = (
        f"{BASE_ENGINE_BUILD_ID}+gdn-{backend.replace('_', '-')}-eager"
        if runtime_app_name in EXPERIMENT_APP_PROFILES
        else BASE_ENGINE_BUILD_ID
    )
    if os.environ.get("RAYLINE_ARC_ENGINE_BUILD_ID") != build_id:
        raise RuntimeError("Rayline ARC runtime engine identity diverged")
    return backend, build_id


GPU_TYPE = "H100"
MODAL_REGION = "us-east"
MAX_SESSIONS = 8
MAX_RESIDENT_TOKENS = MAX_SESSIONS * MAX_SERIALIZED_TOKENS
IDLE_TTL_SECONDS = 5 * 60
CHUNK_SCHEDULE_TOKENS = 8_192

_THIS_DIR = Path(__file__).resolve().parent
_REMOTE_PLUGIN_DIR = "/opt/rayline_arc_io"
_REMOTE_VLLM_DIR = "/opt/vllm-rsp005"
PLUGIN_SOURCE_DIGEST = compute_source_digest(_THIS_DIR / "src" / "rayline_arc_io")
VLLM_RUNTIME_FILES = (
    "vllm/config/model.py",
    "vllm/engine/arg_utils.py",
    "vllm/engine/protocol.py",
    "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
    "vllm/model_executor/layers/pooler/seqwise/heads.py",
    "vllm/model_executor/layers/pooler/seqwise/methods.py",
    "vllm/model_executor/layers/pooler/seqwise/poolers.py",
    "vllm/model_executor/models/gritlm.py",
    "vllm/model_executor/models/qwen3_next.py",
    "vllm/outputs.py",
    "vllm/pooling_params.py",
    "vllm/v1/attention/backends/gdn_attn.py",
    "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/engine/async_llm.py",
    "vllm/v1/engine/output_processor.py",
    "vllm/v1/engine/pooling_session.py",
    "vllm/v1/pool/metadata.py",
    "vllm/v1/worker/gpu_input_batch.py",
    "vllm/v1/worker/gpu_model_runner.py",
)

_expected_runtime_files = repr(VLLM_RUNTIME_FILES)
image = (
    modal.Image.from_registry(CUDA_BASE_IMAGE, add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .uv_pip_install(
        f"vllm=={VLLM_VERSION}",
        extra_index_url=VLLM_WHEEL_INDEX,
        extra_options="--index-strategy unsafe-best-match",
    )
    .add_local_dir(_THIS_DIR, _REMOTE_PLUGIN_DIR, copy=True)
    .run_commands(f"uv pip install --system {_REMOTE_PLUGIN_DIR}")
    .run_commands(
        f"git init {_REMOTE_VLLM_DIR}",
        f"git -C {_REMOTE_VLLM_DIR} remote add origin {VLLM_REPOSITORY}",
        (
            f"git -C {_REMOTE_VLLM_DIR} fetch --depth 1 origin "
            f"{VLLM_COMMIT} {VLLM_BASE_WHEEL_COMMIT}"
        ),
        f"git -C {_REMOTE_VLLM_DIR} checkout --detach {VLLM_COMMIT}",
        f'test "$(git -C {_REMOTE_VLLM_DIR} rev-parse HEAD)" = "{VLLM_COMMIT}"',
        (
            'python3 -c "import subprocess; '
            "got=tuple(subprocess.check_output("
            "['git','-C','"
            + _REMOTE_VLLM_DIR
            + "','diff','--name-only','"
            + VLLM_BASE_WHEEL_COMMIT
            + ".."
            + VLLM_COMMIT
            + "','--','vllm/'],text=True).splitlines()); "
            "expected="
            + _expected_runtime_files
            + '; assert got==expected,(got,expected)"'
        ),
    )
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "HF_HUB_CACHE": "/root/.cache/huggingface/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_CACHE_ROOT": "/root/.cache/vllm",
            "VLLM_LOGGING_LEVEL": "WARNING",
            "RAYLINE_ARC_SESSION_APP_NAME": APP_NAME,
            "RAYLINE_ARC_ENGINE_BUILD_ID": ENGINE_BUILD_ID,
            "RAYLINE_ARC_PLUGIN_SOURCE_DIGEST": PLUGIN_SOURCE_DIGEST,
        }
    )
)

_overlay_commands: list[str] = []
for _runtime_file in VLLM_RUNTIME_FILES:
    _installed_file = f"/usr/local/lib/python3.12/site-packages/{_runtime_file}"
    _overlay_commands.extend(
        (
            f"cp {_REMOTE_VLLM_DIR}/{_runtime_file} {_installed_file}",
            f"cmp -s {_REMOTE_VLLM_DIR}/{_runtime_file} {_installed_file}",
            f"python3 -m py_compile {_installed_file}",
        )
    )
image = image.run_commands(*_overlay_commands)

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("rayline-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("rayline-vllm-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    region=MODAL_REGION,
    cpu=8.0,
    memory=65_536,
    timeout=31 * 60,
    scaledown_window=300,
    max_containers=1,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
@modal.concurrent(max_inputs=32)
class SessionEncoder:
    @modal.enter()
    def start(self) -> None:
        from rayline_arc_io.processor import (  # noqa: PLC0415
            RaylineArcIOProcessor,
        )
        from rayline_arc_io.serializer import (  # noqa: PLC0415
            TokenBlockSerializer,
        )
        from rayline_arc_io.session_api import (  # noqa: PLC0415
            SessionAPIMetadata,
            create_session_app,
        )
        from rayline_arc_io.session_coordinator import (  # noqa: PLC0415
            SessionCoordinator,
        )
        from rayline_arc_io.session_runtime import (  # noqa: PLC0415
            VLLMRetainedPoolingBackendFactory,
            VLLMSessionEngineMetricsProvider,
        )
        from transformers import AutoTokenizer  # noqa: PLC0415
        from vllm.config import PoolerConfig  # noqa: PLC0415
        from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: PLC0415
        from vllm.v1.engine.async_llm import AsyncLLM  # noqa: PLC0415

        expected_digest = os.environ["RAYLINE_ARC_PLUGIN_SOURCE_DIGEST"]
        if installed_source_digest() != expected_digest:
            raise RuntimeError("installed ARC plugin source digest diverged")
        runtime_backend, runtime_build_id = _runtime_profile()

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
        )
        RaylineArcIOProcessor._configure_and_validate_tokenizer(tokenizer)
        engine_args = AsyncEngineArgs(
            model=MODEL_ID,
            tokenizer=MODEL_ID,
            revision=MODEL_REVISION,
            tokenizer_revision=MODEL_REVISION,
            runner="pooling",
            dtype="bfloat16",
            max_model_len=MAX_SERIALIZED_TOKENS,
            max_num_batched_tokens=CHUNK_SCHEDULE_TOKENS,
            max_num_seqs=MAX_SESSIONS,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            gpu_memory_utilization=0.92,
            enforce_eager=True,
            gdn_prefill_backend=runtime_backend,
            pooler_config=PoolerConfig(
                task="embed",
                pooling_type="MEAN",
                use_activation=True,
                enable_chunked_processing=False,
            ),
            enable_logging_iteration_details=True,
            enable_log_requests=False,
        )
        self._engine = AsyncLLM.from_engine_args(engine_args)
        self._coordinator = SessionCoordinator(
            VLLMRetainedPoolingBackendFactory(self._engine),
            max_sessions=MAX_SESSIONS,
            max_resident_tokens=MAX_RESIDENT_TOKENS,
            idle_ttl_seconds=IDLE_TTL_SECONDS,
        )
        self._web_app = create_session_app(
            self._coordinator,
            TokenBlockSerializer(tokenizer),
            SessionAPIMetadata(engine_build_id=runtime_build_id),
            VLLMSessionEngineMetricsProvider(
                self._engine.get_scheduler_load,
                self._coordinator.append_metrics_snapshot,
            ),
        )

    @modal.exit()
    async def stop(self) -> None:
        await self._coordinator.close_all()
        self._engine.shutdown()

    @modal.asgi_app(requires_proxy_auth=True)
    def web(self):
        return self._web_app
