# SPDX-License-Identifier: Apache-2.0

"""Protected Modal deployment for explicit retained Rayline ARC sessions."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rayline_arc_io.constants import MAX_SERIALIZED_TOKENS
from rayline_arc_io.integrity import compute_source_digest, installed_source_digest
from rayline_arc_io.startup_log import capture_startup_log

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
AGT018_APP_PROFILES = {
    "rayline-arc-session-encoder-flashinfer-agt018": "flashinfer",
}
AGT019_APP_PROFILES = {
    "rayline-arc-session-encoder-flashinfer-agt019": "flashinfer",
}
# PERF031 arm 1 only. Arm 0 is the negative control and deliberately has no
# profile: it must deploy DEFAULT_APP_NAME so its ENGINE_BUILD_ID stays the bare
# `vllm@9f5ea81c...`, byte-identical to PERF021's recorded engine identity. A
# `-reference-perf031` profile would stamp `+gdn-torch-reference-eager` and stop
# the control from being identity-matched to the run it must reproduce.
PERF031_APP_PROFILES = {
    "rayline-arc-session-encoder-flashinfer-perf031": "flashinfer",
}
# PERF034 is the saturation cap-raise arm: identical engine profile to PERF033,
# but MAX_SESSIONS and the ingress cap widen for this app name only (below).
PERF034_APP_PROFILES = {
    "rayline-arc-session-encoder-flashinfer-perf034": "flashinfer",
}
# PERF035 is the L4 capacity arm: identical engine profile and identical 8/32
# caps to PERF033, but this app name alone deploys on a 24 GB L4 (below), the
# only GPU class both Modal and the GCP Cloud Run deployment target sell.
PERF035_APP_PROFILES = {
    "rayline-arc-session-encoder-flashinfer-perf035-l4": "flashinfer",
}
# PERF036 is the RTX PRO 6000 capacity arm: identical engine profile and
# identical 8/32 caps to PERF033 and PERF035, but this app name alone deploys
# on the 96 GB RTX PRO 6000 (below), Cloud Run's other GPU class and the first
# card whose prediction scales from a measured cross-GPU anchor (PERF035).
PERF036_APP_PROFILES = {
    "rayline-arc-session-encoder-flashinfer-perf036-rtx6000": "flashinfer",
}
# The standing dev encoder. It is not a packet and closes no run, but it is a
# profile because a profile is the only thing that carries an engine identity:
# a name outside EXPERIMENT_APP_PROFILES gets `torch_reference` and the bare
# `vllm@<sha>` build id, and the router's expected_build_id would then differ
# from every arm PERF033-037 measured. It takes flashinfer for that parity,
# and an L4 (below) because dev consults sit far under the 0.1977 decisions/s
# PERF035 measured on that card. That number is also why this app is evidence
# for nothing: the L4 does not carry the production rate.
DEV_APP_PROFILES = {
    "rayline-arc-session-encoder-dev": "flashinfer",
}
# The standing production encoder takes the proven FlashInfer engine identity
# on an L4 placement. It has its own name so the historical default app
# remains the byte-identical torch-reference arm recorded by closed runs.
PROD_APP_PROFILES = {
    "rayline-arc-session-encoder-prod": "flashinfer",
}
EXPERIMENT_APP_PROFILES = {
    **PERF030_APP_PROFILES,
    **AGT017_APP_PROFILES,
    **AGT018_APP_PROFILES,
    **AGT019_APP_PROFILES,
    **PERF031_APP_PROFILES,
    **PERF034_APP_PROFILES,
    **PERF035_APP_PROFILES,
    **PERF036_APP_PROFILES,
    **DEV_APP_PROFILES,
    **PROD_APP_PROFILES,
}
ALLOWED_APP_NAMES = (DEFAULT_APP_NAME, *SCALEOUT_APP_NAMES, *EXPERIMENT_APP_PROFILES)
APP_NAME = os.environ.get("RAYLINE_ARC_SESSION_APP_NAME", DEFAULT_APP_NAME)
if APP_NAME not in ALLOWED_APP_NAMES:
    raise RuntimeError("unsupported Rayline ARC session app name")
# All encoder deployments scale to zero. Shadow processing tolerates the cold
# start, and its caller owns a bounded completion timeout rather than paying a
# standing GPU cost between recordings.
MIN_CONTAINERS = 0
GPU_SNAPSHOT_ENABLED = APP_NAME in PROD_APP_PROFILES
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


# Placement is deliberately unpinned. The former region="us-east" pin cost a
# 1.75x Modal narrow-region multiplier while PERF011/PERF014 measured it as
# worse, not better: pinning produced 1.042x the PERF009 prepare p50 and 0.994x
# its throughput, and neither preregistered placement gate passed. The measured
# ~0.637s service/transport floor is backend- and region-independent, and
# earlier placement work ruled out simple region distance as its cause.
# Re-pin only with a measurement that clears a placement gate.
# Every recorded run is an H100 run and stays one. PERF035 alone deploys on a
# 24 GB L4 and PERF036 alone on a 96 GB RTX PRO 6000, because those are the
# two GPU classes the deployment target (GCP Cloud Run with GPU) sells and a
# capacity claim for that target has to be measured on that silicon. Scoped to
# the exact app names so no closed run's evidence can change class underneath
# it. The standing dev and production apps take L4s for cost; neither is
# benchmark evidence, so they share a branch isolated from PERF035.
if APP_NAME in PERF035_APP_PROFILES:
    GPU_TYPE = "L4"
elif APP_NAME in PERF036_APP_PROFILES:
    GPU_TYPE = "RTX-PRO-6000"
elif APP_NAME in DEV_APP_PROFILES or APP_NAME in PROD_APP_PROFILES:
    GPU_TYPE = "L4"
else:
    GPU_TYPE = "H100"
# The historical 8 was committed without rationale (4f14763b) and predates the
# frozen corpus's 8 episodes; it is retained for every non-PERF034 app because
# the live stack sizes around it. PERF034 raises its own app to 32 to locate
# the encoder's saturation knee. 32 lanes on the frozen packet-v3 corpus peaks
# at 4,261,735 resident tokens (~51% of the ~70 GiB pool); the worst case
# implied by MAX_RESIDENT_TOKENS (96 GiB) does NOT fit and is excluded only by
# the corpus, a bound preregistered in the PERF034 contract. PERF035 keeps 8
# deliberately: on its 24 GB L4 the 32-lane corpus peak alone is ~49 GiB, and
# even 8 lanes only fit by corpus construction (12.92 GiB against a ~19 GiB
# pool), a bound preregistered in the PERF035 contract. The standing dev and
# production apps share that GPU class and inherit the same 8, but not the
# corpus that makes 8 safe: the coordinator admits in tokens, not GiB, so eight
# maximum-sized live sessions can ask an L4 for more than it has.
MAX_SESSIONS = 32 if APP_NAME in PERF034_APP_PROFILES else 8
MAX_RESIDENT_TOKENS = MAX_SESSIONS * MAX_SERIALIZED_TOKENS
IDLE_TTL_SECONDS = 5 * 60
CHUNK_SCHEDULE_TOKENS = 8_192
# At 32 lanes every in-flight request would exactly hit a 32-input ingress cap,
# and queueing there is invisible to start_lag; 64 keeps ingress unbound so the
# PERF034 sweep measures the encoder, not the front door.
MAX_CONCURRENT_INPUTS = 64 if APP_NAME in PERF034_APP_PROFILES else 32

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
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
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
    cpu=8.0,
    memory=65_536,
    timeout=31 * 60,
    min_containers=MIN_CONTAINERS,
    scaledown_window=300,
    max_containers=1,
    enable_memory_snapshot=GPU_SNAPSHOT_ENABLED,
    experimental_options=(
        {"enable_gpu_snapshot": True} if GPU_SNAPSHOT_ENABLED else {}
    ),
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
class SessionEncoder:
    @modal.enter(snap=GPU_SNAPSHOT_ENABLED)
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
            enable_sleep_mode=GPU_SNAPSHOT_ENABLED,
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
        # Retains vLLM's own engine-sizing lines so the attention-block, mamba
        # page, KV-cache and concurrency figures become deployment-observed
        # instead of source-derived. An empty capture stays empty; the read-only
        # route reports it as `captured: false` rather than as an observation.
        with capture_startup_log() as startup_capture:
            self._engine = AsyncLLM.from_engine_args(engine_args)
        self._startup_log = tuple(startup_capture.lines)
        self._coordinator = SessionCoordinator(
            VLLMRetainedPoolingBackendFactory(self._engine),
            max_sessions=MAX_SESSIONS,
            max_resident_tokens=MAX_RESIDENT_TOKENS,
            idle_ttl_seconds=IDLE_TTL_SECONDS,
        )
        self._web_app = create_session_app(
            self._coordinator,
            TokenBlockSerializer(tokenizer),
            SessionAPIMetadata(
                engine_build_id=runtime_build_id,
                startup_log=self._startup_log,
            ),
            VLLMSessionEngineMetricsProvider(
                self._engine.get_scheduler_load,
                self._coordinator.append_metrics_snapshot,
            ),
        )
        if GPU_SNAPSHOT_ENABLED:
            asyncio.run(self._engine.sleep(level=1))

    @modal.enter(snap=False)
    def restore(self) -> None:
        if GPU_SNAPSHOT_ENABLED:
            asyncio.run(self._engine.wake_up())

    @modal.exit()
    async def stop(self) -> None:
        await self._coordinator.close_all()
        self._engine.shutdown()

    @modal.asgi_app(requires_proxy_auth=True)
    def web(self):
        return self._web_app
