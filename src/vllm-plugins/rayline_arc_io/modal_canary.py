# SPDX-License-Identifier: Apache-2.0

"""Modal CUDA correctness canary for the Rayline ARC Rung A IO plugin.

This is an ephemeral batch app: it starts two loopback-only vLLM servers,
exercises the public synthetic ARC contract, and returns bounded metrics. It
does not deploy a serving endpoint, call a provider, log prompts, or return
embeddings.

Usage from a Python environment containing Modal:

    modal run modal_canary.py --run-id rayline-arc-rung-a-YYYYMMDD \
        --output /tmp/rayline-arc-rung-a.json
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Self

import modal

APP_NAME = "rayline-arc-rung-a-canary-dev"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
VLLM_COMMIT = "98e91a9600eb75b2de14ef27f13b10088d1a1279"
VLLM_VERSION = "0.26.1rc1.dev36+g98e91a960"
VLLM_WHEEL_INDEX = f"https://wheels.vllm.ai/{VLLM_COMMIT}/cu130"
ENGINE_BUILD_ID = f"vllm@{VLLM_COMMIT}"
PLUGIN_VERSION = "rayline-arc-io@0.1.0"
SERIALIZER_VERSION = "mtrouter-token-blocks-v2"
REQUEST_SCHEMA_VERSION = "rayline.arc.pooling-request.v1"
MAX_SERIALIZED_TOKENS = 262_144
EMBEDDING_DIMENSION = 1024

GPU_TYPE = "H100"
FUNCTION_TIMEOUT_SECONDS = 33 * 60
CPU_CORES = 8.0
MEMORY_MIB = 65_536
COST_CEILING_USD = 2.70
PRICING_SNAPSHOT_DATE = "2026-07-28"
H100_USD_PER_SECOND = 0.001097
CPU_CORE_USD_PER_SECOND = 0.0000131
MEMORY_GIB_USD_PER_SECOND = 0.00000222

STARTUP_TIMEOUT_SECONDS = 10 * 60
SHUTDOWN_TIMEOUT_SECONDS = 60
POLL_SECONDS = 1.0
HTTP_OK = 200
GPU_QUERY_FIELD_COUNT = 4
NORM_TOLERANCE = 1e-5
FULL_SCHEDULE_TOKENS = MAX_SERIALIZED_TOKENS
CHUNK_SCHEDULE_TOKENS = 8192
POOLER_CONFIG = {
    "task": "token_embed",
    "tok_pooling_type": "ALL",
    "use_activation": False,
    "enable_chunked_processing": False,
}

SHAPES = {
    "short": {"repeats": 32, "repetitions": 3, "timeout_seconds": 120},
    "multi_chunk": {
        "repeats": 6000,
        "repetitions": 3,
        "timeout_seconds": 300,
    },
    "maximum_contract": {
        "repeats": 90_000,
        "repetitions": 2,
        "timeout_seconds": 900,
    },
}
PRIVACY_MARKER = "ARC_PRIVACY_CANARY_7f03c1"

_THIS_DIR = Path(__file__).resolve().parent
_REMOTE_PLUGIN_DIR = "/opt/rayline_arc_io"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-devel-ubuntu22.04",
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
        }
    )
)

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("rayline-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("rayline-vllm-cache", create_if_missing=True)


def _estimated_cost(elapsed_seconds: float) -> float:
    memory_gib = MEMORY_MIB / 1024
    rate = (
        H100_USD_PER_SECOND
        + CPU_CORES * CPU_CORE_USD_PER_SECOND
        + memory_gib * MEMORY_GIB_USD_PER_SECOND
    )
    return elapsed_seconds * rate


def _read_json(url: str, *, timeout_seconds: int) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read())


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read())


def _server_command(port: int, schedule_tokens: int) -> list[str]:
    return [
        "vllm",
        "serve",
        MODEL_ID,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
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
        str(schedule_tokens),
        "--max-num-seqs",
        "1",
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization",
        "0.92",
        "--io-processor-plugin",
        "rayline_arc_io",
        "--pooler-config",
        json.dumps(POOLER_CONFIG, separators=(",", ":")),
        "--no-enable-log-requests",
    ]


def _log_tail(log_path: Path, max_lines: int = 80) -> str:
    if not log_path.exists():
        return ""
    return "\n".join(log_path.read_text(errors="replace").splitlines()[-max_lines:])


def _wait_until_ready(
    process: subprocess.Popen[bytes],
    port: int,
    log_path: Path,
) -> float:
    started = time.monotonic()
    deadline = started + STARTUP_TIMEOUT_SECONDS
    health_url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM exited during startup with {process.returncode}:\n"
                f"{_log_tail(log_path)}"
            )
        try:
            request = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status == HTTP_OK:
                    return time.monotonic() - started
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"vLLM did not become ready:\n{_log_tail(log_path)}")


def _stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)


class _GpuMonitor:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self.samples = 0
        self.max_memory_used_mib = 0
        self.max_utilization_percent = 0
        self.gpu_name = ""
        self.total_memory_mib = 0

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _poll(self) -> None:
        while not self._stop.is_set():
            command = [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                fields = [
                    field.strip() for field in result.stdout.splitlines()[0].split(",")
                ]
                if len(fields) == GPU_QUERY_FIELD_COUNT:
                    self.gpu_name = fields[0]
                    self.total_memory_mib = int(fields[1])
                    self.max_memory_used_mib = max(
                        self.max_memory_used_mib,
                        int(fields[2]),
                    )
                    self.max_utilization_percent = max(
                        self.max_utilization_percent,
                        int(fields[3]),
                    )
                    self.samples += 1
            self._stop.wait(POLL_SECONDS)


def _turn_text(repeats: int) -> str:
    return (f" {PRIVACY_MARKER} rayline arc canary") * repeats


def _request(run_id: str, mode: str, shape: str, repetition: int) -> dict[str, Any]:
    correlation = f"{run_id}:{mode}:{shape}:{repetition}".encode()
    return {
        "task": "plugin",
        "data": {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "serializer_version": SERIALIZER_VERSION,
            "serving_rung": "A",
            "episode_id_hash": hashlib.sha256(correlation).hexdigest(),
            "turns": [
                {
                    "role": "user",
                    "text": _turn_text(SHAPES[shape]["repeats"]),
                }
            ],
        },
    }


def _validate_response(raw: dict[str, Any]) -> tuple[list[float], dict[str, Any]]:
    data = raw.get("data")
    if not isinstance(data, dict):
        raise TypeError("vLLM plugin response is missing its data object")
    embedding = data.pop("embedding", None)
    if (
        not isinstance(embedding, list)
        or len(embedding) != EMBEDDING_DIMENSION
        or any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in embedding
        )
    ):
        raise ValueError("vLLM plugin returned an invalid embedding")
    expected = {
        "model_revision": MODEL_REVISION,
        "engine_build_id": ENGINE_BUILD_ID,
        "io_plugin_version": PLUGIN_VERSION,
        "pooling_capabilities": ["all_plugin_mean"],
        "cached_prefix_tokens": 0,
    }
    for field, wanted in expected.items():
        if data.get(field) != wanted:
            raise ValueError(
                f"vLLM plugin response {field}={data.get(field)!r}, expected {wanted!r}"
            )
    serialized_tokens = data.get("serialized_tokens")
    if not isinstance(serialized_tokens, int) or serialized_tokens <= 0:
        raise ValueError("vLLM plugin returned an invalid serialized token count")
    norm = math.sqrt(math.fsum(float(value) ** 2 for value in embedding))
    if not math.isfinite(norm) or abs(norm - 1.0) > NORM_TOLERANCE:
        raise ValueError(f"vLLM plugin embedding norm is {norm}, expected 1")
    return [float(value) for value in embedding], {**data, "norm": norm}


def _run_server_mode(
    *,
    run_id: str,
    mode: str,
    port: int,
    schedule_tokens: int,
    shapes: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, list[list[float]]]]:
    log_path = Path(f"/tmp/rayline-arc-{mode}.log")
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            _server_command(port, schedule_tokens),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            startup_seconds = _wait_until_ready(process, port, log_path)
            _read_json(f"http://127.0.0.1:{port}/v1/models", timeout_seconds=30)
            measurements: list[dict[str, Any]] = []
            vectors: dict[str, list[list[float]]] = {}
            for shape in shapes:
                vectors[shape] = []
                spec = SHAPES[shape]
                for repetition in range(spec["repetitions"]):
                    started = time.monotonic()
                    raw = _post_json(
                        f"http://127.0.0.1:{port}/pooling",
                        _request(run_id, mode, shape, repetition),
                        timeout_seconds=spec["timeout_seconds"],
                    )
                    elapsed = time.monotonic() - started
                    embedding, metadata = _validate_response(raw)
                    vectors[shape].append(embedding)
                    measurements.append(
                        {
                            "shape": shape,
                            "repetition": repetition,
                            "elapsed_seconds": elapsed,
                            **metadata,
                        }
                    )
        finally:
            _stop_server(process)

    log_bytes = log_path.read_bytes()
    if PRIVACY_MARKER.encode() in log_bytes:
        raise RuntimeError("raw ARC prompt content appeared in vLLM logs")
    return (
        {
            "mode": mode,
            "schedule_tokens": schedule_tokens,
            "startup_seconds": startup_seconds,
            "measurements": measurements,
            "log_sha256": hashlib.sha256(log_bytes).hexdigest(),
            "privacy_scan_passed": True,
        },
        vectors,
    )


def _numeric_delta(reference: list[float], candidate: list[float]) -> dict[str, Any]:
    differences = [
        right - left for left, right in zip(reference, candidate, strict=True)
    ]
    dot = math.fsum(
        left * right for left, right in zip(reference, candidate, strict=True)
    )
    return {
        "max_abs": max(abs(value) for value in differences),
        "l2": math.sqrt(math.fsum(value * value for value in differences)),
        "cosine_distance": abs(1.0 - dot),
        "synthetic_scores": _synthetic_score_delta(reference, candidate),
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
    return {
        "max_abs": max(abs(a - b) for a, b in zip(left, right, strict=True)),
        "reference_top_two_gap": ordered[0] - ordered[1],
        "selected_arm_reference": max(range(len(left)), key=left.__getitem__),
        "selected_arm_candidate": max(range(len(right)), key=right.__getitem__),
    }


def _summarize_numeric(
    full_vectors: dict[str, list[list[float]]],
    chunked_vectors: dict[str, list[list[float]]],
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

    cross_mode = []
    for shape in ("short", "multi_chunk"):
        cross_mode.append(
            {
                "shape": shape,
                **_numeric_delta(full_vectors[shape][0], chunked_vectors[shape][0]),
            }
        )

    all_deltas = [*repeat_deltas, *cross_mode]
    if any(
        row["synthetic_scores"]["selected_arm_reference"]
        != row["synthetic_scores"]["selected_arm_candidate"]
        for row in all_deltas
    ):
        raise RuntimeError("Rung A canary changed the synthetic selected arm")
    observed_embedding_max_abs = max(row["max_abs"] for row in all_deltas)
    observed_embedding_l2 = max(row["l2"] for row in all_deltas)
    observed_cosine_distance = max(row["cosine_distance"] for row in all_deltas)
    observed_score_max_abs = max(
        row["synthetic_scores"]["max_abs"] for row in all_deltas
    )
    min_top_two_gap = min(
        row["synthetic_scores"]["reference_top_two_gap"] for row in all_deltas
    )
    return {
        "repeat_deltas": repeat_deltas,
        "cross_mode_deltas": cross_mode,
        "observed_maxima": {
            "embedding_max_abs": observed_embedding_max_abs,
            "embedding_l2": observed_embedding_l2,
            "cosine_distance": observed_cosine_distance,
            "synthetic_score_max_abs": observed_score_max_abs,
            "synthetic_min_top_two_gap": min_top_two_gap,
            "selected_arm_parity": 1.0,
        },
    }


@app.function(
    gpu=GPU_TYPE,
    image=image,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    cpu=CPU_CORES,
    memory=MEMORY_MIB,
    timeout=FUNCTION_TIMEOUT_SECONDS,
)
def canary(run_id: str) -> dict[str, Any]:
    started = time.monotonic()
    with _GpuMonitor() as gpu:
        full_report, full_vectors = _run_server_mode(
            run_id=run_id,
            mode="single-schedule",
            port=8001,
            schedule_tokens=FULL_SCHEDULE_TOKENS,
            shapes=("short", "multi_chunk"),
        )
        chunked_report, chunked_vectors = _run_server_mode(
            run_id=run_id,
            mode="chunked-8192",
            port=8002,
            schedule_tokens=CHUNK_SCHEDULE_TOKENS,
            shapes=("short", "multi_chunk", "maximum_contract"),
        )
    elapsed_seconds = time.monotonic() - started
    estimated_cost = _estimated_cost(elapsed_seconds)
    if estimated_cost > COST_CEILING_USD:
        raise RuntimeError(
            f"estimated Modal cost ${estimated_cost:.4f} exceeded "
            f"${COST_CEILING_USD:.2f} ceiling"
        )

    hf_cache.commit()
    vllm_cache.commit()
    maximum_measurements = [
        row
        for row in chunked_report["measurements"]
        if row["shape"] == "maximum_contract"
    ]
    if any(
        row["serialized_tokens"] != MAX_SERIALIZED_TOKENS
        for row in maximum_measurements
    ):
        raise RuntimeError("maximum-contract requests did not reach 262,144 tokens")

    return {
        "schema_version": "rayline.arc.rung-a-modal-canary.v1",
        "run_id": run_id,
        "vllm_commit": VLLM_COMMIT,
        "vllm_version": VLLM_VERSION,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "plugin_version": PLUGIN_VERSION,
        "serializer_version": SERIALIZER_VERSION,
        "dtype": "bfloat16",
        "apc_enabled": False,
        "request_logging_enabled": False,
        "gpu": {
            "requested_type": GPU_TYPE,
            "observed_name": gpu.gpu_name,
            "total_memory_mib": gpu.total_memory_mib,
            "max_memory_used_mib": gpu.max_memory_used_mib,
            "max_utilization_percent": gpu.max_utilization_percent,
            "samples": gpu.samples,
        },
        "servers": [full_report, chunked_report],
        "numeric": _summarize_numeric(full_vectors, chunked_vectors),
        "elapsed_seconds": elapsed_seconds,
        "cost": {
            "pricing_snapshot_date": PRICING_SNAPSHOT_DATE,
            "pricing_source": "https://modal.com/pricing",
            "ceiling_usd": COST_CEILING_USD,
            "estimated_usd": estimated_cost,
            "provider_spend_usd": 0.0,
            "h100_usd_per_second": H100_USD_PER_SECOND,
            "cpu_core_usd_per_second": CPU_CORE_USD_PER_SECOND,
            "memory_gib_usd_per_second": MEMORY_GIB_USD_PER_SECOND,
        },
    }


@app.local_entrypoint()
def main(run_id: str, output: str) -> None:
    report = canary.remote(run_id=run_id)
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "run_id": report["run_id"],
                "output": str(output_path),
                "gpu": report["gpu"],
                "elapsed_seconds": report["elapsed_seconds"],
                "estimated_cost_usd": report["cost"]["estimated_usd"],
                "observed_maxima": report["numeric"]["observed_maxima"],
            },
            indent=2,
            sort_keys=True,
        )
    )
