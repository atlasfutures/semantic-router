# SPDX-License-Identifier: Apache-2.0

"""GPU/runtime helpers for the Rayline ARC Modal correctness canary.

These helpers run only inside the ephemeral GPU container. They start
loopback-only vLLM servers, exercise synthetic inputs, and return bounded
metrics without returning embeddings.
"""

from __future__ import annotations

import hashlib
import importlib
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

APP_NAME = "rayline-arc-rung-a-canary-dev"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
TOKENIZER_REVISION = MODEL_REVISION
TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
EOS_TOKEN_ID = 248046
VLLM_COMMIT = "98e91a9600eb75b2de14ef27f13b10088d1a1279"
VLLM_RUNG_B_COMMIT = "8faf2388c2fab4e86ca37778e74665ac23b3eba4"
VLLM_RUNG_B_BRANCH = "rayline/pl-0039-causal-mean"
VLLM_RUNG_B_REPOSITORY = "https://github.com/davidvgilmore/vllm.git"
VLLM_VERSION = "0.26.1rc1.dev36+g98e91a960"
VLLM_WHEEL_INDEX = f"https://wheels.vllm.ai/{VLLM_COMMIT}/cu130"
ENGINE_BUILD_ID = f"vllm@{VLLM_COMMIT}"
RUNG_B_ENGINE_BUILD_ID = f"vllm@{VLLM_RUNG_B_COMMIT}"
PLUGIN_VERSION = "rayline-arc-io@0.1.0"
SERIALIZER_VERSION = "mtrouter-token-blocks-v2"
REQUEST_SCHEMA_VERSION = "rayline.arc.pooling-request.v1"
MAX_SERIALIZED_TOKENS = 262_144
EMBEDDING_DIMENSION = 1024

GPU_TYPE = "H100"
FUNCTION_TIMEOUT_SECONDS = 20 * 60
CPU_CORES = 8.0
MEMORY_MIB = 65_536
COST_CEILING_USD = 1.65
PRICING_SNAPSHOT_DATE = "2026-07-28"
H100_USD_PER_SECOND = 0.001097
CPU_CORE_USD_PER_SECOND = 0.0000131
MEMORY_GIB_USD_PER_SECOND = 0.00000222

STARTUP_TIMEOUT_SECONDS = 10 * 60
SHUTDOWN_TIMEOUT_SECONDS = 60
POLL_SECONDS = 1.0
HTTP_OK = 200
MAX_HTTP_ERROR_BYTES = 4096
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
RUNG_B_POOLER_CONFIG = {
    "task": "embed",
    "pooling_type": "MEAN",
    "use_activation": True,
    "enable_chunked_processing": False,
}
RUNG_B_RUNTIME_FILES = (
    "vllm/config/model.py",
    "vllm/model_executor/layers/pooler/seqwise/heads.py",
    "vllm/model_executor/layers/pooler/seqwise/methods.py",
    "vllm/model_executor/layers/pooler/seqwise/poolers.py",
    "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/pool/metadata.py",
    "vllm/v1/worker/gpu_input_batch.py",
)
RUNG_B_BOUNDARY_LENGTHS = (
    253_952,
    258_048,
    260_096,
    261_120,
    262_143,
    262_144,
)
RUNG_B_BOUNDARY_TIMEOUT_SECONDS = 120

SHAPES = {
    "short": {"repeats": 32, "repetitions": 3, "timeout_seconds": 120},
    "multi_chunk": {
        "repeats": 6000,
        "repetitions": 3,
        "timeout_seconds": 300,
    },
    "maximum_contract": {
        "repeats": 90_000,
        "repetitions": 1,
        "timeout_seconds": 300,
    },
}
PRIVACY_MARKER = "ARC_PRIVACY_CANARY_7f03c1"


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
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        body = error.read(MAX_HTTP_ERROR_BYTES).decode(errors="replace")
        error.close()
        redacted_body = body.replace(PRIVACY_MARKER, "<redacted>")
        raise RuntimeError(
            f"vLLM /pooling returned HTTP {error.code}: {redacted_body}"
        ) from None


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


def _rung_b_server_command(port: int, schedule_tokens: int) -> list[str]:
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
        "--pooler-config",
        json.dumps(RUNG_B_POOLER_CONFIG, separators=(",", ":")),
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


def _request(
    run_id: str,
    mode: str,
    shape: str,
    repetition: int,
    *,
    serving_rung: str,
) -> dict[str, Any]:
    correlation = f"{run_id}:{mode}:{shape}:{repetition}".encode()
    return {
        "task": "plugin",
        "data": {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "serializer_version": SERIALIZER_VERSION,
            "serving_rung": serving_rung,
            "episode_id_hash": hashlib.sha256(correlation).hexdigest(),
            "turns": [
                {
                    "role": "user",
                    "text": _turn_text(SHAPES[shape]["repeats"]),
                }
            ],
        },
    }


def _rung_b_token_ids(shape: str) -> list[int]:
    # These dependencies are installed only in the remote Modal image.
    arc_turn = importlib.import_module("rayline_arc_io.schemas").ArcTurn
    serializer = importlib.import_module(
        "rayline_arc_io.serializer"
    ).TokenBlockSerializer
    auto_tokenizer = importlib.import_module("transformers").AutoTokenizer

    tokenizer = auto_tokenizer.from_pretrained(
        MODEL_ID,
        revision=TOKENIZER_REVISION,
        split_special_tokens=True,
    )
    tokenizer.split_special_tokens = True
    tokenization = serializer(tokenizer).tokenize(
        [arc_turn(role="user", text=_turn_text(SHAPES[shape]["repeats"]))]
    )
    return list(tokenization.input_ids)


def _rung_b_request(
    token_ids: list[int],
    *,
    use_activation: bool = True,
) -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "input": token_ids,
        "task": "embed",
        "add_special_tokens": False,
        "use_activation": use_activation,
    }


def _validate_rung_b_response(
    raw: dict[str, Any],
    *,
    expected_tokens: int,
    require_normalized: bool = True,
) -> tuple[list[float], dict[str, Any]]:
    data = raw.get("data")
    if not isinstance(data, list) or len(data) != 1:
        raise TypeError("vLLM Rung B response must contain exactly one output")
    item = data[0]
    if not isinstance(item, dict) or item.get("index") != 0:
        raise TypeError("vLLM Rung B response output index is invalid")
    embedding = item.get("data")
    if (
        not isinstance(embedding, list)
        or len(embedding) != EMBEDDING_DIMENSION
        or any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in embedding
        )
    ):
        raise ValueError("vLLM Rung B returned an invalid embedding")
    usage = raw.get("usage")
    if not isinstance(usage, dict) or usage.get("prompt_tokens") != expected_tokens:
        raise ValueError("vLLM Rung B token accounting does not match the serializer")
    normalized = [float(value) for value in embedding]
    norm = math.sqrt(math.fsum(value * value for value in normalized))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError(f"vLLM Rung B embedding norm is invalid: {norm}")
    if require_normalized and abs(norm - 1.0) > NORM_TOLERANCE:
        raise ValueError(f"vLLM Rung B embedding norm is {norm}, expected 1")
    return normalized, {
        "serialized_tokens": expected_tokens,
        "norm": norm,
        "cached_prefix_tokens": 0,
    }


def _validate_response(
    raw: dict[str, Any],
    *,
    engine_build_id: str,
    pooling_capabilities: list[str],
) -> tuple[list[float], dict[str, Any]]:
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
        "serializer_version": SERIALIZER_VERSION,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "eos_token_id": EOS_TOKEN_ID,
        "engine_build_id": engine_build_id,
        "io_plugin_version": PLUGIN_VERSION,
        "pooling_capabilities": pooling_capabilities,
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
    serving_rung: str = "A",
) -> tuple[dict[str, Any], dict[str, list[list[float]]]]:
    log_path = Path(f"/tmp/rayline-arc-{mode}.log")
    if serving_rung == "A":
        command = _server_command(port, schedule_tokens)
        engine_build_id = ENGINE_BUILD_ID
        pooling_capabilities = ["all_plugin_mean"]
    elif serving_rung == "B":
        command = [
            *_rung_b_server_command(port, schedule_tokens),
            "--io-processor-plugin",
            "rayline_arc_io",
        ]
        engine_build_id = RUNG_B_ENGINE_BUILD_ID
        pooling_capabilities = ["chunked_causal_mean"]
    else:
        raise ValueError(f"unsupported plugin canary serving rung {serving_rung!r}")
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            command,
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
                    print(
                        "rayline ARC canary request start: "
                        f"mode={mode} shape={shape} repetition={repetition}",
                        flush=True,
                    )
                    started = time.monotonic()
                    raw = _post_json(
                        f"http://127.0.0.1:{port}/pooling",
                        _request(
                            run_id,
                            mode,
                            shape,
                            repetition,
                            serving_rung=serving_rung,
                        ),
                        timeout_seconds=spec["timeout_seconds"],
                    )
                    elapsed = time.monotonic() - started
                    embedding, metadata = _validate_response(
                        raw,
                        engine_build_id=engine_build_id,
                        pooling_capabilities=pooling_capabilities,
                    )
                    print(
                        "rayline ARC canary request complete: "
                        f"mode={mode} shape={shape} repetition={repetition} "
                        f"serialized_tokens={metadata['serialized_tokens']} "
                        f"elapsed_seconds={elapsed:.3f}",
                        flush=True,
                    )
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


def _run_rung_b_server_mode(
    *,
    mode: str,
    port: int,
    schedule_tokens: int,
    shapes: tuple[str, ...],
) -> tuple[
    dict[str, Any],
    dict[str, list[list[float]]],
    dict[str, list[float]],
]:
    log_path = Path(f"/tmp/rayline-arc-{mode}.log")
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            _rung_b_server_command(port, schedule_tokens),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            startup_seconds = _wait_until_ready(process, port, log_path)
            _read_json(f"http://127.0.0.1:{port}/v1/models", timeout_seconds=30)
            measurements: list[dict[str, Any]] = []
            vectors: dict[str, list[list[float]]] = {}
            raw_mean_vectors: dict[str, list[float]] = {}
            for shape in shapes:
                token_ids = _rung_b_token_ids(shape)
                if shape == "maximum_contract" and (
                    len(token_ids) != MAX_SERIALIZED_TOKENS
                ):
                    raise RuntimeError(
                        "Rung B maximum-contract input did not serialize "
                        "to 262,144 tokens"
                    )
                vectors[shape] = []
                spec = SHAPES[shape]
                raw_embedding, raw_measurement = _measure_raw_rung_b_mean(
                    port,
                    shape,
                    token_ids,
                    spec["timeout_seconds"],
                )
                raw_mean_vectors[shape] = raw_embedding
                measurements.append(raw_measurement)
                for repetition in range(spec["repetitions"]):
                    print(
                        "rayline ARC Rung B request start: "
                        f"mode={mode} shape={shape} repetition={repetition}",
                        flush=True,
                    )
                    started = time.monotonic()
                    raw = _post_json(
                        f"http://127.0.0.1:{port}/pooling",
                        _rung_b_request(token_ids),
                        timeout_seconds=spec["timeout_seconds"],
                    )
                    elapsed = time.monotonic() - started
                    embedding, metadata = _validate_rung_b_response(
                        raw,
                        expected_tokens=len(token_ids),
                    )
                    print(
                        "rayline ARC Rung B request complete: "
                        f"mode={mode} shape={shape} repetition={repetition} "
                        f"serialized_tokens={metadata['serialized_tokens']} "
                        f"elapsed_seconds={elapsed:.3f}",
                        flush=True,
                    )
                    vectors[shape].append(embedding)
                    measurements.append(
                        {
                            "shape": shape,
                            "repetition": repetition,
                            "use_activation": True,
                            "elapsed_seconds": elapsed,
                            **metadata,
                        }
                    )
        finally:
            _stop_server(process)

    log_bytes = log_path.read_bytes()
    if PRIVACY_MARKER.encode() in log_bytes:
        raise RuntimeError("raw ARC prompt content appeared in Rung B vLLM logs")
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
        raw_mean_vectors,
    )


def _measure_raw_rung_b_mean(
    port: int,
    shape: str,
    token_ids: list[int],
    timeout_seconds: int,
) -> tuple[list[float], dict[str, Any]]:
    started = time.monotonic()
    response = _post_json(
        f"http://127.0.0.1:{port}/pooling",
        _rung_b_request(token_ids, use_activation=False),
        timeout_seconds=timeout_seconds,
    )
    elapsed = time.monotonic() - started
    embedding, metadata = _validate_rung_b_response(
        response,
        expected_tokens=len(token_ids),
        require_normalized=False,
    )
    return embedding, {
        "shape": shape,
        "repetition": 0,
        "use_activation": False,
        "elapsed_seconds": elapsed,
        **metadata,
    }


def _run_rung_b_boundary_probe() -> dict[str, Any]:
    mode = "rung-b-boundary-probe"
    port = 8003
    log_path = Path(f"/tmp/rayline-arc-{mode}.log")
    max_token_ids = _rung_b_token_ids("maximum_contract")
    if len(max_token_ids) != MAX_SERIALIZED_TOKENS:
        raise RuntimeError("boundary probe did not serialize the maximum contract")

    measurements: list[dict[str, Any]] = []
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            _rung_b_server_command(port, CHUNK_SCHEDULE_TOKENS),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            startup_seconds = _wait_until_ready(process, port, log_path)
            for token_count in RUNG_B_BOUNDARY_LENGTHS:
                token_ids = [
                    *max_token_ids[: token_count - 1],
                    EOS_TOKEN_ID,
                ]
                print(
                    "rayline ARC Rung B boundary request start: "
                    f"serialized_tokens={token_count}",
                    flush=True,
                )
                started = time.monotonic()
                try:
                    raw = _post_json(
                        f"http://127.0.0.1:{port}/pooling",
                        _rung_b_request(token_ids),
                        timeout_seconds=RUNG_B_BOUNDARY_TIMEOUT_SECONDS,
                    )
                    elapsed = time.monotonic() - started
                    _validate_rung_b_response(raw, expected_tokens=token_count)
                except (OSError, TimeoutError) as error:
                    elapsed = time.monotonic() - started
                    measurements.append(
                        {
                            "serialized_tokens": token_count,
                            "elapsed_seconds": elapsed,
                            "status": "timeout",
                            "error_type": type(error).__name__,
                            "server_running": process.poll() is None,
                        }
                    )
                    print(
                        "rayline ARC Rung B boundary request timeout: "
                        f"serialized_tokens={token_count} "
                        f"elapsed_seconds={elapsed:.3f}",
                        flush=True,
                    )
                    break
                measurements.append(
                    {
                        "serialized_tokens": token_count,
                        "elapsed_seconds": elapsed,
                        "status": "passed",
                    }
                )
                print(
                    "rayline ARC Rung B boundary request complete: "
                    f"serialized_tokens={token_count} "
                    f"elapsed_seconds={elapsed:.3f}",
                    flush=True,
                )
        finally:
            _stop_server(process)

    log_bytes = log_path.read_bytes()
    if PRIVACY_MARKER.encode() in log_bytes:
        raise RuntimeError("raw ARC prompt content appeared in boundary-probe logs")
    return {
        "mode": mode,
        "schedule_tokens": CHUNK_SCHEDULE_TOKENS,
        "request_timeout_seconds": RUNG_B_BOUNDARY_TIMEOUT_SECONDS,
        "startup_seconds": startup_seconds,
        "measurements": measurements,
        "log_sha256": hashlib.sha256(log_bytes).hexdigest(),
        "log_tail": _log_tail(log_path).replace(PRIVACY_MARKER, "<redacted>"),
        "privacy_scan_passed": True,
    }
