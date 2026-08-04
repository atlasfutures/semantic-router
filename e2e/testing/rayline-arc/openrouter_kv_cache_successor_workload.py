#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Public three-worker workload and offline coverage gate for AGT018."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from openrouter_agentic_workload import WORKERS, candidate_case

SCHEMA_VERSION = "rayline.openrouter-kv-cache-successor-workload.v1"
OFFLINE_REPORT_SCHEMA_VERSION = "rayline.openrouter-kv-cache-offline-coverage.v1"
ENCODER_ARTIFACT_ID = "c82-reliability-distillation-gate1-20260720"
ENCODER_MODEL = "Qwen/Qwen3.5-0.8B"
ENCODER_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
ENCODER_DIMENSION = 1024
ENCODER_MANIFEST_SHA256 = (
    "05e1a23105ec9d537d6cc5b1da7a06b01c7536b6c773d119d967d397bb95e043"
)
ROUTING_AXIS_INDEX = 252
ROUTING_TARGETS = {
    "worker-a": 0.006,
    "worker-b": 0.0,
    "worker-c": -0.006,
}
MINIMUM_TOP_TWO_SCORE_GAP = 0.0015

EPISODES = 2
STEPS = 3
MODES = ("retained", "replay")
APPEND_LINES = 120
SEQUENCE_IDS = ("code_patch", "research_synthesis", "incident_correlation")
EXPECTED_SELECTION_TRACES = {
    "code_patch": ("worker-c",) * STEPS,
    "research_synthesis": ("worker-a",) * STEPS,
    "incident_correlation": ("worker-b",) * STEPS,
}
EXPECTED_REQUESTS_PER_DEPLOYMENT = len(SEQUENCE_IDS) * EPISODES * STEPS * len(MODES)


def _code_result(lines: int = 400) -> str:
    return "\n".join(
        f"src/router/session_{index % 7}.py:{20 + index}: "
        f"await state.commit(turn={index}, fenced=True)"
        for index in range(lines)
    )


def _research_result(lines: int = 310) -> str:
    return "\n".join(
        f"document-{index:03d}: queueing observation {index % 11}; "
        f"sample={1000 + index}; source=public-synthetic"
        for index in range(lines)
    )


def _incident_result(lines: int = 250) -> str:
    return "\n".join(
        f"2026-08-03T12:{index // 60:02d}:{index % 60:02d}Z "
        f"worker={index % 3} queue={index % 9} retained={index * 17} "
        f"event={'retry' if index % 37 == 0 else 'complete'}"
        for index in range(lines)
    )


def _incident_source_matches(lines: int = 70) -> str:
    return "\n".join(
        f"src/router/session_{index % 7}.py:{20 + index}: "
        f"await state.commit(turn={index}, fenced=True)"
        for index in range(lines)
    )


def _base_case(sequence_id: str) -> dict[str, Any]:
    index = {
        "code_patch": 0,
        "research_synthesis": 1,
        "incident_correlation": 2,
    }[sequence_id]
    case = copy.deepcopy(candidate_case(index))
    result = {
        "code_patch": _code_result(),
        "research_synthesis": _research_result(),
        "incident_correlation": (
            _incident_result()
            + "\ncorrelated source matches:\n"
            + _incident_source_matches()
        ),
    }[sequence_id]
    case.update(
        {
            "case_id": f"agt018-{sequence_id}-0",
            "scenario": sequence_id,
        }
    )
    case["messages"][3]["content"] = result
    return case


def _append_evidence(step: int) -> str:
    return "\n".join(
        f"public-cache-step={step} sample={index:03d} queue={index % 11} "
        f"retained={step * 1000 + index * 19} status=complete"
        for index in range(APPEND_LINES)
    )


def history_sequences() -> list[dict[str, Any]]:
    sequences: list[dict[str, Any]] = []
    for sequence_id in SEQUENCE_IDS:
        base = _base_case(sequence_id)
        messages = copy.deepcopy(base["messages"])
        states = []
        for step in range(STEPS):
            if step:
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": (
                                "Queueing remains the leading hypothesis; I will "
                                "inspect the next bounded public evidence block."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Update the diagnosis for cache step {step}.\n"
                                f"{_append_evidence(step)}"
                            ),
                        },
                    ]
                )
            states.append(
                {
                    **base,
                    "case_id": f"agt018-{sequence_id}-{step}",
                    "messages": copy.deepcopy(messages),
                    "expected_worker": EXPECTED_SELECTION_TRACES[sequence_id][step],
                }
            )
        sequences.append(
            {
                "sequence_id": sequence_id,
                "expected_selection_trace": list(
                    EXPECTED_SELECTION_TRACES[sequence_id]
                ),
                "states": states,
            }
        )
    return sequences


def normalized_turns(case: Mapping[str, Any]) -> list[dict[str, str]]:
    """Normalize the frozen Chat Completions subset to Rayline turns."""

    turns: list[dict[str, str]] = []
    tool_names: dict[str, str] = {}
    for message in case["messages"]:
        role = str(message["role"])
        if role in {"system", "developer"}:
            continue
        if role == "user":
            turns.append({"role": role, "text": str(message["content"])})
            continue
        if role == "assistant":
            parts = [str(message.get("content") or "")]
            for call in message.get("tool_calls") or []:
                function = call["function"]
                name = str(function["name"])
                tool_names[str(call["id"])] = name
                arguments = json.loads(str(function["arguments"]))
                rendered = json.dumps(
                    arguments,
                    sort_keys=True,
                    ensure_ascii=True,
                    separators=(", ", ": "),
                )
                parts.append(f"[tool_call {name}] {rendered}")
            turns.append(
                {"role": role, "text": "\n".join(part for part in parts if part)}
            )
            continue
        if role == "tool":
            name = tool_names[str(message["tool_call_id"])]
            turns.append(
                {
                    "role": "user",
                    "text": f"[tool_result {name}]\n{message['content']}",
                }
            )
            continue
        raise ValueError(f"unsupported frozen workload role {role!r}")
    return turns


def _normalized_axis(embedding: Sequence[float]) -> float:
    if len(embedding) != ENCODER_DIMENSION:
        raise RuntimeError("offline encoder returned the wrong embedding dimension")
    norm = math.sqrt(math.fsum(float(value) ** 2 for value in embedding))
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError("offline encoder returned a non-finite embedding")
    return float(embedding[ROUTING_AXIS_INDEX]) / norm


def classify_axis(axis: float) -> tuple[str, float]:
    scores = {
        worker: -abs(float(axis) - target) for worker, target in ROUTING_TARGETS.items()
    }
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return ordered[0][0], ordered[0][1] - ordered[1][1]


def evaluate_embeddings(
    embeddings: Mapping[tuple[str, int], Sequence[float]],
) -> dict[str, Any]:
    observations = []
    for sequence_id in SEQUENCE_IDS:
        for step in range(STEPS):
            axis = _normalized_axis(embeddings[(sequence_id, step)])
            selected, gap = classify_axis(axis)
            expected = EXPECTED_SELECTION_TRACES[sequence_id][step]
            observations.append(
                {
                    "sequence_id": sequence_id,
                    "step": step,
                    "selected_worker": selected,
                    "expected_worker": expected,
                    "normalized_routing_axis": axis,
                    "top_two_score_gap": gap,
                }
            )
    if any(row["selected_worker"] != row["expected_worker"] for row in observations):
        raise RuntimeError("offline natural selection trace diverged")
    minimum_gap = min(float(row["top_two_score_gap"]) for row in observations)
    if minimum_gap < MINIMUM_TOP_TWO_SCORE_GAP:
        raise RuntimeError("offline natural selection margin is too small")
    return {
        "observations": observations,
        "minimum_top_two_score_gap": minimum_gap,
        "selected_workers": sorted({row["selected_worker"] for row in observations}),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_paths(runtime_dir: Path) -> tuple[dict[str, Any], Path, Path]:
    manifest_path = runtime_dir / "manifest.json"
    if _sha256(manifest_path) != ENCODER_MANIFEST_SHA256:
        raise RuntimeError("offline encoder manifest identity diverged")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("artifact_id") != ENCODER_ARTIFACT_ID
        or manifest.get("encoder", {}).get("model") != ENCODER_MODEL
        or manifest.get("encoder", {}).get("revision") != ENCODER_REVISION
    ):
        raise RuntimeError("offline encoder artifact contract diverged")
    native = manifest["encoder"]["native"]
    binary_row = next(
        row for row in native["binaries"] if row["target"] == "aarch64-apple-darwin"
    )
    binary = runtime_dir / binary_row["file"]
    model = runtime_dir / native["gguf"]["file"]
    if (
        _sha256(binary) != binary_row["sha256"]
        or _sha256(model) != native["gguf"]["sha256"]
    ):
        raise RuntimeError("offline encoder binary or GGUF identity diverged")
    return manifest, binary, model


def verify_runtime(runtime_dir: Path) -> dict[str, Any]:
    manifest, binary, model = _runtime_paths(runtime_dir)
    process = subprocess.Popen(
        [str(binary), "--model", str(model), "--threads", "8"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("offline encoder process did not expose IPC streams")
    embeddings: dict[tuple[str, int], Sequence[float]] = {}
    telemetry: dict[tuple[str, int], dict[str, Any]] = {}
    try:
        for sequence in history_sequences():
            sequence_id = str(sequence["sequence_id"])
            for step, case in enumerate(sequence["states"]):
                request = {
                    "op": "encode",
                    "episode_id": f"agt018-offline-{sequence_id}",
                    "turns": normalized_turns(case),
                }
                process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
                process.stdin.flush()
                response = json.loads(process.stdout.readline())
                if not response.get("ok"):
                    raise RuntimeError(
                        "offline encoder rejected a frozen workload state"
                    )
                result = response["result"]
                embeddings[(sequence_id, step)] = result["embedding"]
                telemetry[(sequence_id, step)] = {
                    "encode_mode": result["encode_mode"],
                    "serialized_tokens": int(result["serialized_tokens"]),
                    "cached_prefix_tokens": int(result["cached_prefix_tokens"]),
                }
    finally:
        process.terminate()
        process.wait(timeout=5)
    evaluated = evaluate_embeddings(embeddings)
    for row in evaluated["observations"]:
        key = (str(row["sequence_id"]), int(row["step"]))
        row.update(telemetry[key])
    return {
        "schema_version": OFFLINE_REPORT_SCHEMA_VERSION,
        "status": "passed",
        "workload_schema_version": SCHEMA_VERSION,
        "encoder": {
            "artifact_id": manifest["artifact_id"],
            "model": ENCODER_MODEL,
            "revision": ENCODER_REVISION,
            "manifest_sha256": ENCODER_MANIFEST_SHA256,
            "runtime": "llama_cpp_native",
            "device": "metal",
        },
        "expected_selection_traces": {
            key: list(value) for key, value in EXPECTED_SELECTION_TRACES.items()
        },
        **evaluated,
        "all_workers_covered": set(evaluated["selected_workers"]) == set(WORKERS),
        "prompt_text_emitted": False,
        "raw_embeddings_emitted": False,
        "provider_calls": 0,
        "external_gpu_calls": 0,
        "release_qualification_1000_executed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_runtime(args.runtime_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
