#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Build the frozen 128-decision three-arm Rayline parity packet.

The source is Pathfinder's existing public synthetic vLLM parity corpus. This
adapter deliberately does not regenerate prompts or policy data: it verifies
the content-addressed source, selects complete four-turn episodes, and renders
the small protocol-neutral packet consumed by ``rayline_parity_http_probe``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rayline_development_artifact import MANIFEST_SCHEMA
from rayline_parity_http_probe import (
    CORPUS_SCHEMA,
    TOPOLOGY_SCHEMA,
    WORKLOAD_SCHEMA,
)

SOURCE_CORPUS_SCHEMA = "rayline.mtrouter-vllm-parity-corpus.v1"
SOURCE_CASE_SCHEMA = "rayline.mtrouter-vllm-parity-case.v1"
PACKET_SCHEMA = "rayline.vllm.three-arm-packet.v1"
ARMS = ("modal_inprocess", "rayline_remote", "rayline_arc")
MEASURED_CASES = 128
WARMUP_CASES = 8
CONCURRENCY = 8
DECISIONS_PER_EPISODE = 4


class PacketError(ValueError):
    """The public corpus and runtime cannot form an identity-matched packet."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PacketError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise PacketError(f"{label} must be a JSON object")
    return value


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise PacketError(
                        f"source corpus line {line_number} is invalid JSON"
                    ) from error
                if not isinstance(row, dict):
                    raise PacketError(
                        f"source corpus line {line_number} is not an object"
                    )
                rows.append(row)
    except OSError as error:
        raise PacketError(f"cannot read source corpus: {error}") from error
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _runtime_identity(runtime_dir: Path) -> tuple[list[str], dict[str, Any]]:
    manifest = _read_object(runtime_dir / "manifest.json", "runtime manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise PacketError("runtime manifest schema is unsupported")
    workers = manifest.get("workers")
    encoder = manifest.get("encoder")
    if not isinstance(workers, list) or not isinstance(encoder, Mapping):
        raise PacketError("runtime manifest omits workers/encoder")
    worker_ids = [str(worker.get("id") or "") for worker in workers]
    if not worker_ids or "" in worker_ids or len(worker_ids) != len(set(worker_ids)):
        raise PacketError("runtime worker ids must be non-empty and unique")
    checkpoint = manifest.get("source", {}).get("checkpoint", {})
    checkpoint_sha = str(checkpoint.get("sha256") or "")
    if len(checkpoint_sha) != 64:
        raise PacketError("runtime manifest omits the source checkpoint digest")
    return worker_ids, {
        "model": str(encoder.get("model") or ""),
        "revision": str(encoder.get("revision") or ""),
        "serialization": str(encoder.get("serialization") or ""),
        "checkpoint_sha256": checkpoint_sha,
    }


def _case(row: Mapping[str, Any], label: str) -> dict[str, Any]:
    if row.get("schema_version") != SOURCE_CASE_SCHEMA:
        raise PacketError(f"{label} source case schema is unsupported")
    case_id = str(row.get("decision_id") or "")
    episode_id = str(row.get("episode_affinity_hash") or "")
    sequence_index = row.get("sequence_index")
    turns = row.get("turns")
    token_counts = row.get("token_counts")
    input_tokens = (
        token_counts.get("full") if isinstance(token_counts, Mapping) else None
    )
    if (
        not case_id
        or not episode_id
        or not isinstance(sequence_index, int)
        or not isinstance(turns, list)
        or not turns
        or not isinstance(input_tokens, int)
        or isinstance(input_tokens, bool)
        or input_tokens <= 0
    ):
        raise PacketError(f"{label} source case identity/history is invalid")
    messages: list[dict[str, str]] = []
    for turn in turns:
        if not isinstance(turn, Mapping):
            raise PacketError(f"{label} source turn is not an object")
        role = str(turn.get("role") or "")
        content = str(turn.get("text") or "")
        if role not in {"system", "user", "assistant", "tool"} or not content:
            raise PacketError(f"{label} source turn role/text is invalid")
        messages.append({"role": role, "content": content})
    return {
        "case_id": case_id,
        "episode_id": episode_id,
        "input_tokens": input_tokens,
        "messages": messages,
        "_sequence_index": sequence_index,
    }


def _validate_complete_episodes(cases: list[dict[str, Any]], label: str) -> None:
    if len(cases) % DECISIONS_PER_EPISODE:
        raise PacketError(f"{label} does not contain complete episodes")
    for start in range(0, len(cases), DECISIONS_PER_EPISODE):
        episode = cases[start : start + DECISIONS_PER_EPISODE]
        if len({case["episode_id"] for case in episode}) != 1 or [
            case["_sequence_index"] for case in episode
        ] != list(range(DECISIONS_PER_EPISODE)):
            raise PacketError(f"{label} episode order is not a complete 0..3 sequence")


def build_packet(
    *,
    source_corpus_path: Path,
    source_manifest_path: Path,
    runtime_dir: Path,
    output_dir: Path,
    placement_profile: str,
    gpu_class: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise PacketError("output directory already exists")
    if not placement_profile or not gpu_class:
        raise PacketError("placement profile and GPU class must be non-empty")

    source = _read_object(source_manifest_path, "source corpus manifest")
    if source.get("schema_version") != SOURCE_CORPUS_SCHEMA:
        raise PacketError("source corpus manifest schema is unsupported")
    if _sha256(source_corpus_path) != source.get("corpus_sha256"):
        raise PacketError("source corpus digest differs from its manifest")
    if source.get("contains_customer_data") is not False:
        raise PacketError("source corpus is not registered as customer-data-free")

    worker_ids, runtime = _runtime_identity(runtime_dir)
    if list(map(str, source.get("worker_order") or [])) != worker_ids:
        raise PacketError("runtime workers differ from the source corpus topology")
    identity_pairs = {
        "model": (source.get("model"), runtime["model"]),
        "model revision": (source.get("model_revision"), runtime["revision"]),
        "serializer": (source.get("serializer_version"), runtime["serialization"]),
    }
    for label, (corpus_value, runtime_value) in identity_pairs.items():
        if not corpus_value or corpus_value != runtime_value:
            raise PacketError(f"runtime {label} differs from the source corpus")

    rows = _read_rows(source_corpus_path)
    required = MEASURED_CASES + WARMUP_CASES
    if len(rows) < required:
        raise PacketError(f"source corpus must contain at least {required} cases")
    measured = [_case(row, "measured") for row in rows[:MEASURED_CASES]]
    warmup = [_case(row, "warmup") for row in rows[MEASURED_CASES:required]]
    _validate_complete_episodes(measured, "measured corpus")
    _validate_complete_episodes(warmup, "warmup corpus")
    if {case["episode_id"] for case in measured} & {
        case["episode_id"] for case in warmup
    }:
        raise PacketError("warmup and measured episodes must be disjoint")
    for case in [*measured, *warmup]:
        case.pop("_sequence_index")

    output_dir.mkdir(parents=True)
    corpus_path = output_dir / "corpus.json"
    workload_path = output_dir / "workload.json"
    topology_path = output_dir / "topology.json"
    identity_path = output_dir / "identity.json"
    _write_json(
        corpus_path,
        {
            "schema_version": CORPUS_SCHEMA,
            "warmup": warmup,
            "measured": measured,
        },
    )
    _write_json(
        workload_path,
        {
            "schema_version": WORKLOAD_SCHEMA,
            "concurrency": CONCURRENCY,
            "warmup_cases": WARMUP_CASES,
            "measured_cases": MEASURED_CASES,
            "seed": int(source["seed"]),
        },
    )
    identity_maps = {worker_id: worker_id for worker_id in worker_ids}
    _write_json(
        topology_path,
        {
            "schema_version": TOPOLOGY_SCHEMA,
            "canonical_workers": worker_ids,
            "arm_worker_maps": {arm: dict(identity_maps) for arm in ARMS},
        },
    )
    _write_json(
        identity_path,
        {
            "measurement_scope": "architecture_decision_boundary",
            "case_count": MEASURED_CASES,
            "corpus_sha256": _sha256(corpus_path),
            "workload_sha256": _sha256(workload_path),
            "encoder_model": runtime["model"],
            "encoder_revision": runtime["revision"],
            "tokenizer_sha256": str(source["tokenizer_sha256"]),
            "serializer_version": runtime["serialization"],
            "policy_artifact_revision": (f"sha256:{runtime['checkpoint_sha256']}"),
            "gpu_class": gpu_class,
            "worker_topology_sha256": _sha256(topology_path),
            "placement_profile": placement_profile,
            "warm_state": "warm",
            "seed": int(source["seed"]),
        },
    )
    return {
        "schema_version": PACKET_SCHEMA,
        "case_count": MEASURED_CASES,
        "warmup_case_count": WARMUP_CASES,
        "concurrency": CONCURRENCY,
        "corpus_sha256": _sha256(corpus_path),
        "workload_sha256": _sha256(workload_path),
        "worker_topology_sha256": _sha256(topology_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-corpus", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--placement-profile", default="modal-us-east-public-https")
    parser.add_argument("--gpu-class", default="NVIDIA H100 80GB")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        receipt = build_packet(
            source_corpus_path=args.source_corpus,
            source_manifest_path=args.source_manifest,
            runtime_dir=args.runtime_dir,
            output_dir=args.output_dir,
            placement_profile=args.placement_profile,
            gpu_class=args.gpu_class,
        )
    except (OSError, KeyError, TypeError, ValueError, PacketError) as error:
        raise SystemExit(f"cannot build parity packet: {error}") from error
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
