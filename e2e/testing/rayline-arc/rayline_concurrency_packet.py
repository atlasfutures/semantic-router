#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Derive the frozen PERF017 concurrency sweep from the PERF015/016 packet."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rayline_parity_comparator import ARMS, INPUT_TOKEN_BUCKETS
from rayline_parity_http_probe import (
    CORPUS_SCHEMA,
    SWEEP_WORKLOADS,
    WORKLOAD_SCHEMA,
    _input_token_bucket,
    load_packet,
)

PACKET_SCHEMA = "rayline.vllm.concurrency-sweep-packet.v1"
MEASURED_CASES = 32
WARMUP_CASES = 4
DECISIONS_PER_EPISODE = 4


class SweepPacketError(ValueError):
    """The source packet cannot form the registered concurrency sweep."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SweepPacketError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise SweepPacketError(f"{label} must be a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _complete_episode_subset(cases: object, count: int, label: str) -> list[Any]:
    if not isinstance(cases, list) or len(cases) < count:
        raise SweepPacketError(f"source packet has too few {label} cases")
    selected = cases[:count]
    for start in range(0, count, DECISIONS_PER_EPISODE):
        episode = selected[start : start + DECISIONS_PER_EPISODE]
        if (
            len(episode) != DECISIONS_PER_EPISODE
            or len({str(case.get("episode_id") or "") for case in episode}) != 1
        ):
            raise SweepPacketError(f"{label} subset breaks episode boundaries")
    return selected


def build_sweep_packet(*, source_packet_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise SweepPacketError("output directory already exists")
    source_paths = {
        name: source_packet_dir / f"{name}.json"
        for name in ("corpus", "workload", "topology", "identity")
    }
    for arm in ARMS:
        load_packet(
            arm=arm,
            corpus_path=source_paths["corpus"],
            workload_path=source_paths["workload"],
            topology_path=source_paths["topology"],
            identity_path=source_paths["identity"],
        )

    source_corpus = _read_object(source_paths["corpus"], "source corpus")
    source_identity = _read_object(source_paths["identity"], "source identity")
    topology = _read_object(source_paths["topology"], "source topology")
    measured = _complete_episode_subset(
        source_corpus.get("measured"), MEASURED_CASES, "measured"
    )
    warmup = _complete_episode_subset(
        source_corpus.get("warmup"), WARMUP_CASES, "warmup"
    )
    measured_episodes = {str(case["episode_id"]) for case in measured}
    warmup_episodes = {str(case["episode_id"]) for case in warmup}
    if measured_episodes & warmup_episodes:
        raise SweepPacketError("warmup and measured episodes overlap")
    bucket_counts = Counter(
        _input_token_bucket(int(case["input_tokens"])) for case in measured
    )
    if set(bucket_counts) != set(INPUT_TOKEN_BUCKETS):
        raise SweepPacketError("measured subset does not cover every token bucket")

    corpus_path = output_dir / "corpus.json"
    topology_path = output_dir / "topology.json"
    _write_json(
        corpus_path,
        {
            "schema_version": CORPUS_SCHEMA,
            "warmup": warmup,
            "measured": measured,
        },
    )
    _write_json(topology_path, topology)

    cells: dict[str, Any] = {}
    for concurrency, contract in SWEEP_WORKLOADS.items():
        cell_dir = output_dir / "cells" / f"c{concurrency}"
        workload_path = cell_dir / "workload.json"
        identity_path = cell_dir / "identity.json"
        _write_json(
            workload_path,
            {
                "schema_version": WORKLOAD_SCHEMA,
                "concurrency": concurrency,
                "warmup_cases": WARMUP_CASES,
                "measured_cases": MEASURED_CASES,
                "seed": int(source_identity["seed"]),
            },
        )
        identity = dict(source_identity)
        identity.update(
            {
                "measurement_scope": "architecture_decision_concurrency_sweep",
                "case_count": MEASURED_CASES,
                "corpus_sha256": _sha256(corpus_path),
                "workload_sha256": _sha256(workload_path),
                "worker_topology_sha256": _sha256(topology_path),
            }
        )
        _write_json(identity_path, identity)
        cells[f"c{concurrency}"] = {
            "profile": contract.profile,
            "concurrency": concurrency,
            "workload_sha256": _sha256(workload_path),
            "identity_sha256": _sha256(identity_path),
        }

    receipt = {
        "schema_version": PACKET_SCHEMA,
        "source": {name: _sha256(path) for name, path in source_paths.items()},
        "measured_cases": MEASURED_CASES,
        "warmup_cases": WARMUP_CASES,
        "measured_episodes": len(measured_episodes),
        "warmup_episodes": len(warmup_episodes),
        "input_token_buckets": dict(sorted(bucket_counts.items())),
        "corpus_sha256": _sha256(corpus_path),
        "worker_topology_sha256": _sha256(topology_path),
        "cells": cells,
    }
    _write_json(output_dir / "manifest.json", receipt)
    return receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-packet-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        receipt = build_sweep_packet(
            source_packet_dir=args.source_packet_dir,
            output_dir=args.output_dir,
        )
    except (OSError, KeyError, TypeError, ValueError, SweepPacketError) as error:
        raise SystemExit(f"cannot build concurrency packet: {error}") from error
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
