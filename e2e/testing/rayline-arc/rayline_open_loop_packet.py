#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Derive the frozen PERF020 open-loop packet from the PERF017 sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rayline_parity_comparator import ARMS
from rayline_parity_http_probe import SWEEP_WORKLOADS, load_packet

PACKET_SCHEMA = "rayline.vllm.open-loop-packet.v1"
WORKLOAD_SCHEMA = "rayline.vllm.open-loop-workload.v1"
MEASUREMENT_SCOPE = "architecture_decision_open_loop_sweep"
MEASURED_CASES = 32
WARMUP_CASES = 4
MAX_EPISODE_LANES = 8
# The frozen PERF020/PERF021 ladder. It is the default so that re-running this
# script with no flags still reproduces that packet byte-for-byte; a successor
# packet passes its own rungs rather than editing this tuple.
OFFERED_RATES = (0.15, 0.30, 0.45)
ARRIVAL_PROCESS = "seeded_poisson"
COORDINATED_OMISSION_POLICY = "scheduled_arrival"
RATE_PRECISION_TOLERANCE = 1e-9


class OpenLoopPacketError(ValueError):
    """The source packet cannot form the registered open-loop sweep."""


def rate_label(rate: float) -> str:
    scaled = round(rate * 100)
    if scaled <= 0 or abs(rate - scaled / 100) > RATE_PRECISION_TOLERANCE:
        raise OpenLoopPacketError("offered rates require two-decimal precision")
    return f"r{scaled:03d}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise OpenLoopPacketError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise OpenLoopPacketError(f"{label} must be a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def resolve_offered_rates(rates: Sequence[float] | None) -> tuple[float, ...]:
    """Validate a rung set, defaulting to the frozen PERF020/PERF021 ladder."""

    resolved = tuple(OFFERED_RATES if rates is None else rates)
    if not resolved:
        raise OpenLoopPacketError("at least one offered rate is required")
    labels = [rate_label(rate) for rate in resolved]
    if len(set(labels)) != len(labels):
        raise OpenLoopPacketError("offered rates must be distinct")
    if list(resolved) != sorted(resolved):
        raise OpenLoopPacketError("offered rates must ascend")
    return resolved


def build_open_loop_packet(
    *,
    source_packet_dir: Path,
    output_dir: Path,
    offered_rates: Sequence[float] | None = None,
    episode_lanes: int = MAX_EPISODE_LANES,
) -> dict[str, Any]:
    rungs = resolve_offered_rates(offered_rates)
    # The lane count selects which registered sweep cell seeds the packet, and
    # the case counts come from that same contract, so a packet can never mix
    # one cell's lanes with another cell's corpus slice.
    contract = SWEEP_WORKLOADS.get(episode_lanes)
    if contract is None:
        raise OpenLoopPacketError(
            f"no registered sweep workload has {episode_lanes} lanes"
        )
    measured_cases = contract.measured_cases
    warmup_cases = contract.warmup_cases
    if output_dir.exists():
        raise OpenLoopPacketError("output directory already exists")
    source_cell = source_packet_dir / f"cells/c{episode_lanes}"
    source_paths = {
        "manifest": source_packet_dir / "manifest.json",
        "corpus": source_packet_dir / "corpus.json",
        "topology": source_packet_dir / "topology.json",
        "workload": source_cell / "workload.json",
        "identity": source_cell / "identity.json",
    }
    for arm in ARMS:
        load_packet(
            arm=arm,
            corpus_path=source_paths["corpus"],
            workload_path=source_paths["workload"],
            topology_path=source_paths["topology"],
            identity_path=source_paths["identity"],
            workload_contract=contract,
        )
    source_manifest = _read_object(source_paths["manifest"], "source manifest")
    source_identity = _read_object(source_paths["identity"], "source identity")
    corpus = _read_object(source_paths["corpus"], "source corpus")
    topology = _read_object(source_paths["topology"], "source topology")
    if (
        source_manifest.get("measured_cases") != measured_cases
        or source_manifest.get("warmup_cases") != warmup_cases
        or len(corpus.get("measured", [])) != measured_cases
        or len(corpus.get("warmup", [])) != warmup_cases
    ):
        raise OpenLoopPacketError("source packet counts differ")

    corpus_path = output_dir / "corpus.json"
    topology_path = output_dir / "topology.json"
    _write_json(corpus_path, corpus)
    _write_json(topology_path, topology)
    cells: dict[str, Any] = {}
    for offered_rate in rungs:
        label = rate_label(offered_rate)
        cell_dir = output_dir / "cells" / label
        workload_path = cell_dir / "workload.json"
        identity_path = cell_dir / "identity.json"
        _write_json(
            workload_path,
            {
                "schema_version": WORKLOAD_SCHEMA,
                "arrival_process": ARRIVAL_PROCESS,
                "coordinated_omission_policy": COORDINATED_OMISSION_POLICY,
                "offered_rate_rps": offered_rate,
                "max_episode_lanes": episode_lanes,
                "warmup_cases": warmup_cases,
                "measured_cases": measured_cases,
                "seed": int(source_identity["seed"]),
            },
        )
        identity = dict(source_identity)
        identity.update(
            {
                "measurement_scope": MEASUREMENT_SCOPE,
                "case_count": measured_cases,
                "corpus_sha256": _sha256(corpus_path),
                "workload_sha256": _sha256(workload_path),
                "worker_topology_sha256": _sha256(topology_path),
            }
        )
        _write_json(identity_path, identity)
        cells[label] = {
            "offered_rate_rps": offered_rate,
            "workload_sha256": _sha256(workload_path),
            "identity_sha256": _sha256(identity_path),
        }

    receipt = {
        "schema_version": PACKET_SCHEMA,
        "source": {name: _sha256(path) for name, path in source_paths.items()},
        "measured_cases": measured_cases,
        "warmup_cases": warmup_cases,
        "measured_episodes": source_manifest["measured_episodes"],
        "warmup_episodes": source_manifest["warmup_episodes"],
        "input_token_buckets": source_manifest["input_token_buckets"],
        "arrival_process": ARRIVAL_PROCESS,
        "coordinated_omission_policy": COORDINATED_OMISSION_POLICY,
        "corpus_sha256": _sha256(corpus_path),
        "worker_topology_sha256": _sha256(topology_path),
        "cells": cells,
    }
    _write_json(output_dir / "manifest.json", receipt)
    return receipt


def _parse_rate_list(raw: str) -> tuple[float, ...]:
    try:
        return tuple(float(part) for part in raw.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-packet-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--offered-rates",
        type=_parse_rate_list,
        default=None,
        help=(
            "Ascending comma-separated offered arrival rates in requests per "
            "second, two-decimal precision. Defaults to the frozen "
            f"{','.join(str(rate) for rate in OFFERED_RATES)} ladder."
        ),
    )
    parser.add_argument(
        "--episode-lanes",
        type=int,
        default=MAX_EPISODE_LANES,
        help=(
            "Lane count naming the registered sweep cell the packet derives "
            f"from. Defaults to the frozen PERF020 {MAX_EPISODE_LANES}."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        receipt = build_open_loop_packet(
            source_packet_dir=args.source_packet_dir,
            output_dir=args.output_dir,
            offered_rates=args.offered_rates,
            episode_lanes=args.episode_lanes,
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"cannot build open-loop packet: {error}") from error
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
