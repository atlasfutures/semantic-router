#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the bounded PERF025 sticky-versus-forced-failover ARC experiment."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import rayline_scaleout_launcher as scaleout
from rayline_failover_comparator import compare_failover
from rayline_failover_contract import (
    FAILOVER_AFTER_POOLING,
    FAILOVER_ARMS,
    PATHFINDER_AUTHORIZATION_COMMIT,
    PERF025_RUN_ID,
    resolve_launch_contract,
)
from rayline_open_loop_contract import MEASURED_CASES
from rayline_three_arm_budget import budget_receipt
from rayline_three_arm_launcher import LaunchError, _free_port, _stop_process
from rayline_three_arm_telemetry import capture_arc_telemetry

EXPECTED_ARC_REQUESTS = 36


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=PERF025_RUN_ID)
    parser.add_argument("--pathfinder-root", type=Path, required=True)
    parser.add_argument(
        "--packet-dir",
        type=Path,
        default=root / ".agent-harness/rayline-parity/packet-perf020",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=root / ".agent-harness/rayline-parity/staged-runtime",
    )
    parser.add_argument(
        "--router-image",
        default="ghcr.io/vllm-project/semantic-router/vllm-sr:latest",
    )
    return parser.parse_args()


def _preflight(args: argparse.Namespace) -> scaleout.SweepContext:
    contract = resolve_launch_contract(args.run_id)
    return scaleout._preflight_contract(
        args,
        contract,
        PATHFINDER_AUTHORIZATION_COMMIT,
        arms=FAILOVER_ARMS,
    )


def _prepare_arm(
    context: scaleout.SweepContext,
    pair: scaleout.EncoderPairOwnership,
    logical_arm: str,
    work: Path,
) -> scaleout.PreparedArm:
    cell = context.contract.cells[0]
    failover_after = (
        FAILOVER_AFTER_POOLING if logical_arm == "arc_dual_forced_failover" else None
    )
    affinity = scaleout._start_affinity(
        context,
        pair,
        pair.base_urls,
        failover_after_pooling=failover_after,
    )
    prepared = scaleout._prepare_cell(
        context,
        cell,
        work,
        encoder_base_url=affinity.local_base_url.replace(
            "127.0.0.1", "host.docker.internal"
        ),
    )
    ports = {
        name: _free_port() for name in ("pathfinder", "envoy", "router", "metrics")
    }
    local = scaleout._local_cell(context, pair, cell, prepared, ports)
    local.compose_project = (
        f"{context.contract.compose_project_prefix}-{cell.label}-{logical_arm}"
    )
    output = context.output_dir / cell.label
    output.mkdir(exist_ok=True)
    return scaleout.PreparedArm(affinity, prepared, ports, local, output)


def _run_arm(
    context: scaleout.SweepContext,
    pair: scaleout.EncoderPairOwnership,
    logical_arm: str,
    work: Path,
    paid_started: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    cell = context.contract.cells[0]
    arm = _prepare_arm(context, pair, logical_arm, work)
    measured_episodes, warmup_episodes = scaleout._cell_episode_ids(context)
    arc_run_id = f"{context.contract.run_id}:{cell.label}:{logical_arm}"
    arc_started = False
    arc_completed = False
    receipt: dict[str, Any] | None = None
    telemetry: dict[str, Any] | None = None
    state_receipt: dict[str, Any] | None = None
    affinity_receipt: dict[str, Any] | None = None
    failure: BaseException | None = None
    try:
        _router_url, arc_url = scaleout._start_local(
            context, arm.prepared, arm.local, arm.ports
        )
        scaleout.assert_encoder_empty(arm.affinity.client.request)
        scaleout._proxy_json(arm.affinity, "POST", scaleout.AFFINITY_RESET_PATH)
        arc_started = True
        remaining = context.contract.budget.maximum_paid_wall_seconds - (
            time.perf_counter() - paid_started
        )
        if remaining <= 0:
            raise LaunchError("PERF025 paid wall-time ceiling reached")
        receipt = scaleout._probe_cell(
            context,
            cell,
            "rayline_arc",
            arc_url,
            arm.output,
            remaining,
            logical_arm=logical_arm,
        )
        arc_completed = (
            receipt["results"]["completed"] == MEASURED_CASES
            and receipt["results"]["failed"] == 0
        )
        telemetry = capture_arc_telemetry(
            f"http://127.0.0.1:{arm.ports['metrics']}/metrics",
            arm.output / f"{logical_arm}_telemetry.json",
        )
        if sum(telemetry["session_actions"].values()) != EXPECTED_ARC_REQUESTS:
            raise LaunchError("PERF025 ARC telemetry count differs")
    except BaseException as error:
        failure = error
    finally:
        scaleout._cleanup_local(context, arm.local)
        try:
            state_receipt, affinity_receipt = scaleout._finalize_arm_state(
                affinity=arm.affinity,
                arc_run_id=arc_run_id,
                measured_episodes=measured_episodes,
                warmup_episodes=warmup_episodes,
                arc_started=arc_started,
                arc_completed=arc_completed,
                cell_output=arm.output,
                logical_arm=logical_arm,
            )
        except BaseException as cleanup_error:
            if failure is None:
                raise
            raise LaunchError(
                "failover arm execution and state cleanup both failed"
            ) from cleanup_error
        finally:
            _stop_process(arm.affinity.process)
    if failure is not None:
        raise failure
    if (
        receipt is None
        or telemetry is None
        or state_receipt is None
        or affinity_receipt is None
        or not all(arm.local.cleanup.values())
    ):
        raise LaunchError("failover arm cleanup did not complete")
    return (
        receipt,
        affinity_receipt,
        telemetry,
        {
            "local": arm.local.cleanup,
            "encoder_state": state_receipt,
            "affinity_proxy_stopped": arm.affinity.process.poll() is not None,
        },
    )


def _write_manifest(
    context: scaleout.SweepContext,
    comparison: dict[str, Any],
    arm_cleanup: dict[str, Any],
    encoder_cleanup: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    manifest = {
        "schema_version": "rayline.vllm.affinity-failover-run.v1",
        "run_id": context.contract.run_id,
        "source": {
            "semantic_router_commit": context.semantic_head,
            "pathfinder_commit": context.pathfinder_head,
            "engine_build_id": scaleout.IDENTITY.engine_build_id,
            "plugin_version": scaleout.IDENTITY.plugin_version,
            "packet_manifest_sha256": context.contract.packet_manifest_sha256,
            "encoder_app_names": list(scaleout.ENCODER_APP_NAMES),
        },
        "budget": budget_receipt(context.contract.budget, elapsed),
        "comparison_status": comparison["status"],
        "forced_failover_after_pooling": FAILOVER_AFTER_POOLING,
        "arm_cleanup": arm_cleanup,
        "encoder_cleanup": encoder_cleanup,
        "provider_calls": 0,
        "release_qualification_1000_executed": False,
    }
    (context.output_dir / "run-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    context = _preflight(_parse_args())
    paid_started = time.perf_counter()
    pair = scaleout._start_encoder_pair(context)
    receipts: dict[str, dict[str, Any]] = {}
    affinity: dict[str, dict[str, Any]] = {}
    telemetry: dict[str, dict[str, Any]] = {}
    arm_cleanup: dict[str, dict[str, Any]] = {}
    try:
        with tempfile.TemporaryDirectory(
            prefix=context.contract.temporary_prefix
        ) as temp_name:
            temp_root = Path(temp_name)
            for arm in FAILOVER_ARMS:
                (
                    receipts[arm],
                    affinity[arm],
                    telemetry[arm],
                    arm_cleanup[arm],
                ) = _run_arm(context, pair, arm, temp_root / arm, paid_started)
        comparison = compare_failover(receipts, affinity, telemetry)
        (context.output_dir / "comparison.json").write_text(
            json.dumps(comparison, indent=2, sort_keys=True) + "\n"
        )
    finally:
        scaleout._cleanup_encoder_pair(context, pair)
    elapsed = time.perf_counter() - paid_started
    manifest = _write_manifest(
        context,
        comparison,
        arm_cleanup,
        pair.cleanup,
        elapsed,
    )
    print(
        json.dumps(
            {
                "run_id": context.contract.run_id,
                "output_dir": str(context.output_dir),
                "comparison_status": comparison["status"],
                "arm_cleanup": arm_cleanup,
                "encoder_cleanup": pair.cleanup,
                "budget": manifest["budget"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
