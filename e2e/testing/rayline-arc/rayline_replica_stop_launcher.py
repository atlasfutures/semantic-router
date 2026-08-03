#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the source-closed PERF027 staged real-replica-stop experiment."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import rayline_scaleout_launcher as scaleout
from rayline_open_loop_probe import load_open_loop_packet
from rayline_parity_http_probe import JSONClient
from rayline_replica_stop_comparator import (
    BOUNDARY_SCHEMA,
    compare_replica_stop,
)
from rayline_replica_stop_contract import (
    PATHFINDER_AUTHORIZATION_COMMIT,
    PERF027_RUN_ID,
    SESSION_NAMESPACE,
    STOP_ARMS,
    UNAVAILABLE_APP_NAME,
    UNAVAILABLE_REPLICA,
    resolve_launch_contract,
)
from rayline_replica_stop_probe import (
    EXPECTED_POST_BOUNDARY_TURNS,
    EXPECTED_PRELOAD_TURNS,
    run_replica_stop_probe,
)
from rayline_three_arm_budget import budget_receipt
from rayline_three_arm_launcher import LaunchError, _free_port, _stop_process
from rayline_three_arm_telemetry import capture_arc_telemetry

EXPECTED_ARC_REQUESTS = 36


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=PERF027_RUN_ID)
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
        arms=STOP_ARMS,
    )


def _prepare_arm(
    context: scaleout.SweepContext,
    pair: scaleout.EncoderPairOwnership,
    logical_arm: str,
    work: Path,
) -> scaleout.PreparedArm:
    cell = context.contract.cells[0]
    affinity = scaleout._start_affinity(
        context,
        pair,
        pair.base_urls,
        unavailable_replica=(
            UNAVAILABLE_REPLICA if logical_arm == "arc_dual_replica_stop" else None
        ),
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


def _control_boundary() -> dict[str, Any]:
    return {
        "schema_version": BOUNDARY_SCHEMA,
        "action": "control_no_stop",
        "elapsed_seconds": 0.0,
    }


def _stop_boundary(
    context: scaleout.SweepContext,
    pair: scaleout.EncoderPairOwnership,
) -> dict[str, Any]:
    started = time.perf_counter()
    command_started = time.perf_counter()
    command_succeeded = scaleout._stop_app(
        context,
        pair.service_environment,
        UNAVAILABLE_APP_NAME,
    )
    command_seconds = time.perf_counter() - command_started
    if not command_succeeded:
        raise LaunchError("PERF027 exact replica-stop command failed")
    deadline = time.monotonic() + scaleout.MAXIMUM_SCALEDOWN_SECONDS
    unavailable_stopped = False
    unavailable_containers = -1
    survivor_deployed = False
    survivor_containers = -1
    while time.monotonic() < deadline:
        apps = scaleout._named_encoder_apps(context)
        containers = scaleout._named_encoder_containers(context)
        unavailable_rows = [
            row for row in apps if row.get("description") == UNAVAILABLE_APP_NAME
        ]
        survivor_name = scaleout.ENCODER_APP_NAMES[1]
        survivor_rows = [row for row in apps if row.get("description") == survivor_name]
        unavailable_stopped = bool(unavailable_rows) and all(
            row.get("state") == "stopped" and str(row.get("tasks")) == "0"
            for row in unavailable_rows
        )
        unavailable_containers = sum(
            row.get("app_name") == UNAVAILABLE_APP_NAME for row in containers
        )
        survivor_deployed = any(
            row.get("state") == "deployed" and str(row.get("tasks")) == "1"
            for row in survivor_rows
        )
        survivor_containers = sum(
            row.get("app_name") == survivor_name for row in containers
        )
        if (
            unavailable_stopped
            and unavailable_containers == 0
            and survivor_deployed
            and survivor_containers == 1
        ):
            break
        time.sleep(1)
    else:
        raise LaunchError("PERF027 exact replica stop did not converge")
    return {
        "schema_version": BOUNDARY_SCHEMA,
        "action": "stop_exact_app",
        "unavailable_replica": UNAVAILABLE_REPLICA,
        "unavailable_app_name": UNAVAILABLE_APP_NAME,
        "stop_command_succeeded": command_succeeded,
        "stop_command_seconds": command_seconds,
        "convergence_seconds": time.perf_counter() - started - command_seconds,
        "unavailable_app_stopped": unavailable_stopped,
        "unavailable_containers_remaining": unavailable_containers,
        "survivor_app_deployed": survivor_deployed,
        "survivor_containers_running": survivor_containers,
    }


def _execute_staged_arm(
    context: scaleout.SweepContext,
    pair: scaleout.EncoderPairOwnership,
    arm: scaleout.PreparedArm,
    logical_arm: str,
    arc_url: str,
    arc_run_id: str,
    paid_started: float,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    cell = context.contract.cells[0]
    warmup, measured, identity, worker_map, workload = load_open_loop_packet(
        arm="rayline_arc",
        corpus_path=context.packet_dir / "corpus.json",
        workload_path=context.packet_dir / "cells" / cell.label / "workload.json",
        topology_path=context.packet_dir / "topology.json",
        identity_path=context.packet_dir / "cells" / cell.label / "identity.json",
    )
    remaining = context.contract.budget.maximum_paid_wall_seconds - (
        time.perf_counter() - paid_started
    )
    if remaining <= 0:
        raise LaunchError("PERF027 paid wall-time ceiling reached")
    boundary = (
        (lambda: _stop_boundary(context, pair))
        if logical_arm == "arc_dual_replica_stop"
        else _control_boundary
    )
    receipt = run_replica_stop_probe(
        client=JSONClient(arc_url, timeout_seconds=min(180.0, remaining)),
        warmup=warmup,
        measured=measured,
        identity=identity,
        worker_map=worker_map,
        run_id=arc_run_id,
        offered_rate_rps=float(workload["offered_rate_rps"]),
        boundary_callback=boundary,
    )
    (arm.output / f"{logical_arm}.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    completed = (
        receipt["preload"]["completed"] == EXPECTED_PRELOAD_TURNS
        and receipt["preload"]["failed"] == 0
        and receipt["results"]["completed"] == EXPECTED_POST_BOUNDARY_TURNS
        and receipt["results"]["failed"] == 0
    )
    telemetry = capture_arc_telemetry(
        f"http://127.0.0.1:{arm.ports['metrics']}/metrics",
        arm.output / f"{logical_arm}_telemetry.json",
    )
    if sum(telemetry["session_actions"].values()) != EXPECTED_ARC_REQUESTS:
        raise LaunchError("PERF027 ARC telemetry count differs")
    return receipt, telemetry, completed


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
    arc_run_id = f"{context.contract.run_id}:{cell.label}:{SESSION_NAMESPACE}"
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
        receipt, telemetry, arc_completed = _execute_staged_arm(
            context,
            pair,
            arm,
            logical_arm,
            arc_url,
            arc_run_id,
            paid_started,
        )
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
                "replica-stop arm execution and state cleanup both failed"
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
        raise LaunchError("replica-stop arm cleanup did not complete")
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
        "schema_version": "rayline.vllm.replica-stop-run.v1",
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
        "session_namespace": SESSION_NAMESPACE,
        "unavailable_replica": UNAVAILABLE_REPLICA,
        "unavailable_app_name": UNAVAILABLE_APP_NAME,
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
            for arm in STOP_ARMS:
                (
                    receipts[arm],
                    affinity[arm],
                    telemetry[arm],
                    arm_cleanup[arm],
                ) = _run_arm(context, pair, arm, temp_root / arm, paid_started)
        comparison = compare_replica_stop(receipts, affinity, telemetry)
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
