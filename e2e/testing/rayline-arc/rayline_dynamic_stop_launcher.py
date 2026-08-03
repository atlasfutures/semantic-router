#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the source-closed DYN006 three-encoder dynamic drain/stop cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rayline_scaleout_launcher as scaleout
from rayline_concurrency_launcher import LocalCell, PreparedCell
from rayline_concurrency_state import ProtectedEncoderClient, assert_encoder_empty
from rayline_dynamic_arc_config import build_dynamic_config
from rayline_dynamic_capacity_runtime import (
    EncoderFleetOwnership,
    cleanup_encoder_fleet,
    controller_json,
    dynamic_compose,
    named_encoder_containers,
    redis_episode_state,
    register_third_replica,
    start_encoder_fleet,
    stop_draining_replica,
    wait_for_drained_removal,
)
from rayline_dynamic_stop_comparator import (
    CONTROL_BOUNDARY_SCHEMA,
    compare_dynamic_stop,
)
from rayline_dynamic_stop_contract import (
    DYN006_RUN_ID,
    DYNAMIC_STOP_ARMS,
    ENCODER_APP_NAMES,
    ENCODER_REPLICA_IDS,
    EXPECTED_POST_STOP_OWNERS,
    EXPECTED_PRE_BOUNDARY_OWNERS,
    INITIAL_MEMBERSHIP_REVISION,
    MEASURED_EPISODE_IDS,
    MEMBERSHIP_ADOPTION_SECONDS,
    PATHFINDER_AUTHORIZATION_COMMIT,
    SESSION_NAMESPACE,
    WARMUP_EPISODE_ID,
    resolve_launch_contract,
)
from rayline_dynamic_telemetry import capture_dynamic_arc_telemetry
from rayline_open_loop_probe import load_open_loop_packet
from rayline_parity_arc_config import _runtime_manifest
from rayline_parity_http_probe import Case, JSONClient
from rayline_replica_stop_probe import (
    EXPECTED_POST_BOUNDARY_TURNS,
    EXPECTED_PRELOAD_TURNS,
    run_replica_stop_probe,
)
from rayline_three_arm_budget import budget_receipt
from rayline_three_arm_contract import IDENTITY
from rayline_three_arm_launcher import (
    LaunchError,
    _free_port,
    _stop_process,
    _wait_arc_ready,
    _wait_http,
)

EXPECTED_SELECTIONS = 47
HTTP_OK = 200


@dataclass(frozen=True)
class PreparedDynamicArm:
    prepared: PreparedCell
    membership_document: Path
    ports: dict[str, int]
    local: LocalCell
    output: Path


@dataclass(frozen=True)
class DynamicPacket:
    warmup: list[Case]
    measured: list[Case]
    identity: dict[str, Any]
    worker_map: dict[str, str]
    offered_rate_rps: float
    measured_episode_ids: tuple[str, ...]


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DYN006_RUN_ID)
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
        required=True,
    )
    return parser.parse_args()


def _preflight(args: argparse.Namespace) -> scaleout.SweepContext:
    contract = resolve_launch_contract(args.run_id)
    context = scaleout._preflight_contract(
        args,
        contract,
        PATHFINDER_AUTHORIZATION_COMMIT,
        arms=DYNAMIC_STOP_ARMS,
    )
    expected_image = f"rayline-dyn006-router:{context.semantic_head}"
    if context.router_image != expected_image:
        raise LaunchError(f"DYN006 router image must be {expected_image}")
    if named_encoder_containers(context):
        raise LaunchError("a DYN006 encoder replica already has a container")
    return context


def _initial_membership(fleet: EncoderFleetOwnership) -> dict[str, Any]:
    return {
        "schema_version": "rayline.arc.encoder-membership.v1",
        "revision": INITIAL_MEMBERSHIP_REVISION,
        "replicas": [
            {
                "id": ENCODER_REPLICA_IDS[index],
                "base_url": fleet.base_urls[index],
                "state": "active",
            }
            for index in range(2)
        ],
    }


def _prepare_arm(
    context: scaleout.SweepContext,
    fleet: EncoderFleetOwnership,
    logical_arm: str,
    work: Path,
) -> PreparedDynamicArm:
    cell = context.contract.cells[0]
    prepared = scaleout._prepare_cell(
        context,
        cell,
        work,
        encoder_base_url=fleet.base_urls[0],
    )
    template = context.yaml.safe_load(
        (context.semantic_root / "deploy/compose/rayline-arc/config.yaml").read_text()
    )
    dynamic_config = build_dynamic_config(
        template,
        _runtime_manifest(context.runtime_dir),
        artifact_mount_path="/var/lib/vllm-sr/rayline-arc",
        encoder_build_id=IDENTITY.engine_build_id,
        encoder_plugin_version=IDENTITY.plugin_version,
        worker_endpoint="worker-double:8081",
    )
    prepared.arc_config.write_text(
        context.yaml.safe_dump(dynamic_config, sort_keys=False)
    )
    membership_document = work / "membership.json"
    membership_document.write_text(
        json.dumps(_initial_membership(fleet), separators=(",", ":")) + "\n"
    )
    ports = {
        name: _free_port() for name in ("pathfinder", "envoy", "router", "metrics")
    }
    local = scaleout._local_cell(context, fleet, cell, prepared, ports)
    local.compose_project = (
        f"{context.contract.compose_project_prefix}-{cell.label}-{logical_arm}"
    )
    local.compose_environment["RAYLINE_DYNAMIC_MEMBERSHIP_DOCUMENT"] = str(
        membership_document
    )
    output = context.output_dir / cell.label
    output.mkdir(exist_ok=True)
    return PreparedDynamicArm(prepared, membership_document, ports, local, output)


def _start_local(
    context: scaleout.SweepContext,
    arm: PreparedDynamicArm,
) -> tuple[str, str]:
    arm.local.router_log = (arm.prepared.work / "pathfinder.log").open(
        "w", encoding="utf-8"
    )
    arm.local.router_process = subprocess.Popen(
        [
            str(context.pathfinder_python),
            "-m",
            "uvicorn",
            "rayline_router.serving.app:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(arm.ports["pathfinder"]),
            "--log-level",
            "warning",
        ],
        cwd=context.pathfinder_root,
        env=arm.local.router_environment,
        text=True,
        stdout=arm.local.router_log,
        stderr=subprocess.STDOUT,
    )
    router_url = f"http://127.0.0.1:{arm.ports['pathfinder']}"
    _wait_http(f"{router_url}/healthz", 180)
    dynamic_compose(context, arm.local, "up", "--build", "--detach")
    _wait_http(f"http://127.0.0.1:{arm.ports['router']}/health", 60)
    _wait_arc_ready(f"http://127.0.0.1:{arm.ports['metrics']}/metrics", 180)
    return router_url, f"http://127.0.0.1:{arm.ports['envoy']}"


def _cleanup_local(context: scaleout.SweepContext, arm: PreparedDynamicArm) -> None:
    _stop_process(arm.local.router_process)
    arm.local.cleanup["pathfinder_stopped"] = (
        arm.local.router_process is None or arm.local.router_process.poll() is not None
    )
    if arm.local.router_log is not None:
        arm.local.router_log.close()
    result = dynamic_compose(
        context,
        arm.local,
        "down",
        "--volumes",
        "--remove-orphans",
        check=False,
    )
    arm.local.cleanup["compose_removed"] = result.returncode == 0


def _rendezvous_owner(raw_episode_id: str) -> str:
    episode_hash = hashlib.sha256(raw_episode_id.encode()).hexdigest()
    return max(
        ENCODER_REPLICA_IDS,
        key=lambda replica_id: hashlib.sha256(
            episode_hash.encode() + b"\x00" + replica_id.encode()
        ).digest(),
    )


def _capacity_canary_episode(arc_run_id: str) -> str:
    for index in range(10_000):
        raw = f"{arc_run_id}:capacity-canary-{index}"
        if _rendezvous_owner(raw) == ENCODER_REPLICA_IDS[2]:
            return raw
    raise LaunchError("could not construct an encoder-c capacity canary")


def _expect_chat(
    client: JSONClient,
    *,
    episode: str,
    route_id: str,
    messages: list[dict[str, str]],
    close: bool = False,
) -> None:
    headers = {
        "x-rayline-episode-id": episode,
        "x-rayline-route-id": route_id,
    }
    if close:
        headers["x-rayline-episode-close"] = "true"
    status, _body, response_headers, _elapsed = client.request(
        "POST",
        "/v1/chat/completions",
        body={"model": "auto", "messages": messages, "max_tokens": 1},
        headers=headers,
    )
    if status != HTTP_OK or not response_headers.get("x-vsr-selected-model"):
        raise LaunchError("dynamic ARC canary or close request failed")


def _prove_capacity_adoption(
    context: scaleout.SweepContext,
    fleet: EncoderFleetOwnership,
    arm: PreparedDynamicArm,
    client: JSONClient,
    arc_run_id: str,
) -> tuple[dict[str, Any], str]:
    registration = register_third_replica(context, arm.local, fleet)
    # The router refreshes once per second. Give it two full refresh windows,
    # then make exactly one measured adoption proof so telemetry remains frozen.
    time.sleep(MEMBERSHIP_ADOPTION_SECONDS)
    episode = _capacity_canary_episode(arc_run_id)
    messages = [{"role": "user", "content": "DYN006 capacity adoption canary"}]
    _expect_chat(
        client,
        episode=episode,
        route_id=f"{arc_run_id}:capacity-open",
        messages=messages,
    )
    state = redis_episode_state(context, arm.local, episode)
    owner = str((state or {}).get("encoder_owner") or "")
    if owner != ENCODER_REPLICA_IDS[2]:
        raise LaunchError(f"router did not adopt registered capacity: {owner}")
    _expect_chat(
        client,
        episode=episode,
        route_id=f"{arc_run_id}:capacity-close",
        messages=[*messages, {"role": "user", "content": "close capacity canary"}],
        close=True,
    )
    state = redis_episode_state(context, arm.local, episode)
    if (
        state is None
        or state.get("encoder_owner") != ""
        or state.get("encoder_visited_owners") != []
    ):
        raise LaunchError("capacity canary did not close cleanly")
    return registration, episode


def _unique_cases(cases: Iterable[Case]) -> dict[str, Case]:
    result: dict[str, Case] = {}
    for case in cases:
        result[case.episode_id] = case
    return result


def _owner_vector(
    context: scaleout.SweepContext,
    arm: PreparedDynamicArm,
    arc_run_id: str,
    episode_ids: Iterable[str],
) -> list[int]:
    counts = dict.fromkeys(ENCODER_REPLICA_IDS, 0)
    for episode_id in dict.fromkeys(episode_ids):
        state = redis_episode_state(
            context,
            arm.local,
            f"{arc_run_id}:{episode_id}",
        )
        owner = str((state or {}).get("encoder_owner") or "")
        if owner not in counts:
            raise LaunchError("dynamic episode owner is missing or unknown")
        counts[owner] += 1
    return [counts[replica_id] for replica_id in ENCODER_REPLICA_IDS]


def _close_packet_sessions(
    context: scaleout.SweepContext,
    arm: PreparedDynamicArm,
    client: JSONClient,
    *,
    arc_run_id: str,
    warmup: list[Case],
    measured: list[Case],
    capacity_episode: str,
) -> int:
    cases = {**_unique_cases(warmup), **_unique_cases(measured)}
    for index, (episode_id, case) in enumerate(sorted(cases.items())):
        _expect_chat(
            client,
            episode=f"{arc_run_id}:{episode_id}",
            route_id=f"{arc_run_id}:cleanup-{index}",
            messages=[
                *(dict(message) for message in case.messages),
                {"role": "user", "content": "close DYN006 measured session"},
            ],
            close=True,
        )
    cleared = 1
    for episode_id in cases:
        state = redis_episode_state(
            context,
            arm.local,
            f"{arc_run_id}:{episode_id}",
        )
        if (
            state is None
            or state.get("encoder_owner") != ""
            or state.get("encoder_visited_owners") != []
        ):
            raise LaunchError("dynamic packet session did not close cleanly")
        cleared += 1
    capacity_state = redis_episode_state(context, arm.local, capacity_episode)
    if capacity_state is None or capacity_state.get("encoder_owner") != "":
        raise LaunchError("dynamic capacity session cleanup was lost")
    return cleared


def _control_boundary() -> dict[str, Any]:
    return {
        "schema_version": CONTROL_BOUNDARY_SCHEMA,
        "action": "control_no_mutation",
    }


def _load_dynamic_packet(context: scaleout.SweepContext) -> DynamicPacket:
    cell = context.contract.cells[0]
    warmup, measured, identity, worker_map, workload = load_open_loop_packet(
        arm="rayline_arc",
        corpus_path=context.packet_dir / "corpus.json",
        workload_path=context.packet_dir / "cells" / cell.label / "workload.json",
        topology_path=context.packet_dir / "topology.json",
        identity_path=context.packet_dir / "cells" / cell.label / "identity.json",
    )
    measured_episode_ids = tuple(case.episode_id for case in measured)
    if tuple(sorted(set(measured_episode_ids))) != MEASURED_EPISODE_IDS or {
        case.episode_id for case in warmup
    } != {WARMUP_EPISODE_ID}:
        raise LaunchError("DYN006 packet episode identities differ")
    return DynamicPacket(
        warmup=warmup,
        measured=measured,
        identity=identity,
        worker_map=worker_map,
        offered_rate_rps=float(workload["offered_rate_rps"]),
        measured_episode_ids=measured_episode_ids,
    )


def _remaining_paid_seconds(
    context: scaleout.SweepContext,
    paid_started: float,
    label: str,
) -> float:
    remaining = context.contract.budget.maximum_paid_wall_seconds - (
        time.perf_counter() - paid_started
    )
    if remaining <= 0:
        raise LaunchError(f"DYN006 paid wall-time expired {label}")
    return remaining


def _finish_dynamic_membership(
    context: scaleout.SweepContext,
    fleet: EncoderFleetOwnership,
    arm: PreparedDynamicArm,
    logical_arm: str,
    paid_started: float,
) -> dict[str, Any]:
    treatment = logical_arm == DYNAMIC_STOP_ARMS[1]
    if treatment:
        remaining = _remaining_paid_seconds(context, paid_started, "before removal")
        final_membership = wait_for_drained_removal(
            context,
            arm.local,
            timeout_seconds=min(360.0, remaining),
        )
    else:
        final_membership = controller_json(context, arm.local, "status")
    live_urls = fleet.base_urls[1:] if treatment else fleet.base_urls
    for base_url in live_urls:
        assert_encoder_empty(
            ProtectedEncoderClient(
                base_url,
                fleet.proxy.token_id,
                fleet.proxy.token_secret,
            ).request
        )
    return final_membership


def _execute_arm(
    context: scaleout.SweepContext,
    fleet: EncoderFleetOwnership,
    arm: PreparedDynamicArm,
    *,
    logical_arm: str,
    arc_url: str,
    arc_run_id: str,
    paid_started: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    packet = _load_dynamic_packet(context)
    remaining = _remaining_paid_seconds(context, paid_started, "before measurement")
    client = JSONClient(arc_url, timeout_seconds=min(180.0, remaining))
    registration, capacity_episode = _prove_capacity_adoption(
        context, fleet, arm, client, arc_run_id
    )
    pre_boundary_owners: list[int] = []

    def boundary() -> Mapping[str, Any]:
        nonlocal pre_boundary_owners
        pre_boundary_owners = _owner_vector(
            context,
            arm,
            arc_run_id,
            packet.measured_episode_ids,
        )
        if pre_boundary_owners != list(EXPECTED_PRE_BOUNDARY_OWNERS):
            raise LaunchError("DYN006 pre-boundary owner placement differs")
        if logical_arm == DYNAMIC_STOP_ARMS[1]:
            return stop_draining_replica(context, arm.local, fleet)
        return _control_boundary()

    receipt = run_replica_stop_probe(
        client=client,
        warmup=packet.warmup,
        measured=packet.measured,
        identity=packet.identity,
        worker_map=packet.worker_map,
        run_id=arc_run_id,
        offered_rate_rps=packet.offered_rate_rps,
        boundary_callback=boundary,
    )
    if (
        receipt["preload"]["completed"] != EXPECTED_PRELOAD_TURNS
        or receipt["results"]["completed"] != EXPECTED_POST_BOUNDARY_TURNS
    ):
        raise LaunchError("DYN006 staged decisions did not complete")
    post_boundary_owners = _owner_vector(
        context,
        arm,
        arc_run_id,
        packet.measured_episode_ids,
    )
    expected_post = (
        list(EXPECTED_POST_STOP_OWNERS)
        if logical_arm == DYNAMIC_STOP_ARMS[1]
        else list(EXPECTED_PRE_BOUNDARY_OWNERS)
    )
    if post_boundary_owners != expected_post:
        raise LaunchError("DYN006 post-boundary owner placement differs")
    states_cleared = _close_packet_sessions(
        context,
        arm,
        client,
        arc_run_id=arc_run_id,
        warmup=packet.warmup,
        measured=packet.measured,
        capacity_episode=capacity_episode,
    )
    final_membership = _finish_dynamic_membership(
        context,
        fleet,
        arm,
        logical_arm,
        paid_started,
    )
    telemetry = capture_dynamic_arc_telemetry(
        f"http://127.0.0.1:{arm.ports['metrics']}/metrics",
        arm.output / f"{logical_arm}_telemetry.json",
    )
    if sum(telemetry["session_actions"].values()) != EXPECTED_SELECTIONS:
        raise LaunchError("DYN006 ARC telemetry count differs")
    lifecycle = {
        "pre_boundary_owners": pre_boundary_owners,
        "post_boundary_owners": post_boundary_owners,
        "capacity_registration": registration,
        "final_membership": final_membership,
        "episode_states_cleared": states_cleared,
    }
    (arm.output / f"{logical_arm}.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    (arm.output / f"{logical_arm}_lifecycle.json").write_text(
        json.dumps(lifecycle, indent=2, sort_keys=True) + "\n"
    )
    return receipt, telemetry, lifecycle


def _run_arm(
    context: scaleout.SweepContext,
    fleet: EncoderFleetOwnership,
    logical_arm: str,
    work: Path,
    paid_started: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    arm = _prepare_arm(context, fleet, logical_arm, work)
    cell = context.contract.cells[0]
    arc_run_id = f"{context.contract.run_id}:{cell.label}:{SESSION_NAMESPACE}"
    failure: BaseException | None = None
    receipt: dict[str, Any] | None = None
    telemetry: dict[str, Any] | None = None
    lifecycle: dict[str, Any] | None = None
    try:
        _router_url, arc_url = _start_local(context, arm)
        receipt, telemetry, lifecycle = _execute_arm(
            context,
            fleet,
            arm,
            logical_arm=logical_arm,
            arc_url=arc_url,
            arc_run_id=arc_run_id,
            paid_started=paid_started,
        )
    except BaseException as error:
        failure = error
    finally:
        _cleanup_local(context, arm)
    if failure is not None:
        raise failure
    if (
        receipt is None
        or telemetry is None
        or lifecycle is None
        or not all(arm.local.cleanup.values())
    ):
        raise LaunchError("DYN006 arm cleanup did not complete")
    return receipt, telemetry, lifecycle, dict(arm.local.cleanup)


def _write_manifest(
    context: scaleout.SweepContext,
    comparison: Mapping[str, Any],
    arm_cleanup: Mapping[str, Any],
    encoder_cleanup: Mapping[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    manifest = {
        "schema_version": "rayline.vllm.dynamic-capacity-stop-run.v1",
        "run_id": context.contract.run_id,
        "source": {
            "semantic_router_commit": context.semantic_head,
            "pathfinder_commit": context.pathfinder_head,
            "engine_build_id": IDENTITY.engine_build_id,
            "plugin_version": IDENTITY.plugin_version,
            "packet_manifest_sha256": context.contract.packet_manifest_sha256,
            "encoder_app_names": list(ENCODER_APP_NAMES),
            "router_image": context.router_image,
            "router_image_id": scaleout._run(
                [
                    "docker",
                    "image",
                    "inspect",
                    "--format={{.Id}}",
                    context.router_image,
                ],
                cwd=context.semantic_root,
            ).stdout.strip(),
        },
        "budget": budget_receipt(context.contract.budget, elapsed),
        "comparison_status": comparison["status"],
        "session_namespace": SESSION_NAMESPACE,
        "arm_cleanup": dict(arm_cleanup),
        "encoder_cleanup": dict(encoder_cleanup),
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
    fleet = start_encoder_fleet(context)
    receipts: dict[str, dict[str, Any]] = {}
    telemetry: dict[str, dict[str, Any]] = {}
    lifecycle: dict[str, dict[str, Any]] = {}
    arm_cleanup: dict[str, dict[str, Any]] = {}
    comparison: dict[str, Any]
    try:
        with tempfile.TemporaryDirectory(
            prefix=context.contract.temporary_prefix
        ) as temp_name:
            temp_root = Path(temp_name)
            for logical_arm in DYNAMIC_STOP_ARMS:
                (
                    receipts[logical_arm],
                    telemetry[logical_arm],
                    lifecycle[logical_arm],
                    arm_cleanup[logical_arm],
                ) = _run_arm(
                    context,
                    fleet,
                    logical_arm,
                    temp_root / logical_arm,
                    paid_started,
                )
        comparison = compare_dynamic_stop(receipts, telemetry, lifecycle)
        (context.output_dir / "comparison.json").write_text(
            json.dumps(comparison, indent=2, sort_keys=True) + "\n"
        )
    finally:
        cleanup_encoder_fleet(context, fleet)
    elapsed = time.perf_counter() - paid_started
    manifest = _write_manifest(
        context,
        comparison,
        arm_cleanup,
        fleet.cleanup,
        elapsed,
    )
    print(
        json.dumps(
            {
                "run_id": context.contract.run_id,
                "output_dir": str(context.output_dir),
                "comparison_status": comparison["status"],
                "arm_cleanup": arm_cleanup,
                "encoder_cleanup": fleet.cleanup,
                "budget": manifest["budget"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
