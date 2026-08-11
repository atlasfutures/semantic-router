#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the bounded, state-isolated PERF020 Remote-versus-ARC sweep."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rayline_concurrency_launcher import (
    EncoderOwnership,
    SweepContext,
    _cell_episode_ids,
    _cleanup_encoder,
    _cleanup_local,
    _local_cell,
    _prepare_cell,
    _reset_encoder_after_cell,
    _start_encoder,
    _start_local,
)
from rayline_concurrency_state import StateResetError, assert_encoder_empty
from rayline_open_loop_comparator import compare_open_loop
from rayline_open_loop_contract import (
    OPEN_LOOP_ARMS,
    PERF021_RUN_ID,
    OpenLoopCell,
    OpenLoopRunContract,
    resolve_launch_contract,
)
from rayline_open_loop_probe import load_open_loop_packet
from rayline_parity_http_probe import PROTOCOL_BY_ARM
from rayline_saturation_capacity_contract import (
    resolve_launch_contract as resolve_saturation_capacity_contract,
)
from rayline_saturation_knee_contract import (
    resolve_launch_contract as resolve_saturation_knee_contract,
)
from rayline_saturation_knee_v2_contract import (
    resolve_launch_contract as resolve_saturation_knee_v2_contract,
)
from rayline_saturation_ladder_contract import (
    resolve_launch_contract as resolve_saturation_ladder_contract,
)
from rayline_three_arm_budget import budget_receipt
from rayline_three_arm_contract import IDENTITY
from rayline_three_arm_launcher import (
    LaunchError,
    _assert_pushed,
    _free_port,
    _modal_containers,
    _run,
    _sha256,
)
from rayline_three_arm_telemetry import capture_arc_telemetry

DEPLOYMENT_EVIDENCE_SCHEMA = "rayline.vllm.open-loop-deployment-evidence.v1"


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=PERF021_RUN_ID)
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


def _validate_packet(contract: OpenLoopRunContract, packet_dir: Path) -> list[str]:
    if _sha256(packet_dir / "manifest.json") != contract.packet_manifest_sha256:
        raise LaunchError("open-loop packet manifest digest differs")
    if _sha256(packet_dir / "corpus.json") != contract.corpus_sha256:
        raise LaunchError("open-loop corpus digest differs")
    if _sha256(packet_dir / "topology.json") != contract.topology_sha256:
        raise LaunchError("open-loop topology digest differs")
    manifest = json.loads((packet_dir / "manifest.json").read_text())
    if (
        manifest.get("measured_cases") != contract.measured_cases
        or manifest.get("warmup_cases") != contract.warmup_cases
        or manifest.get("measured_episodes") != contract.measured_episodes
        or manifest.get("warmup_episodes") != contract.warmup_episodes
        or set(manifest.get("cells", {})) != {cell.label for cell in contract.cells}
    ):
        raise LaunchError("open-loop packet shape differs")
    for cell in contract.cells:
        cell_dir = packet_dir / "cells" / cell.label
        if (
            _sha256(cell_dir / "workload.json") != cell.workload_sha256
            or _sha256(cell_dir / "identity.json") != cell.identity_sha256
        ):
            raise LaunchError(f"open-loop {cell.label} digest differs")
        for arm in OPEN_LOOP_ARMS:
            _warmup, _measured, _identity, _worker_map, workload = (
                load_open_loop_packet(
                    arm=arm,
                    corpus_path=packet_dir / "corpus.json",
                    workload_path=cell_dir / "workload.json",
                    topology_path=packet_dir / "topology.json",
                    identity_path=cell_dir / "identity.json",
                )
            )
            if workload["offered_rate_rps"] != cell.offered_rate_rps:
                raise LaunchError(f"open-loop {cell.label} rate differs")
            # The comparator decides saturation against the contract's lane
            # count but never sees the packet, so the equality it depends on
            # is asserted here, where the packet is in hand and nothing has
            # been paid for yet.
            if (
                contract.saturation is not None
                and workload["max_episode_lanes"]
                != contract.saturation.episode_lanes
            ):
                raise LaunchError(f"open-loop {cell.label} episode lanes differ")
    topology = json.loads((packet_dir / "topology.json").read_text())
    return list(map(str, topology["canonical_workers"]))


def _resolve_contract(run_id: str) -> OpenLoopRunContract:
    """Resolve across every registry this launcher serves.

    Each registry independently refuses when nothing in it is launchable, so a
    launcher that serves several must try them all before failing closed. A
    new packet is not launchable until its registry is listed here, which is
    deliberate: adding a contract module must be an explicit act, not an
    implicit one. The failure is loud and costs nothing, because it lands in
    preflight before anything deploys.
    """

    for resolve in (
        resolve_launch_contract,
        resolve_saturation_ladder_contract,
        resolve_saturation_knee_contract,
        resolve_saturation_knee_v2_contract,
        resolve_saturation_capacity_contract,
    ):
        with contextlib.suppress(ValueError):
            return resolve(run_id)
    raise ValueError(
        "no Rayline open-loop sweep, saturation ladder arm, saturation knee "
        "or saturation capacity run is currently launchable"
    )


def _expected_arc_requests(contract: OpenLoopRunContract) -> int:
    """The ARC session actions one cell must record.

    Derived from the contract the packet was validated against in preflight,
    not from a module constant. The count is only ever read after a cell has
    already burned paid GPU time, so a number this launcher merely assumed
    would abort the run at the most expensive possible moment.
    """

    return contract.measured_cases + contract.warmup_cases


def _preflight(args: argparse.Namespace) -> SweepContext:
    contract = _resolve_contract(args.run_id)
    budget_receipt(contract.budget)
    semantic_root = Path(__file__).resolve().parents[3]
    pathfinder_root = args.pathfinder_root.resolve()
    packet_dir = args.packet_dir.resolve()
    runtime_dir = args.runtime_dir.resolve()
    pathfinder_python = pathfinder_root / ".venv/bin/python"
    if not pathfinder_python.is_file():
        raise LaunchError("Pathfinder .venv Python is required")
    semantic_head = _assert_pushed(
        semantic_root, IDENTITY.semantic_branch, "atlasfutures", contract.run_id
    )
    pathfinder_head = _assert_pushed(
        pathfinder_root, IDENTITY.pathfinder_branch, "origin", contract.run_id
    )
    if pathfinder_head != contract.pathfinder_authorization_commit:
        raise LaunchError("Pathfinder authorization head differs")
    worker_ids = _validate_packet(contract, packet_dir)
    _run(["docker", "image", "inspect", args.router_image], cwd=semantic_root)
    _run(
        ["go", "test", "./pkg/selection/raylinearc"],
        cwd=semantic_root / "src/semantic-router",
        environment={
            **os.environ,
            "RAYLINE_ARC_PRIVATE_ARTIFACT_DIR": str(runtime_dir),
        },
        timeout=120,
    )
    if not os.environ.get("HF_TOKEN"):
        raise LaunchError("HF_TOKEN is required for the private checkpoint")
    base_environment = {
        **os.environ,
        "MODAL_ENVIRONMENT": IDENTITY.modal_environment,
    }
    if _modal_containers(
        pathfinder_python,
        pathfinder_root,
        base_environment,
        contract.encoder_app_name,
    ):
        raise LaunchError("protected encoder already has a running container")
    for cell in contract.cells:
        compose_project = f"{contract.compose_project_prefix}-{cell.label}"
        existing = _run(
            ["docker", "ps", "-aq", "--filter", f"name={compose_project}"],
            cwd=semantic_root,
        ).stdout.strip()
        if existing:
            raise LaunchError(f"{compose_project} already exists")

    modal = importlib.import_module("modal")
    yaml = importlib.import_module("yaml")
    hf_hub_download = importlib.import_module("huggingface_hub").hf_hub_download
    if modal.__version__ != IDENTITY.required_modal_version:
        raise LaunchError(
            f"Modal SDK {IDENTITY.required_modal_version} is required; "
            f"found {modal.__version__}"
        )
    checkpoint = Path(
        hf_hub_download(
            IDENTITY.checkpoint_repo,
            IDENTITY.checkpoint_path,
            revision=IDENTITY.checkpoint_revision,
            token=os.environ["HF_TOKEN"],
        )
    )
    if _sha256(checkpoint) != IDENTITY.checkpoint_sha256:
        raise LaunchError("private checkpoint digest differs")
    output_dir = semantic_root / ".agent-harness/rayline-parity" / contract.run_id
    if output_dir.exists():
        raise LaunchError(f"{contract.run_id} output directory already exists")
    output_dir.mkdir(parents=True)
    return SweepContext(
        contract=contract,
        semantic_root=semantic_root,
        pathfinder_root=pathfinder_root,
        pathfinder_python=pathfinder_python,
        packet_dir=packet_dir,
        runtime_dir=runtime_dir,
        router_image=args.router_image,
        semantic_head=semantic_head,
        pathfinder_head=pathfinder_head,
        worker_ids=worker_ids,
        checkpoint=checkpoint,
        output_dir=output_dir,
        base_environment=base_environment,
        modal=modal,
        yaml=yaml,
    )


def _write_deployment_evidence(
    context: SweepContext, encoder: EncoderOwnership
) -> dict[str, Any]:
    """Record what was actually deployed, before any measured cell runs.

    The engine sizing comes from vLLM's own startup logging rather than from
    reading vLLM source. An encoder that captured nothing reports
    `startup_log_captured: false`; that is deliberately not an error, because
    the launcher must not turn a missing observation into a claimed one.
    """

    try:
        startup = encoder.client.request("GET", "/v1/rayline/arc/session/startup-log")
    except StateResetError as error:
        raise LaunchError("encoder startup-log evidence is unavailable") from error
    if startup.get("engine_build_id") != context.contract.encoder_build_id:
        raise LaunchError("deployed encoder build id differs from the contract")
    lines = startup.get("lines")
    if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
        raise LaunchError("encoder startup-log evidence is malformed")
    evidence = {
        "schema_version": DEPLOYMENT_EVIDENCE_SCHEMA,
        "run_id": context.contract.run_id,
        "encoder_app_name": encoder.app_name,
        "encoder_base_url": encoder.base_url,
        "encoder_gpu": context.contract.encoder_gpu,
        "engine_build_id": context.contract.encoder_build_id,
        "gdn_prefill_backend": context.contract.encoder_gdn_prefill_backend,
        "plugin_version": IDENTITY.plugin_version,
        "startup_log_captured": bool(startup.get("captured")) and bool(lines),
        "startup_log": list(lines),
    }
    (context.output_dir / "deployment-evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    return evidence


def _probe_cell(
    context: SweepContext,
    cell: OpenLoopCell,
    arm: str,
    base_url: str,
    output_dir: Path,
    timeout_seconds: float,
    logical_arm: str | None = None,
    session_namespace: str | None = None,
) -> dict[str, Any]:
    receipt_arm = logical_arm or arm
    output = output_dir / f"{receipt_arm}.json"
    cell_dir = context.packet_dir / "cells" / cell.label
    probe_namespace = session_namespace or receipt_arm
    probe_run_id = f"{context.contract.run_id}:{cell.label}:{probe_namespace}"
    _run(
        [
            sys.executable,
            str(
                context.semantic_root
                / "e2e/testing/rayline-arc/rayline_open_loop_probe.py"
            ),
            "--arm",
            arm,
            "--protocol",
            PROTOCOL_BY_ARM[arm],
            "--base-url",
            base_url,
            "--corpus",
            str(context.packet_dir / "corpus.json"),
            "--workload",
            str(cell_dir / "workload.json"),
            "--topology",
            str(context.packet_dir / "topology.json"),
            "--identity",
            str(cell_dir / "identity.json"),
            "--run-id",
            probe_run_id,
            "--output",
            str(output),
            "--timeout-seconds",
            "180",
        ],
        cwd=context.semantic_root,
        timeout=timeout_seconds,
    )
    return json.loads(output.read_text())


def _run_cell(
    context: SweepContext,
    encoder: EncoderOwnership,
    cell: OpenLoopCell,
    work: Path,
    paid_started: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    prepared = _prepare_cell(
        context,
        cell,
        work,
        encoder_base_url=encoder.base_url,
        encoder_build_id=context.contract.encoder_build_id,
    )
    ports = {
        name: _free_port() for name in ("pathfinder", "envoy", "router", "metrics")
    }
    local = _local_cell(context, encoder, cell, prepared, ports)
    local.compose_project = f"{context.contract.compose_project_prefix}-{cell.label}"
    cell_output = context.output_dir / cell.label
    cell_output.mkdir()
    measured_episodes, warmup_episodes = _cell_episode_ids(context)
    arc_run_id = f"{context.contract.run_id}:{cell.label}:rayline_arc"
    arc_started = False
    arc_completed = False
    receipts: dict[str, dict[str, Any]] = {}
    failure: BaseException | None = None
    state_receipt: dict[str, Any] | None = None
    try:
        router_url, arc_url = _start_local(context, prepared, local, ports)
        for arm, base_url in (
            ("rayline_remote", router_url),
            ("rayline_arc", arc_url),
        ):
            remaining = context.contract.budget.maximum_paid_wall_seconds - (
                time.perf_counter() - paid_started
            )
            if remaining <= 0:
                raise LaunchError(
                    f"{context.contract.run_id} paid wall-time ceiling reached"
                )
            if arm == "rayline_arc":
                assert_encoder_empty(encoder.client.request)
                arc_started = True
            receipts[arm] = _probe_cell(
                context, cell, arm, base_url, cell_output, remaining
            )
            if arm == "rayline_arc":
                arc_completed = (
                    receipts[arm]["results"]["completed"]
                    == context.contract.measured_cases
                    and receipts[arm]["results"]["failed"] == 0
                )
        telemetry = capture_arc_telemetry(
            f"http://127.0.0.1:{ports['metrics']}/metrics",
            cell_output / "rayline_arc_telemetry.json",
        )
        expected_arc_requests = _expected_arc_requests(context.contract)
        if sum(telemetry["session_actions"].values()) != expected_arc_requests:
            raise LaunchError("ARC telemetry count differs from open-loop packet")
    except BaseException as error:
        failure = error
    finally:
        _cleanup_local(context, local)
        try:
            state_receipt = _reset_encoder_after_cell(
                encoder=encoder,
                arc_run_id=arc_run_id,
                measured_episodes=measured_episodes,
                warmup_episodes=warmup_episodes,
                arc_started=arc_started,
                arc_completed=arc_completed,
            )
            (cell_output / "state-reset.json").write_text(
                json.dumps(state_receipt, indent=2, sort_keys=True) + "\n"
            )
        except BaseException as cleanup_error:
            if failure is None:
                raise
            raise LaunchError(
                "cell execution and state cleanup both failed"
            ) from cleanup_error
    if failure is not None:
        raise failure
    if not all(local.cleanup.values()) or state_receipt is None:
        raise LaunchError("cell local cleanup did not complete")
    return receipts, {"local": local.cleanup, "encoder_state": state_receipt}


def _write_manifest(
    context: SweepContext,
    comparison: Mapping[str, Any],
    cell_cleanup: Mapping[str, Any],
    encoder_cleanup: Mapping[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    manifest = {
        "schema_version": "rayline.vllm.open-loop-run.v1",
        "run_id": context.contract.run_id,
        "source": {
            "semantic_router_commit": context.semantic_head,
            "pathfinder_commit": context.pathfinder_head,
            "encoder_app_name": context.contract.encoder_app_name,
            "engine_build_id": context.contract.encoder_build_id,
            "gdn_prefill_backend": context.contract.encoder_gdn_prefill_backend,
            "plugin_version": IDENTITY.plugin_version,
            "packet_manifest_sha256": context.contract.packet_manifest_sha256,
            "router_image_id": _run(
                [
                    "docker",
                    "image",
                    "inspect",
                    "--format",
                    "{{.Id}}",
                    context.router_image,
                ],
                cwd=context.semantic_root,
            ).stdout.strip(),
        },
        "budget": budget_receipt(context.contract.budget, elapsed),
        "comparison_status": comparison["status"],
        "cell_cleanup": cell_cleanup,
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
    encoder = _start_encoder(context, app_name=context.contract.encoder_app_name)
    raw_cells: dict[str, dict[str, dict[str, Any]]] = {}
    cell_cleanup: dict[str, Any] = {}
    try:
        _write_deployment_evidence(context, encoder)
        with tempfile.TemporaryDirectory(
            prefix=context.contract.temporary_prefix
        ) as temp_name:
            temp_root = Path(temp_name)
            for cell in context.contract.cells:
                raw_cells[cell.label], cell_cleanup[cell.label] = _run_cell(
                    context,
                    encoder,
                    cell,
                    temp_root / cell.label,
                    paid_started,
                )
        comparison = compare_open_loop(
            raw_cells,
            tuple(cell.offered_rate_rps for cell in context.contract.cells),
            context.contract.measured_cases,
            context.contract.saturation,
        )
        (context.output_dir / "comparison.json").write_text(
            json.dumps(comparison, indent=2, sort_keys=True) + "\n"
        )
    finally:
        _cleanup_encoder(context, encoder)
    elapsed = time.perf_counter() - paid_started
    manifest = _write_manifest(
        context, comparison, cell_cleanup, encoder.cleanup, elapsed
    )
    print(
        json.dumps(
            {
                "run_id": context.contract.run_id,
                "output_dir": str(context.output_dir),
                "comparison_status": comparison["status"],
                "cell_cleanup": cell_cleanup,
                "encoder_cleanup": encoder.cleanup,
                "budget": manifest["budget"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
