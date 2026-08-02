#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run a bounded, state-isolated Remote-versus-ARC concurrency sweep.

The protected encoder stays warm across concurrency cells, while each cell gets
a fresh Pathfinder process, ARC Compose/Redis stack, run-ID namespace, and an
explicit retained-session cleanup proof. The source interlock permits only the
preregistered PERF019 namespace under its frozen conservative budget.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rayline_concurrency_comparator import compare_sweep
from rayline_concurrency_contract import (
    MEASURED_CASES,
    MEASURED_EPISODES,
    PERF019_RUN_ID,
    SWEEP_ARMS,
    WARMUP_CASES,
    WARMUP_EPISODES,
    ConcurrencyRunContract,
    SweepCell,
    resolve_launch_contract,
)
from rayline_concurrency_state import (
    STATE_RECEIPT_SCHEMA,
    ProtectedEncoderClient,
    StateResetError,
    assert_encoder_empty,
    close_cell_sessions,
)
from rayline_parity_http_probe import (
    PROTOCOL_BY_ARM,
    WORKLOAD_PROFILES,
    load_packet,
)
from rayline_three_arm_budget import budget_receipt
from rayline_three_arm_contract import IDENTITY, NON_RUNTIME_SECRET_NAMES
from rayline_three_arm_launcher import (
    LaunchError,
    _assert_pushed,
    _compose,
    _free_port,
    _modal_containers,
    _run,
    _sha256,
    _stop_modal_encoder,
    _stop_process,
    _wait_arc_ready,
    _wait_http,
    derive_pathfinder_config,
)
from rayline_three_arm_telemetry import capture_arc_telemetry

EXPECTED_ARC_REQUESTS = MEASURED_CASES + WARMUP_CASES


@dataclass(frozen=True)
class SweepContext:
    contract: ConcurrencyRunContract
    semantic_root: Path
    pathfinder_root: Path
    pathfinder_python: Path
    packet_dir: Path
    runtime_dir: Path
    router_image: str
    semantic_head: str
    pathfinder_head: str
    worker_ids: list[str]
    checkpoint: Path
    output_dir: Path
    base_environment: dict[str, str]
    modal: Any
    yaml: Any


@dataclass
class EncoderOwnership:
    manager: Any
    proxy: Any
    service_environment: dict[str, str]
    client: ProtectedEncoderClient
    cleanup: dict[str, Any] = field(
        default_factory=lambda: {
            "proxy_token_deleted": False,
            "encoder_containers_remaining": None,
        }
    )


@dataclass(frozen=True)
class PreparedCell:
    work: Path
    config_path: Path
    bundle: Path
    arc_config: Path


@dataclass
class LocalCell:
    compose_project: str
    compose_environment: dict[str, str]
    router_environment: dict[str, str]
    router_process: subprocess.Popen[str] | None = None
    router_log: Any = None
    cleanup: dict[str, bool] = field(
        default_factory=lambda: {
            "pathfinder_stopped": False,
            "compose_removed": False,
        }
    )


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=PERF019_RUN_ID)
    parser.add_argument("--pathfinder-root", type=Path, required=True)
    parser.add_argument(
        "--packet-dir",
        type=Path,
        default=root / ".agent-harness/rayline-parity/packet-perf017",
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


def _validate_packet(
    contract: ConcurrencyRunContract, packet_dir: Path
) -> tuple[list[str], tuple[str, ...], tuple[str, ...]]:
    if _sha256(packet_dir / "manifest.json") != contract.packet_manifest_sha256:
        raise LaunchError("concurrency packet manifest digest differs")
    if _sha256(packet_dir / "corpus.json") != contract.corpus_sha256:
        raise LaunchError("concurrency corpus digest differs")
    if _sha256(packet_dir / "topology.json") != contract.topology_sha256:
        raise LaunchError("concurrency topology digest differs")
    manifest = json.loads((packet_dir / "manifest.json").read_text())
    if (
        manifest.get("measured_cases") != MEASURED_CASES
        or manifest.get("warmup_cases") != WARMUP_CASES
    ):
        raise LaunchError("concurrency packet counts differ")
    for cell in contract.cells:
        cell_dir = packet_dir / "cells" / f"c{cell.concurrency}"
        if (
            _sha256(cell_dir / "workload.json") != cell.workload_sha256
            or _sha256(cell_dir / "identity.json") != cell.identity_sha256
        ):
            raise LaunchError(f"concurrency c{cell.concurrency} digest differs")
        for arm in SWEEP_ARMS:
            load_packet(
                arm=arm,
                corpus_path=packet_dir / "corpus.json",
                workload_path=cell_dir / "workload.json",
                topology_path=packet_dir / "topology.json",
                identity_path=cell_dir / "identity.json",
                workload_contract=WORKLOAD_PROFILES[cell.profile],
            )
    topology = json.loads((packet_dir / "topology.json").read_text())
    corpus = json.loads((packet_dir / "corpus.json").read_text())
    worker_ids = list(map(str, topology["canonical_workers"]))
    measured_episodes = tuple(
        dict.fromkeys(str(case["episode_id"]) for case in corpus["measured"])
    )
    warmup_episodes = tuple(
        dict.fromkeys(str(case["episode_id"]) for case in corpus["warmup"])
    )
    if (
        len(measured_episodes) != MEASURED_EPISODES
        or len(warmup_episodes) != WARMUP_EPISODES
    ):
        raise LaunchError("concurrency episode counts differ")
    return worker_ids, measured_episodes, warmup_episodes


def _preflight(args: argparse.Namespace) -> SweepContext:
    contract = resolve_launch_contract(args.run_id)
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
    worker_ids, _measured, _warmup = _validate_packet(contract, packet_dir)
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
    base_environment = {**os.environ, "MODAL_ENVIRONMENT": IDENTITY.modal_environment}
    if _modal_containers(pathfinder_python, pathfinder_root, base_environment):
        raise LaunchError("protected encoder already has a running container")
    for cell in contract.cells:
        compose_project = f"{contract.compose_project_prefix}-c{cell.concurrency}"
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


def _start_encoder(context: SweepContext) -> EncoderOwnership:
    manager = context.modal.Workspace.from_context().proxy_tokens
    proxy = manager.create()
    service_environment = context.base_environment.copy()
    for name in NON_RUNTIME_SECRET_NAMES:
        service_environment.pop(name, None)
    service_environment.update(
        {
            "RAYLINE_ARC_MODAL_KEY": proxy.token_id,
            "RAYLINE_ARC_MODAL_SECRET": proxy.token_secret,
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    client = ProtectedEncoderClient(
        IDENTITY.encoder_url,
        proxy.token_id,
        proxy.token_secret,
        timeout_seconds=30.0,
    )
    ownership = EncoderOwnership(manager, proxy, service_environment, client)
    try:
        _run(
            [
                str(context.pathfinder_python),
                "-m",
                "modal",
                "deploy",
                str(
                    context.semantic_root
                    / "src/vllm-plugins/rayline_arc_io/modal_session_service.py"
                ),
            ],
            cwd=context.pathfinder_root,
            environment=service_environment,
            timeout=15 * 60,
            capture=False,
        )
        deadline = time.monotonic() + 15 * 60
        last_error: StateResetError | None = None
        while time.monotonic() < deadline:
            try:
                assert_encoder_empty(client.request)
                break
            except StateResetError as error:
                last_error = error
                time.sleep(1)
        else:
            raise LaunchError("protected encoder did not become ready") from last_error
    except BaseException:
        _cleanup_encoder(context, ownership)
        raise
    return ownership


def _prepare_cell(context: SweepContext, cell: SweepCell, work: Path) -> PreparedCell:
    work.mkdir(parents=True)
    config_path = work / "pathfinder.yaml"
    base = context.yaml.safe_load(
        (context.pathfinder_root / "configs/live_gap_c82_coldswitch.yaml").read_text()
    )
    config = derive_pathfinder_config(
        base,
        checkpoint=context.checkpoint,
        decision_log=work / "pathfinder-decisions.jsonl",
        worker_ids=context.worker_ids,
        worker_model_prefix="mock/perf017",
    )
    config_path.write_text(context.yaml.safe_dump(config, sort_keys=False))
    bundle = work / "bundle"
    _run(
        [
            str(context.pathfinder_python),
            "scripts/build_router_bundle.py",
            "--config",
            str(config_path),
            "--output",
            str(bundle),
            "--bundle-version",
            f"{context.contract.run_id}-c{cell.concurrency}-bundle-v1",
            "--pricing-snapshot-version",
            f"{context.contract.run_id}-pricing-v1",
            "--checkpoint",
            str(context.checkpoint),
        ],
        cwd=context.pathfinder_root,
        timeout=120,
    )
    arc_config = work / "arc.yaml"
    _run(
        [
            sys.executable,
            str(
                context.semantic_root
                / "e2e/testing/rayline-arc/rayline_parity_arc_config.py"
            ),
            "--template",
            str(context.semantic_root / "deploy/compose/rayline-arc/config.yaml"),
            "--runtime-dir",
            str(context.runtime_dir),
            "--output",
            str(arc_config),
            "--artifact-mount-path",
            "/var/lib/vllm-sr/rayline-arc",
            "--encoder-base-url",
            IDENTITY.encoder_url,
            "--encoder-build-id",
            IDENTITY.engine_build_id,
            "--encoder-plugin-version",
            IDENTITY.plugin_version,
        ],
        cwd=context.semantic_root,
        timeout=30,
    )
    return PreparedCell(work, config_path, bundle, arc_config)


def _local_cell(
    context: SweepContext,
    encoder: EncoderOwnership,
    cell: SweepCell,
    prepared: PreparedCell,
    ports: Mapping[str, int],
) -> LocalCell:
    compose_environment = {
        **encoder.service_environment,
        "RAYLINE_PARITY_ROUTER_IMAGE": context.router_image,
        "RAYLINE_PARITY_ARC_CONFIG": str(prepared.arc_config),
        "RAYLINE_PARITY_ARC_ARTIFACT_DIR": str(context.runtime_dir),
        "RAYLINE_PARITY_ARC_ENVOY_PORT": str(ports["envoy"]),
        "RAYLINE_PARITY_ARC_ROUTER_PORT": str(ports["router"]),
        "RAYLINE_PARITY_ARC_METRICS_PORT": str(ports["metrics"]),
    }
    router_environment = encoder.service_environment.copy()
    pythonpath = str(context.pathfinder_root / "src")
    if router_environment.get("PYTHONPATH"):
        pythonpath += os.pathsep + router_environment["PYTHONPATH"]
    router_environment.update(
        {
            "PYTHONPATH": pythonpath,
            "RAYLINE_ROUTER_CONFIG": str(prepared.config_path),
            "RAYLINE_ROUTER_BUNDLE_URI": str(prepared.bundle),
            "RAYLINE_ROUTER_DECISION_ONLY": "1",
        }
    )
    return LocalCell(
        compose_project=(
            f"{context.contract.compose_project_prefix}-c{cell.concurrency}"
        ),
        compose_environment=compose_environment,
        router_environment=router_environment,
    )


def _start_local(
    context: SweepContext,
    prepared: PreparedCell,
    local: LocalCell,
    ports: Mapping[str, int],
) -> tuple[str, str]:
    local.router_log = (prepared.work / "pathfinder.log").open("w", encoding="utf-8")
    local.router_process = subprocess.Popen(
        [
            str(context.pathfinder_python),
            "-m",
            "uvicorn",
            "rayline_router.serving.app:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(ports["pathfinder"]),
            "--log-level",
            "warning",
        ],
        cwd=context.pathfinder_root,
        env=local.router_environment,
        text=True,
        stdout=local.router_log,
        stderr=subprocess.STDOUT,
    )
    router_url = f"http://127.0.0.1:{ports['pathfinder']}"
    _wait_http(f"{router_url}/healthz", 180)
    _compose(
        context.semantic_root,
        local.compose_environment,
        local.compose_project,
        "up",
        "--build",
        "--detach",
    )
    _wait_http(f"http://127.0.0.1:{ports['router']}/health", 60)
    _wait_arc_ready(f"http://127.0.0.1:{ports['metrics']}/metrics", 180)
    return router_url, f"http://127.0.0.1:{ports['envoy']}"


def _probe_cell(
    context: SweepContext,
    cell: SweepCell,
    arm: str,
    base_url: str,
    output_dir: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    output = output_dir / f"{arm}.json"
    cell_dir = context.packet_dir / "cells" / f"c{cell.concurrency}"
    probe_run_id = f"{context.contract.run_id}:c{cell.concurrency}:{arm}"
    _run(
        [
            sys.executable,
            str(
                context.semantic_root
                / "e2e/testing/rayline-arc/rayline_parity_http_probe.py"
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
            "--workload-profile",
            cell.profile,
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


def _cleanup_local(context: SweepContext, local: LocalCell) -> None:
    _stop_process(local.router_process)
    local.cleanup["pathfinder_stopped"] = (
        local.router_process is None or local.router_process.poll() is not None
    )
    if local.router_log is not None:
        local.router_log.close()
    result = _compose(
        context.semantic_root,
        local.compose_environment,
        local.compose_project,
        "down",
        "--volumes",
        "--remove-orphans",
        check=False,
    )
    local.cleanup["compose_removed"] = result.returncode == 0


def _cell_episode_ids(context: SweepContext) -> tuple[tuple[str, ...], tuple[str, ...]]:
    corpus = json.loads((context.packet_dir / "corpus.json").read_text())
    measured = tuple(
        dict.fromkeys(str(case["episode_id"]) for case in corpus["measured"])
    )
    warmup = tuple(dict.fromkeys(str(case["episode_id"]) for case in corpus["warmup"]))
    return measured, warmup


def _reset_encoder_after_cell(
    *,
    encoder: EncoderOwnership,
    arc_run_id: str,
    measured_episodes: tuple[str, ...],
    warmup_episodes: tuple[str, ...],
    arc_started: bool,
    arc_completed: bool,
) -> dict[str, Any]:
    if arc_started:
        return close_cell_sessions(
            requester=encoder.client.request,
            probe_run_id=arc_run_id,
            measured_episode_ids=measured_episodes,
            warmup_episode_ids=warmup_episodes,
            require_measured_present=arc_completed,
        )
    empty = assert_encoder_empty(encoder.client.request)
    return {
        "schema_version": STATE_RECEIPT_SCHEMA,
        "measured_episode_count": len(measured_episodes),
        "measured_sessions_closed": 0,
        "measured_sessions_missing": len(measured_episodes),
        "warmup_episode_count": len(warmup_episodes),
        "warmup_sessions_closed": 0,
        "warmup_sessions_missing": len(warmup_episodes),
        "resident_sessions_after_cleanup": empty["resident_sessions"],
        "resident_tokens_after_cleanup": empty["resident_tokens"],
    }


def _run_cell(
    context: SweepContext,
    encoder: EncoderOwnership,
    cell: SweepCell,
    work: Path,
    paid_started: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    prepared = _prepare_cell(context, cell, work)
    ports = {
        name: _free_port() for name in ("pathfinder", "envoy", "router", "metrics")
    }
    local = _local_cell(context, encoder, cell, prepared, ports)
    cell_output = context.output_dir / f"c{cell.concurrency}"
    cell_output.mkdir()
    measured_episodes, warmup_episodes = _cell_episode_ids(context)
    arc_run_id = f"{context.contract.run_id}:c{cell.concurrency}:rayline_arc"
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
                context,
                cell,
                arm,
                base_url,
                cell_output,
                remaining,
            )
            if arm == "rayline_arc":
                arc_completed = (
                    receipts[arm]["results"]["completed"] == MEASURED_CASES
                    and receipts[arm]["results"]["failed"] == 0
                )
        telemetry = capture_arc_telemetry(
            f"http://127.0.0.1:{ports['metrics']}/metrics",
            cell_output / "rayline_arc_telemetry.json",
        )
        action_count = sum(telemetry["session_actions"].values())
        if action_count != EXPECTED_ARC_REQUESTS:
            raise LaunchError("ARC telemetry count differs from the sweep packet")
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


def _cleanup_encoder(context: SweepContext, encoder: EncoderOwnership) -> None:
    try:
        encoder.manager.delete(encoder.proxy.token_id)
        encoder.cleanup["proxy_token_deleted"] = True
    finally:
        _stop_modal_encoder(
            context.pathfinder_python,
            context.pathfinder_root,
            encoder.service_environment,
            context.contract.run_id,
        )
    remaining = _modal_containers(
        context.pathfinder_python,
        context.pathfinder_root,
        encoder.service_environment,
    )
    encoder.cleanup["encoder_containers_remaining"] = len(remaining)


def _write_manifest(
    context: SweepContext,
    comparison: Mapping[str, Any],
    cell_cleanup: Mapping[str, Any],
    encoder_cleanup: Mapping[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    manifest = {
        "schema_version": "rayline.vllm.concurrency-sweep-run.v1",
        "run_id": context.contract.run_id,
        "source": {
            "semantic_router_commit": context.semantic_head,
            "pathfinder_commit": context.pathfinder_head,
            "engine_build_id": IDENTITY.engine_build_id,
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
    encoder = _start_encoder(context)
    raw_cells: dict[int, dict[str, dict[str, Any]]] = {}
    cell_cleanup: dict[str, Any] = {}
    try:
        with tempfile.TemporaryDirectory(
            prefix=context.contract.temporary_prefix
        ) as temp_name:
            temp_root = Path(temp_name)
            for cell in context.contract.cells:
                raw_cells[cell.concurrency], cell_cleanup[f"c{cell.concurrency}"] = (
                    _run_cell(
                        context,
                        encoder,
                        cell,
                        temp_root / f"c{cell.concurrency}",
                        paid_started,
                    )
                )
        comparison = compare_sweep(raw_cells)
        (context.output_dir / "comparison.json").write_text(
            json.dumps(comparison, indent=2, sort_keys=True) + "\n"
        )
    finally:
        _cleanup_encoder(context, encoder)
    elapsed = time.perf_counter() - paid_started
    manifest = _write_manifest(
        context,
        comparison,
        cell_cleanup,
        encoder.cleanup,
        elapsed,
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
