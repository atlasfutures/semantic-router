#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the bounded PERF022 single-versus-dual ARC affinity experiment."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import subprocess
import tempfile
import time
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from rayline_concurrency_launcher import (
    LocalCell,
    PreparedCell,
    SweepContext,
    _cell_episode_ids,
    _cleanup_local,
    _local_cell,
    _prepare_cell,
    _reset_encoder_after_cell,
    _start_local,
)
from rayline_concurrency_state import (
    ProtectedEncoderClient,
    StateResetError,
    assert_encoder_empty,
)
from rayline_open_loop_contract import (
    MEASURED_CASES,
    MEASURED_EPISODES,
    WARMUP_CASES,
    WARMUP_EPISODES,
)
from rayline_open_loop_launcher import _probe_cell
from rayline_open_loop_probe import load_open_loop_packet
from rayline_scaleout_comparator import compare_scaleout
from rayline_scaleout_contract import (
    ENCODER_APP_NAMES,
    MAXIMUM_SCALEDOWN_SECONDS,
    PATHFINDER_AUTHORIZATION_COMMIT,
    PERF024_RUN_ID,
    SCALEOUT_ARMS,
    OpenLoopCell,
    ScaleoutRunContract,
    resolve_launch_contract,
)
from rayline_three_arm_budget import budget_receipt
from rayline_three_arm_contract import IDENTITY, NON_RUNTIME_SECRET_NAMES
from rayline_three_arm_launcher import (
    LaunchError,
    _assert_pushed,
    _free_port,
    _run,
    _sha256,
    _stop_process,
)
from rayline_three_arm_telemetry import capture_arc_telemetry

EXPECTED_ARC_REQUESTS = 36
EXPECTED_EPISODES = 9
HTTP_OK = 200
AFFINITY_STATS_PATH = "/v1/rayline/affinity/stats"
AFFINITY_RESET_PATH = "/v1/rayline/affinity/reset"
SERVICE_PATH = Path(__file__).resolve().parents[3] / (
    "src/vllm-plugins/rayline_arc_io/modal_session_service.py"
)


@dataclass
class EncoderPairOwnership:
    manager: Any
    proxy: Any
    service_environment: dict[str, str]
    base_urls: tuple[str, str]
    cleanup: dict[str, Any] = field(
        default_factory=lambda: {
            "proxy_token_deleted": False,
            "encoder_apps_stopped": False,
            "encoder_containers_remaining": None,
        }
    )


@dataclass
class AffinityOwnership:
    process: subprocess.Popen[str]
    client: ProtectedEncoderClient
    local_base_url: str


@dataclass(frozen=True)
class PreparedArm:
    affinity: AffinityOwnership
    prepared: PreparedCell
    ports: dict[str, int]
    local: LocalCell
    output: Path


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=PERF024_RUN_ID)
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


def _validate_packet(contract: ScaleoutRunContract, packet_dir: Path) -> list[str]:
    if _sha256(packet_dir / "manifest.json") != contract.packet_manifest_sha256:
        raise LaunchError("scale-out packet manifest digest differs")
    if _sha256(packet_dir / "corpus.json") != contract.corpus_sha256:
        raise LaunchError("scale-out corpus digest differs")
    if _sha256(packet_dir / "topology.json") != contract.topology_sha256:
        raise LaunchError("scale-out topology digest differs")
    manifest = json.loads((packet_dir / "manifest.json").read_text())
    if (
        manifest.get("measured_cases") != MEASURED_CASES
        or manifest.get("warmup_cases") != WARMUP_CASES
        or manifest.get("measured_episodes") != MEASURED_EPISODES
        or manifest.get("warmup_episodes") != WARMUP_EPISODES
        or not {cell.label for cell in contract.cells}.issubset(
            set(manifest.get("cells", {}))
        )
    ):
        raise LaunchError("scale-out packet shape differs")
    for cell in contract.cells:
        cell_dir = packet_dir / "cells" / cell.label
        if (
            _sha256(cell_dir / "workload.json") != cell.workload_sha256
            or _sha256(cell_dir / "identity.json") != cell.identity_sha256
        ):
            raise LaunchError(f"scale-out {cell.label} digest differs")
        _warmup, _measured, _identity, _workers, workload = load_open_loop_packet(
            arm="rayline_arc",
            corpus_path=packet_dir / "corpus.json",
            workload_path=cell_dir / "workload.json",
            topology_path=packet_dir / "topology.json",
            identity_path=cell_dir / "identity.json",
        )
        if workload["offered_rate_rps"] != cell.offered_rate_rps:
            raise LaunchError(f"scale-out {cell.label} rate differs")
    topology = json.loads((packet_dir / "topology.json").read_text())
    return list(map(str, topology["canonical_workers"]))


def _named_encoder_containers(context: SweepContext) -> list[dict[str, Any]]:
    result = _run(
        [
            str(context.pathfinder_python),
            "-m",
            "modal",
            "container",
            "list",
            "--json",
        ],
        cwd=context.pathfinder_root,
        environment=context.base_environment,
        timeout=30,
    )
    return [
        row
        for row in json.loads(result.stdout)
        if row.get("app_name") in ENCODER_APP_NAMES
    ]


def _named_encoder_apps(context: SweepContext) -> list[dict[str, Any]]:
    result = _run(
        [
            str(context.pathfinder_python),
            "-m",
            "modal",
            "app",
            "list",
            "--json",
        ],
        cwd=context.pathfinder_root,
        environment=context.base_environment,
        timeout=30,
    )
    return [
        row
        for row in json.loads(result.stdout)
        if row.get("description") in ENCODER_APP_NAMES
    ]


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
    if pathfinder_head != PATHFINDER_AUTHORIZATION_COMMIT:
        raise LaunchError("Pathfinder PERF022 authorization head differs")
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
    base_environment = {**os.environ, "MODAL_ENVIRONMENT": IDENTITY.modal_environment}
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
    context = SweepContext(
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
    if _named_encoder_containers(context):
        raise LaunchError("a PERF022 encoder replica already has a container")
    for cell in contract.cells:
        for arm in SCALEOUT_ARMS:
            project = f"{contract.compose_project_prefix}-{cell.label}-{arm}"
            existing = _run(
                ["docker", "ps", "-aq", "--filter", f"name={project}"],
                cwd=semantic_root,
            ).stdout.strip()
            if existing:
                raise LaunchError(f"{project} already exists")
    output_dir.mkdir(parents=True)
    return context


def _direct_client(
    base_url: str, key: str, secret: str, timeout_seconds: float = 180.0
) -> httpx.Client:
    return httpx.Client(
        base_url=base_url,
        headers={"Modal-Key": key, "Modal-Secret": secret},
        timeout=httpx.Timeout(timeout_seconds, connect=10.0, pool=10.0),
    )


def _warm_replica(base_url: str, key: str, secret: str, label: str) -> None:
    episode_hash = hashlib.sha256(label.encode()).hexdigest()
    payload = {
        "schema_version": "rayline.arc.session-pooling-request.v1",
        "serializer_version": "mtrouter-token-blocks-v2",
        "serving_rung": "B",
        "episode_id_hash": episode_hash,
        "turns": [{"role": "user", "text": "Rayline scale-out warmup"}],
    }
    with _direct_client(base_url, key, secret) as client:
        response = client.post("/v1/rayline/arc/session/pooling", json=payload)
        if response.status_code != HTTP_OK:
            raise LaunchError("encoder replica warmup failed")
        closed = client.delete(f"/v1/rayline/arc/session/{episode_hash}")
        if closed.status_code != HTTP_OK or closed.json().get("closed") is not True:
            raise LaunchError("encoder replica warmup cleanup failed")


def _deployed_encoder_url(context: SweepContext, app_name: str) -> str:
    cls = context.modal.Cls.from_name(
        app_name,
        "SessionEncoder",
        environment_name=IDENTITY.modal_environment,
    )
    url = cls().web.get_web_url()
    if not url:
        raise LaunchError("encoder replica web URL is unavailable")
    return url.rstrip("/")


def _start_encoder_pair(context: SweepContext) -> EncoderPairOwnership:
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
    ownership = EncoderPairOwnership(manager, proxy, service_environment, ("", ""))
    try:
        urls: list[str] = []
        for app_name in ENCODER_APP_NAMES:
            deploy_environment = {
                **service_environment,
                "RAYLINE_ARC_SESSION_APP_NAME": app_name,
            }
            _run(
                [
                    str(context.pathfinder_python),
                    "-m",
                    "modal",
                    "deploy",
                    str(SERVICE_PATH),
                ],
                cwd=context.pathfinder_root,
                environment=deploy_environment,
                timeout=15 * 60,
                capture=False,
            )
            urls.append(_deployed_encoder_url(context, app_name))
        ownership.base_urls = (urls[0], urls[1])
        deadline = time.monotonic() + 15 * 60
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                for index, base_url in enumerate(ownership.base_urls):
                    _warm_replica(
                        base_url,
                        proxy.token_id,
                        proxy.token_secret,
                        f"{context.contract.run_id}:replica-{index}",
                    )
                    direct = ProtectedEncoderClient(
                        base_url,
                        proxy.token_id,
                        proxy.token_secret,
                        timeout_seconds=30.0,
                    )
                    assert_encoder_empty(direct.request)
                break
            except (httpx.HTTPError, StateResetError, LaunchError) as error:
                last_error = error
                time.sleep(1)
        else:
            raise LaunchError("encoder replicas did not become ready") from last_error
    except BaseException as error:
        try:
            _cleanup_encoder_pair(context, ownership)
        except BaseException as cleanup_error:
            raise LaunchError(
                "encoder startup and exact-app cleanup both failed"
            ) from cleanup_error
        raise error
    return ownership


def _stop_app(context: SweepContext, environment: Mapping[str, str], app: str) -> bool:
    result = _run(
        [
            str(context.pathfinder_python),
            "-m",
            "modal",
            "app",
            "stop",
            "-y",
            app,
        ],
        cwd=context.pathfinder_root,
        environment=environment,
        timeout=120,
        check=False,
    )
    return result.returncode == 0


def _cleanup_encoder_pair(
    context: SweepContext, ownership: EncoderPairOwnership
) -> None:
    for app in ENCODER_APP_NAMES:
        # Final exact-name inventory below, rather than one CLI response, is
        # the cleanup source of truth. Always continue to token deletion.
        with contextlib.suppress(BaseException):
            _stop_app(context, ownership.service_environment, app)
    try:
        ownership.manager.delete(ownership.proxy.token_id)
        ownership.cleanup["proxy_token_deleted"] = True
    finally:
        deadline = time.monotonic() + MAXIMUM_SCALEDOWN_SECONDS
        while True:
            apps = _named_encoder_apps(context)
            ownership.cleanup["encoder_apps_stopped"] = all(
                row.get("state") == "stopped" and str(row.get("tasks")) == "0"
                for row in apps
            )
            ownership.cleanup["encoder_containers_remaining"] = len(
                _named_encoder_containers(context)
            )
            if (
                ownership.cleanup["encoder_apps_stopped"]
                and ownership.cleanup["encoder_containers_remaining"] == 0
            ):
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(1)
    if (
        not ownership.cleanup["proxy_token_deleted"]
        or not ownership.cleanup["encoder_apps_stopped"]
        or ownership.cleanup["encoder_containers_remaining"] != 0
    ):
        raise LaunchError("encoder pair cleanup did not reach exact-name zero")


def _start_affinity(
    context: SweepContext,
    pair: EncoderPairOwnership,
    upstreams: tuple[str, ...],
) -> AffinityOwnership:
    port = _free_port()
    command = [
        str(context.pathfinder_python),
        str(Path(__file__).with_name("rayline_affinity_proxy.py")),
        "--listen-host",
        "0.0.0.0",
        "--port",
        str(port),
    ]
    for upstream in upstreams:
        command.extend(("--upstream", upstream))
    process = subprocess.Popen(
        command,
        cwd=context.semantic_root,
        env=pair.service_environment,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    local_base_url = f"http://127.0.0.1:{port}"
    client = ProtectedEncoderClient(
        local_base_url,
        pair.proxy.token_id,
        pair.proxy.token_secret,
        timeout_seconds=180.0,
    )
    deadline = time.monotonic() + 15 * 60
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise LaunchError("affinity proxy exited during readiness")
        try:
            assert_encoder_empty(client.request)
            return AffinityOwnership(process, client, local_base_url)
        except StateResetError as error:
            last_error = error
            time.sleep(0.25)
    _stop_process(process)
    raise LaunchError("affinity proxy did not become ready") from last_error


def _proxy_json(ownership: AffinityOwnership, method: str, path: str) -> dict[str, Any]:
    request = urllib.request.Request(ownership.local_base_url + path, method=method)
    with urllib.request.urlopen(request, timeout=10) as response:
        if response.status != HTTP_OK:
            raise LaunchError("affinity control endpoint failed")
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise LaunchError("affinity control endpoint returned a non-object")
    return value


def _finalize_arm_state(
    *,
    affinity: AffinityOwnership,
    arc_run_id: str,
    measured_episodes: tuple[str, ...],
    warmup_episodes: tuple[str, ...],
    arc_started: bool,
    arc_completed: bool,
    cell_output: Path,
    logical_arm: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state_receipt = _reset_encoder_after_cell(
        encoder=affinity,
        arc_run_id=arc_run_id,
        measured_episodes=measured_episodes,
        warmup_episodes=warmup_episodes,
        arc_started=arc_started,
        arc_completed=arc_completed,
    )
    (cell_output / f"{logical_arm}_state-reset.json").write_text(
        json.dumps(state_receipt, indent=2, sort_keys=True) + "\n"
    )
    affinity_receipt = _proxy_json(affinity, "GET", AFFINITY_STATS_PATH)
    (cell_output / f"{logical_arm}_affinity.json").write_text(
        json.dumps(affinity_receipt, indent=2, sort_keys=True) + "\n"
    )
    return state_receipt, affinity_receipt


def _prepare_arm(
    context: SweepContext,
    pair: EncoderPairOwnership,
    cell: OpenLoopCell,
    logical_arm: str,
    work: Path,
) -> PreparedArm:
    upstreams = pair.base_urls[:1] if logical_arm == "arc_single" else pair.base_urls
    affinity = _start_affinity(context, pair, upstreams)
    prepared = _prepare_cell(
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
    local: LocalCell = _local_cell(context, pair, cell, prepared, ports)
    local.compose_project = (
        f"{context.contract.compose_project_prefix}-{cell.label}-{logical_arm}"
    )
    output = context.output_dir / cell.label
    output.mkdir(exist_ok=True)
    return PreparedArm(affinity, prepared, ports, local, output)


def _run_arm(
    context: SweepContext,
    pair: EncoderPairOwnership,
    cell: OpenLoopCell,
    logical_arm: str,
    work: Path,
    paid_started: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    arm = _prepare_arm(context, pair, cell, logical_arm, work)
    measured_episodes, warmup_episodes = _cell_episode_ids(context)
    arc_run_id = f"{context.contract.run_id}:{cell.label}:{logical_arm}"
    arc_started = False
    arc_completed = False
    receipt: dict[str, Any] | None = None
    state_receipt: dict[str, Any] | None = None
    affinity_receipt: dict[str, Any] | None = None
    failure: BaseException | None = None
    try:
        _router_url, arc_url = _start_local(context, arm.prepared, arm.local, arm.ports)
        assert_encoder_empty(arm.affinity.client.request)
        _proxy_json(arm.affinity, "POST", AFFINITY_RESET_PATH)
        arc_started = True
        remaining = context.contract.budget.maximum_paid_wall_seconds - (
            time.perf_counter() - paid_started
        )
        if remaining <= 0:
            raise LaunchError("PERF022 paid wall-time ceiling reached")
        receipt = _probe_cell(
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
            raise LaunchError("ARC telemetry count differs from PERF022")
    except BaseException as error:
        failure = error
    finally:
        _cleanup_local(context, arm.local)
        try:
            state_receipt, affinity_receipt = _finalize_arm_state(
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
                "scale-out arm execution and state cleanup both failed"
            ) from cleanup_error
        finally:
            _stop_process(arm.affinity.process)
    if failure is not None:
        raise failure
    if (
        receipt is None
        or state_receipt is None
        or affinity_receipt is None
        or not all(arm.local.cleanup.values())
    ):
        raise LaunchError("scale-out arm cleanup did not complete")
    return (
        receipt,
        affinity_receipt,
        {
            "local": arm.local.cleanup,
            "encoder_state": state_receipt,
            "affinity_proxy_stopped": arm.affinity.process.poll() is not None,
        },
    )


def _write_manifest(
    context: SweepContext,
    comparison: Mapping[str, Any],
    arm_cleanup: Mapping[str, Any],
    encoder_cleanup: Mapping[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    manifest = {
        "schema_version": "rayline.vllm.affinity-scaleout-run.v1",
        "run_id": context.contract.run_id,
        "source": {
            "semantic_router_commit": context.semantic_head,
            "pathfinder_commit": context.pathfinder_head,
            "engine_build_id": IDENTITY.engine_build_id,
            "plugin_version": IDENTITY.plugin_version,
            "packet_manifest_sha256": context.contract.packet_manifest_sha256,
            "encoder_app_names": list(ENCODER_APP_NAMES),
        },
        "budget": budget_receipt(context.contract.budget, elapsed),
        "comparison_status": comparison["status"],
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
    pair = _start_encoder_pair(context)
    raw_cells: dict[str, dict[str, dict[str, Any]]] = {}
    raw_affinity: dict[str, dict[str, dict[str, Any]]] = {}
    arm_cleanup: dict[str, dict[str, Any]] = {}
    try:
        with tempfile.TemporaryDirectory(
            prefix=context.contract.temporary_prefix
        ) as temp_name:
            temp_root = Path(temp_name)
            for cell in context.contract.cells:
                raw_cells[cell.label] = {}
                raw_affinity[cell.label] = {}
                arm_cleanup[cell.label] = {}
                for arm in SCALEOUT_ARMS:
                    receipt, affinity, cleanup = _run_arm(
                        context,
                        pair,
                        cell,
                        arm,
                        temp_root / cell.label / arm,
                        paid_started,
                    )
                    raw_cells[cell.label][arm] = receipt
                    raw_affinity[cell.label][arm] = affinity
                    arm_cleanup[cell.label][arm] = cleanup
        comparison = compare_scaleout(raw_cells, raw_affinity)
        (context.output_dir / "comparison.json").write_text(
            json.dumps(comparison, indent=2, sort_keys=True) + "\n"
        )
    finally:
        _cleanup_encoder_pair(context, pair)
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
