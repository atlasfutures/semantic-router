#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run one bounded, preregistered Rayline three-arm packet.

This launcher owns one protected Modal H100 encoder, one local Pathfinder
process, and one local ARC Compose project. It refuses mutation until source,
packet, artifact, image, credentials, cost, and zero-container preflights pass.
The 1,000-case release qualification is intentionally unreachable here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rayline_parity_comparator import ARMS, compare_receipts
from rayline_parity_http_probe import PROTOCOL_BY_ARM, load_packet
from rayline_three_arm_budget import budget_receipt
from rayline_three_arm_contract import (
    IDENTITY,
    NON_RUNTIME_SECRET_NAMES,
    PERF016_RUN_ID,
    RunContract,
    resolve_launch_contract,
)
from rayline_three_arm_telemetry import capture_arc_telemetry

MAX_CLEANUP_SECONDS = 180
STABLE_ZERO_SECONDS = 65
HTTP_OK = 200


class LaunchError(RuntimeError):
    """A fail-closed preflight, launch, or cleanup condition was observed."""


@dataclass(frozen=True)
class LaunchContext:
    contract: RunContract
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


@dataclass(frozen=True)
class PreparedWorkload:
    work: Path
    config_path: Path
    bundle: Path
    arc_config: Path


@dataclass
class OwnedResources:
    manager: Any
    proxy: Any
    service_environment: dict[str, str]
    compose_environment: dict[str, str]
    router_environment: dict[str, str]
    router_process: subprocess.Popen[str] | None = None
    router_log: Any = None
    cleanup: dict[str, Any] = field(
        default_factory=lambda: {
            "pathfinder_stopped": False,
            "compose_removed": False,
            "proxy_token_deleted": False,
            "encoder_containers_remaining": None,
        }
    )


CHILD_OUTPUT_TAIL_CHARS = 4000


def _tail(stream: str | None, label: str) -> str:
    if not stream:
        return f"{label}: <empty>"
    text = stream.strip()
    if len(text) > CHILD_OUTPUT_TAIL_CHARS:
        text = "...\n" + text[-CHILD_OUTPUT_TAIL_CHARS:]
    return f"{label}:\n{text}"


def _run(
    command: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str] | None = None,
    timeout: float | None = None,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=dict(environment) if environment is not None else None,
            text=True,
            capture_output=capture,
            timeout=timeout,
            check=check,
        )
    except subprocess.CalledProcessError as error:
        # `capture_output=True` swallows the child's diagnosis, and
        # CalledProcessError's own message names only the command and the
        # exit code. A probe or bundle-build failure then surfaced as a
        # traceback with no cause, which cost real diagnostic cycles. The
        # environment is never echoed, so no secret reaches this message.
        raise LaunchError(
            f"{command[0]} exited {error.returncode}\n"
            f"command: {' '.join(command)}\n"
            f"{_tail(error.stderr, 'stderr')}\n"
            f"{_tail(error.stdout, 'stdout')}"
        ) from error


def _git(root: Path, *args: str) -> str:
    return _run(["git", *args], cwd=root, timeout=30).stdout.strip()


def _assert_pushed(root: Path, branch: str, remote: str, run_id: str) -> str:
    head = _git(root, "rev-parse", "HEAD")
    if _git(root, "branch", "--show-current") != branch:
        raise LaunchError(f"{root.name} must be on {branch}")
    if _git(root, "status", "--porcelain"):
        raise LaunchError(f"{root.name} must be clean before {run_id}")
    if _git(root, "rev-parse", f"{remote}/{branch}") != head:
        raise LaunchError(f"{root.name} must be pushed before {run_id}")
    return head


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_pathfinder_config(
    base: Mapping[str, Any],
    *,
    checkpoint: Path,
    decision_log: Path,
    worker_ids: list[str],
    worker_model_prefix: str = "mock/perf015",
    encoder_base_url: str = IDENTITY.encoder_url,
    encoder_build_id: str = IDENTITY.engine_build_id,
) -> dict[str, Any]:
    """Derive the local Pathfinder router config for one cell.

    The encoder is a parameter because the Pathfinder router resolves its
    encoder ONLY from ``router.mtrouter_vllm_base_url`` in this file -- it
    reads no environment override -- so the `pathfinder_transaction` arms
    reach the encoder through here, not through the ARC config. A run that
    owns an experiment-profile app, or that fronts the encoder with a local
    affinity proxy, must say so here or its local router would silently talk
    to the default frozen app while ARC talked to the run's own.

    The defaults are PERF020/PERF021's frozen identity, so a run that does not
    override the encoder derives a byte-identical config.
    """

    derived = copy.deepcopy(dict(base))
    router = derived.get("router")
    workers = derived.get("workers")
    if not isinstance(router, dict) or not isinstance(workers, list):
        raise LaunchError("Pathfinder C82 config omits router/workers")
    observed_workers = [str(worker.get("id") or "") for worker in workers]
    if observed_workers != worker_ids:
        raise LaunchError("Pathfinder workers differ from the frozen topology")
    router.update(
        {
            "checkpoint_path": str(checkpoint),
            "log_path": str(decision_log),
            "mtrouter_device": "cpu",
            "mtrouter_incremental_encode": False,
            "mtrouter_encoder_backend": "vllm",
            "mtrouter_vllm_base_url": encoder_base_url,
            "mtrouter_vllm_expected_build_id": encoder_build_id,
            "mtrouter_vllm_expected_plugin_version": IDENTITY.plugin_version,
            "mtrouter_vllm_timeout_s": 180.0,
            "mtrouter_vllm_connect_timeout_s": 10.0,
            "mtrouter_vllm_modal_key_env": "RAYLINE_ARC_MODAL_KEY",
            "mtrouter_vllm_modal_secret_env": "RAYLINE_ARC_MODAL_SECRET",
            "trace_store": "memory",
        }
    )
    required_prices = (
        "estimated_input_cost_per_token",
        "estimated_cache_read_cost_per_token",
        "estimated_cache_write_cost_per_token",
        "estimated_output_cost_per_token",
    )
    for index, worker in enumerate(workers):
        if not isinstance(worker, dict) or any(
            price not in worker for price in required_prices
        ):
            raise LaunchError("Pathfinder worker omits an artifact price")
        worker.update(
            {
                "backend": "mock",
                "model": f"{worker_model_prefix}-{index}",
                "api_key_env": "",
            }
        )
    return derived


def _modal_containers(
    python: Path,
    root: Path,
    environment: Mapping[str, str],
    app_name: str = IDENTITY.encoder_app_name,
) -> list[str]:
    result = _run(
        [str(python), "-m", "modal", "container", "list", "--json"],
        cwd=root,
        environment=environment,
        timeout=30,
    )
    return sorted(
        str(row["container_id"])
        for row in json.loads(result.stdout)
        if row.get("app_name") == app_name
    )


def _stop_modal_encoder(
    python: Path,
    root: Path,
    environment: Mapping[str, str],
    run_id: str,
    app_name: str = IDENTITY.encoder_app_name,
) -> None:
    # The app name is a parameter because a run may own an experiment-profile
    # encoder app. Stopping and counting must address the app that was
    # actually deployed, or a profiled run would leak a live H100 while
    # reporting the default app clean.
    _run(
        [
            str(python),
            "-m",
            "modal",
            "app",
            "stop",
            app_name,
            "--yes",
        ],
        cwd=root,
        environment=environment,
        timeout=30,
        check=False,
    )
    deadline = time.monotonic() + MAX_CLEANUP_SECONDS
    zero_since: float | None = None
    while time.monotonic() < deadline:
        containers = _modal_containers(python, root, environment, app_name)
        if containers:
            zero_since = None
            for container in containers:
                _run(
                    [
                        str(python),
                        "-m",
                        "modal",
                        "container",
                        "stop",
                        container,
                        "--yes",
                    ],
                    cwd=root,
                    environment=environment,
                    timeout=30,
                    check=False,
                )
        elif zero_since is None:
            zero_since = time.monotonic()
        elif time.monotonic() - zero_since >= STABLE_ZERO_SECONDS:
            return
        time.sleep(1)
    raise LaunchError(f"{run_id} encoder cleanup did not reach stable zero")


def _wait_http(url: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == HTTP_OK:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.25)
    raise LaunchError(f"service did not become ready: {url}")


def _wait_arc_ready(url: str, timeout_seconds: float) -> None:
    marker = 'llm_rayline_arc_component_ready{component="artifact_head_encoder"} 1'
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if marker in response.read().decode(errors="replace"):
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.25)
    raise LaunchError("ARC artifact/head/encoder component did not become ready")


def _stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _compose(
    semantic_root: Path,
    environment: Mapping[str, str],
    compose_project: str,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            "docker",
            "compose",
            "--project-name",
            compose_project,
            "--file",
            str(semantic_root / "deploy/compose/rayline-arc/compose.parity.yaml"),
            *args,
        ],
        cwd=semantic_root,
        environment=environment,
        timeout=180,
        check=check,
    )


def _probe(
    semantic_root: Path,
    packet_dir: Path,
    output_dir: Path,
    arm: str,
    base_url: str,
    timeout_seconds: float,
    run_id: str,
) -> dict[str, Any]:
    output = output_dir / f"{arm}.json"
    _run(
        [
            sys.executable,
            str(semantic_root / "e2e/testing/rayline-arc/rayline_parity_http_probe.py"),
            "--arm",
            arm,
            "--protocol",
            PROTOCOL_BY_ARM[arm],
            "--base-url",
            base_url,
            "--corpus",
            str(packet_dir / "corpus.json"),
            "--workload",
            str(packet_dir / "workload.json"),
            "--topology",
            str(packet_dir / "topology.json"),
            "--identity",
            str(packet_dir / "identity.json"),
            "--run-id",
            f"{run_id}:{arm}",
            "--output",
            str(output),
            "--timeout-seconds",
            "180",
        ],
        cwd=semantic_root,
        timeout=timeout_seconds,
    )
    return json.loads(output.read_text())


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=PERF016_RUN_ID)
    parser.add_argument("--pathfinder-root", type=Path, required=True)
    parser.add_argument(
        "--packet-dir",
        type=Path,
        default=root / ".agent-harness/rayline-parity/packet-v3",
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


def _preflight(args: argparse.Namespace) -> LaunchContext:
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
    for arm in ARMS:
        load_packet(
            arm=arm,
            corpus_path=packet_dir / "corpus.json",
            workload_path=packet_dir / "workload.json",
            topology_path=packet_dir / "topology.json",
            identity_path=packet_dir / "identity.json",
        )
    topology = json.loads((packet_dir / "topology.json").read_text())
    worker_ids = list(map(str, topology["canonical_workers"]))
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
    if _modal_containers(pathfinder_python, pathfinder_root, base_environment):
        raise LaunchError("protected encoder already has a running container")
    existing = _run(
        [
            "docker",
            "ps",
            "-aq",
            "--filter",
            f"name={contract.compose_project}",
        ],
        cwd=semantic_root,
    ).stdout.strip()
    if existing:
        raise LaunchError(f"{contract.run_id} Compose project already exists")

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
    return LaunchContext(
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


def _prepare_workload(context: LaunchContext, work: Path) -> PreparedWorkload:
    config_path = work / "pathfinder.yaml"
    decision_log = work / "pathfinder-decisions.jsonl"
    base = context.yaml.safe_load(
        (context.pathfinder_root / "configs/live_gap_c82_coldswitch.yaml").read_text()
    )
    config = derive_pathfinder_config(
        base,
        checkpoint=context.checkpoint,
        decision_log=decision_log,
        worker_ids=context.worker_ids,
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
            f"{context.contract.run_id}-bundle-v1",
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
    return PreparedWorkload(work, config_path, bundle, arc_config)


def _owned_resources(
    context: LaunchContext, prepared: PreparedWorkload, ports: Mapping[str, int]
) -> OwnedResources:
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
    compose_environment = {
        **service_environment,
        "RAYLINE_PARITY_ROUTER_IMAGE": context.router_image,
        "RAYLINE_PARITY_ARC_CONFIG": str(prepared.arc_config),
        "RAYLINE_PARITY_ARC_ARTIFACT_DIR": str(context.runtime_dir),
        "RAYLINE_PARITY_ARC_ENVOY_PORT": str(ports["envoy"]),
        "RAYLINE_PARITY_ARC_ROUTER_PORT": str(ports["router"]),
        "RAYLINE_PARITY_ARC_METRICS_PORT": str(ports["metrics"]),
    }
    router_environment = service_environment.copy()
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
    return OwnedResources(
        manager, proxy, service_environment, compose_environment, router_environment
    )


def _execute_arms(
    context: LaunchContext,
    prepared: PreparedWorkload,
    resources: OwnedResources,
    ports: Mapping[str, int],
    paid_started: float,
) -> dict[str, Any]:
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
        environment=resources.service_environment,
        timeout=15 * 60,
        capture=False,
    )
    resources.router_log = (prepared.work / "pathfinder.log").open(
        "w", encoding="utf-8"
    )
    router_url = f"http://127.0.0.1:{ports['pathfinder']}"
    resources.router_process = subprocess.Popen(
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
        env=resources.router_environment,
        text=True,
        stdout=resources.router_log,
        stderr=subprocess.STDOUT,
    )
    _wait_http(f"{router_url}/healthz", 180)
    _compose(
        context.semantic_root,
        resources.compose_environment,
        context.contract.compose_project,
        "up",
        "--build",
        "--detach",
    )
    _wait_http(f"http://127.0.0.1:{ports['router']}/health", 60)
    _wait_arc_ready(f"http://127.0.0.1:{ports['metrics']}/metrics", 180)
    receipts = []
    for arm, base_url in (
        ("modal_inprocess", router_url),
        ("rayline_remote", router_url),
        ("rayline_arc", f"http://127.0.0.1:{ports['envoy']}"),
    ):
        remaining = context.contract.budget.maximum_paid_wall_seconds - (
            time.perf_counter() - paid_started
        )
        if remaining <= 0:
            raise LaunchError(
                f"{context.contract.run_id} paid wall-time ceiling reached"
            )
        receipts.append(
            _probe(
                context.semantic_root,
                context.packet_dir,
                context.output_dir,
                arm,
                base_url,
                remaining,
                context.contract.run_id,
            )
        )
    capture_arc_telemetry(
        f"http://127.0.0.1:{ports['metrics']}/metrics",
        context.output_dir / "rayline_arc_telemetry.json",
    )
    comparison = compare_receipts(receipts)
    (context.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    return comparison


def _cleanup(context: LaunchContext, resources: OwnedResources) -> None:
    _stop_process(resources.router_process)
    resources.cleanup["pathfinder_stopped"] = (
        resources.router_process is None or resources.router_process.poll() is not None
    )
    if resources.router_log is not None:
        resources.router_log.close()
    result = _compose(
        context.semantic_root,
        resources.compose_environment,
        context.contract.compose_project,
        "down",
        "--volumes",
        "--remove-orphans",
        check=False,
    )
    resources.cleanup["compose_removed"] = result.returncode == 0
    try:
        resources.manager.delete(resources.proxy.token_id)
        resources.cleanup["proxy_token_deleted"] = True
    finally:
        _stop_modal_encoder(
            context.pathfinder_python,
            context.pathfinder_root,
            resources.service_environment,
            context.contract.run_id,
        )
    remaining = _modal_containers(
        context.pathfinder_python,
        context.pathfinder_root,
        resources.service_environment,
    )
    resources.cleanup["encoder_containers_remaining"] = len(remaining)


def _run_paid_packet(
    context: LaunchContext, prepared: PreparedWorkload
) -> tuple[dict[str, Any], dict[str, Any], float]:
    ports = {
        "pathfinder": _free_port(),
        "envoy": _free_port(),
        "router": _free_port(),
        "metrics": _free_port(),
    }
    resources = _owned_resources(context, prepared, ports)
    paid_started = time.perf_counter()
    try:
        comparison = _execute_arms(context, prepared, resources, ports, paid_started)
    finally:
        _cleanup(context, resources)
    return comparison, resources.cleanup, time.perf_counter() - paid_started


def _write_manifest(
    context: LaunchContext, cleanup: Mapping[str, Any], elapsed: float
) -> dict[str, Any]:
    manifest = {
        "schema_version": "rayline.vllm.three-arm-run.v1",
        "run_id": context.contract.run_id,
        "source": {
            "semantic_router_commit": context.semantic_head,
            "pathfinder_commit": context.pathfinder_head,
            "engine_build_id": IDENTITY.engine_build_id,
            "plugin_version": IDENTITY.plugin_version,
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
        "cleanup": cleanup,
        "release_qualification_1000_executed": False,
    }
    (context.output_dir / "run-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    context = _preflight(_parse_args())
    with tempfile.TemporaryDirectory(
        prefix=context.contract.temporary_prefix
    ) as temp_name:
        prepared = _prepare_workload(context, Path(temp_name))
        comparison, cleanup, elapsed = _run_paid_packet(context, prepared)
    manifest = _write_manifest(context, cleanup, elapsed)
    print(
        json.dumps(
            {
                "run_id": context.contract.run_id,
                "output_dir": str(context.output_dir),
                "comparison_status": comparison["status"],
                "cleanup": cleanup,
                "budget": manifest["budget"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
