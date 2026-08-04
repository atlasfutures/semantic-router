#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Launch and clean up the bounded three-model OpenRouter ARC canary."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import modal
from modal_fullstack_inputs import CANDIDATE_PROMPTS
from openrouter_agentic_preflight_contract import (
    ENVIRONMENT_KEY as TRANSPORT_PREFLIGHT_ENV,
)
from openrouter_agentic_preflight_contract import (
    validate_report,
)
from openrouter_encoder_runtime import (
    assert_encoder_inactive,
    cleanup_encoder,
    deploy_encoder,
    encoder_containers,
    packet_encoder,
    pin_encoder_singleton,
    verify_encoder_deployment,
    wait_protected_encoder,
)
from openrouter_fullstack_packets import packet_catalog
from openrouter_fullstack_state import (
    EncoderDeployment,
    LaunchOutcome,
    RunPacket,
    RuntimeState,
)
from openrouter_fullstack_state import (
    paid_wall_timeout as _paid_wall_timeout,
)
from openrouter_key_management import (
    create_ephemeral_key as _create_ephemeral_key,
)
from openrouter_key_management import (
    delete_ephemeral_key as _delete_ephemeral_key,
)
from openrouter_key_management import (
    ephemeral_key_usage as _ephemeral_key_usage,
)
from openrouter_kv_cache_deployment_evidence import persist as persist_kv_evidence
from openrouter_kv_cache_matched_contract import (
    AGT017_RESOURCE_BUDGET,
    FLASHINFER_APP_NAME,
    FLASHINFER_CLASS_NAME,
    FLASHINFER_ENGINE_BUILD_ID,
    matched_budget_receipt,
)
from openrouter_kv_cache_matched_contract import (
    RUN_ID as AGT017_RUN_ID,
)
from openrouter_launch_authority import verify_source_authority
from openrouter_provider_preflight import run_preflight as run_provider_preflight
from openrouter_provider_preflight_contract import (
    encode_report as encode_provider_preflight,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILE = REPO_ROOT / "deploy/compose/rayline-arc/compose.yaml"
OPENROUTER_ENVOY_FILE = REPO_ROOT / "deploy/compose/rayline-arc/envoy-openrouter.yaml"
ENCODER_APP_ID = "ap-XtsWCBEWdw1ncu9Kv12Chj"
ENCODER_APP_NAME = "rayline-arc-session-encoder"
ENCODER_CLASS_NAME = "SessionEncoder"
ENCODER_HOST = (
    "atlasfutures-dev--rayline-arc-session-encoder-sessionenc-2d82ac.modal.run"
)
ENCODER_BUILD_ID = "vllm@9f5ea81ca0aa570aea46baf82311a1139c1267ca"
ENCODER_DEPLOYMENT_SOURCE_COMMIT = "0e07fa25410adf2ec2fc8e087dd951436c6b6e0d"
ENCODER_PLUGIN_SOURCE_DIGEST = (
    "1ff4ee4d7a22cc1d74c0cdb0352d3f76f5081b7201fa63e7f8f3dd10af246afd"
)
OPENROUTER_KEY_LIMIT_USD = 0.25
GATEWAY_URL = "http://127.0.0.1:18888"
ROUTER_HEALTH_URL = "http://127.0.0.1:18082/health"
METRICS_URL = "http://127.0.0.1:19190/metrics"
ARC_READY_METRIC = 'llm_rayline_arc_component_ready{component="artifact_head_encoder"}'
MAX_STARTUP_SECONDS = 240
MAX_CANARY_SECONDS = 15 * 60
HTTP_OK = 200
REQUIRED_MODAL_VERSION = "1.5.1"
PUBLIC_REQUEST_LOG_MARKERS = (
    *CANDIDATE_PROMPTS,
    "await state.commit",
    "source=public-synthetic",
    "event=retry",
    "public-cache-step=",
)
SESSION_SERVICE_PATH = (
    REPO_ROOT / "src/vllm-plugins/rayline_arc_io/modal_session_service.py"
)

DEFAULT_ENCODER = EncoderDeployment(
    app_name=ENCODER_APP_NAME,
    class_name=ENCODER_CLASS_NAME,
    app_id=ENCODER_APP_ID,
    expected_host=ENCODER_HOST,
    build_id=ENCODER_BUILD_ID,
    deployment_source_commit=ENCODER_DEPLOYMENT_SOURCE_COMMIT,
    plugin_source_digest=ENCODER_PLUGIN_SOURCE_DIGEST,
)
FLASHINFER_ENCODER = EncoderDeployment(
    app_name=FLASHINFER_APP_NAME,
    class_name=FLASHINFER_CLASS_NAME,
    build_id=FLASHINFER_ENGINE_BUILD_ID,
    deployment_source_commit="runtime-attested-launch-source",
    plugin_source_digest="runtime-attested-launch-source",
    deploy_service_path=SESSION_SERVICE_PATH,
    ephemeral=True,
)


PACKETS = packet_catalog(
    REPO_ROOT,
    DEFAULT_ENCODER,
    FLASHINFER_ENCODER,
    canary_key_limit_usd=OPENROUTER_KEY_LIMIT_USD,
    maximum_canary_seconds=MAX_CANARY_SECONDS,
)


def _run(
    command: list[str],
    *,
    environment: dict[str, str],
    check: bool = True,
    capture_output: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        check=check,
        text=True,
        capture_output=capture_output,
        timeout=timeout,
    )


def _compose_command(packet: RunPacket, *arguments: str) -> list[str]:
    return [
        "docker",
        "compose",
        "--project-name",
        packet.project_name,
        "--file",
        str(COMPOSE_FILE),
        "--file",
        str(packet.compose_override),
        *arguments,
    ]


def _wait_http(url: str) -> None:
    deadline = time.monotonic() + MAX_STARTUP_SECONDS
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == HTTP_OK:
                    return
        except OSError:
            pass
        time.sleep(1)
    raise RuntimeError(f"timed out waiting for {url}")


def _arc_component_ready(metrics: str) -> bool | None:
    prefix = f"{ARC_READY_METRIC} "
    for line in metrics.splitlines():
        if line.startswith(prefix):
            return float(line.removeprefix(prefix)) == 1.0
    return None


def _wait_arc_component_ready(url: str) -> None:
    deadline = time.monotonic() + MAX_STARTUP_SECONDS
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status != HTTP_OK:
                    continue
                ready = _arc_component_ready(response.read().decode())
                if ready is True:
                    return
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise RuntimeError("timed out waiting for Rayline ARC component readiness")


def _scan_logs(
    packet: RunPacket,
    environment: dict[str, str],
    protected_values: tuple[str, ...],
) -> None:
    result = _run(
        _compose_command(packet, "logs", "--no-color"),
        environment=environment,
        check=False,
        capture_output=True,
    )
    logs = result.stdout + result.stderr
    for value in protected_values:
        if value and value in logs:
            raise RuntimeError("protected credential appeared in compose logs")


def _collect_post_run_evidence(
    *,
    environment: dict[str, str],
    protected_values: tuple[str, ...],
    management_key: str,
    key_hash: str,
    packet: RunPacket,
    scan_compose_logs: bool = True,
) -> float:
    if scan_compose_logs:
        _scan_logs(packet, environment, protected_values)
    if not key_hash:
        print(
            "OpenRouter ephemeral key was not created",
            file=sys.stderr,
            flush=True,
        )
        return 0.0
    usage = _ephemeral_key_usage(management_key, key_hash)
    print(
        f"OpenRouter ephemeral key usage: ${usage:.8f}",
        file=sys.stderr,
        flush=True,
    )
    if usage > packet.key_limit_usd:
        raise RuntimeError("OpenRouter key usage exceeded its hard limit")
    return usage


def _cleanup_runtime(
    *,
    environment: dict[str, str],
    manager: Any,
    proxy_token: Any,
    management_key: str,
    key_hash: str,
    modal_command: list[str],
    packet: RunPacket,
    state: RuntimeState,
) -> None:
    print("OpenRouter cleanup: starting", file=sys.stderr, flush=True)
    _run(
        _compose_command(packet, "down", "--volumes", "--remove-orphans"),
        environment=environment,
        check=False,
        capture_output=True,
    )
    try:
        if proxy_token is not None:
            manager.delete(proxy_token.token_id)
    finally:
        try:
            if key_hash:
                _delete_ephemeral_key(management_key, key_hash)
        finally:
            if packet.protected_encoder:
                cleanup_encoder(
                    packet,
                    state,
                    modal_command,
                    environment,
                    cwd=REPO_ROOT,
                )
    print(
        "OpenRouter cleanup: compose down, keys deleted, encoder stopped",
        file=sys.stderr,
        flush=True,
    )


def _runtime_environment(
    *,
    openrouter_key: str,
    modal_key: str,
    modal_secret: str,
    packet: RunPacket,
    encoder_base_url: str = "",
    protected_encoder: bool | None = None,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "OPENROUTER_EPHEMERAL_API_KEY": openrouter_key,
            "RAYLINE_ARC_E2E_PROVIDER_KEY": openrouter_key,
            "RAYLINE_ARC_E2E_CONFIG_PATH": str(packet.config),
            "RAYLINE_ARC_E2E_ENVOY_CONFIG_PATH": str(OPENROUTER_ENVOY_FILE),
        }
    )
    use_protected_encoder = (
        packet.protected_encoder if protected_encoder is None else protected_encoder
    )
    if use_protected_encoder:
        encoder = packet_encoder(packet)
        if not encoder_base_url:
            raise RuntimeError("protected encoder base URL is unresolved")
        environment.update(
            {
                "RAYLINE_ARC_E2E_MODAL_KEY": modal_key,
                "RAYLINE_ARC_E2E_MODAL_SECRET": modal_secret,
                "RAYLINE_ARC_E2E_ENCODER_BASE_URL": encoder_base_url,
                "RAYLINE_ARC_E2E_ENCODER_BUILD_ID": encoder.build_id,
            }
        )
    else:
        environment.update(
            {
                "RAYLINE_ARC_E2E_MODAL_KEY": "public-e2e-modal-key",
                "RAYLINE_ARC_E2E_MODAL_SECRET": "public-e2e-modal-secret",
                "RAYLINE_ARC_E2E_ENCODER_BASE_URL": "http://fake-encoder:8080",
                "RAYLINE_ARC_E2E_ENCODER_BUILD_ID": ("vllm@public-rayline-e2e-build"),
            }
        )
    return environment


def _start_compose(packet: RunPacket, environment: dict[str, str]) -> None:
    _run(
        _compose_command(packet, "down", "--volumes", "--remove-orphans"),
        environment=environment,
        check=False,
        capture_output=True,
    )
    _run(
        _compose_command(packet, "up", "--build", "--detach"),
        environment=environment,
        capture_output=True,
    )
    _wait_http(ROUTER_HEALTH_URL)
    _wait_arc_component_ready(METRICS_URL)


def _compose_service_container_id(
    packet: RunPacket, environment: dict[str, str], service: str
) -> str:
    result = _run(
        _compose_command(packet, "ps", "--quiet", service),
        environment=environment,
        capture_output=True,
    )
    container_id = result.stdout.strip()
    if not container_id or "\n" in container_id:
        raise RuntimeError(f"Compose service {service} did not have one container")
    return container_id


def _run_transport_preflight(
    *, args: argparse.Namespace, packet: RunPacket, state: RuntimeState
) -> dict[str, Any]:
    if packet.preflight_driver is None:
        raise RuntimeError("agentic packet omitted its transport preflight driver")
    result = _run(
        [
            sys.executable,
            str(packet.preflight_driver),
            "--gateway-url",
            GATEWAY_URL,
            "--run-id",
            args.run_id,
            "--timeout-seconds",
            str(args.timeout_seconds),
        ],
        environment=state.environment,
        capture_output=True,
        timeout=5 * 60,
    )
    try:
        decoded = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "agentic transport preflight emitted invalid JSON"
        ) from error
    return validate_report(decoded, require_reuse=False)


def _activate_protected_encoder(
    *,
    packet: RunPacket,
    manager: Any,
    state: RuntimeState,
) -> None:
    envoy_container_id = _compose_service_container_id(
        packet, state.environment, "envoy"
    )
    pin_encoder_singleton(packet, state)
    state.proxy_token = manager.create()
    wait_protected_encoder(state, state.proxy_token)
    state.environment = _runtime_environment(
        openrouter_key=state.ephemeral_key,
        modal_key=state.proxy_token.token_id,
        modal_secret=state.proxy_token.token_secret,
        packet=packet,
        encoder_base_url=state.encoder_base_url,
        protected_encoder=True,
    )
    _run(
        _compose_command(
            packet,
            "up",
            "--detach",
            "--no-deps",
            "--force-recreate",
            "router",
        ),
        environment=state.environment,
        capture_output=True,
    )
    _wait_http(ROUTER_HEALTH_URL)
    _wait_arc_component_ready(METRICS_URL)
    if (
        _compose_service_container_id(packet, state.environment, "envoy")
        != envoy_container_id
    ):
        raise RuntimeError("Envoy was replaced across the protected encoder transition")
    if state.transport_preflight is None:
        raise RuntimeError("protected transition lost its transport preflight")
    state.transport_preflight.update(
        {
            "envoy_container_reused": True,
            "ephemeral_key_reused": True,
        }
    )
    state.environment[TRANSPORT_PREFLIGHT_ENV] = json.dumps(
        state.transport_preflight,
        separators=(",", ":"),
        sort_keys=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=sorted(PACKETS), default="canary")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser.parse_args()


def _execute_runtime(
    *,
    args: argparse.Namespace,
    packet: RunPacket,
    management_key: str,
    manager: Any,
    state: RuntimeState,
) -> None:
    if packet.preflight_driver is not None:
        if not state.ephemeral_key:
            state.ephemeral_key, state.key_hash = _create_ephemeral_key(
                management_key, args.run_id, packet.key_limit_usd
            )
        state.environment = _runtime_environment(
            openrouter_key=state.ephemeral_key,
            modal_key="",
            modal_secret="",
            packet=packet,
            protected_encoder=False,
        )
        _start_compose(packet, state.environment)
        state.compose_started = True
        state.transport_preflight = _run_transport_preflight(
            args=args,
            packet=packet,
            state=state,
        )
        if state.transport_preflight["status"] != "passed":
            failed_stage = state.transport_preflight["failed_stage"]
            failed_worker = state.transport_preflight["failed_worker"]
            status = state.transport_preflight["http_status"]
            raise RuntimeError(
                "agentic transport preflight failed: "
                f"stage={failed_stage}; worker={failed_worker}; HTTP {status}"
            )
        _activate_protected_encoder(packet=packet, manager=manager, state=state)
    else:
        if packet.protected_encoder:
            pin_encoder_singleton(packet, state)
            state.proxy_token = manager.create()
            wait_protected_encoder(state, state.proxy_token)
        if not state.ephemeral_key:
            state.ephemeral_key, state.key_hash = _create_ephemeral_key(
                management_key, args.run_id, packet.key_limit_usd
            )
        state.environment = _runtime_environment(
            openrouter_key=state.ephemeral_key,
            modal_key=(
                state.proxy_token.token_id if state.proxy_token is not None else ""
            ),
            modal_secret=(
                state.proxy_token.token_secret if state.proxy_token is not None else ""
            ),
            packet=packet,
            encoder_base_url=state.encoder_base_url,
        )
        _start_compose(packet, state.environment)
        state.compose_started = True
    _run(
        [
            sys.executable,
            str(packet.driver),
            "--gateway-url",
            GATEWAY_URL,
            "--metrics-url",
            METRICS_URL,
            "--run-id",
            args.run_id,
            "--timeout-seconds",
            str(args.timeout_seconds),
        ],
        environment=state.environment,
        timeout=packet.maximum_seconds,
    )


def _initial_context(
    args: argparse.Namespace, packet: RunPacket
) -> tuple[str, list[str], dict[str, str], Any, RuntimeState]:
    if args.mode == "kv-cache-flashinfer":
        if args.run_id != AGT017_RUN_ID:
            raise SystemExit("FlashInfer launcher only permits the AGT017 run ID")
        matched_budget_receipt()
    if packet.protected_encoder and modal.__version__ != REQUIRED_MODAL_VERSION:
        raise SystemExit(
            f"Modal SDK {REQUIRED_MODAL_VERSION} is required; found {modal.__version__}"
        )
    management_key = os.environ.get("OPENROUTER_MANAGEMENT_KEY", "")
    if not management_key:
        raise SystemExit("OPENROUTER_MANAGEMENT_KEY is required")
    modal_command = [sys.executable, "-m", "modal"]
    environment = os.environ.copy()
    if args.mode == "kv-cache-flashinfer":
        environment["MODAL_ENVIRONMENT"] = "dev"
    verify_source_authority(args.mode, environment, repo_root=REPO_ROOT)
    manager = (
        modal.Workspace.from_context().proxy_tokens
        if packet.protected_encoder
        else None
    )
    return (
        management_key,
        modal_command,
        environment,
        manager,
        RuntimeState(environment=environment),
    )


def _prepare_encoder_runtime(
    packet: RunPacket,
    modal_command: list[str],
    environment: dict[str, str],
    state: RuntimeState,
) -> None:
    if not packet.protected_encoder:
        return
    encoder = packet_encoder(packet)
    if encoder.ephemeral:
        assert_encoder_inactive(packet, modal_command, environment, cwd=REPO_ROOT)
        deploy_encoder(packet, modal_command, environment, state, cwd=REPO_ROOT)
    app, state.encoder_base_url, state.encoder_instance = verify_encoder_deployment(
        packet, modal_command, environment, cwd=REPO_ROOT
    )
    state.encoder_app_id = str(app["app_id"])
    if encoder_containers(
        packet,
        modal_command,
        environment,
        cwd=REPO_ROOT,
        app_id=state.encoder_app_id,
    ):
        raise RuntimeError("protected encoder already has a running container")


def _prepare_provider_preflight(
    args: argparse.Namespace,
    packet: RunPacket,
    management_key: str,
    state: RuntimeState,
) -> None:
    if not packet.provider_preflight:
        return
    state.ephemeral_key, state.key_hash = _create_ephemeral_key(
        management_key, args.run_id, packet.key_limit_usd
    )
    state.environment = _runtime_environment(
        openrouter_key=state.ephemeral_key,
        modal_key="",
        modal_secret="",
        packet=packet,
        protected_encoder=False,
    )
    state.provider_preflight = run_provider_preflight(
        openrouter_key=state.ephemeral_key,
        run_id=args.run_id,
        timeout_seconds=args.timeout_seconds,
    )
    encode_provider_preflight(state.provider_preflight, state.ephemeral_key)
    if state.provider_preflight["status"] != "passed":
        worker = state.provider_preflight["failed_worker"]
        status = state.provider_preflight.get("http_status")
        raise RuntimeError(
            "OpenRouter provider availability preflight failed: "
            f"worker={worker}; HTTP {status or 'transport'}"
        )


def _collect_evidence_safely(
    args: argparse.Namespace,
    packet: RunPacket,
    state: RuntimeState,
    management_key: str,
) -> Exception | None:
    try:
        key_usage = _collect_post_run_evidence(
            environment=state.environment,
            protected_values=(
                management_key,
                state.ephemeral_key,
                state.proxy_token.token_id if state.proxy_token is not None else "",
                (
                    state.proxy_token.token_secret
                    if state.proxy_token is not None
                    else ""
                ),
                *PUBLIC_REQUEST_LOG_MARKERS,
            ),
            management_key=management_key,
            key_hash=state.key_hash,
            packet=packet,
            scan_compose_logs=state.compose_started,
        )
        persist_kv_evidence(
            repo_root=REPO_ROOT,
            mode=args.mode,
            run_id=args.run_id,
            usage=key_usage,
            packet=packet,
            state=state,
        )
    except Exception as error:
        print(
            "OpenRouter post-run evidence collection failed",
            file=sys.stderr,
            flush=True,
        )
        return error
    return None


def _cleanup_safely(
    packet: RunPacket,
    state: RuntimeState,
    manager: Any,
    management_key: str,
    modal_command: list[str],
) -> Exception | None:
    try:
        _cleanup_runtime(
            environment=state.environment,
            manager=manager,
            proxy_token=state.proxy_token,
            management_key=management_key,
            key_hash=state.key_hash,
            modal_command=modal_command,
            packet=packet,
            state=state,
        )
    except Exception as error:
        return error
    return None


def _launch_packet(
    args: argparse.Namespace,
    packet: RunPacket,
    management_key: str,
    modal_command: list[str],
    environment: dict[str, str],
    manager: Any,
    state: RuntimeState,
) -> LaunchOutcome:
    outcome = LaunchOutcome()
    paid_started = None
    previous_alarm_handler = None
    try:
        _prepare_provider_preflight(args, packet, management_key, state)
        if packet.encoder is not None and packet.encoder.ephemeral:
            paid_started = time.monotonic()
            previous_alarm_handler = signal.signal(signal.SIGALRM, _paid_wall_timeout)
            signal.setitimer(
                signal.ITIMER_REAL,
                AGT017_RESOURCE_BUDGET.maximum_paid_wall_seconds,
            )
        _prepare_encoder_runtime(packet, modal_command, environment, state)
        _execute_runtime(
            args=args,
            packet=packet,
            management_key=management_key,
            manager=manager,
            state=state,
        )
    except Exception as error:  # preserve primary failure through exact cleanup
        outcome.run_failure = error
    finally:
        if paid_started is not None:
            outcome.paid_elapsed = time.monotonic() - paid_started
        if previous_alarm_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_alarm_handler)
        outcome.evidence_failure = _collect_evidence_safely(
            args, packet, state, management_key
        )
        outcome.cleanup_failure = _cleanup_safely(
            packet, state, manager, management_key, modal_command
        )
    return outcome


def _raise_outcome(outcome: LaunchOutcome, state: RuntimeState) -> None:
    if (
        outcome.paid_elapsed is not None
        and outcome.paid_elapsed > AGT017_RESOURCE_BUDGET.maximum_paid_wall_seconds
        and outcome.run_failure is None
    ):
        outcome.run_failure = RuntimeError(
            "AGT017 remote arm exceeded its paid wall ceiling"
        )
    if outcome.run_failure is not None:
        if (
            state.provider_preflight is not None
            and state.provider_preflight.get("status") == "failed"
        ):
            print(
                json.dumps(state.provider_preflight, indent=2, sort_keys=True),
                flush=True,
            )
        if (
            state.transport_preflight is not None
            and state.transport_preflight.get("status") == "failed"
        ):
            print(
                json.dumps(state.transport_preflight, indent=2, sort_keys=True),
                flush=True,
            )
        if outcome.evidence_failure is not None:
            outcome.run_failure.add_note("post-run evidence collection also failed")
        if outcome.cleanup_failure is not None:
            outcome.run_failure.add_note("exact resource cleanup also failed")
        raise outcome.run_failure
    if outcome.evidence_failure is not None:
        raise RuntimeError("OpenRouter post-run evidence collection failed") from (
            outcome.evidence_failure
        )
    if outcome.cleanup_failure is not None:
        raise RuntimeError("OpenRouter exact resource cleanup failed") from (
            outcome.cleanup_failure
        )


def main() -> None:
    args = _parse_args()
    packet = PACKETS[args.mode]
    management_key, modal_command, environment, manager, state = _initial_context(
        args, packet
    )
    outcome = _launch_packet(
        args,
        packet,
        management_key,
        modal_command,
        environment,
        manager,
        state,
    )
    _raise_outcome(outcome, state)


if __name__ == "__main__":
    main()
