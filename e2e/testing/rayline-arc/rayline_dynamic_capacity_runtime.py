#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Owned Modal fleet and local control-plane operations for DYN006."""

from __future__ import annotations

import contextlib
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from rayline_concurrency_launcher import LocalCell, SweepContext
from rayline_concurrency_state import (
    ProtectedEncoderClient,
    StateResetError,
    assert_encoder_empty,
)
from rayline_dynamic_arc_config import DYNAMIC_KEY_PREFIX
from rayline_dynamic_stop_contract import (
    DRAINING_MEMBERSHIP_REVISION,
    ENCODER_APP_NAMES,
    ENCODER_REPLICA_IDS,
    MAXIMUM_SCALEDOWN_SECONDS,
    MEMBERSHIP_ADOPTION_SECONDS,
    REGISTERED_MEMBERSHIP_REVISION,
    REMOVED_MEMBERSHIP_REVISION,
    SURVIVOR_COUNT,
    UNAVAILABLE_APP_NAME,
    UNAVAILABLE_REPLICA_ID,
)
from rayline_scaleout_launcher import (
    SERVICE_PATH,
    _deployed_encoder_url,
    _stop_app,
    _warm_replica,
)
from rayline_three_arm_contract import NON_RUNTIME_SECRET_NAMES
from rayline_three_arm_launcher import LaunchError, _run

PARITY_COMPOSE = "deploy/compose/rayline-arc/compose.parity.yaml"
DYNAMIC_COMPOSE = "deploy/compose/rayline-arc/compose.dynamic-parity.yaml"
CONTROLLER_BINARY = "/usr/local/bin/rayline-arc-controller"
REDIS_PASSWORD = "public-rayline-parity-redis-password"
HTTP_OK = 200


@dataclass
class EncoderFleetOwnership:
    manager: Any
    proxy: Any
    service_environment: dict[str, str]
    base_urls: tuple[str, str, str]
    cleanup: dict[str, Any] = field(
        default_factory=lambda: {
            "proxy_token_deleted": False,
            "encoder_apps_stopped": False,
            "encoder_containers_remaining": None,
        }
    )


def dynamic_compose(
    context: SweepContext,
    local: LocalCell,
    *args: str,
    check: bool = True,
    timeout: float = 180,
) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            "docker",
            "compose",
            "--project-name",
            local.compose_project,
            "--file",
            str(context.semantic_root / PARITY_COMPOSE),
            "--file",
            str(context.semantic_root / DYNAMIC_COMPOSE),
            *args,
        ],
        cwd=context.semantic_root,
        environment=local.compose_environment,
        timeout=timeout,
        check=check,
    )


def named_encoder_containers(context: SweepContext) -> list[dict[str, Any]]:
    result = _run(
        [str(context.pathfinder_python), "-m", "modal", "container", "list", "--json"],
        cwd=context.pathfinder_root,
        environment=context.base_environment,
        timeout=30,
    )
    return [
        row
        for row in json.loads(result.stdout)
        if row.get("app_name") in ENCODER_APP_NAMES
    ]


def named_encoder_apps(context: SweepContext) -> list[dict[str, Any]]:
    result = _run(
        [str(context.pathfinder_python), "-m", "modal", "app", "list", "--json"],
        cwd=context.pathfinder_root,
        environment=context.base_environment,
        timeout=30,
    )
    return [
        row
        for row in json.loads(result.stdout)
        if row.get("description") in ENCODER_APP_NAMES
    ]


def start_encoder_fleet(context: SweepContext) -> EncoderFleetOwnership:
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
    ownership = EncoderFleetOwnership(manager, proxy, service_environment, ("", "", ""))
    try:
        urls: list[str] = []
        for app_name in ENCODER_APP_NAMES:
            _run(
                [
                    str(context.pathfinder_python),
                    "-m",
                    "modal",
                    "deploy",
                    str(SERVICE_PATH),
                ],
                cwd=context.pathfinder_root,
                environment={
                    **service_environment,
                    "RAYLINE_ARC_SESSION_APP_NAME": app_name,
                },
                timeout=15 * 60,
                capture=False,
            )
            urls.append(_deployed_encoder_url(context, app_name))
        ownership.base_urls = (urls[0], urls[1], urls[2])
        _wait_encoder_fleet_ready(context, ownership)
    except BaseException:
        try:
            cleanup_encoder_fleet(context, ownership)
        except BaseException as cleanup_error:
            raise LaunchError(
                "dynamic encoder startup and exact cleanup both failed"
            ) from cleanup_error
        raise
    return ownership


def _wait_encoder_fleet_ready(
    context: SweepContext,
    ownership: EncoderFleetOwnership,
) -> None:
    deadline = time.monotonic() + 15 * 60
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            for index, base_url in enumerate(ownership.base_urls):
                _warm_replica(
                    base_url,
                    ownership.proxy.token_id,
                    ownership.proxy.token_secret,
                    f"{context.contract.run_id}:replica-{index}",
                )
                assert_encoder_empty(
                    ProtectedEncoderClient(
                        base_url,
                        ownership.proxy.token_id,
                        ownership.proxy.token_secret,
                        timeout_seconds=30.0,
                    ).request
                )
            return
        except (httpx.HTTPError, StateResetError, LaunchError) as error:
            last_error = error
            time.sleep(1)
    raise LaunchError("dynamic encoder fleet did not become ready") from last_error


def cleanup_encoder_fleet(
    context: SweepContext,
    ownership: EncoderFleetOwnership,
) -> None:
    for app_name in ENCODER_APP_NAMES:
        with contextlib.suppress(BaseException):
            _stop_app(context, ownership.service_environment, app_name)
    try:
        ownership.manager.delete(ownership.proxy.token_id)
        ownership.cleanup["proxy_token_deleted"] = True
    finally:
        deadline = time.monotonic() + MAXIMUM_SCALEDOWN_SECONDS
        while True:
            apps = named_encoder_apps(context)
            ownership.cleanup["encoder_apps_stopped"] = all(
                row.get("state") == "stopped" and str(row.get("tasks")) == "0"
                for row in apps
            )
            ownership.cleanup["encoder_containers_remaining"] = len(
                named_encoder_containers(context)
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
        raise LaunchError("dynamic encoder fleet cleanup did not reach exact zero")


def controller_json(
    context: SweepContext,
    local: LocalCell,
    *args: str,
) -> dict[str, Any]:
    result = dynamic_compose(
        context,
        local,
        "exec",
        "-T",
        "membership-controller",
        CONTROLLER_BINARY,
        *args,
        timeout=30,
    )
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise LaunchError("membership controller returned invalid JSON") from error
    if not isinstance(value, dict):
        raise LaunchError("membership controller returned a non-object")
    return value


def register_third_replica(
    context: SweepContext,
    local: LocalCell,
    fleet: EncoderFleetOwnership,
) -> dict[str, Any]:
    result = controller_json(
        context,
        local,
        "register",
        "--replica-id",
        ENCODER_REPLICA_IDS[2],
        "--base-url",
        fleet.base_urls[2],
    )
    if result != {
        "command": "register",
        "replica_id": ENCODER_REPLICA_IDS[2],
        "revision": REGISTERED_MEMBERSHIP_REVISION,
        "active": 3,
        "draining": 0,
    }:
        raise LaunchError("dynamic capacity registration differs")
    return result


def redis_episode_state(
    context: SweepContext,
    local: LocalCell,
    raw_episode_id: str,
) -> dict[str, Any] | None:
    episode_hash = hashlib.sha256(raw_episode_id.encode()).hexdigest()
    result = dynamic_compose(
        context,
        local,
        "exec",
        "-T",
        "redis",
        "redis-cli",
        "-a",
        REDIS_PASSWORD,
        "--no-auth-warning",
        "GET",
        f"{DYNAMIC_KEY_PREFIX}{episode_hash}:state",
        timeout=15,
    )
    payload = result.stdout.strip()
    if not payload:
        return None
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise LaunchError("dynamic episode state is invalid") from error
    if not isinstance(value, dict):
        raise LaunchError("dynamic episode state is not an object")
    return value


def stop_draining_replica(
    context: SweepContext,
    local: LocalCell,
    fleet: EncoderFleetOwnership,
) -> dict[str, Any]:
    started = time.perf_counter()
    drain = controller_json(
        context,
        local,
        "drain",
        "--replica-id",
        UNAVAILABLE_REPLICA_ID,
    )
    if (
        drain.get("revision") != DRAINING_MEMBERSHIP_REVISION
        or drain.get("active") != SURVIVOR_COUNT
    ):
        raise LaunchError("dynamic drain revision differs")
    time.sleep(MEMBERSHIP_ADOPTION_SECONDS)
    command_started = time.perf_counter()
    command_succeeded = _stop_app(
        context,
        fleet.service_environment,
        UNAVAILABLE_APP_NAME,
    )
    command_seconds = time.perf_counter() - command_started
    if not command_succeeded:
        raise LaunchError("DYN006 exact replica-stop command failed")
    deadline = time.monotonic() + MAXIMUM_SCALEDOWN_SECONDS
    unavailable_stopped = False
    unavailable_containers = -1
    survivors_deployed = False
    survivor_containers = -1
    while time.monotonic() < deadline:
        apps = named_encoder_apps(context)
        containers = named_encoder_containers(context)
        unavailable_rows = [
            row for row in apps if row.get("description") == UNAVAILABLE_APP_NAME
        ]
        survivor_rows = [
            row for row in apps if row.get("description") in ENCODER_APP_NAMES[1:]
        ]
        unavailable_stopped = bool(unavailable_rows) and all(
            row.get("state") == "stopped" and str(row.get("tasks")) == "0"
            for row in unavailable_rows
        )
        unavailable_containers = sum(
            row.get("app_name") == UNAVAILABLE_APP_NAME for row in containers
        )
        survivors_deployed = len(survivor_rows) == SURVIVOR_COUNT and all(
            row.get("state") == "deployed" and str(row.get("tasks")) == "1"
            for row in survivor_rows
        )
        survivor_containers = sum(
            row.get("app_name") in ENCODER_APP_NAMES[1:] for row in containers
        )
        if (
            unavailable_stopped
            and unavailable_containers == 0
            and survivors_deployed
            and survivor_containers == SURVIVOR_COUNT
        ):
            break
        time.sleep(1)
    else:
        raise LaunchError("DYN006 exact replica stop did not converge")
    return {
        "schema_version": "rayline.vllm.dynamic-stop-boundary.v1",
        "action": "drain_then_stop_exact_app",
        "drain_revision": DRAINING_MEMBERSHIP_REVISION,
        "unavailable_replica_id": UNAVAILABLE_REPLICA_ID,
        "unavailable_app_name": UNAVAILABLE_APP_NAME,
        "stop_command_succeeded": command_succeeded,
        "stop_command_seconds": command_seconds,
        "convergence_seconds": time.perf_counter() - started - command_seconds,
        "unavailable_app_stopped": unavailable_stopped,
        "unavailable_containers_remaining": unavailable_containers,
        "survivor_apps_deployed": survivors_deployed,
        "survivor_containers_running": survivor_containers,
    }


def wait_for_drained_removal(
    context: SweepContext,
    local: LocalCell,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = controller_json(context, local, "status")
        if (
            last.get("revision") == REMOVED_MEMBERSHIP_REVISION
            and last.get("active") == SURVIVOR_COUNT
            and last.get("draining") == 0
            and [member.get("id") for member in last.get("members", [])]
            == list(ENCODER_REPLICA_IDS[1:])
        ):
            return last
        time.sleep(2)
    raise LaunchError(f"dynamic drained removal did not converge: {last}")
