#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Deploy, measure, and exactly clean up the one-shot PERF028 GDN A/B."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import importlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modal_session_canary import CanaryClient
from modal_vllm_profile_comparator import run_comparison
from modal_vllm_profile_contract import (
    AUTHORIZATION_COMMIT,
    MODAL_ENVIRONMENT,
    PERF028_BUDGET,
    PREREGISTRATION_COMMIT,
    PROFILE_LABELS,
    PROFILES,
    REQUIRED_MODAL_VERSION,
    RUN_ID,
    SEMANTIC_BRANCH,
    SEMANTIC_REMOTE_REF,
)
from rayline_three_arm_budget import budget_receipt
from rayline_three_arm_contract import NON_RUNTIME_SECRET_NAMES

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_PATH = REPO_ROOT / "src/vllm-plugins/rayline_arc_io/modal_session_service.py"
GIT_SHA_LENGTH = 40
READINESS_DEADLINE_SECONDS = 12 * 60
CLEANUP_DEADLINE_SECONDS = 5 * 60


def _paid_wall_timeout(_signal_number: int, _frame: Any) -> None:
    raise TimeoutError("PERF028 reached its maximum paid wall time")


@dataclass(frozen=True)
class LaunchContext:
    modal: Any
    modal_python: Path
    environment: dict[str, str]
    output_dir: Path
    semantic_head: str
    timeout_seconds: float


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--pathfinder-root", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser.parse_args()


def _run(
    command: list[str],
    *,
    environment: dict[str, str],
    timeout: float | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=check,
    )


def _git(*arguments: str, check: bool = True) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=check,
    ).stdout.strip()


def _verify_source_authority() -> str:
    for label, commit in (
        ("preregistration", PREREGISTRATION_COMMIT),
        ("authorization", AUTHORIZATION_COMMIT),
    ):
        if len(commit) != GIT_SHA_LENGTH:
            raise RuntimeError(f"PERF028 {label} authority is source-closed")
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"PERF028 {label} commit is not in launch source")
    if _git("branch", "--show-current") != SEMANTIC_BRANCH:
        raise RuntimeError("PERF028 semantic branch identity differs")
    if _git("status", "--porcelain"):
        raise RuntimeError("PERF028 requires a clean semantic source checkpoint")
    head = _git("rev-parse", "HEAD")
    if _git("rev-parse", SEMANTIC_REMOTE_REF) != head:
        raise RuntimeError("PERF028 semantic source is not remote-visible")
    return head


def _modal_rows(
    modal_python: Path,
    environment: dict[str, str],
    resource: str,
) -> list[dict[str, Any]]:
    result = _run(
        [str(modal_python), "-m", "modal", resource, "list", "--json"],
        environment=environment,
        timeout=60,
    )
    rows = json.loads(result.stdout)
    if not isinstance(rows, list):
        raise TypeError(f"Modal {resource} inventory is not a list")
    return rows


def _named_apps(
    modal_python: Path, environment: dict[str, str]
) -> list[dict[str, Any]]:
    names = {profile.app_name for profile in PROFILES.values()}
    return [
        row
        for row in _modal_rows(modal_python, environment, "app")
        if row.get("description") in names
    ]


def _named_containers(
    modal_python: Path, environment: dict[str, str]
) -> list[dict[str, Any]]:
    names = {profile.app_name for profile in PROFILES.values()}
    return [
        row
        for row in _modal_rows(modal_python, environment, "container")
        if row.get("app_name") in names
    ]


def _assert_no_active_resources(
    modal_python: Path, environment: dict[str, str]
) -> None:
    active = [
        row
        for row in _named_apps(modal_python, environment)
        if row.get("state") != "stopped" or str(row.get("tasks")) != "0"
    ]
    if active or _named_containers(modal_python, environment):
        raise RuntimeError("a PERF028 Modal resource is already active")


def _deploy(
    modal_python: Path,
    environment: dict[str, str],
    app_name: str,
) -> None:
    deploy_environment = {
        **environment,
        "RAYLINE_ARC_SESSION_APP_NAME": app_name,
    }
    result = _run(
        [str(modal_python), "-m", "modal", "deploy", str(SERVICE_PATH)],
        environment=deploy_environment,
        timeout=15 * 60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"PERF028 deployment failed for {app_name}")


def _wait_ready(client: CanaryClient) -> None:
    deadline = time.monotonic() + READINESS_DEADLINE_SECONDS
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            health, _elapsed = client.request("GET", "/health")
            if (
                health.get("status") == "ok"
                and health.get("resident_sessions") == 0
                and health.get("resident_tokens") == 0
            ):
                return
        except (OSError, RuntimeError, json.JSONDecodeError) as error:
            last_error = error
        time.sleep(2)
    raise RuntimeError("PERF028 protected encoder readiness timed out") from last_error


def _stop_apps(modal_python: Path, environment: dict[str, str]) -> None:
    for profile in PROFILES.values():
        with contextlib.suppress(BaseException):
            _run(
                [
                    str(modal_python),
                    "-m",
                    "modal",
                    "app",
                    "stop",
                    "-y",
                    profile.app_name,
                ],
                environment=environment,
                timeout=120,
                check=False,
            )


def _wait_cleanup(modal_python: Path, environment: dict[str, str]) -> None:
    deadline = time.monotonic() + CLEANUP_DEADLINE_SECONDS
    while True:
        apps = _named_apps(modal_python, environment)
        apps_stopped = all(
            row.get("state") == "stopped" and str(row.get("tasks")) == "0"
            for row in apps
        )
        if apps_stopped and not _named_containers(modal_python, environment):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("PERF028 exact-name cleanup did not reach zero")
        time.sleep(1)


def _prepare(args: argparse.Namespace) -> LaunchContext:
    if args.run_id != RUN_ID:
        raise RuntimeError("PERF028 only permits its preregistered run ID")
    budget_receipt(PERF028_BUDGET)
    semantic_head = _verify_source_authority()
    pathfinder_root = args.pathfinder_root.resolve()
    modal_python = pathfinder_root / ".venv/bin/python"
    if not modal_python.is_file():
        raise RuntimeError("Pathfinder .venv Python is required for Modal 1.5.1")
    if Path(sys.executable).resolve() != modal_python.resolve():
        raise RuntimeError("run PERF028 with the Pathfinder .venv Python")
    modal = importlib.import_module("modal")
    if modal.__version__ != REQUIRED_MODAL_VERSION:
        raise RuntimeError(
            f"Modal SDK {REQUIRED_MODAL_VERSION} is required; found {modal.__version__}"
        )

    output_dir = REPO_ROOT / ".agent-harness/rayline-vllm-profile" / RUN_ID
    if output_dir.exists():
        raise RuntimeError("PERF028 output directory already exists")
    environment = {**os.environ, "MODAL_ENVIRONMENT": MODAL_ENVIRONMENT}
    for name in NON_RUNTIME_SECRET_NAMES:
        environment.pop(name, None)
    _assert_no_active_resources(modal_python, environment)
    output_dir.mkdir(parents=True)
    return LaunchContext(
        modal=modal,
        modal_python=modal_python,
        environment=environment,
        output_dir=output_dir,
        semantic_head=semantic_head,
        timeout_seconds=args.timeout_seconds,
    )


def _profile_clients(context: LaunchContext, proxy: Any) -> dict[str, CanaryClient]:
    clients: dict[str, CanaryClient] = {}
    for label in PROFILE_LABELS:
        profile = PROFILES[label]
        cls = context.modal.Cls.from_name(
            profile.app_name,
            "SessionEncoder",
            environment_name=MODAL_ENVIRONMENT,
        )
        url = cls().web.get_web_url()
        if not url:
            raise RuntimeError(f"PERF028 {label} web URL is unavailable")
        clients[label] = CanaryClient(
            base_url=url.rstrip("/"),
            modal_key=proxy.token_id,
            modal_secret=proxy.token_secret,
            timeout_seconds=context.timeout_seconds,
            expected_engine_build_id=profile.engine_build_id,
        )
    return clients


def _hydrate(clients: dict[str, CanaryClient]) -> None:
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(PROFILE_LABELS))
    try:
        futures = [
            executor.submit(_wait_ready, clients[label]) for label in PROFILE_LABELS
        ]
        for future in futures:
            future.result()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _write_receipt(
    context: LaunchContext,
    *,
    result: dict[str, Any] | None,
    failure_type: str | None,
    cleanup: dict[str, Any],
    elapsed: float,
) -> None:
    cleanup_passed = (
        cleanup["proxy_token_deleted"] is True
        and cleanup["apps_stopped"] is True
        and cleanup["containers_remaining"] == 0
    )
    receipt = {
        "schema_version": "rayline.arc.modal-vllm-profile-launch.v1",
        "run_id": RUN_ID,
        "status": "passed" if result is not None and cleanup_passed else "failed",
        "failure_type": failure_type,
        "semantic_head": context.semantic_head,
        "comparison": result,
        "budget": budget_receipt(PERF028_BUDGET, elapsed),
        "cleanup": cleanup,
        "provider_calls": 0,
        "release_qualification_1000_executed": False,
    }
    (context.output_dir / "report.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )


def _finalize(
    context: LaunchContext,
    *,
    manager: Any,
    proxy: Any,
    started: float,
    result: dict[str, Any] | None,
    failure_type: str | None,
    cleanup: dict[str, Any],
) -> None:
    _stop_apps(context.modal_python, context.environment)
    try:
        manager.delete(proxy.token_id)
        cleanup["proxy_token_deleted"] = True
    finally:
        try:
            _wait_cleanup(context.modal_python, context.environment)
            cleanup["apps_stopped"] = True
            cleanup["containers_remaining"] = 0
        finally:
            _write_receipt(
                context,
                result=result,
                failure_type=failure_type,
                cleanup=cleanup,
                elapsed=time.monotonic() - started,
            )


def _launch(context: LaunchContext) -> dict[str, Any]:
    manager = context.modal.Workspace.from_context().proxy_tokens
    proxy = manager.create()

    started = time.monotonic()
    previous_alarm_handler = signal.signal(signal.SIGALRM, _paid_wall_timeout)
    signal.setitimer(signal.ITIMER_REAL, PERF028_BUDGET.maximum_paid_wall_seconds)
    result: dict[str, Any] | None = None
    failure_type: str | None = None
    cleanup = {
        "proxy_token_deleted": False,
        "apps_stopped": False,
        "containers_remaining": None,
    }
    try:
        for label in PROFILE_LABELS:
            print(f"PERF028 deploy {label}: starting", file=sys.stderr, flush=True)
            _deploy(
                context.modal_python,
                context.environment,
                PROFILES[label].app_name,
            )
        clients = _profile_clients(context, proxy)
        _hydrate(clients)
        result = run_comparison(clients, RUN_ID)
    except BaseException as error:
        failure_type = type(error).__name__
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm_handler)
        _finalize(
            context,
            manager=manager,
            proxy=proxy,
            started=started,
            result=result,
            failure_type=failure_type,
            cleanup=cleanup,
        )
    if result is None:
        raise RuntimeError("PERF028 completed without a comparison result")
    return result


def main() -> None:
    result = _launch(_prepare(_args()))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
