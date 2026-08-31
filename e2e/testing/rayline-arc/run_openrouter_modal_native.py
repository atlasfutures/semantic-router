#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Deploy and run the one-shot AGT014 native-Modal comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal
from modal_fullstack_inputs import CANDIDATE_PROMPTS
from openrouter_key_management import (
    create_ephemeral_key,
    delete_ephemeral_key,
    ephemeral_key_usage,
)
from openrouter_modal_native_benchmark import finalize_report, read_decisions
from openrouter_modal_native_fixture import (
    CHECKPOINT_REMOTE_PATH,
    DECISION_LOG_REMOTE_PATH,
    build_checkpoint,
    router_config_text,
)

RUN_ID = "rayline-openrouter-modal-native-agt014-20260803"
PREREGISTRATION_COMMIT = ""
AUTHORIZATION_COMMIT = ""
REQUIRED_MODAL_VERSION = "1.5.1"
GIT_SHA_LENGTH = 40
PATHFINDER_BRANCH = "codex/rayline-vsr-mvp"
SEMANTIC_BRANCH = "codex/rayline-remote-mvp"
APP_NAME = "rayline-router-openrouter-agt014"
WEBHOOK_LABEL = "router-openrouter-agt014"
ARTIFACT_VOLUME = "rayline-router-openrouter-agt014"
CONTEXT_DICT = "rayline-router-openrouter-agt014"
REGISTRATION_RECEIPTS_DICT = f"{CONTEXT_DICT}-registration-receipts"
OPENROUTER_SECRET = "rayline-openrouter-agt014"
ROUTER_URL = f"https://atlasfutures-dev--{WEBHOOK_LABEL}.modal.run"
GPU = "L40S"
KEY_LIMIT_USD = 0.75
MAXIMUM_PAID_SECONDS = 30 * 60
MAXIMUM_MODAL_COST_USD = 5.0
MAXIMUM_TOTAL_COST_USD = KEY_LIMIT_USD + MAXIMUM_MODAL_COST_USD
PREVIOUS_CONSERVATIVE_USD = 82.835731101543
AUTHORIZED_CUMULATIVE_USD = 134.31282402
BENCHMARK = Path(__file__).with_name("openrouter_modal_native_benchmark.py")
PUBLIC_MARKERS = (*CANDIDATE_PROMPTS, "await state.commit", "source=public-synthetic")


@dataclass(frozen=True)
class LaunchContext:
    semantic_root: Path
    pathfinder_root: Path
    pathfinder_python: Path
    output_dir: Path
    environment: dict[str, str]
    semantic_head: str
    pathfinder_head: str
    timeout_seconds: float


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--pathfinder-root", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser.parse_args()


def _run(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    timeout: float | None = None,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=capture,
        timeout=timeout,
        check=check,
    )


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()


def _verify_source(root: Path, branch: str, remote: str) -> str:
    head = _git(root, "rev-parse", "HEAD")
    if _git(root, "branch", "--show-current") != branch:
        raise RuntimeError(f"{root.name} is not on {branch}")
    if _git(root, "status", "--porcelain"):
        raise RuntimeError(f"{root.name} source is not clean")
    pushed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", head, f"{remote}/{branch}"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if pushed.returncode != 0:
        # Remote-visibility means the launched commit is contained in the
        # remote branch; the branch tip may legitimately have moved past the
        # frozen local checkout, and the launched head stays pinned in the
        # deployment evidence.
        raise RuntimeError(f"{root.name} source is not pushed")
    return head


def _verify_authority(semantic_root: Path) -> None:
    if (
        len(PREREGISTRATION_COMMIT) != GIT_SHA_LENGTH
        or len(AUTHORIZATION_COMMIT) != GIT_SHA_LENGTH
    ):
        raise RuntimeError("AGT014 launch authority is source-closed")
    if _git(
        semantic_root, "merge-base", "--is-ancestor", PREREGISTRATION_COMMIT, "HEAD"
    ):
        pass
    if _git(semantic_root, "merge-base", "--is-ancestor", AUTHORIZATION_COMMIT, "HEAD"):
        pass
    if PREVIOUS_CONSERVATIVE_USD + MAXIMUM_TOTAL_COST_USD > AUTHORIZED_CUMULATIVE_USD:
        raise RuntimeError("AGT014 conservative envelope exceeds user authority")


def _request_json(
    url: str,
    *,
    token: str = "",
    method: str = "GET",
    timeout: float = 30,
) -> dict[str, Any]:
    headers = {"accept": "application/json"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        error.read()
        raise RuntimeError(f"{url} returned HTTP {error.code}") from error


def _wait_ready(url: str, token: str, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + 12 * 60
    last_error = ""
    while time.monotonic() < deadline:
        try:
            report = _request_json(
                f"{url}/readyz", token=token, timeout=timeout_seconds
            )
            if report.get("ready") is True:
                return report
            last_error = str(report)
        except (OSError, RuntimeError, json.JSONDecodeError) as error:
            last_error = type(error).__name__
        time.sleep(2)
    raise RuntimeError(f"native Modal router readiness timed out: {last_error[:120]}")


def _modal_app(environment: dict[str, str], python: Path, root: Path) -> dict[str, Any]:
    result = _run(
        [str(python), "-m", "modal", "app", "list", "--json"],
        cwd=root,
        environment=environment,
        timeout=60,
    )
    matches = [
        row
        for row in json.loads(result.stdout)
        if row.get("description") == APP_NAME and row.get("state") == "deployed"
    ]
    if len(matches) != 1:
        # Stopped apps from prior stopped attempts stay in Modal's listing
        # until they age out, so identity requires exactly one DEPLOYED app.
        raise RuntimeError("native Modal app deployment identity is ambiguous")
    return matches[0]


def _listed_names(
    *,
    python: Path,
    root: Path,
    environment: dict[str, str],
    resource: str,
) -> set[str]:
    result = _run(
        [str(python), "-m", "modal", resource, "list", "--json"],
        cwd=root,
        environment=environment,
        timeout=60,
    )
    return {str(row.get("name") or "") for row in json.loads(result.stdout)}


def _assert_resources_absent(
    *,
    python: Path,
    root: Path,
    environment: dict[str, str],
    wait_seconds: float = 0,
) -> None:
    deadline = time.monotonic() + wait_seconds
    while True:
        apps = _run(
            [str(python), "-m", "modal", "app", "list", "--json"],
            cwd=root,
            environment=environment,
            timeout=60,
        )
        active_apps = [
            row
            for row in json.loads(apps.stdout)
            if row.get("description") == APP_NAME
            and (row.get("state") != "stopped" or str(row.get("tasks")) != "0")
        ]
        if not active_apps:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Modal app {APP_NAME} is still active")
        time.sleep(1)
    expected = {
        "secret": {OPENROUTER_SECRET},
        "dict": {CONTEXT_DICT, REGISTRATION_RECEIPTS_DICT},
        "volume": {ARTIFACT_VOLUME},
    }
    for resource, names in expected.items():
        collisions = names & _listed_names(
            python=python,
            root=root,
            environment=environment,
            resource=resource,
        )
        if collisions:
            raise RuntimeError(
                f"Modal {resource} resource already exists: {sorted(collisions)}"
            )


def _deploy_environment(base: dict[str, str]) -> dict[str, str]:
    return {
        **base,
        "MODAL_ENVIRONMENT": "dev",
        "RAYLINE_ROUTER_MODAL_APP_NAME": APP_NAME,
        "RAYLINE_ROUTER_MODAL_WEBHOOK_LABEL": WEBHOOK_LABEL,
        "RAYLINE_ROUTER_MODAL_ARTIFACT_VOLUME": ARTIFACT_VOLUME,
        "RAYLINE_ROUTER_MODAL_CONTEXT_DICT": CONTEXT_DICT,
        "RAYLINE_ROUTER_MODAL_MAX_CONTAINERS": "1",
        "RAYLINE_ROUTER_MODAL_MAX_INPUTS": "8",
        "RAYLINE_ROUTER_MODAL_GPU": GPU,
        "RAYLINE_ROUTER_OPENROUTER_SECRET": OPENROUTER_SECRET,
        "RAYLINE_LOG_COMMIT_INTERVAL_S": "1",
        "RAYLINE_ROUTER_SERVICE_CACHE_MAX_SERVICES": "1",
    }


def _create_modal_secret(
    *,
    python: Path,
    root: Path,
    environment: dict[str, str],
    openrouter_key: str,
    temporary: Path,
) -> None:
    secret_path = temporary / "modal-secret.json"
    secret_path.write_text(
        json.dumps({"OPENROUTER_API_KEY": openrouter_key}), encoding="utf-8"
    )
    secret_path.chmod(0o600)
    try:
        _run(
            [
                str(python),
                "-m",
                "modal",
                "secret",
                "create",
                OPENROUTER_SECRET,
                "--from-json",
                str(secret_path),
                "--force",
            ],
            cwd=root,
            environment=environment,
            timeout=60,
        )
    finally:
        secret_path.unlink(missing_ok=True)


def _register_context(
    *, token: str, config_text: str, run_id: str, environment: dict[str, str]
) -> None:
    os.environ["MODAL_ENVIRONMENT"] = environment["MODAL_ENVIRONMENT"]
    context = {
        "schema_version": "rayline-router.modal-native-agt014.v1",
        "router_config_text": config_text,
        "router_config_text_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "decision_log": f"/artifacts/{DECISION_LOG_REMOTE_PATH}",
        "run_id": run_id,
        "attempt_id": "attempt-1",
        "training_stage": "openrouter_modal_native_agt014",
        "budget_usd": KEY_LIMIT_USD,
    }
    modal.Dict.from_name(CONTEXT_DICT, create_if_missing=True).put(token, context)


def _delete_modal_resources(
    *, python: Path, root: Path, environment: dict[str, str]
) -> None:
    commands = [
        ["app", "stop", APP_NAME, "--yes"],
        ["secret", "delete", OPENROUTER_SECRET, "--yes"],
        ["dict", "delete", CONTEXT_DICT, "--yes"],
        ["dict", "delete", REGISTRATION_RECEIPTS_DICT, "--yes"],
        ["volume", "delete", ARTIFACT_VOLUME, "--yes"],
    ]
    for arguments in commands:
        _run(
            [str(python), "-m", "modal", *arguments],
            cwd=root,
            environment=environment,
            timeout=120,
            check=False,
        )


def _flush(url: str, token: str, timeout_seconds: float) -> dict[str, Any]:
    return _request_json(
        f"{url}/admin/flush",
        token=token,
        method="POST",
        timeout=timeout_seconds,
    )


def _prepare_launch(args: argparse.Namespace) -> LaunchContext:
    semantic_root = Path(__file__).resolve().parents[3]
    pathfinder_root = Path(args.pathfinder_root).resolve()
    pathfinder_python = pathfinder_root / ".venv/bin/python"
    _verify_authority(semantic_root)
    semantic_head = _verify_source(semantic_root, SEMANTIC_BRANCH, "atlasfutures")
    pathfinder_head = _verify_source(pathfinder_root, PATHFINDER_BRANCH, "origin")
    management_key = os.environ.get("OPENROUTER_MANAGEMENT_KEY", "")
    if not management_key:
        raise SystemExit("OPENROUTER_MANAGEMENT_KEY is required")
    output_dir = semantic_root / ".agent-harness/rayline-modal-native" / RUN_ID
    if output_dir.exists():
        raise SystemExit("AGT014 output directory already exists")
    output_dir.mkdir(parents=True)
    environment = _deploy_environment(os.environ.copy())
    _assert_resources_absent(
        python=pathfinder_python,
        root=pathfinder_root,
        environment=environment,
    )
    return LaunchContext(
        semantic_root=semantic_root,
        pathfinder_root=pathfinder_root,
        pathfinder_python=pathfinder_python,
        output_dir=output_dir,
        environment=environment,
        semantic_head=semantic_head,
        pathfinder_head=pathfinder_head,
        timeout_seconds=args.timeout_seconds,
    )


def _deploy_router(
    context: LaunchContext,
    *,
    checkpoint_path: Path,
    ephemeral_key: str,
    temporary: Path,
) -> dict[str, Any]:
    _create_modal_secret(
        python=context.pathfinder_python,
        root=context.pathfinder_root,
        environment=context.environment,
        openrouter_key=ephemeral_key,
        temporary=temporary,
    )
    for operation, timeout in (("create", 60), ("put", 180)):
        arguments = [
            str(context.pathfinder_python),
            "-m",
            "modal",
            "volume",
            operation,
            ARTIFACT_VOLUME,
        ]
        if operation == "put":
            arguments.extend([str(checkpoint_path), CHECKPOINT_REMOTE_PATH])
        _run(
            arguments,
            cwd=context.pathfinder_root,
            environment=context.environment,
            timeout=timeout,
        )
    _run(
        [
            str(context.pathfinder_python),
            "-m",
            "modal",
            "deploy",
            str(context.pathfinder_root / "scripts/modal_router_server.py"),
        ],
        cwd=context.pathfinder_root,
        environment=context.environment,
        timeout=15 * 60,
        capture=False,
    )
    return _modal_app(
        context.environment, context.pathfinder_python, context.pathfinder_root
    )


def _measure_router(
    context: LaunchContext, *, ephemeral_key: str, router_token: str
) -> dict[str, Any]:
    _register_context(
        token=router_token,
        config_text=router_config_text(),
        run_id=RUN_ID,
        environment=context.environment,
    )
    ready = _wait_ready(ROUTER_URL, router_token, context.timeout_seconds)
    health = _request_json(f"{ROUTER_URL}/healthz", timeout=context.timeout_seconds)
    client_path = context.output_dir / "client.json"
    benchmark_environment = {
        **context.environment,
        "OPENROUTER_EPHEMERAL_API_KEY": ephemeral_key,
        "RAYLINE_MODAL_NATIVE_ROUTER_TOKEN": router_token,
    }
    _run(
        [
            str(context.pathfinder_python),
            str(BENCHMARK),
            "--router-url",
            ROUTER_URL,
            "--run-id",
            RUN_ID,
            "--output",
            str(client_path),
            "--timeout-seconds",
            str(context.timeout_seconds),
        ],
        cwd=context.semantic_root,
        environment=benchmark_environment,
        timeout=MAXIMUM_PAID_SECONDS,
        capture=False,
    )
    flush = _flush(ROUTER_URL, router_token, context.timeout_seconds)
    decisions_path = context.output_dir / "native-decisions.jsonl"
    _run(
        [
            str(context.pathfinder_python),
            "-m",
            "modal",
            "volume",
            "get",
            ARTIFACT_VOLUME,
            DECISION_LOG_REMOTE_PATH,
            str(decisions_path),
            "--force",
        ],
        cwd=context.pathfinder_root,
        environment=context.environment,
        timeout=180,
    )
    return {
        "client_path": client_path,
        "decisions_path": decisions_path,
        "ready": ready,
        "health": health,
        "flush": flush,
    }


def _write_report(
    context: LaunchContext,
    *,
    app: dict[str, Any],
    measurement: dict[str, Any],
    checkpoint: dict[str, Any],
    actual_cost: float,
    paid_started: float,
    protected_values: tuple[str, ...],
) -> None:
    health = measurement["health"]
    deployment = {
        "app_name": APP_NAME,
        "app_id": app["app_id"],
        "router_url": ROUTER_URL,
        "gpu": GPU,
        "max_containers": 1,
        "max_inputs": 8,
        "semantic_router_commit": context.semantic_head,
        "pathfinder_commit": context.pathfinder_head,
        "router_server_build": health.get("router_server_build"),
        "serving_runtime_versions": health.get("serving_runtime_versions"),
        "ready": measurement["ready"],
        "flush": measurement["flush"],
    }
    report = finalize_report(
        client_report=json.loads(measurement["client_path"].read_text()),
        decisions=read_decisions(measurement["decisions_path"]),
        actual_openrouter_cost_usd=actual_cost,
        deployment=deployment,
        checkpoint=checkpoint,
    )
    report.update(
        {
            "paid_wall_seconds": time.perf_counter() - paid_started,
            "maximum_paid_wall_seconds": MAXIMUM_PAID_SECONDS,
            "maximum_total_cost_usd": MAXIMUM_TOTAL_COST_USD,
            "previous_conservative_usd": PREVIOUS_CONSERVATIVE_USD,
            "authorized_cumulative_usd": AUTHORIZED_CUMULATIVE_USD,
        }
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if any(
        value and value in encoded for value in (*protected_values, *PUBLIC_MARKERS)
    ):
        raise RuntimeError("protected value entered AGT014 aggregate report")
    (context.output_dir / "report.json").write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


def _execute_packet(
    context: LaunchContext,
    *,
    management_key: str,
    ephemeral_key: str,
    key_hash: str,
    router_token: str,
    paid_started: float,
) -> float:
    with tempfile.TemporaryDirectory(prefix="rayline-agt014-") as temporary_name:
        temporary = Path(temporary_name)
        checkpoint_path = temporary / "native-openrouter-agentic.pt"
        checkpoint = build_checkpoint(context.pathfinder_root, checkpoint_path)
        app = _deploy_router(
            context,
            checkpoint_path=checkpoint_path,
            ephemeral_key=ephemeral_key,
            temporary=temporary,
        )
        measurement = _measure_router(
            context, ephemeral_key=ephemeral_key, router_token=router_token
        )
        actual_cost = ephemeral_key_usage(management_key, key_hash)
        (context.output_dir / "openrouter-key-usage.json").write_text(
            json.dumps({"usage_usd": actual_cost}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_report(
            context,
            app=app,
            measurement=measurement,
            checkpoint=checkpoint,
            actual_cost=actual_cost,
            paid_started=paid_started,
            protected_values=(management_key, ephemeral_key, router_token),
        )
        return actual_cost


def main() -> None:
    args = _args()
    if args.run_id != RUN_ID:
        raise SystemExit(f"launcher only permits {RUN_ID}")
    if modal.__version__ != REQUIRED_MODAL_VERSION:
        raise SystemExit(
            f"Modal SDK {REQUIRED_MODAL_VERSION} required; found {modal.__version__}"
        )
    management_key = os.environ.get("OPENROUTER_MANAGEMENT_KEY", "")
    if not management_key:
        raise SystemExit("OPENROUTER_MANAGEMENT_KEY is required")
    context = _prepare_launch(args)
    paid_started = time.perf_counter()
    ephemeral_key = ""
    key_hash = ""
    router_token = secrets.token_urlsafe(32)
    actual_cost = 0.0
    primary_error: Exception | None = None
    try:
        ephemeral_key, key_hash = create_ephemeral_key(
            management_key, RUN_ID, KEY_LIMIT_USD
        )
        actual_cost = _execute_packet(
            context,
            management_key=management_key,
            ephemeral_key=ephemeral_key,
            key_hash=key_hash,
            router_token=router_token,
            paid_started=paid_started,
        )
    except Exception as error:
        primary_error = error
    finally:
        if key_hash:
            try:
                actual_cost = ephemeral_key_usage(management_key, key_hash)
            finally:
                delete_ephemeral_key(management_key, key_hash)
        _delete_modal_resources(
            python=context.pathfinder_python,
            root=context.pathfinder_root,
            environment=context.environment,
        )
    if time.perf_counter() - paid_started > MAXIMUM_PAID_SECONDS:
        raise RuntimeError("AGT014 exceeded its paid wall-time ceiling")
    if actual_cost > KEY_LIMIT_USD:
        raise RuntimeError("AGT014 OpenRouter key exceeded its hard limit")
    if primary_error is not None:
        raise primary_error


if __name__ == "__main__":
    main()
