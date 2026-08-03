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
from pathlib import Path
from typing import Any

import modal

from modal_fullstack_inputs import CANDIDATE_PROMPTS
from openrouter_key_management import create_ephemeral_key, delete_ephemeral_key
from openrouter_key_management import ephemeral_key_usage
from openrouter_modal_native_benchmark import finalize_report
from openrouter_modal_native_fixture import (
    CHECKPOINT_REMOTE_PATH,
    DECISION_LOG_REMOTE_PATH,
    build_checkpoint,
    router_config_text,
)

RUN_ID = "rayline-openrouter-modal-native-agt014-20260803"
PREREGISTRATION_COMMIT = "39f683118d07df42a41a683db6291c9119679e6b"
AUTHORIZATION_COMMIT = "5b2e9b9b69150b460999c55321c33ea0c9660dff"
REQUIRED_MODAL_VERSION = "1.5.1"
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
    if _git(root, "rev-parse", f"{remote}/{branch}") != head:
        raise RuntimeError(f"{root.name} source is not pushed")
    return head


def _verify_authority(semantic_root: Path) -> None:
    if len(PREREGISTRATION_COMMIT) != 40 or len(AUTHORIZATION_COMMIT) != 40:
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
        row for row in json.loads(result.stdout) if row.get("description") == APP_NAME
    ]
    if len(matches) != 1 or matches[0].get("state") != "deployed":
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
    *, python: Path, root: Path, environment: dict[str, str]
) -> None:
    apps = _run(
        [str(python), "-m", "modal", "app", "list", "--json"],
        cwd=root,
        environment=environment,
        timeout=60,
    )
    if any(row.get("description") == APP_NAME for row in json.loads(apps.stdout)):
        raise RuntimeError(f"Modal app {APP_NAME} already exists")
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


def _read_decisions(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    args = _args()
    if args.run_id != RUN_ID:
        raise SystemExit(f"launcher only permits {RUN_ID}")
    if modal.__version__ != REQUIRED_MODAL_VERSION:
        raise SystemExit(
            f"Modal SDK {REQUIRED_MODAL_VERSION} required; found {modal.__version__}"
        )
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
    base = os.environ.copy()
    environment = _deploy_environment(base)
    _assert_resources_absent(
        python=pathfinder_python,
        root=pathfinder_root,
        environment=environment,
    )
    paid_started = time.perf_counter()
    ephemeral_key = ""
    key_hash = ""
    router_token = secrets.token_urlsafe(32)
    actual_cost = 0.0
    primary_error: Exception | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="rayline-agt014-") as temporary_name:
            temporary = Path(temporary_name)
            checkpoint_path = temporary / "native-openrouter-agentic.pt"
            checkpoint = build_checkpoint(pathfinder_root, checkpoint_path)
            ephemeral_key, key_hash = create_ephemeral_key(
                management_key, RUN_ID, KEY_LIMIT_USD
            )
            _create_modal_secret(
                python=pathfinder_python,
                root=pathfinder_root,
                environment=environment,
                openrouter_key=ephemeral_key,
                temporary=temporary,
            )
            _run(
                [
                    str(pathfinder_python),
                    "-m",
                    "modal",
                    "volume",
                    "create",
                    ARTIFACT_VOLUME,
                ],
                cwd=pathfinder_root,
                environment=environment,
                timeout=60,
            )
            _run(
                [
                    str(pathfinder_python),
                    "-m",
                    "modal",
                    "volume",
                    "put",
                    ARTIFACT_VOLUME,
                    str(checkpoint_path),
                    CHECKPOINT_REMOTE_PATH,
                ],
                cwd=pathfinder_root,
                environment=environment,
                timeout=180,
            )
            _run(
                [
                    str(pathfinder_python),
                    "-m",
                    "modal",
                    "deploy",
                    str(pathfinder_root / "scripts/modal_router_server.py"),
                ],
                cwd=pathfinder_root,
                environment=environment,
                timeout=15 * 60,
                capture=False,
            )
            app = _modal_app(environment, pathfinder_python, pathfinder_root)
            _register_context(
                token=router_token,
                config_text=router_config_text(),
                run_id=RUN_ID,
                environment=environment,
            )
            ready = _wait_ready(ROUTER_URL, router_token, args.timeout_seconds)
            health = _request_json(
                f"{ROUTER_URL}/healthz", timeout=args.timeout_seconds
            )
            client_path = output_dir / "client.json"
            benchmark_environment = {
                **environment,
                "OPENROUTER_EPHEMERAL_API_KEY": ephemeral_key,
                "RAYLINE_MODAL_NATIVE_ROUTER_TOKEN": router_token,
            }
            _run(
                [
                    str(pathfinder_python),
                    str(BENCHMARK),
                    "--router-url",
                    ROUTER_URL,
                    "--run-id",
                    RUN_ID,
                    "--output",
                    str(client_path),
                    "--timeout-seconds",
                    str(args.timeout_seconds),
                ],
                cwd=semantic_root,
                environment=benchmark_environment,
                timeout=MAXIMUM_PAID_SECONDS,
                capture=False,
            )
            flush = _flush(ROUTER_URL, router_token, args.timeout_seconds)
            decisions_path = output_dir / "native-decisions.jsonl"
            _run(
                [
                    str(pathfinder_python),
                    "-m",
                    "modal",
                    "volume",
                    "get",
                    ARTIFACT_VOLUME,
                    DECISION_LOG_REMOTE_PATH,
                    str(decisions_path),
                    "--force",
                ],
                cwd=pathfinder_root,
                environment=environment,
                timeout=180,
            )
            actual_cost = ephemeral_key_usage(management_key, key_hash)
            deployment = {
                "app_name": APP_NAME,
                "app_id": app["app_id"],
                "router_url": ROUTER_URL,
                "gpu": GPU,
                "max_containers": 1,
                "max_inputs": 8,
                "semantic_router_commit": semantic_head,
                "pathfinder_commit": pathfinder_head,
                "router_server_build": health.get("router_server_build"),
                "serving_runtime_versions": health.get("serving_runtime_versions"),
                "ready": ready,
                "flush": flush,
            }
            report = finalize_report(
                client_report=json.loads(client_path.read_text()),
                decisions=_read_decisions(decisions_path),
                actual_openrouter_cost_usd=actual_cost,
                deployment=deployment,
                checkpoint=checkpoint,
            )
            report["paid_wall_seconds"] = time.perf_counter() - paid_started
            report["maximum_paid_wall_seconds"] = MAXIMUM_PAID_SECONDS
            report["maximum_total_cost_usd"] = MAXIMUM_TOTAL_COST_USD
            report["previous_conservative_usd"] = PREVIOUS_CONSERVATIVE_USD
            report["authorized_cumulative_usd"] = AUTHORIZED_CUMULATIVE_USD
            encoded = json.dumps(report, indent=2, sort_keys=True)
            for protected in (
                management_key,
                ephemeral_key,
                router_token,
                *PUBLIC_MARKERS,
            ):
                if protected and protected in encoded:
                    raise RuntimeError(
                        "protected value entered AGT014 aggregate report"
                    )
            (output_dir / "report.json").write_text(encoded + "\n", encoding="utf-8")
            print(encoded)
    except Exception as error:
        primary_error = error
    finally:
        if key_hash:
            try:
                actual_cost = ephemeral_key_usage(management_key, key_hash)
            finally:
                delete_ephemeral_key(management_key, key_hash)
        _delete_modal_resources(
            python=pathfinder_python,
            root=pathfinder_root,
            environment=environment,
        )
    if time.perf_counter() - paid_started > MAXIMUM_PAID_SECONDS:
        raise RuntimeError("AGT014 exceeded its paid wall-time ceiling")
    if actual_cost > KEY_LIMIT_USD:
        raise RuntimeError("AGT014 OpenRouter key exceeded its hard limit")
    if primary_error is not None:
        raise primary_error


if __name__ == "__main__":
    main()
