#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Deploy, measure, and remove the one-shot AGT017 native Modal router."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any

import modal
import run_openrouter_modal_native as support
from openrouter_key_management import (
    create_ephemeral_key,
    delete_ephemeral_key,
    ephemeral_key_usage,
)
from openrouter_kv_cache_matched_contract import (
    AUTHORIZATION_COMMIT,
    MAX_COMPLETION_TOKENS,
    NATIVE_APP_NAME,
    NATIVE_WEBHOOK_LABEL,
    OPENROUTER_KEY_LIMIT_USD_PER_ARM,
    PATHFINDER_BRANCH,
    PREREGISTRATION_COMMIT,
    RUN_ID,
    SEMANTIC_BRANCH,
    matched_budget_receipt,
)
from openrouter_modal_native_fixture import (
    DECISION_LOG_REMOTE_PATH,
    build_checkpoint,
    router_config_text,
)
from openrouter_provider_preflight import run_preflight as run_provider_preflight
from openrouter_provider_preflight_contract import (
    encode_report as encode_provider_preflight,
)

GIT_SHA_LENGTH = 40
REQUIRED_MODAL_VERSION = "1.5.1"
APP_NAME = NATIVE_APP_NAME
WEBHOOK_LABEL = NATIVE_WEBHOOK_LABEL
ARTIFACT_VOLUME = APP_NAME
CONTEXT_DICT = APP_NAME
REGISTRATION_RECEIPTS_DICT = f"{CONTEXT_DICT}-registration-receipts"
OPENROUTER_SECRET = APP_NAME
ROUTER_URL = f"https://atlasfutures-dev--{WEBHOOK_LABEL}.modal.run"
GPU = "H100"
KEY_LIMIT_USD = OPENROUTER_KEY_LIMIT_USD_PER_ARM
MAXIMUM_PAID_SECONDS = 20 * 60
MAXIMUM_CLEANUP_SETTLE_SECONDS = 60
TRAINING_STAGE = "openrouter_kv_cache_agt017"
BENCHMARK = Path(__file__).with_name("openrouter_kv_cache_benchmark.py")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--pathfinder-root", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser.parse_args()


def _configure_support() -> None:
    support.APP_NAME = APP_NAME
    support.WEBHOOK_LABEL = WEBHOOK_LABEL
    support.ARTIFACT_VOLUME = ARTIFACT_VOLUME
    support.CONTEXT_DICT = CONTEXT_DICT
    support.REGISTRATION_RECEIPTS_DICT = REGISTRATION_RECEIPTS_DICT
    support.OPENROUTER_SECRET = OPENROUTER_SECRET
    support.ROUTER_URL = ROUTER_URL
    support.GPU = GPU


def _verify_authority(semantic_root: Path) -> None:
    for name, commit in (
        ("preregistration", PREREGISTRATION_COMMIT),
        ("authorization", AUTHORIZATION_COMMIT),
    ):
        if len(commit) != GIT_SHA_LENGTH:
            raise RuntimeError(f"AGT017 {name} authority is source-closed")
        support._git(
            semantic_root,
            "merge-base",
            "--is-ancestor",
            commit,
            "HEAD",
        )
    matched_budget_receipt()


def _register_context(
    *, token: str, config_text: str, run_id: str, environment: dict[str, str]
) -> None:
    os.environ["MODAL_ENVIRONMENT"] = environment["MODAL_ENVIRONMENT"]
    context = {
        "schema_version": "rayline-router.modal-native-agt017.v1",
        "router_config_text": config_text,
        "router_config_text_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "decision_log": f"/artifacts/{DECISION_LOG_REMOTE_PATH}",
        "run_id": run_id,
        "attempt_id": "attempt-1",
        "training_stage": TRAINING_STAGE,
        "budget_usd": KEY_LIMIT_USD,
    }
    modal.Dict.from_name(CONTEXT_DICT, create_if_missing=True).put(token, context)


def _prepare(
    args: argparse.Namespace,
) -> support.LaunchContext:
    semantic_root = Path(__file__).resolve().parents[3]
    pathfinder_root = Path(args.pathfinder_root).resolve()
    pathfinder_python = pathfinder_root / ".venv/bin/python"
    _verify_authority(semantic_root)
    semantic_head = support._verify_source(
        semantic_root, SEMANTIC_BRANCH, "atlasfutures"
    )
    pathfinder_head = support._verify_source(
        pathfinder_root, PATHFINDER_BRANCH, "origin"
    )
    output_dir = semantic_root / ".agent-harness/rayline-kv-cache" / RUN_ID
    if output_dir.exists():
        raise RuntimeError("AGT017 output directory already exists")
    output_dir.mkdir(parents=True)
    environment = support._deploy_environment(os.environ.copy())
    support._assert_resources_absent(
        python=pathfinder_python,
        root=pathfinder_root,
        environment=environment,
    )
    context = support.LaunchContext(
        semantic_root=semantic_root,
        pathfinder_root=pathfinder_root,
        pathfinder_python=pathfinder_python,
        output_dir=output_dir,
        environment=environment,
        semantic_head=semantic_head,
        pathfinder_head=pathfinder_head,
        timeout_seconds=args.timeout_seconds,
    )
    return context


def _measure(
    context: support.LaunchContext,
    *,
    ephemeral_key: str,
    router_token: str,
) -> dict[str, Any]:
    config_text = router_config_text(
        training_stage=TRAINING_STAGE,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        app_title="Rayline AGT017",
    )
    _register_context(
        token=router_token,
        config_text=config_text,
        run_id=RUN_ID,
        environment=context.environment,
    )
    ready = support._wait_ready(ROUTER_URL, router_token, context.timeout_seconds)
    health = support._request_json(
        f"{ROUTER_URL}/healthz", timeout=context.timeout_seconds
    )
    client_path = context.output_dir / "native-client.json"
    journal_path = context.output_dir / "native-journal.jsonl"
    benchmark_environment = {
        **context.environment,
        "OPENROUTER_EPHEMERAL_API_KEY": ephemeral_key,
        "RAYLINE_MODAL_NATIVE_ROUTER_TOKEN": router_token,
    }
    decisions_path = context.output_dir / "native-decisions.jsonl"
    benchmark_error: Exception | None = None
    evidence_error: Exception | None = None
    flush: dict[str, Any] | None = None
    try:
        support._run(
            [
                str(context.pathfinder_python),
                str(BENCHMARK),
                "--deployment",
                "native_modal",
                "--base-url",
                ROUTER_URL,
                "--run-id",
                RUN_ID,
                "--output",
                str(client_path),
                "--journal",
                str(journal_path),
                "--timeout-seconds",
                str(context.timeout_seconds),
            ],
            cwd=context.semantic_root,
            environment=benchmark_environment,
            timeout=MAXIMUM_PAID_SECONDS,
            capture=False,
        )
    except Exception as error:
        benchmark_error = error
    try:
        flush = support._flush(ROUTER_URL, router_token, context.timeout_seconds)
        support._run(
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
    except Exception as error:
        evidence_error = error
    if benchmark_error is not None:
        if evidence_error is not None:
            benchmark_error.add_note(
                "native decision evidence recovery also failed after the workload"
            )
        raise benchmark_error
    if evidence_error is not None:
        raise evidence_error
    if flush is None:
        raise RuntimeError("native decision log flush evidence was unavailable")
    return {
        "ready": ready,
        "health": health,
        "flush": flush,
        "client_path": client_path,
        "decisions_path": decisions_path,
    }


def _write_deployment(
    context: support.LaunchContext,
    *,
    app: dict[str, Any],
    measurement: dict[str, Any],
    checkpoint: dict[str, Any],
) -> None:
    health = measurement["health"]
    deployment = {
        "architecture": "native Pathfinder Modal router with local retained KV",
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
        "checkpoint": checkpoint,
        "ready": measurement["ready"],
        "flush": measurement["flush"],
    }
    (context.output_dir / "native-deployment.json").write_text(
        json.dumps(deployment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _execute(
    context: support.LaunchContext,
    *,
    management_key: str,
    ephemeral_key: str,
    key_hash: str,
    router_token: str,
) -> float:
    with tempfile.TemporaryDirectory(prefix="rayline-agt017-") as temporary_name:
        temporary = Path(temporary_name)
        checkpoint_path = temporary / "native-openrouter-kv-cache.pt"
        checkpoint = build_checkpoint(context.pathfinder_root, checkpoint_path)
        app = support._deploy_router(
            context,
            checkpoint_path=checkpoint_path,
            ephemeral_key=ephemeral_key,
            temporary=temporary,
        )
        measurement = _measure(
            context,
            ephemeral_key=ephemeral_key,
            router_token=router_token,
        )
        usage = ephemeral_key_usage(management_key, key_hash)
        (context.output_dir / "native-key-usage.json").write_text(
            json.dumps({"usage_usd": usage}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_deployment(
            context,
            app=app,
            measurement=measurement,
            checkpoint=checkpoint,
        )
        return usage


def _remember_cleanup_error(
    current: Exception | None,
    error: Exception,
    note: str,
) -> Exception:
    if current is None:
        return error
    current.add_note(note)
    return current


def _cleanup(
    context: support.LaunchContext,
    *,
    management_key: str,
    key_hash: str,
    usage: float,
) -> tuple[float, Exception | None]:
    cleanup_error: Exception | None = None
    if key_hash:
        try:
            usage = ephemeral_key_usage(management_key, key_hash)
            (context.output_dir / "native-key-usage.json").write_text(
                json.dumps({"usage_usd": usage}, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except Exception as error:
            cleanup_error = error
        try:
            delete_ephemeral_key(management_key, key_hash)
        except Exception as error:
            cleanup_error = _remember_cleanup_error(
                cleanup_error,
                error,
                "ephemeral OpenRouter key deletion also failed",
            )
    try:
        support._delete_modal_resources(
            python=context.pathfinder_python,
            root=context.pathfinder_root,
            environment=context.environment,
        )
        support._assert_resources_absent(
            python=context.pathfinder_python,
            root=context.pathfinder_root,
            environment=context.environment,
            wait_seconds=MAXIMUM_CLEANUP_SETTLE_SECONDS,
        )
    except Exception as error:
        cleanup_error = _remember_cleanup_error(
            cleanup_error,
            error,
            "native Modal resource cleanup also failed",
        )
    return usage, cleanup_error


def _validate_completion(
    *,
    paid_started: float | None,
    usage: float,
    primary_error: Exception | None,
    cleanup_error: Exception | None,
) -> None:
    if (
        paid_started is not None
        and time.perf_counter() - paid_started > MAXIMUM_PAID_SECONDS
    ):
        raise RuntimeError("AGT017 exceeded its paid wall-time ceiling")
    if usage > KEY_LIMIT_USD:
        raise RuntimeError("AGT017 native OpenRouter key exceeded its hard limit")
    if primary_error is not None:
        if cleanup_error is not None:
            primary_error.add_note("AGT017 cleanup also failed")
        raise primary_error
    if cleanup_error is not None:
        raise cleanup_error


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
    _configure_support()
    context = _prepare(args)
    paid_started: float | None = None
    ephemeral_key = ""
    key_hash = ""
    router_token = secrets.token_urlsafe(32)
    usage = 0.0
    primary_error: Exception | None = None
    try:
        ephemeral_key, key_hash = create_ephemeral_key(
            management_key, RUN_ID, KEY_LIMIT_USD
        )
        provider_preflight = run_provider_preflight(
            openrouter_key=ephemeral_key,
            run_id=RUN_ID,
            timeout_seconds=args.timeout_seconds,
        )
        encode_provider_preflight(provider_preflight, ephemeral_key)
        (context.output_dir / "native-provider-preflight.json").write_text(
            json.dumps(provider_preflight, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if provider_preflight["status"] != "passed":
            worker = provider_preflight["failed_worker"]
            status = provider_preflight.get("http_status")
            raise RuntimeError(
                "OpenRouter provider availability preflight failed: "
                f"worker={worker}; HTTP {status or 'transport'}"
            )
        paid_started = time.perf_counter()
        usage = _execute(
            context,
            management_key=management_key,
            ephemeral_key=ephemeral_key,
            key_hash=key_hash,
            router_token=router_token,
        )
    except Exception as error:
        primary_error = error
    finally:
        usage, cleanup_error = _cleanup(
            context,
            management_key=management_key,
            key_hash=key_hash,
            usage=usage,
        )
    _validate_completion(
        paid_started=paid_started,
        usage=usage,
        primary_error=primary_error,
        cleanup_error=cleanup_error,
    )


if __name__ == "__main__":
    main()
