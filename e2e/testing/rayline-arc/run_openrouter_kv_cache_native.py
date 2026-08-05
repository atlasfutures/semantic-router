#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Deploy, measure, and remove a one-shot native Modal KV router."""

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
from openrouter_kv_cache_agt019_contract import (
    ARTIFACT_REVISION as AGT019_ARTIFACT_REVISION,
)
from openrouter_kv_cache_agt019_contract import (
    AUTHORIZATION_COMMIT as AGT019_AUTHORIZATION_COMMIT,
)
from openrouter_kv_cache_agt019_contract import (
    MAX_COMPLETION_TOKENS as AGT019_MAX_COMPLETION_TOKENS,
)
from openrouter_kv_cache_agt019_contract import (
    NATIVE_APP_NAME as AGT019_NATIVE_APP_NAME,
)
from openrouter_kv_cache_agt019_contract import (
    NATIVE_WEBHOOK_LABEL as AGT019_NATIVE_WEBHOOK_LABEL,
)
from openrouter_kv_cache_agt019_contract import (
    PATHFINDER_BRANCH as AGT019_PATHFINDER_BRANCH,
)
from openrouter_kv_cache_agt019_contract import (
    PREREGISTRATION_COMMIT as AGT019_PREREGISTRATION_COMMIT,
)
from openrouter_kv_cache_agt019_contract import RUN_ID as AGT019_RUN_ID
from openrouter_kv_cache_agt019_contract import (
    SEMANTIC_BRANCH as AGT019_SEMANTIC_BRANCH,
)
from openrouter_kv_cache_agt019_contract import (
    SOURCE_CLOSED_KEY_LIMIT_USD_PER_ARM as AGT019_KEY_LIMIT_USD,
)
from openrouter_kv_cache_agt019_contract import (
    SOURCE_CLOSED_MAXIMUM_PAID_WALL_SECONDS as AGT019_MAXIMUM_PAID_SECONDS,
)
from openrouter_kv_cache_agt019_contract import (
    agt019_budget_receipt,
)
from openrouter_kv_cache_matched_contract import (
    ARTIFACT_REVISION,
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
from openrouter_kv_cache_successor_contract import (
    ARTIFACT_REVISION as AGT018_ARTIFACT_REVISION,
)
from openrouter_kv_cache_successor_contract import (
    AUTHORIZATION_COMMIT as AGT018_AUTHORIZATION_COMMIT,
)
from openrouter_kv_cache_successor_contract import (
    MAX_COMPLETION_TOKENS as AGT018_MAX_COMPLETION_TOKENS,
)
from openrouter_kv_cache_successor_contract import (
    NATIVE_APP_NAME as AGT018_NATIVE_APP_NAME,
)
from openrouter_kv_cache_successor_contract import (
    NATIVE_WEBHOOK_LABEL as AGT018_NATIVE_WEBHOOK_LABEL,
)
from openrouter_kv_cache_successor_contract import (
    PATHFINDER_BRANCH as AGT018_PATHFINDER_BRANCH,
)
from openrouter_kv_cache_successor_contract import (
    PREREGISTRATION_COMMIT as AGT018_PREREGISTRATION_COMMIT,
)
from openrouter_kv_cache_successor_contract import RUN_ID as AGT018_RUN_ID
from openrouter_kv_cache_successor_contract import (
    SEMANTIC_BRANCH as AGT018_SEMANTIC_BRANCH,
)
from openrouter_kv_cache_successor_contract import (
    SOURCE_CLOSED_KEY_LIMIT_USD_PER_ARM as AGT018_KEY_LIMIT_USD,
)
from openrouter_kv_cache_successor_contract import (
    SOURCE_CLOSED_MAXIMUM_PAID_WALL_SECONDS as AGT018_MAXIMUM_PAID_SECONDS,
)
from openrouter_kv_cache_successor_contract import (
    successor_budget_receipt,
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
APP_TITLE = "Rayline AGT017"
CONTEXT_SCHEMA_VERSION = "rayline-router.modal-native-agt017.v1"
RUN_LABEL = "AGT017"
AGT017_APP_NAME = APP_NAME
AGT017_WEBHOOK_LABEL = WEBHOOK_LABEL
AGT017_RUN_ID = RUN_ID
AGT017_PREREGISTRATION_COMMIT = PREREGISTRATION_COMMIT
AGT017_AUTHORIZATION_COMMIT = AUTHORIZATION_COMMIT
AGT017_MAX_COMPLETION_TOKENS = MAX_COMPLETION_TOKENS
AGT017_PATHFINDER_BRANCH = PATHFINDER_BRANCH
AGT017_SEMANTIC_BRANCH = SEMANTIC_BRANCH
AGT017_ARTIFACT_REVISION = ARTIFACT_REVISION
AGT017_KEY_LIMIT_USD = KEY_LIMIT_USD
AGT017_MAXIMUM_PAID_SECONDS = MAXIMUM_PAID_SECONDS


GENERATIONS: dict[str, dict[str, Any]] = {
    "agt017": {
        "app_name": AGT017_APP_NAME,
        "webhook_label": AGT017_WEBHOOK_LABEL,
        "key_limit_usd": AGT017_KEY_LIMIT_USD,
        "maximum_paid_seconds": AGT017_MAXIMUM_PAID_SECONDS,
        "training_stage": "openrouter_kv_cache_agt017",
        "benchmark": "openrouter_kv_cache_benchmark.py",
        "app_title": "Rayline AGT017",
        "context_schema_version": "rayline-router.modal-native-agt017.v1",
        "run_label": "AGT017",
        "run_id": AGT017_RUN_ID,
        "preregistration_commit": AGT017_PREREGISTRATION_COMMIT,
        "authorization_commit": AGT017_AUTHORIZATION_COMMIT,
        "max_completion_tokens": AGT017_MAX_COMPLETION_TOKENS,
        "pathfinder_branch": AGT017_PATHFINDER_BRANCH,
        "semantic_branch": AGT017_SEMANTIC_BRANCH,
        "artifact_revision": AGT017_ARTIFACT_REVISION,
    },
    "agt018": {
        "app_name": AGT018_NATIVE_APP_NAME,
        "webhook_label": AGT018_NATIVE_WEBHOOK_LABEL,
        "key_limit_usd": AGT018_KEY_LIMIT_USD,
        "maximum_paid_seconds": AGT018_MAXIMUM_PAID_SECONDS,
        "training_stage": "openrouter_kv_cache_agt018",
        "benchmark": "openrouter_kv_cache_successor_benchmark.py",
        "app_title": "Rayline AGT018",
        "context_schema_version": "rayline-router.modal-native-agt018.v1",
        "run_label": "AGT018",
        "run_id": AGT018_RUN_ID,
        "preregistration_commit": AGT018_PREREGISTRATION_COMMIT,
        "authorization_commit": AGT018_AUTHORIZATION_COMMIT,
        "max_completion_tokens": AGT018_MAX_COMPLETION_TOKENS,
        "pathfinder_branch": AGT018_PATHFINDER_BRANCH,
        "semantic_branch": AGT018_SEMANTIC_BRANCH,
        "artifact_revision": AGT018_ARTIFACT_REVISION,
    },
    "agt019": {
        "app_name": AGT019_NATIVE_APP_NAME,
        "webhook_label": AGT019_NATIVE_WEBHOOK_LABEL,
        "key_limit_usd": AGT019_KEY_LIMIT_USD,
        "maximum_paid_seconds": AGT019_MAXIMUM_PAID_SECONDS,
        "training_stage": "openrouter_kv_cache_agt019",
        # AGT019 reuses the AGT018 workload verbatim; only comparability moved.
        "benchmark": "openrouter_kv_cache_successor_benchmark.py",
        "app_title": "Rayline AGT019",
        "context_schema_version": "rayline-router.modal-native-agt019.v1",
        "run_label": "AGT019",
        "run_id": AGT019_RUN_ID,
        "preregistration_commit": AGT019_PREREGISTRATION_COMMIT,
        "authorization_commit": AGT019_AUTHORIZATION_COMMIT,
        "max_completion_tokens": AGT019_MAX_COMPLETION_TOKENS,
        "pathfinder_branch": AGT019_PATHFINDER_BRANCH,
        "semantic_branch": AGT019_SEMANTIC_BRANCH,
        "artifact_revision": AGT019_ARTIFACT_REVISION,
    },
}


def _configure_generation(generation: str) -> None:
    identity = GENERATIONS.get(generation)
    if identity is None:
        raise ValueError(f"unknown native KV generation {generation!r}")
    global APP_NAME
    global WEBHOOK_LABEL
    global ARTIFACT_VOLUME
    global CONTEXT_DICT
    global REGISTRATION_RECEIPTS_DICT
    global OPENROUTER_SECRET
    global ROUTER_URL
    global KEY_LIMIT_USD
    global MAXIMUM_PAID_SECONDS
    global TRAINING_STAGE
    global BENCHMARK
    global APP_TITLE
    global CONTEXT_SCHEMA_VERSION
    global RUN_LABEL
    global RUN_ID
    global PREREGISTRATION_COMMIT
    global AUTHORIZATION_COMMIT
    global MAX_COMPLETION_TOKENS
    global PATHFINDER_BRANCH
    global SEMANTIC_BRANCH
    global ARTIFACT_REVISION

    APP_NAME = identity["app_name"]
    WEBHOOK_LABEL = identity["webhook_label"]
    ARTIFACT_VOLUME = APP_NAME
    CONTEXT_DICT = APP_NAME
    REGISTRATION_RECEIPTS_DICT = f"{CONTEXT_DICT}-registration-receipts"
    OPENROUTER_SECRET = APP_NAME
    ROUTER_URL = f"https://atlasfutures-dev--{WEBHOOK_LABEL}.modal.run"
    KEY_LIMIT_USD = identity["key_limit_usd"]
    MAXIMUM_PAID_SECONDS = identity["maximum_paid_seconds"]
    TRAINING_STAGE = identity["training_stage"]
    BENCHMARK = Path(__file__).with_name(identity["benchmark"])
    APP_TITLE = identity["app_title"]
    CONTEXT_SCHEMA_VERSION = identity["context_schema_version"]
    RUN_LABEL = identity["run_label"]
    RUN_ID = identity["run_id"]
    PREREGISTRATION_COMMIT = identity["preregistration_commit"]
    AUTHORIZATION_COMMIT = identity["authorization_commit"]
    MAX_COMPLETION_TOKENS = identity["max_completion_tokens"]
    PATHFINDER_BRANCH = identity["pathfinder_branch"]
    SEMANTIC_BRANCH = identity["semantic_branch"]
    ARTIFACT_REVISION = identity["artifact_revision"]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generation",
        choices=("agt017", "agt018", "agt019"),
        default="agt017",
    )
    parser.add_argument("--run-id", default="")
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
            raise RuntimeError(f"{RUN_LABEL} {name} authority is source-closed")
        support._git(
            semantic_root,
            "merge-base",
            "--is-ancestor",
            commit,
            "HEAD",
        )
    if RUN_LABEL == "AGT017":
        matched_budget_receipt()
    elif KEY_LIMIT_USD <= 0 or MAXIMUM_PAID_SECONDS <= 0:
        raise RuntimeError(f"{RUN_LABEL} budget authority is source-closed")
    elif RUN_LABEL == "AGT018":
        successor_budget_receipt()
    elif RUN_LABEL == "AGT019":
        agt019_budget_receipt()
    else:
        raise RuntimeError(f"{RUN_LABEL} budget authority is unbound")


def _register_context(
    *, token: str, config_text: str, run_id: str, environment: dict[str, str]
) -> None:
    os.environ["MODAL_ENVIRONMENT"] = environment["MODAL_ENVIRONMENT"]
    context = {
        "schema_version": CONTEXT_SCHEMA_VERSION,
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
    semantic_head = support._verify_source(
        semantic_root, SEMANTIC_BRANCH, "atlasfutures"
    )
    pathfinder_head = support._verify_source(
        pathfinder_root, PATHFINDER_BRANCH, "origin"
    )
    output_dir = semantic_root / ".agent-harness/rayline-kv-cache" / RUN_ID
    if output_dir.exists():
        raise RuntimeError(f"{RUN_LABEL} output directory already exists")
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
        app_title=APP_TITLE,
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
    with tempfile.TemporaryDirectory(
        prefix=f"rayline-{RUN_LABEL.lower()}-"
    ) as temporary_name:
        temporary = Path(temporary_name)
        checkpoint_path = temporary / "native-openrouter-kv-cache.pt"
        checkpoint = build_checkpoint(
            context.pathfinder_root,
            checkpoint_path,
            artifact_id=ARTIFACT_REVISION,
        )
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
        raise RuntimeError(f"{RUN_LABEL} exceeded its paid wall-time ceiling")
    if usage > KEY_LIMIT_USD:
        raise RuntimeError(f"{RUN_LABEL} native OpenRouter key exceeded its hard limit")
    if primary_error is not None:
        if cleanup_error is not None:
            primary_error.add_note(f"{RUN_LABEL} cleanup also failed")
        raise primary_error
    if cleanup_error is not None:
        raise cleanup_error


def main() -> None:
    args = _args()
    _configure_generation(args.generation)
    requested_run_id = args.run_id or RUN_ID
    if requested_run_id != RUN_ID:
        raise SystemExit(f"launcher only permits {RUN_ID}")
    args.run_id = requested_run_id
    semantic_root = Path(__file__).resolve().parents[3]
    _verify_authority(semantic_root)
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
