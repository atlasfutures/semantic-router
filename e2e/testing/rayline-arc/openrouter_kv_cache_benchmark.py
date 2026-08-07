#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Bounded retained-versus-replay OpenRouter workload for Rayline encoders."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import openrouter_agentic_benchmark as base
from modal_fullstack_canary import (
    SESSION_METRIC,
    _episode_id,
    _metric_values,
    _read_metrics,
)
from modal_http import connection_for_url as _connection
from modal_http import request_following_result_redirects
from openrouter_agentic_stage_metrics import (
    encoder_client_from_environment,
    encoder_stage_delta,
    read_encoder_snapshot,
    router_snapshot,
    router_stage_delta,
)
from openrouter_agentic_workload import WORKERS
from openrouter_agentic_workload import candidate_case as _candidate_case
from openrouter_fullstack_canary import (
    OpenRouterHTTPError,
    _attempt_count,
    _http_error,
)
from openrouter_kv_cache_journal import append as append_journal
from openrouter_kv_cache_journal import initialize as initialize_journal
from openrouter_kv_cache_matched_contract import MAX_COMPLETION_TOKENS

HTTP_OK = 200
EPISODES = 2
STEPS = 3
MODES = ("retained", "replay")
EXPECTED_REQUESTS = EPISODES * STEPS * len(MODES)
APPEND_LINES = 120


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deployment", choices=("native_modal", "remote_vllm"), required=True
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--metrics-url", default="")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser.parse_args()


def _append_evidence(step: int) -> str:
    return "\n".join(
        f"public-cache-step={step} sample={index:03d} queue={index % 11} "
        f"retained={step * 1000 + index * 19} status=complete"
        for index in range(APPEND_LINES)
    )


def history_states() -> list[dict[str, Any]]:
    base_case = _candidate_case(2)
    messages = copy.deepcopy(base_case["messages"])
    states = [
        {
            **base_case,
            "case_id": "kv-history-0",
            "messages": copy.deepcopy(messages),
        }
    ]
    for step in range(1, STEPS):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": (
                        "Queueing remains the leading hypothesis; I will inspect "
                        "the next bounded public evidence block."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Update the diagnosis for cache step {step}.\n"
                        f"{_append_evidence(step)}"
                    ),
                },
            ]
        )
        states.append(
            {
                **base_case,
                "case_id": f"kv-history-{step}",
                "messages": copy.deepcopy(messages),
            }
        )
    return states


def _payload(deployment: str, case: dict[str, Any]) -> dict[str, Any]:
    # This builds the measured requests for both arms, and deliberately carries
    # no `temperature`. Each arm's router already pins temperature per worker
    # from the manifest — the Go dispatch sets it from the manifest or deletes
    # it, and the native adapter applies the worker spec — so a client value is
    # redundant here. It is also unsuppressable: the native router forwards a
    # request temperature verbatim and treats a manifest `None` as a no-op, so
    # a worker whose model does not advertise the parameter (worker-b's
    # gpt-5.6-luna) would fail 404 "No endpoints found" under
    # `require_parameters`. Workers a and c are unchanged on the wire; their
    # manifests still pin 0.
    return {
        "model": "rayline/router" if deployment == "native_modal" else "auto",
        "messages": case["messages"],
        "tools": case["tools"],
        "tool_choice": "none",
        "max_tokens": MAX_COMPLETION_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def _request_headers(deployment: str, episode_id: str) -> dict[str, str]:
    headers = {
        "content-type": "application/json",
        "x-rayline-episode-id": episode_id,
    }
    if deployment == "native_modal":
        token = os.environ.get("RAYLINE_MODAL_NATIVE_ROUTER_TOKEN", "")
        if not token:
            raise RuntimeError("native Modal router token is missing")
        headers.update(
            {
                "authorization": f"Bearer {token}",
                # Native Pathfinder's current KV key includes session_id but
                # not episode_id when a registered run_id is present.
                "x-rayline-session": episode_id,
            }
        )
    return headers


def _read_stream(connection: Any, response: Any, started: float) -> dict[str, Any]:
    first_token: float | None = None
    response_model = ""
    provider = ""
    usage: dict[str, Any] = {}
    events = 0
    done = False
    try:
        while line := response.readline():
            stripped = line.decode(errors="replace").strip()
            if not stripped.startswith("data:"):
                continue
            data = stripped.removeprefix("data:").strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            events += 1
            response_model = str(event.get("model") or response_model)
            provider = str(event.get("provider") or provider)
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            if first_token is None and base._event_emits_token(event):
                first_token = time.perf_counter() - started
    finally:
        connection.close()
    if not done or not events or first_token is None:
        raise RuntimeError("KV comparison received an incomplete OpenAI stream")
    return {
        "first_token_seconds": first_token,
        "total_seconds": time.perf_counter() - started,
        "response_model": response_model,
        "provider": provider,
        "usage": usage,
        "data_events": events,
    }


def _request_once(
    *,
    deployment: str,
    base_url: str,
    case: dict[str, Any],
    episode_id: str,
    timeout_seconds: float,
    started: float,
) -> dict[str, Any]:
    headers = _request_headers(deployment, episode_id)
    connection, response = request_following_result_redirects(
        connection_factory=_connection,
        method="POST",
        url=f"{base_url.rstrip('/')}/v1/chat/completions",
        body=json.dumps(_payload(deployment, case), separators=(",", ":")).encode(),
        headers=headers,
        timeout_seconds=timeout_seconds,
    )
    worker_header = (
        "x-rayline-worker" if deployment == "native_modal" else "x-vsr-selected-model"
    )
    selected_worker = response.getheader(worker_header, "")
    request_id = response.getheader("x-rayline-request-id", "")
    upstream_millis = response.getheader("x-envoy-upstream-service-time", "")
    attempt_header = response.getheader("x-envoy-attempt-count", "")
    if response.status != HTTP_OK:
        body = response.read()
        retry_after = response.getheader("retry-after")
        connection.close()
        raise _http_error(
            endpoint=f"{deployment} KV comparison",
            status_code=response.status,
            body=body,
            retry_after=retry_after,
            external_attempts=_attempt_count(attempt_header),
        )
    stream = _read_stream(connection, response, started)
    if selected_worker not in WORKERS:
        raise RuntimeError("KV comparison response omitted a known selected worker")
    if deployment == "native_modal" and not request_id:
        raise RuntimeError("native Modal response omitted its decision request ID")
    if not base._response_model_matches(
        stream["response_model"], WORKERS[selected_worker]
    ):
        raise RuntimeError("KV comparison returned the wrong OpenRouter model")
    if (
        deployment == "remote_vllm"
        and stream["provider"] not in base.PROVIDER_NAMES[selected_worker]
    ):
        raise RuntimeError("KV comparison used a provider outside its frozen order")
    usage = stream.pop("usage")
    attempts = _attempt_count(attempt_header)
    return {
        **stream,
        "selected_worker": selected_worker,
        "request_id": request_id,
        "external_attempts": attempts,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "cost_usd": float(usage.get("cost") or 0),
        "envoy_upstream_service_seconds": (
            float(upstream_millis) / 1000 if upstream_millis else None
        ),
    }


def _request(
    *,
    deployment: str,
    base_url: str,
    case: dict[str, Any],
    episode_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = _request_once(
        deployment=deployment,
        base_url=base_url,
        case=case,
        episode_id=episode_id,
        timeout_seconds=timeout_seconds,
        started=started,
    )
    result["client_attempts"] = 1
    return result


def _action_delta(before: str, after: str) -> str:
    left = _metric_values(before, SESSION_METRIC)
    right = _metric_values(after, SESSION_METRIC)
    changed = [
        action
        for action in sorted(set(left) | set(right))
        if math.isclose(right.get(action, 0) - left.get(action, 0), 1.0)
    ]
    if len(changed) != 1:
        raise RuntimeError("remote vLLM request did not emit one session action")
    return changed[0]


def _remote_request(
    *,
    args: argparse.Namespace,
    encoder_client: Any,
    case: dict[str, Any],
    episode_id: str,
) -> dict[str, Any]:
    router_before_text = _read_metrics(args.metrics_url, args.timeout_seconds)
    encoder_before = read_encoder_snapshot(encoder_client)
    result = _request(
        deployment=args.deployment,
        base_url=args.base_url,
        case=case,
        episode_id=episode_id,
        timeout_seconds=args.timeout_seconds,
    )
    router_after_text = _read_metrics(args.metrics_url, args.timeout_seconds)
    encoder_after = read_encoder_snapshot(encoder_client)
    result["session_action"] = _action_delta(router_before_text, router_after_text)
    result["router_stage"] = router_stage_delta(
        before=router_snapshot(router_before_text),
        after=router_snapshot(router_after_text),
        path="arc",
        results=[result],
    )
    result["encoder_stage"] = encoder_stage_delta(
        before=encoder_before,
        after=encoder_after,
        requests=1,
    )
    return result


def _expected_action(deployment: str, mode: str, step: int) -> str | None:
    if deployment != "remote_vllm":
        return None
    return "appended" if mode == "retained" and step > 0 else "created"


def _journal_path(args: argparse.Namespace) -> Path | None:
    raw = getattr(args, "journal", "")
    return Path(raw) if raw else None


def _journal_failure(error: Exception) -> dict[str, Any]:
    report: dict[str, Any] = {"error_class": type(error).__name__}
    if isinstance(error, OpenRouterHTTPError):
        report.update(
            {
                "http_status": error.status_code,
                "error_category": error.error_category,
                "error_type": error.error_type,
                "provider_code": error.provider_code,
                "external_attempts": error.external_attempts,
                "client_attempts": int(getattr(error, "client_attempts", 1)),
                "retry_statuses": list(getattr(error, "retry_statuses", ())),
            }
        )
    return report


def _run_cell(
    *,
    args: argparse.Namespace,
    encoder_client: Any,
    case: dict[str, Any],
    episode_id: str,
    mode: str,
    episode: int,
    step: int,
) -> dict[str, Any]:
    result = (
        _remote_request(
            args=args,
            encoder_client=encoder_client,
            case=case,
            episode_id=episode_id,
        )
        if encoder_client is not None
        else _request(
            deployment=args.deployment,
            base_url=args.base_url,
            case=case,
            episode_id=episode_id,
            timeout_seconds=args.timeout_seconds,
        )
    )
    expected = _expected_action(args.deployment, mode, step)
    if expected is not None and result["session_action"] != expected:
        raise RuntimeError("remote vLLM session action diverged")
    result.update(
        {
            "mode": mode,
            "episode": episode,
            "step": step,
            "case_id": case["case_id"],
        }
    )
    return result


def _run_workload(
    *,
    args: argparse.Namespace,
    encoder_client: Any,
    journal_path: Path | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    states = history_states()
    for episode in range(EPISODES):
        for step, case in enumerate(states):
            mode_order = MODES if (episode + step) % 2 == 0 else tuple(reversed(MODES))
            for mode in mode_order:
                results.append(
                    _run_recorded_cell(
                        args=args,
                        encoder_client=encoder_client,
                        journal_path=journal_path,
                        case=case,
                        mode=mode,
                        episode=episode,
                        step=step,
                        ordinal=len(results) + 1,
                    )
                )
    return results


def _run_recorded_cell(
    *,
    args: argparse.Namespace,
    encoder_client: Any,
    journal_path: Path | None,
    case: dict[str, Any],
    mode: str,
    episode: int,
    step: int,
    ordinal: int,
) -> dict[str, Any]:
    label = f"{args.deployment}:{mode}:episode-{episode}"
    if mode == "replay":
        label += f":step-{step}"
    cell = {
        "ordinal": ordinal,
        "deployment": args.deployment,
        "run_id": args.run_id,
        "mode": mode,
        "episode": episode,
        "step": step,
        "case_id": case["case_id"],
    }
    try:
        result = _run_cell(
            args=args,
            encoder_client=encoder_client,
            case=case,
            episode_id=_episode_id(args.run_id, label),
            mode=mode,
            episode=episode,
            step=step,
        )
    except Exception as error:
        if journal_path is not None:
            append_journal(
                journal_path,
                {**cell, "event": "request_failed", "error": _journal_failure(error)},
            )
        raise
    if journal_path is not None:
        append_journal(
            journal_path,
            {**cell, "event": "request_succeeded", "result": result},
        )
    return result


def _validate_results(results: list[dict[str, Any]]) -> None:
    if len(results) != EXPECTED_REQUESTS:
        raise RuntimeError("KV comparison request envelope diverged")
    for episode in range(EPISODES):
        for step in range(STEPS):
            pair = [
                result
                for result in results
                if result["episode"] == episode and result["step"] == step
            ]
            if (
                len(pair) != len(MODES)
                or len({r["selected_worker"] for r in pair}) != 1
            ):
                raise RuntimeError("retained/replay selection parity failed")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.deployment == "remote_vllm" and not args.metrics_url:
        raise RuntimeError("remote vLLM comparison requires the router metrics URL")
    encoder_client = (
        encoder_client_from_environment(args.timeout_seconds)
        if args.deployment == "remote_vllm"
        else None
    )
    journal_path = _journal_path(args)
    if journal_path is not None:
        initialize_journal(journal_path)
    started = time.perf_counter()
    results = _run_workload(
        args=args,
        encoder_client=encoder_client,
        journal_path=journal_path,
    )
    _validate_results(results)
    return {
        "schema_version": "rayline.openrouter-kv-cache-client.v1",
        "run_id": args.run_id,
        "deployment": args.deployment,
        "status": "client_passed",
        "workload": {
            "episodes": EPISODES,
            "steps": STEPS,
            "modes": list(MODES),
            "requests": EXPECTED_REQUESTS,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "base_case": "agentic-02",
            "append_lines_per_step": APPEND_LINES,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
    }


def main() -> None:
    args = _args()
    report = run(args)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    for name in (
        "OPENROUTER_EPHEMERAL_API_KEY",
        "RAYLINE_MODAL_NATIVE_ROUTER_TOKEN",
        "RAYLINE_ARC_E2E_MODAL_KEY",
        "RAYLINE_ARC_E2E_MODAL_SECRET",
    ):
        value = os.environ.get(name, "")
        if value and value in encoded:
            raise RuntimeError("credential entered KV comparison client report")
    Path(args.output).write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
