#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Bounded three-model ARC canary against pinned OpenRouter endpoints."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from modal_encoder_warmup import warm_encoder_from_environment
from modal_fullstack_canary import (
    _episode_id,
    _failure_total,
    _message_has_output,
    _metric_values,
    _read_metrics,
    _summary,
)
from modal_fullstack_inputs import CANDIDATE_PROMPTS
from modal_http import connection_for_url as _connection
from modal_http import request_following_result_redirects

HTTP_OK = 200
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_TOKENS = 8
MAX_COVERAGE_REQUESTS = 24
DIRECT_REQUESTS_PER_MODEL = 1
ROUTED_REQUESTS_PER_MODEL = 1
STREAM_REQUESTS = 1
MAX_PROVIDER_REQUESTS = (
    MAX_COVERAGE_REQUESTS
    + (DIRECT_REQUESTS_PER_MODEL + ROUTED_REQUESTS_PER_MODEL) * 3
    + STREAM_REQUESTS
)
MAX_RETRIES_PER_REQUEST = 1
MAX_EXTERNAL_ATTEMPTS = MAX_PROVIDER_REQUESTS * (MAX_RETRIES_PER_REQUEST + 1)
REQUEST_PACING_SECONDS = 1.0
DEFAULT_RETRY_DELAY_SECONDS = 2.0
MIN_RETRY_DELAY_SECONDS = 1.0
MAX_RETRY_DELAY_SECONDS = 30.0
RETRIABLE_HTTP_STATUSES = frozenset({429, 503})
MAX_REPORTED_PROVIDER_COST_USD = 0.10
PUBLIC_GATEWAY_AUTHORIZATION = "Bearer public-openrouter-fullstack-canary"
SESSION_METRIC = "llm_rayline_arc_session_actions_total"
WORKERS = {
    "worker-a": "deepseek/deepseek-v4-flash",
    "worker-b": "openai/gpt-5.6-luna",
    "worker-c": "z-ai/glm-5.2",
}
PROVIDER_SLUGS = {
    "worker-a": "fireworks",
    "worker-b": "openai",
    "worker-c": "fireworks",
}
PROVIDER_NAMES = {
    "worker-a": "Fireworks",
    "worker-b": "OpenAI",
    "worker-c": "Fireworks",
}
TEMPERATURES: dict[str, int | None] = {
    "worker-a": 0,
    "worker-b": None,
    "worker-c": 0,
}


class OpenRouterHTTPError(RuntimeError):
    """Privacy-safe pre-response provider error eligible for bounded retry."""

    def __init__(
        self,
        *,
        endpoint: str,
        status_code: int,
        retry_after_seconds: float,
        error_type: str,
        provider_code: str,
    ) -> None:
        details = [f"{endpoint} returned HTTP {status_code}"]
        if error_type:
            details.append(f"error_type={error_type}")
        if provider_code:
            details.append(f"provider_code={provider_code}")
        super().__init__("; ".join(details))
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds
        self.error_type = error_type
        self.provider_code = provider_code

    @property
    def retriable(self) -> bool:
        return self.status_code in RETRIABLE_HTTP_STATUSES


def _bounded_error_token(value: Any) -> str:
    if not isinstance(value, (str, int)):
        return ""
    token = str(value)[:64]
    return token if re.fullmatch(r"[A-Za-z0-9_.:-]+", token) else ""


def _retry_after_seconds(value: str | None) -> float:
    delay = DEFAULT_RETRY_DELAY_SECONDS
    if value:
        try:
            delay = float(value)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                pass
    return min(
        MAX_RETRY_DELAY_SECONDS,
        max(MIN_RETRY_DELAY_SECONDS, delay),
    )


def _http_error(
    *,
    endpoint: str,
    status_code: int,
    body: bytes,
    retry_after: str | None,
) -> OpenRouterHTTPError:
    error_type = ""
    provider_code = ""
    try:
        decoded = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        decoded = None
    error = decoded.get("error") if isinstance(decoded, dict) else None
    metadata = error.get("metadata") if isinstance(error, dict) else None
    if isinstance(metadata, dict):
        error_type = _bounded_error_token(metadata.get("error_type"))
        provider_code = _bounded_error_token(metadata.get("provider_code"))
    return OpenRouterHTTPError(
        endpoint=endpoint,
        status_code=status_code,
        retry_after_seconds=_retry_after_seconds(retry_after),
        error_type=error_type,
        provider_code=provider_code,
    )


def _chat_request(
    *,
    model: str,
    prompt: str,
    direct_openrouter: bool,
    provider_slug: str,
    temperature: int | None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }
    if temperature is not None:
        request["temperature"] = temperature
    if direct_openrouter:
        if not provider_slug:
            raise ValueError("direct OpenRouter calls require a provider slug")
        request.update(
            {
                "provider": {
                    "order": [provider_slug],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                },
                "reasoning": {"enabled": False, "effort": "none"},
            }
        )
    return request


def _chat_once(
    *,
    base_url: str,
    model: str,
    prompt: str,
    authorization: str,
    timeout_seconds: float,
    episode_id: str = "",
    direct_openrouter: bool = False,
    provider_slug: str = "",
    expected_provider_name: str = "",
    temperature: int | None = 0,
) -> dict[str, Any]:
    if direct_openrouter and not expected_provider_name:
        raise ValueError("direct OpenRouter calls require an expected provider")
    request = _chat_request(
        model=model,
        prompt=prompt,
        direct_openrouter=direct_openrouter,
        provider_slug=provider_slug,
        temperature=temperature,
    )
    headers = {
        "authorization": authorization,
        "content-type": "application/json",
    }
    if episode_id:
        headers["x-rayline-episode-id"] = episode_id
    started = time.perf_counter()
    connection, response = request_following_result_redirects(
        connection_factory=_connection,
        method="POST",
        url=f"{base_url.rstrip('/')}/chat/completions",
        body=json.dumps(request, separators=(",", ":")).encode(),
        headers=headers,
        timeout_seconds=timeout_seconds,
    )
    body = response.read()
    latency_seconds = time.perf_counter() - started
    selected_worker = response.getheader("x-vsr-selected-model", "")
    retry_after = response.getheader("retry-after")
    connection.close()
    if response.status != HTTP_OK:
        raise _http_error(
            endpoint="chat endpoint",
            status_code=response.status,
            body=body,
            retry_after=retry_after,
        )
    decoded = json.loads(body)
    choices = decoded.get("choices") if isinstance(decoded, dict) else None
    if (
        not isinstance(choices, list)
        or not choices
        or not isinstance(choices[0], dict)
        or not _message_has_output(choices[0].get("message"))
    ):
        raise RuntimeError("OpenRouter response omitted generated output")
    usage = decoded.get("usage")
    if not isinstance(usage, dict) or not isinstance(usage.get("cost"), (int, float)):
        raise TypeError("OpenRouter response omitted usage cost")
    provider = str(decoded.get("provider") or "")
    expected_provider = expected_provider_name or PROVIDER_NAMES.get(
        selected_worker, ""
    )
    if not expected_provider or provider != expected_provider:
        raise RuntimeError("OpenRouter did not use the pinned provider")
    return {
        "latency_seconds": latency_seconds,
        "selected_worker": selected_worker,
        "response_model": str(decoded.get("model") or ""),
        "provider": provider,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "cost_usd": float(usage["cost"]),
    }


def _chat(
    *,
    base_url: str,
    model: str,
    prompt: str,
    authorization: str,
    timeout_seconds: float,
    episode_id: str = "",
    direct_openrouter: bool = False,
    provider_slug: str = "",
    expected_provider_name: str = "",
    temperature: int | None = 0,
) -> dict[str, Any]:
    attempts = 0
    while True:
        attempts += 1
        try:
            result = _chat_once(
                base_url=base_url,
                model=model,
                prompt=prompt,
                authorization=authorization,
                timeout_seconds=timeout_seconds,
                episode_id=episode_id,
                direct_openrouter=direct_openrouter,
                provider_slug=provider_slug,
                expected_provider_name=expected_provider_name,
                temperature=temperature,
            )
            result["external_attempts"] = attempts
            time.sleep(REQUEST_PACING_SECONDS)
            return result
        except OpenRouterHTTPError as error:
            if not error.retriable or attempts > MAX_RETRIES_PER_REQUEST:
                raise
            print(
                f"OpenRouter transient HTTP {error.status_code}; "
                f"retrying attempt {attempts + 1}/{MAX_RETRIES_PER_REQUEST + 1} "
                f"after {error.retry_after_seconds:g}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(error.retry_after_seconds)


def _validate_model(result: dict[str, Any], expected_model: str) -> None:
    response_model = str(result["response_model"])
    if response_model != expected_model and not response_model.startswith(
        f"{expected_model}-"
    ):
        raise RuntimeError("OpenRouter response model did not match the selected arm")


def _cover_models(
    *, gateway_url: str, run_id: str, timeout_seconds: float
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    selected_prompts: dict[str, str] = {}
    results: list[dict[str, Any]] = []
    for index, prompt in enumerate(CANDIDATE_PROMPTS[:MAX_COVERAGE_REQUESTS]):
        result = _chat(
            base_url=f"{gateway_url.rstrip('/')}/v1",
            model="auto",
            prompt=prompt,
            authorization=PUBLIC_GATEWAY_AUTHORIZATION,
            timeout_seconds=timeout_seconds,
            episode_id=_episode_id(run_id, f"openrouter-coverage-{index}"),
        )
        worker = str(result["selected_worker"])
        if worker not in WORKERS:
            raise RuntimeError("ARC gateway omitted the selected OpenRouter worker")
        _validate_model(result, WORKERS[worker])
        selected_prompts.setdefault(worker, prompt)
        results.append(result)
        print(
            f"OpenRouter coverage: {index + 1} request(s), "
            f"{len(selected_prompts)}/{len(WORKERS)} models",
            file=sys.stderr,
            flush=True,
        )
        if set(selected_prompts) == set(WORKERS):
            break
    if set(selected_prompts) != set(WORKERS):
        raise RuntimeError(
            "candidate prompt set did not exercise all OpenRouter models"
        )
    return selected_prompts, results


def _direct_and_routed(
    *,
    gateway_url: str,
    openrouter_key: str,
    selected_prompts: dict[str, str],
    run_id: str,
    timeout_seconds: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    direct: dict[str, list[dict[str, Any]]] = {}
    routed: dict[str, list[dict[str, Any]]] = {}
    for worker, model in WORKERS.items():
        direct[worker] = [
            _chat(
                base_url=OPENROUTER_BASE_URL,
                model=model,
                prompt=selected_prompts[worker],
                authorization=f"Bearer {openrouter_key}",
                timeout_seconds=timeout_seconds,
                direct_openrouter=True,
                provider_slug=PROVIDER_SLUGS[worker],
                expected_provider_name=PROVIDER_NAMES[worker],
                temperature=TEMPERATURES[worker],
            )
            for _index in range(DIRECT_REQUESTS_PER_MODEL)
        ]
        routed[worker] = [
            _chat(
                base_url=f"{gateway_url.rstrip('/')}/v1",
                model="auto",
                prompt=selected_prompts[worker],
                authorization=PUBLIC_GATEWAY_AUTHORIZATION,
                timeout_seconds=timeout_seconds,
                episode_id=_episode_id(run_id, f"openrouter-routed-{worker}-{index}"),
            )
            for index in range(ROUTED_REQUESTS_PER_MODEL)
        ]
        for result in [*direct[worker], *routed[worker]]:
            _validate_model(result, model)
        if any(result["selected_worker"] != worker for result in routed[worker]):
            raise RuntimeError("ARC selection changed for a frozen OpenRouter prompt")
    return direct, routed


def _stream_once(
    *,
    gateway_url: str,
    prompt: str,
    run_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": "auto",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "stream": True,
        },
        separators=(",", ":"),
    ).encode()
    started = time.perf_counter()
    connection, response = request_following_result_redirects(
        connection_factory=_connection,
        method="POST",
        url=f"{gateway_url.rstrip('/')}/v1/chat/completions",
        body=payload,
        headers={
            "authorization": PUBLIC_GATEWAY_AUTHORIZATION,
            "content-type": "application/json",
            "x-rayline-episode-id": _episode_id(run_id, "openrouter-stream"),
        },
        timeout_seconds=timeout_seconds,
    )
    selected_worker = response.getheader("x-vsr-selected-model", "")
    if response.status != HTTP_OK:
        body = response.read()
        retry_after = response.getheader("retry-after")
        connection.close()
        raise _http_error(
            endpoint="streaming gateway",
            status_code=response.status,
            body=body,
            retry_after=retry_after,
        )
    first_event_seconds: float | None = None
    data_events = 0
    saw_done = False
    usage: dict[str, Any] | None = None
    provider = ""
    response_model = ""
    while line := response.readline():
        stripped = line.decode(errors="replace").strip()
        if not stripped.startswith("data:"):
            continue
        data = stripped.removeprefix("data:").strip()
        if data == "[DONE]":
            saw_done = True
            break
        event = json.loads(data)
        data_events += 1
        if first_event_seconds is None:
            first_event_seconds = time.perf_counter() - started
        if isinstance(event, dict):
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            provider = str(event.get("provider") or provider)
            response_model = str(event.get("model") or response_model)
    total_seconds = time.perf_counter() - started
    connection.close()
    if not saw_done or not data_events or first_event_seconds is None:
        raise RuntimeError("streaming OpenRouter response was incomplete")
    if selected_worker != "worker-a":
        raise RuntimeError("streaming ARC selection changed for the frozen prompt")
    if provider != PROVIDER_NAMES["worker-a"]:
        raise RuntimeError("streaming OpenRouter response used the wrong provider")
    if not isinstance(usage, dict) or not isinstance(usage.get("cost"), (int, float)):
        raise TypeError("streaming OpenRouter response omitted usage cost")
    result = {
        "selected_worker": selected_worker,
        "response_model": response_model,
        "provider": provider,
        "time_to_first_event_seconds": first_event_seconds,
        "total_seconds": total_seconds,
        "data_events": data_events,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "cost_usd": float(usage["cost"]),
    }
    _validate_model(result, WORKERS[selected_worker])
    return result


def _stream(
    *,
    gateway_url: str,
    prompt: str,
    run_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    attempts = 0
    while True:
        attempts += 1
        try:
            result = _stream_once(
                gateway_url=gateway_url,
                prompt=prompt,
                run_id=run_id,
                timeout_seconds=timeout_seconds,
            )
            result["external_attempts"] = attempts
            time.sleep(REQUEST_PACING_SECONDS)
            return result
        except OpenRouterHTTPError as error:
            if not error.retriable or attempts > MAX_RETRIES_PER_REQUEST:
                raise
            print(
                f"OpenRouter transient streaming HTTP {error.status_code}; "
                f"retrying attempt {attempts + 1}/{MAX_RETRIES_PER_REQUEST + 1} "
                f"after {error.retry_after_seconds:g}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(error.retry_after_seconds)


def _result_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    external_attempts = sum(
        int(result.get("external_attempts", 1)) for result in results
    )
    return {
        "requests": len(results),
        "external_attempts": external_attempts,
        "retries": external_attempts - len(results),
        "latency": _summary([float(result["latency_seconds"]) for result in results]),
        "prompt_tokens": sum(int(result["prompt_tokens"]) for result in results),
        "completion_tokens": sum(
            int(result["completion_tokens"]) for result in results
        ),
        "cost_usd": sum(float(result["cost_usd"]) for result in results),
        "response_models": sorted(
            {str(result["response_model"]) for result in results}
        ),
        "providers": sorted({str(result["provider"]) for result in results}),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--metrics-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    openrouter_key = os.environ.get("OPENROUTER_EPHEMERAL_API_KEY", "")
    if not openrouter_key:
        raise SystemExit("OPENROUTER_EPHEMERAL_API_KEY is required")
    print("OpenRouter encoder warmup: starting", file=sys.stderr, flush=True)
    encoder_warmup = warm_encoder_from_environment(
        timeout_seconds=args.timeout_seconds,
        connection_factory=_connection,
    )
    selected_prompts, coverage = _cover_models(
        gateway_url=args.gateway_url,
        run_id=args.run_id,
        timeout_seconds=args.timeout_seconds,
    )
    direct, routed = _direct_and_routed(
        gateway_url=args.gateway_url,
        openrouter_key=openrouter_key,
        selected_prompts=selected_prompts,
        run_id=args.run_id,
        timeout_seconds=args.timeout_seconds,
    )
    stream = _stream(
        gateway_url=args.gateway_url,
        prompt=selected_prompts["worker-a"],
        run_id=args.run_id,
        timeout_seconds=args.timeout_seconds,
    )
    all_nonstream = [
        *coverage,
        *(result for results in direct.values() for result in results),
        *(result for results in routed.values() for result in results),
    ]
    provider_cost = sum(float(result["cost_usd"]) for result in all_nonstream) + float(
        stream["cost_usd"]
    )
    if provider_cost > MAX_REPORTED_PROVIDER_COST_USD:
        raise RuntimeError("OpenRouter reported cost exceeded the canary gate")
    actual_provider_requests = len(all_nonstream) + STREAM_REQUESTS
    if actual_provider_requests > MAX_PROVIDER_REQUESTS:
        raise RuntimeError("OpenRouter request count exceeded the canary bound")
    actual_external_attempts = sum(
        int(result.get("external_attempts", 1)) for result in all_nonstream
    ) + int(stream.get("external_attempts", 1))
    if actual_external_attempts > MAX_EXTERNAL_ATTEMPTS:
        raise RuntimeError(
            "OpenRouter external attempt count exceeded the canary bound"
        )

    metrics = _read_metrics(args.metrics_url, args.timeout_seconds)
    session_actions = _metric_values(metrics, SESSION_METRIC)
    expected_creates = (
        len(coverage) + len(WORKERS) * ROUTED_REQUESTS_PER_MODEL + STREAM_REQUESTS
    )
    if session_actions.get("created", 0) < expected_creates:
        raise RuntimeError("router metrics omitted OpenRouter session creates")
    if _failure_total(metrics) != 0:
        raise RuntimeError("router recorded an ARC selection failure")

    report = {
        "schema_version": "rayline.arc.openrouter-fullstack-canary.v1",
        "run_id": args.run_id,
        "status": "passed",
        "models": WORKERS,
        "pinned_providers": PROVIDER_NAMES,
        "encoder_warmup": encoder_warmup,
        "coverage": {
            **_result_summary(coverage),
            "selection_counts": {
                worker: sum(result["selected_worker"] == worker for result in coverage)
                for worker in WORKERS
            },
        },
        "direct": {
            worker: _result_summary(results) for worker, results in direct.items()
        },
        "routed": {
            worker: _result_summary(results) for worker, results in routed.items()
        },
        "stream": stream,
        "actual_provider_requests": actual_provider_requests,
        "maximum_provider_requests": MAX_PROVIDER_REQUESTS,
        "actual_external_attempts": actual_external_attempts,
        "maximum_external_attempts": MAX_EXTERNAL_ATTEMPTS,
        "provider_retries": actual_external_attempts - actual_provider_requests,
        "reported_provider_cost_usd": provider_cost,
        "maximum_reported_provider_cost_usd": MAX_REPORTED_PROVIDER_COST_USD,
        "session_actions_created": session_actions["created"],
        "selection_failures": 0,
        "automatic_prefix_cache_enabled": False,
        "release_qualification_1000_executed": False,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if openrouter_key in encoded:
        raise RuntimeError("OpenRouter credential entered the report")
    print(encoded)


if __name__ == "__main__":
    main()
