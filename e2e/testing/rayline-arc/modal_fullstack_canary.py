#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Bounded real-encoder -> Semantic Router -> real-vLLM worker canary."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
import os
import re
import sys
import time
from typing import Any
from urllib.parse import urlparse

from modal_http import request_following_result_redirects

HTTP_OK = 200
WORKERS = {
    "worker-a": "synthetic/provider-a",
    "worker-b": "synthetic/provider-b",
}
SESSION_METRIC = "llm_rayline_arc_session_actions_total"
FAILURE_METRIC = "llm_rayline_arc_selection_failures_total"
CONCURRENT_REQUESTS = 4
DIRECT_SAMPLES_PER_WORKER = 3
MAX_CANDIDATE_REQUESTS = 24
CANDIDATE_PROMPTS = (
    "Reply with one word: amber.",
    "Return only the number four.",
    "Write a tiny Python addition function.",
    "Name one planet in the solar system.",
    "Say hello in French.",
    "Summarize why rain falls in one sentence.",
    "Respond with valid JSON containing one boolean.",
    "Translate good morning into German.",
    "Give a two-word title for a calm ocean painting.",
    "State whether ten is larger than nine.",
    "Write one cheerful adjective.",
    "Write one serious adjective.",
    "Explain a database index in five words.",
    "Name a common Linux command.",
    "Complete this sequence: 2, 4, 6,",
    "Give one example of a mammal.",
    "Answer yes or no: is ice cold?",
    "Describe a red square very briefly.",
    "Write the lowercase alphabet's first letter.",
    "Return an empty JSON object.",
    "Name one musical instrument.",
    "What is one plus one?",
    "Use one word to describe a mountain.",
    "Reply with the word done.",
)


def _connection(base_url: str, timeout_seconds: float) -> tuple[Any, str]:
    parsed = urlparse(base_url)
    if not parsed.hostname:
        raise ValueError("URL omitted hostname")
    connection_type = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = connection_type(
        parsed.hostname,
        parsed.port,
        timeout=timeout_seconds,
    )
    return connection, parsed.path.rstrip("/")


def _episode_id(run_id: str, label: str) -> str:
    digest = hashlib.sha256(f"{run_id}:{label}".encode()).hexdigest()[:32]
    return f"real-workers-{digest}"


def _message_has_output(message: object) -> bool:
    if not isinstance(message, dict):
        return False
    return any(
        isinstance(message.get(key), str) and bool(message[key])
        for key in ("content", "reasoning_content")
    )


def _nonstream_chat(
    *,
    base_url: str,
    model: str,
    prompt: str,
    authorization: str,
    timeout_seconds: float,
    episode_id: str = "",
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8,
            "temperature": 0,
            "stream": False,
        },
        separators=(",", ":"),
    ).encode()
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
        url=f"{base_url.rstrip('/')}/v1/chat/completions",
        body=payload,
        headers=headers,
        timeout_seconds=timeout_seconds,
    )
    body = response.read()
    elapsed = time.perf_counter() - started
    selected_worker = response.getheader("x-vsr-selected-model", "")
    connection.close()
    if response.status != HTTP_OK:
        raise RuntimeError(f"chat endpoint returned HTTP {response.status}")
    decoded = json.loads(body)
    choices = decoded.get("choices") if isinstance(decoded, dict) else None
    if (
        not isinstance(choices, list)
        or not choices
        or not isinstance(choices[0], dict)
        or not _message_has_output(choices[0].get("message"))
    ):
        raise RuntimeError("chat endpoint omitted generated output")
    usage = decoded.get("usage") or {}
    return {
        "latency_seconds": elapsed,
        "response_model": decoded.get("model", ""),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "selected_worker": selected_worker,
    }


def _stream_chat(
    *,
    base_url: str,
    prompt: str,
    timeout_seconds: float,
    episode_id: str,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": "auto",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8,
            "temperature": 0,
            "stream": True,
        },
        separators=(",", ":"),
    ).encode()
    started = time.perf_counter()
    connection, response = request_following_result_redirects(
        connection_factory=_connection,
        method="POST",
        url=f"{base_url.rstrip('/')}/v1/chat/completions",
        body=payload,
        headers={
            "authorization": "Bearer public-modal-fullstack-canary",
            "content-type": "application/json",
            "x-rayline-episode-id": episode_id,
        },
        timeout_seconds=timeout_seconds,
    )
    selected_worker = response.getheader("x-vsr-selected-model", "")
    if response.status != HTTP_OK:
        response.read()
        connection.close()
        raise RuntimeError(f"streaming gateway returned HTTP {response.status}")
    first_event_seconds: float | None = None
    data_events = 0
    saw_done = False
    while line := response.readline():
        stripped = line.decode(errors="replace").strip()
        if not stripped.startswith("data:"):
            continue
        data = stripped.removeprefix("data:").strip()
        if data == "[DONE]":
            saw_done = True
            break
        json.loads(data)
        data_events += 1
        if first_event_seconds is None:
            first_event_seconds = time.perf_counter() - started
    elapsed = time.perf_counter() - started
    connection.close()
    if not saw_done or not data_events or first_event_seconds is None:
        raise RuntimeError("streaming gateway response was incomplete")
    if selected_worker not in WORKERS:
        raise RuntimeError("streaming gateway omitted selected worker")
    return {
        "selected_worker": selected_worker,
        "time_to_first_event_seconds": first_event_seconds,
        "total_seconds": elapsed,
        "data_events": data_events,
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "p50_seconds": _percentile(values, 0.50),
        "p95_seconds": _percentile(values, 0.95),
        "max_seconds": max(values),
    }


def _metric_values(metrics: str, name: str) -> dict[str, float]:
    values: dict[str, float] = {}
    pattern = re.compile(
        rf'^{re.escape(name)}\{{[^}}]*action="([^"]+)"[^}}]*\}}\s+([0-9.eE+-]+)$'
    )
    for line in metrics.splitlines():
        if match := pattern.match(line):
            values[match.group(1)] = float(match.group(2))
    return values


def _failure_total(metrics: str) -> float:
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in metrics.splitlines()
        if line.startswith(FAILURE_METRIC + "{")
    )


def _read_metrics(url: str, timeout_seconds: float) -> str:
    connection, prefix = _connection(url, timeout_seconds)
    connection.request("GET", prefix or "/")
    response = connection.getresponse()
    body = response.read().decode()
    connection.close()
    if response.status != HTTP_OK:
        raise RuntimeError(f"metrics endpoint returned HTTP {response.status}")
    return body


def _direct_sample(
    *,
    worker_url: str,
    model: str,
    authorization: str,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    _nonstream_chat(
        base_url=worker_url,
        model=model,
        prompt="Reply with the word ready.",
        authorization=authorization,
        timeout_seconds=timeout_seconds,
    )
    return [
        _nonstream_chat(
            base_url=worker_url,
            model=model,
            prompt="Reply with one short word.",
            authorization=authorization,
            timeout_seconds=timeout_seconds,
        )
        for _index in range(DIRECT_SAMPLES_PER_WORKER)
    ]


def _direct_baselines(
    *,
    worker_urls: dict[str, str],
    authorization: str,
    timeout_seconds: float,
) -> dict[str, list[dict[str, Any]]]:
    print("real-worker direct baselines: starting", file=sys.stderr, flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(WORKERS)) as executor:
        direct_futures = {
            worker: executor.submit(
                _direct_sample,
                worker_url=worker_urls[worker],
                model=model,
                authorization=authorization,
                timeout_seconds=timeout_seconds,
            )
            for worker, model in WORKERS.items()
        }
        direct = {worker: future.result() for worker, future in direct_futures.items()}
    print("real-worker direct baselines: complete", file=sys.stderr, flush=True)
    return direct


def _cover_workers(
    *, gateway_url: str, run_id: str, timeout_seconds: float
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    selected_prompts: dict[str, str] = {}
    gateway_results: list[dict[str, Any]] = []
    for index, prompt in enumerate(CANDIDATE_PROMPTS[:MAX_CANDIDATE_REQUESTS]):
        result = _nonstream_chat(
            base_url=gateway_url,
            model="auto",
            prompt=prompt,
            authorization="Bearer public-modal-fullstack-canary",
            timeout_seconds=timeout_seconds,
            episode_id=_episode_id(run_id, f"coverage-{index}"),
        )
        selected_worker = result["selected_worker"]
        if selected_worker not in WORKERS:
            raise RuntimeError("gateway omitted selected worker")
        if result["response_model"] != WORKERS[selected_worker]:
            raise RuntimeError("gateway response model did not match selected worker")
        selected_prompts.setdefault(selected_worker, prompt)
        gateway_results.append(result)
        print(
            f"gateway coverage: {index + 1} request(s), "
            f"{len(selected_prompts)}/2 workers",
            file=sys.stderr,
            flush=True,
        )
        if set(selected_prompts) == set(WORKERS):
            break
    if set(selected_prompts) != set(WORKERS):
        raise RuntimeError("candidate prompt set did not exercise both real workers")
    return selected_prompts, gateway_results


def _concurrent_gateway_requests(
    *,
    gateway_url: str,
    run_id: str,
    timeout_seconds: float,
    selected_prompts: dict[str, str],
) -> tuple[list[dict[str, Any]], float]:
    concurrent_prompts = [
        selected_prompts["worker-a"],
        selected_prompts["worker-b"],
        selected_prompts["worker-a"],
        selected_prompts["worker-b"],
    ]
    print("real-worker concurrency: starting", file=sys.stderr, flush=True)
    concurrent_started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=CONCURRENT_REQUESTS
    ) as executor:
        futures = [
            executor.submit(
                _nonstream_chat,
                base_url=gateway_url,
                model="auto",
                prompt=prompt,
                authorization="Bearer public-modal-fullstack-canary",
                timeout_seconds=timeout_seconds,
                episode_id=_episode_id(run_id, f"concurrent-{index}"),
            )
            for index, prompt in enumerate(concurrent_prompts)
        ]
        concurrent_results = [future.result() for future in futures]
    concurrent_wall = time.perf_counter() - concurrent_started
    concurrent_workers = {result["selected_worker"] for result in concurrent_results}
    if concurrent_workers != set(WORKERS):
        raise RuntimeError("concurrent gateway requests did not reach both workers")
    return concurrent_results, concurrent_wall


def _validate_metrics(
    *, metrics_url: str, timeout_seconds: float, expected_creates: int
) -> dict[str, float]:
    metrics = _read_metrics(metrics_url, timeout_seconds)
    session_actions = _metric_values(metrics, SESSION_METRIC)
    if session_actions.get("created", 0.0) < expected_creates:
        raise RuntimeError("router metrics omitted real-worker session creates")
    if _failure_total(metrics) != 0:
        raise RuntimeError("router recorded a Rayline ARC selection failure")
    return session_actions


def _build_report(
    *,
    run_id: str,
    direct: dict[str, list[dict[str, Any]]],
    gateway_results: list[dict[str, Any]],
    concurrent_results: list[dict[str, Any]],
    concurrent_wall: float,
    stream: dict[str, Any],
    session_actions: dict[str, float],
) -> dict[str, Any]:
    direct_summary = {
        worker: {
            "latency": _summary(
                [float(result["latency_seconds"]) for result in results]
            ),
            "completion_tokens": sum(
                int(result["completion_tokens"]) for result in results
            ),
        }
        for worker, results in direct.items()
    }
    all_gateway = [*gateway_results, *concurrent_results]
    return {
        "schema_version": "rayline.arc.modal-fullstack-canary.v1",
        "run_id": run_id,
        "status": "passed",
        "real_workers": sorted(WORKERS),
        "direct": direct_summary,
        "gateway": {
            "coverage_requests": len(gateway_results),
            "selection_counts": {
                worker: sum(
                    result["selected_worker"] == worker for result in all_gateway
                )
                for worker in WORKERS
            },
            "latency": _summary(
                [float(result["latency_seconds"]) for result in all_gateway]
            ),
            "completion_tokens": sum(
                int(result["completion_tokens"]) for result in all_gateway
            ),
        },
        "concurrency": {
            "requests": CONCURRENT_REQUESTS,
            "wall_seconds": concurrent_wall,
            "requests_per_second": CONCURRENT_REQUESTS / concurrent_wall,
            "wall_to_sum_latency_ratio": concurrent_wall
            / sum(float(result["latency_seconds"]) for result in concurrent_results),
        },
        "stream": stream,
        "session_actions_created": session_actions["created"],
        "selection_failures": 0,
        "provider_calls": 0,
        "automatic_prefix_cache_enabled": False,
        "release_qualification_1000_executed": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--metrics-url", required=True)
    parser.add_argument("--worker-a-url", required=True)
    parser.add_argument("--worker-b-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    worker_api_key = os.environ.get("RAYLINE_ARC_WORKER_API_KEY", "")
    if not worker_api_key:
        raise SystemExit("RAYLINE_ARC_WORKER_API_KEY is required")
    direct = _direct_baselines(
        worker_urls={"worker-a": args.worker_a_url, "worker-b": args.worker_b_url},
        authorization=f"Bearer {worker_api_key}",
        timeout_seconds=args.timeout_seconds,
    )
    selected_prompts, gateway_results = _cover_workers(
        gateway_url=args.gateway_url,
        run_id=args.run_id,
        timeout_seconds=args.timeout_seconds,
    )
    concurrent_results, concurrent_wall = _concurrent_gateway_requests(
        gateway_url=args.gateway_url,
        run_id=args.run_id,
        timeout_seconds=args.timeout_seconds,
        selected_prompts=selected_prompts,
    )
    print("real-worker streaming: starting", file=sys.stderr, flush=True)
    stream = _stream_chat(
        base_url=args.gateway_url,
        prompt=selected_prompts["worker-a"],
        timeout_seconds=args.timeout_seconds,
        episode_id=_episode_id(args.run_id, "stream"),
    )
    session_actions = _validate_metrics(
        metrics_url=args.metrics_url,
        timeout_seconds=args.timeout_seconds,
        expected_creates=len(gateway_results) + len(concurrent_results) + 1,
    )
    report = _build_report(
        run_id=args.run_id,
        direct=direct,
        gateway_results=gateway_results,
        concurrent_results=concurrent_results,
        concurrent_wall=concurrent_wall,
        stream=stream,
        session_actions=session_actions,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
