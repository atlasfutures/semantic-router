#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Drive one frozen Rayline parity arm and emit an aggregate-only receipt.

The three protocols deliberately exercise each architecture's real decision
boundary. Modal uses its decision-only route API, Remote uses the Pathfinder
prepare/commit/settle transaction, and ARC uses the normal Semantic Router gateway
with a zero-cost worker double. Raw cases and endpoint credentials never enter
the receipt.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import hmac
import http.client
import json
import math
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from rayline_parity_comparator import ARMS, INPUT_SCHEMA, validate_receipt

CORPUS_SCHEMA = "rayline.vllm.three-arm-corpus.v1"
WORKLOAD_SCHEMA = "rayline.vllm.three-arm-workload.v1"
TOPOLOGY_SCHEMA = "rayline.vllm.three-arm-topology.v1"
PROTOCOL_BY_ARM = {
    "modal_inprocess": "modal_route",
    "rayline_remote": "pathfinder_transaction",
    "rayline_arc": "openai_gateway",
}
TRANSACTION_SCHEMA = "rayline-router.selection-transaction.v1"
EXPECTED_CONCURRENCY = 8
EXPECTED_MEASURED_CASES = 128
EXPECTED_WARMUP_CASES = 8
MIN_CANONICAL_WORKERS = 2
HTTP_OK = 200
TOKEN_BUCKET_8K = 8192
TOKEN_BUCKET_32K = 32768
TOKEN_BUCKET_128K = 131072
INPUT_TOKEN_BUCKET_NAMES = (
    "lt_8192",
    "8192_to_32767",
    "32768_to_131071",
    "gte_131072",
)


class ProbeError(ValueError):
    """The input packet or observed serving response is invalid."""


@dataclass(frozen=True)
class Case:
    case_id: str
    episode_id: str
    input_tokens: int
    messages: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class WorkloadContract:
    """One registered packet size and admission level."""

    profile: str
    concurrency: int
    warmup_cases: int
    measured_cases: int


DIRECTIONAL_WORKLOAD = WorkloadContract(
    profile="directional-128-c8",
    concurrency=EXPECTED_CONCURRENCY,
    warmup_cases=EXPECTED_WARMUP_CASES,
    measured_cases=EXPECTED_MEASURED_CASES,
)
SWEEP_WORKLOADS = {
    **{
        concurrency: WorkloadContract(
            profile=f"sweep-32-c{concurrency}",
            concurrency=concurrency,
            warmup_cases=4,
            measured_cases=32,
        )
        for concurrency in (1, 4, 8)
    },
    # The PERF034 cap-raise cell: every lane of the full 128-case directional
    # corpus, one lane per measured episode. The 32-case sweep cells above are
    # a strict prefix of the same corpus, which is what keeps the identity
    # chain to PERF020 checkable after the digests change.
    32: WorkloadContract(
        profile="sweep-128-c32",
        concurrency=32,
        warmup_cases=8,
        measured_cases=128,
    ),
}
WORKLOAD_PROFILES = {
    contract.profile: contract
    for contract in (DIRECTIONAL_WORKLOAD, *SWEEP_WORKLOADS.values())
}


class JSONClient:
    """Small thread-local HTTP client with no automatic retries."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float,
        authorization: str = "",
    ) -> None:
        parsed = urlsplit(base_url.rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ProbeError("base URL must be absolute HTTP(S)")
        if parsed.query or parsed.fragment:
            raise ProbeError("base URL must not contain a query or fragment")
        self._scheme = parsed.scheme
        self._host = parsed.hostname
        self._port = parsed.port
        self._prefix = parsed.path.rstrip("/")
        self._timeout = timeout_seconds
        self._authorization = authorization
        self._local = threading.local()

    def _connection(self) -> http.client.HTTPConnection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection_type = (
                http.client.HTTPSConnection
                if self._scheme == "https"
                else http.client.HTTPConnection
            )
            connection = connection_type(
                self._host,
                self._port,
                timeout=self._timeout,
            )
            self._local.connection = connection
        return connection

    def close_thread_connection(self) -> None:
        """Close the caller thread's keep-alive connection, if one exists."""
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None

    def request(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[int, dict[str, Any], dict[str, str], float]:
        encoded = None if body is None else json.dumps(body).encode()
        request_headers = {"accept": "application/json"}
        if encoded is not None:
            request_headers["content-type"] = "application/json"
        if self._authorization:
            request_headers["authorization"] = self._authorization
        if headers:
            request_headers.update(headers)
        started = time.perf_counter()
        connection = self._connection()
        try:
            connection.request(
                method,
                f"{self._prefix}{path}",
                body=encoded,
                headers=request_headers,
            )
            response = connection.getresponse()
            raw = response.read()
        except (OSError, http.client.HTTPException):
            connection.close()
            self._local.connection = None
            raise
        elapsed = time.perf_counter() - started
        response_headers = {key.lower(): value for key, value in response.getheaders()}
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError as error:
            raise ProbeError(f"{method} {path} returned non-JSON") from error
        if not isinstance(payload, dict):
            raise ProbeError(f"{method} {path} returned a non-object body")
        return response.status, payload, response_headers, elapsed


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ProbeError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ProbeError(f"{label} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ProbeError(f"{label} keys differ from the frozen schema")


def load_corpus(path: Path) -> tuple[list[Case], list[Case]]:
    corpus = _read_json(path, "corpus")
    _exact_keys(corpus, {"schema_version", "warmup", "measured"}, "corpus")
    if corpus["schema_version"] != CORPUS_SCHEMA:
        raise ProbeError("unsupported corpus schema")

    def cases(raw_cases: object, label: str) -> list[Case]:
        if not isinstance(raw_cases, list):
            raise ProbeError(f"corpus.{label} must be a list")
        parsed: list[Case] = []
        for index, raw in enumerate(raw_cases):
            if not isinstance(raw, Mapping):
                raise ProbeError(f"corpus.{label}[{index}] must be an object")
            _exact_keys(
                raw,
                {"case_id", "episode_id", "input_tokens", "messages"},
                "case",
            )
            case_id = str(raw["case_id"])
            episode_id = str(raw["episode_id"])
            input_tokens = raw["input_tokens"]
            messages_raw = raw["messages"]
            if (
                not case_id
                or not episode_id
                or not isinstance(input_tokens, int)
                or isinstance(input_tokens, bool)
                or input_tokens <= 0
                or not isinstance(messages_raw, list)
            ):
                raise ProbeError("case identity/messages are invalid")
            messages: list[dict[str, str]] = []
            for message in messages_raw:
                if not isinstance(message, Mapping):
                    raise ProbeError("case message must be an object")
                _exact_keys(message, {"role", "content"}, "case message")
                role = str(message["role"])
                content = str(message["content"])
                if role not in {"system", "user", "assistant", "tool"} or not content:
                    raise ProbeError("case message role/content is invalid")
                messages.append({"role": role, "content": content})
            if not messages:
                raise ProbeError("case must contain at least one message")
            parsed.append(Case(case_id, episode_id, input_tokens, tuple(messages)))
        return parsed

    warmup = cases(corpus["warmup"], "warmup")
    measured = cases(corpus["measured"], "measured")
    ids = [case.case_id for case in [*warmup, *measured]]
    if len(ids) != len(set(ids)):
        raise ProbeError("case_id values must be unique")
    return warmup, measured


def load_packet(
    *,
    arm: str,
    corpus_path: Path,
    workload_path: Path,
    topology_path: Path,
    identity_path: Path,
    workload_contract: WorkloadContract = DIRECTIONAL_WORKLOAD,
) -> tuple[list[Case], list[Case], dict[str, Any], dict[str, str]]:
    warmup, measured = load_corpus(corpus_path)
    workload = _read_json(workload_path, "workload")
    _exact_keys(
        workload,
        {
            "schema_version",
            "concurrency",
            "warmup_cases",
            "measured_cases",
            "seed",
        },
        "workload",
    )
    if workload["schema_version"] != WORKLOAD_SCHEMA:
        raise ProbeError("unsupported workload schema")
    if workload["concurrency"] != workload_contract.concurrency:
        raise ProbeError(
            f"{workload_contract.profile} requires concurrency "
            f"{workload_contract.concurrency}"
        )
    if (
        workload["warmup_cases"] != len(warmup)
        or len(warmup) != workload_contract.warmup_cases
    ):
        raise ProbeError("workload warmup count differs from corpus")
    if (
        workload["measured_cases"] != len(measured)
        or len(measured) != workload_contract.measured_cases
    ):
        raise ProbeError(
            f"{workload_contract.profile} requires "
            f"{workload_contract.measured_cases} measured cases"
        )

    identity = _read_json(identity_path, "identity")
    if identity.get("case_count") != len(measured):
        raise ProbeError("identity case_count differs from corpus")
    if identity.get("seed") != workload["seed"]:
        raise ProbeError("identity seed differs from workload")
    digest_checks = {
        "corpus_sha256": _sha256(corpus_path),
        "workload_sha256": _sha256(workload_path),
        "worker_topology_sha256": _sha256(topology_path),
    }
    for field, digest in digest_checks.items():
        if identity.get(field) != digest:
            raise ProbeError(f"identity {field} differs from file digest")

    topology = _read_json(topology_path, "topology")
    _exact_keys(
        topology,
        {"schema_version", "canonical_workers", "arm_worker_maps"},
        "topology",
    )
    if topology["schema_version"] != TOPOLOGY_SCHEMA:
        raise ProbeError("unsupported topology schema")
    canonical = topology["canonical_workers"]
    maps = topology["arm_worker_maps"]
    if (
        not isinstance(canonical, list)
        or len(canonical) < MIN_CANONICAL_WORKERS
        or len(canonical) != len(set(map(str, canonical)))
        or not isinstance(maps, Mapping)
        or set(maps) != set(ARMS)
        or not isinstance(maps.get(arm), Mapping)
    ):
        raise ProbeError("topology worker maps are invalid")
    worker_map = {str(key): str(value) for key, value in maps[arm].items()}
    if set(worker_map.values()) != set(map(str, canonical)):
        raise ProbeError("arm worker map does not cover the canonical topology")
    return warmup, measured, identity, worker_map


def _episode_key(case: Case, seed: int, run_id: str) -> str:
    key = f"rayline-three-arm:{seed}:{run_id}".encode()
    return (
        "hmac-sha256:"
        + hmac.new(
            key,
            case.episode_id.encode(),
            hashlib.sha256,
        ).hexdigest()
    )


def _expect_ok(
    response: tuple[int, dict[str, Any], dict[str, str], float],
    operation: str,
) -> tuple[dict[str, Any], dict[str, str], float]:
    status, body, headers, elapsed = response
    if status != HTTP_OK:
        raise ProbeError(f"{operation} returned HTTP {status}")
    return body, headers, elapsed


def _select_modal(client: JSONClient, case: Case, run_id: str) -> tuple[str, float]:
    body, _headers, elapsed = _expect_ok(
        client.request(
            "POST",
            "/v1/route",
            body={
                "model": "rayline/router",
                "messages": list(case.messages),
                "metadata": {
                    "rayline_session_id": f"{run_id}:{case.episode_id}",
                    "rayline_task_id": case.case_id,
                },
            },
        ),
        "Modal route",
    )
    selected = str(body.get("selected_worker") or body.get("worker_id") or "")
    if not selected:
        raise ProbeError("Modal route omitted selected worker")
    return selected, elapsed


def _remote_capabilities(client: JSONClient) -> tuple[str, list[str]]:
    body, _headers, _elapsed = _expect_ok(
        client.request("GET", "/v1/route/capabilities"),
        "Remote capabilities",
    )
    bundle = str(body.get("bundle_version") or "")
    candidates = body.get("workers")
    if not bundle or not isinstance(candidates, list) or not candidates:
        raise ProbeError("Remote capabilities omitted bundle/workers")
    return bundle, [str(candidate) for candidate in candidates]


def _select_remote(
    client: JSONClient,
    case: Case,
    run_id: str,
    *,
    seed: int,
    bundle: str,
    candidates: list[str],
) -> tuple[str, float]:
    decision_id = f"{run_id}:{case.case_id}"
    prepared, _headers, prepare_elapsed = _expect_ok(
        client.request(
            "POST",
            "/v1/route/prepare",
            body={
                "schema_version": TRANSACTION_SCHEMA,
                "decision_id": decision_id,
                "episode_key": _episode_key(case, seed, run_id),
                "bundle_version": bundle,
                "candidates": candidates,
                "request": {
                    "protocol": "openai.chat.completions",
                    "model": "rayline/router",
                    "messages": list(case.messages),
                },
            },
        ),
        "Remote prepare",
    )
    if prepared.get("state") != "prepared":
        raise ProbeError("Remote prepare did not enter prepared state")
    selected = str(prepared.get("selected_worker") or "")
    committed, _headers, commit_elapsed = _expect_ok(
        client.request(
            "POST",
            "/v1/route/commit",
            body={
                "schema_version": TRANSACTION_SCHEMA,
                "decision_id": decision_id,
                "receipt": prepared.get("receipt"),
                "status_code": 200,
            },
        ),
        "Remote commit",
    )
    if committed.get("state") != "committed":
        raise ProbeError("Remote commit did not enter committed state")
    settled, _headers, settle_elapsed = _expect_ok(
        client.request(
            "POST",
            "/v1/route/settle",
            body={
                "schema_version": TRANSACTION_SCHEMA,
                "decision_id": decision_id,
                "receipt": prepared.get("receipt"),
                "outcome": {
                    "outcome_class": "success",
                    "status_code": 200,
                    "input_tokens": case.input_tokens,
                    "output_tokens": 0,
                    "latency_ms": 0,
                    "cost_usd": 0,
                    "error_type": None,
                },
            },
        ),
        "Remote settle",
    )
    if settled.get("state") != "settled":
        raise ProbeError("Remote settle did not enter settled state")
    if not selected:
        raise ProbeError("Remote prepare omitted selected worker")
    return selected, prepare_elapsed + commit_elapsed + settle_elapsed


def _select_arc(client: JSONClient, case: Case, run_id: str) -> tuple[str, float]:
    _body, headers, elapsed = _expect_ok(
        client.request(
            "POST",
            "/v1/chat/completions",
            body={
                "model": "auto",
                "messages": list(case.messages),
                "max_tokens": 1,
            },
            headers={
                "x-rayline-episode-id": f"{run_id}:{case.episode_id}",
                "x-rayline-route-id": f"{run_id}:{case.case_id}",
            },
        ),
        "ARC gateway",
    )
    selected = headers.get("x-vsr-selected-model", "")
    if not selected:
        raise ProbeError("ARC gateway omitted x-vsr-selected-model")
    return selected, elapsed


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _input_token_bucket(input_tokens: int) -> str:
    if input_tokens < TOKEN_BUCKET_8K:
        return "lt_8192"
    if input_tokens < TOKEN_BUCKET_32K:
        return "8192_to_32767"
    if input_tokens < TOKEN_BUCKET_128K:
        return "32768_to_131071"
    return "gte_131072"


def _latency_by_input_tokens(
    measured: list[Case], successes: Mapping[str, tuple[str, float]]
) -> dict[str, Any]:
    samples: dict[str, dict[str, Any]] = {
        name: {"scheduled": 0, "latencies": []} for name in INPUT_TOKEN_BUCKET_NAMES
    }
    for case in measured:
        bucket = samples[_input_token_bucket(case.input_tokens)]
        bucket["scheduled"] += 1
        outcome = successes.get(case.case_id)
        if outcome is not None:
            bucket["latencies"].append(outcome[1])
    rendered: dict[str, Any] = {}
    for name, sample in samples.items():
        latencies = sample["latencies"]
        completed = len(latencies)
        rendered[name] = {
            "scheduled": sample["scheduled"],
            "completed": completed,
            "failed": sample["scheduled"] - completed,
            "selection_latency_seconds": (
                {
                    "p50": _percentile(latencies, 0.50),
                    "p95": _percentile(latencies, 0.95),
                    "p99": _percentile(latencies, 0.99),
                }
                if latencies
                else None
            ),
        }
    return rendered


def _selector(
    *,
    arm: str,
    client: JSONClient,
    run_id: str,
    identity: Mapping[str, Any],
) -> Callable[[Case], tuple[str, float]]:
    if arm == "modal_inprocess":
        return lambda case: _select_modal(client, case, run_id)
    if arm == "rayline_remote":
        bundle, candidates = _remote_capabilities(client)
        return lambda case: _select_remote(
            client,
            case,
            run_id,
            seed=int(identity["seed"]),
            bundle=bundle,
            candidates=candidates,
        )
    return lambda case: _select_arc(client, case, run_id)


def _select_episode(
    cases: list[Case],
    select: Callable[[Case], tuple[str, float]],
    worker_map: Mapping[str, str],
) -> list[tuple[str, str, float] | None]:
    outcomes: list[tuple[str, str, float] | None] = []
    for case in cases:
        try:
            observed, latency = select(case)
            canonical = worker_map.get(observed)
            if canonical is None:
                raise ProbeError("measured request selected an unmapped worker")
            outcomes.append((case.case_id, canonical, latency))
        except (OSError, ProbeError):
            outcomes.append(None)
    return outcomes


def run_probe(
    *,
    arm: str,
    protocol: str,
    client: JSONClient,
    warmup: list[Case],
    measured: list[Case],
    identity: Mapping[str, Any],
    worker_map: Mapping[str, str],
    run_id: str,
    concurrency: int = EXPECTED_CONCURRENCY,
) -> dict[str, Any]:
    if PROTOCOL_BY_ARM.get(arm) != protocol:
        raise ProbeError("arm/protocol pair is not registered")
    select = _selector(arm=arm, client=client, run_id=run_id, identity=identity)

    for case in warmup:
        observed, _latency = select(case)
        if observed not in worker_map:
            raise ProbeError("warmup selected an unmapped worker")

    # A Rayline episode is an ordered transaction stream. Running two turns
    # from the same episode concurrently would measure a race the production
    # contract forbids and would invalidate the encoder KV-cache comparison.
    # Parallelize episode lanes while preserving case order within each lane.
    episode_lanes: dict[str, list[Case]] = {}
    for case in measured:
        episode_lanes.setdefault(case.episode_id, []).append(case)

    started = time.perf_counter()
    successes: dict[str, tuple[str, float]] = {}
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(_select_episode, cases, select, worker_map)
            for cases in episode_lanes.values()
        ]
        for future in concurrent.futures.as_completed(futures):
            for outcome in future.result():
                if outcome is None:
                    failures += 1
                else:
                    case_id, canonical, latency = outcome
                    successes[case_id] = (canonical, latency)
    duration = time.perf_counter() - started
    ordered_successes = [
        (case.case_id, *successes[case.case_id])
        for case in measured
        if case.case_id in successes
    ]
    latencies = [latency for _case_id, _worker, latency in ordered_successes]
    if not latencies:
        raise ProbeError("arm completed zero measured decisions")
    trace = json.dumps(
        [[case_id, worker] for case_id, worker, _latency in ordered_successes],
        separators=(",", ":"),
    ).encode()
    receipt = {
        "schema_version": INPUT_SCHEMA,
        "arm": arm,
        "run_id": run_id,
        "identity": dict(identity),
        "results": {
            "scheduled": len(measured),
            "completed": len(ordered_successes),
            "failed": failures,
            "duration_seconds": duration,
            "throughput_rps": len(ordered_successes) / duration,
            "selection_latency_seconds": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "p99": _percentile(latencies, 0.99),
            },
            "selected_worker_trace_sha256": hashlib.sha256(trace).hexdigest(),
            "provider_calls": 0,
            "latency_by_input_tokens": _latency_by_input_tokens(measured, successes),
        },
    }
    return validate_receipt(receipt)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--protocol", required=True, choices=PROTOCOL_BY_ARM.values())
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--authorization-env", default="")
    parser.add_argument(
        "--workload-profile",
        choices=WORKLOAD_PROFILES,
        default=DIRECTIONAL_WORKLOAD.profile,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    authorization = ""
    if args.authorization_env:
        token = os.environ.get(args.authorization_env, "")
        if not token:
            raise SystemExit("authorization environment variable is empty")
        authorization = f"Bearer {token}"
    try:
        workload_contract = WORKLOAD_PROFILES[args.workload_profile]
        warmup, measured, identity, worker_map = load_packet(
            arm=args.arm,
            corpus_path=args.corpus,
            workload_path=args.workload,
            topology_path=args.topology,
            identity_path=args.identity,
            workload_contract=workload_contract,
        )
        report = run_probe(
            arm=args.arm,
            protocol=args.protocol,
            client=JSONClient(
                args.base_url,
                timeout_seconds=args.timeout_seconds,
                authorization=authorization,
            ),
            warmup=warmup,
            measured=measured,
            identity=identity,
            worker_map=worker_map,
            run_id=args.run_id,
            concurrency=workload_contract.concurrency,
        )
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (OSError, ProbeError, ValueError) as error:
        raise SystemExit(f"invalid parity arm packet: {error}") from error


if __name__ == "__main__":
    main()
