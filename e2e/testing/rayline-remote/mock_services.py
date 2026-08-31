#!/usr/bin/env python3
"""Contract-faithful Rayline decision plane and fake OpenAI providers."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

TRANSACTION_VERSION = "rayline-router.selection-transaction.v1"
CAPABILITIES_VERSION = "rayline-router.selection-capabilities.v1"
BUNDLE_VERSION = "rayline-remote-e2e-bundle-v1"
ROUTER_KEY = "public-e2e-rayline-key"
PROVIDER_KEY = "public-e2e-provider-key"
LEASE_SECONDS = 30.0
WORKERS = ("remote-worker-a", "remote-worker-b")
HTTP_SUCCESS_MIN = 200
HTTP_SUCCESS_MAX = 299


def json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def message_text(body: dict[str, Any]) -> str:
    parts: list[str] = []
    for message in body.get("messages", []):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


class QuietHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("request body must be an object")
        return value

    def send_json(self, status: int, value: object) -> None:
        payload = json_bytes(value)
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        with contextlib.suppress(BrokenPipeError):
            self.wfile.write(payload)

    def send_error_code(self, status: int, code: str) -> None:
        self.send_json(
            status,
            {
                "schema_version": TRANSACTION_VERSION,
                "error": {"code": code, "message": "bounded fixture failure"},
            },
        )


@dataclass
class Episode:
    route_call_index: int = 0
    previous_worker: str = ""
    previous_input_tokens: int | None = None
    pending_decision: str = ""


@dataclass
class Transaction:
    decision_id: str
    digest: str
    receipt: str
    episode_key: str
    route_call_index: int
    selected_worker: str
    response: dict[str, Any]
    state: str = "prepared"
    status_code: int = 0
    abort_reason: str = ""
    outcome: dict[str, Any] | None = None


@dataclass
class RaylineState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    mode: str = "normal"
    receipt_index: int = 0
    episodes: dict[str, Episode] = field(default_factory=dict)
    transactions: dict[str, Transaction] = field(default_factory=dict)
    prepares: list[dict[str, Any]] = field(default_factory=list)
    operation_counts: dict[str, int] = field(default_factory=dict)

    def count(self, operation: str) -> None:
        self.operation_counts[operation] = self.operation_counts.get(operation, 0) + 1

    def reset(self) -> None:
        with self.lock:
            self.mode = "normal"
            self.receipt_index = 0
            self.episodes.clear()
            self.transactions.clear()
            self.prepares.clear()
            self.operation_counts.clear()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "mode": self.mode,
                "episodes": {
                    key: {
                        "route_call_index": episode.route_call_index,
                        "previous_worker": episode.previous_worker,
                        "previous_input_tokens": episode.previous_input_tokens,
                        "pending": bool(episode.pending_decision),
                    }
                    for key, episode in self.episodes.items()
                },
                "prepares": list(self.prepares),
                "operation_counts": dict(self.operation_counts),
                "transactions": {
                    decision_id: {
                        "state": transaction.state,
                        "route_call_index": transaction.route_call_index,
                        "selected_worker": transaction.selected_worker,
                        "status_code": transaction.status_code,
                        "abort_reason": transaction.abort_reason,
                        "outcome": transaction.outcome,
                    }
                    for decision_id, transaction in self.transactions.items()
                },
            }


RAYLINE_STATE = RaylineState()


def worker_row(
    worker_id: str,
    model: str,
    thinking_mode: str,
) -> dict[str, Any]:
    return {
        "id": worker_id,
        "backend": "mock",
        "model": model,
        "thinking_mode": thinking_mode,
        "openrouter_allow_fallbacks": False,
        "capability_tags": ["text"],
        "pricing_snapshot_version": "rayline-remote-e2e-prices-v1",
        "per_token_prices": {
            "input": 0.000002,
            "cache_read": 0.000001,
            "cache_write": 0.0000025,
            "output": 0.000004,
        },
    }


class RaylineHandler(QuietHandler):
    def authorized(self) -> bool:
        if self.headers.get("authorization") == f"Bearer {ROUTER_KEY}":
            return True
        self.send_error_code(401, "unauthorized")
        return False

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(200, {"status": "ok"})
            return
        if self.path == "/observed":
            self.send_json(200, RAYLINE_STATE.snapshot())
            return
        if self.path == "/v1/route/capabilities":
            if not self.authorized():
                return
            self.send_json(
                200,
                {
                    "schema_version": CAPABILITIES_VERSION,
                    "transaction_schema_version": TRANSACTION_VERSION,
                    "bundle_version": BUNDLE_VERSION,
                    "protocols": ["openai.chat.completions"],
                    "operations": [
                        "prepare",
                        "renew",
                        "commit",
                        "abort",
                        "settle",
                    ],
                    # The outside worker proves the request candidate mask,
                    # rather than the global catalog, is authoritative.
                    "workers": [
                        "outside-worker",
                        "remote-worker-a",
                        "remote-worker-b",
                    ],
                    "lease_seconds": LEASE_SECONDS,
                    "pending_journal": "bounded_in_process_single_replica",
                },
            )
            return
        if self.path == "/v1/workers":
            if not self.authorized():
                return
            self.send_json(
                200,
                [
                    worker_row(
                        "outside-worker",
                        "provider/outside",
                        "disabled",
                    ),
                    worker_row(
                        "remote-worker-a",
                        "provider/a",
                        "disabled",
                    ),
                    worker_row(
                        "remote-worker-b",
                        "provider/b",
                        "enabled",
                    ),
                ],
            )
            return
        self.send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path == "/reset":
            RAYLINE_STATE.reset()
            self.send_json(200, {"status": "reset"})
            return
        if self.path == "/mode":
            body = self.read_json()
            with RAYLINE_STATE.lock:
                RAYLINE_STATE.mode = str(body.get("mode") or "normal")
            self.send_json(200, {"status": "ok"})
            return
        if not self.path.startswith("/v1/route/"):
            self.send_json(404, {"error": "not_found"})
            return
        if not self.authorized():
            return
        try:
            body = self.read_json()
        except (ValueError, json.JSONDecodeError):
            self.send_error_code(400, "invalid_json")
            return
        operation = self.path.rsplit("/", 1)[-1]
        with RAYLINE_STATE.lock:
            RAYLINE_STATE.count(operation)
            mode = RAYLINE_STATE.mode
        if operation == "prepare":
            self.prepare(body, mode)
            return
        self.lifecycle(operation, body, mode)

    def prepare(self, body: dict[str, Any], mode: str) -> None:
        if mode == "timeout_prepare":
            time.sleep(1.5)
            self.send_error_code(503, "selection_timeout")
            return
        required = {
            "schema_version",
            "decision_id",
            "episode_key",
            "bundle_version",
            "candidates",
            "request",
        }
        candidates = body.get("candidates")
        request = body.get("request")
        if (
            set(body) != required
            or body.get("schema_version") != TRANSACTION_VERSION
            or body.get("bundle_version") != BUNDLE_VERSION
            or not isinstance(candidates, list)
            or candidates != list(WORKERS)
            or not isinstance(request, dict)
            or request.get("protocol") != "openai.chat.completions"
            or "model" in request
            or not str(body.get("episode_key", "")).startswith("hmac-sha256:")
        ):
            self.send_error_code(400, "invalid_prepare")
            return
        decision_id = str(body["decision_id"])
        episode_key = str(body["episode_key"])
        digest = hashlib.sha256(json_bytes(body)).hexdigest()
        with RAYLINE_STATE.lock:
            previous = RAYLINE_STATE.transactions.get(decision_id)
            if previous is not None:
                if previous.digest != digest:
                    self.send_error_code(409, "decision_id_conflict")
                    return
                response = dict(previous.response)
            else:
                episode = RAYLINE_STATE.episodes.setdefault(
                    episode_key,
                    Episode(),
                )
                if episode.pending_decision:
                    self.send_error_code(409, "episode_busy")
                    return
                RAYLINE_STATE.receipt_index += 1
                receipt = f"rsel_fixture_{RAYLINE_STATE.receipt_index:08d}"
                response = {
                    "schema_version": TRANSACTION_VERSION,
                    "decision_id": decision_id,
                    "receipt": receipt,
                    "state": "prepared",
                    "selected_worker": "remote-worker-b",
                    "policy": "outside-worker-masked-to-b",
                    "route_call_index": episode.route_call_index,
                    "bundle_version": BUNDLE_VERSION,
                    "lease_expires_at": time.strftime(
                        "%Y-%m-%dT%H:%M:%S.000Z",
                        time.gmtime(time.time() + LEASE_SECONDS),
                    ),
                }
                transaction = Transaction(
                    decision_id=decision_id,
                    digest=digest,
                    receipt=receipt,
                    episode_key=episode_key,
                    route_call_index=episode.route_call_index,
                    selected_worker="remote-worker-b",
                    response=dict(response),
                )
                RAYLINE_STATE.transactions[decision_id] = transaction
                episode.pending_decision = decision_id
                RAYLINE_STATE.prepares.append(
                    {
                        "decision_id": decision_id,
                        "episode_key": episode_key,
                        "candidates": list(candidates),
                        "route_call_index": episode.route_call_index,
                        "previous_worker": episode.previous_worker,
                        "selected_worker": transaction.selected_worker,
                    }
                )
        if mode == "wrong_worker":
            response["selected_worker"] = "outside-worker"
        elif mode == "wrong_bundle":
            response["bundle_version"] = "wrong-bundle"
        elif mode == "wrong_decision":
            response["decision_id"] = "rt_wrong-decision"
        elif mode == "malformed_receipt":
            response["receipt"] = "not a valid receipt"
        self.send_json(200, response)

    # The fixture keeps the complete wire state machine in one visible seam.
    def lifecycle(  # noqa: C901, PLR0912
        self,
        operation: str,
        body: dict[str, Any],
        mode: str,
    ) -> None:
        if body.get("schema_version") != TRANSACTION_VERSION:
            self.send_error_code(400, "unsupported_schema")
            return
        decision_id = str(body.get("decision_id") or "")
        receipt = str(body.get("receipt") or "")
        with RAYLINE_STATE.lock:
            transaction = RAYLINE_STATE.transactions.get(decision_id)
            if transaction is None or transaction.receipt != receipt:
                self.send_error_code(409, "receipt_conflict")
                return
            episode = RAYLINE_STATE.episodes[transaction.episode_key]
            if operation == "renew":
                if mode == "renew_fail":
                    self.send_error_code(410, "selection_lease_expired")
                    return
                response = self.state_response(transaction)
                response["lease_expires_at"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%S.000Z",
                    time.gmtime(time.time() + LEASE_SECONDS),
                )
            elif operation == "commit":
                if mode == "commit_fail":
                    self.send_error_code(503, "state_commit_unavailable")
                    return
                status_code = body.get("status_code")
                if (
                    not isinstance(status_code, int)
                    or status_code < HTTP_SUCCESS_MIN
                    or status_code > HTTP_SUCCESS_MAX
                ):
                    self.send_error_code(400, "invalid_commit_status")
                    return
                if transaction.state == "prepared":
                    episode.route_call_index += 1
                    episode.previous_worker = transaction.selected_worker
                    episode.pending_decision = ""
                    transaction.state = "committed"
                    transaction.status_code = status_code
                response = self.state_response(transaction)
                if mode == "commit_wrong_receipt":
                    response["receipt"] = "rsel_wrong_receipt"
            elif operation == "abort":
                reason = str(body.get("reason") or "")
                if transaction.state == "prepared":
                    transaction.state = "aborted"
                    transaction.abort_reason = reason
                    episode.pending_decision = ""
                response = self.state_response(transaction)
            elif operation == "settle":
                outcome = body.get("outcome")
                if not isinstance(outcome, dict):
                    self.send_error_code(400, "invalid_outcome")
                    return
                if transaction.state == "committed":
                    transaction.state = "settled"
                    transaction.outcome = dict(outcome)
                    if isinstance(outcome.get("input_tokens"), int):
                        episode.previous_input_tokens = int(outcome["input_tokens"])
                response = self.state_response(transaction)
            else:
                self.send_error_code(404, "unknown_operation")
                return
        self.send_json(200, response)

    @staticmethod
    def state_response(transaction: Transaction) -> dict[str, Any]:
        return {
            "schema_version": TRANSACTION_VERSION,
            "decision_id": transaction.decision_id,
            "receipt": transaction.receipt,
            "state": transaction.state,
            "route_call_index": transaction.route_call_index,
        }


@dataclass
class ProviderState:
    identity: str
    lock: threading.Lock = field(default_factory=threading.Lock)
    requests: list[dict[str, Any]] = field(default_factory=list)

    def append(self, body: dict[str, Any], auth_ok: bool) -> None:
        text = message_text(body)
        summary = {
            "model": body.get("model"),
            "auth_ok": auth_ok,
            "provider_5xx": "REMOTE_PROVIDER_5XX" in text,
            "provider_delay": "REMOTE_PROVIDER_DELAY" in text,
            "hold": "REMOTE_HOLD" in text,
            "stream": "REMOTE_STREAM" in text,
            "stream_break": "REMOTE_STREAM_BREAK" in text,
            "no_usage": "REMOTE_NO_USAGE" in text,
        }
        with self.lock:
            self.requests.append(summary)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "identity": self.identity,
                "count": len(self.requests),
                "requests": list(self.requests),
            }

    def reset(self) -> None:
        with self.lock:
            self.requests.clear()


PROVIDER_STATE = ProviderState("")


class ProviderHandler(QuietHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(200, {"status": "ok"})
            return
        if self.path == "/observed":
            self.send_json(200, PROVIDER_STATE.snapshot())
            return
        self.send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path == "/reset":
            PROVIDER_STATE.reset()
            self.send_json(200, {"status": "reset"})
            return
        if self.path != "/v1/chat/completions":
            self.send_json(404, {"error": "not_found"})
            return
        try:
            body = self.read_json()
        except (ValueError, json.JSONDecodeError):
            self.send_json(400, {"error": {"message": "invalid request"}})
            return
        auth_ok = self.headers.get("authorization") == f"Bearer {PROVIDER_KEY}"
        PROVIDER_STATE.append(body, auth_ok)
        text = message_text(body)
        if not auth_ok:
            self.send_json(401, {"error": {"message": "missing key"}})
            return
        if "REMOTE_PROVIDER_5XX" in text:
            self.send_json(503, {"error": {"message": "fixture failure"}})
            return
        if "REMOTE_PROVIDER_DELAY" in text:
            time.sleep(3)
        if "REMOTE_HOLD" in text:
            time.sleep(0.6)
        if "REMOTE_STREAM_BREAK" in text:
            self.send_sse(break_stream=True)
            return
        if "REMOTE_STREAM" in text:
            self.send_sse(break_stream=False)
            return
        response = {
            "id": f"chatcmpl-{PROVIDER_STATE.identity}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"provider-{PROVIDER_STATE.identity}",
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        if "REMOTE_NO_USAGE" not in text:
            response["usage"] = {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            }
        self.send_json(200, response)

    def send_sse(self, *, break_stream: bool) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        first = {
            "id": f"chatcmpl-{PROVIDER_STATE.identity}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": f"provider/{PROVIDER_STATE.identity}",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": f"provider-{PROVIDER_STATE.identity}",
                    },
                    "finish_reason": None,
                }
            ],
        }
        with contextlib.suppress(BrokenPipeError):
            self.write_chunk(b"data: " + json_bytes(first) + b"\n\n")
            self.wfile.flush()
            if break_stream:
                # Closing a chunked response without the terminal zero-length
                # chunk is an observable truncation, not a valid close-
                # delimited response.
                self.close_connection = True
                return
            final = {
                "id": f"chatcmpl-{PROVIDER_STATE.identity}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": f"provider/{PROVIDER_STATE.identity}",
                "choices": [],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                },
            }
            self.write_chunk(b"data: " + json_bytes(final) + b"\n\n")
            self.write_chunk(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        self.close_connection = True

    def write_chunk(self, payload: bytes) -> None:
        self.wfile.write(f"{len(payload):x}\r\n".encode())
        self.wfile.write(payload)
        self.wfile.write(b"\r\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("service", choices=("rayline", "provider"))
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--identity", default="")
    args = parser.parse_args()
    handler: type[BaseHTTPRequestHandler]
    if args.service == "rayline":
        handler = RaylineHandler
    else:
        if args.identity not in {"a", "b"}:
            raise SystemExit("provider identity must be a or b")
        PROVIDER_STATE.identity = args.identity
        PROVIDER_STATE.reset()
        handler = ProviderHandler
    server = ThreadingHTTPServer(("0.0.0.0", args.port), handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
