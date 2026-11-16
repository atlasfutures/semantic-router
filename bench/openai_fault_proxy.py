#!/usr/bin/env python3
"""OpenAI-compatible fault injection proxy for live routing benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

HTTP_OK = 200
HTTP_BAD_GATEWAY = 502
HTTP_UNAVAILABLE = 503
SESSION_HEADER = "x-session-id"
# Stream faults (--stream-fault) act on streamed calls ("stream": true) only.
# Each is keyed by the call's conversation: a digest of its "messages" (Chat,
# Messages) or "input" (Responses). A re-send of the same turn carries the
# same conversation, so it is the key's second call; the next turn adds a
# message, so it is a new key whose first call is faulted again.
STREAM_FAULT_STALL_ONCE = "stall-once"
STREAM_FAULT_STALL_TWICE = "stall-twice"
STREAM_FAULT_CUT_AFTER_CONTENT = "cut-after-content"
STREAM_FAULTS = (
    STREAM_FAULT_STALL_ONCE,
    STREAM_FAULT_STALL_TWICE,
    STREAM_FAULT_CUT_AFTER_CONTENT,
)
STALL_POLL_SECONDS = 0.25
STREAM_UPSTREAM_TIMEOUT_SECONDS = 600
HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


@dataclass(frozen=True)
class FaultPolicy:
    fail_turns: frozenset[int]
    fail_phases: frozenset[str]
    fail_status: int
    fail_session_mod: int
    fail_session_remainder: int
    fail_once_per_session: bool
    delay_ms: int = 0
    delay_jitter_ms: int = 0
    stream_fault: str = ""
    stall_seconds: float = 120.0


@dataclass(frozen=True)
class RequestShape:
    session_id: str
    session_index: int
    turn: int | None
    phase: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=18090)
    parser.add_argument("--upstream-base-url", required=True)
    parser.add_argument("--fail-turns", default="")
    parser.add_argument("--fail-phases", default="")
    parser.add_argument("--fail-status", type=int, default=HTTP_UNAVAILABLE)
    parser.add_argument("--fail-session-mod", type=int, default=1)
    parser.add_argument("--fail-session-remainder", type=int, default=0)
    parser.add_argument("--repeat-failures", action="store_true")
    parser.add_argument("--delay-ms", type=int, default=0)
    parser.add_argument("--delay-jitter-ms", type=int, default=0)
    parser.add_argument(
        "--stream-fault",
        choices=("", *STREAM_FAULTS),
        default="",
        help=(
            "fault streamed calls per conversation: stall-once (first call "
            "answers 200 and SSE headers, then no bytes for --stall-seconds), "
            "stall-twice (first two calls stall), cut-after-content (first "
            "call relays its first content event, then closes before the end)"
        ),
    )
    parser.add_argument(
        "--stall-seconds",
        type=float,
        default=120.0,
        help="how long a stalled call stays silent; set it above the first-content limit",
    )
    parser.add_argument("--log-jsonl", type=Path, default=None)
    return parser.parse_args()


def parse_policy(args: argparse.Namespace) -> FaultPolicy:
    return FaultPolicy(
        fail_turns=frozenset(parse_int_csv(args.fail_turns)),
        fail_phases=frozenset(parse_str_csv(args.fail_phases)),
        fail_status=args.fail_status,
        fail_session_mod=args.fail_session_mod,
        fail_session_remainder=args.fail_session_remainder,
        fail_once_per_session=not args.repeat_failures,
        delay_ms=args.delay_ms,
        delay_jitter_ms=args.delay_jitter_ms,
        stream_fault=args.stream_fault,
        stall_seconds=args.stall_seconds,
    )


def delay_seconds(policy: FaultPolicy) -> float:
    """Return the latency to apply before handling a request, in seconds."""
    if policy.delay_ms <= 0 and policy.delay_jitter_ms <= 0:
        return 0.0
    total_ms = float(max(policy.delay_ms, 0))
    if policy.delay_jitter_ms > 0:
        total_ms += random.uniform(0.0, policy.delay_jitter_ms)
    return total_ms / 1000.0


def parse_int_csv(raw: str) -> set[int]:
    if not raw.strip():
        return set()
    return {int(value.strip()) for value in raw.split(",") if value.strip()}


def parse_str_csv(raw: str) -> set[str]:
    return {value.strip() for value in raw.split(",") if value.strip()}


class FaultProxyHandler(BaseHTTPRequestHandler):
    # HTTP/1.1 keep-alive avoids a TCP handshake + thread per request under soak
    # load; safe because every response path sets an explicit Content-Length.
    protocol_version = "HTTP/1.1"

    upstream: ClassVar[str]
    fault_policy: ClassVar[FaultPolicy]
    seen_failures: ClassVar[set[str]]
    seen_lock: ClassVar[threading.Lock]
    stream_calls: ClassVar[dict[str, int]]
    log_path: ClassVar[Path | None]
    log_lock: ClassVar[threading.Lock]

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(HTTP_OK, {"status": "ok"})
            return
        self.forward(b"")

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        shape = request_shape(body, self.headers)
        with self.seen_lock:
            inject_fault = should_inject_fault(
                shape, self.fault_policy, self.seen_failures
            )
        delay = delay_seconds(self.fault_policy)
        if delay > 0:
            time.sleep(delay)
        if inject_fault:
            message = (
                f"fault proxy injected {self.fault_policy.fail_status} for "
                f"session={shape.session_id} turn={shape.turn} phase={shape.phase}"
            )
            self.record_event("injected", shape, self.fault_policy.fail_status)
            self.send_json(
                self.fault_policy.fail_status,
                {"error": {"message": message, "type": "fault_proxy_injected"}},
                {"x-vsr-fault-injected": "true"},
            )
            return
        payload = parse_json_body(body)
        if self.fault_policy.stream_fault and payload.get("stream") is True:
            key = stream_fault_key(payload, self.headers.get(SESSION_HEADER, ""))
            with self.seen_lock:
                call = self.stream_calls.get(key, 0) + 1
                self.stream_calls[key] = call
            fault = stream_fault_for_call(self.fault_policy.stream_fault, call)
            extra = {"key": key, "call": call}
            if fault == "stall":
                self.stall(shape, extra)
                return
            self.forward_stream(body, shape, extra, cut_after_content=fault == "cut")
            return
        self.forward(body, shape)

    def stall(self, shape: RequestShape, extra: dict[str, Any]) -> None:
        """Answer 200 and SSE headers, then send nothing and close."""
        self.close_connection = True
        self.send_response(HTTP_OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("x-vsr-fault-injected", "stall")
        self.end_headers()
        self.wfile.flush()
        self.record_event("stalled", shape, HTTP_OK, extra)
        deadline = time.monotonic() + self.fault_policy.stall_seconds
        while time.monotonic() < deadline:
            time.sleep(STALL_POLL_SECONDS)

    def forward_stream(
        self,
        body: bytes,
        shape: RequestShape,
        extra: dict[str, Any],
        cut_after_content: bool = False,
    ) -> None:
        """Relay a streamed call line by line; with cut_after_content, close it
        right after its first content event, before any terminal event."""
        target = join_url(self.upstream, self.path)
        headers = {
            name: value
            for name, value in filtered_headers(self.headers).items()
            if name.lower() != "accept-encoding"
        }
        request = urllib.request.Request(
            target, data=body, headers=headers, method=self.command
        )
        try:
            response = urllib.request.urlopen(
                request, timeout=STREAM_UPSTREAM_TIMEOUT_SECONDS
            )
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            self.send_response(exc.code)
            self.forward_headers(dict(exc.headers), len(payload))
            self.end_headers()
            self.wfile.write(payload)
            self.record_event("upstream_error", shape, exc.code, extra)
            return
        except Exception as exc:  # pragma: no cover - network errors vary
            self.record_event("proxy_error", shape, HTTP_BAD_GATEWAY, extra)
            self.send_json(
                HTTP_BAD_GATEWAY,
                {"error": {"message": str(exc), "type": "fault_proxy_error"}},
            )
            return
        with response:
            self.close_connection = True
            self.send_response(response.status)
            for name, value in response.headers.items():
                if name.lower() not in HOP_BY_HOP_HEADERS:
                    self.send_header(name, value)
            self.send_header("Connection", "close")
            if cut_after_content:
                self.send_header("x-vsr-fault-injected", "cut-after-content")
            self.end_headers()
            content_seen = False
            for line in response:
                self.wfile.write(line)
                self.wfile.flush()
                if not cut_after_content:
                    continue
                content_seen = content_seen or sse_line_has_content(line)
                # Cut once the content event is complete (its blank line),
                # so the client has parsed it.
                if content_seen and not line.strip():
                    self.record_event(
                        "cut_after_content", shape, response.status, extra
                    )
                    return
        self.record_event("forwarded_stream", shape, response.status, extra)

    def forward(self, body: bytes, shape: RequestShape | None = None) -> None:
        target = join_url(self.upstream, self.path)
        headers = filtered_headers(self.headers)
        request = urllib.request.Request(
            target,
            data=body if body else None,
            headers=headers,
            method=self.command,
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
                self.send_response(response.status)
                self.forward_headers(dict(response.headers), len(payload))
                self.end_headers()
                self.wfile.write(payload)
                if shape is not None:
                    self.record_event("forwarded", shape, response.status)
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            self.send_response(exc.code)
            self.forward_headers(dict(exc.headers), len(payload))
            self.end_headers()
            self.wfile.write(payload)
            if shape is not None:
                self.record_event("upstream_error", shape, exc.code)
        except Exception as exc:  # pragma: no cover - network errors vary
            if shape is not None:
                self.record_event("proxy_error", shape, HTTP_BAD_GATEWAY)
            self.send_json(
                HTTP_BAD_GATEWAY,
                {"error": {"message": str(exc), "type": "fault_proxy_error"}},
            )

    def forward_headers(self, headers: dict[str, str], length: int) -> None:
        for name, value in headers.items():
            if name.lower() not in HOP_BY_HOP_HEADERS:
                self.send_header(name, value)
        self.send_header("Content-Length", str(length))

    def send_json(
        self,
        status: int,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(raw)

    def record_event(
        self,
        action: str,
        shape: RequestShape,
        status: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if self.log_path is None:
            return
        event = {
            **(extra or {}),
            "action": action,
            "session_id": shape.session_id,
            "session_index": shape.session_index,
            "turn": shape.turn,
            "phase": shape.phase,
            "status": status,
        }
        with self.log_lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps(event, sort_keys=True) + "\n")

    def log_message(self, _fmt: str, *_args: Any) -> None:
        return


def make_handler(
    upstream_base_url: str,
    policy: FaultPolicy,
    log_jsonl: Path | None = None,
) -> type[BaseHTTPRequestHandler]:
    class ConfiguredFaultProxyHandler(FaultProxyHandler):
        upstream: ClassVar[str] = upstream_base_url.rstrip("/")
        fault_policy: ClassVar[FaultPolicy] = policy
        seen_failures: ClassVar[set[str]] = set()
        seen_lock: ClassVar[threading.Lock] = threading.Lock()
        stream_calls: ClassVar[dict[str, int]] = {}
        log_path: ClassVar[Path | None] = log_jsonl
        log_lock: ClassVar[threading.Lock] = threading.Lock()

    return ConfiguredFaultProxyHandler


def filtered_headers(headers: Any) -> dict[str, str]:
    return {
        name: value
        for name, value in headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS
    }


def join_url(upstream_base_url: str, path: str) -> str:
    return upstream_base_url.rstrip("/") + "/" + path.lstrip("/")


def request_shape(body: bytes, headers: Any) -> RequestShape:
    payload = parse_json_body(body)
    text = "\n".join(message_texts(payload))
    session_id = headers.get(SESSION_HEADER, "") or session_id_from_text(text)
    session_id = session_id or "unknown-session"
    return RequestShape(
        session_id=session_id,
        session_index=session_index(session_id),
        turn=turn_from_text(text),
        phase=phase_from_text(text),
    )


def parse_json_body(body: bytes) -> dict[str, Any]:
    if not body:
        return {}
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def message_texts(payload: dict[str, Any]) -> list[str]:
    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        return []
    texts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        texts.extend(content_texts(content))
    return texts


def content_texts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    texts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            value = item.get("text")
            if isinstance(value, str):
                texts.append(value)
        elif isinstance(item, str):
            texts.append(item)
    return texts


def session_id_from_text(text: str) -> str:
    match = re.search(r"\bSession\s+(\d+)\b", text)
    return f"session-{match.group(1)}" if match else ""


def session_index(session_id: str) -> int:
    match = re.search(r"(\d+)(?!.*\d)", session_id)
    return int(match.group(1)) if match else 0


def turn_from_text(text: str) -> int | None:
    match = re.search(r"\bturn\s+(\d+)\b", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def phase_from_text(text: str) -> str:
    lowered = text.lower()
    if "provided tool result" in lowered or "tool result ready" in lowered:
        return "tool_loop"
    if "previous response state" in lowered:
        return "provider_state"
    if "task direction changed" in lowered:
        return "topic_drift"
    if "harder planning step" in lowered:
        return "frontier_turn"
    if "idle pause" in lowered:
        return "idle_boundary"
    return "user_turn"


def stream_fault_key(payload: dict[str, Any], session: str = "") -> str:
    """Name a streamed call's conversation: a digest of its session header,
    model, system prompt, Responses parent and messages (or input). A
    re-send changes none of these; it may change only provider routing."""
    conversation = {
        "session": session,
        "model": payload.get("model"),
        "system": payload.get("system"),
        "previous_response_id": payload.get("previous_response_id"),
        "messages": payload.get("messages", payload.get("input")),
    }
    raw = json.dumps(conversation, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def stream_fault_for_call(mode: str, call: int) -> str:
    """The fault for a conversation's nth streamed call: "stall", "cut" or ""."""
    if mode == STREAM_FAULT_STALL_ONCE and call == 1:
        return "stall"
    if mode == STREAM_FAULT_STALL_TWICE and call <= 2:
        return "stall"
    if mode == STREAM_FAULT_CUT_AFTER_CONTENT and call == 1:
        return "cut"
    return ""


def sse_line_has_content(line: bytes) -> bool:
    """Report whether an SSE data line carries content: text, reasoning or a
    tool call (Chat), a content block delta (Messages), or an output delta
    (Responses)."""
    text = line.decode("utf-8", errors="replace").strip()
    if not text.startswith("data:"):
        return False
    try:
        event = json.loads(text[len("data:") :].strip())
    except json.JSONDecodeError:
        return False
    if not isinstance(event, dict):
        return False
    kind = event.get("type")
    if kind == "content_block_delta":
        delta = event.get("delta")
        return isinstance(delta, dict) and any(
            delta.get(field) for field in ("text", "thinking", "partial_json")
        )
    if (
        isinstance(kind, str)
        and kind.startswith("response.")
        and kind.endswith(".delta")
    ):
        return bool(event.get("delta"))
    for choice in event.get("choices") or []:
        delta = choice.get("delta") if isinstance(choice, dict) else None
        if not isinstance(delta, dict):
            continue
        for field in ("content", "reasoning", "reasoning_content", "refusal"):
            if delta.get(field):
                return True
        for detail in delta.get("reasoning_details") or []:
            if isinstance(detail, dict) and (
                detail.get("text") or detail.get("summary")
            ):
                return True
        if any(tool_call_has_arguments(call) for call in delta.get("tool_calls") or []):
            return True
    return False


def tool_call_has_arguments(call: Any) -> bool:
    """A tool call delta counts as content once it carries arguments; its
    bare start (id and name only) does not."""
    function = call.get("function") if isinstance(call, dict) else None
    return isinstance(function, dict) and bool(function.get("arguments"))


def should_inject_fault(
    shape: RequestShape, policy: FaultPolicy, seen_failures: set[str]
) -> bool:
    if not policy.fail_turns and not policy.fail_phases:
        return False
    if policy.fail_turns and shape.turn not in policy.fail_turns:
        return False
    if policy.fail_phases and shape.phase not in policy.fail_phases:
        return False
    if policy.fail_session_mod > 0:
        remainder = shape.session_index % policy.fail_session_mod
        if remainder != policy.fail_session_remainder:
            return False
    if not policy.fail_once_per_session:
        return True
    if shape.session_id in seen_failures:
        return False
    seen_failures.add(shape.session_id)
    return True


def main() -> int:
    args = parse_args()
    policy = parse_policy(args)
    handler = make_handler(args.upstream_base_url, policy, args.log_jsonl)
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), handler)
    print(
        json.dumps(
            {
                "listen": f"http://{args.listen_host}:{args.listen_port}",
                "upstream_base_url": args.upstream_base_url,
                "fail_turns": sorted(policy.fail_turns),
                "fail_phases": sorted(policy.fail_phases),
                "fail_status": policy.fail_status,
                "fail_session_mod": policy.fail_session_mod,
                "fail_session_remainder": policy.fail_session_remainder,
                "fail_once_per_session": policy.fail_once_per_session,
                "delay_ms": policy.delay_ms,
                "delay_jitter_ms": policy.delay_jitter_ms,
                "stream_fault": policy.stream_fault,
                "stall_seconds": policy.stall_seconds,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
