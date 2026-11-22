import importlib.util
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

HTTP_OK = 200
HTTP_UNAVAILABLE = 503
DELAY_MS = 100
MIN_DELAY_SECONDS = 0.09


def load_fault_proxy_module():
    path = Path(__file__).with_name("openai_fault_proxy.py")
    spec = importlib.util.spec_from_file_location("openai_fault_proxy", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class UpstreamEchoHandler(BaseHTTPRequestHandler):
    seen_paths: ClassVar[list[str]] = []

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length)
        json.loads(raw.decode("utf-8"))
        UpstreamEchoHandler.seen_paths.append(self.path)
        payload = {
            "id": "resp_echo",
            "model": "small",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {},
        }
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(HTTP_OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _fmt, *_args):
        return


def start_server(handler_cls):
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def stop_server(server, thread):
    server.shutdown()
    thread.join(timeout=2)


def post_turn(base_url, session_idx, turn, prompt_suffix=""):
    prompt = f"Session {session_idx}, turn {turn}. {prompt_suffix or 'continue'}"
    payload = {"model": "auto", "messages": [{"role": "user", "content": prompt}]}
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-session-id": f"session-{session_idx}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, dict(response.headers), json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), json.loads(exc.read())


def test_fault_proxy_injects_selected_turn_once_then_recovers(tmp_path):
    fault = load_fault_proxy_module()
    UpstreamEchoHandler.seen_paths = []
    upstream, upstream_thread = start_server(UpstreamEchoHandler)
    policy = fault.FaultPolicy(
        fail_turns=frozenset({1}),
        fail_phases=frozenset(),
        fail_status=HTTP_UNAVAILABLE,
        fail_session_mod=1,
        fail_session_remainder=0,
        fail_once_per_session=True,
    )
    handler = fault.make_handler(
        f"http://127.0.0.1:{upstream.server_port}",
        policy,
        tmp_path / "faults.jsonl",
    )
    proxy, proxy_thread = start_server(handler)

    try:
        base_url = f"http://127.0.0.1:{proxy.server_port}"
        assert post_turn(base_url, 0, 0)[0] == HTTP_OK

        status, headers, payload = post_turn(base_url, 0, 1)
        assert status == HTTP_UNAVAILABLE
        assert headers["x-vsr-fault-injected"] == "true"
        assert payload["error"]["type"] == "fault_proxy_injected"

        assert post_turn(base_url, 0, 1)[0] == HTTP_OK
        assert post_turn(base_url, 0, 2)[0] == HTTP_OK
    finally:
        stop_server(proxy, proxy_thread)
        stop_server(upstream, upstream_thread)

    events = [
        json.loads(line)
        for line in (tmp_path / "faults.jsonl").read_text().splitlines()
    ]
    assert [event["action"] for event in events] == [
        "forwarded",
        "injected",
        "forwarded",
        "forwarded",
    ]
    assert UpstreamEchoHandler.seen_paths == [
        "/v1/chat/completions",
        "/v1/chat/completions",
        "/v1/chat/completions",
    ]


def test_fault_proxy_session_modulo_selects_subset():
    fault = load_fault_proxy_module()
    upstream, upstream_thread = start_server(UpstreamEchoHandler)
    policy = fault.FaultPolicy(
        fail_turns=frozenset({0}),
        fail_phases=frozenset(),
        fail_status=HTTP_UNAVAILABLE,
        fail_session_mod=2,
        fail_session_remainder=1,
        fail_once_per_session=True,
    )
    handler = fault.make_handler(f"http://127.0.0.1:{upstream.server_port}", policy)
    proxy, proxy_thread = start_server(handler)

    try:
        base_url = f"http://127.0.0.1:{proxy.server_port}"
        assert post_turn(base_url, 0, 0)[0] == HTTP_OK
        assert post_turn(base_url, 1, 0)[0] == HTTP_UNAVAILABLE
    finally:
        stop_server(proxy, proxy_thread)
        stop_server(upstream, upstream_thread)


def test_fault_proxy_delays_forwarded_and_injected_responses():
    fault = load_fault_proxy_module()
    upstream, upstream_thread = start_server(UpstreamEchoHandler)
    policy = fault.FaultPolicy(
        fail_turns=frozenset({1}),
        fail_phases=frozenset(),
        fail_status=HTTP_UNAVAILABLE,
        fail_session_mod=1,
        fail_session_remainder=0,
        fail_once_per_session=True,
        delay_ms=DELAY_MS,
    )
    handler = fault.make_handler(f"http://127.0.0.1:{upstream.server_port}", policy)
    proxy, proxy_thread = start_server(handler)

    try:
        base_url = f"http://127.0.0.1:{proxy.server_port}"
        started = time.monotonic()
        forwarded_status = post_turn(base_url, 0, 0)[0]
        forwarded_elapsed = time.monotonic() - started

        started = time.monotonic()
        injected_status, injected_headers, injected_payload = post_turn(base_url, 0, 1)
        injected_elapsed = time.monotonic() - started
    finally:
        stop_server(proxy, proxy_thread)
        stop_server(upstream, upstream_thread)

    assert forwarded_status == HTTP_OK
    assert forwarded_elapsed >= MIN_DELAY_SECONDS
    assert injected_status == HTTP_UNAVAILABLE
    assert injected_headers["x-vsr-fault-injected"] == "true"
    assert injected_payload["error"]["type"] == "fault_proxy_injected"
    assert injected_elapsed >= MIN_DELAY_SECONDS


def test_fault_proxy_delay_seconds_defaults_to_no_sleep():
    fault = load_fault_proxy_module()
    policy = fault.FaultPolicy(
        fail_turns=frozenset(),
        fail_phases=frozenset(),
        fail_status=HTTP_UNAVAILABLE,
        fail_session_mod=1,
        fail_session_remainder=0,
        fail_once_per_session=True,
    )
    assert policy.delay_ms == 0
    assert policy.delay_jitter_ms == 0
    assert fault.delay_seconds(policy) == 0.0

    jittered = fault.FaultPolicy(
        fail_turns=frozenset(),
        fail_phases=frozenset(),
        fail_status=HTTP_UNAVAILABLE,
        fail_session_mod=1,
        fail_session_remainder=0,
        fail_once_per_session=True,
        delay_ms=DELAY_MS,
        delay_jitter_ms=DELAY_MS,
    )
    for _ in range(20):
        seconds = fault.delay_seconds(jittered)
        assert MIN_DELAY_SECONDS <= seconds <= 2 * DELAY_MS / 1000.0


def test_fault_proxy_phase_inference_targets_tool_loops():
    fault = load_fault_proxy_module()
    assert fault.phase_from_text("Use the provided tool result now") == "tool_loop"
    assert fault.phase_from_text("Continue from the previous response state") == (
        "provider_state"
    )
    assert fault.phase_from_text("This follows an idle pause") == "idle_boundary"


SSE_EVENTS = [
    b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
    b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n',
    b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
    b"data: [DONE]\n\n",
]
STALL_SECONDS = 0.6
READ_TIMEOUT_SECONDS = 0.3


class UpstreamSSEHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    calls: ClassVar[int] = 0

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(content_length)
        UpstreamSSEHandler.calls += 1
        self.close_connection = True
        self.send_response(HTTP_OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        for event in SSE_EVENTS:
            self.wfile.write(event)
            self.wfile.flush()

    def log_message(self, _fmt, *_args):
        return


def stream_proxy(tmp_path, mode):
    fault = load_fault_proxy_module()
    UpstreamSSEHandler.calls = 0
    upstream, upstream_thread = start_server(UpstreamSSEHandler)
    policy = fault.FaultPolicy(
        fail_turns=frozenset(),
        fail_phases=frozenset(),
        fail_status=HTTP_UNAVAILABLE,
        fail_session_mod=1,
        fail_session_remainder=0,
        fail_once_per_session=True,
        stream_fault=mode,
        stall_seconds=STALL_SECONDS,
    )
    handler = fault.make_handler(
        f"http://127.0.0.1:{upstream.server_port}", policy, tmp_path / "faults.jsonl"
    )
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    return (upstream, upstream_thread), (proxy, proxy_thread)


def post_stream(base_url, messages):
    payload = {"model": "auto", "stream": True, "messages": messages}
    request = urllib.request.Request(
        f"{base_url}/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    response = urllib.request.urlopen(request, timeout=READ_TIMEOUT_SECONDS)
    status = response.status
    chunks, timed_out = [], False
    try:
        while chunk := response.read1(65536):
            chunks.append(chunk)
    except (TimeoutError, OSError):
        timed_out = True
    response.close()
    return status, b"".join(chunks), timed_out


def stream_events(tmp_path):
    return [
        json.loads(line)
        for line in (tmp_path / "faults.jsonl").read_text().splitlines()
    ]


def run_stream_turns(tmp_path, mode, calls):
    upstream, proxy = stream_proxy(tmp_path, mode)
    turn = [{"role": "user", "content": "Session 0, turn 1."}]
    try:
        base_url = f"http://127.0.0.1:{proxy[0].server_port}"
        results = [post_stream(base_url, turn) for _ in range(calls)]
    finally:
        stop_server(*proxy)
        stop_server(*upstream)
    return results


def test_stream_fault_stall_once_stalls_first_call_then_passes(tmp_path):
    results = run_stream_turns(tmp_path, "stall-once", 2)
    status, body, timed_out = results[0]
    fault = load_fault_proxy_module()
    # One SSE comment, then silence: the comment starts a router's
    # first-content timer, and no content ever follows.
    assert (status, body, timed_out) == (HTTP_OK, fault.STALL_COMMENT, True)
    assert results[1] == (HTTP_OK, b"".join(SSE_EVENTS), False)
    assert UpstreamSSEHandler.calls == 1
    events = stream_events(tmp_path)
    assert [(e["action"], e["call"]) for e in events] == [
        ("stalled", 1),
        ("forwarded_stream", 2),
    ]
    assert events[0]["key"] == events[1]["key"]


def test_stream_fault_stall_twice_stalls_first_two_calls(tmp_path):
    results = run_stream_turns(tmp_path, "stall-twice", 3)
    assert [r[2] for r in results] == [True, True, False]
    assert UpstreamSSEHandler.calls == 1
    assert [e["action"] for e in stream_events(tmp_path)] == [
        "stalled",
        "stalled",
        "forwarded_stream",
    ]


def test_stream_fault_cut_after_content_ends_before_done(tmp_path):
    results = run_stream_turns(tmp_path, "cut-after-content", 2)
    assert results[0] == (HTTP_OK, b"".join(SSE_EVENTS[:2]), False)
    assert results[1] == (HTTP_OK, b"".join(SSE_EVENTS), False)
    assert [e["action"] for e in stream_events(tmp_path)] == [
        "cut_after_content",
        "forwarded_stream",
    ]


def test_stream_fault_key_follows_the_conversation():
    fault = load_fault_proxy_module()
    first = {"messages": [{"role": "user", "content": "a"}], "provider": {}}
    resent = {
        "messages": [{"role": "user", "content": "a"}],
        "provider": {"ignore": ["x"]},
    }
    next_turn = {
        "messages": [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
    }
    assert fault.stream_fault_key(first) == fault.stream_fault_key(resent)
    assert fault.stream_fault_key(first) != fault.stream_fault_key(next_turn)
    responses_a = {"input": "hi", "previous_response_id": "resp_a"}
    responses_b = {"input": "hi", "previous_response_id": "resp_b"}
    assert fault.stream_fault_key(responses_a) != fault.stream_fault_key(responses_b)


def test_sse_line_has_content_by_wire():
    fault = load_fault_proxy_module()
    assert not fault.sse_line_has_content(SSE_EVENTS[0])
    assert fault.sse_line_has_content(SSE_EVENTS[1])
    assert not fault.sse_line_has_content(b"data: [DONE]\n")
    assert not fault.sse_line_has_content(b": keep-alive\n")
    assert fault.sse_line_has_content(
        b'data: {"type":"content_block_delta","delta":{"text":"x"}}\n'
    )
    assert not fault.sse_line_has_content(b'data: {"type":"message_start"}\n')
    bare_tool_start = (
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1",'
        b'"function":{"name":"f","arguments":""}}]}}]}\n'
    )
    assert not fault.sse_line_has_content(bare_tool_start)
    assert fault.sse_line_has_content(
        b'data: {"choices":[{"delta":{"refusal":"no"}}]}\n'
    )
    assert fault.sse_line_has_content(
        b'data: {"choices":[{"delta":{"reasoning_details":'
        b'[{"type":"reasoning.text","text":"t"}]}}]}\n'
    )
    assert fault.sse_line_has_content(
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
        b'"function":{"arguments":"{\\"a\\""}}]}}]}\n'
    )
    assert not fault.sse_line_has_content(
        b'data: {"type":"content_block_delta","delta":{"partial_json":""}}\n'
    )
    assert fault.sse_line_has_content(
        b'data: {"type":"response.output_text.delta","delta":"x"}\n'
    )
