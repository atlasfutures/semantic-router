#!/usr/bin/env python3
"""Contract-faithful fake encoder and provider for ARC stack acceptance."""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
ENGINE_BUILD_ID = "vllm@public-rayline-e2e-build"
PLUGIN_VERSION = "rayline-arc-io@0.1.0"
SERIALIZER_VERSION = "mtrouter-token-blocks-v2"
REQUEST_SCHEMA_VERSION = "rayline.arc.pooling-request.v1"
EMBEDDING_DIMENSION = 1024
EPISODE_HASH_LENGTH = 64
PROVIDER_API_KEY = "public-e2e-provider-key"


def _validated_pooling_data(body: dict[str, Any]) -> dict[str, Any]:
    """Enforce the plugin's strict request contract like the real endpoint."""
    if body.get("task", "plugin") != "plugin":
        raise ValueError("task must be 'plugin'")
    data = body["data"]
    if not isinstance(data, dict):
        raise ValueError("data must be an object")
    allowed = {
        "schema_version",
        "serializer_version",
        "serving_rung",
        "episode_id_hash",
        "turns",
    }
    unexpected = set(data) - allowed
    if unexpected:
        raise ValueError(f"unexpected fields: {sorted(unexpected)}")
    if data["schema_version"] != REQUEST_SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    if data["serializer_version"] != SERIALIZER_VERSION:
        raise ValueError("unsupported serializer_version")
    if data["serving_rung"] != "B":
        raise ValueError("unexpected serving rung")
    episode_hash = data["episode_id_hash"]
    if (
        not isinstance(episode_hash, str)
        or len(episode_hash) != EPISODE_HASH_LENGTH
        or any(character not in "0123456789abcdef" for character in episode_hash)
    ):
        raise ValueError("episode_id_hash must be a lowercase SHA256 hex digest")
    _validate_turns(data["turns"])
    return data


def _validate_turns(turns: Any) -> None:
    if not isinstance(turns, list) or not turns:
        raise ValueError("turns must be a non-empty list")
    for turn in turns:
        if not isinstance(turn, dict) or set(turn) != {"role", "text"}:
            raise ValueError("turns must contain role/text objects")
        if turn["role"] not in ("user", "assistant"):
            raise ValueError("turn role must be user or assistant")
        if not isinstance(turn["text"], str):
            raise ValueError("turn text must be a string")


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode()


def _message_text(body: dict[str, Any]) -> str:
    parts: list[str] = []
    for message in body.get("messages", []):
        content = message.get("content", "")
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
        payload = _json_bytes(value)
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_health(self) -> None:
        self.send_json(200, {"status": "ok"})


class EncoderState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active_by_episode: dict[str, int] = {}
        self.active_global = 0
        self.max_same_episode = 0
        self.max_global = 0
        self.request_index = 0

    def next_request_index(self) -> int:
        with self.lock:
            self.request_index += 1
            return self.request_index

    def begin(self, episode_hash: str) -> None:
        with self.lock:
            active = self.active_by_episode.get(episode_hash, 0) + 1
            self.active_by_episode[episode_hash] = active
            self.active_global += 1
            self.max_same_episode = max(self.max_same_episode, active)
            self.max_global = max(self.max_global, self.active_global)

    def end(self, episode_hash: str) -> None:
        with self.lock:
            active = self.active_by_episode.get(episode_hash, 0) - 1
            if active <= 0:
                self.active_by_episode.pop(episode_hash, None)
            else:
                self.active_by_episode[episode_hash] = active
            self.active_global -= 1

    def reset(self) -> None:
        with self.lock:
            self.active_by_episode.clear()
            self.active_global = 0
            self.max_same_episode = 0
            self.max_global = 0

    def snapshot(self) -> dict[str, int]:
        with self.lock:
            return {
                "max_same_episode": self.max_same_episode,
                "max_global": self.max_global,
            }


ENCODER_STATE = EncoderState()


class EncoderHandler(QuietHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_health()
            return
        if self.path == "/stats":
            self.send_json(200, ENCODER_STATE.snapshot())
            return
        self.send_json(404, {"error": "not_found"})

    def send_vllm_error(self, status: int, message: str, err_type: str) -> None:
        # Mirror vLLM's ErrorResponse envelope so clients see the real shape.
        self.send_json(
            status,
            {"error": {"message": message, "type": err_type, "code": status}},
        )

    def do_POST(self) -> None:
        if self.path == "/reset":
            ENCODER_STATE.reset()
            self.send_json(200, {"status": "reset"})
            return
        if self.path != "/pooling":
            self.send_vllm_error(404, "Not Found", "NotFoundError")
            return
        if (
            self.headers.get("Modal-Key") != "public-e2e-modal-key"
            or self.headers.get("Modal-Secret") != "public-e2e-modal-secret"
        ):
            self.send_vllm_error(
                401, "modal-proxy: missing credentials", "Unauthorized"
            )
            return
        try:
            body = self.read_json()
            data = _validated_pooling_data(body)
            episode_hash = data["episode_id_hash"]
            turns = data["turns"]
            text = "\n".join(str(turn.get("text", "")) for turn in turns)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.send_vllm_error(400, str(error), "BadRequestError")
            return

        ENCODER_STATE.begin(episode_hash)
        try:
            if "ARC_DELAY" in text:
                time.sleep(0.4)
            if "ARC_ENCODER_FAIL" in text:
                self.send_vllm_error(503, "Service Unavailable", "InternalServerError")
                return
            sign = -1.0 if "ARC_ROUTE_B" in text else 1.0
            embedding = [sign] + [0.0] * (EMBEDDING_DIMENSION - 1)
            token_count = max(1, sum(len(str(turn.get("text", ""))) for turn in turns))
            self.send_json(
                200,
                {
                    # Mirror the vLLM IOProcessorResponse envelope exactly,
                    # including the engine-owned correlation fields.
                    "request_id": f"pool-{ENCODER_STATE.next_request_index()}",
                    "created_at": int(time.time()),
                    "data": {
                        "embedding": embedding,
                        "serialized_tokens": token_count,
                        "full_history_tokens": token_count,
                        "truncated_tokens": 0,
                        "cached_prefix_tokens": 0,
                        "serializer_version": SERIALIZER_VERSION,
                        "model": "Qwen/Qwen3.5-0.8B",
                        "model_revision": MODEL_REVISION,
                        "tokenizer_revision": MODEL_REVISION,
                        "tokenizer_sha256": TOKENIZER_SHA256,
                        "eos_token_id": 248046,
                        "engine_build_id": ENGINE_BUILD_ID,
                        "io_plugin_version": PLUGIN_VERSION,
                        "pooling_capabilities": ["chunked_causal_mean"],
                    },
                },
            )
        finally:
            ENCODER_STATE.end(episode_hash)


class ProviderState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: list[dict[str, Any]] = []

    def append(self, body: dict[str, Any]) -> int:
        with self.lock:
            self.requests.append(body)
            return len(self.requests)

    def reset(self) -> None:
        with self.lock:
            self.requests.clear()

    def snapshot(self) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.requests)


PROVIDER_STATE = ProviderState()


class ProviderHandler(QuietHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_health()
            return
        if self.path == "/observed":
            self.send_json(200, {"requests": PROVIDER_STATE.snapshot()})
            return
        self.send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path == "/reset":
            PROVIDER_STATE.reset()
            self.send_json(200, {"status": "reset"})
            return
        if self.headers.get("Authorization") != f"Bearer {PROVIDER_API_KEY}":
            # Prove the router injected exactly the artifact-owned credential
            # rather than forwarding a caller-supplied Authorization header.
            self.send_json(
                401,
                {
                    "error": {
                        "message": "missing or wrong provider credential",
                        "type": "AuthenticationError",
                        "code": 401,
                    }
                },
            )
            return
        try:
            body = self.read_json()
        except (ValueError, json.JSONDecodeError):
            self.send_json(400, {"error": "invalid_request"})
            return
        sequence = PROVIDER_STATE.append(body)
        text = _message_text(body)
        if "ARC_PROVIDER_DELAY" in text:
            time.sleep(1)
        if "ARC_PROVIDER_TRANSPORT" in text:
            self.close_connection = True
            return
        if "ARC_PROVIDER_5XX" in text:
            self.send_json(503, {"error": "synthetic_provider_failure"})
            return
        if "ARC_STREAM_ABORT" in text:
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("transfer-encoding", "chunked")
            self.send_header("x-e2e-provider-sequence", str(sequence))
            self.end_headers()
            payload = b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
            self.wfile.flush()
            self.close_connection = True
            return
        self.send_json(
            200,
            {
                "id": f"synthetic-{sequence}",
                "object": "chat.completion",
                "model": body.get("model", ""),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("service", choices=("encoder", "provider"))
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    handler = EncoderHandler if args.service == "encoder" else ProviderHandler
    server = ThreadingHTTPServer(("0.0.0.0", args.port), handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
