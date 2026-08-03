#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Bounded episode-affinity proxy for explicit retained-session replicas."""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

import httpx

STATS_SCHEMA = "rayline.vllm.episode-affinity-stats.v1"
FAILOVER_STATS_SCHEMA = "rayline.vllm.episode-affinity-failover-stats.v2"
SESSION_POOLING_PATH = "/v1/rayline/arc/session/pooling"
SESSION_CLOSE_PREFIX = "/v1/rayline/arc/session/"
STATE_PATH = "/health"
STATS_PATH = "/v1/rayline/affinity/stats"
RESET_PATH = "/v1/rayline/affinity/reset"
MAX_REQUEST_BYTES = 4 * 1024 * 1024
MAX_RESPONSE_BYTES = 512 * 1024
MAX_REPLICAS = 8
MIN_FAILOVER_REPLICAS = 2
EPISODE_HASH = re.compile(r"^[0-9a-f]{64}$")


class AffinityProxyError(RuntimeError):
    """A bounded proxy contract failed without exposing request payloads."""


@dataclass(frozen=True)
class ForwardedResponse:
    status_code: int
    content_type: str
    body: bytes


class AffinityState:
    """Own deterministic placement, upstream pooling, and aggregate counters."""

    def __init__(
        self,
        upstreams: list[str],
        *,
        modal_key: str,
        modal_secret: str,
        timeout_seconds: float,
        failover_after_pooling: int | None = None,
    ) -> None:
        if not 1 <= len(upstreams) <= MAX_REPLICAS:
            raise ValueError("affinity proxy requires between one and eight replicas")
        normalized: list[str] = []
        for value in upstreams:
            parsed = urlsplit(value.rstrip("/"))
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "affinity upstream must be an absolute HTTP(S) base URL"
                )
            normalized.append(value.rstrip("/"))
        if len(set(normalized)) != len(normalized):
            raise ValueError("affinity upstreams must be unique")
        if bool(modal_key) != bool(modal_secret):
            raise ValueError("affinity upstream credentials must be paired")
        if failover_after_pooling is not None and (
            failover_after_pooling < 1 or len(normalized) < MIN_FAILOVER_REPLICAS
        ):
            raise ValueError(
                "forced failover requires two replicas and a positive threshold"
            )
        self.upstreams = tuple(normalized)
        self._failover_after_pooling = failover_after_pooling
        self._headers = {"accept": "application/json"}
        if modal_key:
            self._headers.update({"Modal-Key": modal_key, "Modal-Secret": modal_secret})
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds, connect=10.0, pool=10.0),
            limits=httpx.Limits(
                max_connections=64,
                max_keepalive_connections=32,
                keepalive_expiry=60.0,
            ),
        )
        self._lock = threading.Lock()
        self._requests = [0 for _ in self.upstreams]
        self._session_pooling_requests = [0 for _ in self.upstreams]
        self._session_deletes = [0 for _ in self.upstreams]
        self._placement_by_episode: dict[str, int] = {}
        self._visited_by_episode: dict[str, set[int]] = {}
        self._pooling_count_by_episode: dict[str, int] = {}
        self._affinity_mismatches = 0
        self._failover_pooling_requests = 0
        self._failover_episodes: set[str] = set()
        self._close_fanout_requests = 0
        self._session_rebuild_responses = 0

    def close(self) -> None:
        self._client.close()

    def replica_for_hash(self, episode_hash: str) -> int:
        if not EPISODE_HASH.fullmatch(episode_hash):
            raise AffinityProxyError("episode hash is invalid")
        return int(episode_hash[:16], 16) % len(self.upstreams)

    def _observe(
        self,
        episode_hash: str,
        replica: int,
        *,
        deletion: bool,
        expected_remap: bool = False,
    ) -> None:
        with self._lock:
            previous = self._placement_by_episode.setdefault(episode_hash, replica)
            if previous != replica and not expected_remap:
                self._affinity_mismatches += 1
            self._visited_by_episode.setdefault(episode_hash, set()).add(replica)
            self._requests[replica] += 1
            if deletion:
                self._session_deletes[replica] += 1
                if self._failover_after_pooling is not None:
                    self._close_fanout_requests += 1
            else:
                self._session_pooling_requests[replica] += 1

    def _pooling_replica(self, episode_hash: str) -> tuple[int, bool]:
        primary = self.replica_for_hash(episode_hash)
        with self._lock:
            count = self._pooling_count_by_episode.get(episode_hash, 0) + 1
            self._pooling_count_by_episode[episode_hash] = count
            failed_over = (
                self._failover_after_pooling is not None
                and count > self._failover_after_pooling
            )
            if failed_over:
                self._failover_pooling_requests += 1
                self._failover_episodes.add(episode_hash)
        if failed_over:
            return (primary + 1) % len(self.upstreams), True
        return primary, False

    def _forward(
        self,
        replica: int,
        method: str,
        path: str,
        body: bytes | None = None,
    ) -> ForwardedResponse:
        headers = dict(self._headers)
        if body is not None:
            headers["content-type"] = "application/json"
        try:
            response = self._client.request(
                method,
                self.upstreams[replica] + path,
                content=body,
                headers=headers,
            )
        except httpx.HTTPError as error:
            raise AffinityProxyError("upstream transport failed") from error
        content = response.content
        if len(content) > MAX_RESPONSE_BYTES:
            raise AffinityProxyError("upstream response exceeded the bounded limit")
        return ForwardedResponse(
            response.status_code,
            response.headers.get("content-type", "application/json"),
            content,
        )

    def forward_session_pooling(self, body: bytes) -> ForwardedResponse:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as error:
            raise AffinityProxyError("session request is not JSON") from error
        if not isinstance(payload, dict):
            raise AffinityProxyError("session request must be an object")
        episode_hash = payload.get("episode_id_hash")
        if not isinstance(episode_hash, str):
            raise AffinityProxyError("session request omits episode hash")
        replica, failed_over = self._pooling_replica(episode_hash)
        self._observe(
            episode_hash,
            replica,
            deletion=False,
            expected_remap=failed_over,
        )
        response = self._forward(replica, "POST", SESSION_POOLING_PATH, body)
        if failed_over and response.status_code == HTTPStatus.OK:
            try:
                payload = json.loads(response.body)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and payload.get("session_action") == "created":
                with self._lock:
                    self._session_rebuild_responses += 1
        return response

    def forward_close(self, episode_hash: str) -> ForwardedResponse:
        if self._failover_after_pooling is None:
            replica = self.replica_for_hash(episode_hash)
            self._observe(episode_hash, replica, deletion=True)
            return self._forward(
                replica,
                "DELETE",
                SESSION_CLOSE_PREFIX + episode_hash,
            )
        with self._lock:
            replicas = sorted(self._visited_by_episode.get(episode_hash, set()))
        if not replicas:
            replicas = [self.replica_for_hash(episode_hash)]
        closed = False
        for replica in replicas:
            self._observe(
                episode_hash,
                replica,
                deletion=True,
                expected_remap=True,
            )
            response = self._forward(
                replica,
                "DELETE",
                SESSION_CLOSE_PREFIX + episode_hash,
            )
            if response.status_code != HTTPStatus.OK:
                return response
            try:
                payload = json.loads(response.body)
            except json.JSONDecodeError as error:
                raise AffinityProxyError("session close response is invalid") from error
            if not isinstance(payload, dict) or not isinstance(
                payload.get("closed"), bool
            ):
                raise AffinityProxyError("session close contract failed")
            closed = closed or payload["closed"]
        return self._json_response({"closed": closed})

    def aggregate_health(self) -> ForwardedResponse:
        with ThreadPoolExecutor(max_workers=len(self.upstreams)) as pool:
            responses = list(
                pool.map(
                    lambda replica: self._forward(replica, "GET", STATE_PATH),
                    range(len(self.upstreams)),
                )
            )
        decoded: list[dict[str, Any]] = []
        for response in responses:
            if response.status_code != HTTPStatus.OK:
                raise AffinityProxyError("an affinity replica is not healthy")
            try:
                value = json.loads(response.body)
            except json.JSONDecodeError as error:
                raise AffinityProxyError(
                    "an affinity health response is invalid"
                ) from error
            if not isinstance(value, dict) or value.get("status") != "ok":
                raise AffinityProxyError("an affinity health contract failed")
            decoded.append(value)
        capabilities = decoded[0].get("pooling_capabilities")
        if any(item.get("pooling_capabilities") != capabilities for item in decoded):
            raise AffinityProxyError("affinity replica capabilities differ")
        integer_fields = (
            "resident_sessions",
            "resident_tokens",
            "max_sessions",
            "max_resident_tokens",
        )
        totals: dict[str, int] = {}
        for field in integer_fields:
            values = [item.get(field) for item in decoded]
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in values
            ):
                raise AffinityProxyError("affinity health counts are invalid")
            totals[field] = sum(values)
        return self._json_response(
            {
                "status": "ok",
                **totals,
                "pooling_capabilities": capabilities,
            }
        )

    def stats(self) -> ForwardedResponse:
        with self._lock:
            unique = [0 for _ in self.upstreams]
            for replicas in self._visited_by_episode.values():
                for replica in replicas:
                    unique[replica] += 1
            primary = [0 for _ in self.upstreams]
            for replica in self._placement_by_episode.values():
                primary[replica] += 1
            payload = {
                "schema_version": (
                    FAILOVER_STATS_SCHEMA
                    if self._failover_after_pooling is not None
                    else STATS_SCHEMA
                ),
                "replicas": len(self.upstreams),
                "requests_by_replica": list(self._requests),
                "session_pooling_requests_by_replica": list(
                    self._session_pooling_requests
                ),
                "session_deletes_by_replica": list(self._session_deletes),
                "unique_sessions_by_replica": unique,
                "affinity_mismatches": self._affinity_mismatches,
            }
            if self._failover_after_pooling is not None:
                payload.update(
                    {
                        "primary_sessions_by_replica": primary,
                        "failover_after_pooling": self._failover_after_pooling,
                        "failover_pooling_requests": self._failover_pooling_requests,
                        "failover_sessions": len(self._failover_episodes),
                        "close_fanout_requests": self._close_fanout_requests,
                        "session_rebuild_responses": self._session_rebuild_responses,
                    }
                )
        return self._json_response(payload)

    def reset_stats(self) -> ForwardedResponse:
        with self._lock:
            self._requests = [0 for _ in self.upstreams]
            self._session_pooling_requests = [0 for _ in self.upstreams]
            self._session_deletes = [0 for _ in self.upstreams]
            self._placement_by_episode.clear()
            self._visited_by_episode.clear()
            self._pooling_count_by_episode.clear()
            self._affinity_mismatches = 0
            self._failover_pooling_requests = 0
            self._failover_episodes.clear()
            self._close_fanout_requests = 0
            self._session_rebuild_responses = 0
        return self._json_response({"status": "reset"})

    @staticmethod
    def _json_response(payload: dict[str, Any]) -> ForwardedResponse:
        return ForwardedResponse(
            HTTPStatus.OK,
            "application/json",
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
        )


class AffinityRequestHandler(BaseHTTPRequestHandler):
    server: AffinityHTTPServer

    def log_message(self, _format: str, *_args: object) -> None:
        """Never log paths, request bodies, identities, or credentials."""

    def _write(self, response: ForwardedResponse) -> None:
        self.send_response(response.status_code)
        self.send_header("content-type", response.content_type)
        self.send_header("content-length", str(len(response.body)))
        self.end_headers()
        self.wfile.write(response.body)

    def _error(self, status: HTTPStatus, code: str) -> None:
        self._write(
            ForwardedResponse(
                status,
                "application/json",
                json.dumps({"error": code}, separators=(",", ":")).encode(),
            )
        )

    def do_GET(self) -> None:
        try:
            if self.path == STATE_PATH:
                self._write(self.server.state.aggregate_health())
            elif self.path == STATS_PATH:
                self._write(self.server.state.stats())
            else:
                self._error(HTTPStatus.NOT_FOUND, "not_found")
        except AffinityProxyError:
            self._error(HTTPStatus.BAD_GATEWAY, "upstream")

    def do_POST(self) -> None:
        if self.path == RESET_PATH:
            self._write(self.server.state.reset_stats())
            return
        if self.path != SESSION_POOLING_PATH:
            self._error(HTTPStatus.NOT_FOUND, "not_found")
            return
        raw_length = self.headers.get("content-length", "")
        try:
            length = int(raw_length)
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "content_length")
            return
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request_limit")
            return
        body = self.rfile.read(length)
        try:
            self._write(self.server.state.forward_session_pooling(body))
        except AffinityProxyError:
            self._error(HTTPStatus.BAD_REQUEST, "invalid_request")

    def do_DELETE(self) -> None:
        if not self.path.startswith(SESSION_CLOSE_PREFIX):
            self._error(HTTPStatus.NOT_FOUND, "not_found")
            return
        episode_hash = self.path.removeprefix(SESSION_CLOSE_PREFIX)
        try:
            self._write(self.server.state.forward_close(episode_hash))
        except AffinityProxyError:
            self._error(HTTPStatus.BAD_REQUEST, "invalid_request")


class AffinityHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], state: AffinityState) -> None:
        self.state = state
        super().__init__(address, AffinityRequestHandler)

    def server_close(self) -> None:
        try:
            self.state.close()
        finally:
            super().server_close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--upstream", action="append", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--failover-after-pooling", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    key = os.environ.get("RAYLINE_ARC_MODAL_KEY", "")
    secret = os.environ.get("RAYLINE_ARC_MODAL_SECRET", "")
    if not key or not secret:
        raise SystemExit("affinity proxy requires protected endpoint credentials")
    state = AffinityState(
        args.upstream,
        modal_key=key,
        modal_secret=secret,
        timeout_seconds=args.timeout_seconds,
        failover_after_pooling=args.failover_after_pooling,
    )
    server = AffinityHTTPServer((args.listen_host, args.port), state)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
