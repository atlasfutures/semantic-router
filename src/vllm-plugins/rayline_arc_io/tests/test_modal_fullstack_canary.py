# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
HTTP_PATH = REPO_ROOT / "e2e/testing/rayline-arc/modal_http.py"
WARMUP_PATH = REPO_ROOT / "e2e/testing/rayline-arc/modal_encoder_warmup.py"
EXPECTED_MAX_RESULT_REDIRECTS = 2
EXPECTED_HTTP_OK = 200


def _http_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("modal_http", HTTP_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResponse:
    def __init__(self, status: int, *, location: str = "", body: bytes = b"") -> None:
        self.status = status
        self.location = location
        self.body = body
        self.read_called = False

    def getheader(self, name: str, default: str = "") -> str:
        return self.location if name.lower() == "location" else default

    def read(self) -> bytes:
        self.read_called = True
        return self.body


class FakeConnection:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.requests: list[tuple[str, str, bytes | None, dict[str, str]]] = []
        self.closed = False

    def request(
        self,
        method: str,
        target: str,
        *,
        body: bytes | None,
        headers: dict[str, str],
    ) -> None:
        self.requests.append((method, target, body, headers))

    def getresponse(self) -> FakeResponse:
        return self.response

    def close(self) -> None:
        self.closed = True


def _fake_connection_factory(
    module: ModuleType,
    responses: list[FakeResponse],
) -> tuple[list[FakeConnection], Callable[[str, float], tuple[FakeConnection, str]]]:
    connections = [FakeConnection(response) for response in responses]
    pending = iter(connections)

    def connection(_url: str, _timeout: float) -> tuple[FakeConnection, str]:
        item = next(pending)
        parsed = module.urlparse(_url)
        return item, parsed.path.rstrip("/")

    return connections, connection


def test_result_redirect_is_bounded_same_origin_get() -> None:
    module = _http_module()
    assert module.MAX_RESULT_REDIRECTS == EXPECTED_MAX_RESULT_REDIRECTS
    redirect = FakeResponse(
        module.HTTP_SEE_OTHER,
        location="/v1/chat/completions?modal_result=public-test",
    )
    final = FakeResponse(EXPECTED_HTTP_OK)
    connections, connection_factory = _fake_connection_factory(
        module, [redirect, final]
    )
    headers = {"authorization": "Bearer public-test-only"}

    returned_connection, returned_response = module.request_following_result_redirects(
        connection_factory=connection_factory,
        method="POST",
        url="https://worker.example/v1/chat/completions",
        body=b"{}",
        headers=headers,
        timeout_seconds=1,
    )

    assert returned_connection is connections[1]
    assert returned_response is final
    assert connections[0].requests == [("POST", "/v1/chat/completions", b"{}", headers)]
    assert connections[1].requests == [
        (
            "GET",
            "/v1/chat/completions?modal_result=public-test",
            None,
            headers,
        )
    ]
    assert redirect.read_called
    assert connections[0].closed


def test_result_redirect_refuses_cross_origin_credentials() -> None:
    module = _http_module()
    redirect = FakeResponse(
        module.HTTP_SEE_OTHER,
        location="https://attacker.example/result",
    )
    connections, connection_factory = _fake_connection_factory(module, [redirect])

    with pytest.raises(RuntimeError, match="cross-origin"):
        module.request_following_result_redirects(
            connection_factory=connection_factory,
            method="POST",
            url="https://worker.example/v1/chat/completions",
            body=b"{}",
            headers={"authorization": "Bearer public-test-only"},
            timeout_seconds=1,
        )

    assert len(connections[0].requests) == 1
    assert connections[0].closed


def test_encoder_warmup_uses_protected_health_without_session() -> None:
    http_module = _http_module()
    sys.modules["modal_http"] = http_module
    spec = importlib.util.spec_from_file_location("modal_encoder_warmup", WARMUP_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    response = FakeResponse(
        EXPECTED_HTTP_OK,
        body=json.dumps(
            {
                "status": "ok",
                "pooling_capabilities": [
                    "resumable_causal_mean",
                    "chunked_causal_mean",
                ],
            }
        ).encode(),
    )
    connections, connection_factory = _fake_connection_factory(http_module, [response])

    result = module.warm_encoder(
        base_url="https://encoder.example",
        modal_key="public-modal-key",
        modal_secret="public-modal-secret",
        timeout_seconds=1,
        connection_factory=connection_factory,
    )

    assert result["status"] == "ok"
    assert connections[0].requests == [
        (
            "GET",
            "/health",
            None,
            {
                "Modal-Key": "public-modal-key",
                "Modal-Secret": "public-modal-secret",
            },
        )
    ]
    assert connections[0].closed
