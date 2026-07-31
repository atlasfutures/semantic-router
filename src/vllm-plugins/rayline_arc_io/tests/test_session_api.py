# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from http import HTTPStatus

from fastapi.testclient import TestClient
from rayline_arc_io.schemas import ArcTurn
from rayline_arc_io.serializer import TokenBlockSerializer
from rayline_arc_io.session_api import SessionAPIMetadata, create_session_app
from rayline_arc_io.session_coordinator import (
    RetainedPoolingOutput,
    SessionCoordinator,
)

REBUILT_BACKEND_COUNT = 2


class TinyTokenizer:
    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        split_special_tokens: bool,
    ) -> list[int]:
        assert add_special_tokens is False
        assert split_special_tokens is True
        return [ord(character) for character in text]


@dataclass
class FakeBackend:
    cumulative: list[int] = field(default_factory=list)
    closed: bool = False

    async def append(self, token_ids: tuple[int, ...]) -> RetainedPoolingOutput:
        self.cumulative.extend(token_ids)
        return RetainedPoolingOutput(
            embedding=tuple([1.0] + [0.0] * 1023),
            cumulative_token_ids=tuple(self.cumulative),
        )

    async def close(self) -> None:
        self.closed = True


class FakeFactory:
    def __init__(self) -> None:
        self.backends: list[FakeBackend] = []

    def __call__(self, _episode_id_hash: str) -> FakeBackend:
        backend = FakeBackend()
        self.backends.append(backend)
        return backend


def request_body(turns: list[dict[str, str]]) -> dict:
    return {
        "schema_version": "rayline.arc.session-pooling-request.v1",
        "serializer_version": "mtrouter-token-blocks-v2",
        "serving_rung": "B",
        "episode_id_hash": "a" * 64,
        "turns": turns,
    }


def build_client() -> tuple[TestClient, FakeFactory]:
    factory = FakeFactory()
    coordinator = SessionCoordinator(
        factory,
        max_sessions=2,
        max_resident_tokens=10_000,
        idle_ttl_seconds=60,
    )
    app = create_session_app(
        coordinator,
        TokenBlockSerializer(TinyTokenizer(), max_tokens=4096),
        SessionAPIMetadata(engine_build_id="vllm@retained-session-test"),
    )
    return TestClient(app), factory


def test_session_http_contract_appends_only_exact_suffix() -> None:
    client, factory = build_client()
    first_turns = [{"role": "user", "text": "task"}]
    second_turns = [
        *first_turns,
        {"role": "assistant", "text": "answer"},
    ]

    first = client.post(
        "/v1/rayline/arc/session/pooling",
        json=request_body(first_turns),
    )
    second = client.post(
        "/v1/rayline/arc/session/pooling",
        json=request_body(second_turns),
    )

    assert first.status_code == HTTPStatus.OK
    assert second.status_code == HTTPStatus.OK
    first_data = first.json()
    second_data = second.json()
    assert first_data["schema_version"] == ("rayline.arc.session-pooling-response.v1")
    assert first_data["session_action"] == "created"
    assert second_data["session_action"] == "appended"
    assert second_data["retained_prefix_tokens"] == first_data["serialized_tokens"]
    assert (
        second_data["retained_prefix_tokens"] + second_data["appended_tokens"]
        == second_data["serialized_tokens"]
    )
    assert second_data["pooling_capabilities"] == [
        "chunked_causal_mean",
        "resumable_causal_mean",
    ]
    assert len(factory.backends) == 1


def test_session_http_contract_rebuilds_after_non_prefix_history() -> None:
    client, factory = build_client()
    first_turns = [{"role": "user", "text": "task"}]
    changed_turns = [{"role": "user", "text": "different"}]
    assert (
        client.post(
            "/v1/rayline/arc/session/pooling",
            json=request_body(first_turns),
        ).status_code
        == HTTPStatus.OK
    )

    response = client.post(
        "/v1/rayline/arc/session/pooling",
        json=request_body(changed_turns),
    )

    assert response.status_code == HTTPStatus.OK
    assert response.json()["session_action"] == "rebuilt"
    assert response.json()["retained_prefix_tokens"] == 0
    assert len(factory.backends) == REBUILT_BACKEND_COUNT
    assert factory.backends[0].closed is True


def test_validation_error_never_echoes_request_text() -> None:
    client, _ = build_client()
    secret_text = "never-echo-this-prompt"
    invalid = request_body([{"role": "user", "text": secret_text}])
    invalid["serving_rung"] = "A"

    response = client.post(
        "/v1/rayline/arc/session/pooling",
        json=invalid,
    )

    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert secret_text not in response.text
    assert response.json()["error"] == "invalid_request"


def test_health_and_explicit_close_are_bounded() -> None:
    client, factory = build_client()
    turns = [{"role": "user", "text": "task"}]
    assert (
        client.post(
            "/v1/rayline/arc/session/pooling",
            json=request_body(turns),
        ).status_code
        == HTTPStatus.OK
    )

    health = client.get("/health")
    closed = client.delete("/v1/rayline/arc/session/" + "a" * 64)
    health_after = client.get("/health")

    assert health.json()["resident_sessions"] == 1
    assert closed.json() == {"closed": True}
    assert health_after.json()["resident_sessions"] == 0
    assert factory.backends[0].closed is True


def test_request_schema_remains_compatible_with_arc_turn_model() -> None:
    assert ArcTurn(role="user", text="task").role == "user"
