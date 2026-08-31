# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError
from rayline_arc_io.constants import EOS_TOKEN_ID
from rayline_arc_io.schemas import (
    ArcPoolingRequest,
    ArcPoolingResponse,
    ArcSessionPoolingRequest,
    ArcSessionPoolingResponse,
)

SERIALIZED_TOKENS = 16


def valid_request() -> dict:
    return {
        "schema_version": "rayline.arc.pooling-request.v1",
        "serializer_version": "mtrouter-token-blocks-v2",
        "serving_rung": "A",
        "episode_id_hash": "a" * 64,
        "turns": [{"role": "user", "text": "hello"}],
    }


def test_request_is_strict_and_explicit() -> None:
    request = ArcPoolingRequest.model_validate(valid_request())
    assert request.serving_rung == "A"
    assert request.turns[0].text == "hello"
    rung_b = valid_request()
    rung_b["serving_rung"] = "B"
    assert ArcPoolingRequest.model_validate(rung_b).serving_rung == "B"

    for field in (
        "schema_version",
        "serializer_version",
        "serving_rung",
        "episode_id_hash",
    ):
        invalid = valid_request()
        del invalid[field]
        with pytest.raises(ValidationError):
            ArcPoolingRequest.model_validate(invalid)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("episode_id_hash", "A" * 64),
        ("serving_rung", "C"),
        ("serializer_version", "other"),
        ("turns", []),
        ("turns", [{"role": "tool", "text": "no"}]),
        ("turns", [{"role": "user", "text": 7}]),
    ],
)
def test_request_rejects_contract_drift(field: str, value: object) -> None:
    invalid = valid_request()
    invalid[field] = value
    with pytest.raises(ValidationError):
        ArcPoolingRequest.model_validate(invalid)


def test_request_rejects_unknown_fields() -> None:
    invalid = valid_request()
    invalid["model"] = "untrusted"
    with pytest.raises(ValidationError):
        ArcPoolingRequest.model_validate(invalid)


def test_response_exposes_the_full_readiness_contract() -> None:
    response = ArcPoolingResponse(
        embedding=[0.0] * 1024,
        serialized_tokens=SERIALIZED_TOKENS,
        full_history_tokens=SERIALIZED_TOKENS,
        truncated_tokens=0,
        cached_prefix_tokens=0,
        serializer_version="mtrouter-token-blocks-v2",
        model="Qwen/Qwen3.5-0.8B",
        model_revision="model-revision",
        tokenizer_revision="tokenizer-revision",
        tokenizer_sha256="a" * 64,
        eos_token_id=EOS_TOKEN_ID,
        engine_build_id="vllm@immutable-build",
        io_plugin_version="rayline-arc-io@0.1.0",
        pooling_capabilities=["all_plugin_mean"],
    )
    assert response.tokenizer_sha256 == "a" * 64
    assert response.eos_token_id == EOS_TOKEN_ID


def test_session_schema_is_versioned_and_rung_b_only() -> None:
    request = ArcSessionPoolingRequest.model_validate(
        {
            **valid_request(),
            "schema_version": "rayline.arc.session-pooling-request.v1",
            "serving_rung": "B",
        }
    )
    assert request.serving_rung == "B"

    with pytest.raises(ValidationError):
        ArcSessionPoolingRequest.model_validate(
            {
                **valid_request(),
                "schema_version": "rayline.arc.session-pooling-request.v1",
                "serving_rung": "A",
            }
        )


def test_session_response_exposes_retained_state_accounting() -> None:
    response = ArcSessionPoolingResponse(
        schema_version="rayline.arc.session-pooling-response.v1",
        embedding=[0.0] * 1024,
        serialized_tokens=16,
        full_history_tokens=16,
        truncated_tokens=0,
        retained_prefix_tokens=12,
        appended_tokens=4,
        session_action="appended",
        session_revision=2,
        serializer_version="mtrouter-token-blocks-v2",
        model="Qwen/Qwen3.5-0.8B",
        model_revision="model-revision",
        tokenizer_revision="tokenizer-revision",
        tokenizer_sha256="a" * 64,
        eos_token_id=EOS_TOKEN_ID,
        engine_build_id="vllm@immutable-build",
        io_plugin_version="rayline-arc-io@0.1.0",
        pooling_capabilities=["chunked_causal_mean", "resumable_causal_mean"],
    )
    assert (
        response.retained_prefix_tokens + response.appended_tokens == SERIALIZED_TOKENS
    )
