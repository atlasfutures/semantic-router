# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError
from rayline_arc_io.schemas import ArcPoolingRequest


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

    for field in ("serializer_version", "serving_rung", "episode_id_hash"):
        invalid = valid_request()
        del invalid[field]
        with pytest.raises(ValidationError):
            ArcPoolingRequest.model_validate(invalid)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("episode_id_hash", "A" * 64),
        ("serving_rung", "B"),
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
