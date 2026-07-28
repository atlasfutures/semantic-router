# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from rayline_arc_io.schemas import ArcTurn
from rayline_arc_io.serializer import TokenBlockSerializer

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_PATH = (
    REPO_ROOT
    / "src"
    / "semantic-router"
    / "pkg"
    / "selection"
    / "raylinearc"
    / "testdata"
    / "token_block_goldens.v1.json"
)


class GoldenTokenizer:
    """Replay authoritative block encodings without network or local caches."""

    def __init__(self, encodings: dict[str, list[int]]) -> None:
        self.encodings = encodings

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return self.encodings[text]


def _tokenizer_for_case(
    case: dict,
    eos_token_id: int,
    context_header_ids: list[int],
) -> GoldenTokenizer:
    turns = case["turns"]
    expected = case["expected"]
    task_is_first = bool(turns) and turns[0]["role"] == "user"
    task_text = turns[0]["text"] if task_is_first else ""
    context_turns = turns[1:] if task_is_first else turns
    described_full_tokens = len(expected["task_ids"])
    described_full_tokens += len(expected["context_header_ids"])
    described_full_tokens += sum(
        len(block["header_ids"]) + len(block["content_ids"]) + 1
        for block in expected["history_blocks"]
    )
    assert described_full_tokens == expected["full_tokens"]
    encodings = {
        "[Task]\n" + task_text: expected["task_ids"][:-1],
        "[Context]": context_header_ids,
    }
    if context_turns:
        assert expected["context_header_ids"][-1] == eos_token_id
        assert expected["context_header_ids"][:-1] == context_header_ids

    turn_number = 0
    for turn, block in zip(
        context_turns,
        expected["history_blocks"],
        strict=True,
    ):
        if turn["role"] == "user":
            turn_number += 1
        encodings[f"[Turn {turn_number} - {turn['role']}]\n"] = block["header_ids"]
        encodings[turn["text"]] = block["content_ids"]
    return GoldenTokenizer(encodings)


def test_all_authoritative_token_block_goldens() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text())
    context_header_ids = next(
        case["expected"]["context_header_ids"][:-1]
        for case in fixture["cases"]
        if case["expected"]["context_header_ids"]
    )

    for case in fixture["cases"]:
        tokenizer = _tokenizer_for_case(
            case,
            fixture["provenance"]["eos_token_id"],
            context_header_ids,
        )
        serializer = TokenBlockSerializer(
            tokenizer,
            eos_token_id=fixture["provenance"]["eos_token_id"],
            max_tokens=case["limits"]["max_tokens"],
            min_recent_turns=case["limits"]["min_recent_turns"],
            min_recent_tokens=case["limits"]["min_recent_tokens"],
        )
        turns = [ArcTurn.model_validate(turn) for turn in case["turns"]]
        result = serializer.tokenize(turns)
        assert list(result.input_ids) == case["expected"]["input_ids"], case["id"]
        assert result.full_tokens == case["expected"]["full_tokens"], case["id"]
        assert result.truncated_tokens == case["expected"]["truncated_tokens"], case[
            "id"
        ]
