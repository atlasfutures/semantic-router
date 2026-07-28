# SPDX-License-Identifier: Apache-2.0

"""Exact Python port of Rayline's mtrouter-token-blocks-v2 serializer."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .constants import (
    EOS_TOKEN_ID,
    MAX_SERIALIZED_TOKENS,
    MIN_RECENT_TOKENS,
    MIN_RECENT_TURNS,
    MIN_SERIALIZED_TOKENS,
)
from .schemas import ArcTurn


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


@dataclass(frozen=True)
class TokenizationResult:
    input_ids: tuple[int, ...]
    full_tokens: int
    truncated_tokens: int


@dataclass(frozen=True)
class _HistoryBlock:
    header_ids: tuple[int, ...]
    content_ids: tuple[int, ...]
    eos: int

    def input_ids(self) -> tuple[int, ...]:
        return (*self.header_ids, *self.content_ids, self.eos)

    def recent_tail(self, max_tokens: int) -> tuple[int, ...]:
        if max_tokens == 0:
            return ()
        if max_tokens == 1:
            return (self.eos,)
        if max_tokens <= len(self.header_ids) + 1:
            return (*self.header_ids[: max_tokens - 1], self.eos)

        content_budget = max_tokens - len(self.header_ids) - 1
        content_start = max(0, len(self.content_ids) - content_budget)
        return (*self.header_ids, *self.content_ids[content_start:], self.eos)


class TokenBlockSerializer:
    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        eos_token_id: int = EOS_TOKEN_ID,
        max_tokens: int = MAX_SERIALIZED_TOKENS,
        min_recent_turns: int = MIN_RECENT_TURNS,
        min_recent_tokens: int = MIN_RECENT_TOKENS,
    ) -> None:
        if eos_token_id < 0:
            raise ValueError("ARC tokenization requires a non-negative EOS token ID")
        if (
            max_tokens < MIN_SERIALIZED_TOKENS
            or min_recent_turns < 1
            or min_recent_tokens < 1
        ):
            raise ValueError("invalid ARC tokenization limits")
        self._tokenizer = tokenizer
        self._eos = eos_token_id
        self._max_tokens = max_tokens
        self._min_recent_turns = min_recent_turns
        self._min_recent_tokens = min_recent_tokens

    def _encode(self, text: str) -> tuple[int, ...]:
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        if not isinstance(token_ids, list) or any(
            not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0
            for token_id in token_ids
        ):
            raise TypeError("tokenizer returned invalid ARC token IDs")
        return tuple(token_ids)

    def _encode_block(self, text: str) -> tuple[int, ...]:
        return (*self._encode(text), self._eos)

    def _truncate_task(
        self, input_ids: tuple[int, ...], max_tokens: int
    ) -> tuple[int, ...]:
        if max_tokens == 0:
            return ()
        if len(input_ids) <= max_tokens:
            return input_ids
        if max_tokens == 1:
            return (self._eos,)
        return (*input_ids[: max_tokens - 1], self._eos)

    def tokenize(self, turns: Sequence[ArcTurn]) -> TokenizationResult:
        if not turns:
            raise ValueError("ARC histories must contain at least one turn")

        task_is_first = turns[0].role == "user"
        task_text = turns[0].text if task_is_first else ""
        context_start = 1 if task_is_first else 0

        task_ids_full = self._encode_block("[Task]\n" + task_text)
        context_header_ids = self._encode_block("[Context]")

        blocks: list[_HistoryBlock] = []
        turn_number = 0
        for turn in turns[context_start:]:
            if turn.role == "user":
                turn_number += 1
            blocks.append(
                _HistoryBlock(
                    header_ids=self._encode(f"[Turn {turn_number} - {turn.role}]\n"),
                    content_ids=self._encode(turn.text),
                    eos=self._eos,
                )
            )

        full_context_tokens = 0
        if blocks:
            full_context_tokens = len(context_header_ids) + sum(
                len(block.header_ids) + len(block.content_ids) + 1 for block in blocks
            )
        full_tokens = len(task_ids_full) + full_context_tokens

        if not blocks:
            task_ids = self._truncate_task(task_ids_full, self._max_tokens)
            return TokenizationResult(
                input_ids=task_ids,
                full_tokens=full_tokens,
                truncated_tokens=max(0, full_tokens - len(task_ids)),
            )

        newest_ids = blocks[-1].input_ids()
        newest_budget = min(self._min_recent_tokens, len(newest_ids))
        reserved_context = min(
            self._max_tokens - 1,
            len(context_header_ids) + newest_budget,
        )
        task_limit = max(1, self._max_tokens - reserved_context)
        task_ids = self._truncate_task(task_ids_full, task_limit)
        available = self._max_tokens - len(task_ids) - len(context_header_ids)

        selected_reversed: list[tuple[int, ...]] = []
        for block in reversed(blocks):
            block_ids = block.input_ids()
            if len(block_ids) <= available:
                available -= len(block_ids)
                selected_reversed.append(block_ids)
                continue
            if len(selected_reversed) < self._min_recent_turns and available > 0:
                partial = block.recent_tail(available)
                if partial:
                    available -= len(partial)
                    selected_reversed.append(partial)
            break

        input_ids = (
            *task_ids,
            *context_header_ids,
            *(token_id for block in reversed(selected_reversed) for token_id in block),
        )
        if len(input_ids) > self._max_tokens:
            raise RuntimeError("ARC tokenization exceeded the context limit")
        return TokenizationResult(
            input_ids=input_ids,
            full_tokens=full_tokens,
            truncated_tokens=max(0, full_tokens - len(input_ids)),
        )
