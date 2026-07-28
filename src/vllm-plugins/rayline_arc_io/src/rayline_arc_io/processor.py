# SPDX-License-Identifier: Apache-2.0

"""vLLM adapter for the frozen Rayline ARC Rung A contract."""

import hashlib
import importlib
import os
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pydantic import ValidationError
from vllm.config import VllmConfig
from vllm.inputs import PromptType, TokensPrompt
from vllm.outputs import PoolingRequestOutput
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer

from .constants import (
    EMBEDDING_DIMENSION,
    EOS_TOKEN,
    EOS_TOKEN_ID,
    MAX_SERIALIZED_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    PLUGIN_VERSION,
    RUNG_A_CAPABILITIES,
    TOKENIZER_PROBE_IDS,
    TOKENIZER_PROBE_TEXT,
    TOKENIZER_REVISION,
    TOKENIZER_SHA256,
)
from .pooling import fp32_masked_mean_l2
from .schemas import ArcPoolingRequest, ArcPoolingResponse
from .serializer import TokenBlockSerializer, TokenizationResult

_ENGINE_BUILD_PATTERN = re.compile(r"^vllm@[A-Za-z0-9][A-Za-z0-9._:+/@-]{6,127}$")
_PENDING_TTL_SECONDS = 30 * 60
_MAX_PENDING_REQUESTS = 4096
_HASH_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class _PendingRequest:
    tokenization: TokenizationResult
    created_at: float


class RaylineArcIOProcessor(IOProcessor[ArcPoolingRequest, ArcPoolingResponse]):
    """Serialize structured ARC turns and pool Qwen hidden states in FP32."""

    def __init__(self, vllm_config: VllmConfig, renderer: BaseRenderer):
        super().__init__(vllm_config, renderer)
        self.renderer = renderer
        self._engine_build_id = self._validate_server_config(vllm_config)
        tokenizer = renderer.get_tokenizer()
        self._configure_and_validate_tokenizer(tokenizer)
        self._serializer = TokenBlockSerializer(tokenizer)
        self._pending: OrderedDict[str, _PendingRequest] = OrderedDict()
        self._pending_lock = threading.Lock()

    @staticmethod
    def _validate_server_config(vllm_config: VllmConfig) -> str:
        model = vllm_config.model_config
        failures = RaylineArcIOProcessor._model_config_failures(model)
        failures.extend(
            RaylineArcIOProcessor._pooler_config_failures(model.pooler_config)
        )
        if vllm_config.cache_config.enable_prefix_caching:
            failures.append("automatic prefix caching must be disabled for Rung A")

        engine_build_id = os.environ.get("RAYLINE_ARC_ENGINE_BUILD_ID", "")
        if not _ENGINE_BUILD_PATTERN.fullmatch(engine_build_id):
            failures.append(
                "RAYLINE_ARC_ENGINE_BUILD_ID must match 'vllm@<immutable-build-id>'"
            )
        if failures:
            raise ValueError(
                "invalid Rayline ARC vLLM configuration: " + "; ".join(failures)
            )
        return engine_build_id

    @staticmethod
    def _model_config_failures(model: Any) -> list[str]:
        checks = (
            (model.model == MODEL_ID, f"model={model.model!r}, expected {MODEL_ID!r}"),
            (
                model.tokenizer == MODEL_ID,
                f"tokenizer={model.tokenizer!r}, expected {MODEL_ID!r}",
            ),
            (
                model.revision == MODEL_REVISION,
                f"revision={model.revision!r}, expected {MODEL_REVISION!r}",
            ),
            (
                model.tokenizer_revision == TOKENIZER_REVISION,
                f"tokenizer_revision={model.tokenizer_revision!r}, expected {TOKENIZER_REVISION!r}",
            ),
            (
                model.dtype is torch.bfloat16,
                f"dtype={model.dtype!r}, expected torch.bfloat16",
            ),
            (
                model.max_model_len == MAX_SERIALIZED_TOKENS,
                f"max_model_len={model.max_model_len!r}, expected {MAX_SERIALIZED_TOKENS}",
            ),
            (
                model.embedding_size == EMBEDDING_DIMENSION,
                f"embedding_size={model.embedding_size!r}, expected {EMBEDDING_DIMENSION}",
            ),
        )
        return [message for valid, message in checks if not valid]

    @staticmethod
    def _pooler_config_failures(pooler: Any) -> list[str]:
        if pooler is None:
            return ["pooler_config is missing"]
        checks = (
            (
                pooler.task == "token_embed",
                f"pooler task={pooler.task!r}, expected 'token_embed'",
            ),
            (
                pooler.tok_pooling_type == "ALL",
                f"token pooling type={pooler.tok_pooling_type!r}, expected 'ALL'",
            ),
            (
                pooler.use_activation is False,
                f"pooler use_activation={pooler.use_activation!r}, expected False",
            ),
            (
                not pooler.enable_chunked_processing,
                "pooler enable_chunked_processing must be false",
            ),
        )
        return [message for valid, message in checks if not valid]

    @staticmethod
    def _configure_and_validate_tokenizer(tokenizer: Any) -> None:
        if tokenizer.eos_token != EOS_TOKEN or tokenizer.eos_token_id != EOS_TOKEN_ID:
            raise ValueError(
                "Rayline ARC tokenizer EOS mismatch: "
                f"got {tokenizer.eos_token!r}/{tokenizer.eos_token_id!r}, "
                f"expected {EOS_TOKEN!r}/{EOS_TOKEN_ID}"
            )

        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None or not hasattr(backend, "encode_special_tokens"):
            raise ValueError(
                "Rayline ARC requires a fast tokenizer with encode_special_tokens"
            )
        backend.encode_special_tokens = True
        if backend.encode_special_tokens is not True:
            raise ValueError("failed to disable special-token parsing for ARC content")

        RaylineArcIOProcessor._verify_tokenizer_file(tokenizer)
        probe_ids = tokenizer.encode(TOKENIZER_PROBE_TEXT, add_special_tokens=False)
        if tuple(probe_ids) != TOKENIZER_PROBE_IDS:
            raise ValueError(
                f"Rayline ARC tokenizer behavioral fingerprint mismatch: got {tuple(probe_ids)!r}"
            )

    @staticmethod
    def _verify_tokenizer_file(tokenizer: Any) -> None:
        tokenizer_path = RaylineArcIOProcessor._resolve_tokenizer_file(tokenizer)
        digest = hashlib.sha256()
        with tokenizer_path.open("rb") as tokenizer_file:
            while chunk := tokenizer_file.read(_HASH_CHUNK_BYTES):
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != TOKENIZER_SHA256:
            raise ValueError(
                "Rayline ARC tokenizer.json SHA256 mismatch: "
                f"got {actual_sha256}, expected {TOKENIZER_SHA256}"
            )

    @staticmethod
    def _resolve_tokenizer_file(tokenizer: Any) -> Path:
        name_or_path = str(getattr(tokenizer, "name_or_path", ""))
        if name_or_path:
            local_candidate = Path(name_or_path).expanduser() / "tokenizer.json"
            if local_candidate.is_file():
                return local_candidate

        try:
            transformers_hub = importlib.import_module("transformers.utils.hub")
            resolved = transformers_hub.cached_file(
                MODEL_ID,
                "tokenizer.json",
                revision=TOKENIZER_REVISION,
                local_files_only=True,
            )
        except (ImportError, OSError) as error:
            raise ValueError(
                "Rayline ARC could not resolve the pinned tokenizer.json from "
                "the local Hugging Face cache"
            ) from error
        if not resolved:
            raise ValueError(
                "Rayline ARC could not resolve the pinned tokenizer.json from "
                "the local Hugging Face cache"
            )
        return Path(resolved)

    def parse_data(self, data: object) -> ArcPoolingRequest:
        if not isinstance(data, dict):
            raise TypeError("Rayline ARC request data must be a dictionary")
        try:
            return ArcPoolingRequest.model_validate(data)
        except ValidationError as error:
            raise ValueError(f"invalid Rayline ARC request: {error}") from error

    def merge_pooling_params(
        self,
        params: PoolingParams | None = None,
    ) -> PoolingParams:
        if params is None:
            params = PoolingParams()
        params.task = "token_embed"
        params.use_activation = False
        params.dimensions = None
        params.skip_reading_prefix_cache = True
        return params

    def _expire_pending_locked(self, now: float) -> None:
        while self._pending:
            first_key = next(iter(self._pending))
            if now - self._pending[first_key].created_at <= _PENDING_TTL_SECONDS:
                break
            self._pending.popitem(last=False)

    def pre_process(
        self,
        prompt: ArcPoolingRequest,
        request_id: str | None = None,
        **kwargs: Any,
    ) -> PromptType:
        if not request_id:
            raise ValueError("Rayline ARC online requests require a vLLM request_id")
        tokenization = self._serializer.tokenize(prompt.turns)
        now = time.monotonic()
        with self._pending_lock:
            self._expire_pending_locked(now)
            if request_id in self._pending:
                raise ValueError(f"duplicate Rayline ARC request_id {request_id!r}")
            if len(self._pending) >= _MAX_PENDING_REQUESTS:
                raise RuntimeError("Rayline ARC pending-request cache is full")
            self._pending[request_id] = _PendingRequest(tokenization, now)
        return TokensPrompt(prompt_token_ids=list(tokenization.input_ids))

    def post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_id: str | None = None,
        **kwargs: Any,
    ) -> ArcPoolingResponse:
        if not request_id:
            raise ValueError("Rayline ARC online responses require a vLLM request_id")
        with self._pending_lock:
            pending = self._pending.pop(request_id, None)
        if pending is None:
            raise ValueError(
                f"unknown or expired Rayline ARC request_id {request_id!r}"
            )
        if len(model_output) != 1:
            raise ValueError(
                f"Rayline ARC requires exactly one pooling output, got {len(model_output)}"
            )

        output = model_output[0]
        if output.request_id != request_id:
            raise ValueError(
                "Rayline ARC vLLM request ID mismatch: "
                f"got {output.request_id!r}, expected {request_id!r}"
            )
        if not output.finished:
            raise ValueError("Rayline ARC received an unfinished pooling output")
        if tuple(output.prompt_token_ids) != pending.tokenization.input_ids:
            raise ValueError("Rayline ARC vLLM prompt token IDs changed in flight")
        if output.num_cached_tokens != 0:
            raise ValueError(
                f"Rayline ARC Rung A forbids cached prefix tokens, got {output.num_cached_tokens}"
            )

        embedding = fp32_masked_mean_l2(
            output.outputs.data,
            expected_tokens=len(pending.tokenization.input_ids),
        )
        return ArcPoolingResponse(
            embedding=embedding,
            serialized_tokens=len(pending.tokenization.input_ids),
            full_history_tokens=pending.tokenization.full_tokens,
            truncated_tokens=pending.tokenization.truncated_tokens,
            cached_prefix_tokens=0,
            model_revision=MODEL_REVISION,
            engine_build_id=self._engine_build_id,
            io_plugin_version=PLUGIN_VERSION,
            pooling_capabilities=list(RUNG_A_CAPABILITIES),
        )
