# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from rayline_arc_io.constants import (
    EOS_TOKEN,
    EOS_TOKEN_ID,
    MODEL_ID,
    MODEL_REVISION,
    TOKENIZER_PROBE_IDS,
    TOKENIZER_PROBE_TEXT,
)


class _IOProcessor:
    @classmethod
    def __class_getitem__(cls, item):
        return cls

    def __init__(self, vllm_config, renderer):
        self.vllm_config = vllm_config


class _TokensPrompt(dict):
    def __init__(self, *, prompt_token_ids):
        super().__init__(prompt_token_ids=prompt_token_ids)
        self.prompt_token_ids = prompt_token_ids


class _PoolingParams:
    def __init__(self):
        self.task = None
        self.use_activation = None
        self.dimensions = 99
        self.skip_reading_prefix_cache = None


def _install_vllm_stubs() -> None:
    modules = {
        "vllm": ModuleType("vllm"),
        "vllm.config": ModuleType("vllm.config"),
        "vllm.inputs": ModuleType("vllm.inputs"),
        "vllm.outputs": ModuleType("vllm.outputs"),
        "vllm.plugins": ModuleType("vllm.plugins"),
        "vllm.plugins.io_processors": ModuleType("vllm.plugins.io_processors"),
        "vllm.plugins.io_processors.interface": ModuleType(
            "vllm.plugins.io_processors.interface"
        ),
        "vllm.pooling_params": ModuleType("vllm.pooling_params"),
        "vllm.renderers": ModuleType("vllm.renderers"),
    }
    modules["vllm.config"].VllmConfig = object
    modules["vllm.inputs"].PromptType = object
    modules["vllm.inputs"].TokensPrompt = _TokensPrompt
    modules["vllm.outputs"].PoolingRequestOutput = object
    modules["vllm.plugins.io_processors.interface"].IOProcessor = _IOProcessor
    modules["vllm.pooling_params"].PoolingParams = _PoolingParams
    modules["vllm.renderers"].BaseRenderer = object
    sys.modules.update(modules)


_install_vllm_stubs()
processor_module = importlib.import_module("rayline_arc_io.processor")
RaylineArcIOProcessor = processor_module.RaylineArcIOProcessor


class _Backend:
    encode_special_tokens = False


class _Tokenizer:
    eos_token = EOS_TOKEN
    eos_token_id = EOS_TOKEN_ID

    def __init__(self):
        self.backend_tokenizer = _Backend()

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        if text == TOKENIZER_PROBE_TEXT:
            return list(TOKENIZER_PROBE_IDS)
        return [ord(character) for character in text]


class _Renderer:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or _Tokenizer()

    def get_tokenizer(self):
        return self.tokenizer


def _config(**overrides):
    pooler = SimpleNamespace(
        task="token_embed",
        tok_pooling_type="ALL",
        use_activation=False,
        enable_chunked_processing=False,
    )
    model = SimpleNamespace(
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        max_model_len=262_144,
        embedding_size=1024,
        pooler_config=pooler,
    )
    cache = SimpleNamespace(enable_prefix_caching=False)
    config = SimpleNamespace(model_config=model, cache_config=cache)
    for path, value in overrides.items():
        target_name, attribute = path.split(".", maxsplit=1)
        setattr(getattr(config, target_name), attribute, value)
    return config


def _request_data():
    return {
        "schema_version": "rayline.arc.pooling-request.v1",
        "serializer_version": "mtrouter-token-blocks-v2",
        "serving_rung": "A",
        "episode_id_hash": "b" * 64,
        "turns": [{"role": "user", "text": "hello"}],
    }


@pytest.fixture(autouse=True)
def pinned_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("RAYLINE_ARC_ENGINE_BUILD_ID", "vllm@test-build-1206891")
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer_file.write_bytes(b"public synthetic tokenizer fixture")
    monkeypatch.setattr(_Tokenizer, "name_or_path", str(tmp_path), raising=False)
    monkeypatch.setattr(
        processor_module,
        "TOKENIZER_SHA256",
        hashlib.sha256(tokenizer_file.read_bytes()).hexdigest(),
    )


def test_adapter_serializes_and_returns_normalized_contract() -> None:
    processor = RaylineArcIOProcessor(_config(), _Renderer())
    request = processor.parse_data(_request_data())
    prompt = processor.pre_process(request, request_id="request-1")
    params = processor.merge_pooling_params(_PoolingParams())

    assert params.task == "token_embed"
    assert params.use_activation is False
    assert params.dimensions is None
    assert params.skip_reading_prefix_cache is True

    token_count = len(prompt.prompt_token_ids)
    states = torch.zeros((token_count, 1024), dtype=torch.bfloat16)
    states[:, 3] = 2
    output = SimpleNamespace(
        request_id="request-1",
        prompt_token_ids=prompt.prompt_token_ids,
        num_cached_tokens=0,
        finished=True,
        outputs=SimpleNamespace(data=states),
    )

    response = processor.post_process([output], request_id="request-1")

    assert response.embedding[3] == pytest.approx(1.0)
    assert sum(response.embedding[:3]) == 0
    assert response.serialized_tokens == token_count
    assert response.full_history_tokens == token_count
    assert response.truncated_tokens == 0
    assert response.cached_prefix_tokens == 0
    assert response.engine_build_id == "vllm@test-build-1206891"
    assert response.pooling_capabilities == ["all_plugin_mean"]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"model_config.revision": "main"}, "revision="),
        ({"model_config.max_model_len": 8192}, "max_model_len="),
        ({"cache_config.enable_prefix_caching": True}, "prefix caching"),
    ],
)
def test_adapter_rejects_server_drift(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        RaylineArcIOProcessor(_config(**overrides), _Renderer())


def test_adapter_rejects_tokenizer_fingerprint_drift() -> None:
    tokenizer = _Tokenizer()
    tokenizer.eos_token_id = 0
    with pytest.raises(ValueError, match="EOS mismatch"):
        RaylineArcIOProcessor(_config(), _Renderer(tokenizer))


def test_adapter_rejects_tokenizer_file_drift(tmp_path) -> None:
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer_file.write_bytes(b"different tokenizer")
    tokenizer = _Tokenizer()
    tokenizer.name_or_path = str(tmp_path)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        RaylineArcIOProcessor(_config(), _Renderer(tokenizer))


def test_adapter_rejects_cached_rung_a_output() -> None:
    processor = RaylineArcIOProcessor(_config(), _Renderer())
    request = processor.parse_data(_request_data())
    prompt = processor.pre_process(request, request_id="request-2")
    output = SimpleNamespace(
        request_id="request-2",
        prompt_token_ids=prompt.prompt_token_ids,
        num_cached_tokens=1,
        finished=True,
        outputs=SimpleNamespace(data=torch.ones((len(prompt.prompt_token_ids), 1024))),
    )
    with pytest.raises(ValueError, match="forbids cached prefix"):
        processor.post_process([output], request_id="request-2")


def test_adapter_requires_immutable_engine_build_id(monkeypatch) -> None:
    monkeypatch.setenv("RAYLINE_ARC_ENGINE_BUILD_ID", "latest")
    with pytest.raises(ValueError, match="ENGINE_BUILD_ID"):
        RaylineArcIOProcessor(_config(), _Renderer())
