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


class _ForkPoolingStates:
    """Models the fork's PoolingStates causal-MEAN accumulator surface."""

    def __init__(self):
        self.mean_pool_sum = None
        self.mean_pool_count = 0


class _ForkMeanPool:
    def _forward_accumulate(self):
        raise NotImplementedError


class _ForkScheduler:
    def schedule(self):
        # Mirrors the fork's exact-max pooling change: the attestation reads
        # this method's source for the reservation marker.
        sampled_token_reservation = 0
        return sampled_token_reservation


def _install_vllm_stubs() -> None:
    modules = {
        "vllm": ModuleType("vllm"),
        "vllm.config": ModuleType("vllm.config"),
        "vllm.inputs": ModuleType("vllm.inputs"),
        "vllm.model_executor": ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": ModuleType("vllm.model_executor.layers"),
        "vllm.model_executor.layers.pooler": ModuleType(
            "vllm.model_executor.layers.pooler"
        ),
        "vllm.model_executor.layers.pooler.seqwise": ModuleType(
            "vllm.model_executor.layers.pooler.seqwise"
        ),
        "vllm.model_executor.layers.pooler.seqwise.methods": ModuleType(
            "vllm.model_executor.layers.pooler.seqwise.methods"
        ),
        "vllm.outputs": ModuleType("vllm.outputs"),
        "vllm.plugins": ModuleType("vllm.plugins"),
        "vllm.plugins.io_processors": ModuleType("vllm.plugins.io_processors"),
        "vllm.plugins.io_processors.interface": ModuleType(
            "vllm.plugins.io_processors.interface"
        ),
        "vllm.pooling_params": ModuleType("vllm.pooling_params"),
        "vllm.renderers": ModuleType("vllm.renderers"),
        "vllm.v1": ModuleType("vllm.v1"),
        "vllm.v1.core": ModuleType("vllm.v1.core"),
        "vllm.v1.core.sched": ModuleType("vllm.v1.core.sched"),
        "vllm.v1.core.sched.scheduler": ModuleType("vllm.v1.core.sched.scheduler"),
        "vllm.v1.pool": ModuleType("vllm.v1.pool"),
        "vllm.v1.pool.metadata": ModuleType("vllm.v1.pool.metadata"),
    }
    modules["vllm.config"].VllmConfig = object
    modules["vllm.inputs"].PromptType = object
    modules["vllm.inputs"].TokensPrompt = _TokensPrompt
    modules["vllm.model_executor.layers.pooler.seqwise.methods"].MeanPool = (
        _ForkMeanPool
    )
    modules["vllm.outputs"].PoolingRequestOutput = object
    modules["vllm.plugins.io_processors.interface"].IOProcessor = _IOProcessor
    modules["vllm.pooling_params"].PoolingParams = _PoolingParams
    modules["vllm.renderers"].BaseRenderer = object
    modules["vllm.v1.core.sched.scheduler"].Scheduler = _ForkScheduler
    modules["vllm.v1.pool.metadata"].PoolingStates = _ForkPoolingStates
    sys.modules.update(modules)


_install_vllm_stubs()
processor_module = importlib.import_module("rayline_arc_io.processor")
RaylineArcIOProcessor = processor_module.RaylineArcIOProcessor
installed_source_digest = importlib.import_module(
    "rayline_arc_io.integrity"
).installed_source_digest


class _Tokenizer:
    eos_token = EOS_TOKEN
    eos_token_id = EOS_TOKEN_ID

    def __init__(self):
        self.split_special_tokens = False

    def encode(self, text, *, add_special_tokens, split_special_tokens):
        assert add_special_tokens is False
        assert split_special_tokens is True
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
        seq_pooling_type=None,
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


def _rung_b_config():
    config = _config()
    config.model_config.pooler_config = SimpleNamespace(
        task="embed",
        seq_pooling_type="MEAN",
        tok_pooling_type=None,
        use_activation=True,
        enable_chunked_processing=False,
    )
    return config


def _request_data(serving_rung="A"):
    return {
        "schema_version": "rayline.arc.pooling-request.v1",
        "serializer_version": "mtrouter-token-blocks-v2",
        "serving_rung": serving_rung,
        "episode_id_hash": "b" * 64,
        "turns": [{"role": "user", "text": "hello"}],
    }


@pytest.fixture(autouse=True)
def pinned_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("RAYLINE_ARC_ENGINE_BUILD_ID", "vllm@test-build-1206891")
    monkeypatch.setenv(
        "RAYLINE_ARC_PLUGIN_SOURCE_DIGEST",
        installed_source_digest(),
    )
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer_file.write_bytes(b"public synthetic tokenizer fixture")
    monkeypatch.setattr(_Tokenizer, "name_or_path", str(tmp_path), raising=False)
    monkeypatch.setattr(
        processor_module,
        "TOKENIZER_SHA256",
        hashlib.sha256(tokenizer_file.read_bytes()).hexdigest(),
    )


def test_adapter_serializes_and_returns_normalized_contract() -> None:
    tokenizer = _Tokenizer()
    processor = RaylineArcIOProcessor(_config(), _Renderer(tokenizer))
    assert tokenizer.split_special_tokens is True
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
        request_id="request-1-0",
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


def test_rung_b_adapter_returns_in_engine_causal_mean() -> None:
    processor = RaylineArcIOProcessor(_rung_b_config(), _Renderer())
    request = processor.parse_data(_request_data("B"))
    prompt = processor.pre_process(request, request_id="request-b")
    params = processor.merge_pooling_params(_PoolingParams())

    assert params.task == "embed"
    assert params.use_activation is True
    assert params.dimensions is None
    assert params.skip_reading_prefix_cache is True

    embedding = torch.zeros(1024)
    embedding[7] = 1
    output = SimpleNamespace(
        request_id="request-b-0",
        prompt_token_ids=prompt.prompt_token_ids,
        num_cached_tokens=0,
        finished=True,
        outputs=SimpleNamespace(data=embedding),
    )

    response = processor.post_process([output], request_id="request-b")

    assert response.embedding[7] == 1
    assert response.pooling_capabilities == ["chunked_causal_mean"]


def test_adapter_rejects_request_for_inactive_rung() -> None:
    processor = RaylineArcIOProcessor(_rung_b_config(), _Renderer())
    request = processor.parse_data(_request_data("A"))
    with pytest.raises(ValueError, match="does not match deployed Rung B"):
        processor.pre_process(request, request_id="request-rung-mismatch")


@pytest.mark.parametrize(
    "embedding",
    [
        torch.ones(1023),
        torch.full((1024,), float("nan")),
        torch.ones(1024),
    ],
)
def test_rung_b_adapter_rejects_invalid_embedding(embedding) -> None:
    processor = RaylineArcIOProcessor(_rung_b_config(), _Renderer())
    request = processor.parse_data(_request_data("B"))
    prompt = processor.pre_process(request, request_id="request-invalid-b")
    output = SimpleNamespace(
        request_id="request-invalid-b-0",
        prompt_token_ids=prompt.prompt_token_ids,
        num_cached_tokens=0,
        finished=True,
        outputs=SimpleNamespace(data=embedding),
    )
    with pytest.raises(ValueError, match="Rung B"):
        processor.post_process([output], request_id="request-invalid-b")


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
        request_id="request-2-0",
        prompt_token_ids=prompt.prompt_token_ids,
        num_cached_tokens=1,
        finished=True,
        outputs=SimpleNamespace(data=torch.ones((len(prompt.prompt_token_ids), 1024))),
    )
    with pytest.raises(ValueError, match="forbid cached prefix"):
        processor.post_process([output], request_id="request-2")


def test_adapter_rejects_unindexed_engine_request_id() -> None:
    processor = RaylineArcIOProcessor(_config(), _Renderer())
    request = processor.parse_data(_request_data())
    prompt = processor.pre_process(request, request_id="request-3")
    output = SimpleNamespace(
        request_id="request-3",
        prompt_token_ids=prompt.prompt_token_ids,
        num_cached_tokens=0,
        finished=True,
        outputs=SimpleNamespace(data=torch.ones((len(prompt.prompt_token_ids), 1024))),
    )
    with pytest.raises(ValueError, match=r"expected 'request-3-0'"):
        processor.post_process([output], request_id="request-3")


def test_adapter_requires_immutable_engine_build_id(monkeypatch) -> None:
    monkeypatch.setenv("RAYLINE_ARC_ENGINE_BUILD_ID", "latest")
    with pytest.raises(ValueError, match="ENGINE_BUILD_ID"):
        RaylineArcIOProcessor(_config(), _Renderer())


def test_adapter_requires_matching_plugin_source_digest(monkeypatch) -> None:
    monkeypatch.setenv("RAYLINE_ARC_PLUGIN_SOURCE_DIGEST", "c" * 64)
    with pytest.raises(ValueError, match="plugin source does not match"):
        RaylineArcIOProcessor(_config(), _Renderer())


def test_adapter_requires_wellformed_plugin_source_digest(monkeypatch) -> None:
    monkeypatch.delenv("RAYLINE_ARC_PLUGIN_SOURCE_DIGEST")
    with pytest.raises(ValueError, match="PLUGIN_SOURCE_DIGEST"):
        RaylineArcIOProcessor(_config(), _Renderer())


def test_adapter_attests_fork_pooling_state(monkeypatch) -> None:
    metadata = sys.modules["vllm.v1.pool.metadata"]

    class _StockPoolingStates:
        pass

    monkeypatch.setattr(metadata, "PoolingStates", _StockPoolingStates)
    with pytest.raises(ValueError, match="causal-MEAN accumulator"):
        RaylineArcIOProcessor(_config(), _Renderer())


def test_adapter_attests_fork_scheduler_reservation(monkeypatch) -> None:
    scheduler = sys.modules["vllm.v1.core.sched.scheduler"]

    class _StockScheduler:
        def schedule(self):
            return None

    monkeypatch.setattr(scheduler, "Scheduler", _StockScheduler)
    with pytest.raises(ValueError, match="exact-max pooling reservation"):
        RaylineArcIOProcessor(_config(), _Renderer())


def test_parse_data_errors_never_echo_turn_content() -> None:
    processor = RaylineArcIOProcessor(_config(), _Renderer())
    oversized = _request_data()
    oversized["turns"] = [
        {"role": "user", "text": "CANARY-PROMPT " + "y" * (16 * 1024 * 1024)}
    ]
    invalid_role = _request_data()
    invalid_role["turns"] = [{"role": "CANARY-ROLE", "text": "CANARY-PROMPT"}]

    for payload in (oversized, invalid_role):
        with pytest.raises(ValueError) as excinfo:
            processor.parse_data(payload)
        message = str(excinfo.value)
        assert "CANARY" not in message
        assert "invalid Rayline ARC request" in message


def test_pending_cache_evicts_oldest_when_full(monkeypatch) -> None:
    processor = RaylineArcIOProcessor(_config(), _Renderer())
    monkeypatch.setattr(processor_module, "_MAX_PENDING_REQUESTS", 2)
    request = processor.parse_data(_request_data())
    processor.pre_process(request, request_id="req-a")
    processor.pre_process(request, request_id="req-b")
    processor.pre_process(request, request_id="req-c")
    with processor._pending_lock:
        assert list(processor._pending) == ["req-b", "req-c"]
