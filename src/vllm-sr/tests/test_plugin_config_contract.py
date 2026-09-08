"""The plugin configuration contract shared by `vllm-sr validate` and the router.

The router decodes a decision plugin's `configuration` block into a Go struct
(`pkg/config/plugin_config.go`); `vllm-sr validate` checks the same block
against a pydantic model (`cli/validator.py:306`). Both are hand-written.
"""

from cli.models import (
    Decision,
    MemoryPluginConfig,
    PluginConfig,
    PluginType,
    RouterReplayPluginConfig,
    Routing,
    UserConfig,
)
from cli.validator import validate_plugin_configurations


def _validate(plugin_type: str, configuration: dict) -> list:
    """Run the plugin check `vllm-sr validate` runs, over one plugin block."""
    config = UserConfig(
        version="0.3",
        routing=Routing(
            decisions=[
                Decision(
                    name="d",
                    priority=1,
                    modelRefs=[],
                    plugins=[
                        PluginConfig(type=plugin_type, configuration=configuration)
                    ],
                )
            ]
        ),
    )
    return validate_plugin_configurations(config)


# Keys the router reads, paired with the Go field that reads them. The CLI
# model declares none of them, so `vllm-sr validate` cannot check them.
ROUTER_HONOURED_KEYS = [
    (MemoryPluginConfig, "hybrid_search", "plugin_config.go:175"),
    (MemoryPluginConfig, "hybrid_mode", "plugin_config.go:176"),
    (MemoryPluginConfig, "reflection", "plugin_config.go:177"),
    (RouterReplayPluginConfig, "max_tool_trace_bytes", "plugin_config.go:245"),
    (RouterReplayPluginConfig, "max_tool_trace_steps", "plugin_config.go:252"),
]

# One misspelling of a key each of these models does declare.
MISSPELLED_KEYS = [
    (PluginType.MEMORY.value, {"enabled": True, "retrieval_limitt": 4}),
    (PluginType.ROUTER_REPLAY.value, {"enabled": True, "max_recordss": 10}),
    (PluginType.REQUEST_PARAMS.value, {"max_tokens_limitt": 4096}),
    (PluginType.SYSTEM_PROMPT.value, {"system_prompt": "hi", "modee": "replace"}),
]

# `context_compression` has a strict CLI model, absent from `config_models`.
CONTEXT_COMPRESSION_TYPO = {"enabled": True, "modee": "auto"}


def test_plugin_validator_rejects_unknown_keys():
    """A misspelled plugin configuration key must be reported as invalid.

    Pydantic defaults to `extra="ignore"`, and of the plugin config models in
    `cli/models.py` only `ContextCompressionPluginConfig:894` forbids extras.
    """
    accepted = [
        plugin_type
        for plugin_type, configuration in MISSPELLED_KEYS
        if not _validate(plugin_type, configuration)
    ]
    assert not accepted, f"a misspelled configuration key is accepted for {accepted}"


def test_plugin_contract_carries_router_honoured_keys():
    """Every key the router honours must be one `vllm-sr validate` can check."""
    missing = [
        f"{model.__name__} omits {key}, read by {where}"
        for model, key, where in ROUTER_HONOURED_KEYS
        if key not in model.model_fields
    ]
    assert not missing, f"the router reads keys the CLI does not model: {missing}"
    assert _validate(
        PluginType.CONTEXT_COMPRESSION.value, CONTEXT_COMPRESSION_TYPO
    ), "context_compression has a strict model that the validator never applies"


def test_unknown_plugin_key_passes_validation_today():
    """Pins the present behaviour so a change to it shows up in the diff."""
    unchecked = MISSPELLED_KEYS + [
        (PluginType.CONTEXT_COMPRESSION.value, CONTEXT_COMPRESSION_TYPO)
    ]
    rejected = [
        plugin_type
        for plugin_type, configuration in unchecked
        if _validate(plugin_type, configuration)
    ]
    assert not rejected, f"an unknown key is now rejected for {rejected}"
