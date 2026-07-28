# Rayline ARC vLLM IO Processor

This installable plugin owns the frozen `mtrouter-token-blocks-v2` serializer
and the Rung A FP32 masked-mean/L2 pooling adapter for PL-0039. vLLM owns all
Qwen inference and token-hidden-state production; the plugin does not contain
or load the Rayline policy head.

The plugin is deliberately fail closed. It accepts only the pinned
`Qwen/Qwen3.5-0.8B` model and tokenizer revision, BF16, a 262,144-token context,
`token_embed`/`ALL` pooling without activation, and Rung A with automatic
prefix caching disabled. Startup also proves the real EOS and a pinned
literal-special-token probe.

Install it into the same environment as vLLM:

```bash
uv pip install ./src/vllm-plugins/rayline_arc_io
```

The serving process must set an immutable build identifier:

```bash
export RAYLINE_ARC_ENGINE_BUILD_ID=vllm@<image-or-source-revision>
```

The deployment command and readiness contract are added by the later PL-0039
deployment task. vLLM's `/pooling` API carries the strict ARC request inside
its standard plugin envelope and wraps the strict ARC response as `data`:

```json
{
  "task": "plugin",
  "data": {
    "schema_version": "rayline.arc.pooling-request.v1",
    "serializer_version": "mtrouter-token-blocks-v2",
    "serving_rung": "A",
    "episode_id_hash": "<64-lowercase-hex>",
    "turns": [{"role": "user", "text": "public synthetic input"}]
  }
}
```

Unit tests are host-independent:

```bash
uv run --project src/vllm-plugins/rayline_arc_io --extra test pytest
```
