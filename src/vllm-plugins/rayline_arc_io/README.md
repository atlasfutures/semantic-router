# Rayline ARC vLLM IO Processor

This installable plugin owns the frozen `mtrouter-token-blocks-v2` serializer
and the Rung A/B pooling adapters for PL-0039. vLLM owns all Qwen inference
and Rung B's FP32 causal mean; the plugin does not contain or load the Rayline
policy head.

The plugin is deliberately fail closed. It accepts only the pinned
`Qwen/Qwen3.5-0.8B` model and tokenizer revision, BF16, a 262,144-token context,
and automatic prefix caching disabled. Rung A accepts `token_embed`/`ALL`
without activation for diagnostics. Production Rung B accepts only
`embed`/causal `MEAN` with activation and returns one normalized vector rather
than transporting the token-hidden-state matrix. Startup also proves the real EOS and a pinned
literal-special-token probe. Every tokenizer call sets
`split_special_tokens=true`; relying on a backend tokenizer attribute is not
equivalent through the Transformers wrapper.

Install it into the same environment as vLLM:

```bash
uv pip install ./src/vllm-plugins/rayline_arc_io
```

The serving process must set an immutable build identifier:

```bash
export RAYLINE_ARC_ENGINE_BUILD_ID=vllm@<image-or-source-revision>
```

The protected Modal Rung B deployment is defined in `modal_service.py`. Deploy
it only after the CUDA correctness gate passes:

```bash
modal deploy src/vllm-plugins/rayline_arc_io/modal_service.py
```

The endpoint requires Modal proxy authentication. Configure Semantic Router
with environment-variable names for the corresponding `Modal-Key` and
`Modal-Secret`; never put their values in router YAML.

vLLM's `/pooling` API carries the strict ARC request inside its standard
plugin envelope and wraps the strict ARC response as `data`:

```json
{
  "task": "plugin",
  "data": {
    "schema_version": "rayline.arc.pooling-request.v1",
    "serializer_version": "mtrouter-token-blocks-v2",
    "serving_rung": "B",
    "episode_id_hash": "<64-lowercase-hex>",
    "turns": [{"role": "user", "text": "public synthetic input"}]
  }
}
```

Unit tests are host-independent:

```bash
uv run --project src/vllm-plugins/rayline_arc_io --extra test pytest
```

## Retained-session endpoint

`modal_session_service.py` is a separate, versioned comparison arm. It embeds
the proven `AsyncPoolingSession` API instead of exposing retained state through
vLLM's stateless `/pooling` contract:

```bash
modal deploy src/vllm-plugins/rayline_arc_io/modal_session_service.py
```

The protected endpoint accepts the complete reconstructible history at
`POST /v1/rayline/arc/session/pooling`:

```json
{
  "schema_version": "rayline.arc.session-pooling-request.v1",
  "serializer_version": "mtrouter-token-blocks-v2",
  "serving_rung": "B",
  "episode_id_hash": "<64-lowercase-hex>",
  "turns": [{"role": "user", "text": "public synthetic input"}]
}
```

An exact token extension appends only its suffix. An identical retry reuses the
last result, and any other history closes the old live request and rebuilds
from the supplied full history. Sessions are ephemeral: TTL, LRU pressure,
container restart, or an affinity miss can discard them without affecting
correctness because every request remains reconstructible. Per-episode work is
serialized; independent episodes may execute concurrently. The deployment
bounds both resident sessions and total retained tokens, and exposes those
counts at `GET /health`. `DELETE /v1/rayline/arc/session/{episode_id_hash}`
releases one idle session explicitly.

`GET /v1/rayline/arc/session/metrics` exposes a versioned, aggregate-only
diagnostic snapshot. Coordinator fields report tokenization time, request
in-flight and peak in-flight counts, actual same-session lock contention and
wait time, backend-append concurrency, latency, failures, and successfully
appended tokens. The protected vLLM deployment also supplies curated scheduler
gauges and cumulative queue, inference, end-to-end, and prompt-token metrics.
Those cumulative timings are scoped to completed retained appends, not whole
sessions or ordinary terminal requests. Scheduler occupancy comes from a
cached in-process vLLM snapshot, so reading the endpoint does not collect the
Prometheus registry. The endpoint has no request, episode, prompt, embedding,
or credential labels; unavailable engine telemetry is represented explicitly
rather than as zero.

The same protected service also exposes a compatibility `POST /pooling` route
for Pathfinder's strict stateless v1 client. Each call runs as one randomly
namespaced ephemeral append and closes its backend before returning the normal
vLLM plugin envelope. It therefore preserves `cached_prefix_tokens: 0` and
never turns the stateless contract into cross-request cache reuse, while still
making append-scoped scheduler and latency metrics available for
transaction-path capacity receipts.

The Modal MVP pins the service to one container so successive turns reach the
same cache owner and the GPU cost envelope remains enforceable. Production
horizontal scale requires cache-aware affinity or an explicit shared session
directory; ordinary round-robin scaling is correct only by rebuilding and does
not preserve the KV-reuse performance claim.

The response reports `retained_prefix_tokens`, `appended_tokens`,
`session_action`, and `session_revision`. These are explicit live-session
metrics and must not be interpreted as vLLM automatic prefix-cache hits;
automatic prefix caching remains disabled.

## Real generation workers

`modal_generation_workers.py` is separate, ephemeral test infrastructure for
the real-worker full-stack rung. It deploys two revision-pinned L4 vLLM OpenAI
servers, requires an injected `RAYLINE_ARC_WORKER_API_KEY`, disables automatic
prefix caching, and caps each worker at one container. Use
`e2e/testing/rayline-arc/run_modal_fullstack.py` to create short-lived
credentials, deploy the workers, exercise both routing paths plus streaming,
and stop the worker app during guaranteed cleanup. Run it with the pinned SDK:

```bash
UV_LOCKED=0 uv run --with modal==1.5.1 python \
  e2e/testing/rayline-arc/run_modal_fullstack.py --run-id <registered-run-id>
```

The launcher uses that same interpreter for Modal deploy and cleanup commands,
so its proxy-token API and serialized function graph cannot silently come from
different SDK versions.
