# PL-0039 Rayline ARC Orchestrator (C82)

## Goal

Implement the frozen Rayline C82 switch-aware orchestrator as an experimental
`rayline_arc` selection algorithm. vLLM performs the Qwen3.5-0.8B encoder
inference; Semantic Router owns the artifact-verified F32 head, cache-aware
policy, transactional episode state, provider dispatch, and observability.

This is architecture-parity work, not a model-promotion claim. C82 did not pass
its private held-test promotion gate, so the feature remains experimental until
a separately frozen successor passes the research repository's promotion
process. Private evaluation results do not belong in this public plan.

## Scope

- A dedicated vLLM pooling deployment for
  `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`.
- An out-of-tree vLLM IO Processor plugin implementing the exact
  `mtrouter-token-blocks-v2` input contract and returning normalized
  1024-dimensional history embeddings.
- A staged vLLM path: an uncached `token_embed`/`ALL` correctness bootstrap,
  then a required core enhancement for causal `MEAN` across chunked prefill,
  then automatic-prefix-cache summaries only when the production-economics
  phase gate is opened.
- An artifact-pinned Semantic Router selector that evaluates the frozen C82
  head and policy without Python or Candle inference in the router process.
- A transactional episode store with same-episode serialization and
  commit-on-success semantics.
- Exact C82 provider/request shaping, safe decision telemetry, reference
  configuration, deployment assets, and CPU/GPU integration tests.

Non-goals:

- Retraining, changing, quantizing, or claiming promotion of C82.
- Serving the encoder through Semantic Router's existing generic text
  embedding provider. That API cannot carry structured turns or prove the
  frozen serializer contract.
- Running the C82 head inside vLLM. It is a small deterministic F32 policy head,
  not the SLM inference workload.
- Reusing Semantic Router's generic session-aware switch gate on top of C82.
  C82 already incorporates the previous arm, cache-loss cost, and stay margin;
  another gate would double-penalize switching.
- Making the encoder cache transactional. vLLM's cache is content-addressed
  acceleration only; failed upstream dispatches must roll back episode policy
  state, not harmless reusable encoder blocks.
- Spending the frozen holdout. Development uses artifact goldens, synthetic
  sequences, smoke inputs, and approved dev data only.

## Research Baseline

All conclusions in this plan are pinned to:

| Surface | Revision | Evidence |
| --- | --- | --- |
| Research apparatus | `rayline-ai/m4-alpha-route-2@c0a113a4` | C82 gate/postmortem/performance history |
| Rayline implementation | `davidvgilmore/rayline@9187b0ad7c504934a627486bc8bf67ac2e251e6f` | PR `rayline-ai/rayline#59` |
| llama.cpp enhancement | `davidvgilmore/llama.cpp@8c5d694fe7e28e8973349b634a72fe7683ecc940` | cumulative mean pooling state |
| Semantic Router | `vllm-project/semantic-router@c224fcfc892405d1ebc4794f206bf763c5835aa4` | current selection/extproc/config seams |
| vLLM | `vllm-project/vllm@1206891822ca8befe421879a4230ef42a3fc93be` | current pooling, hybrid APC, plugin APIs |
| C82 artifact | private immutable `rayline.mtrouter-runtime.v3` snapshot | mounted at runtime; repo/revision stay in private deployment config |

Local validation completed before this plan:

- `rustup run 1.88.0 cargo test -p rayline-mtrouter --locked`: pass.
- Rayline C82 doctor on Metal: ready; head parity `1.0`, maximum head drift
  `4.768e-7`; encoder parity `1.0`, cached/clean maximum embedding drift `0`,
  adjusted top-two-gap drift `0.001377`; 8,192-token cached delta about 75 ms
  versus about 1,599 ms clean on this machine.
- Semantic Router `make agent-validate`: pass.
- Semantic Router development images built with `make vllm-sr-dev`.
- The CPU agent stack started, passed `make agent-smoke-local`, and stopped
  cleanly. The harness currently needs `.venv-agent/bin` on `PATH`.
- vLLM source is prepared locally. Qwen3.5's CUDA/hybrid pooling path cannot be
  validated on the development Mac; GPU correctness and performance tests in
  this plan run through Modal Linux GPU jobs.

### Why vLLM PR #40804 Is Input, Not the Implementation

Open PR `vllm-project/vllm#40804` demonstrates a valuable primitive:
FP32 prompt-hidden-state sums associated with physical prefix-cache blocks.
It is not suitable to cherry-pick:

- it adds an embedding side channel to the generation runner rather than
  making the pooling runner correct;
- it explicitly does not support multi-turn use;
- it knowingly skips decode-state contributions that later turns could reuse;
- it indexes only cache group zero and notes that multi-group models need more
  work;
- it moves results through `kv_transfer_params`;
- its pre-run check is failing and it has no maintainer approval.

Before changing vLLM, re-check #40804 and #48214 for movement. Coordinate with
or build on an accepted upstream primitive when one exists; do not create a
second incompatible cache-summary mechanism.

### Phased Encoder Delivery

The encoder path has three intentionally separate rungs:

1. **Rung A -- bootstrap without a vLLM core change.** Request `token_embed`
   (`ALL` pooling), let the IO Processor compute the masked FP32 mean and L2
   normalization in-process, and keep APC disabled. vLLM already carries ALL
   hidden states across chunked prefill. This proves tokenizer, Qwen3.5 pooling
   conversion, serializer, head, policy, and end-to-end routing correctness.
   It is not a production serving shape: a 262,144-token BF16/1024 history is
   roughly 0.5 GiB of hidden state and every turn recomputes the full history.
2. **Rung B -- required causal-MEAN correctness.** Add an FP32 running
   sum/count to pooling request state so `MEAN` consumes scheduled slices and
   emits only when the prompt completes. This is contract-critical because a
   maximum-length history cannot be issued as one viable prefill step. It also
   fixes the current inconsistency in which hybrid pooling advertises chunked
   prefill support while `MeanPool` rejects partial prefill at runtime.
3. **Rung C -- phase-gated APC economics.** Add FP32 prefix-block summaries,
   hybrid-group alignment, and the complete cache lifecycle only after Rung B
   canary data and expected router load show that uncached per-turn inference
   misses the declared latency, memory, or cost budget. This is a separate
   upstream design and review unit. It must not hold Rungs A/B or the
   experimental Semantic Router integration hostage to maintainer latency.

The Rung C gate records expected QPS/context distribution, cold and repeated
turn latency, GPU memory, raw Modal cost, a maximum implementation/maintenance
budget, and the responsible human maintainer. If the gate stays closed, record
APC as an explicit deferred optimization with the evidence and trigger that
would reopen it; APC-only acceptance rows do not block the experimental
feature exit. A production-readiness claim does require Rung C.

An alternative considered for Rung C is router-side accumulation: vLLM would
return suffix sums and token ranges while the fenced episode store retains
8,192-aligned FP32 checkpoints. This reduces worker cache-lifecycle complexity
but makes numerical encoder state authoritative in Redis and couples cache
reconciliation to episode transactions. Prefer worker-side content-addressed
summaries if Rung C opens; revisit router-side accumulation only through an
explicit plan amendment with parity and state-size evidence.

## Architecture

```text
client
  │ explicit episode id + chat/response request
  ▼
Envoy ──► Semantic Router ext_proc
            1. normalize request into frozen ARC turns
            2. acquire fenced episode transaction
            3. call ARC vLLM /pooling endpoint
            4. run frozen F32 head + policy
            5. shape and dispatch selected arm
            6. commit transaction on upstream 2xx headers; otherwise abort
                         │
                         ▼
              vLLM pooling deployment
                IO plugin: token-blocks-v2
                Qwen3.5-0.8B BF16
                Rung A: ALL -> plugin FP32 mean
                Rung B: chunked causal MEAN
                Rung C: hybrid APC FP32 sums (gated)
```

The RPC boundary is deliberate. Semantic Router remains a routing control
plane and vLLM remains the model inference engine. The router must not load the
Qwen encoder through Candle, ONNX, or a Python subprocess.

## Frozen C82 Contract

The runtime must reject startup when any required artifact field, tensor,
shape, order, revision, or hash differs.

### Encoder and Serialization

- model: `Qwen/Qwen3.5-0.8B`
- revision: `2fc06364715b967f1860aea9cf38778875588b17`
- dtype: BF16
- attention reference: SDPA
- output: masked mean, L2-normalized, 1024 dimensions
- maximum serialized tokens: 262,144
- minimum retained recent turns: 1
- minimum retained recent tokens: 64
- serialization: `mtrouter-token-blocks-v2`
- incremental checkpoint unit: 8,192 tokens
- reference physical decode batch: 512 tokens

Tokenization uses the pinned model tokenizer with
`add_special_tokens=false` and special-token parsing disabled for literal
content. Tokenize each block's header and content separately before
concatenation; joining the strings before tokenization changes BPE boundaries.
Each logical block appends the exact EOS token ID verified against the artifact
and GGUF reference:

1. The first user message becomes `[Task]\n<text>` and is not repeated in
   context.
2. When later turns exist, append `[Context]`.
3. Append `[Turn N - <role>]\n<text>` for each later turn. `N` increments on
   user roles.
4. Truncation retains the task prefix plus newest complete turns; when needed,
   retain the header and tail of the oldest included recent turn. Match
   Rayline's exact boundary behavior for 0/1-token budgets and EOS placement.

The protocol adapter must port Rayline's deterministic tool rendering,
including sorted compact JSON, ASCII escaping, `[tool_call <name>]`,
`[tool_result <name>]`, `[tool error]`, and Python-style `True`/`False`
coercion. OpenAI Chat Completions, Anthropic Messages, and Responses input must
first normalize to one `[]Turn{role,text}` contract:

- Anthropic Messages is authoritative. Match Rayline exactly: drop system and
  unknown roles; drop image and thinking blocks; preserve text order; resolve
  `tool_result.tool_use_id` through the cross-message tool-name map; join text
  and tool calls with `\n` and tool results with `\n\n`.
- OpenAI Chat Completions drops system/developer messages, preserves
  user/assistant text, expands assistant `tool_calls` in wire order, parses each
  JSON-string `arguments` value and renders it through the same canonical JSON
  path, and maps `role=tool` through `tool_call_id` to the recorded name.
  Malformed tool argument JSON fails closed because it has no Rayline-equivalent
  value.
- Responses preserves input/output item order, maps message text to the same
  user/assistant turn rules, maps `function_call` and
  `function_call_output` through `call_id`, and applies the Chat Completions
  argument/result rules. System/developer instructions are dropped.
- Rich blocks with a Rayline-defined drop behavior are dropped for parity;
  structurally unknown items, unresolved tool IDs, and malformed supported
  fields fail closed. Do not invent visible placeholders for omitted images or
  thinking.

Golden fixtures must prove that equivalent protocol requests produce identical
turns and token IDs. Include multiple tool calls, cross-message name
resolution, non-string text coercion, Unicode/ASCII escaping, literal
`<|im_end|>` content, exact EOS, separately tokenized header/content, dropped
rich blocks, and every 0/1-token truncation boundary.

### Head

- artifact schema: `rayline.mtrouter-runtime.v3`
- weights: `model.safetensors`; verify the SHA256 declared by the mounted
  immutable manifest
- all required tensors are F32
- `meta_mean` and `meta_std` remain required and shape/hash checked even though
  the frozen inference path does not consume them; document that intentional
  non-use so a cleanup cannot silently change artifact compatibility
- history embedding: 1024
- arm embedding: 64
- joint input: 1154 =
  `history[1024] + candidate[64] + previous[64] + same_arm[1] + log1p(turn)[1]`
- arm metadata encoder:
  `8 → Linear(32) → ReLU → Linear(32)`, concatenate a learned 16-vector,
  then `Linear(48,64) → LayerNorm(eps=1e-5)`
- Q network:
  `Linear(1154,256) → ReLU → Linear(256,256) → ReLU → Linear(256,1)`
- dropout `0.1` is training-only and must not execute at inference.
- Normalize the 1024-vector before head evaluation and reject zero,
  non-finite, or wrong-dimensional inputs.
- Stable first-index argmax is part of the contract.

The candidate arms and their immutable tensor order come only from the mounted
manifest. Public tests use synthetic arm IDs. Private provider/model IDs and
thinking variants stay in private deployment configuration and are
cross-checked against the manifest at startup.

### Policy

- previous-arm stay margin: `0.05`
- cold-switch penalty: `1.0 * estimated lost-cache USD`
- cold-switch upgrade exemption: enabled when the candidate input rate is
  greater than the previous arm's rate
- stay-margin upgrade exemption: disabled
- reference worker: the immutable logical arm declared by the artifact
- prices are the artifact's immutable cache-aware snapshot; mutable live prices
  may be observed but must not affect decisions
- expected prefix-cache hit ratio:
  `0.8` through 150 idle seconds, linear decay to `0.2` at 300 seconds, then
  `0.2`
- lost-cache cost for a candidate is:
  `miss_tokens * max(input_rate - cache_read_rate, 0)`
- apply cold-switch penalties first, choose stable argmax, then retain the
  previous arm when the tentative advantage is `<= 0.05`
- episode commit increments the turn, sets the previous arm, and updates only
  the selected arm's `{last_used,last_input_tokens}` warmth.

## Semantic Router Design

### Configuration

Add a typed block to `config.AlgorithmConfig` and the canonical/reference/docs
surfaces:

```yaml
algorithm:
  type: rayline_arc
  on_error: fail_closed
  rayline_arc:
    artifact_dir: /var/lib/vllm-sr/rayline-arc
    artifact_revision: ${RAYLINE_ARC_ARTIFACT_REVISION}
    encoder:
      base_url: http://rayline-arc-encoder:8000
      model: Qwen/Qwen3.5-0.8B
      model_revision: 2fc06364715b967f1860aea9cf38778875588b17
      expected_build_id: ${RAYLINE_ARC_VLLM_BUILD_ID}
      connect_timeout_seconds: 5
      total_timeout_seconds: 180
      max_retries: 1
    episode:
      id_header: x-rayline-episode-id
      backend: redis
      key_prefix: vsr:rayline-arc:
      acquire_timeout_seconds: 30
      lease_ttl_seconds: 60
      idle_ttl_seconds: 900
      max_in_memory_episodes: 1024
```

Validation rules:

- `type=rayline_arc` requires its block and `on_error=fail_closed`.
- The public schema accepts a mounted artifact revision and verifies it against
  the mounted manifest; it does not hardcode the private C82 commit in source.
  This deployment profile pins the immutable C82 revision.
- The decision has exactly the artifact-declared `modelRefs` in manifest arm
  order.
- Every logical arm resolves to a configured backend and provider model ID
  matching the artifact. The two thinking variants remain distinct logical
  arms even when their provider model ID is shared.
- Redis is required outside explicit development mode. The memory backend is
  bounded, reaped, documented as single-replica/sticky-session only, and uses
  the same transaction interface.
- The episode header is explicit and nonempty. Do not fall back to the current
  changing message-hash session ID.
- Secrets use existing credential resolution or `*_env` fields and are never
  serialized by canonical config export.
- Router Learning is disabled for ARC decisions. Validation rejects a
  configuration that could run `applyRouterLearning` after ARC selection, and
  an integration test proves no post-selection model override is possible.
- The encoder total timeout is a request budget, not a fixed per-attempt
  timeout. The production value is frozen after the maximum-context Modal
  canary with headroom for measured p99; 30 seconds is not an acceptable
  assumed default for the 262,144-token path. Retries consume the same budget.

### Contribution and Ownership Boundary

Keep public OSS changes usable without access to the private C82 bundle:

- Semantic Router upstream-intended work is schema-generic artifact loading
  for `rayline.mtrouter-runtime.v3`, synthetic public fixtures, fail-closed
  algorithm plumbing, the episode-store interface/backends, revision/capability
  readiness, and generic artifact-owned dispatch validation.
- The IO Processor package and serializer are upstreamable when their fixtures
  contain no private artifact data. Deployment-only C82 pins, model/provider
  arm values, private goldens, and HF download credentials stay in private
  configuration or mounted runtime artifacts.
- vLLM upstream work is limited to generally correct pooling behavior: the
  hybrid support-flag fix, Rung B chunked causal mean, and, if opened, Rung C
  prefix-summary lifecycle. No C82 names, private pins, or Semantic
  Router-specific channels belong in vLLM core.

Do not open either public PR with private artifact contents, private golden
values, provider credentials, deployment URLs, or a source-code dependency on
the private HF repository. Commits intended for Semantic Router or vLLM PRs
must follow each repository's contribution and sign-off rules.

### Artifact Loader and F32 Runtime

Create `pkg/selection/raylinearc` as the coherent C82 concern, with a thin
`selection.Selector` adapter in `pkg/selection/rayline_arc.go`.

The loader:

- accepts a mounted local artifact directory; deployment tooling downloads the
  private HF snapshot before router startup;
- verifies `manifest.json`, every core file hash, schema, model/revision,
  encoder contract, arm order, prices, policy, and golden declarations;
- implements the minimal SafeTensors reader needed here: bounded 8-byte header
  length, bounded JSON header, checked offsets, exact tensor allowlist, F32
  only, no overlapping/out-of-range data;
- rejects missing and unexpected required tensors and validates every shape;
- constructs immutable in-memory tensors once at startup;
- runs head goldens at startup and keeps the algorithm unready on any drift.

Do not route head inference through the existing legacy `modelselection`
package. The current selection seam is `pkg/selection`.

The implementation is generic to the declared runtime schema. Tests use a
small synthetic, redistributable manifest/tensor fixture; the private C82
artifact is mounted only in private startup and acceptance jobs.

### Encoder Client

Use a dedicated client, not `pkg/embedding.Provider`. It posts structured
`{episode_id_hash, turns}` data to the vLLM `/pooling` plugin endpoint and
expects:

```json
{
  "embedding": [1024 finite floats],
  "serialized_tokens": 123,
  "full_history_tokens": 123,
  "truncated_tokens": 0,
  "cached_prefix_tokens": 0,
  "model_revision": "2fc063...",
  "engine_build_id": "vllm@<commit-or-image-digest>",
  "io_plugin_version": "rayline-arc-io@<version>",
  "pooling_capabilities": ["all_plugin_mean", "chunked_causal_mean"]
}
```

The raw episode ID is not needed by vLLM and must not cross the boundary; a
request correlation ID or SHA256 may be sent for diagnostics. Bound response
bytes, reject wrong dimensions, model/tokenizer/plugin/build revisions,
capability sets, or non-finite values, use contextual timeouts, and retry only
pre-response transport failures. Retry behavior must not multiply an unbounded
request budget. Readiness verifies the exact vLLM image digest or build commit,
plugin version, serializer version, model revision, tokenizer fingerprint, EOS
ID, and active Rung capabilities; verifying the model alone is insufficient.

### Strict Selection Errors

`selectModelFromCandidates` currently converts invalid context, missing
selectors, selector errors, invalid results, and unknown selected models into
the first candidate. That behavior is incompatible with an artifact-pinned
orchestrator.

Change the call chain to return an error and honor `algorithm.on_error`:

- preserve today's default-candidate behavior for all existing algorithms;
- for `fail_closed`, return an immediate JSON 503 before upstream dispatch;
- record a bounded error class, not prompt or artifact content;
- never advance episode state, record a successful session decision, or route
  the first arm after an ARC error.

`rayline_arc` must be registered as `TierExperimental` in the selection and
config catalogs, fragment catalog, reference config, docs, and tier tests.

### Episode Transaction

Replace eager ARC use of `recordAgenticSessionDecision` with an explicit
transaction lifecycle:

```go
type EpisodeStore interface {
    Prepare(ctx, episodeID, workerCount) (Lease, EpisodeState, error)
    Commit(ctx, lease, expectedVersion, EpisodeState) error
    Abort(ctx, lease) error
}
```

`Prepare` serializes requests for the same episode and returns a fenced lease.
The prepared lease lives on `RequestContext`; exactly one terminal path owns
finalization.

- Commit synchronously on the first upstream response headers whose status is
  2xx, before returning the ext_proc response.
- Abort on non-2xx headers, pre-header transport failure, immediate response,
  handler error, panic recovery, EOF/cancel/deadline before headers, or lease
  loss.
- A streaming body failure after successful 2xx headers does not undo the
  commit; this matches Rayline's response-header commit point.
- Add a deferred idempotent finalizer in `OpenAIRouter.Process` so no exit path
  leaks a lease.
- Redis uses owner tokens, monotonically fenced versions, atomic Lua
  compare-and-set/commit/release, bounded acquisition, lease renewal, and TTL.
  A stale or lost owner can never commit.
- Persist warmth timestamps as validated UTC Unix epoch milliseconds from the
  router's wall clock; a process-monotonic clock cannot cross replicas or
  restarts. Clamp negative idle durations caused by bounded clock skew, reject
  implausibly future timestamps, and test restart/skew behavior with a fake
  clock.
- The memory backend uses per-episode mutexes, bounded LRU/idle reaping, and
  idempotent release. Unknown failures still reach a terminal release path.
- Store only arm/turn/warmth metadata; never prompts, tool arguments, API keys,
  or embeddings.

After a successful commit, mirror the result into existing session telemetry.
Do not let that mirror become the authoritative C82 state.

### Dispatch Contract

Normal endpoint selection and credential injection stay in Semantic Router.
An ARC-specific body mutator applies the artifact-owned worker contract after
model selection:

- exact upstream provider model ID;
- exact OpenRouter provider order;
- `allow_fallbacks=false`;
- `require_parameters=true`;
- exact reasoning/effort body fields for thinking-on/off variants;
- artifact retry limits remain an observed contract, but retries must have one
  authoritative owner. Prefer the existing data-plane/provider owner and
  reject conflicting duplicate retry configuration.

Cross-check configured endpoint/model mappings against the artifact at startup.
Fail startup rather than silently routing an arm to a different provider,
reasoning mode, fallback policy, or price identity.

### Observability and Privacy

Add metrics and structured trace fields for:

- artifact and encoder revision;
- selected/prior arm index and hashed logical ID;
- raw and adjusted scores;
- switch cost, miss tokens, stay/exemption flags;
- serialized/full/truncated/cached token counts;
- encoder/head/lease/upstream-header latency;
- episode transaction outcome and bounded failure class;
- state backend, cache hit mode, and eviction counts.

Never log raw episode IDs, prompts, tool input/results, embeddings, keys,
authorization headers, or full mutated request bodies. Hash episode IDs with
SHA256 only where correlation is explicitly enabled. Expose an ARC readiness
component that fails until artifact verification, head goldens, encoder
model/tokenizer/build/plugin/capability probes, dimension probe, and
episode-store probe all pass.

Start vLLM with prompt/request logging disabled and no debug endpoint that
echoes token IDs or bodies. Extend the privacy canary across Envoy, Semantic
Router, Redis diagnostics, the ARC vLLM container, and fake-provider logs; the
gate fails if any raw prompt, tool payload, episode ID, embedding, key, or
authorization canary appears.

## vLLM Design

### IO Processor Plugin

Add an installable Python package under
`src/vllm-plugins/rayline_arc_io/` in this repository and install it into the
ARC vLLM image through the `vllm.io_processor_plugins` entry-point group.

The plugin:

- owns typed request/response validation for `/pooling`;
- stores `renderer.get_tokenizer()` and verifies tokenizer/model revisions plus
  a real EOS token at startup;
- ports the Rayline token-block serializer exactly and returns
  `TokensPrompt(prompt_token_ids=...)`;
- negotiates an explicit serving Rung: Rung A requests `token_embed`/`ALL` and
  performs the masked FP32 mean plus L2 normalization in the output processor;
  Rungs B/C request causal `MEAN` with activation enabled;
- returns the normalized embedding plus token/cache metadata;
- rejects empty histories, unsupported roles/content, inputs over request byte
  limits, and revision drift;
- includes pure Python golden tests that compare token IDs and truncation
  boundaries to the immutable artifact/Rayline fixtures.

Do not use `EmbedIOProcessor.enable_chunked_processing`: it splits text into
independent model calls and averages their embeddings, losing causal context.

### Core Cached-Mean Primitive

Generalize causal `MEAN`; do not add a C82-only pooling enum and do not route
through generation's `kv_transfer_params`. Land Rung B and Rung C as separate
changes with separate tests and review:

Rung B:

1. Extend `PoolingStates` with an FP32 running sum/count and explicit lifecycle.
2. Make `MeanPool.forward` consume only the scheduled slice, accumulate by
   request, return output only when the prompt is complete, and remove the
   current partial-prefill error.
3. Correct `ModelConfig.is_chunked_prefill_supported` so advertised hybrid
   pooling support cannot reach a runtime path that rejects partial prefill.
4. Enable causal `MEAN` chunked prefill only after the running-state capability
   is present.

Rung C, only after its gate opens:

5. Add a worker-side prefix-pool cache that stores FP32 hidden-state sums for
   completed logical cache blocks. It is allocated only for causal mean pooling.
6. Seed a new request's running sum/count from every fully cached prefix block,
   then add newly computed suffix tokens.
7. Bind summaries to one explicitly selected cache group whose logical token
   coverage is valid for all groups. Handle block allocation, zero, copy/fork,
   eviction, preemption/resume, and request abort. Never infer lifecycle only
   from “slot zero happened to be written.”
8. For hybrid models, validate that the scheduler's computed prefix is aligned
   across attention and GDN groups. Reject unsupported layouts rather than
   indexing group zero optimistically.
9. Enable causal `MEAN` prefix caching in
   `ModelConfig.is_prefix_caching_supported`, including hybrid models only when
   the cache summary and hybrid alignment capabilities are present.
10. Preserve zero allocation and zero hot-path work for other pooling types and
   generative runners.

The ARC serving profile uses Qwen3.5 hybrid prefix caching with
`mamba_cache_mode=align`, an 8,192-token Mamba block/checkpoint, and
`max_num_batched_tokens >= 8192`. Leave `prefix_match_unit` at a value valid for
all cache groups; do not force 8,192 when attention blocks are smaller.

### Upstream Coordination

The vLLM change is a separate, human-owned upstream contribution. Follow
vLLM's `AGENTS.md`: search duplicates again, open/attach an issue or RFC before
substantial work, explain the relationship to #40804, and do not publish an
agent-only PR.

Human-owned publication is non-blocking for work that does not depend on a
maintainer decision. The coding agent prepares a complete issue/RFC draft,
patch evidence, and duplicate analysis for a human to file; it records the
pending action and proceeds with synthetic-fixture Semantic Router work, Rung
A, and isolated Rung B tests. Split the current hybrid chunked-support
advertisement/runtime contradiction into the smallest credible bugfix or issue
before proposing cache summaries. Re-check upstream state at each vLLM loop,
but do not idle while another plan task is actionable.

The experimental Semantic Router feature exit requires Rungs A/B. Rung C is
required only when its phase gate opens and for any production-readiness claim.

## Planned Code Surfaces

Semantic Router:

- `src/semantic-router/pkg/config/decision_config.go`
- `src/semantic-router/pkg/config/validator_decision.go`
- `src/semantic-router/pkg/config/routing_surface_catalog.go`
- `src/semantic-router/pkg/selection/rayline_arc.go`
- `src/semantic-router/pkg/selection/raylinearc/{types,manifest,safetensors,head,policy,encoder_client,episode_store,episode_memory,episode_redis}.go`
- `src/semantic-router/pkg/selection/factory.go`
- `src/semantic-router/pkg/extproc/req_filter_classification*.go`
- `src/semantic-router/pkg/extproc/{request_context,processor_core,processor_res_header,session_policy}.go`
- new focused extproc files for ARC input normalization, dispatch, and
  transaction finalization
- `config/algorithm/selection/rayline-arc.yaml`
- `config/config.yaml`, configuration docs, tutorial, Helm/Compose/agent
  profiles, readiness and metrics docs
- `src/vllm-plugins/rayline_arc_io/`
- ARC vLLM image/deployment assets and Modal GPU acceptance entrypoint

vLLM:

- `vllm/v1/pool/metadata.py`
- `vllm/model_executor/layers/pooler/seqwise/methods.py`
- `vllm/v1/worker/gpu_model_runner.py`
- a focused worker cache-summary module (Rung C only)
- scheduler/model-runner output only where lifecycle metadata is genuinely
  required
- `vllm/config/model.py` and argument/config validation
- unit tests for mean accumulation, block lifecycle, and hybrid alignment
- GPU E2E tests for full/chunked requests plus APC-hit, preempted, aborted, and
  concurrent cache-summary requests only if Rung C opens

Keep modules cohesive. Do not extend legacy `pkg/modelselection`, fold episode
storage into extproc orchestration, or mix provider dispatch into the numerical
head.

## Numeric Validation Policy

Every tolerance names the compared paths, hardware/dtype, repetitions, and
decision consumer. Do not reuse one threshold across unlike comparisons:

- **Artifact head golden versus Go head:** use the manifest-declared absolute
  tolerance and require selected-arm parity `1.0`. Record whether the reference
  and implementation use fused multiply-add; Rust's `f32::mul_add` and a Go
  scalar loop need not be bit-identical. Required-but-unused artifact tensors
  still undergo shape/hash checks.
- **Same vLLM build, full versus chunked (Rung A/B):** compare raw means,
  normalized embeddings, head scores, adjusted top-two gaps, and selected arms.
  Run the Modal canary first, repeat enough times to characterize CUDA
  reduction/atomic nondeterminism, then freeze the smallest tolerances that
  cover observed variance plus explicit headroom before the broader matrix.
- **Same vLLM build, cold versus APC hit (Rung C):** freeze a separate
  cache-path tolerance from canary evidence. The existing `<= 0.005` adjusted
  top-two-gap limit is the outer decision-safety bound; do not assume
  `<= 1e-5` score drift is achievable before measurement.
- **Cross-engine Rayline llama.cpp versus vLLM CUDA:** require selected-arm
  parity and adjusted top-two-gap drift `<= 0.005` on tie-free goldens. Report
  score/embedding drift, but do not apply the within-engine `1e-5` target.

Numerical parity fixtures must be tie-free by more than twice their declared
score/gap tolerance. Test stable first-index tie behavior separately with a
synthetic exact-tie head. A threshold may be tightened after evidence; it is
never widened merely to make a failing implementation pass.

## Acceptance Matrix

### Deterministic CPU Tests

- SafeTensors malformed-header, overflow, overlap, wrong dtype/shape/hash, and
  tensor-allowlist rejection.
- F32 head goldens meet the manifest-declared Go-versus-artifact tolerance and
  selection parity `1.0`, with FMA behavior recorded.
- Policy parity for first-index ties, cold/warm decay boundaries, upgrade
  exemption, stay margin equality, immutable prices, and commit-only state.
- Protocol normalization and tokenizer fixtures match Rayline token IDs exactly,
  including separate header/content tokenization, literal special-token text,
  protocol role/drop rules, multiple tools, Python-style scalar coercion,
  Unicode, sorted tool JSON, EOS, turn numbering, and every truncation boundary.
- Strict algorithm errors produce 503 and never fall back to arm zero.
- Config/canonical round trip, secret redaction, catalogs, fragments, reference
  config, and docs contracts.
- Memory and Redis transaction tests cover contention, timeout, renewal, lost
  lease, stale commit, panic, cancel, non-2xx, idempotent finalization, reaping,
  and one successful commit.

### vLLM GPU Tests

Run on pinned Linux CUDA hardware through Modal:

- Rung A `token_embed`/ALL plus plugin FP32 mean covers short, multi-chunk, and
  maximum-contract inputs with APC disabled.
- Qwen3.5-0.8B loads as a pooling model at the pinned revision.
- Full prefill output is finite, normalized, and 1024-dimensional.
- Rung B full and chunked means meet the frozen within-vLLM tolerance and
  selected-arm parity `1.0`; repeated runs bound CUDA reduction variance.
- If Rung C opens, cold and APC-hit paths meet their separately frozen
  cache-path tolerance, adjusted top-two-gap drift `<= 0.005`, and selected-arm
  parity `1.0`.
- If Rung C opens, cached-prefix accounting equals the scheduler's actually
  reused tokens; Qwen hybrid groups work in align mode at 8,192-token
  boundaries; and block reuse/copy/eviction, abort, preemption/resume, mixed
  cached/cold batches, and concurrent identical prefixes cannot consume stale
  sums.
- Non-MEAN pooling and generation benchmarks show no allocated summary cache
  and no material regression.
- Record clean versus cached latency and GPU memory on the same hardware and
  commit, labeled as a development measurement. Set performance thresholds
  only after this canary evidence exists.

### End-to-End Semantic Router Tests

- Start Envoy, Semantic Router, Redis, ARC vLLM, and fake provider backends.
- Route every artifact golden case and prove selected arm, thinking mode,
  provider pin, fallback policy, and request body.
- Prove Router Learning is disabled/bypassed and cannot override the ARC arm
  after selection.
- Two concurrent requests for one episode serialize; different episodes do not.
- 2xx response headers commit exactly once.
- 4xx/5xx, encoder failure, head failure, provider transport failure, client
  cancel before headers, Redis loss, and process panic never advance state.
- A post-2xx streaming abort preserves the already committed state.
- Restart with Redis and continue the prior episode version; memory mode is
  explicitly tested as non-durable.
- Logs/traces across Envoy, Router, Redis, vLLM, and fake providers contain
  hashes and numeric telemetry but no test prompt, tool payload, embedding, raw
  episode ID, or key canary.

Required Semantic Router gates include changed Go package tests, config/docs
contract tests, E2E, `make agent-lint`, `make agent-validate`, and an
`agent-report` for the exact changed-file set. Required vLLM gates follow its
test-selection tooling plus the dedicated CUDA tests. Do not run paid model
evaluation or a frozen holdout as part of implementation validation.

## Exit Criteria

- The exact immutable artifact passes startup verification and head goldens.
- Semantic Router calls vLLM—not Candle/ONNX/local Python—for every ARC encoder
  decision.
- The IO plugin's token IDs match the Rayline serializer fixtures exactly.
- Rungs A/B meet their full/chunked numerical and arm-parity gates. If the Rung
  C gate opens, APC-hit paths also meet their separately frozen gates; if it
  stays closed, the deferral evidence and reopening trigger are recorded.
- Readiness verifies the exact vLLM build/image, IO plugin, serializer,
  tokenizer/EOS, model revision, and required serving-Rung capabilities.
- Episode state is serialized, fenced, bounded, terminal on every failure, and
  committed only on upstream 2xx headers.
- Every arm dispatches the artifact-pinned model, provider, thinking mode,
  fallback policy, and immutable price identity.
- ARC errors fail closed; no path silently selects the first/default candidate.
- Existing algorithms and non-MEAN vLLM workloads preserve behavior and
  performance.
- CPU, Redis, GPU, and full-stack tests pass; readiness and privacy tests pass.
- Router Learning cannot override ARC, and vLLM prompt logging is disabled and
  covered by the cross-container privacy canary.
- Reference config, deployment docs, operational rollback, and experimental
  status are documented.
- Public Semantic Router/vLLM changes contain no private artifact contents,
  pins, credentials, or goldens.
- The plan's task list is complete with commit/test evidence recorded here.

## Task List

- [x] T1 Re-check upstream duplicates; prepare the human-owned issue/RFC
      package, including the hybrid chunked-support contradiction and the
      relationship to #40804/#48214; record any pending human publication
      action without blocking independent tasks.
- [x] T2 Add frozen artifact loader, SafeTensors reader, F32 head, policy, and
      CPU golden/parity tests using a schema-generic loader and public synthetic
      fixture.
- [x] T3 Add typed `rayline_arc` config, validation, catalogs, fragment,
      canonical/reference surfaces, experimental tier, engine/capability pins,
      and Router Learning exclusion.
- [ ] T4 Add the specified protocol-to-turn normalization and cross-protocol
      Rayline golden fixtures, including drop/coercion/tokenization traps.
- [ ] T5 Package the ARC vLLM IO Processor plugin, exact token serializer, and
      Rung A `token_embed`/ALL plugin-side FP32 mean.
- [ ] T6 Establish Rung A full/multi-chunk/max-context Qwen3.5 correctness on
      Modal; freeze timeout and within-path numeric budgets from the canary.
- [ ] T7 Implement vLLM chunked causal mean state and unit tests.
- [ ] T8 Run the Rung C phase gate. If opened, implement prefix-block FP32 sums,
      hybrid cache lifecycle, APC support, and CUDA tests; if closed, record the
      owner, evidence, and quantitative reopening trigger.
- [ ] T9 Add ARC encoder client, selector adapter, strict error propagation,
      build/plugin/capability readiness, privacy-safe telemetry, and Router
      Learning bypass.
- [ ] T10 Add memory and Redis episode transactions plus terminal-path tests.
- [ ] T11 Add artifact-owned ARC dispatch mutation and provider-contract
      validation.
- [ ] T12 Add Compose/Helm/Modal deployment profiles and full-stack E2E.
- [ ] T13 Run all affected Semantic Router and vLLM gates; record exact commits,
      branches, hardware, commands, Modal cost, results, and rollback procedure.
- [ ] T14 Re-run the acceptance matrix on the merged code paths and close this
      plan only when every required or opened-phase exit criterion has evidence.

## Next Action

T4: add the exact protocol-to-turn normalization contract and cross-protocol
public fixtures, including Rayline drop rules, tool rendering/coercion,
malformed-field failure, Unicode, EOS, turn numbering, and truncation
boundaries. The human publication action recorded in Loop 1 remains
non-blocking.

## Loop Evidence

### Loop 1 — T1 upstream duplicate audit and handoff (2026-07-28)

Status: complete. No vLLM core or runtime implementation changed.

Live upstream findings:

- `vllm-project/vllm#40804` remains open and conflicted at
  `b42df0395f6bc2d947ec739be61879d9687abb86`. It couples FP32 prefix sums to
  the generation runner and `kv_transfer_params`; maintainer comments on
  2026-05-18/21 explicitly prefer keeping this behavior out of GPU-runner core.
- `vllm-project/vllm#48214` remains open and conflicted at
  `521cbfd8cba1fe464ee6c34fef32ddf77816ea55`. It fixes an async host-buffer
  race between pooling prefill chunks; it does not implement sequence-MEAN
  accumulation.
- Open duplicate searches for `mean pooling chunked prefill` and
  `MeanPool partial prefill` found no causal-MEAN implementation. `#48791`
  concerns Model Runner V2 sequence-pooling enablement, not accumulated
  causal MEAN.
- On current vLLM `98e91a9600eb75b2de14ef27f13b10088d1a1279`,
  `ModelConfig.is_chunked_prefill_supported` still returns support for hybrid
  pooling models while `MeanPool.forward` rejects every partial prefill.
- `#30672` is closed in favor of hidden-state RFC `#33118`; `#33118` was closed
  by `#33736`. Those paths address hidden-state extraction/generation, not the
  pooling runner contract required here.

Human-ready handoff draft:

> **Title:** [RFC] Correct chunked causal MEAN pooling for hybrid pooling
> models
>
> **Problem:** vLLM's pooling configuration advertises chunked prefill for
> hybrid models, but `MeanPool.forward` raises on partial prefill. Long causal
> embedding requests therefore fail at runtime despite passing configuration
> validation. A 262k-token pooling model cannot rely on one prefill step.
>
> **Proposed first contribution:** keep the change inside the pooling runner.
> Add request-scoped FP32 sum/count state to `PoolingStates`; accumulate only
> scheduled prompt slices; emit the mean only when the prompt completes; clean
> state on completion/abort; and advertise hybrid causal-MEAN support only when
> this capability is present. Preserve zero allocation/work for other pooling
> types and generation. Do not add a generation response side channel,
> `kv_transfer_params`, C82 names, or cache summaries.
>
> **Relationship to existing work:** #40804 provides useful FP32 accumulation
> ideas but is generation-side, multi-turn-incomplete, conflicted, and rejected
> by maintainers in its current runner shape. #48214 fixes a separate buffer
> lifetime race and should be incorporated/retested if it lands. #48791 is a
> Model Runner V2 enablement change, not causal-MEAN state.
>
> **Validation:** unit tests for full versus 2/3-chunk accumulation, mixed
> request completion, abort cleanup, FP32 state, and unchanged non-MEAN paths;
> then a pinned CUDA canary for full/chunked parity on a causal pooling model.
> Prefix-cache summaries are a later, separately gated RFC.

Pending human action: review and file the RFC/issue, then post it to the vLLM
contributors channel and identify a maintainer. The coding agent must not
publish it or an upstream PR.

Repository evidence:

- Semantic Router lane: `rayline/pl-0039`, based on
  `vllm-project/semantic-router@6d2bb8ff` before this loop's commits.
- vLLM fork created: `davidvgilmore/vllm`; implementation lane
  `rayline/pl-0039-causal-mean`, based and pushed at
  `98e91a9600eb75b2de14ef27f13b10088d1a1279`.
- Commands: `make agent-report ENV=cpu ...`; GitHub live PR/issue views and
  duplicate searches; source inspection with `rg`/`sed`;
  `make agent-lint CHANGED_FILES="..."`; `make agent-validate`;
  `make agent-ci-gate CHANGED_FILES="..."`. All gates passed, including
  Markdown/YAML formatting, structural/security checks, and
  `go test ./pkg/config/... -run TestReferenceConfig -count=1`.
- Semantic Router task commit:
  `ab89b1c28f894938d0261dd5b2c9e3a921b3baa0` (signed off), pushed to
  `davidvgilmore/semantic-router:rayline/pl-0039`. A direct push to
  `vllm-project/semantic-router` failed with HTTP 403; the authorized fork push
  succeeded without opening a PR.
- Hardware: local Apple development machine; no GPU/Modal work.
- Paid/Modal cost: `$0.00`; cost ceiling consumed: `$0.00`.

### Loop 2 — T2 artifact runtime, F32 head, and policy (2026-07-28)

Status: complete. No vLLM source changed.

Implementation:

- Added `pkg/selection/raylinearc` with a strict
  `rayline.mtrouter-runtime.v3` manifest loader, source/encoder/architecture/
  policy validation, immutable-price enforcement, lexical and evaluated
  symlink containment, bounded reads, and SHA256 verification over the bytes
  that are parsed.
- Added a bounded F32 SafeTensors reader that rejects malformed headers,
  duplicate/unexpected/missing tensors, non-F32 data, invalid or overflowing
  shapes, non-finite values, overlaps, gaps, and trailing bytes.
- Added the schema-generic ARC arm encoder, residual projection, layer
  normalization, and Q network. Startup goldens require selected-arm parity
  and manifest-declared score tolerance. The reference uses Rust
  `f32::mul_add`; Go uses scalar `math.FMA` in `float64` followed by an F32
  cast, so tolerance—not bit identity—is the portable contract.
- Added the exact cache-aware policy: stable first-index ordering, warmth
  decay boundaries, rounded decayed-prefix cost, cold-switch penalty and
  upgrade exemption, stay-margin equality and upgrade exemption, and
  commit-only state mutation.
- Added public synthetic fixture generation plus malformed-artifact, numeric,
  head, policy, state, and exact-tie tests. A separately mounted immutable
  runtime also passed the generic compatibility test; its private location,
  pins, arms, and contents were not copied into this repository or evidence.

Repository evidence:

- Semantic Router commits, all signed off:
  `5112689156f444c4846a0e33a8edec570706a9ab`,
  `c6310ed872d36da210ac97909e8f65c12560c772`,
  `3982a8538d1a459d603a8a59926655694d4744cc`, and
  `42517f50362a088d32e544191079990e4212ba9a`.
- vLLM lane remained clean at
  `98e91a9600eb75b2de14ef27f13b10088d1a1279`.
- Commands passed:
  `go test ./pkg/selection/raylinearc -count=1`;
  `go test -race ./pkg/selection/raylinearc -count=1`;
  `go vet ./pkg/selection/raylinearc`;
  the opt-in mounted-runtime compatibility test;
  `make agent-lint CHANGED_FILES="<eight ARC Go files>"`;
  `make agent-validate`;
  `make test-semantic-router`;
  and `make agent-ci-gate CHANGED_FILES="<eight ARC Go files>"`.
- `make agent-report` classified the exact change as
  `routing-policy-change` over `routing_policy, algorithm_selection` and
  required the local CPU stack. `make agent-dev ENV=cpu` built the arm64
  images. `make agent-serve-local ENV=cpu` and `make agent-smoke-local`
  passed with `.venv-agent/bin` on `PATH`, after an initial harness-only
  `vllm-sr: command not found` invocation. `vllm-sr stop` removed the stack
  and network cleanly.
- Hardware: local Apple Silicon, arm64 CPU/Docker; no CUDA/GPU execution.
- Paid/Modal cost: `$0.00`; cost ceiling consumed: `$0.00`.

### Loop 3 — T3 typed ARC configuration contract (2026-07-28)

Status: complete. No vLLM source changed.

Implementation:

- Added the typed Go and Python `rayline_arc` artifact, encoder, episode, and
  Redis configuration families. Validation requires immutable artifact/build/
  plugin pins, the exact Qwen model revision and serializer, bounded HTTP
  timeouts/retries, declared serving capabilities, secret-by-environment Redis
  credentials, and bounded development-only memory mode.
- Added strict decision contracts: `on_error=fail_closed`,
  `adaptations.mode=bypass`, and at least two unique `modelRefs` in artifact
  order. Exact manifest arm mapping remains a startup responsibility in T9.
- Added the experimental routing-surface tier, selection method, canonical
  reference config, public fragment, exhaustive coverage tests, CLI parity,
  docs, and the algorithm catalog/tutorial.
- The raw Dashboard config surface already preserves the typed block. The
  structured Signal DSL builder cannot express mandatory decision
  `adaptations`; TD045 records that intentional narrower surface and its
  compile/decompile/UI exit criteria. The canonical debt doc was added to the
  repository manifest.
- Live recheck: vLLM #40804 remains open at
  `b42df0395f6bc2d947ec739be61879d9687abb86`; #48214 remains open at
  `521cbfd8cba1fe464ee6c34fef32ddf77816ea55`. Neither replaces the planned
  causal-MEAN path.

Repository evidence:

- Semantic Router task commit:
  `9b20dbbc53c6a8905ea23d1e1f8b3679ada38cd3` (signed off).
- vLLM lane remained clean at
  `98e91a9600eb75b2de14ef27f13b10088d1a1279`.
- Commands passed:
  `go test ./pkg/config/... ./pkg/selection/... -count=1`;
  focused Python ARC/config tests (`39 passed`);
  `make agent-report ENV=cpu CHANGED_FILES="<28 files>"`;
  `make agent-lint CHANGED_FILES="<28 files>"`;
  `make agent-validate`;
  `make vllm-sr-test`;
  `make test-semantic-router`;
  `make vllm-sr-test-integration`;
  and `make agent-ci-gate CHANGED_FILES="<28 files>"`.
  The first changed-file lint found only three new cyclomatic-complexity
  violations and Python formatting/magic-number findings; validators were
  decomposed, constants added, and the exact gate passed on rerun.
- `make agent-dev ENV=cpu` rebuilt current images.
  `make agent-serve-local ENV=cpu` and `make agent-smoke-local` passed with
  `.venv-agent/bin` on `PATH`; `vllm-sr stop` removed every runtime,
  observability, Redis, and Postgres container plus the network.
- Elapsed loop duration: about 28 minutes from the prior evidence commit to the
  cohesive task commit.
- Hardware: local Apple Silicon, arm64 CPU/Docker; no CUDA/GPU execution.
- Paid/Modal cost: `$0.00`; cost ceiling consumed: `$0.00`.

## Operating Rules

- Re-read this plan's checkbox/evidence state at the start of every loop; work
  the first actionable unchecked task and keep its tests green before advancing.
- Update this task list and append exact branch/commit, command, hardware,
  duration, cost, and result evidence after every completed loop.
- Re-read the nearest `AGENTS.md` before touching config, extproc, or vLLM.
- Start each repository from current `main` on one descriptive lane branch and
  one clone. Commit cohesive passing slices with required sign-off and push the
  authorized lane the same day; never push directly to `main`.
- Preserve existing algorithm fallback behavior; strict failure is opt-in and
  mandatory for ARC.
- Use Modal for GPU experiments, right-size hardware, stop apps after tests,
  predeclare a run-specific time/cost ceiling, and record costs/artifacts
  according to the research apparatus rules.
- Never run the same unchanged paid/CUDA command a third time after two
  equivalent failures. Diagnose or change the approach, advance another
  independent task, and record the evidence/blocker before spending again.
- Human-owned issue/PR publication does not block independent implementation.
  Prepare the artifact and smallest required human action, then continue.
- Never delete or weaken a test/gate, widen a tolerance, or change the frozen
  artifact merely to obtain a pass.
- Never run locked/paid evaluation or use held-out task IDs in this lane.
- Never commit or publish the private bundle, C82 pins/goldens in public source,
  model cache, generated results, deployment URLs, credentials, or secrets.
- Stop only for a genuine external blocker; record the exact failing command,
  output class, and smallest user/maintainer action needed.

## Goal-Loop Prompt (Under 2,000 Characters)

```text
Implement PL-0039 at /Users/davidgilmore/Documents/vllm-semantic-router/docs/agent/plans/pl-0039-rayline-arc-orchestrator.md in a persistent goal loop. Use its checkbox/evidence ledger as memory. Complete Rungs A/B; execute Rung C only if its phase gate opens.

Each loop: inspect git state and re-read PL-0039 plus the nearest AGENTS.md; choose the first actionable task; re-check upstream duplicates; implement the smallest cohesive slice in vllm-semantic-router and, only where specified, /Users/davidgilmore/Documents/vllm; run focused tests and required gates; fix failures; update PL-0039 with checkbox, branch/commit, command, hardware, duration, cost, and result evidence. Use one lane branch/clone per repo, signed commits where required, and push authorized lanes the same day—never main.

Human issue/PR publication is non-blocking: prepare the handoff, record the pending action, and advance independent work. Never publish an agent-only vLLM PR. If the same paid/CUDA command fails twice unchanged, do not run it a third time; diagnose or change approach, advance another task, and record the blocker. Predeclare Modal time/cost ceilings and stop apps after tests.

Preserve the split: vLLM owns pinned Qwen inference, exact serialization, and pooling; Semantic Router owns schema-generic artifact verification, F32 head/policy, fenced transactions, fail-closed selection, and dispatch. Never use Candle/ONNX, fall back to arm zero, allow Router Learning to override ARC, weaken 2xx-only commit, log sensitive input, mutate the artifact, weaken a gate/tolerance, expose private artifacts/pins/goldens in either public repo, or spend holdout/paid evals.

Finish only when required CPU, Redis, Modal CUDA, privacy, and E2E gates pass on the exact final commits, plus APC gates if Rung C opened. Otherwise continue; if externally blocked after safe alternatives, leave the plan active and report evidence and the smallest human action.
```

## Related Docs

- `docs/agent/architecture-guardrails.md`
- `docs/agent/feature-complete-checklist.md`
- `src/semantic-router/pkg/config/AGENTS.md`
- `src/semantic-router/pkg/extproc/AGENTS.md`
- `https://github.com/rayline-ai/rayline/pull/59`
- `https://github.com/vllm-project/vllm/pull/40804`
- `https://github.com/vllm-project/vllm/pull/48214`
- `https://docs.vllm.ai/en/latest/models/pooling_models/`
- `https://docs.vllm.ai/en/latest/design/io_processor_plugins/`
- `https://docs.vllm.ai/en/stable/design/prefix_caching/`
