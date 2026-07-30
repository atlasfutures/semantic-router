# Rayline-on-vLLM Parity Implementation

Status: adopted child blueprint for PL-0041 RSP-003 through RSP-010. Updated
2026-07-30 after review against the transactional-routing architecture and the
published stateless encoder implementation.

This document refines the open cache choice in the
[Rayline vLLM serving boundary](rayline-vllm-serving-boundary.md). It uses the
immutable identities and pass/fail thresholds in the
[performance qualification contract](../benchmarks/rayline-vllm-performance-contract.md).

## Relationship to Transactional Routing

This is not a second top-level Rayline architecture. The transactional-routing
design is the parent system contract:

- Semantic Router owns request normalization, candidate gating, provider
  identity, credentials, dispatch, streaming, and execution truth.
- Pathfinder owns the policy artifact, committed **routing** state, pending
  selection receipts, same-episode fencing, and worker choice.
- dedicated Rayline vLLM owns reconstructible model execution and encoder
  acceleration;
- worker vLLMs or providers own generation and worker-local KV.

This document is the child workstream that replaces only Pathfinder's encoder
execution backend and then qualifies its cache. It does not move provider
lifecycle into Pathfinder or transaction authority into vLLM.

Conversation history follows the request, not the routing-state store.
Semantic Router supplies the complete current request history to Pathfinder on
every prepare. Pathfinder combines that input with committed routing facts such
as route version and previous worker, then sends the complete canonical turns
to the encoder. Pathfinder does not need to persist prompts to make encoder
cache loss reconstructible.

For the current MVP, no broader `TransactionalSelector` or cross-protocol
normalization project is required. The generic selection-transaction owner and
`rayline_remote` selector already provide the lifecycle seam. OpenAI Responses
and Anthropic Messages remain later protocol expansions, not parity
prerequisites.

## Recommendation

Keep the existing `rayline_remote` ownership model and replace only
Pathfinder's model-execution backend:

```text
                         authoritative state and policy
                    +-----------------------------------+
                    | Pathfinder                        |
                    | committed routing state           |
                    | prepare / commit / abort / settle  |
                    | C82 policy head and worker choice  |
                    +----------------+------------------+
                                     |
                         full canonical history on every encode
                                     |
                                     v
                    +-----------------------------------+
                    | dedicated Rayline vLLM             |
                    | Qwen3.5 encoder                    |
                    | causal-MEAN pooling                |
                    | continuous batching               |
                    | paired KV + mean-summary cache     |
                    +-----------------------------------+

 client -> Envoy -> Semantic Router -> selected worker vLLM/provider
                    |
                    +-- HTTP lifecycle, credentials, dispatch, and streaming
```

The cross-turn acceleration should use vLLM automatic prefix caching plus a
small causal-MEAN summary cache. A reusable checkpoint is a pair:

```text
token boundary C
    |
    +-- model state after tokens [0, C)       (vLLM KV/hybrid cache)
    |
    `-- FP32 sum(hidden[0:C]) and count C      (pooling summary cache)
```

Neither half is valid alone. On each full-history request, vLLM finds the
greatest boundary for which both halves exist, restores the pair, and executes
only the remaining suffix. Missing, evicted, mismatched, or corrupt state
causes a full or earlier-checkpoint recompute. It never changes the input or
the policy decision.

Use the current Modal implementation as the numeric and operational oracle
during qualification. Keep it as the rollback backend until the vLLM path
passes every gate in this document.

## Current Delivery State

The implementation is intentionally between Rungs A and B:

- PL-0040's transactional prepare/renew/commit/abort/settle lifecycle is
  complete for one Pathfinder replica.
- Pathfinder
  [`7f13de3d`](https://github.com/atlasfutures/pathfinder/commit/7f13de3d10855ea44245717f9ccb50d55ea40e93)
  provides one local/remote encoder seam, the strict stateless v1 pooling
  client, checkpoint identity validation, bounded response handling, and
  readiness propagation.
- The deterministic parity corpus and receipt runner are the next
  credential-free implementation slice.
- Actual pinned-model GPU parity, cross-episode batching, the paired-cache v2
  protocol, and the production-shaped five-service run are not complete.
- Pathfinder ADR 0059 remains proposed and requires human acceptance before
  the serving boundary is considered ratified.

## What “Parity” Means

Parity is not bit-identical floating-point output from two different inference
runtimes. It is the following observable contract:

| Dimension | Required result |
| --- | --- |
| Serialization | Identical `mtrouter-token-blocks-v2` tokens for identical canonical turns |
| Model identity | Exact pinned Qwen3.5 weights, tokenizer, dtype, context limit, and pooling contract |
| Policy identity | Exact C82 artifact, worker ordering, price snapshot, and adjustment logic |
| Selection correctness | Zero selected-worker flips on the frozen parity corpus |
| Numeric correctness | Adjusted top-two policy-gap drift at or below `5e-3` |
| State behavior | Same-episode turns are serialized; cache loss changes latency only |
| Incremental work | A warm strict extension replays no more than one 8,192-token tail before new tokens |
| Exact retry | An identical full token sequence may use a bounded cached final embedding after the paired-prefix MVP proves necessary |
| Memory | One enforceable engine-wide residency bound, no unbounded per-episode state |
| Failure | Restart, eviction, mismatch, timeout, or affinity loss rebuilds from the complete request history supplied through Pathfinder |
| End to end | Existing prepare/commit/abort/settle and Semantic Router streaming semantics remain unchanged |

The frozen performance contract adds the actual latency, throughput, cache-hit,
memory, and completion thresholds.

## Current Implementations

### Current Modal Rayline

Pathfinder's `KVEncodeSession` is a cross-request session. It retains:

- the exact cached token prefix;
- Qwen attention and recurrent `past_key_values`;
- an FP32 hidden-state sum aligned with the retained model state;
- a total FP32 hidden-state sum including the unaligned tail;
- the final hidden state; and
- bounded, process-owned residency metadata.

The encoder advances in an 8,192-token grid. At a non-aligned endpoint it
temporarily executes the tail, records the output sum, and rewinds hybrid
recurrent state to the last aligned checkpoint. The next turn re-executes at
most 8,191 old tokens before processing the new turn.

This design also handles Qwen3.5's hybrid recurrence through `CacheRewinder`.
An exact repeated token sequence can reuse the prior total sum without another
forward pass. Prefix replacement, shrinkage, truncation, sub-grid inputs,
session loss, and OOM decline to a full encode.

The important invariant is not the Python session object. It is that the model
state and hidden-state sum describe the same token boundary.

### David's causal-MEAN vLLM branch

The inspected vLLM baseline is
[`davidvgilmore/vllm@162bcefe1b41c5bb35eccc2f2219ea39e2c74bb7`](https://github.com/davidvgilmore/vllm/commit/162bcefe1b41c5bb35eccc2f2219ea39e2c74bb7).
It correctly adds:

- an FP32 `mean_pool_sum` and `mean_pool_count` to `PoolingStates`;
- causal-MEAN accumulation across scheduler chunks within one request;
- vectorized segmented sums for mixed batches;
- incomplete-request handling in pooling heads;
- Qwen/GritLM encoder-only support; and
- tests for chunking, mixed completion, cleanup, and batching cost.

It deliberately sets `skip_reading_prefix_cache` for causal MEAN. A normal KV
hit would skip the hidden states that the pooler needs, so using automatic
prefix caching without a corresponding sum/count would return an incorrect
mean.

Its accumulator is also cleaned when the request completes. It therefore
solves long-request scheduler chunking, not Modal's cross-turn reuse.

### Why the existing streaming-input API is not the answer

The current vLLM `StreamingInput` path explicitly rejects `PoolingParams`. It
also models one live generation request whose input arrives in chunks, not a
set of independently transactional Rayline turns separated by worker
dispatches and commits.

Extending it would require a custom session protocol, paused GPU-resident
requests, TTL and fencing behavior, affinity, eviction, and pooling outputs
after every append. That recreates Pathfinder's session manager inside the
inference engine and weakens vLLM's normal request/cache lifecycle.

It remains a prototype comparator, not the recommended production primitive.

## Target Cache Algorithm

### Checkpoint grid

Use the same 8,192-token checkpoint grid as Modal:

```text
request tokens
0-------------8192-------------16384-------------N
              ^                 ^
              paired            paired
              checkpoint        checkpoint

cache lookup H ----------------------------------^  longest model-state hit
paired restore C ----------------^                  greatest summary <= H
execute suffix                  [C................N)
```

For the frozen 262,144-token maximum, one sequence can have at most 32 aligned
summary checkpoints. Each 1,024-dimensional FP32 sum is 4 KiB, so all summary
vectors for a maximum-length sequence occupy about 128 KiB. Model state, not
the summary vectors, remains the dominant memory cost.

The grid is more important than request endpoints. An arbitrary endpoint may
fall inside a vLLM hash/cache block and cannot necessarily be used as the
prefix of a longer request. An aligned checkpoint is stable across later
extensions and matches the current Modal replay bound.

### Cache identity

The summary key is content-addressed:

```text
encoder contract fingerprint
+ tenant/cache-domain salt
+ terminal chained block hash at boundary C
+ boundary token count C
```

The encoder contract fingerprint includes:

- model ID and immutable weights revision;
- tokenizer ID/revision and digest;
- serializer and IO-plugin versions;
- causal-MEAN implementation version;
- hidden width, accumulator dtype, model execution dtype, and normalization;
- context limit and checkpoint grid; and
- vLLM build capability version.

The C82 policy artifact is validated by Pathfinder but does not need to be
part of the encoder cache key: a policy-head change cannot change the encoder
hidden states. It remains part of the end-to-end readiness receipt.

Do not use a raw episode ID as the cache key. Identical authorized prefixes
may safely share content-addressed acceleration. Use vLLM's cache salt or an
equivalent opaque cache-domain digest when tenants must not share cache
entries. A hashed episode identifier may be carried only as an affinity and
bounded-telemetry hint.

### Lookup

For every request:

1. Pathfinder sends the complete canonical turn history, including the
   candidate turn. Delta-only requests are not allowed.
2. The frozen IO plugin serializes and tokenizes the turns.
3. vLLM performs its hybrid-aware automatic-prefix-cache lookup and obtains
   the longest model-state hit `H`.
4. The pooling-summary manager finds the greatest 8,192-token boundary
   `C <= H` whose contract key and FP32 sum/count are present.
5. vLLM truncates the adopted cached blocks to `C`, restores
   `mean_pool_sum` and `mean_pool_count = C`, and schedules tokens `[C, N)`.
6. If no paired checkpoint exists, `C = 0` and the request is a full encode.
7. The pooler emits new sparse summaries whenever execution crosses a grid
   boundary.
8. At completion, vLLM returns the normalized 1,024-dimensional embedding and
   cache telemetry. Pathfinder runs the unchanged C82 head and routing logic.

An available KV hit with no summary is deliberately reduced to an earlier
paired boundary or zero. An available summary with no corresponding model
state is ignored. This fail-closed pairing rule is the central correctness
property.

### Exact-result fast path

After the paired-prefix MVP, measure whether retaining Modal's zero-forward
behavior for an exact repeat materially improves retry or duplicate traffic.
If it does, add a bounded final-result entry keyed by the complete canonical
token digest and encoder contract. It stores only the final normalized
embedding and token count.

This path is useful for idempotent retries and duplicate prepares. It does not
participate in strict-prefix extension and does not become authoritative
episode state. Pathfinder's transaction idempotency remains the first duplicate
request defense.

The first paired-cache prototype may omit this path. Qualification must report
the difference explicitly and must not claim retry-heavy performance parity
until the measured need is resolved. Exact-result caching is not a correctness
gate for stateless or paired-prefix selection parity.

### Hybrid Qwen state

The target model is hybrid attention plus recurrent state. The implementation
must use and test vLLM's hybrid cache coordination with
`mamba_cache_mode=align`. All cache groups must agree on boundary `C` before a
summary is restored.

Configure prefix-cache retention so the 8,192-token replay boundaries remain
reachable. A full-attention hit beyond a missing recurrent checkpoint is not a
paired hit. Connector or replica-loaded state must satisfy the same all-group
boundary rule.

The first parity release should use one local encoder engine. External cache
connectors and cross-replica transfer are a later optimization; they must not
be required for correctness or the initial end-to-end MVP.

## Concrete vLLM Changes

The implementation should remain a narrow extension of David's causal-MEAN
work and normal vLLM request scheduling.

### 1. Versioned capability

Add an opt-in capability such as:

```text
causal_mean_prefix_summary.v1
```

Do not change generic MEAN-pooling cache behavior globally. Existing pooling
models continue to skip prefix reads unless they explicitly negotiate this
capability.

Readiness must prove:

- `chunked_causal_mean`;
- `causal_mean_prefix_summary.v1`;
- hybrid APC with aligned Mamba state;
- the 8,192-token checkpoint grid;
- FP32 sum/count restoration;
- exact model/tokenizer/plugin pins; and
- the engine incarnation.

### 2. Pooling state and sparse snapshots

Extend `vllm/v1/pool/metadata.py` so `PoolingStates` can be initialized from a
validated prefix sum/count and can expose sparse checkpoint snapshots.

Extend `MeanPool` in
`vllm/model_executor/layers/pooler/seqwise/methods.py` to:

- preserve the current batched FP32 accumulation fast path;
- split a request segment only when it crosses a checkpoint boundary;
- snapshot the cumulative sum at that boundary;
- export snapshots before final request cleanup; and
- validate count, dimension, dtype, device, and finite values.

There should be no per-request GPU kernel launch in the common short-request
case. Boundary snapshots are sparse and should be copied to CPU only when a
new checkpoint is created.

### 3. Scheduler-owned summary manager

Add a bounded `PoolingPrefixSummaryManager` beside the KV-cache manager. It
owns CPU copies of:

```text
(contract, terminal block hash, token count) -> FP32 sum
```

The scheduler already knows request block hashes and the reconciled model-cache
hit. It should:

- look up the greatest paired checkpoint;
- truncate adopted blocks to that boundary;
- pass the restored sum/count through `NewRequestData`; and
- ingest sparse summary updates returned by the model runner.

Moving a 4 KiB vector from CPU to the model device once per cache hit is
negligible relative to even a small suffix forward. Keeping the summary index
in the scheduler also makes admission, eviction, and telemetry deterministic.

The summary manager must have an explicit LRU/byte bound. For the frozen
600,000-token logical budget, `ceil(600000 / 8192) = 74` divergent checkpoint
rows is a useful initial bound, plus a separately bounded allowance for active
requests. The final limit should be derived from the configured model-cache
byte budget and proven by the memory qualification, not left implicit.

### 4. Paired cache admission and fallback

Modify the causal-MEAN admission path in the scheduler and KV-cache manager:

- run the normal local hybrid APC lookup;
- allow a prefix read only under the new capability;
- clamp its result to the greatest available summary boundary;
- initialize restored pooling state on the model runner;
- retain the 8,192-token cache boundaries; and
- report why a longer model hit was declined.

Do not merely change `skip_reading_prefix_cache` from `true` to `false`.
Without the pairing and restore logic, that one-line change is a correctness
bug.

Summary eviction does not have to synchronously evict model blocks. A stale KV
entry is harmless because lookup clamps to a paired boundary. A stale summary
is harmless because it cannot be adopted without a model-state hit. Coupled
eviction events are still desirable to avoid wasted memory and improve
observability.

### 5. Optional final-result cache

If the measured retry workload justifies it, add the small exact-result cache
at the engine-core or pooling-serving layer. On an exact key hit, return a
normal finished pooling output through the same post-processing path, with
cache mode `exact_result`. It must obey the same identity, salt, LRU, reset,
and telemetry rules as prefix summaries.

### 6. Preemption and restart

Current vLLM pooling state is cleaned when a preempted request recomputes. The
restored-prefix state must follow the same rule:

- preemption with preserved model blocks may preserve the in-request sum;
- recompute resets to the last externally paired checkpoint, not an arbitrary
  partial accumulator;
- engine restart changes the incarnation and clears both summary/result
  managers; and
- cache-load or connector failure falls back to local recompute.

Every fallback must be visible as a bounded reason code.

## Pathfinder Changes

Pathfinder should expose one artifact-bound encoder seam:

```text
encode(canonical_turns, encoder_contract, deadline) -> EncoderResult
```

Implementations:

- `LocalTransformersEncoder`: the current Modal-compatible
  `KVEncodeSession` implementation and rollback oracle;
- `RemoteVLLMEncoder`: the strict full-history pooling client.

`EncoderResult` should contain:

- normalized embedding;
- exact model, tokenizer, serializer, plugin, and engine identities;
- full, restored-prefix, recomputed, and truncated token counts;
- `full`, `paired_prefix`, or `exact_result` mode;
- bounded miss/rebuild reason;
- queue, tokenize, forward, pool, and total timing; and
- engine incarnation.

Pathfinder retains:

- committed routing state: route version, previous worker, bounded outcome
  facts, and pending transaction ownership;
- same-episode transaction fencing;
- C82 artifact loading and policy-head execution;
- candidate masking, worker ordering, prices, and adjustment logic;
- prepare/renew/commit/abort/settle;
- retry/idempotency semantics; and
- the decision trace.

Semantic Router supplies the complete current conversation history in the
prepare request. Pathfinder canonicalizes that request for the encoder but
does not need to persist prompt history in its routing-state store.

The remote encoder must never receive only a suffix. Full history makes cache
loss, replica-affinity loss, and engine restart reconstructible. It also lets
vLLM content hashes independently verify every claimed prefix.

The current local KV store is not layered on top of the remote vLLM cache.
Exactly one encoder backend owns acceleration for a request.

### Cross-episode concurrency seam

Pathfinder's transaction coordinator already marks an episode busy before
calling the selector and releases its journal lock while different episodes
select. The remaining serialization point is the process-wide
`RouterService._policy_select_lock`: the transactional prepare path calls
`_policy_select`, so remote HTTP encodes currently reach vLLM one at a time.
The legacy eager route path also has a one-thread pre-worker segment, but that
segment is not the immediate limiter for `/v1/route/prepare`.

Before throughput qualification:

- declare concurrency at the policy implementation boundary rather than
  assuming every policy is thread-safe;
- allow the immutable MTRouter estimator plus `VLLMMTRouterEncoder` client to
  select concurrently for different prepared episodes;
- retain the existing journal's one-in-flight prepare fence for the same
  episode;
- preserve serialization for mutable RNG/round-robin policies;
- prove with a blocking fake encoder that different episodes overlap and the
  same episode does not; and
- include encoder in-flight concurrency and queue time in receipts.

This seam is required to exercise vLLM continuous batching. Scaling the
encoder or adding a paired cache before removing this lock would benchmark a
Pathfinder mutex rather than vLLM.

## Semantic Router Changes

No new selection abstraction is required. `rayline_remote` already uses
Semantic Router's natural policy seam:

- selectors receive the request-scoped candidate set;
- Pathfinder returns one allowed worker;
- Semantic Router resolves that worker to configured provider/model identity;
- the existing selection transaction commits on first upstream 2xx headers,
  aborts before that boundary, and settles actual response usage; and
- Semantic Router continues to own credentials, request mutation, dispatch,
  streaming, and client errors.

The relevant existing seams are:

- [`pkg/selection/selector.go`](../../src/semantic-router/pkg/selection/selector.go);
- [`rayline_remote_selector.go`](../../src/semantic-router/pkg/extproc/rayline_remote_selector.go);
- [`selection_transaction.go`](../../src/semantic-router/pkg/extproc/selection_transaction.go); and
- the `rayline_remote` registration in
  [`router_selection.go`](../../src/semantic-router/pkg/extproc/router_selection.go).

For the current OpenAI Chat MVP, only configuration, readiness propagation,
metrics, and the production-shaped test composition should change in this
repository. The Rayline model, CUDA runtime, encoder cache, and policy head do
not move into the Semantic Router Go process. A broader typed normalized
request or public `TransactionalSelector` abstraction should be introduced
only when another protocol or selector needs it; it is not a prerequisite for
vLLM parity.

## Wire Contract

The cached implementation must use a new version rather than weakening the
existing ARC v1 response, which correctly requires zero cached-prefix tokens.
A representative request is:

```json
{
  "task": "plugin",
  "data": {
    "schema_version": "rayline.arc.pooling-request.v2",
    "serializer_version": "mtrouter-token-blocks-v2",
    "encoder_contract": "<opaque pinned digest>",
    "cache_domain": "<opaque digest>",
    "episode_affinity_hash": "<optional opaque digest>",
    "turns": [
      {"role": "user", "text": "example only"}
    ]
  }
}
```

The response extends the strict identity fields with:

```json
{
  "schema_version": "rayline.arc.pooling-response.v2",
  "embedding": ["1024 finite numbers"],
  "engine_incarnation": "<opaque>",
  "encode_mode": "paired_prefix",
  "full_tokens": 26624,
  "restored_prefix_tokens": 24576,
  "recomputed_tokens": 2048,
  "checkpoint_grid": 8192,
  "cache_reason": "paired_hit"
}
```

The actual schema should retain the existing bounded-field, exact-key, finite
number, and identity checks. Prompts, token IDs, raw episode IDs, cache
tensors, authorization data, and provider receipts must never be logged.

## Alternatives

| Design | Correctness potential | Batching and memory | Operational fit | Decision |
| --- | --- | --- | --- | --- |
| vLLM APC plus paired summary | High; content verifies every prefix | Uses normal continuous batching and shared block cache | Reconstructible and engine-native | **Recommended** |
| Explicit pinned vLLM episode session | High if it recreates Modal rewind exactly | Pins per-session state and needs affinity/TTL/fencing | Duplicates Pathfinder state lifecycle in vLLM | Prototype comparator only |
| Stateless vLLM full history | High and simplest | Cost grows with full context every turn | Good baseline and fallback | Required first rung |
| Current Modal Transformers | Proven current behavior | No vLLM scheduler/continuous batching | Existing oracle and rollback | Keep through qualification |
| Put policy head in vLLM | Possible but unnecessary | Couples artifact rollout to GPU engine | Blurs policy/transaction ownership | Reject |
| Put encoder in Semantic Router | Not viable in the Go proxy | Couples CUDA/model failures to HTTP proxy | Violates current process boundary | Reject |
| One engine for routing and worker generation | Cache and queue interference | Different models and workloads compete | Coupled scaling and failures | Reject |

Paired APC is the preferred hypothesis, not a result declared before
measurement. Time-box both feasibility spikes rather than production-hardening
two designs. The paired-APC spike must first prove that Qwen3.5's aligned
hybrid state can be restored at the same boundary as the mean summary. A
pinned-session comparator needs only enough lifecycle, memory, and latency
evidence to expose a decisive advantage or liability. Select one in RSP-005
with a recorded decision; implementation convenience alone is not enough to
accept the session design's larger affinity and lifecycle surface.

## Implementation Rungs

### Rung A: freeze and replay the oracle

- Export a deterministic, non-customer parity corpus from the frozen
  serializer.
- Record Modal full-encode, Modal KV-delta, embeddings, raw policy outputs,
  adjusted scores/gaps, and selected workers.
- Include all cache-decline and hybrid-rewind shapes.
- Store identity and workload digests with the receipt.

Exit: repeatable oracle receipt with zero internal Modal full-versus-delta
selection flips.

### Rung B: stateless remote vLLM

- Use Pathfinder's implemented encoder interface and strict full-history v1
  client at `7f13de3d`.
- Use David's causal-MEAN branch with prefix reads disabled.
- Compare vLLM full-history output against both Modal modes.

Exit: zero selection flips, gap drift within `5e-3`, and deterministic fallback
to the local encoder.

### Rung B.5: concurrent remote selection

- Add an explicit thread-safety capability for policy implementations.
- Let different-episode MTRouter remote selections bypass the process-wide
  policy lock and overlap at the encoder.
- Keep one in-flight prepare per episode and keep mutable policies serialized.
- Prove overlap, cancellation, readiness, and bounded shutdown behavior without
  a GPU.

Exit: at least two different-episode prepares are observed concurrently by a
blocking fake encoder, same-episode prepares remain fenced, and no other policy
changes behavior.

### Rung C: paired local prefix cache

- Implement the summary manager, state restore, sparse snapshots, cache
  clamping, and reason codes. Add the exact-result path only if the measured
  retry workload justifies it.
- Run one Pathfinder replica and one local Rayline vLLM engine.
- Keep connectors and cross-replica cache transfer disabled.

Exit: all correctness/failure tests pass and every warm strict extension
restores a paired boundary or reports an explicit rebuild.

### Rung D: production-shaped local stack

- Start Envoy, Semantic Router, actual Pathfinder, dedicated Rayline vLLM,
  and two real worker vLLM endpoints through the repository's normal image
  flow.
- Test streaming and non-streaming requests, worker identity, first-2xx commit,
  abort, and usage settlement.

Exit: reproducible credential-free end-to-end receipt.

### Rung E: qualification

- Run all frozen ARC/Remote/cache variants.
- Run router-only and full-stack load ladders.
- Measure cold, warm, eviction, restart, affinity-miss, and maximum-context
  behavior.
- Compare the selected vLLM cache with current Modal on identical L40S
  hardware.

Exit: every hard gate in the performance contract passes, or the design is
rejected with a report identifying the failed cells.

### Rung F: optional scale work

Only after the single-engine result passes:

- evaluate cache-aware encoder affinity;
- extend a vLLM KV connector to carry paired summary identity/state;
- test replica loss and cache transfer;
- address durable Pathfinder pending transactions before multi-replica
  production; and
- repeat qualification at the intended replica count.

## Verification Plan

### 1. Serializer and identity goldens

For every fixture, compare:

- canonical turn ordering and roles;
- serialized byte digest;
- token IDs and token count;
- truncation point;
- model/tokenizer/plugin/config identities; and
- worker order, prices, and policy artifact.

Fixtures include short history, Unicode, tools, a large tool result, an empty
assistant turn, near-maximum context, exact repeat, strict extension, prefix
replacement, shrinkage, and truncation.

Any token mismatch blocks later numeric comparisons.

### 2. Pooling math unit tests

Test David's existing cases plus:

- full one-shot mean versus chunked mean;
- restored prefix sum plus suffix versus full mean;
- a scheduler chunk crossing one and several grid boundaries;
- mixed batches with full, restored, and incomplete requests;
- exact count and shape validation;
- BF16 hidden states with FP32 accumulation;
- NaN/Inf, corrupt dimension, wrong count, and wrong contract rejection;
- preemption with preserved blocks;
- preemption with recompute;
- cleanup after completion, abort, and error; and
- no retained tensor aliases to a whole batch allocation.

Small deterministic model tests may use tensor `allclose`. The pinned model
uses the policy-level parity gate because floating reduction order can vary
with batching.

### 3. Paired-cache invariant tests

Exercise this matrix:

| Model-state cache | Summary cache | Expected action |
| --- | --- | --- |
| miss | miss | full encode |
| hit | miss | clamp to earlier pair or full encode |
| miss | hit | ignore summary and full encode |
| hit at 24,576 | summary at 24,576 | restore 24,576 |
| hit at 26,000 | summary at 24,576 | restore 24,576 and replay tail |
| hit at 24,576 | wrong contract summary | full/earlier rebuild |
| hybrid groups disagree | summary exists | clamp to all-group boundary |
| exact final-result hit | n/a | return identical final embedding |

Assert restored token counts, forwarded token counts, output equivalence,
reason codes, and cleanup. Fault-inject hash mismatch, summary corruption,
eviction between lookup and schedule, and engine-incarnation change.

### 4. Frozen policy parity

Run at least 1,000 multi-turn decisions across the four frozen workload shapes
in these modes:

1. Modal full encode;
2. Modal `KVEncodeSession`;
3. vLLM full history with prefix reads disabled;
4. vLLM paired-prefix cache;
5. vLLM pinned-session prototype, if implemented.

Record:

- embedding max absolute error and cosine similarity;
- raw C82 outputs;
- adjusted worker scores;
- top-two adjusted gap drift;
- selected worker;
- full/restored/recomputed token counts; and
- cache/fallback reason.

Hard gates:

- zero selected-worker flips;
- maximum adjusted top-two gap drift `<= 5e-3`;
- no silent cache decline;
- no protocol or identity mismatch; and
- cache loss produces the same decision as a forced full encode.

### 5. Hybrid and lifecycle tests

Run the pinned Qwen3.5 model on the target GPU and cover:

- all context/turn-size cells;
- every grid boundary and boundary-minus/plus-one;
- repeated partial tails;
- long sequences crossing multiple recurrent checkpoints;
- same-episode concurrent prepare attempts;
- different-episode continuous batching;
- a blocking remote encoder proving different-episode calls overlap instead of
  queuing behind `RouterService._policy_select_lock`;
- a mutable-policy control proving RNG/round-robin selection remains
  serialized;
- cancellation during prefill;
- encoder timeout;
- forced preemption;
- summary and model-block eviction;
- Pathfinder restart;
- vLLM restart and new incarnation;
- OOM injection/fallback;
- cache-domain and contract rotation; and
- maximum-context rejection/truncation.

Same-episode concurrency must be fenced before encoder-state mutation.
Different episodes must remain batchable.

### 6. End-to-end tests with real LLM endpoints

The production-shaped test starts:

```text
Envoy
  -> Semantic Router
       -> Pathfinder
            -> dedicated Rayline vLLM (Qwen3.5-0.8B)
       -> worker vLLM A (pinned Qwen3-8B)
       -> worker vLLM B (same pin, distinct route identity)
```

Required scenarios:

- streaming and non-streaming success through each worker;
- candidate masking and exact worker-to-provider resolution;
- commit exactly once after first upstream 2xx headers;
- abort exactly once on pre-2xx failure;
- actual usage/cost settlement after response completion;
- selected-worker endpoint timeout and non-2xx;
- Rayline encoder timeout/restart;
- warm multi-turn cache reuse;
- forced cold rebuild with identical worker selection; and
- direct/static controls using the same workers and request corpus.

Fake workers remain in CPU CI for deterministic lifecycle coverage. Real
workers are the GPU qualification rung, not a replacement for hermetic tests.

### 7. Performance qualification

Use the exact workload, run duration, repetitions, load ladder, and metrics in
the frozen performance contract. In particular:

- warm for at least 2 minutes and 200 decisions;
- measure at least 10 minutes and 1,000 decisions per cell;
- run three seeded repetitions;
- use at least 50 isolated cold/restart/eviction trials per shape;
- measure concurrency `1, 4, 16, 32, 64, 128`;
- measure offered rates `1, 2, 4, 8, 12, 16` decisions/s; and
- include queueing time to avoid coordinated omission.

The key cache/performance gates are:

- at least 90% of eligible warm prefix tokens reused;
- vLLM paired-cache p95 no worse than `1.10x` current Modal KV p95;
- at least 8 decisions/s at 64 active episodes within 1,000 ms p95 and
  2,000 ms p99;
- full-stack throughput and output tokens/s at least 90% of static control;
- routing adds at most 1,000 ms p95 TTFT at 4 downstream starts/s;
- maximum-context cold rebuild at most 60 seconds p95;
- at least 99.9% completion/acceptance in required cells;
- no OOM or unbounded growth; and
- non-cache GPU allocation drift no more than 5% after warmup.

Report full-history compute, restored tokens, replayed old tail, new-turn
tokens, scheduler queue, GPU utilization, cache bytes, summary entries,
evictions, and rebuild reasons. A hit-rate number without forwarded-token and
latency evidence is insufficient.

## Acceptance Checklist

The vLLM implementation has complete Rayline parity only when all are true:

- [x] Pathfinder's local and remote encoders share one strict interface at
      `atlasfutures/pathfinder@7f13de3d`.
- [x] The stateless remote v1 client sends full canonical history on every
      encode and rejects cached-prefix claims.
- [ ] Stateless vLLM passes frozen numeric and selection parity first.
- [ ] Different-episode remote selections overlap while same-episode prepares
      remain fenced.
- [ ] Every prefix hit restores paired model and pooling state at one boundary.
- [ ] A KV-only or summary-only hit is impossible to consume.
- [ ] Warm strict extensions replay at most 8,191 old tokens.
- [ ] The measured retry workload decides whether a bounded exact-result cache
      is required; any implemented path passes identity and memory gates.
- [ ] Qwen3.5 hybrid state is proven correct at and around every grid boundary.
- [ ] Cache identity binds every encoder-affecting artifact and cache domain.
- [ ] Engine restart, eviction, mismatch, and affinity loss rebuild safely.
- [ ] Memory has one enforced byte/token bound and bounded summary/result LRUs.
- [ ] Same-episode mutation is fenced and different episodes still batch.
- [ ] Semantic Router transaction, streaming, dispatch, and settlement tests pass.
- [ ] The real five-service local stack reaches both real worker endpoints.
- [ ] All hard correctness and performance qualification gates pass.
- [ ] The local Modal backend remains a tested one-configuration rollback.
- [ ] Receipts contain exact code/model/config identities and no sensitive data.

## Rollout and Kill Criteria

Use these operational modes:

```text
local_transformers
remote_vllm_full
remote_vllm_paired_cache
```

The implementation currently configures `mtrouter_encoder_backend` as
`local|vllm`; `vllm` means the stateless v1 mode. RSP-006 must freeze whether
paired caching is a new backend value or a separately negotiated v2
capability before changing configuration.

Promotion order is `local_transformers` -> `remote_vllm_full` shadow ->
`remote_vllm_full` serving -> `remote_vllm_paired_cache` shadow ->
`remote_vllm_paired_cache` serving.

Immediately disable paired caching, while retaining stateless vLLM, on:

- any selection flip attributable to cache state;
- gap drift above `5e-3`;
- a consumed unpaired checkpoint;
- unexplained restored-token counts;
- summary corruption or identity bypass;
- unbounded memory or OOM;
- engine-incarnation confusion; or
- cache-hit output differing from a forced full encode beyond the gate.

Roll back from remote vLLM to local Transformers on persistent encoder
readiness failure, timeout budget breach, or stateless numeric failure. No
episode-state migration is needed because Pathfinder remains authoritative.

## Remaining Production Work

Passing this design establishes a production-shaped single-Pathfinder,
single-encoder deployment and a measured scale envelope. It does not by itself
resolve:

- durable pending transactions across Pathfinder replicas (TD046);
- cross-replica encoder-cache transfer;
- cache-aware load balancing and autoscaling;
- production SLO alerting;
- upstreaming and long-term maintenance of the vLLM fork; or
- policy/artifact promotion governance.

Those are subsequent decisions. None should be hidden inside the parity MVP.
