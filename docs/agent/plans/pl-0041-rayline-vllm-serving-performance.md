# PL-0041 Rayline vLLM Serving and Performance Qualification

## Goal

Turn the completed PL-0040 protocol MVP into a production-shaped, measured
Rayline serving design.

The plan must answer four questions with runnable evidence:

1. Can the Rayline backbone be hosted by vLLM while Pathfinder remains the
   remote policy and episode-state authority?
2. Can Rayline's existing cross-turn KV-delta behavior be preserved when model
   execution moves out of the Pathfinder process?
3. Does the complete path work against both self-hosted and external real LLM
   endpoints?
4. Under a realistic multi-episode workload, where do `rayline_arc` and
   `rayline_remote` saturate, and what latency, throughput, memory, and
   operational costs does each design impose?

Status: active on 2026-07-31. The stateless end-to-end MVP parity gate passes;
full-corpus quality/regret, cross-request KV, concurrency, and throughput
qualification remain open. Current published implementation heads:

- Semantic Router
  [`atlasfutures/semantic-router:codex/rayline-remote-mvp`](https://github.com/atlasfutures/semantic-router/tree/codex/rayline-remote-mvp)
  at `8c7171ebb2569241836960c273f679107b23a678`.
- Pathfinder
  [`atlasfutures/pathfinder:codex/rayline-vsr-mvp`](https://github.com/atlasfutures/pathfinder/tree/codex/rayline-vsr-mvp)
  at `05c4f1df7e1654897fec291e338426b810b1af98`.
- vLLM integration
  [`atlasfutures/vllm:codex/rayline-vsr-mvp`](https://github.com/atlasfutures/vllm/tree/codex/rayline-vsr-mvp)
  at `6ef6e84425d4493566a95ffcdfcb79f3c27abc46`.
- David's reviewed vLLM causal-MEAN input
  [`davidvgilmore/vllm:rayline/pl-0039-causal-mean`](https://github.com/davidvgilmore/vllm/tree/rayline/pl-0039-causal-mean)
  at `162bcefe1b41c5bb35eccc2f2219ea39e2c74bb7`.

## Scope

### Parent and Child Architecture

The transactional-routing architecture is the parent system contract. Semantic
Router owns HTTP normalization, candidate gating, credentials, dispatch,
streaming, and execution truth. Pathfinder owns the policy artifact, committed
routing state, pending selection receipts, same-episode fencing, and worker
choice. The vLLM parity and cache work is a child execution workstream that
replaces only Pathfinder's encoder backend.

Conversation history follows each prepare request. Semantic Router sends the
complete current history to Pathfinder, and Pathfinder forwards its canonical
form to the encoder alongside its committed routing facts. Pathfinder does not
need to persist prompts to make cache loss reconstructible.

The current OpenAI Chat MVP already has the required transaction seam. A
broader public transactional-selector abstraction and OpenAI Responses or
Anthropic Messages normalization are deferred until another protocol or
selector requires them.

### Recommended Deployment Boundary

The default topology is a separate vLLM process integrated into the Rayline
deployment, not an LLM engine embedded in the Semantic Router Go process:

```text
                                      decision plane
                                +-----------------------+
                                | Pathfinder            |
Client                          | policy + transactions |
  |                             | episode authority     |
  v                             +-----------+-----------+
Envoy -> Semantic Router ------------------>| prepare
              |                             |
              |                             v
              |                   dedicated Rayline vLLM
              |                   pooling/KV engine
              |                             |
              |<--------- selected worker --+
              |
              +--------> worker vLLM A / worker vLLM B / external provider
                              data plane
```

For the first GPU MVP, use one Pathfinder replica and one dedicated Rayline
encoder replica. They may be placed on the same node and communicate over the
cluster network or localhost, but remain separate processes with separate
health, resource, and rollout boundaries.

This is still "hosted in the vLLM framework": the Rayline model uses vLLM's
pooling runner, scheduler, model lifecycle, GPU memory manager, and IO processor
plugin. Pathfinder calls that engine and retains the small policy head,
transaction journal, and episode state. Semantic Router never loads Python,
CUDA, model weights, or Rayline KV tensors.

The Rayline encoder must not share one vLLM engine with a downstream worker
model. They use different model identities, runner contracts, scaling signals,
and cache lifetimes. A colocated-GPU experiment may be measured as a cost
variant, but the default benchmark and deployment use dedicated engines so
worker generation cannot evict or queue behind decision-plane state.

### Why Not Put Everything in One vLLM Server?

The existing IO plugin is a good seam for strict request serialization and
pooling output. It is not the owner of:

- prepare, renew, commit, abort, and settle transactions;
- the worker allowlist and bundle contract;
- durable episode state;
- provider credentials or dispatch; or
- fail-closed response lifecycle behavior.

Moving those responsibilities into a vLLM plugin would couple policy releases,
GPU scaling, transaction recovery, and provider semantics to an inference
engine extension. It would also make a GPU process restart an authority change
rather than a reconstructible cache miss.

Two alternatives remain legitimate experiments:

- **Pathfinder embeds `AsyncLLM`**: removes one local HTTP hop, but couples the
  API and GPU engine failure domains and scales them together.
- **Pathfinder and vLLM as containers in one Pod**: preserves process
  separation and localhost latency, but forces 1:1 scaling and duplicates
  weights when Pathfinder is replicated.

The benchmark may measure these shapes, but neither replaces the default
separate-service boundary without an explicit architecture decision.

### The Existing Cache Implementations Are Different

Pathfinder already has real cross-turn KV reuse:

- `KVEncodeSession` retains `past_key_values`, a running FP32 hidden-state sum,
  last hidden state, token prefix, and chunk-aligned resume position.
- `KVSessionStore` serializes same-episode mutation, isolates service
  incarnations, evicts whole sessions, and treats the cache as optional.
- `KVMemoryBudget` is the process-global residency owner and bounds total
  cached tokens.
- Cache loss, replacement, truncation, and sub-chunk requests fall back to a
  full encode; committed episode state remains authoritative.

David's vLLM fork currently solves a different boundary:

- it allows causal MEAN pooling to accumulate across scheduler chunks within
  one long request;
- its pooling state is cleaned when that request finishes; and
- causal MEAN deliberately skips automatic prefix-cache reads, because a KV
  hit would skip hidden states needed by the mean accumulator.

The reference ARC deployment correspondingly runs with
`--no-enable-prefix-caching`. Its `chunked_causal_mean` capability bounds one
long prefill; it does not yet reuse an earlier turn's KV blocks on the next
request.

RSP-005 must choose and prove one vLLM cross-request design:

1. **Prefix-cache extension, preferred hypothesis.** Enable automatic prefix
   caching and persist or reconstruct the causal-MEAN sum/count at the matched
   block boundary. A hit restores both model cache state and pooling state;
   restoring only KV is incorrect.
2. **Pinned episode-session extension.** Add an explicit, bounded session
   contract that retains vLLM-owned cache state between pooling requests and
   mirrors Pathfinder's existing prefix, rewind, eviction, and fallback
   behavior.

An IO processor change alone is not accepted as proof. The engine/scheduler
must expose and test the cache lifecycle it actually owns. RSP-005 time-boxes
both feasibility spikes and records the winning design; it does not
production-harden both or treat the preferred hypothesis as a predetermined
result.

### Cache and State Contract

The target contract keeps correctness separate from acceleration:

- The complete current request history supplied through Semantic Router and
  Pathfinder's committed routing state are the reconstructible inputs.
- vLLM's KV and pooling accumulator are reconstructible, non-durable
  acceleration state.
- Every encoder request is bound to the immutable model, tokenizer,
  serializer, bundle, and policy revisions.
- The cache identity is derived from the opaque episode key plus canonical
  token-prefix identity; it never uses a raw user episode ID.
- A cache hit reports the engine incarnation, matched prefix length, encode
  mode, evictions, and rebuild reason using bounded telemetry.
- A miss, eviction, engine restart, affinity miss, or rejected session rebuilds
  from the complete current request and must preserve the same selection.
- Same-episode concurrent requests are fenced before cache mutation.
- GPU residency has one enforceable owner per engine and a measured bound.

Horizontal scale requires cache-aware affinity for performance, not
correctness. A request reaching another encoder replica may be slower because
it rebuilds, but it must not make a different policy decision outside the
frozen numeric tolerance.

### ARC and Remote Comparison

The experiment must not conflate policy placement with cache placement:

| Variant | Policy/state owner | Encoder | Cross-turn cache today | Extra decision-plane hop |
| --- | --- | --- | --- | --- |
| Static route baseline | Semantic Router config | none | n/a | none |
| `rayline_arc` current | Semantic Router | dedicated vLLM pooling | no; full history per request | VSR to encoder |
| `rayline_arc` plus KV | Semantic Router | dedicated vLLM pooling | target experiment | VSR to encoder |
| `rayline_remote` current | Pathfinder | in-process Transformers | yes; `KVEncodeSession` | VSR to Pathfinder |
| `rayline_remote` vLLM bridge | Pathfinder | dedicated vLLM pooling | no; full history per request | VSR to Pathfinder to encoder |
| `rayline_remote` vLLM plus KV | Pathfinder | dedicated vLLM pooling | target design | VSR to Pathfinder to encoder |

The current trade is expected to be workload-dependent:

- ARC has fewer network and transaction boundaries and a smaller failure
  surface, but its current full-history encode cost grows with episode depth.
- Remote adds prepare/renew/commit/settle work and another service to operate,
  but its existing delta path makes steady-state encode cost depend mainly on
  the new turn rather than the complete prefix.
- Once both modes use the same vLLM KV primitive, the comparison isolates the
  true cost of remote authority: network, transaction, state-store, cache
  affinity, and independent scaling.

The fair comparison pins the same encoder model/revision, tokenizer,
serializer, policy artifact, worker order, price snapshot, request corpus,
hardware class, worker endpoints, and warm/cold state.

### End-to-End Test Rungs

The work keeps deterministic tests and real endpoints as separate evidence:

1. **Rung 0 — protocol fixture.** Existing Envoy + Semantic Router + fake
   Rayline + fake providers receipt.
2. **Rung 1 — actual Pathfinder.** Existing actual Pathfinder service + fake
   providers receipt.
3. **Rung 2 — real Rayline model.** Actual Pathfinder + actual Rayline encoder
   on GPU + fake providers. This isolates decision latency, selection parity,
   KV behavior, and memory at zero provider spend.
4. **Rung 3 — self-hosted real workers.** Actual Pathfinder + actual Rayline
   encoder + two actual OpenAI-compatible vLLM worker endpoints. This is the
   reproducible end-to-end performance environment.
5. **Rung 4 — external provider canary.** The same stack dispatches to two
   frozen OpenAI-compatible external model IDs through VSR-owned credentials.
   This proves live transport, usage, cost settlement, and provider failure
   behavior; it is not used as the primary throughput benchmark.

Rung 4 is explicit opt-in only. It requires a dedicated key with a provider
spend limit, a test-level upper bound, frozen non-alias model IDs, small token
limits, single concurrency, and a sanitized receipt. Baseline CI never needs a
credential or paid call.

### Performance Workload

The benchmark has two layers:

- **Router-only:** selected worker endpoints return an immediate synthetic 2xx.
  This measures the maximum selection-plane throughput and decomposes VSR,
  Pathfinder, encoder, policy-head, transaction, and state-store latency.
- **Full stack:** selected workers are real vLLM generation endpoints. This
  measures client-visible time to first token, inter-token latency, output
  throughput, end-to-end request throughput, and whether the router starves
  worker serving.

Use both closed-loop multi-turn sessions and open-loop arrivals. The frozen
workload matrix includes:

- short, growing, large-tool-dump, near-maximum-context, and cache-replacement
  episodes;
- cold start, warm cache hit, cache miss, eviction, encoder restart, and
  Pathfinder restart;
- streaming and non-streaming Chat Completions;
- incremental turn sizes such as small chat turns, ordinary tool results, and
  a large tool dump;
- episode concurrency at 1 and progressively higher levels until saturation;
- uniform and skewed episode popularity to exercise affinity and eviction; and
- a direct-to-worker baseline plus every applicable row in the ARC/Remote
  variant table.

Record at least:

- client end-to-end latency, TTFT, inter-token latency, and errors;
- accepted requests/second and output tokens/second;
- prepare, encoder queue, tokenize, model forward, pool, policy-head,
  renew/commit, provider, and settle latency at p50/p95/p99;
- vLLM scheduler queue depth, prompt throughput, GPU utilization, allocated and
  reserved memory, cache hit tokens, resident tokens, evictions, rebuilds, and
  refusals;
- selection parity, selected-worker distribution, state advancement, and
  dispatch identity; and
- the exact code, artifact, config, model, GPU, driver, and workload revisions.

External-provider latency is reported separately from local vLLM performance so
WAN and provider queue variance cannot be mistaken for router cost.

### Scope Boundaries

In scope:

- a real vLLM-backed Pathfinder encoder seam;
- a measured cross-request KV prototype and one selected implementation;
- a local GPU composition with actual Rayline and worker models;
- an opt-in external OpenAI-compatible provider canary;
- a reproducible ARC-versus-Remote benchmark and recommendation;
- failure, restart, eviction, privacy, and bounded-memory evidence; and
- the config, metrics, docs, and receipts required to repeat the result.

Not in scope:

- making paid provider tests a PR merge gate;
- treating external-provider throughput as a stable system benchmark;
- sharing one vLLM engine between the Rayline model and worker models as the
  production default;
- online policy training or artifact promotion;
- native Anthropic Messages or OpenAI Responses support;
- production traffic rollout, SLO alerting, or autoscaling policy; and
- silently resolving TD046. Durable, multi-replica pending transactions remain
  a separate production requirement.

## Exit Criteria

- One reviewed architecture decision selects the Rayline model-hosting and
  cache design, and records why the rejected designs lost.
- Actual Pathfinder can use the frozen Rayline model through a pinned vLLM
  build and strict readiness contract.
- Cross-turn cache hits in the selected vLLM design restore both model KV and
  causal-MEAN pooling state, or the design explicitly proves an equivalent
  session mechanism.
- Full encode, current Pathfinder KV, and vLLM KV select the same worker over
  the fixed parity corpus: zero selection flips and adjusted top-two gap drift
  within the existing `5e-3` gate.
- Cache loss, eviction, affinity miss, and encoder restart rebuild correctly
  from the complete current request supplied through Semantic Router.
- GPU residency stays within the configured bound; OOM, unbounded session
  growth, silent cache drift, and secret-bearing telemetry are release
  blockers.
- The complete local stack reaches two actual vLLM worker endpoints for both
  streaming and non-streaming requests, commits only after first 2xx headers,
  and settles actual usage.
- The opt-in external-provider canary reaches two frozen real model IDs, stays
  under its hard cost cap, and emits a credential-free receipt.
- A versioned report compares every viable ARC/Remote row at identical
  hardware, workload, and model pins, including cold/warm p50/p95/p99,
  saturation throughput, TTFT impact, GPU memory, and cache effectiveness.
- Before the measured run, RSP-001 freezes a numeric product latency budget and
  target request-start rate. The report states whether the selection plane
  sustains at least 2x that downstream start rate and stays inside the agreed
  p95 TTFT budget.
- The report ends with a concrete deployment recommendation, capacity envelope,
  rollback trigger, and list of remaining production blockers.
- Baseline CPU CI remains deterministic, credential-free, and passing.

## Task List

- [x] **RSP-001 — Freeze targets and experiment contract.** Record the target
  request-start rate, TTFT budget, context/turn distributions, concurrency
  ladder, GPU classes, model pins, cost ceiling, repetitions, warmup, and
  statistical summary before measuring. Frozen as
  `rayline-vllm-perf.v1` in
  `docs/benchmarks/rayline-vllm-performance-contract.md`.
- [ ] **RSP-002 — Decide the serving boundary.** Write the architecture
  decision comparing a separate vLLM service, same-Pod sidecar, embedded
  `AsyncLLM`, and current in-process Transformers execution. The detailed
  boundary is drafted in
  `docs/architecture/rayline-vllm-serving-boundary.md`; Pathfinder ADR 0059 is
  proposed at
  [`fb3a4b94`](https://github.com/atlasfutures/pathfinder/commit/fb3a4b9455653eb9f8e490ca414aaa90a24e0a55)
  and still requires human acceptance.
- [x] **RSP-003 — Extract Pathfinder's encoder seam.** Make local Transformers
  and remote vLLM implementations satisfy one strict, artifact-bound interface
  with identical canonical serialization and telemetry. Implemented in
  [`atlasfutures/pathfinder@7f13de3d`](https://github.com/atlasfutures/pathfinder/commit/7f13de3d10855ea44245717f9ccb50d55ea40e93):
  the local backend preserves the accepted Transformers/KV behavior, while the
  remote backend loads only the C82 policy head and fails closed on encoder
  identity drift.
- [x] **RSP-004 — Build the stateless vLLM bridge and pass the MVP parity
  smoke.** Reuse the pinned IO plugin
  and causal-MEAN fork to serve full-history Pathfinder encodes; prove numeric
  and selection parity before adding cross-request caching. Fork vLLM under
  `atlasfutures` before publishing any new vLLM change. The strict client,
  configuration, readiness probe, bounded response handling, and policy-head
  integration landed with RSP-003 at `7f13de3d`. The deterministic exact-token
  corpus, mode runner, strict comparator, and sanitized receipt landed at
  [`f580f961`](https://github.com/atlasfutures/pathfinder/commit/f580f9618787b90b6d876c33d510b9505f084327).
  The pinned L40S comparison completed on 2026-07-30 and **failed the strict
  gate**: 1,000/1,000 decisions and exact token-count parity passed, maximum
  adjusted top-two gap drift was `0.003936` against the `0.005` limit, but four
  boundary decisions selected a different worker. Sequential diagnostic
  runtime was 4,297.6 seconds locally and 1,203.2 seconds through vLLM
  (`3.57x` faster); this is not a throughput claim. The run also exposed a
  seam mismatch: local Transformers returns an unnormalized FP32 mean while
  Rung B returns a normalized vector, although C82 normalizes both before
  scoring. Evidence and private artifact pins are recorded in
  [`atlasfutures/pathfinder@5295fdb5`](https://github.com/atlasfutures/pathfinder/commit/5295fdb57adece07d1a62c0aa447143c0e9f3224).
  The first remediation rung is complete at
  [`atlasfutures/pathfinder@b280b585`](https://github.com/atlasfutures/pathfinder/commit/b280b5856e71d0f5375eb0fc13920357ca4f1a50):
  the encoder seam now declares `l2-normalized-fp32.v1`, the v2 comparator
  rejects non-unit vectors, and a six-decision RSP-004S corpus contains all
  four historical flips plus large-tool and near-maximum coverage. Offline
  canonicalization reduced the meaningful embedding maximum absolute error to
  `0.00112024`; explicit local pre-normalization changed C82 q-values by at
  most `3.5763e-7` and changed zero raw argmax decisions. The original scale
  mismatch was therefore a comparator defect, not the flip cause. Kernel
  direction drift at policy boundaries remains open. The sanitized diagnostic
  and smoke inputs are privately pinned at
  `rayline-ai/router-artifacts@d73fae3a526ff4d350d462b93b453792099a08b9`.
  No provider call or GPU spend was used for this remediation.
  The bounded execution-alignment follow-up then isolated scheduler, eager,
  Transformers model-implementation, GDN, Q/K projection, normalization,
  FlexAttention, and Triton-attention variants. The first strict MVP pass uses
  David's causal-MEAN path, Transformers-ordered Torch-reference GDN
  preparation, memory-bounded Triton attention, and the existing
  cheap-default selection margin set to `0.002` on both local and remote
  contracts. Its receipt passes all eight hard gates over six decisions and
  426,979 tokens: zero selection flips, exact token-count and contract
  identity, minimum embedding cosine `0.9999849695`, and maximum adjusted
  top-two gap drift `0.0011914223` against the `0.005` gate. The guard changes
  one local near-tie and zero remote decisions in offline replay; it is an MVP
  stability contract, not evidence of production quality non-regression.
  Private artifacts are pinned at
  `rayline-ai/router-artifacts@306ca8c40470820f36d3decb5bfd9414552b5b7a`.
  The reproducible controller and result ledger are published at
  [`atlasfutures/pathfinder@05c4f1df`](https://github.com/atlasfutures/pathfinder/commit/05c4f1df7e1654897fec291e338426b810b1af98).
  Measured infrastructure spend across successful and preserved failed arms
  was `$1.1961`; adding the conservative `$1` preflight/preemption reserve
  yields `$2.1961`, below the `$20` cap. All fourteen Modal apps were verified
  stopped with zero tasks.
- [ ] **RSP-004Q — Complete production parity and stability qualification.**
  Replay a larger task-disjoint, quality-labeled development set to measure
  task quality, cost, churn, and regret from the `0.002` stability margin, then
  run the existing 1,000-decision, 41.2-million-token qualification corpus
  through the winning local and remote contracts. TD048 remains open until
  both gates pass. Preparation, input pinning, and dry-run validation are
  authorized; launching either paid 1,000-decision arm requires a fresh,
  explicit user confirmation after the readiness packet is presented.
- [x] **RSP-004A — Enable cross-episode remote selection concurrency.** Add an
  explicit policy thread-safety capability, allow immutable MTRouter remote
  selections for different prepared episodes to overlap, retain the existing
  same-episode transaction fence, and keep mutable policies serialized. Prove
  the boundary with a blocking fake encoder before throughput or cache
  qualification. Implemented at
  [`atlasfutures/pathfinder@ce661e5f`](https://github.com/atlasfutures/pathfinder/commit/ce661e5ffe62301dcad307b9bc4b242324019497): undeclared and mutable policies
  remain serialized, remote MTRouter declares concurrent safety, independent
  episode prepares overlap, failures release capacity, and `/readyz` reports
  bounded policy-selection in-flight and queue-wait metrics. TD047 remains
  open only for the measured router-only receipt proving more than one request
  reaches the encoder/vLLM boundary in the real stack.
- [ ] **RSP-005 — Prototype both KV designs.** Measure the automatic-prefix-
  cache plus pooling-accumulator extension against explicit pinned sessions.
  Test hybrid-model rewind, eviction, batching, replica affinity, and restart;
  select one with a recorded decision.
- [ ] **RSP-006 — Implement and harden vLLM KV reuse.** Add bounded cache
  ownership, exact fallback, same-episode fencing, privacy-safe metrics, and
  full-vs-incremental parity gates.
- [ ] **RSP-007 — Add the production-shaped local stack.** Compose Envoy,
  Semantic Router, Pathfinder, dedicated Rayline vLLM, state store, and two
  worker vLLM endpoints through the normal local image flow.
- [ ] **RSP-008 — Add the benchmark harness.** Drive frozen open- and
  closed-loop workloads, collect synchronized client/component/GPU metrics, and
  emit one versioned machine-readable receipt plus a human report. Do not start
  the concurrency ladder until RSP-004A removes the transactional path's
  process-wide policy-selection lock for concurrent-safe MTRouter execution;
  otherwise encoder calls serialize before vLLM and the benchmark cannot
  exercise continuous batching.
- [ ] **RSP-009 — Run router-only qualification.** Find cold/warm latency,
  cache break-even, saturation, memory envelope, and failure behavior without
  provider spend.
- [ ] **RSP-010 — Run self-hosted full-stack qualification.** Compare direct,
  static, ARC, and Remote variants against identical real vLLM worker
  endpoints.
- [ ] **RSP-011 — Add the external-provider canary.** Use one dedicated,
  spend-limited OpenAI-compatible key and two immutable low-cost model IDs;
  validate dispatch, streaming, usage, cost, and sanitized logs.
- [ ] **RSP-012 — Publish the comparison and next decision.** State whether the
  design holds at the frozen target, choose the deployment shape, size the
  capacity envelope, and route HA journal work to TD046 rather than hiding it
  in benchmark notes.

## Next Action

The end-to-end stateless MVP is complete, RSP-004A's implementation boundary is
landed, and the first task-disjoint stability preflight has resolved the policy
ordering:

1. Treat the original post-stay `0.002` guard as rejected. On 60 canonical C82
   dev attempts it changed 40/524 decisions (`7.63%`) and increased switches
   `14→30`, failing the frozen behavior gate. The sanitized replay is pinned at
   `rayline-ai/router-artifacts@b947be95f9181058270b572d285c7efde5b5b074`.
2. Use the explicit `pre_stay` candidate for the next smoke. It changed 0/524
   decisions and preserved 14 switches on the same replay, so it removes the
   churn regression, but the result is `insufficient_power`, not a quality
   pass: no route-0 action changed. Evidence is pinned at
   `rayline-ai/router-artifacts@e7f862ede913559a4985b8354296b580ab1f919d`.
3. Recanary the six-case local/remote parity corpus with `pre_stay`, then build
   targeted same-state route-0 evidence around the `0.002` boundary. A clean
   smoke cannot substitute for the missing quality/regret evidence.
4. Prepare the **RSP-004Q** 1,000-decision local and remote commands, pins,
   budget envelope, and cleanup checks. Stop at the launch boundary and wait
   for the user's explicit confirmation; readiness alone is not authorization.
5. Then proceed to RSP-005 cache feasibility and the router-only performance
   ladder. Keep the stateless full-history path as the reconstructible
   correctness fallback.

The completed 2026-07-30 full run remains RSP-004Q attempt 1 and a failed
receipt; it is not renamed or reinterpreted after the fact. The v1 plugin
continues to reject cached-prefix tokens until RSP-005 chooses and versions a
cross-request cache design. RSP-002 remains pending until a Pathfinder human
accepts ADR 0059.

RSP-004A now replaces the process-wide `_policy_select_lock` with a
default-serialized executor and an explicit concurrency-safe capability. The
transaction coordinator still rejects a second prepare for the same episode;
different episodes may overlap only when the concrete policy opts in. The
legacy eager route still has a one-thread `AsyncStateCoordinator` segment, but
it is a separate follow-up rather than the current `/v1/route/prepare` blocker.

## Operating Rules

- Use the repo's normal local image flow; do not invent another Semantic Router
  serve path.
- Keep the Rayline model engine, Pathfinder authority, and worker data plane as
  distinct owners even when colocated.
- Pin code commits, model and tokenizer revisions, artifacts, serializer,
  prices, and GPU class in every receipt.
- Treat KV as a reconstructible optimization. A miss may cost latency but never
  correctness or a different state transition.
- Freeze benchmark inputs and pass/fail budgets before a measured run.
- Never compare external-provider latency directly with local worker
  throughput.
- Do not log prompts, tools, raw episode IDs, receipts, authorization headers,
  cache tensors, or secrets.
- Run the smallest reported gate first and drive every affected gate to green.
- Add behavior-visible E2E coverage for config, startup, API, dispatch, or
  lifecycle changes.
- Use signed-off commits for work intended for review.
- Keep TD046 open until durable pending transactions and multi-replica fencing
  are implemented and tested.
- Keep TD047 open until a router-only receipt shows concurrent-safe MTRouter
  selections reaching the real encoder/vLLM boundary at observed in-flight
  concurrency above one; the in-process fencing tests are already green.
- Keep TD048 open until the stability margin passes task-disjoint
  quality/regret evaluation and the full RSP-004Q parity qualification.

## Related Docs

- [pl-0039-rayline-arc-orchestrator.md](pl-0039-rayline-arc-orchestrator.md)
- [pl-0040-rayline-remote-mvp.md](pl-0040-rayline-remote-mvp.md)
- [Rayline vLLM serving boundary](../../../docs/architecture/rayline-vllm-serving-boundary.md)
- [Rayline-on-vLLM parity implementation](../../../docs/architecture/rayline-vllm-parity-design.md)
- [Rayline vLLM performance contract](../../../docs/benchmarks/rayline-vllm-performance-contract.md)
- [Rayline ARC tutorial](../../../website/docs/tutorials/algorithm/selection/rayline-arc.md)
- [Rayline Remote tutorial](../../../website/docs/tutorials/algorithm/selection/rayline-remote.md)
- [TD046](../tech-debt/td-046-rayline-remote-durable-journal-gap.md)
- [TD047](../tech-debt/td-047-rayline-remote-cross-episode-selection-serialization.md)
- [TD048](../tech-debt/td-048-rayline-vllm-selection-stability-gap.md)
- [Pathfinder ADR 0059 proposal](https://github.com/atlasfutures/pathfinder/blob/fb3a4b9455653eb9f8e490ca414aaa90a24e0a55/docs/adr/0059-rayline-vllm-serving-boundary.md)
- [Pathfinder stateless vLLM encoder implementation](https://github.com/atlasfutures/pathfinder/commit/7f13de3d10855ea44245717f9ccb50d55ea40e93)
- Pathfinder `docs/adr/0021-service-owned-kv-sessions.md`
- Pathfinder `docs/adr/0023-process-global-kv-memory-owner.md`
- Pathfinder `docs/history/2026-07-22-mtrouter-c82-perf-smoke.md`
- Pathfinder `docs/history/2026-07-26-kvdelta-s9-p95refined-recanary.md`
