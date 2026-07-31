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

Status: active on 2026-07-31. The stateless end-to-end MVP parity gate, retained
engine gate, versioned HTTP/client integration, and real-GPU concurrent gateway
E2E pass. The explicit pinned-session design is selected; its 100–200-case
development qualification and production hardening remain in progress. Current published
implementation heads:

- Semantic Router
  [`atlasfutures/semantic-router:codex/rayline-remote-mvp`](https://github.com/atlasfutures/semantic-router/tree/codex/rayline-remote-mvp)
  at `29219dd0` for the capability-gated retained-session client and hermetic
  full-stack checkpoint.
- Pathfinder
  [`atlasfutures/pathfinder:codex/rayline-vsr-mvp`](https://github.com/atlasfutures/pathfinder/tree/codex/rayline-vsr-mvp)
  at `9e4678b0` for the registered retained-session canary.
- vLLM integration
  [`atlasfutures/vllm:codex/rayline-vsr-mvp`](https://github.com/atlasfutures/vllm/tree/codex/rayline-vsr-mvp)
  at `b1049f6dd95c27d2e1b052eebc3b1a7f9f41195f`.
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

RSP-005 considered two vLLM cross-request designs:

1. **Prefix-cache extension, rejected for the MVP.** Enable automatic prefix
   caching and persist or reconstruct the causal-MEAN sum/count at the matched
   block boundary. A hit restores both model cache state and pooling state;
   restoring only KV is incorrect.
2. **Pinned episode-session extension, selected.** Add an explicit, bounded session
   contract that retains vLLM-owned cache state between pooling requests and
   mirrors Pathfinder's existing prefix, rewind, eviction, and fallback
   behavior.

The prefix-cache variant was rejected because vLLM's block-cache lifecycle does
not own the matching causal-MEAN sum/count. Restoring KV without that
accumulator is numerically wrong; coupling two independently evicted state
stores would add a second cache-lifecycle protocol before the MVP has a measured
need for it.

The selected variant keeps one live pooling request as the owner of both model
KV/GDN state and the causal-MEAN accumulator. vLLM commit `b1049f6d` adds a
strict one-append/one-output `AsyncPoolingSession`. A real NVIDIA L40S canary
processed 3,072 session tokens versus 7,680 cumulative replay tokens, with
minimum cosine `0.9999889556`, maximum absolute drift `0.0005071524`, and
one-shot/session latency ratios of `1.27x` and `2.14x` on turns 2 and 3. The
verified private evidence is pinned at
`rayline-ai/router-artifacts@6e387884239951ff29f48363c1adcf6c49e74d67`.

The Semantic Router checkpoints at `4f14763b` and `29219dd0` add the next
lifecycle boundary: a separate authenticated ASGI endpoint, full-history
exact-prefix validation, same-episode serialization, independent-session
concurrency, identical-request reuse, mismatch rebuild, TTL/LRU eviction,
global session/token residency bounds, explicit close/health APIs, and a
capability-gated Go client with bounded metrics. The normal `/pooling` v1
contract stays stateless. Capability `resumable_causal_mean` selects the
session wire and requires `chunked_causal_mean`; automatic prefix caching
remains disabled.

The deployed H100 HTTP canary
`rayline-arc-session-http-shp001-20260731` passed `created → appended → reused
→ rebuilt`, retained the exact 11-token prefix while appending 35 tokens, and
returned zero resident sessions after explicit cleanup. Two independent
episodes overlapped in `0.775s` wall time versus individual request latencies
of `0.661s` and `0.760s`. The real gateway canary
`rayline-arc-modal-gateway-mgp003-20260731` then traversed Envoy, Semantic
Router, the protected Modal ASGI endpoint, retained vLLM state, Rayline scoring,
and the synthetic provider. Both requests returned HTTP 200 and selected
`worker-b`; the warm end-to-end latencies were `0.337s` and `0.424s`. Router
metrics recorded one `created`, one `appended`, and zero selection failures.
The Modal service disables automatic prefix caching and has a five-minute
scale-to-zero window. At the pinned H100/CPU/memory price snapshot, one entire
31-minute single-container timeout envelope is about `$2.50`, below the `$20`
cap; this canary used only a fraction of that envelope and made zero paid
provider calls.

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

### Frozen 128-Case Development Qualification

The next paid rung is fixed at 128 public synthetic history states: four turns
for eight episodes in each of `short`, `medium`, `tool_dump`, and `long`
shapes. Every retained result is compared in memory with a fresh full-history
replay through the same pinned H100 engine. The driver refuses any case count
other than 128 and hard-caps the development surface at 200; it cannot launch
the held 1,000-case release packet.

The qualification passes only if all of these gates hold:

- minimum retained/replay cosine similarity is at least `0.9999`;
- maximum embedding absolute drift is at most `0.01`;
- maximum four-arm synthetic-head score drift is at most `0.005`, with zero
  selected-arm flips;
- retained appended tokens are at most 75% of full-replay serialized tokens;
- eight independent episodes overlap with wall time at most 85% of their
  summed individual latencies;
- identical same-episode requests produce exactly `created` plus `reused`;
- the ninth resident episode evicts the LRU session and reconstructs with
  parity;
- explicit affinity loss reconstructs with parity; and
- cleanup returns both resident sessions and resident tokens to zero.

The Modal MVP is pinned to one container. This is an intentional deployment
constraint: it makes cache affinity and the cost bound enforceable while
`@modal.concurrent(max_inputs=32)` still permits cross-episode batching. The
single-container 31-minute timeout envelope is about `$2.50` at the pinned
price snapshot. Before a later multi-replica qualification, add cache-aware
affinity or an explicit session directory and freeze a new cost envelope.

#### Development Qualification Result — 2026-07-31

Run `rayline-arc-session-qualification-sqp001-20260731` passed all frozen gates
on one NVIDIA H100 with automatic prefix caching disabled:

| Signal | Result | Gate |
|---|---:|---:|
| History states | 128 | exactly 128 |
| Minimum cosine similarity | `0.9999814` | at least `0.9999` |
| Maximum absolute drift | `0.0006667` | at most `0.01` |
| Maximum synthetic score drift | `0.0002751` | at most `0.005` |
| Synthetic selected-arm flips | `0` | `0` |
| Retained/full-replay token ratio | `0.4004` | at most `0.75` |
| Eight-way create/append wall-to-sum ratio | `0.1420` / `0.1388` | at most `0.85` |
| Same-episode actions | `created`, `reused` | exact match |
| LRU / affinity-loss rebuild cosine | `1.0` / `0.9999967` | at least `0.9999` |
| Residency after cleanup | `0` sessions, `0` tokens | both `0` |

Retained latency was `0.841` / `0.910` / `1.010` seconds at p50/p95/p99,
versus `0.856` / `1.041` / `1.073` seconds for full replay. The retained path
therefore saved 60% of serialized token work, but the client-visible latency
benefit at this workload was modest: about 1.8% at p50, 12.6% at p95, and 5.8%
at p99. The retained maximum of `96.892` seconds is the one cold-start request
and is reported separately from the warm percentiles.

The complete driver took `421.998` seconds (`0.303` history states/second).
At the pinned combined H100/CPU/memory rate, client elapsed time represents
about `$0.567`; including the configured five-minute idle scale-down window is
a conservative `$0.970` attempt estimate, below the `$2.50` timeout envelope.
Provider calls and provider spend were zero. The sanitized receipt is pinned at
`rayline-ai/router-artifacts@4b8a0b308d7980b5782cb8b41ac454874e8c7e16`
under `runs/rayline-arc-session-qualification-sqp001-20260731`.

This closes the 100–200 case development rung, not release qualification. It
does not prove multi-container affinity, real worker-generation throughput, or
production traffic behavior. The separate 1,000-case packet was not executed
and remains confirmation-gated.

### Frozen Real-Worker Full-Stack Canary

The next rung is a bounded self-hosted generation canary, not another parity
qualification. It deploys two separate OpenAI-compatible vLLM endpoints on
NVIDIA L4 containers, both serving the pinned `Qwen/Qwen3.5-0.8B` revision
under the artifact's `synthetic/provider-a` and `synthetic/provider-b` model
identities. The existing protected H100 session encoder remains the routing
model. Semantic Router sends a generated bearer credential to the workers and
Modal proxy credentials to the encoder; all credentials are deleted or made
unreachable during cleanup.

The fixed workload contains at most 37 generation requests:

- one warm-up plus three measured direct requests to each real worker;
- at most 24 public candidate prompts, stopping as soon as the gateway has
  selected both workers;
- four concurrent gateway requests split across both selected paths; and
- one streaming gateway request that must reach `[DONE]`.

The public synthetic artifact can raise gateway completion limits to 128
tokens, so the gateway side is bounded to at most 3,712 generated tokens; the
direct side adds at most 64. The driver does not accept a case-count argument
and contains no path to the held 1,000-case packet. The launcher also applies
a 15-minute whole-canary deadline around the driver; expiration enters the same
unconditional credential, compose, and worker cleanup path.

The canary passes only if:

- both direct vLLM endpoints generate a valid OpenAI-compatible response;
- the real encoder and policy route at least one request to each worker;
- each gateway response's model identity matches its selected-worker header;
- the four-request concurrent phase reaches both workers and reports its
  wall-to-summed-latency ratio and requests/second;
- the streaming phase emits at least one data event and terminates with
  `[DONE]`;
- router metrics report every session create and zero ARC selection failures;
- compose logs contain none of the ephemeral credentials; and
- cleanup removes the compose stack and volumes, deletes the Modal proxy
  token, and stops both L4 workers.

ARC worker artifacts now distinguish the legacy default `openrouter` dispatch
from `openai_compatible`. The latter must pin its exact `provider_base_url`,
cannot carry OpenRouter provider fields, omits the OpenRouter request payload,
and owns `chat_template_kwargs.enable_thinking` in its signed `extra_body`.
Startup fails closed if config URL, credential environment identity, model,
pricing, reasoning mode, or auth shape diverges from that artifact contract.

At the 2026-07-31 Modal rate snapshot, each 15-minute L4/4-CPU/16-GiB timeout
envelope is `$0.278928`; both workers total `$0.557856`. Including the existing
single-container H100 encoder's `$2.499617` timeout envelope gives a combined
worst-case `$3.057473`, with zero external-provider spend. Normal success is
expected to be much lower because the workers are stopped immediately and the
encoder scales to zero. This rung measures actual generation and dispatch, but
it remains a small canary rather than a saturation benchmark.

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
  preparation, memory-bounded Triton attention, and an exploratory global
  cheap-default selection margin set to `0.002` on both local and remote
  contracts. Its receipt passes all eight hard gates over six decisions and
  426,979 tokens: zero selection flips, exact token-count and contract
  identity, minimum embedding cosine `0.9999849695`, and maximum adjusted
  top-two gap drift `0.0011914223` against the `0.005` gate. That guard changed
  one local near-tie and zero remote decisions in the smoke, but later quality
  evidence rejects it; this receipt remains historical execution evidence, not
  an accepted policy contract.
  Private artifacts are pinned at
  `rayline-ai/router-artifacts@306ca8c40470820f36d3decb5bfd9414552b5b7a`.
  The reproducible controller and result ledger are published at
  [`atlasfutures/pathfinder@05c4f1df`](https://github.com/atlasfutures/pathfinder/commit/05c4f1df7e1654897fec291e338426b810b1af98).
  Measured infrastructure spend across successful and preserved failed arms
  was `$1.1961`; adding the conservative `$1` preflight/preemption reserve
  yields `$2.1961`, below the `$20` cap. All fourteen Modal apps were verified
  stopped with zero tasks.
  The explicit `pre_stay` contract has now also passed the registered six-case
  local/remote recanary. All eight hard gates passed with 0/6 selection flips,
  exact token counts, maximum top-two-gap drift `0.001191`, and minimum
  embedding cosine `0.99998497`. The remote arm ran in an isolated L40S
  container, made zero provider calls, and its seven-file private bundle was
  round-trip verified at
  `rayline-ai/router-artifacts@b82e0afc2da53e6268dc72ba13a23df7e863e9c0`.
  This closes the reordered-policy smoke only; it does not supply the missing
  route-0 quality/regret evidence.
  A subsequent 178-state, source-lineage-disjoint C9 route-0 screen rejects the
  global `0.002` Flash-off default outright. It crossed model families on four
  decisions; three scorable changes had mean reward delta `-0.1667` and worst
  task delta `-0.5`, while one unscorable change failed closed. The replacement
  rule is restricted to Flash thinking-on versus the same base model's
  thinking-off arm within `0.0005`. It made zero cross-model changes and was
  inert on both the 178-state screen and all 524 canonical C82 dev decisions,
  preserving the historical 14 switches. Those are scope and compatibility
  screens, not powered changed-action quality evidence. Both private offline
  bundles are round-trip verified at
  `rayline-ai/router-artifacts@d4a2d67b10b0e435c70de10a320c2b0590d520e8`.
  The narrow-rule L40S recanary then passed all hard gates: 6/6 decisions, zero
  flips, exact token counts, `0.0011912882` maximum gap drift, and
  `0.9999849696` minimum embedding cosine. Its seven-file private bundle is
  round-trip verified at
  `rayline-ai/router-artifacts@b707b2715018edaa269e08e16f1755491d79fd06`;
  measured infrastructure was `$0.155999`, provider calls were zero, and the
  Modal app stopped with zero tasks.
- [ ] **RSP-004Q — Complete production parity and stability qualification.**
  The global `0.002` candidate is rejected. The selected qualification contract
  is the `0.0005` same-model thinking tie-break, whose two offline screens are
  compatible but underpowered because it fired zero times. The exact
  1,000-decision, 41.2-million-token local and remote launch packet is now
  frozen and registered at Pathfinder `63eead46`: source, input, model, plugin,
  timeouts, acceptance gates, cleanup checks, and a cumulative conservative
  `$14.484864` envelope are pinned against the `$20` cap. The launcher defaults
  to packet-only mode and refuses Modal execution unless both
  `--execute-paid-1000` and `RSP-004Q-1000-CONFIRMED` are supplied. Actual
  1,000-decision arms launched: zero. Await fresh user confirmation before
  either arm. TD048 remains open for both the held full-corpus parity result and
  genuinely powered changed-action quality evidence (or an explicit reviewed
  decision accepting the narrow same-model canonicalization without it).
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
- [ ] **RSP-005 — Prove the selected explicit session end to end.** The engine
  gate, local HTTP lifecycle, capability-gated Go client, hermetic restart and
  Redis-loss stack, and real-GPU HTTP/concurrency/rebuild canaries pass. The
  automatic-prefix-cache design is rejected for the MVP. Record batching,
  eviction, affinity, and restart behavior in the development qualification
  before closing this rung.
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

The end-to-end stateless MVP is complete, the retained engine canary passes,
RSP-004A's implementation boundary is landed, and RSP-004Q is fully prepared
but held:

1. Treat the original post-stay `0.002` guard as rejected. On 60 canonical C82
   dev attempts it changed 40/524 decisions (`7.63%`) and increased switches
   `14→30`, failing the frozen behavior gate. The sanitized replay is pinned at
   `rayline-ai/router-artifacts@b947be95f9181058270b572d285c7efde5b5b074`.
2. Retire the global rule rather than promoting its `pre_stay` ordering. The
   targeted 178-state route-0 screen observed four cross-model changes, one
   unscorable change, mean paired reward delta `-0.1667`, and worst delta
   `-0.5`; it is rejected fail-closed. This screen excludes the C82 source
   lineages but does not claim complete task-identity disjointness. Evidence is
   pinned at
   `rayline-ai/router-artifacts@d4a2d67b10b0e435c70de10a320c2b0590d520e8`.
3. Use only the narrow `0.0005` same-model thinking tie-break. It changed 0/178
   targeted route-0 states and 0/524 historical decisions, with zero cross-model
   changes and switches preserved at 14. This establishes scope and historical
   compatibility, not changed-action task quality.
4. Treat the narrow-rule recanary as the final live readiness gate before the
   full corpus: it passed 6/6 with zero flips, exact token counts, `0.001191`
   maximum gap drift, `0.99998497` minimum embedding cosine, zero provider
   calls, and stopped cleanup. Its bundle is pinned at
   `rayline-ai/router-artifacts@b707b2715018edaa269e08e16f1755491d79fd06`.
5. Keep the frozen **RSP-004Q** packet at Pathfinder `63eead46` held. It is
   registered, digest-verified, dual-interlocked, and budgeted at a cumulative
   conservative `$14.484864` against the `$20` cap. Actual 1,000-case arms
   launched remain zero; only explicit user confirmation may change that.
6. Treat the RSP-005 MVP path as end-to-end proven: the capability-gated client,
   hermetic stack, protected H100 session endpoint, concurrent sessions,
   rebuild path, and real Semantic Router gateway are green.
7. Next, run a stratified 100–200-case development qualification for parity,
   latency, throughput, residency, eviction, affinity loss, and restart. Keep
   stateless full-history replay as the comparison and reconstructible fallback.
8. Keep the 1,000-case release qualification held until every smaller rung is
   green and the user explicitly confirms execution.

The completed 2026-07-30 full run remains RSP-004Q attempt 1 and a failed
receipt; it is not renamed or reinterpreted after the fact. The v1 plugin
continues to reject cached-prefix tokens. The separate session v1 wire reports
retained and appended tokens rather than mislabelling live-request reuse as an
automatic prefix-cache hit. RSP-002 remains pending until a Pathfinder human
accepts ADR 0064 (rayline-vllm-serving-boundary).

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
- Keep TD048 open until the narrow stability rule gains powered changed-action
  quality/regret evidence (or an explicit reviewed acceptance of its
  canonicalization semantics) and the full RSP-004Q parity qualification
  passes.

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
