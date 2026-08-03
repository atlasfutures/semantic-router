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

Status: active on 2026-08-02. The stateless end-to-end MVP parity gate, retained
engine gate, versioned HTTP/client integration, and real-GPU concurrent gateway
E2E pass. The explicit pinned-session design is selected, its 128-case
development qualification passes, and the first bounded full-stack performance
packet is complete. Its direct-only comparison was confounded by completion and
time-order variance; the source-frozen static-gateway follow-up passes and
isolates the protected encoder/session request as the dominant measured ARC
cost. Two encoder-only concurrency packets prove eight-way admission and a
complete 92-call workload, and the distinct eight-call PERF007 microprobe now
proves multi-request vLLM scheduling with a pre-execution batch width of seven.
PERF009 now closes the transaction-path concurrency gap with 128 capped
prepare/abort transactions through Pathfinder and the protected encoder at
`10.263 req/s`, Pathfinder in-flight `8`, encoder in-flight `7`, and vLLM
scheduled batch width `6`. PERF011 completes the first placement comparison:
pinning both components to Modal `us-east` did not improve p50 or throughput,
and PERF014 removes the largest region confound: a London Pathfinder calling
an explicitly `us-east` encoder reproduced PERF011's slower encoder time
without colocation. PERF015 then completed the source-frozen three-interface
packet with exact worker-trace parity and zero failures. ARC retained sessions
improved throughput by 35.9% over eager and 26.4% over Remote, but every arm
failed the immutable absolute SLO gates on 42k-token-average histories. The
PERF016 repeat reproduced the ARC/Remote throughput direction at `1.256x`,
localized the largest p95 wins to histories above 32k tokens, and proved 1.206M
tokens were explicitly retained across 102 session appends. Its queued `<8k`
tail regressed, so concurrency and service time still need separation. The
PERF017 launch stopped before measurement when its first protected health
request timed out during the Modal cold start and escaped the intended
readiness loop. Cleanup left zero encoder tasks/containers, sweep processes,
or new local stacks. The run is closed and conservatively charged USD
0.29027808. PERF018 is the identity-equivalent retry: the 32-turn
Remote-versus-ARC cells at concurrency `1`, `4`, and `8`, fresh
Pathfinder/ARC/Redis state, packet, placement, and gates remain frozen; only
the new run namespace and transient startup transport normalization change.
The PERF017 failure receipt is privately round-trip verified. PERF018 then
tolerated cold start and completed c1 Remote 32/32 at `0.314 rps`, but its
pre-ARC state gate caught a retained session created by ARC startup readiness.
The gate prevented a contaminated comparison; cleanup reached zero and the two
aggregate receipts are privately pinned at
`rayline-ai/router-artifacts@cb14a91e`. PERF019 fixed the production readiness
probe and passed all six Remote/ARC receipts at concurrency `1`, `4`, and `8`.
ARC throughput was `1.204x`, `1.209x`, and `1.207x` Remote while its p95 latency
was `0.931x`, `0.871x`, and `0.774x`; both arms scaled only about `1.05-1.07x`
from c1, exposing the shared single-encoder saturation boundary. All state and
resource cleanup passed, provider calls stayed zero, and private receipts are
pinned at `rayline-ai/router-artifacts@1bc01b2b`. The source interlock is closed
after the one authorized execution. PERF020 is implemented and locally
validated as the next bounded diagnostic: it replays the same 32 measured turns
per arm under seeded-Poisson open-loop arrivals at `0.15`, `0.30`, and `0.45`
decisions per second. It preserves per-episode ordering and fresh state per
cell, and records scheduled-arrival latency, service latency, start lag,
backlog, drain time, achieved start rate, and completion throughput. Its pass
status is integrity-only; the first overloaded rate is a preregistered
diagnostic, not a new production SLO. The complete 5,160-resource-second
envelope is USD 6.9344208 and leaves USD 19.27874248 reserve under current
authority. PERF020 completed once and failed integrity because Remote completed
only 16/32 turns in every cell while ARC completed 32/32. The sparse schedule
left 10.3-52.4 seconds between same-episode turns, beyond the direct Uvicorn
idle keep-alive; every lane consequently alternated one successful fresh
connection with one failed stale connection. ARC's Envoy boundary reconnected
and passed. Cleanup reached zero throughout, provider spend was zero, and the
observed infrastructure upper estimate was USD 1.63206983. The result also
showed that this 32-sample Poisson seed realized 1.24 times each nominal rate,
so successor knee logic must compare achieved starts with the realized
schedule rate. PERF020's source interlock is closed; its evidence cannot support
a Remote-versus-ARC capacity comparison. PERF021 is implemented locally as an
identity-equivalent successor under a new namespace. Its direct and ARC clients
close only the caller thread's connection after a complete routing decision,
so Remote still keeps prepare/commit/settle on one connection but cannot reuse
it after an idle server timeout. Strict v2 receipts now carry the realized
arrival rate derived from the frozen schedule span; all six PERF020 v1 receipts
remain replayable. The successor's USD 6.9344208 full envelope would bring the
cumulative conservative maximum to USD 66.66615137 and leave USD 17.64667265
reserve under the previous authority. The additional USD 50 authority raises
that full-envelope reserve to USD 67.64667265. Its implementation is pushed at
`7c685cca`, renewed authority at `8caf6b49`, immutable Pathfinder
preregistration at `ae205109`, attestation at `86f43d09`, and authorization at
`b53434ab`. PERF021 then passed its one execution: all six arms completed 32/32
with zero failures and identical traces, all telemetry and state-reset gates
reconciled, and cleanup reached zero. The source interlock is closed again.
The realized single-H100 saturation knee is bracketed between `0.1862` and
`0.3724` decisions per second; ARC throughput was 6.9%, 9.2%, and 13.5% above
Remote at the three ordered cells. The
independent endpoint therefore remains the MVP default while
retained KV is a measured optimization, not a production-readiness claim. The
separately held quality qualification and HA journal remain open; another
transaction concurrency proof is not required. PERF022's exact preregistration,
attestation, Pathfinder authorization, and source authorization are pushed at
`edfb58a2`, `6bb425d7`, `24b4a3d6`, and `06a4ba4a`. Its one launch stopped
before GPU hydration or measurement at Modal class-method endpoint lookup and
is closed with complete exact-name cleanup. PERF023 completed its exact four
arms and passed the performance comparator, but the launcher failed after both
stops because Modal app state had not yet converged to zero. Independent
cleanup verification reached exact zero; close PERF023 without retry and use
its measurements only as diagnostic evidence. PERF024 is the
identity-equivalent, cleanup-stabilized successor and passed, followed by the
identity-corrected PERF026 forced-remap packet and PERF027 real-replica-stop
packet. PERF027 proved the expected single-survivor capacity penalty: throughput
fell to `0.5929x` control while p50 service latency rose to `3.1924x`.
Semantic Router now implements the static `rayline.arc.encoder-failover.v1`
production contract with active/draining membership, persisted v2 owner
affinity, one explicitly status-gated remap, ambiguous-failure fail-closed
behavior, aggregate metrics, and explicit final-turn close fanout. A
two-encoder Envoy/Semantic Router/Redis integration stack exercises failover,
recovery, restart, Redis loss, cleanup, and privacy without a GPU or provider
call. The source-exact integration, IO-plugin tests, full serialized Semantic
Router suite, and repository CI gate pass. DYN006 then exercised the production
controller with three exact H100 encoder apps: both arms registered ready C as
revision 2 and placed measured sessions `[2,3,3]`; treatment drained/stopped A,
failed over exactly two sessions to `[0,4,4]`, and observed TTL removal at
revision 4. Its capacity gate passed at `0.8668x` control throughput, `1.0785x`
p50, and `1.4688x` p95 service latency. Cleanup reached stable zero and the
aggregate-only evidence is privately pinned at
`rayline-ai/router-artifacts@fb75f38d20c7fdd1a2565bce52b9dd094bc3285c`.
The launcher-window infrastructure upper estimate was `$4.08003918826349`,
bringing cumulative observed accounting to `$77.72054280274334` under the
`$134.31282402` authority. Fleet provisioning/operator integration remains
TD050; DYN006 cannot retry and the 1,000-case qualification remains held.
The single-router OpenRouter agentic packets through AGT006 are also closed:
AGT002 proved real ARC generation and a 16/0/8 natural DS4/MiMo/HY3 mix but
stopped at an obsolete three-worker coverage gate; AGT003 and AGT004 stopped
on the first static DS4 probe with a transient-looking HTTP 404. The zero-H100
DGN001 follow-up then proved direct, pinned-static, and unpinned-static requests
all succeed against the same real route, ruling out a deterministic model,
provider, path, or credential rewrite bug. AGT005 exposed process-local Modal
session affinity; DGN002 proved the singleton lifecycle correction. AGT006 then
passed singleton warmup and direct DS4/Baidu key readiness but again received
HTTP 404 on the first static gateway probe. It produced no performance result,
all run authorities are closed, and the 1,000-case qualification remains held.
Current published implementation heads:

- Semantic Router
  [`atlasfutures/semantic-router:codex/rayline-remote-mvp`](https://github.com/atlasfutures/semantic-router/tree/codex/rayline-remote-mvp)
  contains the capability-gated
  retained-session client, hermetic stack, bounded direct/static/ARC diagnostic,
  fixed three-model OpenRouter transport and retry contracts, and mandatory ARC
  readiness preflight. The protected session service explicitly enables vLLM
  iteration-detail capture, the minimal batch probe, and protected stateless
  pooling compatibility used by the Pathfinder transaction lane and an
  explicit `us-east` placement pin for controlled comparison. It also contains
  the static native-ARC replica membership, affinity, failover, close, metrics,
  and two-encoder integration contract derived from PERF024/PERF026/PERF027.
- Pathfinder
  [`atlasfutures/pathfinder:codex/rayline-vsr-mvp`](https://github.com/atlasfutures/pathfinder/tree/codex/rayline-vsr-mvp)
  PERF016 launch source at `78b9310a4b5ef46353c88ee31a30d38bde475d94`
  for the registered
  retained-session, real-endpoint and OpenRouter canaries, plus the closed,
  artifact-pinned direct/static/ARC stage, encoder diagnostics, PERF009 remote
  transaction-capacity result, PERF011 placement comparator, and PERF014
  explicitly region-pinned remote control, and the completed PERF015 result
  with private aggregate receipts and app-owned cleanup.
- vLLM integration
  [`atlasfutures/vllm:codex/rayline-vsr-mvp`](https://github.com/atlasfutures/vllm/tree/codex/rayline-vsr-mvp)
  at `9f5ea81ca0aa570aea46baf82311a1139c1267ca` for append-scoped
  timing, process-lifetime scheduler occupancy, and pre-execution scheduled
  batch-width telemetry.
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

### PERF022 Bounded Affinity Scale-Out Phase

PERF021 places the first overloaded single-H100 cell at a realized `0.3724`
decisions per second and shows further backlog at `0.5586`. The next justified
deployment experiment is therefore horizontal encoder scale-out, not a larger
qualification packet or another transaction-concurrency proof.

PERF022 keeps the frozen PERF021 corpus, seed, topology, model, artifact,
serializer, worker trace, and `r030`/`r045` schedules. It compares only two ARC
deployment shapes:

```text
arc_single
Semantic Router -> local affinity proxy -> encoder A (one H100 container)

arc_dual_affinity
Semantic Router -> local affinity proxy -+-> encoder A (one H100 container)
                                         +-> encoder B (one H100 container)
```

The two explicit Modal apps are
`rayline-arc-session-encoder-a` and `rayline-arc-session-encoder-b`; each keeps
the proven `max_containers=1` boundary. The proxy selects a replica from the
first 64 bits of the opaque episode SHA-256 modulo the replica count. Pooling,
subsequent appends, and explicit close for one episode therefore reach the same
process-local retained session. Both arms traverse the proxy so its local hop
is symmetric. The proxy records only bounded aggregate counts, never prompt
content, raw episode IDs, credentials, or request paths.

Each arm/cell owns a fresh Semantic Router, Pathfinder, Redis, Compose project,
proxy, and retained-session namespace. The encoder pair remains fixed across
the four arms so model hydration is outside the comparison, but every arm must
start and finish with zero resident sessions and tokens. The run contains four
receipts, 128 measured turns, 16 warmups, and zero provider or generation
calls. It passes integrity only when all turns complete, worker traces match,
provider calls stay zero, ARC telemetry records exactly 36 session actions per
arm, nine sessions are closed, both treatment replicas receive at least one
episode, and affinity mismatches remain zero. Reported throughput, service and
scheduled latency, backlog, and drain ratios are diagnostic rather than a new
production SLO.

The launcher is fail-closed until its signed Semantic Router implementation,
Pathfinder preregistration, self-attestation, and distinct authorization commit
are all remote-visible. Its two-replica envelope is 5,160 resource-seconds per
replica: 2,400 seconds paid wall time, 2,460 seconds for an orphaned request,
and 300 seconds scale-down. At the pinned Modal price snapshot this is
`$13.8688416`; added to the `$61.80928732218463` prior conservative total it
would reach `$75.67812892218463`, leaving `$58.63469509781537` under the current
`$134.31282402` authority. The launcher-window estimate also charges both
replicas for all elapsed time. No whole-run retry exists, and the separately
held 1,000-case qualification remains unreachable.

PERF022 is deliberately an experiment-side deployment proof. It does not yet
claim a production service directory, replica membership protocol, failover,
rebalance, shared cache, or HA transaction journal. Those boundaries remain
required before turning the local deterministic proxy into a supported public
deployment mode.

PERF022 launched once and stopped before GPU hydration or measurement. Modal
SDK 1.5.1 rejected `Function.from_name("SessionEncoder.web")`: class methods
must be resolved through `Cls.from_name("SessionEncoder")` and an instance.
The cleanup path then exposed a second pre-measurement defect because
`modal app stop` lacked `-y` and waited for confirmation until its timeout.
Manual recovery stopped the exact app A, verified app B was never deployed,
deleted the run's proxy token, and found zero named encoder containers, local
containers, affinity proxies, provider calls, warmups, measured turns, or
1,000-case qualification calls. The conservative launcher-to-verified-cleanup
upper estimate is `$1.01328552`, bringing the observed cumulative upper to
`$62.82257284218463`. PERF022 is closed without retry.

PERF023 is the identity-equivalent successor. It changes only the run and
resource namespaces, class-method endpoint lookup, noninteractive exact-app
cleanup, and prior-cost basis. Its full two-replica envelope remains
`$13.8688416`, which would bring the conservative cumulative maximum to
`$76.69141444218463` and leave `$57.62140957781537` under current authority.
Its corrected source is pushed at `dabac197`; immutable preregistration,
attestation, and Pathfinder authorization are pushed at `bef9a117`, `01263c5d`,
and `057f3d26`. Only the distinct signed source authorization checkpoint opens
its one-shot resolver.

PERF023 completed all four measurement arms and its strict comparison passed:
128/128 measured turns, 16 warmups, zero failures or providers, one shared
worker trace, zero affinity mismatches, exact 36-pooling/nine-close accounting
per arm, and zero retained state after every arm. At `r030`, dual affinity
improved completion throughput `1.3696x`, reduced p95 service latency to
`0.3338x`, reduced drain to `0.3975x`, and lowered final-arrival backlog by two.
At `r045`, the ratios were `1.3188x`, `0.5594x`, and `0.5740x`, with unchanged
final backlog. Both treatment replicas received sessions.

The packet is nevertheless not a pass: both exact app stops and token deletion
succeeded, but the launcher immediately read Modal's still-converging app state
and raised before writing its run manifest. Independent verification seconds
later found both apps stopped with zero tasks, zero named containers, zero local
containers or proxies, and no run token. The conservative 1,040-second upper is
`$2.7952704`, bringing the observed cumulative upper to `$65.61784324218463`.
PERF023 is closed without retry; its complete performance receipt is diagnostic.

PERF024 preserves every measurement input and adds only a bounded stable-zero
poll after noninteractive stop and token deletion. Its `$13.8688416` full
envelope would bring the cumulative conservative maximum to
`$79.48668484218463`, leaving `$54.82613917781537` under current authority. It
is preregistered, self-attested, and externally authorized at `e5ba2084`,
`a524d0d8`, and `739270a1`; only a distinct signed source checkpoint may open
its one execution. PERF024 passed that execution: 128/128 measured turns and 16
warmups completed with zero failures or providers, the worker trace and all
affinity/telemetry/state-reset gates matched, and bounded cleanup reached exact
zero. At `r030`, dual affinity improved throughput `1.1442x`, reduced p95
service latency to `0.7209x`, reduced drain to `0.7169x`, and lowered final
backlog by two. At `r045`, the ratios were `1.3990x`, `0.4849x`, and `0.5003x`,
with final backlog lower by one. The conservative 1,013.995-second resource
upper is `$2.725376389638886`, bringing the cumulative observed upper to
`$68.343219631823516`. Source authority is closed; the 1,000-case qualification
remains held.

PERF025 is the next bounded deployment phase. It keeps the passed two-replica
topology and exact `r030` packet, then compares dual sticky affinity with dual
forced affinity loss. The treatment remaps every episode to its peer after two
pooling calls: turn three must recreate from the supplied full history, turn
four must append on the peer, and close must fan out to both visited replicas.
Across the warmup plus eight measured episodes, the sticky arm must report nine
creates and 27 appends; the treatment must report 18 creates, 18 appends, nine
peer rebuild responses, 18 failover pooling requests, and 18 close attempts.
Both arms must complete 32/32, preserve the exact selected-worker trace, and
return both replicas to zero state. The comparator reports throughput, latency,
drain, backlog, appended-token work, and retained-token ratios without adding
an absolute SLO.

This is controlled affinity-loss injection, not a claim of real outage
detection, service discovery, membership, or ambiguous-transport retry. Those
production gaps are tracked in TD050. PERF025 has one `r030` cell, no provider
or generation path, no whole-run retry, and a `$7.4182176` full two-replica
envelope. That would bring the cumulative conservative maximum to
`$75.761437231823516` and leave `$58.551386788176484` under current authority.
Its implementation was pushed source-closed at `c449a396`. The distinct
Pathfinder preregistration, attestation, and authorization checkpoints are
pushed at `652bc815`, `adc8965a`, and `c1b080f4`; only a separate signed source
checkpoint may open its one execution.

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
The real-worker launcher separately pins the protected encoder URL and exact
`vllm@b1049f6d...` build identity, allows 180 seconds for its first retained
probe through strictly numeric config, and leaves the hermetic fake URL/build
as compose defaults. This separation was added after `rwe001` correctly failed readiness:
that first packet switched only worker endpoints, so fresh Modal credentials
were sent to the fake encoder and the real H100 service was never invoked.
The follow-up `rwe002` reached and passed that protected encoder probe, but its
launcher waited for `/health` on the Envoy generation listener. Envoy correctly
returned `503` for that non-generation route while the router API health port
was already `200`; the corrected launcher waits on the router API and still
sends all measured generation requests through Envoy. `rwe003` passed both of
those readiness boundaries and entered the direct-worker baseline, then Modal
rejected worker hydration before vLLM started. The worker Secret existed in the
local deploy graph only when the ephemeral key environment variable was set,
while remote import reconstructed an empty conditional list. The resulting
four-object-versus-three-dependency mismatch made no generation or provider
call. The next packet makes that Secret dependency structurally unconditional
across both import environments while retaining `server_command`'s fail-closed
authentication check.

`rwe004` validated that correction on both live L4 functions, then stopped at
the next pre-generation boundary: the pinned vLLM build rejected the legacy
`--disable-log-requests` flag. In this build request logging is already opt-in
through `--enable-log-requests`. An exact-commit source audit found every other
configured flag, including `--no-enable-prefix-caching`; the next packet removes
only the unsupported flag. It also pins Modal SDK `1.5.1` and invokes its CLI
through the same Python interpreter as the proxy-token API, eliminating a
local-library/system-CLI version split observed during zero-cost preflight.

`rwe005` then loaded the pinned model on both L4 workers. Its two cold direct
requests crossed Modal Web Functions' documented 150-second synchronous HTTP
window, which returns `303` while the original request continues. The canary's
low-level client treated that continuation as terminal, so generation completion
is recorded as unknown rather than zero. The next packet follows at most two
same-origin result redirects using `GET`, and refuses to forward the worker
bearer credential across origins.

`rwe006` live-validated that continuation path and completed all eight direct
OpenAI-compatible generations across both pinned L4 workers. The protected
encoder and ARC selection then processed the first routed request, and ARC
correctly replaced the public caller credential with the artifact-owned worker
key. That exposed a fixture wiring error: both selected routes in the hermetic
`envoy.yaml` still targeted `fake-provider`, which rejected the real-worker key
with `401`. No routed request reached a real worker. Cleanup left zero compose
containers or volumes and stopped both the generation app and exact H100
encoder container. Its credential-scanned private receipt is pinned at
`rayline-ai/router-artifacts@a76e51e5715df881fb4dea8641ee6c9f6b120294`;
the conservative attempt estimate is `$0.632927`, bringing the session plus
real-worker work to about `$2.632270`.

The correction keeps the default hermetic Envoy fixture unchanged and adds a
launcher-selected `envoy-real-workers.yaml`. It maps `worker-a` and `worker-b`
to separate DNS/TLS Modal clusters with system-CA validation, exact SNI, and
host rewrite, while ARC remains the sole owner of the upstream bearer
mutation. The dedicated fixture contains no credential. Local tests validate the launch selection, route
separation, absence of embedded authorization, compose interpolation, and
Envoy v1.34 configuration. A new Pathfinder experiment ID and signed Semantic
Router commit are required before that corrected path consumes paid resources.

`rwe007` started that dedicated Envoy fixture and again passed all eight direct
real-worker generations. Router startup had already warmed and validated the
protected encoder, but the L4 cold baselines took about six minutes, exceeding
the encoder's five-minute idle scale-down window. The first routed request then
triggered a second H100 cold start and exhausted the 180-second end-to-end
timeout before ARC produced a worker selection. No request reached either
routed worker. Cleanup again reached zero local and Modal resources. The
private receipt is pinned at
`rayline-ai/router-artifacts@e4570094ec30d369d738ab6127e302cccdf68bc0`;
its conservative `$1.144849` estimate brings session plus real-worker work to
about `$3.777119`.

The next packet adds one authenticated, continuation-aware `GET /health`
against the exact protected encoder immediately after the direct baselines.
It validates the retained pooling capability set, allocates no encoder session,
adds no generation request or prompt, and reports its latency. This makes
encoder residency at the start of routed measurement explicit without
extending the deployed H100 idle window or weakening the routed timeout.

`rwe008` validated that correction: both direct workers passed, the protected
post-baseline warmup restored the encoder, and 24/24 routed requests completed
through the protected H100 encoder, ARC, dedicated Envoy TLS route, and a real
vLLM worker. All 24 selected the same arm, so the preregistered two-arm coverage
gate correctly stopped before concurrency and streaming. This is a synthetic
fixture-axis problem rather than another transport failure: the test head reads
embedding coordinate zero, which did not separate the frozen public candidates.
The private receipt is pinned at
`rayline-ai/router-artifacts@5c2d4580caa4e7ff8e281dbd242207897e27fc36`;
its conservative `$1.050590` estimate brings session plus real-worker work to
about `$4.827709`.

The next bounded packet is an H100-only candidate-axis probe over those exact
24 public prompts. It ranks coordinates first by minority-sign count, then by
minimum and median absolute normalized margin, with the lowest coordinate as
the deterministic final tie-break. It requires at least six candidates on each
side and a `0.0001` minimum absolute margin. Its receipt contains only the
selected coordinate, sign counts, aggregate margins, timings, and cleanup—not
prompt text or raw embeddings. A passing result may update only the public
synthetic head and goldens on a new signed commit; the subsequent two-L4
real-worker canary requires its own preregistration and must retain every
coverage, concurrency, streaming, metrics, privacy, and cleanup gate.

`cap001` passed that gate on coordinate 252. Its 24 protected real embeddings
split 12/12 with zero exact ties; the minimum, median, and maximum absolute
normalized margins were `0.00129066`, `0.00513827`, and `0.01461668`. The
first attempt collected all embeddings but deferred session closes beyond the
service's three-session residency, so the oldest close correctly returned
`closed=false`; the non-material retry closed each session immediately and
finished with zero residents and zero running H100 containers. Across both
attempts, its conservative cost is `$0.193519`, bringing the session and
real-worker work to about `$5.021228`. The aggregate-only private receipt is
pinned at `rayline-ai/router-artifacts@e1cc725b829ec7fea88e708ee46e558cb56e0ef5`.
The public synthetic fixture now reads coordinate 252 in both its signed head
and hermetic fake encoder. This is deterministic deployment-test plumbing, not
a learned policy or task-quality claim.

`rwe009` then passed the complete bounded real-worker packet. Both direct L4
vLLM workers returned all eight baseline generations; the protected H100
encoder warmup passed; coordinate 252 reached both workers after four coverage
requests; four concurrent routed requests reached both arms at `0.845 req/s`;
and the routed stream emitted 34 data events before `[DONE]`. The router
reported nine session creates and zero ARC selection failures. In total, 17
real generation requests completed with zero provider calls. Compose and Modal
prompt scans found no public prompt bodies, the exact credential scan passed,
and cleanup left zero compose containers, volumes, L4 tasks, or H100/L4
containers. The digest-verified private receipt is pinned at
`rayline-ai/router-artifacts@2e76a0a7b4bb0d418c375c52d2bafd7c2d358992`.
Its conservative `$0.809061` cost brings all session-probe and real-worker work
to about `$5.830288`. This closes the real-endpoint MVP acceptance gate; it
does not yet establish saturation capacity or task-quality generalization.

The next self-hosted packet is frozen as a bounded first performance rung,
separate from both `rwe009` and the held release corpus. It compares direct
requests with the identical prompt/model distribution routed through ARC at
client concurrency `1, 2, 4, 8`, with two fixed waves at each level. It then
runs five ARC-only soak waves at concurrency four. Direct warmup and baseline
use eight generation calls, prompt coverage stops by 24 calls, the measured
ladder uses 60 calls, and the soak uses 20 calls: at most 112 real generations,
with 80 in the measured packet and exactly 40 measured requests per worker.
There is no duration-unbounded loop or case-count argument.

The packet passes only if all 80 measured generations succeed, every frozen
prompt keeps selecting its discovered worker, both workers are exercised at
every balanced level, and the five-wave soak completes without an error. vLLM
Prometheus counters must advance by exactly 40 successes per worker, prompt and
generation tokens must advance, TTFT, E2E, and queue-time histogram counts must
each advance by 40, and preemptions must remain zero. A background sampler
records observed request concurrency, queue depth, and KV utilization without
logging requests. The receipt reports per-wave p50/p95/max, throughput, direct
versus ARC throughput and p95 ratios, router actions, and component metrics.
It does not assign a production SLO or claim saturation from the maximum-eight
ladder. The launcher enables vLLM metrics, retains disabled request-body
logging and automatic prefix caching, refuses an already-running encoder, and
requires zero exact encoder containers after cleanup.

The self-hosted packet retains the established `$3.057473` maximum resource
envelope. Added to observed spend through `rwe009`, its conservative cumulative
envelope is `$8.887761`, below the `$20` cap. It requires a new signed Semantic
Router implementation commit and Pathfinder preregistration under
`rayline-arc-real-workers-perf001-20260731` before any Modal launch.

The preregistered packet completed on the exact frozen implementation and
passed every correctness, metrics, privacy, and cleanup gate. All 80 measured
requests completed with balanced 40/40 worker selections, each worker advanced
exactly 40 vLLM successes and latency-histogram observations, preemptions and
router selection failures remained zero, and the live sampler observed no
request queue. Four coverage requests plus eight direct setup/baseline calls
brought total generations to 92, below the 112-call maximum. The private
aggregate receipt round-trips byte-for-byte at
`rayline-ai/router-artifacts@e6cf0245ec9f97f0626939ba7cc7826d67497363`.
Cleanup independently found zero compose resources, generation-worker tasks or
containers, and encoder containers.

The performance result is a negative production-readiness signal. ARC/direct
throughput ratios were `0.256`, `0.218`, `0.157`, and `0.083` at concurrency
`1`, `2`, `4`, and `8`. At concurrency eight, maximum-wave p95 was `18.81s`
through ARC versus `1.88s` direct. The subsequent concurrency-four ARC soak
recovered to `1.70` requests per summed wave second with `2.59s` maximum-wave
p95, so the bounded evidence points to variable decision-plane contention or
queueing rather than a simple generation-worker saturation limit. Direct and
ARC completion-token counts also differed, which prevents treating the ratios
as a clean per-token service-capacity curve. The next self-hosted rung must add
stage-level traces around Envoy, Semantic Router, Pathfinder policy execution,
and the protected encoder, then isolate encoder concurrency and per-episode
session scheduling before assigning an SLO or capacity number. Conservative
span accounting charges `$0.926876` for this packet and brings cumulative
observed upper-bound spend to `$6.757164`.

The follow-up diagnostic is source-bounded before any new GPU launch. It adds
a specified-model gateway control between direct vLLM and full ARC, and sends
the same discovered prompt, 32-token limit, artifact-matching temperature,
thinking flag, and fixed seed through all three paths. Two waves at concurrency
one and four produce 30 measured generations, exactly 15 per worker, with at
most 62 total generations including warmup and coverage. Every self-hosted
gateway response must report exactly one Envoy attempt; local vLLM 429s remain
backpressure and are not retried.

The packet records client p50/p95/max and throughput, Envoy upstream service
time, the explicitly approximate client-minus-upstream residual, Semantic
Router routing and Rayline encoder histograms, and vLLM TTFT, E2E, queue,
preemption, running, waiting, token, and KV-utilization metrics. Direct requests
must add zero router observations, specified-model requests must add routing
but zero encoder observations, and ARC requests must add both. Completion-token
totals must match across all three paths per worker. This fixes the principal
confounder in `perf001`; it remains a diagnostic at concurrency one and four,
not a production saturation or SLO claim.

The source-frozen packet completed once and passed all correctness, metrics,
parity, privacy, and cleanup gates. It launched 42 self-hosted generations, 30
measured and exactly 15 per worker, with zero retries, selection failures,
preemptions, or worker queues. Execution fields and completion-token totals
matched across direct, specified-model gateway, and ARC paths.

The static-gateway arm is the causal baseline. ARC/static throughput was
`0.748` at concurrency one and `0.755` at concurrency four; ARC added `0.351s`
and `0.596s` to client p95. Static routing cost only `0.329ms` and `0.146ms`
mean, while ARC routing mean was `0.367s` and `0.597s`. The protected encoder
consumed `0.363s` and `0.595s`, more than 99% of the measured routing stage.
Static and ARC worker E2E means were nearly identical at both levels, maximum
worker queue depth remained zero, and maximum KV utilization was `0.00301`.
The next self-hosted rung should therefore instrument protected-encoder
in-flight and queue behavior directly, not increase packet size.

The aggregate-only receipt and manifest are private and byte-for-byte verified
at
`rayline-ai/router-artifacts@9592cdc676fedcba1512e071772f2771285a8793`;
Pathfinder closes the experiment at
[`dc4abee9`](https://github.com/atlasfutures/pathfinder/commit/dc4abee91c794ca91742a7501fade97aefa485cb).
Independent inventory found zero compose resources, zero generation-worker
tasks, and zero protected-encoder tasks. There were zero provider calls and
`$0` provider charge. Conservatively charging the full `$3.0574728` resource
envelope yields a `$14.23246402` cumulative upper bound and `$5.76753598`
headroom under the user cap. The held 1,000-case packet remains uninvoked.

The encoder-only observability rung then added aggregate coordinator counters
for tokenization, request and backend in-flight peaks, same-session lock
contention, append latency, failures, and token work, plus a curated view of
vLLM scheduler gauges and completed-request histograms. The implementation is
source-frozen at
[`83782ab9`](https://github.com/atlasfutures/semantic-router/commit/83782ab99316869b6eab47efc20dbc31a73a833a)
after the first attempt exposed and fixed a launcher failure-visibility gap.

The second source-frozen attempt failed at the first concurrency-one wave for
a substantive lifecycle mismatch: one retained append and explicit close both
returned HTTP 200, but vLLM's standard queue, inference, end-to-end, and
prompt-token completed-request histogram deltas remained `0/0/0/0` throughout
a ten-second settlement window. Those terminal-request histograms therefore do
not describe retained-stream appends, or even advance on this close path in the
exact pinned runtime. The packet stopped before accepting any throughput or
concurrency result. Full Prometheus-registry reads also took roughly
`165-230ms` at the warmed HTTP boundary, so they are too invasive for the
planned `20ms` sampler.

The aggregate-only failure receipt is private and byte-for-byte verified at
`rayline-ai/router-artifacts@28a3f5cf5b82a20f7b6f93f245d825a70e7f5685`,
path `runs/rayline-arc-encoder-service-perf004-20260801`. Both attempts deleted
their proxy tokens and left zero protected-encoder containers. Conservative
accounting charges both full H100 envelopes, bringing the cumulative ceiling
to `$19.23169762` and leaving `$0.76830238` under the cap. No further
full-envelope GPU packet may launch without renewed budget authority. The next
implementation must add append-scoped retained telemetry inside the vLLM seam
and serve cached aggregate snapshots before restoring the ladder.

The user-requested realistic data-plane packet is a separate three-model
OpenRouter canary, not an external-provider load test. A public synthetic
three-arm head maps the protected real encoder's coordinate 252 into positive,
near-zero, and negative regions. Its workers are pinned to:

- `deepseek/deepseek-v4-flash`;
- `moonshotai/kimi-k3`; and
- `z-ai/glm-5.2`.

All three OpenRouter arms pin the common `fireworks` provider, disable provider
fallbacks and reasoning, require declared parameters, and cap each completion
at eight tokens. This holds the hosting provider constant so observed model
latency and dispatch behavior are not confounded by OpenRouter's own provider
selection. Up to 24 routed public prompts may discover all three arms; one
direct and one routed request then exercise each model, followed by one routed
stream. The total is at most 31 paid model calls. Failure to cover all three
arms stops the packet; it is not repaired by changing the head or prompt set
after launch.

The OpenRouter launcher requires a management credential supplied at runtime
from 1Password. It creates a one-run API key with a server-enforced `$0.25`
limit, passes only that ephemeral key to Semantic Router, reads sanitized key
usage, scans compose logs for all credentials, and deletes both the OpenRouter
key and Modal proxy token unconditionally. The receipt must report OpenRouter's
per-response usage accounting and stay below a stricter `$0.10` aggregate
cost gate. Every response must identify Fireworks and the selected model, ARC
must cover all three workers without selection drift, streaming must reach
`[DONE]`, router failures must remain zero, and the compose stack, credentials,
and exact H100 encoder must all reach zero after cleanup.

The external packet's conservative envelope is the established `$2.499617`
protected-H100 timeout envelope plus the `$0.25` provider-key limit, or
`$2.749617`. If both new packets consume their full envelopes, cumulative
spend is bounded at `$11.637378`, leaving more than `$8.36` below the user cap.
It requires its own Pathfinder preregistration under
`rayline-arc-openrouter-orc001-20260731`. Neither packet executes or relaxes
the held 1,000-case release qualification.

ORC001 failed closed on its first coverage request with HTTP 503. Encoder
health warmup passed, but the router made zero encoder pooling calls and no
selected provider generation was observed. A zero-cost exact-config
reproducer emitted `llm_rayline_arc_component_ready=0` with failure class
`artifact_dispatch_contract`: the backend refs declared `provider=openrouter`,
which changed the canonical transport profile type even though the artifact
requires the OpenAI-compatible `openai` transport. Fireworks pinning remains
in the artifact dispatch fields. Commit `7e4672ff` corrects all three profiles,
adds a config-to-artifact contract test, and makes ARC component readiness a
mandatory pre-provider launcher gate. The full Semantic Router suite, CI gate,
and initial/resume/Redis-loss Rayline compose workflow pass.

The failed aggregate receipt is privately pinned and byte-for-byte verified at
`rayline-ai/router-artifacts@f8860b6b3ac12f45c1fb1965e39d199d8d21f156`.
Cleanup removed every compose resource, deleted the one-run OpenRouter and
Modal proxy credentials, and returned the exact encoder app to zero
containers. Because cleanup deleted the ephemeral key before its usage read,
provider spend is not asserted as zero; it is conservatively bounded at the
key's `$0.25` hard limit. Together with a 90-second full-run H100 span estimate,
the attempt upper bound is `$0.370949`, bringing observed cumulative
upper-bound spend to `$7.128113`.

The materially fixed retry is separately preregistered as
`rayline-arc-openrouter-orc002-20260731` at Pathfinder `358c0eee`. Models,
Fireworks pin, prompts, request and token limits, `$0.10` reported-cost gate,
`$0.25` key limit, privacy rules, and cleanup contract are unchanged. Its new
preflight must observe component readiness equal to one before any explicit
warmup or provider request. Observed prior spend plus its full packet envelope
is `$9.877730`; even the deliberately over-conservative all-rungs envelope is
`$14.386995`, below the `$20` cap.

Both permitted ORC002 attempts passed component readiness and completed the
real encoder selection call, but the first coverage request returned HTTP 503.
A hermetic three-arm request then reached its fake provider with the exact
DeepSeek/Fireworks payload and HTTP 200. The same route using a fake encoder,
real OpenRouter cluster, and intentionally invalid credential proved the 503
was Envoy's local fallback: ext-proc resolves the provider profile into
`/api/v1/chat/completions` and clears the route cache, but the Envoy worker
routes rematched only `/v1/`. No external model generation was observed.

Commit `db20cf48` changes all three matches to the resolved `/api/v1/` path and
removes the duplicate prefix rewrite. The focused contract tests, CI gate, and
three-phase Rayline compose workflow pass; the fixed zero-generation external
probe reaches OpenRouter and returns the expected 401 for its invalid key,
proving DNS, TLS, path rematching, auth forwarding, and upstream reachability.
ORC002's aggregate two-attempt receipt is privately pinned at
`rayline-ai/router-artifacts@49fdbe75edf9bb1bdd7d3031e8f12085f6f8d3e8`.
Conservative accounting retains both deleted keys' full limits and a combined
193-second H100 span, charging at most `$0.759371` and bringing cumulative
upper-bound spend to `$7.887484`.

The otherwise unchanged ORC003 packet is preregistered at Pathfinder
`88524778`. Observed prior spend plus its full `$2.749617` packet envelope is
`$10.637101`; the deliberately over-conservative all-rungs envelope is
`$17.136611`, still below the `$20` cap. The held 1,000-case packet remains
uninvoked.

ORC003 passed component readiness, the real retained encoder, corrected route
rematching, and three real Fireworks generations. Those successful coverage
responses validated output, per-response usage, provider identity, selected
worker, and selected model; two of three arms were covered. The fourth
coverage call returned HTTP 429. The source-frozen driver discarded both
`Retry-After` and structured provider error metadata and failed immediately,
so this is a failed full packet after verified external dispatch rather than a
router-path regression. Cleanup deleted the one-run OpenRouter key and Modal
proxy credential, removed every compose resource, and returned the exact H100
encoder app to zero containers.

The aggregate-only ORC003 receipt is private and exact-round-trip verified at
`rayline-ai/router-artifacts@e5ba12a8c5a031f46ca08e121c4488d17ce9e488`.
Conservative accounting retains the deleted key's full `$0.25` limit and a
108-second H100 span, charging at most `$0.395139` and bringing cumulative
observed upper-bound spend to `$8.282623`.

Commit `ecdb173a` adds diagnostic-canary resilience without claiming a
production data-plane fix: sequential requests are paced by one second;
pre-response HTTP 429 or 503 may retry once with the same Rayline episode ID;
`Retry-After` is honored within a 1–30 second clamp with a two-second default;
and no stream retries after HTTP 200. Error reporting emits only bounded type
and provider-code tokens, never the raw provider message. The receipt records
successful logical requests separately from external wire attempts, with hard
limits of 31 and 62 respectively. TD049 tracks the remaining production retry
ownership gap.

The exact ORC004 packet is preregistered at Pathfinder `b39b5c2a`. Models,
Fireworks pin, disabled fallback and reasoning, public prompts, eight-token
limit, `$0.10` reported-cost gate, `$0.25` one-run key limit, privacy rules,
and cleanup contract remain unchanged. The internal per-request retry is the
only authorized retry; no whole-run retry is allowed. Observed prior spend plus
the full packet envelope is `$11.032240`. The deliberately conservative
all-rungs envelope is `$19.886228`, leaving `$0.113772` below the user cap, so
any packet failure stops paid execution. The held 1,000-case packet remains
uninvoked.

ORC004 reproduced the same external limit after exercising the new boundary
exactly as designed. Three coverage requests again completed verified
Fireworks generations and covered two arms. Logical request four returned
HTTP 429; the driver reused the same Rayline episode and request after its
two-second default because no `Retry-After` header was present, and the second
external attempt also returned 429. Thus four logical calls produced five wire
attempts and three confirmed generations. The request pacing, retry limit,
privacy-safe exception, and fail-closed behavior worked, but three-model
coverage failed before the direct, routed-comparison, or streaming phases.

The private aggregate receipt is exact-round-trip verified at
`rayline-ai/router-artifacts@e060a95e4f1a03f1e369b31b271c9fc731c8ed24`;
Pathfinder records the closed result at `02f23596`. Cleanup again removed all
compose resources, deleted both transient credentials, and returned the exact
encoder app to zero containers. Conservative accounting retains the deleted
key's full `$0.25` limit and charges `$0.142707` for the 106.19-second H100
span, bringing cumulative observed upper-bound spend to `$8.675330`.

A zero-generation public endpoint inventory after the run gives a plausible,
but not conclusive, provider-side explanation. The ordinary Fireworks Kimi K3
endpoint reported degraded status `-2` and 93.91% trailing-30-minute uptime;
its premium `fireworks/fast` endpoint reported status `0` and 97.47% uptime.
The ordinary Fireworks endpoints for DeepSeek V4 Flash and GLM 5.2 reported
status `0`. Because the failed driver's selected worker was not persisted, the
Kimi correlation is an inference rather than a proven per-request attribution.

Paid execution stops at this boundary. A future packet must choose its claim
before changing transport policy:

- pin `fireworks/fast` only for Kimi to preserve one provider family while
  accepting a 50% higher Kimi token price and a non-identical service tier;
- allow OpenRouter fallbacks to test realistic availability, while giving up a
  controlled provider comparison; or
- pin one currently healthy provider per model, which preserves reproducible
  provider identity but measures a heterogeneous provider/model bundle.

Any option requires a new immutable artifact revision, source-frozen driver,
preregistration, and explicit bounded packet. At ORC004 closure the plan
authorized no additional paid generation. The held 1,000-case packet remains
uninvoked.

On 2026-08-01 the user selected the third option and authorized one bounded
continuation packet under the existing `$20` total cap. ORC005 replaces only
the degraded Kimi arm with `openai/gpt-5.6-luna`; it does not select the
premium Fireworks Fast endpoint. Every generation still traverses the
OpenRouter API and ephemeral OpenRouter credential. The immutable artifact
revision is `public-rayline-arc-openrouter-luna-v2`, with these execution
contracts:

- `worker-a`: `deepseek/deepseek-v4-flash`, pinned to standard provider slug
  `fireworks`, with `temperature=0`;
- `worker-b`: `openai/gpt-5.6-luna`, pinned to standard provider slug `openai`,
  with the `temperature` parameter omitted because the pinned endpoint does
  not advertise it; and
- `worker-c`: `z-ai/glm-5.2`, pinned to standard provider slug `fireworks`,
  with `temperature=0`.

All three contracts keep provider fallbacks and reasoning disabled and require
declared parameters. `fireworks/fast`, OpenAI Flex, and OpenAI Priority are not
allowed. The source-frozen price snapshot records Luna at `$0.10` prompt,
`$0.01` cache-read, `$0.125` cache-write, and `$0.60` completion per million
tokens from the
[OpenRouter Luna endpoint inventory](https://openrouter.ai/api/v1/models/openai/gpt-5.6-luna/endpoints).
The DeepSeek and GLM prices remain pinned to their standard Fireworks endpoint
snapshots. The compose backend profile remains the OpenAI-compatible transport
to `https://openrouter.ai/api/v1` for every worker; artifact fields, not the
transport profile name, own each downstream provider pin.

ORC005 retains the 31-logical-call and 62-wire-attempt ceilings, one-second
sequential pacing, one same-episode retry only for pre-response HTTP 429/503,
eight-token completions, `$0.10` reported-cost gate, `$0.25` ephemeral-key hard
limit, and exact cleanup/privacy gates. It must be preregistered against the
signed Semantic Router implementation commit before one launch; there is no
whole-packet retry. The `$2.749617` packet envelope added to the `$8.675330`
cumulative observed upper bound yields `$11.424947`, leaving more than `$8.57`
under the user cap. This authorization does not include the held 1,000-case
qualification.

ORC005 is now closed with no whole-packet retry. The protected encoder and
component-readiness gates passed. Four routed coverage generations exercised
all three workers for the first time and validated the exact
DeepSeek/Fireworks, Luna/OpenAI, and GLM/Fireworks contracts through
OpenRouter; a direct worker-a baseline also completed. The routed worker-a
comparison then returned HTTP 429, reused the same episode for its one allowed
retry after two seconds, and returned HTTP 429 again. Six logical requests
therefore made seven wire attempts and produced five confirmed generations;
the remaining comparisons and stream were not executed, so the full packet
gate failed.

OpenRouter's one-run key reported `$0.00004472` usage before deletion. The
failed-run launcher did not persist exact wall timing, so final conservative
accounting charges the full `$2.4996168` H100 timeout envelope rather than
inventing an observed infrastructure amount. The resulting cumulative upper
bound is `$11.17499122`, leaving more than `$8.82` under the user cap. The
aggregate-only receipt and manifest are private and exact-round-trip verified
at `rayline-ai/router-artifacts@ca708efafa93526c8f298a457ad7662fc737c9b7`;
Pathfinder records closure at
[`623933be`](https://github.com/atlasfutures/pathfinder/commit/623933be4008d180714a1be0091c6233f834747e).
Cleanup found zero compose resources and zero protected encoder containers and
deleted both transient credentials. This proves the OpenRouter-only Luna
topology can reach every arm without Fireworks Fast; it does not establish a
stable provider-throughput or latency result, and TD049 remains open.

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
  bounded policy-selection in-flight and queue-wait metrics. PERF009 completes
  the measured real-stack receipt with Pathfinder in-flight `8`, encoder
  in-flight `7`, and vLLM scheduled batch width `6`.
- [x] **RSP-005 — Prove the selected explicit session end to end.** The engine
  gate, local HTTP lifecycle, capability-gated Go client, hermetic restart and
  Redis-loss stack, and real-GPU HTTP/concurrency/rebuild canaries pass. The
  automatic-prefix-cache design is rejected for the MVP. Record batching,
  eviction, affinity, and restart behavior in the development qualification
  before closing this rung. Closed by the 128-case development qualification:
  retained/full replay parity, cross-episode overlap, LRU eviction, affinity
  loss, restart/rebuild, and zero-residency cleanup all passed.
- [x] **RSP-006 — Implement and harden vLLM KV reuse.** Add bounded cache
  ownership, exact fallback, same-episode fencing, privacy-safe metrics, and
  full-vs-incremental parity gates. Closed for the single-container MVP by the
  explicit retained-session implementation and its development qualification;
  multi-replica affinity remains production follow-up, not an unbounded cache
  correctness dependency.
- [ ] **RSP-007 — Add the production-shaped local stack.** Compose Envoy,
  Semantic Router, Pathfinder, dedicated Rayline vLLM, state store, and two
  worker vLLM endpoints through the normal local image flow.
- [ ] **RSP-008 — Add the benchmark harness.** Drive frozen open- and
  closed-loop workloads, collect synchronized client/component/GPU metrics, and
  emit one versioned machine-readable receipt plus a human report. Do not start
  the concurrency ladder until RSP-004A removes the transactional path's
  process-wide policy-selection lock for concurrent-safe MTRouter execution;
  otherwise encoder calls serialize before vLLM and the benchmark cannot
  exercise continuous batching. The identity-locked comparator, deterministic
  packet adapter, three protocol drivers, private runtime stager, generated ARC
  config, and ARC Compose profile are implemented under
  `e2e/testing/rayline-arc/` and `deploy/compose/rayline-arc/`. The packet
  preserves 32 ordered four-turn episode lanes at concurrency eight. Its
  Remote path commits and settles the synthetic 2xx result with the exact
  serialized input-token count instead of aborting successful turns. The
  private C82 runtime passes the real Go loader/golden preflight, and the ARC
  local gateway reaches component readiness plus HTTP 200 through Envoy and a
  worker double. The source/pin/budget fail-closed PERF015 launcher now owns one
  protected H100 app, one local Pathfinder process, and one exact Compose
  project, with a USD 10.1597328 worst-case resource envelope and 65-second
  stable-zero cleanup. PERF015 completed all three 128-turn arms with exact
  trace parity and zero failures. ARC achieved `0.318 rps` and
  `9.09s/76.70s/98.72s` p50/p95/p99 versus Remote `0.251 rps` and
  `14.09s/90.54s/99.92s`; relative gates passed, but every arm failed the
  frozen 8 rps / 1s p95 / 2s p99 absolute gates. Receipts are pinned at
  `rayline-ai/router-artifacts@6e391a8b`. The zero-spend follow-up implements
  v2 arm receipts with four fixed input-length buckets and an aggregate ARC
  telemetry sidecar captured before teardown; legacy v1 receipts remain
  replayable, but mixed schemas fail closed. PERF016 completed that exact
  repeat with trace parity, zero failures, all relative gates passing, and all
  absolute gates failing. Its ARC/Remote throughput ratio was `1.256x`; ARC p95
  ratios were `0.536x` from 32k to below 128k tokens and `0.499x` at or above
  128k, but `1.116x` below 8k under the queued lane mix. Aggregate ARC telemetry
  records 34 creates, 102 appends, zero rebuilds, and 1.206M retained tokens.
  Receipts are pinned at `rayline-ai/router-artifacts@5bf052df`.
  PERF017 derived eight complete measured episodes plus one disjoint warmup
  episode into a 32-turn packet spanning all four length buckets. It runs only
  Remote and ARC at concurrency 1, 4, and 8 against one warm encoder. Every
  cell owns a fresh Pathfinder process and ARC Compose/Redis stack; Remote must
  leave zero encoder residency before ARC, and exact namespaced ARC sessions
  are deleted and verified empty before the next cell. Six v2 receipts must
  complete 32/32 with zero provider calls and one shared worker trace. The
  comparator reports per-cell ARC/Remote ratios and per-arm `c4/c1` and
  `c8/c1` scaling without inventing a new absolute SLO. The 3,960-second full
  resource envelope was USD 5.3217648. Its first cold-start health request
  timed out before any cell ran, cleanup reached zero, and the one-shot ID is
  closed with a conservative USD 0.29027808 charge. PERF018 preserved every
  workload and acceptance detail under a new namespace and fixed only
  transient readiness transport handling. It then completed c1 Remote 32/32 at
  `0.314 rps` before its pre-ARC state gate found ARC startup readiness's
  retained session still resident. Cleanup reached zero and receipts are
  pinned at `rayline-ai/router-artifacts@cb14a91e`. PERF019 closed that
  production readiness session after probing and passed the fixed six-arm
  packet. ARC/Remote throughput was `1.204x`, `1.209x`, and `1.207x` at c1, c4,
  and c8; ARC p95 ratios were `0.931x`, `0.871x`, and `0.774x`. Both arms gained
  only about 5-7% throughput from c1 to c8, so the single remote encoder is
  already the shared bottleneck. Cleanup reached stable zero and aggregate-only
  receipts are privately pinned at `rayline-ai/router-artifacts@1bc01b2b`.
  Close PERF019 without retry. PERF020 freezes the same 32-turn packet into
  seeded-Poisson open-loop cells at `0.15`, `0.30`, and `0.45` decisions per
  second, with queue-inclusive scheduled latency, start lag, client backlog,
  and drain time separated from selector service time. Each rate gets fresh
  Pathfinder/ARC/Redis state and both arms must complete 32/32 with zero
  provider calls, matching worker traces, 36 ARC session actions, and empty
  retained state after cleanup. Saturation-knee reporting is diagnostic; only
  integrity controls pass/fail. The full USD 6.9344208 envelope is within the
  current cumulative authority. PERF020 executed once and failed integrity:
  Remote alternated fresh-connection success and stale-connection failure,
  completing 16/32 in all three cells, while ARC completed every turn. The
  10.3-52.4 second same-episode gaps exceed the direct server's idle keep-alive.
  Cleanup passed with zero residents and containers; the launcher-window upper
  estimate was USD 1.63206983 and providers remained unused. Close PERF020.
  PERF021 must close a thread-local client connection only after each complete
  decision transaction and compare achieved start rate with the schedule's
  realized rate. PERF021 implements that narrow delta with replay compatibility
  for all PERF020 v1 receipts. Its signed implementation, preregistration,
  attestation, and registry authorization are remote-visible; the source
  resolver pinned `b53434ab` for one execution. PERF021 passed with 192/192
  measured turns, zero failures, exact trace and telemetry parity, zero
  providers, and complete cleanup; close its source and registry authority.
  PERF022 implements the resulting bounded scale-out experiment: one ARC
  replica versus two explicit one-container ARC replicas behind the same
  deterministic episode-affinity proxy at only `r030` and `r045`. The
  one launch stopped at class-method endpoint lookup before GPU hydration or
  measurement and is closed. Exact cleanup reached zero after noninteractive
  manual recovery; its conservative upper is `$1.01328552`. PERF023 preserved
  the packet and topology under a new namespace while correcting only
  `Cls.from_name` endpoint resolution and `modal app stop -y`. All four arms
  completed and the comparator passed, but the run failed after measurement
  because exact-app stop state was checked before Modal's asynchronous state
  transition converged. Independent verification reached stable zero and the
  complete measurements remain diagnostic only. PERF024 preserved the exact
  packet under a new namespace, changed only cleanup verification to poll for
  bounded stable zero, and passed all 128 measured turns, comparison, and
  cleanup gates. Its two-replica throughput gain was `1.1442x` at `r030` and
  `1.3990x` at `r045`; private evidence is pinned at
  `rayline-ai/router-artifacts@cd832e8d`. PERF025 is source-closed with one
  `r030` sticky-versus-forced-remap cell to measure cross-replica rebuild and
  cleanup cost. Providers, generation, runtime-added cells, whole-run retry,
  and the 1,000-case qualification remain unreachable.
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
7. Treat the 128-case development qualification as complete: parity, latency,
   throughput, residency, eviction, affinity loss, restart, and cleanup passed
   with zero selection flips and no provider calls. Keep stateless full-history
   replay as the reconstructible fallback.
8. Treat the bounded real-endpoint MVP gate as complete at `rwe009`: protected
   retained H100 encoder, ARC, dedicated Envoy TLS routes, two real L4 vLLM
   workers, concurrent routing, streaming, metrics, privacy, and cleanup pass.
9. Treat the original direct-versus-ARC ladder as a confounded regression
   baseline, not a capacity curve. The source-frozen static-control diagnostic
   passed with exact token parity and measured ARC/static throughput ratios of
   `0.748` and `0.755` at concurrency one and four. More than 99% of ARC routing
   time was the protected encoder/session request, while generation-worker
   queues stayed empty. The subsequent encoder-only work replaced vLLM
   terminal-request histograms and hot Prometheus polling with append-scoped
   timing plus cached scheduler peaks. PERF005 completed all 92 calls with
   coordinator in-flight max `8`, but failed because its occupancy metric was
   sampled after one-step pooling requests completed. PERF006 moved the
   observation before output removal and emitted a complete failure receipt.
   At concurrency eight, create and append reached `5.742` and `5.831 req/s`;
   p95 latency was `1.407s` and `1.393s`; mean vLLM queue time was only
   `0.030ms` and `0.044ms`; coordinator in-flight max was `8`; and all 92
   requests passed. The engine reported waiting max `8` but scheduled max `0`,
   which traced to `ObservabilityConfig.enable_logging_iteration_details=False`,
   not proof of absent batching. Semantic Router `d70a35bd` enables that signal
   while keeping request logging disabled. The distinct PERF007 microprobe
   then passed one frozen eight-call wave: coordinator in-flight max `8`,
   pre-execution scheduled max `7`, waiting max `8`, zero failures, `4.208
   req/s`, `1.899s` p95, and `0.023ms` mean engine queue time. This accepts
   batch existence, not saturation capacity or an SLO. The v4 plugin source
   digest remains
   `67a9015c0c0399d4846930a9836982dd62c4a42f537af9f6c8917eb3beed23e5`.
   PERF005, PERF006, and PERF007 are privately pinned at
   `rayline-ai/router-artifacts@462cc5cefdba03ceb66284611dfa1f4da1652b98`,
   `rayline-ai/router-artifacts@67c44b5a188960a270756da3e62afc97f6d5d8be`,
   and
   `rayline-ai/router-artifacts@2ffc810d8494dd23e3811dff49b8cb2da7a4a014`.
   PERF008 then stopped before its soak because the cold readiness call exceeded
   Pathfinder's readiness TTL and the harness incorrectly required exactly two
   encoder calls. It made zero soak prepares and was closed without retry.
   PERF009 proved encoder metrics were zero before readiness and allowed the one
   legitimate TTL refresh. It passed all 128 capped prepare/abort transactions
   in `12.473s` at `10.263 req/s`; prepare latency was `0.721s` p50, `1.036s`
   p95, and `1.484s` p99. Pathfinder in-flight reached `8`, encoder backend
   in-flight reached `7`, vLLM scheduled batch width reached `6`, all 128
   encoder calls succeeded, and final residency, provider calls, and provider
   spend were zero. Mean encoder queue time was `0.013ms`, versus `196.347ms`
   inference and `273.647ms` encoder e2e. The private receipt is pinned at
   `rayline-ai/router-artifacts@f1fab622034e913400b6cc6962d020cbd7eeea98`.
   Conservative accounting charges both full envelopes and is now
   `$31.72978162`, leaving `$8.27021838` below the approved `$40` cap; the
   PERF009 launcher-window upper estimate was `$0.217977`. No prior packet may
   be retried or reinterpreted.
   PERF010 then failed during local module import before its budget guard or any
   external mutation; it created no app, credential, CPU/GPU resource, or cost
   and was closed without retry. PERF011 added a detonated direct-script
   regression test and completed the otherwise unchanged same-region packet.
   All 128 transactions passed at `10.199 req/s`; prepare latency was `0.752s`
   p50, `1.086s` p95, and `1.112s` p99. Pathfinder and encoder in-flight both
   reached `8`, scheduled batch width reached `7`, every encoder call
   succeeded, provider traffic was zero, and final residency was zero. Against
   PERF009, p50 and throughput ratios were `1.042` and `0.994`, so neither
   strong-placement threshold passed; p99 improved to `0.749x`, but mean
   encoder inference/e2e grew to `2.377x/2.413x` while queue time remained only
   `0.018ms`. This rejects colocation as an obvious p50/throughput win without
   claiming a pure network cause. The immutable private receipt is pinned at
   `rayline-ai/router-artifacts@02d01f19d6c481b5a2113ea8ece5065e0185a221`.
   Conservative accounting is now `$34.31359042`, leaving `$5.68640958` below
   the approved `$40` cap; the PERF011 launcher-window upper estimate was
   `$0.448730`.
   PERF012 then attempted the region-controlled remote topology: London
   Pathfinder to an explicitly `us-east` encoder. Its first zero-metrics call
   timed out at 90 seconds while vLLM was compiling the Qwen GDN Triton warmup
   kernel, before any prepare or provider call. It is charged the full
   `$2.4996168` envelope and was not retried. A disconnected Modal request
   continued creating replacement containers after zero inventories; exact app
   shutdown was required to cancel it. PERF013 failed closed at the existing-
   container preflight before creating a token, deployment, or cost.
   The separately preregistered PERF014 raised only the protected encoder
   cold-start deadline to 240 seconds and made cleanup own the stable exact app
   name across redeploy IDs. It passed 128/128 transactions in `14.623s` at
   `8.753 req/s`; prepare p50/p95/p99 were `0.950s/1.286s/1.308s`, Pathfinder
   and encoder in-flight both reached `8`, vLLM scheduled `7`, and failures,
   contention, residency, provider calls, and provider spend were zero. Its
   encoder inference/e2e means were `0.502s/0.614s`: `1.076x/0.930x` PERF011,
   but `2.558x/2.244x` PERF009. Because the explicitly pinned remote run
   reproduced the colocated encoder time without colocating Pathfinder,
   region/host/warmup/batching variability is the stronger explanation for the
   original encoder gap; one sample does not establish causality. End-to-end
   p50 and throughput were worse than both prior samples, so the evidence does
   not justify colocation.
   The PERF014 receipt is privately pinned at
   `rayline-ai/router-artifacts@81ab491a303c5e7b45e5706400fe748a1568ba50`.
   Conservative accounting through PERF014 is `$39.31282402`; its launcher-
   window upper estimate was `$0.330942`. Keep the independent endpoint as the
   MVP default.
   The user then authorized another `$20`, and PERF015 ran one exact
   source-frozen 128-turn packet per arm against the same protected `us-east`
   H100. Eager, Remote, and ARC each completed 128/128 with zero failures and
   the same worker-trace digest. ARC reached `0.318 rps`, `9.09s` p50, `76.70s`
   p95, and `98.72s` p99 versus Remote `0.251 rps`, `14.09s`, `90.54s`, and
   `99.92s`, and eager `0.234 rps`, `15.82s`, `96.14s`, and `108.09s`.
   Therefore every relative gate passed, including ARC/Remote throughput
   `1.264x` and p95 `0.847x`, while all arms failed the frozen absolute 8 rps,
   1s p95, and 2s p99 gates. This supports retained KV but rejects production
   SLO qualification on histories averaging about 42k tokens and peaking near
   248k. The observed launcher-window resource upper estimate was `$2.467912`,
   providers were unused, cleanup reached 65 seconds of stable zero, and five
   aggregate receipts are privately verified at
   `rayline-ai/router-artifacts@6e391a8b77394d730af2117ccc79482dd45c65de`.
   The cumulative full-envelope maximum is `$49.47255682` under the
   `$59.31282402` authority, leaving `$9.8402672` conservative reserve. The
   PERF016 then repeated the exact packet with v2 receipts. All arms completed
   128/128 with the same worker trace, and all relative gates passed again. ARC
   reached `0.349 rps` and `10.19s/63.07s/85.09s` p50/p95/p99 versus Remote
   `0.277 rps` and `12.04s/82.42s/92.18s`. Its ARC/Remote throughput ratio
   (`1.256x`) nearly duplicated PERF015 (`1.264x`). ARC explicitly retained
   1,205,793 of 5,703,416 full-history tokens through 34 creates and 102 appends
   with zero rebuilds. The p95 win concentrated above 32k tokens, while `<8k`
   was `1.116x` Remote under the queued lane mix. The run used a `$2.268730`
   launcher-window resource upper estimate, cleaned to stable zero, and is
   privately pinned at
   `rayline-ai/router-artifacts@5bf052dffeaa5ffbfb5cc333741e18aaba81c9e0`.
   Close PERF016 without retry. Preregister a bounded concurrency sweep with
   state isolation between cells before another launch; do not increase
   qualification size opportunistically. That PERF017 implementation and exact
   packet were ready, but PERF017 failed before measurement because the first
   cold-start health read timed out outside the readiness loop. Its cleanup is
   complete and the one-shot ID is closed. Normalize transient startup
   transport failures, preregister the otherwise identity-equivalent PERF018,
   and execute it once. PERF018 reached c1 Remote but correctly failed before
   ARC when startup readiness's retained session violated the empty-state
   gate. Close PERF018, make the production readiness probe close its session,
   and preregister otherwise identical PERF019. The additional USD 20 authority
   opened exactly one execution. PERF019 passed all six arms and cleanup gates;
   close its source interlock without retry and keep the held 1,000-case
   qualification closed. Implement and execute PERF020 exactly once after its
   signed source and registry checkpoints are both remote-visible. Use no
   provider or generation endpoint, do not add rate cells at runtime, and close
   launch authority after success or failure. Use its scheduled-arrival
   latency and final-arrival backlog to bracket the single-encoder knee before
   considering a multi-replica affinity experiment. PERF020 failed exactly
   once on direct-client stale idle connections and is closed. Prepare PERF021
   under a new ID with transaction-boundary connection close and realized-rate
   diagnostics; do not reinterpret the valid ARC-only curve as parity evidence.
   PERF021 passed its one authorized run and is closed: 192/192 measured turns,
   zero failures, exact trace/telemetry parity, complete cleanup, and zero
   provider spend. Preregister the next bounded deployment phase from the
   measured `0.1862`-to-`0.3724` single-H100 knee; do not reopen PERF021.
   PERF022 was that phase but stopped before GPU hydration or measurement when
   Modal rejected function-style lookup of a class web method; close it without
   retry and privately verify its aggregate failure receipt. PERF023 completed
   the identity-equivalent four-arm packet and its strict comparator passed,
   but it failed after measurement because cleanup verification raced Modal's
   asynchronous app-stop convergence. Close it without retry, privately verify
   its complete aggregate evidence, and retain the results only as diagnostic.
   PERF024 was the identity-equivalent successor and passed all four arms,
   comparator, state-reset, and stable-zero cleanup gates. Its private aggregate
   evidence is byte-verified at `rayline-ai/router-artifacts@cd832e8d`; close it
   without retry. Implement PERF025 as one `r030` dual-sticky versus dual-forced-
   remap cell. Remap every episode after its second pooling request, require the
   peer to recreate from full history, preserve the exact selected-worker trace,
   fan close out to both visited replicas, and report token-work plus latency
   cost under a `$7.4182176` envelope. Keep source closed until distinct
   Pathfinder preregistration, self-attestation, authorization, and source-pin
   checkpoints are pushed; then execute it once and close authority after any
   outcome. PERF025 completed 64/64 measured turns with zero failures or
   provider calls, exact selected-worker trace parity, nine peer-created
   responses, 18 fanout closes, and stable-zero cleanup. Its generated
   comparator passed, but the independent preregistration audit found that
   `_probe_cell` included the logical arm in the episode-hash namespace. The
   sticky primary distribution was `[3,6]`; failover stats v1 exposed only the
   treatment's visited distribution `[9,9]`, so it could not attest matching
   primary placement. Close PERF025 without retry and retain only its
   correctness, reconstruction, fanout, and cleanup evidence; its performance
   ratios are confounded and inadmissible as a clean failover-cost estimate.
   PERF026 is the identity-corrected successor. Give both logical arms the same
   explicit `shared-affinity` probe/session namespace while retaining distinct
   receipt names, emit failover-stats v2 with
   `primary_sessions_by_replica`, and require that vector to exactly equal the
   sticky arm's unique-session vector before computing performance ratios. Use
   the same r030 packet, arm order, fault boundary, topology, and `$7.4182176`
   envelope, with prior observed accounting `$70.1005119398672`; source-close
   it until distinct Pathfinder preregistration, self-attestation,
   authorization, and source-pin checkpoints are pushed. Execute it once after
   those gates and close authority after any outcome. Do not interpret fault
   injection as production membership or outage detection; keep TD050 and the
   held 1,000-case qualification open.
   PERF026 passed its one authorized execution and is closed: 64/64 measured
   turns, zero failures or providers, exact selected-worker trace, nine peer
   reconstructions, 18 fanout closes, stable-zero cleanup, and exact `[7,2]`
   cross-arm primary-placement identity. Forced remap increased appended-token
   work by `1.0575x` and p50 service latency by `1.1371x`; throughput was
   `1.0176x`, p95 service latency `0.9533x`, drain `0.9524x`, and final backlog
   unchanged. Treat the mixed latency/throughput direction as one bounded sample,
   not a speedup claim. Its observed launcher-window resource upper estimate is
   `$1.834964`, bringing cumulative observed accounting to
   `$71.9354755968929`. Privately verify the aggregate packet before registry
   closure. The next justified live phase is a staged real-replica-stop packet,
   not another forced-remap sample: preload an identical bounded set of retained
   sessions through two turns, stop one exact app only in treatment, require
   bounded transport-failure detection and remap to the surviving replica, then
   compare turns three and four against a no-stop control. Preregister a new ID,
   frozen stop boundary, survivor capacity, retry/idempotency semantics,
   aggregate failure/rebuild/fanout metrics, cleanup, and budget before opening
   source. Keep production exposure blocked by TD050 and keep qualification held.
   PERF027 is that source-closed staged packet. Use run ID
   `rayline-replica-stop-perf027-20260803`, the shared
   `shared-replica-stop` namespace, and the exact r030 corpus whose eight
   measured episodes place `[4,4]` across replicas; the one warmup episode makes
   the all-session vector `[5,4]`. In each arm, warm four turns and preload turns
   one and two for all eight measured episodes before the boundary. The control
   performs no stop. Treatment stops exact app
   `rayline-arc-session-encoder-a`, waits for stopped/zero-container state while
   encoder B remains deployed with one container, then measures only turns three
   and four on the same seeded post-boundary schedule. The proxy may retry an
   unavailable primary only after that orchestration proof, on explicit
   404/410/502/503/504 or transport failure; cache the episode remap so exactly
   four detections produce eight peer pooling calls and four created responses.
   Require identical preload and post-boundary worker traces, `[5,4]` primary
   identity, 16/16 post-boundary completions per arm, eight survivor closes,
   five unavailable-owner close skips, survivor zero state, exact final app/
   container zero, and aggregate-only evidence. The stop-convergence duration
   is reported but excluded from request latency. Freeze the same `$7.4182176`
   two-replica envelope from prior observed accounting `$71.9354755968929`;
   keep providers, generation workers, whole-run retry, and qualification out.
   PERF027 passed its one authorized launch and is closed. Both arms completed
   16/16 preload and 16/16 post-boundary decisions with zero failures or
   providers and exact preload/post-boundary worker traces. Treatment proved
   app A stopped with zero containers while app B retained one container,
   detected exactly four affected primaries through explicit HTTP failures,
   rebuilt four sessions with eight failover pooling calls, closed all eight
   measured sessions on the survivor, skipped five unavailable-owner closes,
   and ended with zero retained sessions/tokens and stable-zero resources.
   The real stop converged in `10.909s`, excluded from request latency. Under
   the surviving single encoder, post-boundary throughput was `0.1371 rps`
   versus control `0.2312 rps` (`0.5929x`), p50 service latency was `8.268s`
   versus `2.590s` (`3.1924x`), p95 was `76.219s` versus `28.707s`
   (`2.6551x`), and drain was `76.227s` versus `28.716s` (`2.6545x`). The
   small one-schedule result is capacity-impact evidence, not an SLO or
   variance claim. Treatment appended-token work was `1.0263x` control while
   retained work was `0.8945x`. Observed launcher-window resource upper cost
   was `$1.705028`, bringing cumulative observed accounting to
   `$73.64050361447986`. The ten aggregate-only files are byte-for-byte
   verified at
   `rayline-ai/router-artifacts@2c38ad5760961b04f80c4d2c9d5c1bd85c78ae41`.
   Stop further live expansion: use PERF024/PERF026/PERF027 to implement the
   versioned production membership, health, idempotency, observability, close,
   and rollout contract tracked by TD050. Keep the 1,000-case qualification
   held. That static production contract is now implemented as
   `rayline.arc.encoder-failover.v1`: two to eight stable replicas, explicit
   active/draining state, deterministic new-episode affinity, persisted v2
   owner/visited state, one status-gated remap, ambiguous-failure fail-closed,
   concurrent close fanout, and low-cardinality metrics. The real Envoy,
   Semantic Router, Redis, and two-fake-encoder integration passes failover,
   survivor stickiness, cooldown recovery, router restart, Redis loss, cleanup,
   and privacy. No GPU or provider was used and the implementation phase spent
   `$0`, so cumulative observed accounting stays
   `$73.64050361447986`; `$60.67232040552014` remains under the
   `$134.31282402` cap before the required `$3` reserve. Do not add another paid
   performance run from the present evidence: PERF027 already proves the
   expected single-survivor capacity penalty. Reopen live measurement only
   after a preregistered change to survivor capacity, automatic membership, or
   another deployment variable capable of changing that boundary. Track
   automatic provider/controller discovery under TD050 and keep the 1,000-case
   qualification held.
10. Treat ORC001 and ORC002 as closed local-contract failures and ORC003,
    ORC004, and ORC005 as closed provider-limit failures, all with complete
    cleanup and private aggregate receipts. ORC005 proves three-arm coverage
    through OpenRouter with DeepSeek/GLM pinned to standard Fireworks and Luna
    pinned to standard OpenAI, without Fireworks Fast or fallbacks; it still
    fails the complete direct/routed/streaming gate after the routed worker-a
    retry exhausts. Do not rerun it or interpret provider latency as a stable
    throughput benchmark. Preregister a traced self-hosted diagnostic that
    isolates the decision plane before another capacity packet.
11. Treat the production retry ownership implementation as hermetically green:
    Envoy retries one OpenRouter 429/503 below a single Rayline decision,
    honors bounded `Retry-After`, reports logical and wire counts separately,
    and never retries after HTTP 200. Self-hosted vLLM routes have no retry
    policy. The external canary no longer retries, but TD049 remains pending
    until a separately authorized real-provider confirmation passes.
12. Keep the 1,000-case release qualification held until the user explicitly
    confirms execution.

## TD050 Dynamic Membership Continuation

### Goal

Replace the manual, static retained-encoder replica list with an optional,
reviewed Redis membership source. The request contract remains
`rayline.arc.encoder-failover.v1`: deterministic affinity, persisted owner and
visited-owner state, one status-gated remap, ambiguous-failure fail-closed,
and explicit close fanout do not change.

### Task List

- [x] DYN001: Add the versioned Redis membership document and a runtime
  snapshot reader. Each router starts only after reading a valid two-to-eight
  member document and atomically adopts newer revisions without replacing a
  stable replica identity's endpoint.
- [x] DYN002: Add controller-owned active-to-draining and drain-completion
  operations. A controller must wait the idle boundary, prove no persisted
  owner or visited-owner references remain, and use a compare-and-set revision
  before removing a draining member.
- [x] DYN003: Preserve close and failure safety across snapshot changes,
  including retained clients for a removed owner until router shutdown and
  fail-closed behavior for an invalid or unavailable membership source.
- [x] DYN004: Add config/CLI parity, focused unit coverage, and the two-encoder
  Envoy/Semantic Router/Redis integration case for active, draining, and
  controller-confirmed removal.
- [x] DYN005: Deliver a standalone least-privilege membership controller
  command and image. `status`, `drain`, `reconcile`, and continuous `run`
  consume the canonical router config, resolve the write credential only in
  the controller process, emit privacy-safe JSON, and drive the Compose
  active-to-draining-to-removed acceptance without fabricating those
  revisions in the test harness.
- [x] DYN006a: Add idempotent controller `register`, readiness-safe router
  adoption of newly registered clients, controller-driven scale-out in the
  hermetic E2E, and focused contract tests.
- [x] DYN006b: Implement a source-closed three-encoder real-stop launcher,
  strict comparator, aggregate dynamic lifecycle telemetry, exact-image pin,
  and one-shot cleanup/budget interlocks.
- [x] DYN006c: Push signed source and registry checkpoints, open the exact
  one-shot authority, execute the preregistered cell once, privately verify
  aggregate evidence, and close launch authority after success or failure.

### DYN006 Result

DYN006 passed its single authorized execution. Both arms completed 16/16
preload and 16/16 post-boundary decisions with matching selected-worker traces,
zero failures, and zero provider calls. Controller registration produced
revision 2 and exact `[2,3,3]` placement in both arms. Treatment drain produced
revision 3, exact app A stopped in `0.876s` and converged in `16.576s`, two
affected sessions failed over to B/C, and the five-minute idle boundary ended
at revision 4 with `[0,4,4]` ownership and active B/C only.

Control completed at `0.2390 rps`, with `3.470s` p50 and `26.456s` p95 service
latency. Treatment completed at `0.2072 rps`, with `3.743s` p50 and `38.859s`
p95. Ratios were `0.8668x` throughput, `1.0785x` p50, and `1.4688x` p95, passing
the frozen `>=0.75x` throughput and `<=2.0x` latency gates. All 94 gateway
selections reconciled; treatment recorded exactly two failovers and two
unavailable-owner closes. All apps, containers, local stacks, episode states,
and the proxy token reached stable zero.

The exact source authorization was `493b2149`, permanent source closure is
`e260440d`, Pathfinder result closure is `028a37c6`, and the eight-file private
aggregate bundle is byte-for-byte verified at
`rayline-ai/router-artifacts@fb75f38d20c7fdd1a2565bce52b9dd094bc3285c`.
The 1,000-case qualification was not executed.

### Next Action

Continue the single-router end-to-end serving proof below. Park
multi-router transactional consistency in
[GitHub issue #2756](https://github.com/vllm-project/semantic-router/issues/2756)
and leave Kubernetes fleet provisioning under TD050. AGT001 through AGT006,
DGN001, DGN002, and DYN006 are permanently closed; do not rerun them. AGT006
passed direct key readiness but stopped on the first static gateway probe, so
no serving-performance result is admissible. Diagnose that seam without
reinterpreting the failed packet. Keep the 1,000-case qualification held until
explicit user confirmation.

### AGT001 OpenRouter Agentic Serving Diagnostic

AGT001 answers one bounded deployment question: what client-visible throughput,
TTFT, and end-to-end latency does the complete Rayline ARC serving path deliver
for small realistic agentic requests when generation is provided by three real
models through OpenRouter, and how much routing overhead does it add relative to
both direct OpenRouter and a specified-model static gateway?

The frozen generation pool is:

- `deepseek/deepseek-v4-flash`, pinned through OpenRouter to standard Baidu;
- `xiaomi/mimo-v2.5`, pinned through OpenRouter to standard Xiaomi; and
- `tencent/hy3`, pinned through OpenRouter to standard Tencent.

Provider fallback and reasoning are disabled, and Fireworks Fast, Kimi, GLM,
and Luna are absent. The source snapshot records the 2026-08-03 endpoint prices
and health contract. All requests carry public synthetic coding, research, or
incident-triage histories with an assistant tool call, a bounded tool result,
and a final synthesis turn. Output is streamed and capped at 96 tokens.

The benchmark first discovers exactly two frozen cases per selected worker and
requires all three workload shapes. It then compares the same six payloads
through `direct`, `gateway_static`, and full `arc` paths at concurrency one and
four. Two serial waves plus one doubled concurrency-four wave produce exactly
72 measured generations; coverage stops by 24 generations. The source contains
no request-count argument or duration-unbounded loop. The report includes:

- completed requests/second and output tokens/second;
- client TTFT and end-to-end p50/p95/max;
- per-model/provider latency, tokens, and cost;
- Envoy upstream service time, logical requests, external attempts, retries,
  and retry exhaustion;
- exact ARC selection coverage, session creates, failures, and cleanup; and
- ARC/static throughput and latency deltas beside the existing pure-Modal
  diagnostic reference (`0.748x`/`0.755x` throughput at c1/c4 and
  `+0.351s`/`+0.596s` p95).

The pure-Modal absolute throughput is not an interchangeable model benchmark:
it used Qwen3.5-0.8B generation on two Modal L4 workers and shorter inputs. Only
normalized ARC-versus-static overhead is compared directly. AGT001's absolute
OpenRouter numbers describe the requested hybrid deployment: local
Envoy/Semantic Router/Redis, the protected Modal H100 Rayline encoder, and real
OpenRouter generation.

The ephemeral OpenRouter key has a `$0.75` server-enforced limit and the report
has a stricter `$0.50` provider-cost gate. The 30-minute protected-H100 ceiling
is `$4.9992336`; the total full envelope is therefore `$5.7492336`. Added to
the `$77.72054280274334` cumulative observed accounting, the conservative
maximum is `$83.46977640274334`, below the existing `$134.31282402` authority.
The launcher must delete both transient credentials, stop the exact encoder,
remove Compose state, scan for credentials and public request bodies, and
reach stable zero after success or failure.

- [x] AGT001a: Freeze the requested model/provider pool, public tool-use
  workload, three-path comparator, source bounds, and focused unit tests.
- [x] AGT001b: Pass the source-exact artifact/config checks, focused Python
  suite, Rayline Compose integration, and repository gates.
- [x] AGT001c: Push the signed Semantic Router checkpoint, preregister and
  authorize the exact one-shot packet in Pathfinder, then execute once.
- [x] AGT001d: Privately pin the aggregate failure receipt, record that no
  pure-Modal comparison is admissible, close authority, and verify stable zero
  cleanup.

#### AGT001 Result

The single authorized AGT001 attempt closed during ARC startup on 2026-08-03,
before workload discovery or any generation request. The configured protected
encoder URL returned HTTP 404 and the Modal app recorded no request from the
attempt, so the router marked `artifact_head_encoder` not ready. OpenRouter
reported exactly `$0.00000000` of ephemeral-key usage. The launcher deleted the
ephemeral key and Modal proxy token, removed the Compose stack, stopped the
exact encoder container inventory to stable zero, and its signed source closure
clears the authorization pin. No latency or throughput inference is admissible
from AGT001 and it cannot be rerun.

Use a new AGT002 registry ID to replace the stale deployed-endpoint assumption
with an explicit source-pinned deployment of the current retained-session
encoder before creating the OpenRouter key. Preserve the frozen models,
providers, workloads, counts, metrics, cost gates, and pure-Modal comparison;
only the encoder deployment lifecycle and resulting identity pins may change.

### AGT002 Source-Pinned Encoder Retry

AGT002 preserves AGT001's exact three OpenRouter models/providers, public
agentic payloads, request counts, concurrency cells, output cap, direct/static/
ARC paths, metrics, retry bounds, cost ceilings, privacy contract, and
pure-Modal comparison doctrine. Its only experimental correction is the
external encoder lifecycle.

The retained-session encoder was explicitly deployed from Semantic Router
`0e07fa25` with plugin source digest `1ff4ee4d7a22`, vLLM `9f5ea81c`, and Qwen
revision `2fc06364`. Modal assigned deployed app
`ap-XtsWCBEWdw1ncu9Kv12Chj`; its protected route returns HTTP 401 without a
proxy token and its inventory is deployed with zero tasks. Before any
OpenRouter key creation, the launcher must:

1. attest that exact deployed app ID/name, zero-task state, protected route,
   deployment-source commit, plugin digest, vLLM build, and model revision;
2. refuse any pre-existing encoder container;
3. create a transient Modal proxy token and obtain a healthy zero-session,
   zero-token response from the protected endpoint, allowing the cold H100 to
   initialize; and
4. only then create the server-limited OpenRouter key and start Compose.

If encoder initialization fails, no OpenRouter key or provider request may
exist. Cleanup remains unconditional and stops the exact new app's container.
The packet retains the `$0.75` OpenRouter hard limit, `$0.50` reported-cost
gate, 30-minute H100 limit, `$5.7492336` total envelope, and existing
`$134.31282402` authority. AGT001 used zero provider and GPU spend, so the
conservative cumulative envelope remains `$83.46977640274334`.

- [x] AGT002a: Deploy and attest the current zero-task protected encoder; move
  encoder health before OpenRouter key creation; update exact cleanup ownership
  and focused tests.
- [x] AGT002b: Pass repository lint, focused tests, source-exact config startup,
  and the hermetic Rayline Compose suite.
- [x] AGT002c: Push the signed source checkpoint and complete distinct
  preregistration, source attestation, authorization, and launch-pin commits.
- [x] AGT002d: Execute once, privately pin the aggregate receipt, permanently
  close authority, verify stable zero, and compare normalized ARC/static
  overhead with the pure-Modal reference.

AGT002's only authorized attempt failed its preregistered coverage gate before
measurement. Across the maximum 24 discovery requests, C82 selected
DeepSeek/Baidu 16 times, MiMo/Xiaomi zero times, and HY3/Tencent 8 times; the
required two cases per worker therefore could not be frozen. No direct, static,
or ARC measurement cell ran, so no TTFT, latency, throughput, retry, or
pure-Modal performance comparison is admissible. OpenRouter charged
`$0.01228052`; the full launch-wall H100 upper estimate is `$0.769326504`.
Cleanup returned Compose and the encoder to stable zero, independently proved
the ephemeral key absent, and closed source authority at Semantic Router
`b8caee17`. The private aggregate failure receipt is pinned at
`rayline-ai/router-artifacts@53c13911` with SHA-256 `d0d307920420847f7f1b267276c256b261d447001b6017efd97ad1df3b4b6024`;
Pathfinder closes the packet at `6cdc1c4f`.

The next packet must treat natural ARC model share as a result instead of a
precondition: prove each native endpoint separately, allow a zero-share worker
in the routed workload, and freeze each static control to ARC's observed
assignment for the same request. That preserves a realistic model mix while
still measuring direct, static, and ARC overhead. Any such change requires a
new registry ID and authorization chain; AGT002 cannot retry.

### AGT003 Natural-Mix Measurement

AGT003 keeps the exact DS4 Flash/Baidu, MiMo V2.5/Xiaomi, and HY3/Tencent pool,
provider pins, public agentic histories, 96-token cap, 24-request discovery,
six-case and 72-request measured workload, concurrency cells, retry bounds,
cost ceilings, protected encoder deployment, privacy contract, and cleanup.
It changes only the failed coverage interpretation:

1. issue one specified-model gateway reachability probe to each exact endpoint;
2. run all 24 ARC discovery requests and report their natural model share;
3. require at least two active workers and select six balanced cases spanning
   all three scenario shapes, while allowing the third worker to have zero
   natural share; and
4. freeze direct and static controls to each selected case's observed ARC
   worker, then run the unchanged direct/static/ARC measurement cells.

The three reachability probes increase the logical provider ceiling from 96 to
99 and the two-attempt wire ceiling from 192 to 198. The `$0.75` key limit,
`$0.50` reported-cost gate, H100 limit, user budget, and held 1,000-case
qualification do not change. A distinct AGT003 registration and authorization
chain is required.

- [x] AGT003a: Implement and validate endpoint probes, natural-share selection,
  zero-share reporting, fixed request bounds, and aggregate-only evidence.
- [x] AGT003b: Preregister and complete the distinct source-attestation,
  authorization, and final launch-pin chain.
- [x] AGT003c: Execute once, privately pin the aggregate result, permanently
  close authority, verify stable zero, and compare normalized overhead with the
  pure-Modal reference.

AGT003's only authorized attempt failed its first specified-model gateway probe
with HTTP 404 before a completed response. Natural ARC discovery and all
measurement cells remained untouched, and OpenRouter recorded exactly `$0`
usage. The launcher returned Compose and the encoder to stable zero, proved the
ephemeral key absent, and closed source authority at `b1b291b2`. Its private
aggregate receipt is pinned at `rayline-ai/router-artifacts@3402e2ce` with
SHA-256 `bc655d826e1ba56d2d41ffe290ef4e0cc25c0d341b5c2c88aa454c43bdf2e920`;
Pathfinder closes the packet at `65ed3832`.

The specified-model control had supplied a public placeholder Authorization
value. Unlike ARC dispatch, which explicitly overwrites caller authorization
with its artifact-owned credential, specified-model routing uses the configured
credential path and must receive no caller credential from this unauthenticated
benchmark ingress. The hermetic stack already proves that a headerless static
control becomes exactly one config-owned provider credential and rewrites
`worker-a` to its external provider model ID.

### AGT004 Config-Owned Static Credentials

AGT004 removes caller Authorization from all gateway requests. Direct
OpenRouter requests continue to use the ephemeral key; specified-model gateway
requests rely on the router's config-owned credential; ARC requests rely on the
artifact-owned credential. The exact AGT003 models, providers, endpoint probes,
24-request natural-mix discovery, six selected cases, 72 measured requests,
request/attempt bounds, retry policy, cost ceilings, protected encoder,
privacy, cleanup, and pure-Modal comparison doctrine remain unchanged.

- [x] AGT004a: Implement and validate path-specific credential ownership,
  including the hermetic static-control provider-key and model-rewrite proof.
- [x] AGT004b: Preregister and complete the distinct source-attestation,
  authorization, and final launch-pin chain.
- [x] AGT004c: Execute once, privately pin the aggregate result, permanently
  close authority, verify stable zero, and compare normalized overhead with the
  pure-Modal reference.

AGT004's only attempt failed its first headerless worker-a static probe with
HTTP 404, exactly as AGT003 had. It therefore produced no discovery or measured
requests and OpenRouter usage remained `$0`. Cleanup returned the key, proxy,
Compose, and encoder-container inventories to zero. The aggregate-only receipt
is private and exact-round-trip verified at
`rayline-ai/router-artifacts@1eb0037c` with SHA-256
`8c647dd2010794c3da70356d4676cdd408e3f4706730cb486aceb21b756ab809`;
Semantic Router `5c9f0e7a` and Pathfinder `4e6ab4dc` close the attempt. Removing
caller Authorization consequently falsified the duplicate-credential theory,
but the failure still did not identify the malformed seam.

### DGN001 Real Static-Mutation Diagnostic

DGN001 used the same agentic config, Envoy route, router image, and local
contract-faithful encoder without starting Modal. Its one authorized packet
sent three one-token DS4-Flash requests:

1. direct OpenRouter with the Baidu pin;
2. headerless worker-a static routing with the Baidu pin; and
3. the same headerless worker-a static route without a provider pin.

All three returned HTTP 200 and the external
`deepseek/deepseek-v4-flash` model ID. The pinned static path used Baidu; the
unpinned static path used Morph. This proves the real gateway rewrites the
worker alias, provider model, path, and config-owned credential correctly. It
falsifies a deterministic static-mutation defect, but the direct-first order
does not distinguish transient endpoint availability from new-key or first-
request propagation. No latency or throughput inference is admissible from
three diagnostic calls. OpenRouter reported `$0`, no H100 started, and cleanup
reached zero. The aggregate receipt is private and byte-verified at
`rayline-ai/router-artifacts@86510f14` with SHA-256
`40d7a444844ba024492acdeed7ed42d17603fc518b67778e52fd9ad21a3eb274`;
Pathfinder closes DGN001 at `43d76aca`.

### AGT005 Key-Ready Natural-Mix Measurement

AGT005 preserves AGT004's exact model/provider pool, protected encoder,
headerless gateway credential ownership, three static endpoint probes,
24-request natural-mix discovery, six selected cases, 72 measured
direct/static/ARC requests, concurrency-one and -four cells, 96-token measured
cap, retry policy, metrics, privacy, cleanup, and normalized pure-Modal
comparison. It adds only one direct DS4-Flash/Baidu readiness request with a
one-token cap before the existing static probes.

The readiness canary receives at most two attempts and may treat an initial
HTTP 404, 429, or 503 as transient. All ordinary direct calls retain only the
existing 429/503 retry set; gateway retries remain owned by Envoy. The packet
therefore increases from 99 to 100 logical provider requests and from 198 to
200 maximum external attempts, while measured traffic remains exactly 72
requests. The `$0.75` key hard limit, `$0.50` reported-cost gate, 30-minute H100
limit, and `$5.7492336` packet envelope do not change. Charging the observed
AGT004 upper bound and DGN001's zero spend first gives a prior cumulative upper
estimate of `$79.110389914743`; the full AGT005 envelope reaches
`$84.859623514743`, leaving `$49.453200505257` under the current
`$134.31282402` authority.

- [x] AGT005a: Validate the one-token key-readiness probe, 100/200 request
  bounds, aggregate v3 report, source-exact config startup, and hermetic ARC
  stack; then push a signed source-closed checkpoint.
- [x] AGT005b: Preregister AGT005 in Pathfinder and complete distinct source
  attestation, one-attempt authorization, and final Semantic Router launch pin.
- [x] AGT005c: Execute once, privately pin the aggregate receipt, permanently
  close both authorities, prove stable-zero cleanup, and report real TTFT,
  latency, throughput, retry, cost, natural mix, and normalized ARC/static
  comparison. The 1,000-case qualification remains held.

The signed source-closed implementation is `29eb128f`, Pathfinder
preregistration is `9b115765`, the signed Semantic Router preregistration
attestation is `e9aea88b`, the distinct Pathfinder authorization is
`15657a24`, and its finalized registry attestation is `2b31fdcd`. The final
source pin names only that last remote-visible registry state.

AGT005's only attempt passed protected health, then failed before the direct
key-readiness canary because ARC's retained-session startup probe was not
transactionally affine. Modal's aggregate system log shows the probe `POST`
returned HTTP 200 from one H100 container, while the required `DELETE`
cold-started a second container and returned HTTP 200 after `78.9s`. The
process-local session was absent there, so its bounded `{closed:false}`
contract correctly left `artifact_head_encoder` not ready. OpenRouter usage
was exactly `$0`; no discovery or measured request ran. Cleanup returned the
key, proxy, Compose, volume, and encoder-container inventories to zero. The
conservative 188-second H100 upper estimate is `$0.522142176`, bringing the
cumulative observed upper estimate to `$79.632532090743`. The private failure
receipt is byte-verified at `rayline-ai/router-artifacts@1086cddd` with
SHA-256 `cbe8138dd8b7b95bfa247073f7d2098935766546c8fb7d275ba7b44dbf830170`;
Semantic Router `5acb9406` and Pathfinder `70154044` close the run. No
performance inference is admissible.

### DGN002 Singleton Retained-Session Affinity

DGN002 tests the deployment invariant exposed by AGT005 without creating an
OpenRouter key or starting generation. The benchmark launcher temporarily
overrides the exact deployed Modal class to `min_containers=1`,
`max_containers=1`, `buffer_containers=0`, and a 300-second scale-down window
before protected health. Unconditional cleanup restores `min_containers=0`
with the source-frozen remaining settings, then stops the exact container and
proves stable zero. This is not merely a warm-start optimization: the retained
KV/session owner is process-local, so one live singleton is required for
correctness during a run.

The paid diagnostic will create one transient Modal proxy, require empty
health, create and explicitly close one public synthetic retained session,
require empty health again, restore scale-to-zero, delete the proxy, and stop
the exact container. It records aggregate status, create/close success, token
counts, and resource upper bound only. No OpenRouter credential, model route,
prompt, embedding, raw episode ID, or timestamp may enter the receipt.

- [x] DGN002a: Validate singleton pin/restore ownership, source-close the
  launcher, and pass focused tests, repository lint, and hermetic ARC
  acceptance.
- [x] DGN002b: Preregister and execute one zero-provider retained-session
  create/close diagnostic, privately pin the aggregate receipt, and prove
  autoscaler plus container cleanup.
- [x] DGN002c: If affinity passes, preregister AGT006 as AGT005's otherwise
  unchanged full successor. If it fails, stop and redesign the state owner;
  do not hide the result with request retries.

DGN002 passed its only authorized attempt. With the exact Modal class pinned,
one protected container created an 11-token session at revision 1 in `1.258s`,
explicitly closed it in `0.456s`, and returned empty health in `0.441s`.
Cleanup restored the zero-minimum autoscaler, deleted the transient proxy, and
stopped the exact container with zero tasks. The 84-second conservative H100
upper estimate is `$0.233297568`, bringing cumulative observed accounting to
`$79.865829658743`; provider spend remained zero. The private aggregate
receipt is byte-verified at `rayline-ai/router-artifacts@7c970c93` with
SHA-256 `175267bb1da22c6970faf8dc6cb1197a322189b7430ca21e40ffca25bcb2ca14`;
Pathfinder closes the diagnostic at `5246afce`. This proves lifecycle affinity,
not throughput or high availability.

### AGT006 Singleton-Pinned Natural-Mix Measurement

AGT006 is AGT005 under a new run/state namespace with exactly one experimental
correction: DGN002's benchmark-owned Modal singleton lifecycle surrounds the
complete session-bearing window. The three OpenRouter models/providers,
one-token direct key readiness, three static endpoint probes, 24 ARC discovery
requests, natural-share case selection, six cases, 72 measured requests,
direct/static/ARC paths, concurrency one and four, 96-token measured cap,
100/200 request bounds, retry rules, aggregate metrics, privacy, and normalized
pure-Modal comparison remain unchanged.

Before proxy health, the launcher pins the exact deployed class to
`min=1/max=1/buffer=0`; unconditional cleanup restores `min=0/max=1/buffer=0`,
deletes credentials and Compose/Redis state, and stops the exact container.
The `$0.75` OpenRouter hard limit, `$0.50` report gate, 30-minute H100 ceiling,
and `$5.7492336` packet envelope are unchanged. Charging DGN002 first gives a
prior cumulative observed upper estimate of `$79.865829658743`; a full AGT006
envelope reaches `$85.615063258743`, leaving `$48.697760761257` under the
current `$134.31282402` authority.

- [x] AGT006a: Source-close and validate the singleton lifecycle, focused
  benchmark tests, repository lint, hermetic full-stack acceptance, updated
  performance contract, and execution plan.
- [x] AGT006b: Preregister AGT006 and complete distinct Semantic Router
  attestation, Pathfinder one-attempt authorization, registry attestation, and
  final source launch pin.
- [x] AGT006c: Execute once, privately pin the aggregate receipt, close both
  authorities, prove autoscaler and resource cleanup, and report the real E2E
  throughput, TTFT, latency, retry, cost, natural mix, and normalized
  ARC/static comparison. The 1,000-case qualification remains held.

The remote-visible launch chain is Pathfinder preregistration `25ef39da`,
Semantic Router source attestation `eb33a209`, Pathfinder authorization
`f97d502d`, and Pathfinder registry attestation `5df342cc`. The final signed
Semantic Router source checkpoint pins both Pathfinder authorities and is the
only checkpoint permitted to launch the one AGT006 attempt.

AGT006's only attempt passed the singleton encoder warmup and one-token direct
DS4/Baidu readiness, then the first specified-model static gateway request
returned HTTP 404 before a completed streamed response. Discovery and all 72
measured requests did not run, so there is no TTFT, latency, throughput,
natural-mix, or ARC/static comparison. OpenRouter usage was zero. Cleanup
restored the zero-minimum autoscaler, removed Compose and Redis state, deleted
both transient credentials, and left the protected app deployed with zero
tasks and containers. The 109-second H100 upper estimate is `$0.302731368`,
bringing cumulative observed accounting to `$80.168561026743`. The private
aggregate receipt is byte-verified at `rayline-ai/router-artifacts@ee2d6fc8`
with SHA-256
`d28b926c06cb94eddb56cab922376f233dee4e115aa94a68373de807a30bfc2b`.
Semantic Router closes source authority at `ae323259`, and Pathfinder closes
the registry at `e3908495`. The 1,000-case qualification remains held.

### DGN003 Gateway-Shape Isolation

DGN003 is a no-H100, six-request diagnostic for the remaining AGT006 failure
seam. It uses the exact agentic config, Envoy route, router image, fake encoder
contract, DS4/Baidu pin, and first public synthetic agentic case. In fixed
order it sends direct/static requests at one token, then two interleaved
direct/static pairs at the exact 96-token measured cap. It permits no
client-owned retry; Envoy may make at most one existing 429/503 retry per
gateway request. The packet therefore has six logical requests and twelve
external attempts at most.

The aggregate receipt records only path, token cap, status, bounded error
category/type/code, response model/provider, completion tokens, and attempt
count. It persists no request body, provider error message, credential,
episode ID, latency, or timestamp. This packet cannot support performance
inference. Its ephemeral OpenRouter key is hard-capped at `$0.05`; there is no
Modal H100 or proxy token. Charging the complete envelope gives a cumulative
upper bound of `$80.218561026743`, leaving `$54.094262993257` under the current
`$134.31282402` authority.

- [x] DGN003a: Source-close the local fake-encoder mode, six-request driver,
  privacy-safe upstream classifier, and focused/hermetic validation.
- [ ] DGN003b: Preregister and authorize exactly one diagnostic attempt, then
  pin the pushed registry authority in source.
- [ ] DGN003c: Execute once, privately pin and close the aggregate receipt,
  prove key/Compose/Modal inventories at zero, and choose the AGT007 correction
  from the observed direct/static and one/96-token matrix.

If direct and static behave alike, treat AGT006 as an upstream transient and
add only a bounded static readiness retry under a new full packet. If direct
succeeds while static fails, inspect the sanitized upstream category and fix
the gateway transport contract before another H100 run. If only 96-token calls
fail, correct the provider/request-shape contract. In every branch, keep the
1,000-case qualification held.

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
- [TD048](../tech-debt/td-048-rayline-vllm-selection-stability-gap.md)
- [Pathfinder ADR 0059 proposal](https://github.com/atlasfutures/pathfinder/blob/fb3a4b9455653eb9f8e490ca414aaa90a24e0a55/docs/adr/0059-rayline-vllm-serving-boundary.md)
- [Pathfinder stateless vLLM encoder implementation](https://github.com/atlasfutures/pathfinder/commit/7f13de3d10855ea44245717f9ccb50d55ea40e93)
- Pathfinder `docs/adr/0021-service-owned-kv-sessions.md`
- Pathfinder `docs/adr/0023-process-global-kv-memory-owner.md`
- Pathfinder `docs/history/2026-07-22-mtrouter-c82-perf-smoke.md`
- Pathfinder `docs/history/2026-07-26-kvdelta-s9-p95refined-recanary.md`
