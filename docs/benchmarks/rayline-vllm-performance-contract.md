# Rayline vLLM Performance Qualification Contract

Contract version: `rayline-vllm-perf.v1`  
Frozen: 2026-07-30 for PL-0041 RSP-001

This contract freezes the inputs and pass/fail interpretation before the
PL-0041 measured runs. A measured result may reject the design or motivate a
new, explicitly versioned contract. It must not silently move these thresholds.

## Product and Headroom Targets

The representative deployment target is:

- 4 downstream LLM request starts per second;
- 64 simultaneously active multi-turn episodes;
- 80% streaming and 20% non-streaming Chat Completions;
- routing adds no more than 1,000 ms at p95 to client-visible time to first
  token relative to static routing at the same offered load; and
- the selection plane sustains at least 8 decisions per second, 2x the target
  downstream start rate, while selection latency remains at or below 1,000 ms
  p95 and 2,000 ms p99.

These are qualification targets, not claims about observed production demand.
The report must state pass or fail without extrapolating beyond the measured
capacity envelope.

## Immutable Identity Pins

| Layer | Frozen identity |
| --- | --- |
| Semantic Router protocol baseline | `atlasfutures/semantic-router@33716d1106f42cf38565a296cd71c338f89a959c` |
| Pathfinder protocol baseline | `atlasfutures/pathfinder@5eeed94cf7bf3d5c1d79407f56d84e5af173a33b` |
| vLLM causal-MEAN baseline | `davidvgilmore/vllm@162bcefe1b41c5bb35eccc2f2219ea39e2c74bb7` |
| Rayline encoder | `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17` |
| C82 policy package | private `rayline-ai/mtrouter-c82@a06a4cc194761cfb39f92549ba305b0a8173a3d4` |
| C82 checkpoint | `provenance/source/mtrouter_estimator.pt`, LFS SHA256 `c2b0e63216c11f1496b47b22dff9f6c83baa6ef065e205a34897deff7493920f` |
| Serializer | `mtrouter-token-blocks-v2`, source tag `serializer-src-01b692ca14003693` |
| Pooling | causal masked MEAN, FP32 accumulator, L2-normalized 1024-dimensional result |
| Encoder execution | BF16, 262,144-token maximum, 8,192-token chunk grid |
| Self-hosted worker model | `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218` |
| Worker topology | two independent vLLM engines, identical model pin, distinct endpoints and served identities |
| Primary GPU | NVIDIA L40S 48 GB for each Rayline encoder and worker process |
| Reference GPU | NVIDIA H100 80 GB encoder rerun for hardware sensitivity only |

The two self-hosted workers deliberately use the same model and hardware. Their
routing identities remain distinct from their backend served-model identities.
This isolates routing placement, queueing, and transport from differences in
model speed or output quality.

Every receipt must add the exact implementation heads under test, container
image digests, plugin source digest, CUDA and driver versions, vLLM launch
arguments, GPU product and memory, tokenizer digest, config digest, benchmark
corpus digest, and random seed. A changed identity requires a new contract
version before a measured run.

## Workload

The corpus is deterministic, synthetic, and token-count controlled through the
frozen serializer. It contains no customer prompts or raw episode identifiers.
The seed is `20260730`.

### Episode shape mix

| Share | Shape | Prefix before measured turn | New serialized turn |
| ---: | --- | ---: | ---: |
| 50% | short chat | 2,048–8,192 tokens | 512 tokens |
| 30% | growing agent | 8,192–65,536 tokens | 2,048 tokens |
| 15% | large tool result | 65,536–131,072 tokens | 16,384 tokens |
| 5% | near maximum | 245,760 tokens | 2,048 tokens |

All counts are post-serialization. The generator must keep the final request at
or below 262,144 tokens and record actual full, serialized, cached-prefix, and
truncated counts. No truncation is permitted in the steady-state target cell;
truncation is a separate failure/fallback case.

Run both episode-popularity distributions:

- uniform over the active episode set; and
- skewed, with a deterministic Zipf exponent of 1.2.

Same-episode turns are serial. Different episodes may overlap and batch.
A separate correctness case deliberately races two same-episode prepares and
must prove fencing; it is not mixed into throughput results.

### Output and transport mix

- 80% of full-stack requests stream; 20% do not.
- Output limits are 64 tokens for 50%, 256 tokens for 35%, and 1,024 tokens
  for 15% of requests.
- Sampling is deterministic: temperature 0 and a frozen request seed where the
  backend supports it.
- Router-only workers return an immediate synthetic 2xx with deterministic
  usage. Full-stack workers execute the pinned 8B model.

### Load ladders

Closed-loop episode concurrency is `1, 4, 16, 32, 64, 128`.

Open-loop offered selection rates are `1, 2, 4, 8, 12, 16` decisions per
second. Arrivals use a seeded Poisson process. The benchmark does not use
client-side coordinated omission: latency begins at scheduled arrival time,
including queueing before a request can be issued.

The required target cells are:

- router-only: 64 active episodes at 8 selections/s;
- full stack: 64 active episodes at 4 downstream starts/s; and
- direct/static control: the same full-stack load and workers without a
  semantic selection call.

Continue the ladder until saturation or 16 selections/s. Saturation is the
first cell where any of these holds: fewer than 99.9% scheduled requests
complete, the decision p95 exceeds 1,000 ms, or queue depth has a positive
least-squares slope over the final five minutes.

## Run Discipline and Statistics

For every steady-state cell:

1. start from a declared cold or warm state;
2. warm for at least 2 minutes and 200 completed decisions;
3. measure for at least 10 minutes and 1,000 completed decisions, whichever
   takes longer; and
4. repeat three times with seeds `20260730`, `20260731`, and `20260732`.

Report each repetition separately and report min/median/max across
repetitions. Also report p50, p95, and p99 from mergeable histograms, never
from averages of percentiles. Compute 95% bootstrap confidence intervals with
10,000 resamples for throughput, p95 selection latency, and added TTFT.
All three repetitions must meet a hard qualification threshold.

Cold rebuild, eviction, affinity miss, and restart cases use at least 50
isolated trials per shape. Parity uses a fixed corpus of at least 1,000
multi-turn decisions spanning all four shapes and every fallback reason.

## Measurements

The machine-readable receipt must contain:

- scheduled, accepted, completed, failed, retried, and cancelled requests;
- client latency, TTFT, inter-token latency, and output tokens/s;
- selection end-to-end, VSR, prepare, encoder queue, tokenize, model forward,
  pool, policy head, renew, commit, provider, and settle timing;
- vLLM prompt and generation throughput, running/waiting requests, scheduler
  queue, GPU utilization, allocated/reserved memory, and cache block use;
- full, delta, matched-prefix, rebuilt, evicted, refused, and resident tokens;
- cache hit/miss reason, engine incarnation, and replica-affinity result;
- selected-worker distribution and dispatch identity;
- parity drift and selection flips; and
- all immutable identity pins listed above.

Telemetry must not contain prompts, tools, raw episode IDs, authorization
headers, provider receipts, cache tensors, or secrets.

## Qualification Gates

### Correctness and identity

- Every component proves its frozen identity before traffic.
- Full encode, current Pathfinder KV, stateless vLLM, and selected vLLM KV
  choose the same worker on the parity corpus: zero selection flips and
  adjusted top-two gap drift at or below `5e-3`.
- There are zero protocol, state-transition, dispatch-identity, streaming-
  lifecycle, or silent-fallback errors.
- Cache loss, eviction, affinity miss, and encoder restart rebuild from
  committed Pathfinder history and preserve the selection.

### Router-only capacity

At 64 active episodes and 8 offered decisions/s:

- at least 99.9% of scheduled decisions complete;
- selection latency is at most 1,000 ms p95 and 2,000 ms p99;
- the final-five-minute queue-depth slope is not positive at the 95%
  confidence level; and
- no retry or timeout is required to meet the latency gate.

### Full-stack behavior

At 64 active episodes and 4 downstream starts/s:

- routed request acceptance is at least 99.9%;
- p95 TTFT added by routing is at most 1,000 ms versus static routing at the
  same offered load;
- completed request throughput and output tokens/s are each at least 90% of
  the static control; and
- streaming commits only after the first upstream 2xx headers, with usage
  settled from the actual response.

### Cache and memory

- Warm eligible requests reuse at least 90% of their eligible prefix tokens.
- Selected vLLM KV p95 is no worse than 1.10 times current Pathfinder KV p95
  on the same workload and GPU.
- A maximum-context cold full rebuild is at most 60 seconds p95.
- Logical encoder residency is bounded at 600,000 tokens across all sessions,
  or by a byte bound demonstrated to be no less strict on the pinned model.
- There is no OOM, unbounded session growth, or silent eviction.
- After warmup, non-cache GPU allocation does not drift upward by more than 5%
  between the first and final two-minute windows.

Failure of a hard gate is a result, not permission to revise the gate. The
report may additionally show the highest load that passed.

## ARC and Remote Comparison Matrix

Run every viable row against the same workload and worker endpoints:

| Variant | Required |
| --- | --- |
| Direct worker, no Semantic Router | yes |
| Static Semantic Router route | yes |
| Current `rayline_arc`, full history | yes |
| `rayline_remote`, current in-process Pathfinder KV | yes |
| `rayline_remote`, stateless vLLM full history | yes |
| `rayline_remote`, selected vLLM cross-turn KV | yes |
| `rayline_arc`, selected vLLM cross-turn KV | if the selected primitive can serve ARC without a different cache contract |
| Same-Pod or embedded variant | diagnostic only |

The comparison report must separate selection-plane cost from downstream
generation time and show cold/warm latency, saturation throughput, memory,
cache effectiveness, failures, and operational ownership for each row.

## Three-Arm Directional Parity Packet

PERF015 is a directional router-only comparison across the
Modal-compatible in-process Rayline reference interface, `rayline_remote` with
the retained vLLM session service's stateless compatibility path, and
`rayline_arc` with that service's retained-session path. It is not the
1,000-case release qualification and must not be presented as a production
saturation result or as a measurement of Modal policy-process placement.

Before any arm launches, all three input receipts must declare the exact same:

- 128-case public synthetic corpus and workload digests;
- encoder model/revision, tokenizer digest, serializer, and policy artifact;
- GPU class, warm-state declaration, seed, placement profile, and worker
  topology digest; and
- measurement scope and selected-worker trace construction.

The first packet uses the router-only scope, eight-way closed-loop admission,
zero provider calls, and identical worker doubles. Cold start is recorded
separately and excluded from the warm percentile calculation. The three arms
run sequentially against the same pinned GPU class and placement profile so
their resource envelopes cannot overlap silently.

For this controlled architecture-boundary packet, the client, Pathfinder
reference process, Remote transaction process, and ARC gateway run on the same
London host; all encoder calls traverse the same public HTTPS path to the
single Modal `us-east` H100. The identity is therefore
`london-policy-us-east-encoder-public-https`. The arm name
`modal_inprocess` denotes the current eager `/v1/route` interface and optimistic
state transition, not a claim that its policy process is executing inside
Modal. A separate placement study is required for that deployment comparison.

The 128 measured decisions are 32 complete four-turn episodes selected from
the already content-addressed public parity corpus; eight warmup decisions are
two separate complete episodes. Episode lanes may overlap up to concurrency
eight, but turns within one episode remain serial. The Modal reference uses
`POST /v1/route`; Remote uses `prepare → synthetic-2xx commit → settle`; ARC
uses the normal OpenAI gateway and an immediate worker double. Remote never
aborts a successful measured turn: doing so would suppress committed
`previous_worker`, route-index, and input-token state and compare different
policy semantics. Its settle record carries the corpus's exact post-serializer
input-token count. All three paths make zero real provider calls.

New machine-readable inputs use `rayline.vllm.three-arm-input.v2`. The
comparator still validates v1 receipts for historical replay, rejects mixed
v1/v2 arms, and emits `rayline.vllm.three-arm-comparison.v2` for v2 inputs.
V2 adds scheduled/completed/failed counts and p50/p95/p99 for fixed `<8k`,
`8k–<32k`, `32k–<128k`, and `≥128k` input-token buckets; empty buckets carry a
null latency. The comparator rejects unknown fields, missing or duplicate arms,
malformed SHA-256 identities, mismatched case or bucket totals, inconsistent
throughput arithmetic, non-monotonic latency percentiles, and any identity
mismatch before producing a comparison. Failed gates remain evidence.

Absolute gates for every arm remain the frozen router-only targets:

- at least 99.9% completion;
- at least 8 decisions per second;
- at most 1,000 ms p95 selection latency; and
- at most 2,000 ms p99 selection latency.

Directional parity additionally requires both `rayline_remote` and
`rayline_arc` to achieve at least 90% of the Modal in-process throughput and
no more than 1.10 times its p95 latency. All three selected-worker trace
digests must match. The ARC-versus-Remote ratios are reported but do not add a
fourth promotion gate; they diagnose the cost of the extra Pathfinder
authority boundary.

The newly authorized incremental infrastructure ceiling is USD 20 above the
previous conservative USD 39.31282402 envelope. No launch may consume more
than USD 15 of that increment; at least USD 5 remains reserved for a diagnosed
rerun or cleanup discrepancy. Packet generation and local validation spend
USD 0. A launcher must fail before mutation unless its own exact maximum cost,
app names, cleanup owner, and stable-zero rule fit inside the USD 15 packet
ceiling.

PERF015 owns the exact Modal app `rayline-arc-session-encoder`, the local
Compose project `rayline-three-arm-perf015`, and a 65-second stable-zero cleanup
window. Its 90-minute paid wall limit plus one worst-case 31-minute orphan
request and five-minute scale-down tail bound the H100/CPU/memory envelope at
USD 10.1597328. The cumulative conservative maximum is USD 49.47255682, leaving
USD 9.8402672 under the new authority and zero provider spend.

### PERF015 result

The single preregistered run completed all 128 measured turns on every arm
with zero failures and an exact selected-worker trace match. The source-frozen
histories averaged about 42k input tokens and peaked near 248k.

| Arm | Throughput | p50 | p95 | p99 |
| --- | ---: | ---: | ---: | ---: |
| Eager local interface (`modal_inprocess`) | 0.234 rps | 15.82 s | 96.14 s | 108.09 s |
| Remote prepare/commit/settle | 0.251 rps | 14.09 s | 90.54 s | 99.92 s |
| ARC retained session/KV | 0.318 rps | 9.09 s | 76.70 s | 98.72 s |

All relative gates passed. ARC throughput was `1.359x` eager and `1.264x`
Remote; its p95 ratios were `0.798x` and `0.847x`. The overall comparator
status remains `failed` because every arm missed the immutable 8 rps, 1-second
p95, and 2-second p99 absolute gates. This is positive evidence for retained
KV, not production SLO qualification. The eager label still does not imply a
Modal-hosted policy process.

The launcher window, including cleanup, was 1,836.41 seconds. Its conservative
resource upper estimate is USD 2.4679120734; provider spend is zero. Cleanup
stopped Pathfinder, removed the exact Compose project and volumes, deleted the
ephemeral proxy, and held the protected encoder at zero containers for 65
seconds. Five aggregate-only receipts are privately round-trip verified at
`rayline-ai/router-artifacts@6e391a8b77394d730af2117ccc79482dd45c65de`.
The 1,000-case qualification was not executed.

The zero-spend follow-up captures one additional
`rayline.vllm.arc-telemetry.v1` sidecar before teardown. It persists only
aggregate component readiness, session create/append/rebuild/reuse counts,
full/serialized/retained/appended/cached/truncated token sums and counts, and
cache-miss token sums and counts. Session actions must reconcile with ARC
request counts or the launcher fails closed.

### PERF016 preregistered repeat

PERF016 is one repeat of PERF015, not a new load cell. It keeps the same
source-frozen packet, arm order, eight-way closed-loop admission, 8 warmup plus
128 measured turns per arm, London policy placement, public HTTPS path to one
protected Modal `us-east` H100, synthetic worker doubles, and absolute and
relative gates. It changes only the receipt surface: all arms must emit
`rayline.vllm.three-arm-input.v2`, and the launcher must capture the reconciled
`rayline.vllm.arc-telemetry.v1` sidecar before teardown.

The run ID is `rayline-three-arm-repeat-perf016-20260802`; it owns Compose
project `rayline-three-arm-perf016` and the same exact protected encoder app.
The closed PERF015 ID is not launchable from the generalized launcher. PERF016
has no whole-run retry: a failure closes this ID, and any source, workload,
placement, order, threshold, or cost change requires a new registry entry.

The run passes its evidence-integrity gate only if every arm schedules and
completes exactly 128 measured turns with zero failures and provider calls,
the three worker-trace digests match, every input-length bucket reconciles to
the arm totals, and ARC session-action counts reconcile to ARC requests. The
immutable absolute and relative performance gates still determine comparator
status. Results will be reported both overall and for `<8k`, `8k–<32k`,
`32k–<128k`, and `≥128k` histories. Two sequential samples may show whether
the PERF015 direction repeats; they do not establish a powered variance,
causal-placement, or saturation claim.

The paid wall limit is 40 minutes. Adding one worst-case 31-minute orphan
request and a five-minute scale-down tail bounds the resource exposure at
4,560 seconds and USD 6.1280928. Charging the complete PERF015 envelope first
produces a cumulative conservative maximum of USD 55.60064962 under the USD
59.31282402 authority, leaving USD 3.7121744. Provider calls and spend remain
zero, and the held 1,000-case qualification is unreachable from this launcher.

### PERF016 result

PERF016 completed every integrity gate: all three arms finished 128/128 turns
with zero failures or provider calls, their worker-trace digest exactly matched
PERF015, all v2 token buckets reconciled, and ARC telemetry reconciled 34
session creates plus 102 appends to its 136 measured-plus-warmup requests. No
session rebuilt or truncated.

| Arm | Throughput | p50 | p95 | p99 |
| --- | ---: | ---: | ---: | ---: |
| Eager local interface (`modal_inprocess`) | 0.259 rps | 14.33 s | 84.92 s | 94.88 s |
| Remote prepare/commit/settle | 0.277 rps | 12.04 s | 82.42 s | 92.18 s |
| ARC retained session/KV | 0.349 rps | 10.19 s | 63.07 s | 85.09 s |

All relative gates passed again. ARC throughput was `1.346x` eager and
`1.256x` Remote; its p95 ratios were `0.743x` and `0.765x`. PERF015's
ARC/Remote throughput ratio was `1.264x`, so the central direction repeated
almost exactly despite roughly 10% run-to-run throughput variation. Every arm
again failed the immutable absolute SLO gates.

ARC processed 5,703,416 full-history tokens as 1,205,793 retained and
4,497,623 appended tokens, a 21.14% retained share. Automatic-prefix-cached
tokens remain zero by design; explicit retained-session tokens are the relevant
cross-turn reuse measure. Compared with Remote, ARC p95 was `0.536x` for the
24 cases from 32k to below 128k tokens and `0.499x` for the 11 cases at or above
128k. It was `1.116x` on the 55 cases below 8k. Because these buckets include
client queueing at concurrency eight, especially lane-order head-of-line
blocking, they diagnose where to measure next rather than define an isolated
encoder service curve.

The launcher window, including cleanup, was 1,688.19 seconds, with a USD
2.2687301622 resource upper estimate and zero provider spend. Seven
aggregate-only files are privately round-trip verified at
`rayline-ai/router-artifacts@5bf052dffeaa5ffbfb5cc333741e18aaba81c9e0`.
The exact Modal app stopped with zero tasks and held zero containers for 65
seconds. PERF016 is closed without retry, and no three-arm experiment is
currently launchable from the source tree.

### PERF017 concurrency sweep result

PERF017 was the next bounded diagnostic, not a larger qualification. It derives
the first eight complete measured episodes and the first disjoint warmup episode
from the PERF015/016 packet. Every cell therefore contains 4 warmup turns and
32 measured turns, with 13 `<8k`, 11 `8k–<32k`, 5 `32k–<128k`, and 3 `≥128k`
measured histories. The derived corpus digest is
`72bbb22c6a8673d78cb4eadbce46ffd88f882f91f1880b4163e117f4679b1105`;
the topology digest remains
`ad0970c68d2e6b035c187d193f3da8ca49f48a68267bd323e0d66c9d44bcfddd`.
The concurrency-one, -four, and -eight workload digests are respectively
`a350a92ee0f38c3feb72407e9590da29b9ef70da2ca466d3959358c0999f8230`,
`2a5cc697004b95c9384489663b7d6d67e69e78c63b2b606c22e55ed58d02e5fb`,
and `a7cc6948e731fb8277bbc0d9b79a4b21539515402c3d2d1b146885056c31ebca`.

Each concurrency cell runs Remote and then ARC against the same corpus and one
already-warm protected encoder. Eager is excluded because PERF017 isolates the
production interface choice. Between cells, the launcher stops the fresh
Pathfinder process, deletes the fresh ARC Compose project and Redis volume,
closes every namespaced retained encoder session, and requires zero resident
sessions and tokens before advancing. Remote must also leave the encoder empty
before ARC begins. Run IDs participate in both Remote's HMAC episode key and
ARC's raw episode ID before hashing, so no cell can reuse another cell's state.

The evidence gate requires six valid v2 receipts, 32/32 completions and zero
failures or provider calls in every arm, matching Remote/ARC traces within each
cell and one trace across all cells, reconciled token buckets, 36 ARC actions
including warmup per cell, and successful local and encoder-state cleanup. The
comparison reports ARC/Remote latency and throughput at concurrency 1, 4, and
8 plus each arm's `c4/c1` and `c8/c1` scaling. Those ratios are diagnostic; no
new absolute performance threshold is invented from the two prior samples.

The run ID is `rayline-concurrency-sweep-perf017-20260802`. Its exact 30-minute
paid wall, 31-minute orphan request, and five-minute scale-down tail total 3,960
resource seconds and USD 5.3217648. Charging the full PERF016 envelope first
would make the cumulative conservative maximum USD 60.92241442. Preserving the
frozen USD 3 reserve requires cumulative authority of USD 63.92241442. The user
approved the proposed additional USD 5, raising cumulative authority to USD
64.31282402 and leaving USD 3.3904096 after the full envelope. The signed,
pushed source checkpoint set `LAUNCHABLE_CONTRACT` to PERF017. The one-shot
launch failed before any benchmark cell or provider request: the first
protected `/health` request exceeded its 30-second socket read timeout during
the Modal cold start, and that transport exception escaped the intended
15-minute readiness loop. The exact encoder app then stopped with zero tasks
and containers, and no local cell stack or sweep process remained. The failure
is conservatively charged as 216 resource seconds, or USD 0.29027808, from the
reported 119.535-second deploy, one 30-second request timeout, 65-second
stable-zero cleanup, and rounding. PERF017 is closed without retry.

### PERF018 identity-equivalent startup retry (authorized, unlaunched)

PERF018 changes only the run/state namespace and the readiness transport
handling: `TimeoutError` and `URLError` now become the same sanitized state
error already retried by the fixed 15-minute startup loop. The packet, source
corpus, topology, concurrency cells, arm order, placement, model, thresholds,
state-isolation rules, and six-receipt acceptance gate are otherwise identical
to PERF017. Its run ID is
`rayline-concurrency-sweep-perf018-20260802`, so it cannot append to or reuse
the failed run's output or retained-session namespace.

Charging the PERF017 startup failure makes the new prior conservative total
USD 55.89092770. PERF018 retains the same 3,960-second, USD 5.3217648 full
resource envelope, producing a cumulative maximum of USD 61.21269250 and
leaving USD 3.10013152 under the existing USD 64.31282402 authority. No new
authority is needed. The failure receipt is privately round-trip verified at
`rayline-ai/router-artifacts@d109f1201abf8c39cd824e637bf841872bb2bbf9`.
Pathfinder `01b78615` closes PERF017 and preregisters PERF018; the signed,
pushed Semantic Router checkpoint now opens only the new one-shot ID. The held
1,000-case qualification remains unreachable.

## External Provider Canary

This canary proves transport and settlement only; it is excluded from local
throughput comparisons.

- opt-in, single concurrency, at most 8 paid requests;
- at most 128 output tokens per request;
- hard observed provider-spend cap of USD 1.00 for the run;
- dedicated credential with an account-side USD 5.00 spend limit;
- frozen C82 worker/model routes
  `deepseek/deepseek-v4-flash@thinking-off` through `baidu/fp8` and
  `xiaomi/mimo-v2.5-pro@thinking-off` through `xiaomi/fp8`;
- provider fallback disabled and parameter support required; and
- abort before spend if the model, provider route, price snapshot, or
  credential scope cannot be proven.

The sanitized receipt records provider model and route identity, status,
streaming behavior, usage, and settled cost, but never the credential,
authorization headers, prompt content, or provider receipt body.

## Evidence Lineage

The frozen choices build on:

- Pathfinder `docs/history/2026-07-22-mtrouter-c82-perf-smoke.md`;
- Pathfinder `docs/history/2026-07-22-kvdelta-serving-phase1.md`;
- Pathfinder
  `docs/history/2026-07-26-kvdelta-s9-p95refined-recanary.md`;
- Pathfinder `docs/adr/0021-service-owned-kv-sessions.md`;
- Pathfinder `docs/adr/0023-process-global-kv-memory-owner.md`; and
- PL-0039's pinned Rayline ARC serializer, IO plugin, and causal-MEAN runtime
  receipts.
