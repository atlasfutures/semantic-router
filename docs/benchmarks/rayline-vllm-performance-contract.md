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

### PERF018 identity-equivalent startup retry result

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
authority was needed. The PERF017 failure receipt is privately round-trip
verified at
`rayline-ai/router-artifacts@d109f1201abf8c39cd824e637bf841872bb2bbf9`.
Pathfinder `01b78615` closed PERF017 and preregistered PERF018; the signed,
pushed Semantic Router checkpoint opened only the new one-shot ID.

PERF018 tolerated the cold-start timeouts, warmed one H100, and completed the
concurrency-one Remote arm 32/32 with zero failures or provider calls. It
measured `0.314 rps` with `1.04s/17.63s/17.67s` p50/p95/p99 and the expected
worker trace. Before ARC began, the new empty-state gate found the encoder was
not empty. Sanitized service logs and source inspection identify one retained
session created by ARC's startup readiness probe before Remote's 36 stateless
pooling requests. `EncoderClient.Probe` exercised the retained-session wire but
never deleted that fixed readiness session. The gate therefore prevented a
contaminated ARC comparison.

The exact app lifetime plus stable-zero cleanup is conservatively charged as
433 resource seconds, or USD 0.58190004. The app stopped with zero tasks and
containers, and no local process or Compose project remained. The aggregate
failure receipt and completed Remote receipt are privately round-trip verified
at `rayline-ai/router-artifacts@cb14a91eadc836e5a83ed5a04e9eb3aacceeb2f8`.
PERF018 is closed without retry.

### PERF019 retained-readiness cleanup retry (implemented and authorized)

PERF019 preserves PERF018's packet, placement, arm order, model, state-reset
rules, thresholds, and resource envelope under a new run/state namespace. Its
only semantic fix makes retained-session readiness transactional: after the
probe exercises the exact session wire, it issues an authenticated bounded
DELETE for the hashed readiness session, fails startup if the confirmed session
cannot be closed, and also attempts cleanup after an invalid probe response.

Charging both failed attempts makes PERF019's prior conservative total USD
56.47282774. Its unchanged USD 5.3217648 full envelope produces a cumulative
maximum of USD 61.79459254. The user approved another USD 20, raising cumulative
authority from USD 64.31282402 to USD 84.31282402 and leaving USD 22.51823148
after the full envelope. The source interlock opens only PERF019 after the exact
fix, registry pins, and signed pushed authorization checkpoints pass. The held
1,000-case qualification remains unreachable.

The single authorized PERF019 execution passed all six identity-locked
receipts. Remote throughput at concurrency 1, 4, and 8 was `0.240`, `0.255`,
and `0.253 rps`; ARC reached `0.289`, `0.308`, and `0.305 rps`, respectively.
ARC/Remote throughput ratios were `1.204x`, `1.209x`, and `1.207x`. ARC p95
latency was `21.25s`, `42.27s`, and `62.80s` versus Remote's `22.83s`, `48.53s`,
and `81.16s`. Every arm completed 32/32 with zero failures and provider calls,
one worker trace matched within and across cells, and every cell cleaned to zero
resident sessions and tokens. Pathfinder, Compose, Redis volumes, the ephemeral
proxy token, and the protected encoder all cleaned successfully.

The 1,210.55-second launcher window has a conservative infrastructure upper
estimate of USD 1.626833 and USD 0 provider spend. Fourteen aggregate-only files
are privately round-trip verified at
`rayline-ai/router-artifacts@1bc01b2bcb7c39e38ced1bfc630d3d6909d88cfd`.
The source interlock is closed after this success; PERF019 cannot retry, and the
held 1,000-case qualification remains unreachable.

## External Provider Agentic Qualification

This opt-in packet measures the complete single-router serving path against
real OpenRouter generation. Absolute provider latency and throughput describe
this packet only; they are not interchangeable with the self-hosted Modal
worker results because the generation models, prompt lengths, and provider
queues differ. Only normalized ARC/static ratios and latency deltas may be
compared across the two environments.

The frozen model pool is:

- `deepseek/deepseek-v4-flash@thinking-off`, pinned to standard Baidu;
- `xiaomi/mimo-v2.5@thinking-off`, pinned to standard Xiaomi; and
- `tencent/hy3@thinking-off`, pinned to standard Tencent.

Provider fallback is disabled and parameter support is required. No Fireworks
Fast, Kimi, or GLM route is permitted. The agentic histories are public
synthetic multi-turn code-patch, research-synthesis, and incident-triage
requests with tool schemas and bounded tool results. C82 routes 24 discovery
requests, and natural model share is a reported result rather than a
three-model coverage precondition. Six cases spanning all three scenario
shapes and at least two active workers are frozen for comparison.

Each frozen request runs through direct OpenRouter, specified-model static
gateway, and ARC paths at concurrency one and four. The measured packet is
exactly 72 requests with a 96-output-token cap. Before discovery, one direct
DS4/Baidu request capped at one output token establishes new-key/provider
readiness, followed by one specified-model gateway reachability request for
each worker. The complete AGT007 bound is 100 logical provider requests and
203 external attempts. The direct key-readiness canary may retry one initial
404/429/503. Each static endpoint-readiness probe may retry one initial 404;
that client retry set excludes 429/503 because those remain owned by Envoy.
Discovery and measured gateway calls do not retry 404, and ordinary direct
calls retry one pre-response 429/503.

The OpenRouter ephemeral key has a USD 0.75 server-side hard limit, and the
aggregate report rejects provider cost above USD 0.50. The protected Modal
encoder has a 30-minute paid-wall limit. Because retained sessions and KV are
process-local, the launcher must temporarily pin the exact deployed class to
one warm container for the entire session-bearing run; a scale-to-zero
transition between append or close requests is a correctness failure, not a
cache miss. Cleanup restores the source-frozen zero-minimum autoscaler before
stopping the exact container. The receipt reports aggregate request and
output-token throughput, TTFT, end-to-end and Envoy upstream latency, tokens,
retries, cost, provider/model identity, C82 natural mix, and ARC-versus-static
ratios. It persists no credential, authorization header, prompt, tool output,
routing anchor, per-request assignment, raw error body, or timestamp. Cleanup
must delete transient OpenRouter and Modal credentials, remove Compose and
Redis state, stop the exact encoder container, and retain the deployed app at
zero tasks. The 1,000-case qualification is unreachable from this launcher.

AGT006's only authorized attempt passed singleton encoder warmup and the
one-token direct DS4/Baidu readiness request, then the first DS4/Baidu static
gateway probe returned HTTP 404. OpenRouter usage remained USD 0, discovery and
all 72 measured requests did not run, and no performance inference is
admissible. Cleanup restored the zero-minimum autoscaler, removed all transient
credentials and Compose state, stopped the exact encoder container, and left
the deployed app at zero tasks. The source interlock is closed; AGT006 cannot
retry, and the 1,000-case qualification remains held.

The no-H100 DGN003 follow-up then interleaved six exact DS4/Baidu requests:
direct/static at one token and two direct/static pairs at 96 tokens. All six
completed with HTTP 200, exact model/provider identity, and one wire attempt;
the two static 96-token calls each emitted 96 tokens. This rules out a
deterministic specified-model rewrite or 96-token agentic request-shape defect.
Together with the intermittent AGT003/004/006 404s, it supports one bounded
HTTP 404 retry only in static endpoint readiness, not in measured traffic.
DGN003 spent USD 0.00033251 on OpenRouter and zero on Modal.

AGT007 then retried the first static 96-token DS4/Baidu readiness request once,
but both client attempts returned HTTP 404 after direct key readiness passed.
OpenRouter usage remained USD 0 and no discovery or measured request ran. This
falsifies the hypothesis that one generic readiness retry is sufficient. The
remaining sequencing difference is that DGN003 primed the gateway with a
one-token static request before its successful 96-token static calls. AGT007 is
closed and cannot retry; no performance inference is admissible.

DGN004 tested that remaining sequencing hypothesis with a fresh key. Static 96
tokens as the gateway's first request, a static one-token prime, static 96
tokens after the prime, and a direct 96-token control all completed with HTTP
200, exact DS4/Baidu identity, and one wire attempt. OpenRouter and Modal both
reported USD 0. This falsifies gateway priming as the explanation for the
intermittent full-stack 404. Another paid-encoder packet is not admissible until
the same OpenRouter/Envoy endpoint path is proven before encoder startup or the
full-stack edge failure gains privacy-safe observability. DGN004 is closed, and
the 1,000-case qualification remains held.

AGT008 moves that uncertainty outside the paid encoder window. With the same
ephemeral key, Compose project, router image, agentic config, and Envoy process,
it first starts against the public fake encoder and runs direct one-token
DS4/Baidu readiness plus 96-token static probes for DS4/Baidu, MiMo/Xiaomi, and
HY3/Tencent. Only a 4/4 pass may pin the protected singleton H100 encoder. The
launcher then recreates only the router with the protected endpoint and verifies
that the Envoy container and ephemeral key were preserved. The original
protected key readiness, three endpoint probes, 24-case natural-mix discovery,
and 72 measured direct/static/ARC calls remain unchanged.

The complete AGT008 bound is 104 logical provider requests and 214 external
attempts. Its aggregate v4 receipt includes the pre-encoder preflight but admits
performance inference only from the original 72 measured requests. The
preflight adds no H100 exposure; the protected window remains capped at 30
minutes. The ephemeral key remains hard-limited to USD 0.75, aggregate reported
provider cost remains capped at USD 0.50, and the complete conservative packet
envelope remains USD 5.7492336. The 1,000-case qualification is unreachable and
held.

AGT008's only attempt failed inside that four-request pre-encoder subprocess,
so the protected H100 never started. OpenRouter reported USD 0, no discovery or
measured request ran, and cleanup left Compose, key, and encoder inventories at
zero while retaining the deployed app at zero tasks. The launcher captured but
did not propagate the subprocess's privacy-safe error stream, so the failed
probe and bounded provider category are unknown. No performance inference is
admissible. AGT008 is closed and cannot retry; a successor must preserve the
zero-H100 gate while returning a structured aggregate failure receipt.

AGT009 is that otherwise identity-equivalent successor. A preflight HTTP
failure returns a bounded JSON receipt containing only stage, worker, status,
error category/type/provider code, completed requests, attempts, and completed
cost. Compose build output is captured so stdout contains only the aggregate
success or failure receipt. Raw provider text remains forbidden. All model,
provider, workload, retry, measurement, cost, H100, privacy, cleanup, and
1,000-case bounds remain unchanged from AGT008.

AGT009 stopped at its pre-encoder gate after direct DS4 readiness and the DS4
static probe completed. The MiMo v2.5 static probe returned HTTP 404
`no_endpoints`; the structured receipt records two completed requests, three
external attempts, and USD 0.0003326904 cost. Source reconstruction found the
exhausted 404 retry was not accumulated into the terminal exception, so the
actual minimum was four wire attempts. No H100, discovery, or measurement
request started, so the run provides transport evidence but no latency,
throughput, TTFT, natural-mix, ARC/static, or pure-Modal performance inference.

AGT010 preserves those three exact model identities while replacing each single
provider pin with a bounded OpenRouter provider order. Native providers remain
first; fallbacks are limited by validation to Baidu/StreamLake/DeepInfra for
DS4, Xiaomi/Parasail/Venice/Novita for MiMo, and Tencent/DeepInfra/Novita for
HY3. Every result reports its actual provider, conservative routing metadata
uses the maximum rate in each order, and exhausted client retries now propagate
their full wire-attempt count. No model substitution or Fireworks Fast model is
allowed. All workload, measurement, H100, cost, privacy, cleanup, and held
qualification bounds remain unchanged.

AGT010 then stopped at fake-encoder router readiness before any provider or H100
request because the existing Rayline ARC manifest contract requires exactly one
provider and rejects `openrouter_allow_fallbacks=true`. Cost was zero and no
performance inference is admissible. The next contract keeps automatic provider
fallback disabled but permits a bounded unique provider order with the preferred
provider first; router-controlled request retry and post-response provider
validation remain authoritative.

AGT011 makes that provider order a first-class Rayline ARC contract. The order
must be non-empty, contain unique non-empty slugs, and start with the preferred
provider. Automatic OpenRouter fallback remains forbidden. The exact agentic
artifact and direct/static controls all send the same order with
`allow_fallbacks=false`; successful responses outside the order fail closed and
actual provider identities remain in aggregate evidence.

The one authorized AGT011 run passed all 104 provider requests without retry and
completed the 72-request measurement. Natural ARC mix was DS4/MiMo/HY3
`16/0/8`; MiMo passed reachability through Venice but was not naturally selected
into the six-case measured set. ARC/static throughput was `0.762x` at c1 and
`0.588x` at c4. ARC minus static TTFT p95 was `+0.622s` and `+3.389s`; E2E p95
was `+0.008s` and `+2.804s`. The prior pure-Modal normalized throughput result
was `0.748x`/`0.755x`, so c1 is similar while c4 is not parity. The comparison
remains diagnostic because the prior target models and prompt lengths differ
and the new sample has only 12 requests per path/concurrency cell. Private
aggregate evidence is pinned at `rayline-ai/router-artifacts`, revision
`6039a41b8902445ef2ddf5f944cf3b2a60b4b544`, SHA-256
`0c2a6492e981e6c61915e686974ab062084badea7ffbbb232d24f6b848da6d31`.

### Retained-KV successor resilience contract

The next native-versus-vLLM retained-KV packet must preserve evidence and stop
unavailable-provider runs before GPU launch. Each measured logical request has
one crash-durable JSONL outcome. Successful events contain only routing identity
and numeric performance/usage evidence; failed events contain only bounded
status/category/type/code and attempt counts. Prompt, tool, response/error text,
credential, raw episode identity, and timestamps are forbidden.

Before either H100 paid timer begins, one shared direct OpenRouter gate sends a
one-output-token request to each exact model/provider order. It may retry one
pre-response 429/503 and must prove DS4 Flash, MiMo V2.5, and HY3 identity. This
gate establishes availability only; its latency and throughput are not
admissible performance evidence.

Measured routed traffic does not gain a client-owned retry loop. Native
Pathfinder and remote Envoy each own one bounded 429/503 retry below a single
Rayline selection transaction. Client replay would repeat semantic selection
and invalidate the session-action and decision-join evidence. Reports must keep
logical routed requests, server-owned provider attempts, and direct-preflight
attempts distinct.

The workload has two explicitly different coverage claims:

- the AGT018 natural semantic-cache lane requires the frozen public code,
  research, and incident/source-correlation histories to select worker-c,
  worker-a, and worker-b across all three growing states under both encoder
  architectures;
- the stratified static serving lane must reach all three models but is never
  evidence that the semantic classifier naturally covers those workers.

Exact native offline evaluation passed with traces `C/C/C`, `A/A/A`, and
`B/B/B`, serialized histories from `8,194` through `16,204` tokens, and minimum
top-two score gap `0.0019787615092044693` against a `0.0015` gate. This closes
the native prompt-discovery part of TD051 but not cross-architecture parity.
The vLLM-hosted encoder must reproduce all nine states before routed provider
measurement.

That successor seam is implemented source-closed. The protected encoder probe
runs after activation and before routed traffic, checks exact session
create/append revisions and retained-prefix accounting, evaluates the nine
embeddings without persisting them, and closes every probe session on success or
failure. The paired v3 reporter requires the offline and remote traces to equal
the frozen trace, joins every native decision, validates all retained/replay
cells, and emits aggregate whole-run, per-history, and per-model latency, first-
token, router/encoder, token-work, throughput, retry, provider, and cost
evidence. Empty authority pins plus zero key/time ceilings remained mandatory
until the reviewed 2026-08-05 budget checkpoints bound both pins and replaced
the ceilings with authorized per-arm key limits and a 20-minute paid wall
under fresh `$10` user authority. The per-arm key limit is `$0.15`: the
AGT017-era `$0.05` proved unable to host the successor workload because
OpenRouter's limit check counts in-flight pre-authorization holds and returned
HTTP 402 at request 35 of 36 with only `$0.025` settled usage.

The successor is
`rayline-openrouter-kv-cache-agt018-20260804`, artifact
`public-rayline-arc-openrouter-kv-cache-v4`, report schema
`rayline.openrouter-kv-cache-comparison.v3`. It bounds 36 routed requests per
deployment plus six total direct availability probes: 78 logical provider
requests and 156 worst-case external attempts. AGT017 remains historical and
cannot be retried or reinterpreted.

On 2026-08-05 three consecutive AGT018 availability preflights failed on
worker-a with upstream HTTP 429 before any GPU or measured spend. Diagnostic
probes with the byte-exact frozen payload showed the frozen worker-a order
Baidu/StreamLake/DeepInfra had collapsed to Baidu alone (the other two return
404 under `require_parameters`) while Baidu's shared upstream pool reported
`tpm_rate_limit_exceeded` intermittently. Artifact v3, which carried that
order, was retired without ever being deployed. Artifact v4 re-vets worker-a
to Baidu/GMICloud/SiliconFlow — both alternates are fp8 endpoints that passed
the byte-exact frozen payload, and OpenRouter falls through a rate-limited
provider to the next pinned entry with fallbacks still disabled — and adopts
conservative maximum-rate pricing across the amended order. Workers b and c,
the workload, the request envelope, both retry policies, and the nine-state
encoder gate are unchanged.

AGT018d executed both arms once on 2026-08-05. The vLLM-hosted encoder
reproduced all nine frozen states before routed measurement and both arms
completed `36/36` requests with exact selected-worker trace parity. Nine of
ten acceptance gates passed; `matched_completion_policy` failed because
worker-b was served by Xiaomi natively but fell through to Venice remotely,
yielding different completion token sets, so the run is recorded as
`failed_acceptance` and strict cross-deployment E2E comparability is not
claimed. Retained token-work savings were `44.6%` native and `59.0%` remote;
the remote arm ran `1.60x` native serial throughput with `0.75x` observed
first-token time, while the native router stayed about `2.46x` faster per
decision. Both authority pins are permanently closed with the recorded
result; a successor packet must reconcile multi-provider fallthrough with
completion matching.

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
