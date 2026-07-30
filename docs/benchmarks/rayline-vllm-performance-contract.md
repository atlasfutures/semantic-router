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
