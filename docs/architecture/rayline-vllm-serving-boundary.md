# Rayline vLLM Serving Boundary

Status: proposed for PL-0041. Pathfinder human endorsement is tracked by the
corresponding proposed architecture decision.

## Decision

Run the Rayline encoder in a dedicated vLLM service. Keep Pathfinder as the
policy, transaction, and authoritative episode-state owner. Keep Semantic
Router as the HTTP lifecycle and worker-dispatch owner.

The production-shaped boundary is separate-process, integrated deployment:

```text
                                        decision plane
                              +-----------------------------+
                              | Pathfinder                  |
                              | policy head                 |
                              | prepare / commit / settle   |
                              | authoritative episode state |
                              +--------------+--------------+
                                             |
                                             | strict pooling request
                                             v
+--------+     +-------+     +-----------------------------+
| client | --> | Envoy | --> | Semantic Router             |
+--------+     +-------+     | request lifecycle + dispatch|
                             +-------------+---------------+
                                           |
                                           | selected worker
                                           v
                            +--------------+---------------+
                            | worker vLLM A / worker vLLM B |
                            | or external provider          |
                            +-------------------------------+

                              +-----------------------------+
                              | dedicated Rayline vLLM      |
                              | tokenizer + causal MEAN     |
                              | scheduler + reconstructible |
                              | KV / pooling cache          |
                              +-----------------------------+
```

The Pathfinder-to-encoder arrow is intentionally independent of the
Semantic-Router-to-worker arrow. The encoder and workers have different model
identities, cache lifetimes, resource requirements, health contracts, and
scaling signals.

## Ownership

| Concern | Sole owner | Failure meaning |
| --- | --- | --- |
| Client HTTP and streaming lifecycle | Envoy and Semantic Router | Fail the client request closed |
| Worker allowlist and dispatch identity | Semantic Router request plus Pathfinder policy contract | Reject an unknown or ambiguous worker |
| Prepare, renew, commit, abort, and settle | Pathfinder | Preserve or reject the authoritative transition |
| Current conversation history | Semantic Router request, forwarded through Pathfinder | Complete encode input; not persisted as routing state |
| Committed route version, previous worker, and bounded outcomes | Pathfinder | Authoritative policy state used alongside the current request |
| Canonical Rayline serialization | Frozen IO plugin and parity fixtures | Readiness or request rejection |
| Model forward and causal-MEAN pooling | Dedicated Rayline vLLM | Reconstructible encode failure |
| Cross-turn model KV and pooling accumulator | Dedicated Rayline vLLM | Cache miss followed by a full rebuild |
| C82 policy artifact and small policy head | Pathfinder | Readiness failure; never silently substitute |
| Worker generation KV | Each worker vLLM | Worker-local cache miss; never shared with Rayline |
| Provider credentials and response usage | Semantic Router | Credentials never cross into Pathfinder or Rayline vLLM |

Correctness must not depend on the Rayline vLLM cache. Semantic Router supplies
the complete current request history on every prepare, so Pathfinder can
reissue a full encode after an engine restart, eviction, replica-affinity miss,
or rejected cache entry without persisting prompt history.

## Stateless Bridge First

RSP-004 reuses the existing strict PL-0039 IO plugin and David's causal-MEAN
vLLM fork without changing cache semantics. Pathfinder forwards the complete
current request history received from Semantic Router through the existing
`rayline.arc.pooling-request.v1` plugin envelope:

```json
{
  "task": "plugin",
  "data": {
    "schema_version": "rayline.arc.pooling-request.v1",
    "serializer_version": "mtrouter-token-blocks-v2",
    "serving_rung": "B",
    "episode_id_hash": "<64 lowercase hex characters>",
    "turns": [
      {"role": "user", "text": "sanitized example"}
    ]
  }
}
```

The response must be the strict `ArcPoolingResponse` already checked by the
ARC client: one normalized 1024-dimensional embedding plus exact model,
tokenizer, serializer, engine, plugin, token-count, and pooling-capability
identity. Pathfinder feeds that embedding into the same C82 policy head used
by the local Transformers implementation.

The stateless bridge has these rules:

- Semantic Router supplies the complete current request history; Pathfinder
  combines it with authoritative routing state and constructs the canonical
  turn list; vLLM owns tokenization and pooling.
- The episode value on the wire is a one-way opaque digest, never a user
  episode identifier.
- Every request has a bounded deadline and an unambiguous request ID.
- A timeout or identity mismatch aborts the selection transaction. It does not
  fall through to an arbitrary worker.
- Readiness proves the exact model, tokenizer, serializer, plugin, causal-MEAN
  capability, precision, context length, and vLLM build before traffic.
- Full-history parity is established before a cross-request cache is enabled.
- The v1 contract continues to require zero cached prefix tokens. RSP-005 must
  introduce a new capability/version rather than weakening v1 checks in place.

## Cross-Episode Concurrency

The selection-transaction journal already fences a second prepare for the same
episode and releases its lock while different episodes select. The immediate
transactional-path limiter was Pathfinder's process-wide
`RouterService._policy_select_lock`, which wrapped the policy call made by
`/v1/route/prepare`.

RSP-004A adds an explicit per-policy concurrency capability.
Immutable MTRouter selection through the remote encoder may overlap across
different episodes; same-episode prepares remain fenced and mutable policies
remain serialized. Deterministic blocking tests prove the seam, and PERF009
proves the real transaction path reaches Pathfinder in-flight `8`, encoder
in-flight `7`, and vLLM scheduled batch width `6`.

## Cross-Turn Cache Contract

David's current vLLM change accumulates causal MEAN across scheduler chunks
within one request. Pathfinder's current `KVEncodeSession` instead retains
`past_key_values` and an FP32 hidden-state sum across requests. Those are
different cache boundaries.

RSP-005 must prototype both viable vLLM designs:

1. Extend automatic prefix caching so a block hit restores the causal-MEAN
   sum and count at the same matched boundary as model KV.
2. Add an explicit, bounded vLLM session whose engine-owned KV, FP32 sum/count,
   prefix identity, and lifecycle survive between requests.

Either prototype must satisfy all of the following before selection:

- The cache key binds engine incarnation, model, tokenizer, serializer,
  artifact, and canonical token-prefix identity.
- Same-episode mutation is fenced; different episodes remain batchable.
- A hit restores both model state and pooling state. KV without the
  corresponding sum/count is not a hit.
- Prefix replacement, truncation, hybrid-cache rewind, sub-chunk input, and
  incompatible identity decline to an exact full encode.
- Residency has one engine-global enforceable bound. Whole-session or
  whole-block eviction is deterministic and observable.
- Cache telemetry contains counts, reasons, and opaque identities only.
- A miss can increase latency but cannot change the selected worker outside
  the frozen parity tolerance.

The prototype decision is intentionally not made in this document. RSP-005
must compare correctness, batching, memory efficiency, operational complexity,
and latency with measured evidence.

## Deployment Shapes

### Default: separate services

Pathfinder and Rayline vLLM have independent Deployments, health checks,
resource requests, rollouts, and replica counts. Cache-aware affinity is a
performance optimization. A non-affine request rebuilds from Pathfinder state.

This shape is the qualification baseline because it preserves independent
scaling and failure domains and makes GPU resource use visible.

### Allowed experiment: same Pod, separate containers

A Pathfinder container and Rayline vLLM sidecar may communicate over localhost.
This removes most network variance but forces 1:1 rollout and scaling, and can
duplicate encoder weights when Pathfinder is replicated. It may win at one
replica but is not the default without measurement and a new decision.

### Rejected default: embedded `AsyncLLM`

Embedding vLLM inside Pathfinder removes an HTTP hop, but couples API health,
Python dependency resolution, GPU engine lifecycle, autoscaling, and rollout.
A CUDA or engine failure would also remove the policy/state endpoint. It
remains a benchmark variant, not the production boundary.

### Rejected default: current in-process Transformers

The current implementation proves exact cross-turn KV behavior and remains the
numeric oracle and rollback path during qualification. It does not provide
vLLM continuous batching, scheduler telemetry, standardized model serving, or
an independently scalable encoder plane.

### Rejected: one engine for encoder and worker generation

Rayline pooling and downstream generation do not share one vLLM engine. They
use different models and scheduling contracts; worker generation must not
evict decision-plane state or make routing queue behind long completions.

## Failure and Rollout Rules

- Start with one Pathfinder replica and one dedicated Rayline vLLM replica.
- Drain or fail new prepares before replacing Pathfinder. Keep TD046 open
  until pending transactions are durable across replicas.
- Rayline vLLM restart invalidates acceleration state only. Pathfinder detects
  a new engine incarnation and rebuilds.
- A selected worker is committed only after the existing first-2xx-headers
  boundary. Encoder success alone never commits dispatch.
- Roll back the encoder backend from remote vLLM to local Transformers without
  migrating committed episode state.
- Do not enable a vLLM cross-turn cache until stateless full-history parity,
  readiness identity, bounded timeout, and failure tests pass.
- Qualification thresholds and immutable test identities live in
  `docs/benchmarks/rayline-vllm-performance-contract.md`.
