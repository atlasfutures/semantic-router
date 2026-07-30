# TD048: Rayline vLLM Selection Stability Gap

## Status

Open

## Owner Plan

[PL0041 Rayline vLLM Serving and Performance Qualification](../plans/pl-0041-rayline-vllm-serving-performance.md)

## Release Relevance

The stateless vLLM encoder cannot be promoted to the Rayline decision path
while the same canonical history can select a different worker than the local
Transformers reference.

## Scope

- Pathfinder's local and remote MTRouter encoder result contract
- Rayline ARC Rung B normalization behavior
- parity comparator embedding semantics
- policy decision stability at near-tie and previous-worker stay boundaries
- smoke-versus-qualification corpus sizing

Cross-request KV ownership and throughput are out of scope until this gap is
closed.

## Summary

The first pinned RSP-004 L40S qualification completed 1,000 decisions through
both local Transformers and stateless causal-MEAN vLLM. Seven of eight hard
gates passed, but four worker selections flipped, so the preregistered
zero-flip gate failed.

The same run exposed a distinct interface mismatch. The local encoder returned
the raw FP32 masked mean with vector norms from approximately 109 to 130.
Rung B returned an L2-normalized vector. C82 has
`normalize_embeddings: true`, so both inputs were normalized inside the policy
head and the score comparison remains valid, but the receipt's raw absolute
embedding-error metric compares incompatible scales.

## Evidence

- Pathfinder experiment
  `rayline-vllm-stateless-parity-rsp004-20260730` at
  [`5295fdb5`](https://github.com/atlasfutures/pathfinder/commit/5295fdb51ae0553a15ad4d6ed2dbf9cf3dc71581)
  records the immutable receipt and private artifact revision.
- All 1,000 token counts matched exactly and maximum adjusted top-two gap drift
  was `0.003936`, below the frozen `0.005` limit.
- Decisions `parity-000038` and `parity-000486` reversed near argmax ties.
- Decisions `parity-000286` and `parity-000669` crossed the configured `0.05`
  previous-worker stay threshold by less than `0.001`.
- `LocalMTRouterEncoder` delegates to `LongContextHistoryEmbedder`, whose
  masked-mean path returns an unnormalized FP32 mean.
- Rayline ARC Rung B requires `embed`/`MEAN` with activation and the remote
  client rejects vectors whose norm differs from one.
- The C82 remote-backend identity check requires
  `normalize_embeddings: true`, and the estimator normalizes again before its
  policy head.

## Why It Matters

Worker selection is the semantic output of the router. Small numeric drift is
expected across inference engines, but silently accepting boundary flips would
change cost, quality, and thinking-mode behavior. Conversely, demanding raw
elementwise equality between differently scaled interface outputs obscures the
actual problem.

The first loop also processed 41.2 million full-history tokens per arm and took
71.6 minutes on the local baseline. Using that qualification corpus as the
implementation feedback loop makes correction unnecessarily slow and costly.

## Desired End State

Both encoder backends return the same documented normalized-vector contract,
and parity receipts compare canonical embeddings plus policy outputs. Routing
selection is deterministic under the accepted cross-engine numeric envelope,
with any stability rule explicitly specified and evaluated for quality rather
than introduced by weakening a gate after measurement.

A small boundary-heavy smoke corpus provides fast implementation feedback. The
full 1,000-decision corpus remains the final qualification.

## Exit Criteria

- Local and remote encoder results both satisfy an explicit normalized-vector
  invariant, with focused tests covering full, fallback, and remote paths.
- The comparator refuses incompatible embedding contracts and reports
  meaningful cosine and canonical elementwise error.
- An offline replay isolates normalization, execution-kernel, and policy
  threshold effects for the four observed flips.
- Any proposed tie or stay-boundary stability rule has a frozen definition and
  passes an offline quality/regret gate before policy behavior changes.
- The small RSP-004S live smoke passes with zero selection flips, exact token
  counts, and top-two gap drift at or below `0.005`.
- The full RSP-004Q qualification passes the same gates.
- The evidence is linked from PL0041 and this debt entry is deleted.
