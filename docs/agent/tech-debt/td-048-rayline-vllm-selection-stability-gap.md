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

The same run exposed a distinct interface mismatch. That mismatch is now
remediated: local and remote encoders declare the same normalized FP32
contract, and receipt v2 refuses non-unit vectors. Offline replay shows that
explicit local pre-normalization changes no raw argmax decision, so scale was
a measurement defect rather than the cause of the four flips.

The remaining debt is narrower: small directional differences between the
Transformers and vLLM executions cross near-tie and previous-worker
stay-threshold boundaries.

## Evidence

- Pathfinder experiment
  `rayline-vllm-stateless-parity-rsp004-20260730` at
  [`5295fdb5`](https://github.com/atlasfutures/pathfinder/commit/5295fdb57adece07d1a62c0aa447143c0e9f3224)
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
- Pathfinder
  [`f08128bd`](https://github.com/atlasfutures/pathfinder/commit/f08128bdced1a71d5fb4f1ac6bf724f4936644ee)
  implements `l2-normalized-fp32.v1`, the fail-closed v2 receipt, regression
  coverage, and deterministic smoke derivation.
- The canonicalized frozen observations have maximum embedding absolute error
  `0.00112024` and minimum cosine similarity `0.9999729453`.
- Explicit pre-normalization of the local vectors changes C82 q-values by at
  most `3.5763e-7` and raw argmax on 0 of 1,000 decisions.
- RSP-004S is materialized with six decisions, 426,979 full-history tokens, all
  four historical flips, and one large-tool and one near-maximum-context case.
  Its sanitized inputs and diagnostic are privately pinned at
  `rayline-ai/router-artifacts@d73fae3a526ff4d350d462b93b453792099a08b9`;
  no GPU or provider spend was incurred.

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

Both encoder backends now return the same documented normalized-vector
contract, and parity receipts compare canonical embeddings plus policy
outputs. The remaining desired state is deterministic routing selection under
the accepted cross-engine numeric envelope, with any stability rule explicitly
specified and evaluated for quality rather than introduced by weakening a gate
after measurement.

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
