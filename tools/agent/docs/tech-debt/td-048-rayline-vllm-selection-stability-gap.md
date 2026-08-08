# TD048: Rayline vLLM Selection Stability Gap

## Status

Open — the MVP smoke passes; production quality/regret and full-corpus
qualification remain incomplete.

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

The execution-alignment MVP closed the first six-case smoke. A Torch-reference
GDN path plus Triton attention reduced the original four flips to one
same-model thinking-mode tie. Applying a global cheap-default margin at `0.002`
to both local and remote contracts changed one local decision, changed zero
remote decisions, and produced a strict zero-flip receipt, but later quality
evidence rejects that global contract.

The first task-disjoint preflight rejects the original precedence. Applied
after the previous-worker stay margin, the guard changed 40/524 canonical dev
decisions (`7.63%`) and increased switches from 14 to 30, failing its behavior
gate; no change was at route 0, so it supplied no same-initial-state quality
evidence. Applying the same threshold before stay resolution changed 0/524 and
preserved 14 switches. That candidate is behaviorally safe on this replay but
explicitly `insufficient_power`, not a quality pass.

A targeted 178-state C9 route-0 screen found the global rule crossed model
families four times. Three scorable changes had mean reward delta `-0.1667` and
worst task delta `-0.5`; one unscorable change failed closed. The global rule
is retired, regardless of whether it runs before or after stay resolution.

The replacement is a `0.0005` tie-break restricted to thinking-on/off arms of
the same Flash base model. It made zero cross-model changes and was inert on
both the 178-state screen and all 524 historical decisions, preserving 14
switches. Its dedicated six-case recanary passes all execution-parity gates.
The remaining debt is a powered changed-action quality/regret result for this
narrow rule and the held full 1,000-decision RSP-004Q qualification.

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
- The winning paired local/remote smoke is pinned at
  `rayline-ai/router-artifacts@306ca8c40470820f36d3decb5bfd9414552b5b7a`.
  All eight receipt gates pass: 0/6 selection flips, exact token-count and
  contract identity, minimum embedding cosine `0.9999849695`, and maximum
  top-two gap drift `0.0011914223`.
- The final vLLM implementation is
  [`atlasfutures/vllm@6ef6e844`](https://github.com/atlasfutures/vllm/commit/6ef6e84425d4493566a95ffcdfcb79f3c27abc46);
  the complete Pathfinder experiment ledger is
  [`atlasfutures/pathfinder@05c4f1df`](https://github.com/atlasfutures/pathfinder/commit/05c4f1df7e1654897fec291e338426b810b1af98).
- Measured infrastructure spend was `$1.1961`, or `$2.1961` including the
  conservative `$1` preflight/preemption reserve, under the `$20` cap. All
  fourteen Modal apps were verified stopped with zero tasks.
- The failed post-stay replay is privately pinned at
  `rayline-ai/router-artifacts@b947be95f9181058270b572d285c7efde5b5b074`;
  the behaviorally clean but underpowered pre-stay replay is pinned at
  `rayline-ai/router-artifacts@e7f862ede913559a4985b8354296b580ab1f919d`.
  Both use 60 manifest-authoritative dev attempts and make zero provider or
  Modal GPU calls. Pathfinder records the evaluator, explicit compatibility
  stage, and results at
  [`ce661e5f`](https://github.com/atlasfutures/pathfinder/commit/ce661e5ffe62301dcad307b9bc4b242324019497).
- The explicit pre-stay local/remote recanary is privately pinned at
  `rayline-ai/router-artifacts@b82e0afc2da53e6268dc72ba13a23df7e863e9c0`.
  All eight receipt gates pass over six decisions: zero selection flips, exact
  token and contract identity, maximum top-two-gap drift `0.0011912137`, and
  minimum embedding cosine `0.9999849696`. The candidate runtime was 205.10
  seconds on an isolated L40S, reported `$0.12549` infrastructure, made zero
  provider calls, passed the prompt-log privacy scan, and the Modal app stopped
  with zero tasks. This closes the smoke criterion only; the replay remains
  underpowered for route-0 quality.
- The targeted route-0 screen and the 524-decision narrow-rule replay are
  privately pinned and exact-round-trip verified at
  `rayline-ai/router-artifacts@d4a2d67b10b0e435c70de10a320c2b0590d520e8`.
  The C9 screen contains 178 unique initial states outside C82's four source
  lineages; it is not claimed task-identity-disjoint from every C82 fit row.
  The global rule changed four decisions and is rejected; the narrow rule
  changed zero decisions on both screens and is therefore scope-compatible but
  underpowered.
- The narrow-rule local/remote recanary is privately pinned and exact-round-trip
  verified at
  `rayline-ai/router-artifacts@b707b2715018edaa269e08e16f1755491d79fd06`.
  It passed 6/6 decisions with zero flips, exact token counts,
  `0.0011912882` maximum gap drift, `0.9999849696` minimum embedding cosine,
  zero provider calls, and stopped cleanup. Observed infrastructure was
  `$0.155999`.
- Pathfinder
  [`63eead46`](https://github.com/atlasfutures/pathfinder/commit/63eead4666c7785ceaa02c913bb810ac85280f94)
  records the exact held RSP-004Q packet. The driver is source-frozen at
  `c7dde584`, the confirmation-gated launcher at `565c2afb`, and the cumulative
  conservative envelope is `$14.484864` against the `$20` cap. The launcher
  refuses paid execution without both its dedicated flag and exact confirmation
  token. Actual 1,000-decision arms launched: zero.

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
and the MVP receipt proves deterministic routing selection under the accepted
cross-engine numeric envelope. The global cheap-default rule is no longer a
candidate. The remaining desired state is evidence that the narrow same-model
thinking tie-break preserves task quality and acceptable regret when it
actually fires, plus a passing full qualification receipt.

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
