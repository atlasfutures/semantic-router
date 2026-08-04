# TD051: Rayline Agentic Prompts Do Not Naturally Cover Every Worker

## Status

Open — AGT018 now has exact native offline three-worker coverage and a
source-closed vLLM parity gate, but the vLLM-hosted encoder has not yet executed
the frozen trace.

## Owner Plan

[PL0041 Rayline vLLM Serving and Performance Qualification](../plans/pl-0041-rayline-vllm-serving-performance.md)

## Release Relevance

Real-provider performance claims must distinguish semantic selection coverage
from static endpoint coverage. Conflating them would make a three-model serving
probe look like evidence about the learned router's natural traffic mix.

## Scope

- public synthetic agentic cases used by the OpenRouter qualification;
- natural Rayline selection across worker-a, worker-b, and worker-c; and
- offline evidence required before a real-provider packet claims semantic
  coverage of all three workers.

Provider availability and explicit static dispatch are not blocked by this
entry. They remain valid serving checks when labeled as such.

## Summary

The historical 24-case agentic discovery set selected DS4 Flash and HY3 but no
MiMo V2.5, while AGT017's one retained/replay history selected only DS4 Flash.
AGT018 replaces that narrow workload with three public, model-agnostic growing
agentic histories. Exact native encoder evaluation now selects worker-c for
the code sequence, worker-a for research, and worker-b for incident/source
correlation across all three KV states. TD051 stays open until the pinned
vLLM-hosted encoder reproduces the same trace before routed provider
measurement.

## Evidence

- AGT011 observed a `16/0/8` natural discovery mix across worker-a, worker-b,
  and worker-c. Its static preflight still reached all three endpoints.
- AGT017's completed FlashInfer arm selected worker-a for all 12 requests; its
  MiMo and HY3 endpoints remained configured but unobserved in that lane.
- The pinned native Metal runtime, manifest SHA-256
  `05e1a23105ec9d537d6cc5b1da7a06b01c7536b6c773d119d967d397bb95e043`,
  reproduced natural traces `C/C/C`, `A/A/A`, and `B/B/B` for the three AGT018
  histories. Serialized lengths range from `8,194` to `16,204` tokens; every
  first state is a prefill and every append is a delta with an `8,192`-token
  retained prefix. The minimum head top-two score gap is
  `0.0019787615092044693`, above the frozen `0.0015` gate.
- `openrouter_kv_cache_workload_contract.py` requires natural three-worker
  coverage for AGT018 while keeping stratified static dispatch inadmissible for
  semantic-selection claims.
- The AGT018 remote launch path now evaluates those same nine states directly
  against the protected vLLM encoder before any routed provider measurement,
  enforces the score-margin and retained-prefix contracts, and feeds aggregate-
  only evidence into the v3 reporter. The launch remains source-closed, so this
  is implementation coverage rather than an executed cross-architecture proof.

## Why It Matters

Provider latency and availability vary materially by model. A workload that
only naturally reaches one or two workers cannot characterize the complete
semantic router distribution, while forcing a worker bypasses the classifier
whose behavior is under study.

## Desired End State

A privacy-safe, public, realistic prompt suite naturally and reproducibly
selects every configured worker under both native and vLLM-hosted Rayline,
with a frozen minimum share and cross-architecture selection parity. Static
dispatch remains a separate serving control.

## Exit Criteria

- [x] Offline native discovery proves every configured worker has the frozen
  minimum margin on the exact pinned artifact and encoder revision.
- [x] The prompt suite remains realistic and public; it does not use forced
  model IDs, controlled embeddings, routing-only headers, or model-specific
  anchors.
- Native and remote Rayline produce the same selected-worker trace.
- The real-provider report labels semantic and static coverage separately.
