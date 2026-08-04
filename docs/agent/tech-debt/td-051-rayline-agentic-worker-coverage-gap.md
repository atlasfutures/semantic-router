# TD051: Rayline Agentic Prompts Do Not Naturally Cover Every Worker

## Status

Open — endpoint coverage is explicit, but the current public semantic prompt
distribution has no natural MiMo/worker-b share.

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

The frozen 24-case agentic discovery set selects DS4 Flash and HY3 but has not
selected MiMo V2.5. The retained/replay KV packet is narrower still: its one
realistic history shape selected DS4 Flash for every completed request. A new
workload contract therefore separates the natural semantic-cache lane from an
explicitly stratified three-worker serving lane.

## Evidence

- AGT011 observed a `16/0/8` natural discovery mix across worker-a, worker-b,
  and worker-c. Its static preflight still reached all three endpoints.
- AGT017's completed FlashInfer arm selected worker-a for all 12 requests; its
  MiMo and HY3 endpoints remained configured but unobserved in that lane.
- `openrouter_kv_cache_workload_contract.py` marks stratified static coverage
  as inadmissible for semantic-selection claims.

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

- Offline discovery proves every configured worker has the preregistered
  minimum natural share on the exact pinned artifact and encoder revision.
- The prompt suite remains realistic and public; it does not use forced model
  IDs, controlled embeddings, routing-only headers, or model-specific anchors.
- Native and remote Rayline produce the same selected-worker trace.
- The real-provider report labels semantic and static coverage separately.
