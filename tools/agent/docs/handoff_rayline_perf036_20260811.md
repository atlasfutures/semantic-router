# Handoff: PERF036 — the RTX PRO 6000 capacity run, executed and closed

Status as of 2026-08-11, 21:10Z. Read this before touching the rayline
capacity-measurement lane: `e2e/testing/rayline-arc/`, pl-0041, the pathfinder
experiment registry, or any GPU cost/capacity figure for the ARC encoder.

## TL;DR

- **PERF036 executed once and is fully closed.** Launch gate closed
  (`LAUNCHABLE_CONTRACT = None`, commit `d277497d`), registry complete
  (pathfinder `2642844d`), results in pl-0041 (`becaf9e8`), receipts on HF.
  All pushed. Nothing is launchable from this tree.
- **PROVEN**: measured ceiling 0.8877 dec/s (arc `r144`), inside the
  preregistered band 0.5490–1.0195 — cross-checked by the independent remote
  arm (0.8441) and by Little's-law arithmetic (8 ÷ 9.01 s = 0.888).
- **PROVEN**: zero Modal GPU containers remain — launcher cleanup plus an
  independent `modal container list` check.
- **NOT proven**: the encoder's own knee (the rig was occupancy-bound),
  burst absorption, multi-instance, and TFLOPS scaling beyond this one hop.
- **Next action**: none for an agent. Any further run needs a new registry ID
  and fresh human authorization (§10).

---

## 1. Why this thread exists

PERF035 measured the L4 and falsified the token-model calculator (missed by
2.03×). It left one prediction method standing: scale a *measured* ceiling by
the raw TFLOPS ratio. PERF036 promoted that survivor to a preregistered,
falsifiable prediction and pointed it at Cloud Run's other card, the RTX PRO
6000 96 GB. Same frozen packet, ladder rescaled ×2, `r032` kept byte-identical
to PERF035 as a cross-run anchor. The user approved the run ("i approve" /
"the rtx bench", under $10; envelope $5.6186208).

## 2. PROVEN, with evidence

### 2.1 The band held — first prospective cross-GPU hit

Prediction: 0.1977 (measured L4) × 480/121 TFLOPS = **0.7843** dec/s,
band ±30% = 0.5490–1.0195. Measured, from
`.agent-harness/rayline-parity/rayline-rtx6000-capacity-perf036-20260811/`:

```text
arc     r032 0.3766  r064 0.6082  r096 0.7472  r144 0.8877   (completion dec/s)
remote  r032 0.3744  r064 0.5772  r096 0.7208  r144 0.8441
occupancy: 0.375 / 0.875 / 1.0 / 1.0 (both arms; saturated from r096)
```

Cross-checks (three, independent):

1. The remote sub-arm is a separate client path and lands within 5% at every
   rung, always slower — the same arc-faster ordering as PERF033–035.
2. Little's law from measured residence: 8 ÷ 10.71 s = 0.747 (`r096`),
   8 ÷ 9.01 s = 0.888 (`r144`). Matches completion throughput to 3 figures.
3. The `r032` anchor's realized arrival rate equals PERF035's **to the last
   digit** (0.39722749207287017) — the load generator did the same thing on
   both cards.

### 2.2 The drain clause voided the r144 plateau — third time

Plateau fired at `r144` (gains 0.2359 / 0.2069 under the 0.3333 floor), but
drain-corrected gains are 0.9581 / 0.9558 (within 10% of 1.0) with negative
residence deltas. This is PERF032's finite-corpus signature; the packet's own
preregistered rule discards the plateau verdicts. So **0.8877 is the
eight-lane rig's floor under the encoder ceiling, not the encoder's number.**

### 2.3 Integrity and identity

All 8 cells 32/32, `failed=0`, provider calls 0; `comparison.json` status
"passed" in-run. Trace sha256 equals PERF020's digest
`d9e93cf0f4c636a3838e41938d2ef3ff6e1d66a60860922f84771b3fa5158ac9` in every
cell; ARC telemetry byte-identical across cells and to PERF035 (same corpus).
State resets closed 8/8 sessions with exact zeros. Cross-check: these are
in-artifact fields verified by the contract's 40 pytest tests, which run
against the frozen preregistration, not the run's own reporting.

### 2.4 Cleanup

The launcher stopped the app and deleted the proxy token; an independent
`modal container list` in the `dev` environment then showed zero containers.
Two instruments, both empty.

## 3. NOT proven / still open

- **The encoder's own saturation knee.** Never observed on any card. Every
  packet so far ended rig-bound (occupancy 1.0) with the plateau voided by
  the drain clause. Isolating it needs a wider corpus or more lanes.
- **Burst absorption.** The worst recorded production burst is 2.33 dec/s,
  which exceeds the 0.8877 eight-lane floor. Wider lane counts are unmeasured
  on this card. Do not claim the card absorbs bursts.
- **Multi-instance.** Blocked by session ownership (retained sessions in a
  process-local dict); `max_containers=1` is required. Parked as TD050.
- **TFLOPS scaling generality.** One prospective hit on one hop
  (Ada → Blackwell, GPU-bound workload). L40S/A100 figures remain token-model
  output and are unknown.
- **Identity docs' `gpu_class`.** They still read `NVIDIA H100 80GB` by
  frozen lineage. The actual card is attested only in
  `deployment-evidence.json` (`"encoder_gpu": "RTX-PRO-6000"`). A reader of
  the identity docs alone would name the wrong card.

## 4. UNKNOWN

- `warmup_sessions_missing: 1` in every cell. Same pattern as PERF035,
  harmless to the measurement (warmup only), but never root-caused.
- Why the measurement landed +13.2% above the point prediction. Inside the
  band, so not a falsification — but the sign and size are unexplained.

## 5. Corrections to earlier records

`docs/agent/handoff_rayline_serving_cost_20260810.md` had three stale claims;
a correction banner now sits at its top (same commit as this file):

1. "`902c4ab4` is NOT pushed" — long since pushed; branch is ahead 0.
2. "The vLLM encoder has never run on any GPU except H100" — falsified by
   PERF035 (L4) and PERF036 (RTX PRO 6000).
3. "FlashInfer has never been load-tested end-to-end" — PERF033–036 all ran
   the `gdn-flashinfer-eager` engine build under load.

## 6. Time-boxed facts

| Fact | Expires | Consequence when it does |
|---|---|---|
| Modal dashboard usage/billing view for this run | unknown retention | measured-cost upper bound ($0.9561) becomes unverifiable at the provider; the receipts on HF remain |

Nothing else known to be time-boxed. HF revisions are permanent pins; R2 has
no known TTL; 1Password-held tokens have no known expiry.

## 7. Where things live

- **semantic-router**: branch `codex/rayline-remote-mvp`, HEAD `becaf9e8`,
  pushed (ahead 0). Chain: `fee6176e` authorize → `fcc1fc7d` bind →
  `d277497d` close gate → `becaf9e8` pl-0041 results.
- **pathfinder**: worktree `/Users/chilang/code/pathfinder-rayline-vsr-mvp`,
  branch `codex/rayline-vsr-mvp`, registry completion `2642844d`, pushed.
  Never touch `~/code/pathfinder` itself — it holds the user's uncommitted
  work.
- **Durable receipts**: HF dataset `rayline-ai/router-artifacts`,
  `runs/rayline-rtx6000-capacity-perf036-20260811`, revision `7d62eae9`
  (19 files); aborted-attempt evidence at `…-aborted-docker-down`, revision
  `44483196`.
- **Results page**: R2
  `memex-pr-assets/shared/17613c84-f5fc-4d2f-944c-6fa1271926fb/rayline-arc-measurement-results-20260811.html`
  — section 05C added, verified byte-identical after upload.
- **Disk-only**: `.agent-harness/rayline-parity/rayline-rtx6000-capacity-perf036-20260811{,-aborted-docker-down}`
  (88K + 8K). Redundant copies — the same files are on HF at the pinned
  revisions above. Safe to lose.
- **Worktrees**: `.claude/worktrees/perf034-cap-raise` (this lane, at HEAD);
  `.claude/worktrees/rayline-reconcile` (`e2cc0e17`, a *different* lane —
  untouched by this session, do not clean up unexamined).

## 8. Exact resume commands

Contract tests (ran this session, 40 passed):

```bash
cd /Users/chilang/code/semantic-router/src/vllm-plugins/rayline_arc_io
~/.local/bin/uv run --extra test --with pyyaml python -m pytest \
  tests/test_rayline_rtx6000_capacity_contract.py \
  tests/test_rayline_l4_capacity_contract.py \
  tests/test_modal_session_service.py -q
```

Registry validator (ran this session, "OK: … 586 experiments"):

```bash
cd /Users/chilang/code/pathfinder-rayline-vsr-mvp
unset UV_LOCKED
~/.local/bin/uv run --frozen --extra test python scripts/validate_experiment_registry.py
```

Fetch the live results page (ran this session; `curl` on the public host
failed DNS from this network — use rclone):

```bash
unset HTTPS_PROXY https_proxy
rclone copyto r2-pr-assets:memex-pr-assets/shared/17613c84-f5fc-4d2f-944c-6fa1271926fb/rayline-arc-measurement-results-20260811.html /tmp/results.html
```

Modal commands (the session's working pattern; `modal container list` was run
with it and returned empty — the pattern is required because proxy/color env
vars break the Modal CLI):

```bash
unset HTTPS_PROXY https_proxy FORCE_COLOR CLICOLOR_FORCE
export MODAL_ENVIRONMENT=dev
modal container list
```

YAML gotcha that bit twice: plain scalars in the registry must not contain
`": "` (colon-space) — the validator rejects them; use `" -- "` instead.

## 9. Cost

- This run: measured upper bound **$0.9561** of the $5.6186208 envelope; paid
  window 878.02 s of 2400 s. Provider (Anthropic) spend $0.
- Aborted first attempt: 137.2 s of idle deploy time before the local Docker
  daemon was found down; no cell ran.
- Cumulative conservative total: **$194.459850266383**, exactly $3.00 of
  reserve under the $197.459850266383 authority.
- Resuming this handoff spends nothing. Any new measurement spends GPU money
  and needs fresh authorization first.

## 10. Open decisions for the human

1. **Fund a knee-isolation run?** Three packets in a row ended rig-bound with
   the plateau voided. Finding the encoder's true ceiling needs a wider
   corpus or more lanes, a new registry ID, and new money — and only $3.00
   remains under the current authority, so it also needs a raised authority.
   Recommendation: only if a deployment decision actually depends on the
   ceiling; the floor (3.95× demand single-instance) already answers the
   sizing question asked.
2. **Measure burst absorption before committing to single-instance Cloud
   Run?** The 2.33 dec/s worst burst exceeds the measured eight-lane floor.
   Recommendation: yes, before any production commitment — or accept queueing
   during bursts explicitly.
3. **Unblock multi-instance (TD050)?** An architecture decision (session
   ownership), not a measurement. No recommendation from this thread.
