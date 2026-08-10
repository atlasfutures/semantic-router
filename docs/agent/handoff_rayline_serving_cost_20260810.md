# Handoff: Rayline ARC encoder serving cost — model, findings, and one code change

Status as of 2026-08-10, 14:10Z. Read this before touching
`src/vllm-plugins/rayline_arc_io/modal_session_service.py`, the Modal
deployment, or any cost figure quoted for the Rayline ARC encoder.

## TL;DR

- **One commit exists and is NOT pushed**: `902c4ab4` on
  `codex/rayline-remote-mvp` removes the `region="us-east"` Modal pin.
  Branch is ahead 1 of `atlasfutures/codex/rayline-remote-mvp`.
- **PROVEN**: the KV memory model (12,288 B/token, 18.63 MiB/session) derives
  exactly from the pinned model config. The capacity model reproduces the
  measured saturation knee. Production traffic is effectively 24/7. The region
  pin was measurably *slower*, by this repo's own benchmarks.
- **NOT proven**: the unpinned deployment has never been deployed or measured —
  this change is code-only. FlashInfer has never been load-tested end-to-end.
  The vLLM encoder has never run on any GPU except H100.
- **Next action**: decide whether to push `902c4ab4` (see §10.1). Nothing is
  blocked on an agent.

---

## 1. Why this thread exists

The original ask was to estimate monthly cost of serving traffic like
memex-desktop's on H100/L4 and to build a growth calculator. It re-scoped twice,
and a reader who knows only the original ask will rebuild work already discarded:

1. **The served model is not a generation LLM.** It is the Rayline ARC routing
   encoder — a Qwen3.5-0.8B pooling model under vLLM emitting one 1024-d
   embedding per user turn and **zero output tokens**, plus a Go Q-network head
   that runs on CPU. The first calculator modelled a 30B generation model and is
   superseded (§5).
2. **The Modal cost lane was wrong**, not Modal's prices. That produced the one
   code change in this handoff.

## 2. PROVEN, with evidence

### 2.1 KV memory model — derived exactly from the pinned config

The pinned revision's config is cached locally at
`~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17/config.json`:
24 layers = 18 `linear_attention` + 6 `full_attention`; `num_key_value_heads` 2;
`head_dim` 256; `mamba_ssm_dtype` **float32**; dtype bfloat16.

```
KV_BYTES_PER_TOKEN = 2 (K,V) x 6 full-attn x 2 kv x 256 dim x 2 B  = 12,288 B/token
GDN conv  = (6144, 3) bf16                                          =     36,864 B
GDN ssm   = (16,128,128) fp32                                       =  1,048,576 B
per layer = 1,085,440 B  x 18 linear layers = 19,537,920 B          = 18.6328125 MiB
```

**Cross-check**: these constants independently reproduce the per-session table
in `~/.agent/diagrams/rayline-router-gpu-cost-and-scale.html` (2026-08-03) to
four significant figures — 262,144 tokens gives 3.018 GiB there and 3.019 GiB
here. Two authors, two derivations, same numbers.

**Load-bearing detail**: the fp32 SSM state is not an assumption. vLLM's
`Qwen3_5ForConditionalGenerationConfig.verify_and_update_config` copies the HF
config's `mamba_ssm_dtype="float32"` into `cache_config.mamba_ssm_cache_dtype`
because the deployment leaves it `"auto"`. At bf16 the answer would be 9.63 MiB
and every downstream figure moves.

### 2.2 Capacity model reproduces the measured saturation knee

```
appended = 42,000 x (1 - 0.2114)                     = 33,121 tokens
gpu_bound_dps       = 11,117 / 33,121                = 0.335 dec/s
transport_bound_dps = 1.055 / (0.637 + 2.979)        = 0.291 dec/s
```

Both sit inside the MEASURED PERF021 knee of **0.1862–0.3724 dec/s per H100**
(`.agent-harness/rayline-parity/rayline-open-loop-sweep-perf021-20260802/comparison.json`,
`realized_arrival_rate_rps`, with `overloaded` false at r015 and true at r030).

**Cross-check, on a second GPU and a second code path**: scaling the H100
FlashInfer figure by dense-BF16 TFLOPS predicts an L40S at
`115,764 x (362/989) = 42,373 tok/s`. AGT014 measured **43,600 tok/s** mean
embedding throughput on a real L40S (`deployment.gpu = "L40S"`, native path,
4,752 mean prompt tokens) — a **−2.8%** miss from a prediction made without
reference to it.

### 2.3 Production traffic is effectively 24/7

720 hours, 2026-07-11 → 2026-08-09, `dbt_prod_marts.model_usage_hourly` joined
to a generated hour spine (**the table stores only non-zero hours** — querying it
without a spine silently drops idle hours and was the trap here):

| Metric | Value |
|---|---|
| Diurnal trough → peak | 9.44 → 17.68 tpm = **1.87x** |
| Hours with 0 turns | **0 of 720** |
| Hours below 1 tpm | **0 of 720** |
| Quietest hour of the month | 94 turns = 1.57 tpm |
| Inter-arrival p50 / p99 / max | 3.0s / 33.0s / 1,209s |
| Gaps > 300s (scaledown window) | **48 in 30 days** |

**Cross-check**: two independent pipelines agree — the turn-level mart gives
1.641M cache-read TPM against the LLM proxy's own hourly counters at 1.682M
(within 3%); and the gap query returned 584,176 gaps against 584,177 turns,
which is the arithmetic identity you'd expect if neither query dropped rows.

**Consequence**: scale-to-zero would reclaim `Σ(gap − 300)` = 7,034s = **0.27%
of the month**, spend ~1.2h of that back on cold starts, and stall 48 real user
turns by 78.9–96.9s each. Measured duty is **99.73%**.

### 2.4 The region pin was unjustified and measurably slower

From this repo's own records, not an external source:

- `docs/architecture/rayline-vllm-serving-boundary.md:180-186` — *"PERF011 does
  not justify tighter colocation on latency or throughput alone. Pinning both
  processes to Modal `us-east` produced `1.042x` the PERF009 prepare p50 and
  `0.994x` its throughput; neither preregistered strong-placement gate passed."*
- `pl-0041:1650-1661` — PERF014, the explicitly pinned run, was the **slowest of
  three arms**: 8.753 req/s vs 10.263 (PERF009) and 10.199 (PERF011); p50 0.950s
  vs 0.752s. *"the evidence does not justify colocation."*
- The pin arrived in commit `50941e8f` with an **empty commit body** and no code
  comment; `git blame` shows the prose calling it a *"controlled comparison"* was
  written **22 minutes later**.

Cost effect: Modal's narrow-region multiplier is ×1.75 applied to GPU+CPU+RAM,
so unpinning takes the encoder from **$8.466/hr to $4.838/hr per H100** at the
deployed 8-core/64 GiB request (−42.9%).

### 2.5 The code change is green, with a working negative control

`902c4ab4` — 11 insertions, 4 deletions across two files.

```
6 passed in 0.01s          # tests/test_modal_session_service.py
261 passed, 3 failed       # full package suite
```

**Cross-check (the negative control that matters)**: I stashed the change and
re-ran the full suite on the clean tree. The **identical 3 tests failed**, same
names. They are pre-existing and unrelated:
`test_encoder_identity_is_dynamic_but_timeouts_remain_typed`,
`test_agentic_preflight_returns_bounded_provider_failure`,
`test_launcher_stage_packet_is_closed_and_reuses_agentic_sources`.
Ruff likewise reports the same 9 pre-existing `RUF100` findings before and after,
shifted only by the 5 comment lines added.

## 3. NOT proven / still open

- **The unpinned deployment has never been deployed or run.** This is a
  code-only change. No measurement exists of the encoder running unpinned.
  Placement is now non-deterministic — Modal schedules wherever it has capacity,
  and the client policy process is in London. The evidence says region distance
  is not what drives the 0.637s floor, but that is an inference, not a
  measurement of this configuration.
- **FlashInfer has never been load-tested end-to-end.** Its 9.8586x speedup is
  measured on 12–36 **strictly serial** pooling calls. Every saturation sweep
  (PERF015–027) ran on `torch_reference`. The FlashInfer knee is unmeasured; the
  model *predicts* ~3.9x, bounded by the backend-independent transport floor,
  but that is a structural property of the model, not a result.
- **The vLLM encoder has never run on any GPU except H100.** All L40S / L4 /
  A100 / RTX PRO 6000 throughput in the calculator is TFLOPS-scaled, ±30%. The
  one real L40S datapoint (AGT014) is the **native** path, not the vLLM encoder.
- **The 544-token attention block size is derived from vLLM source, never
  observed.** No vLLM startup log is captured anywhere: `.agent-harness/` holds
  250 `.json` + 28 `.jsonl` and **zero** `.log`/`.txt`.
- **Cold start has only ever been captured as one opaque wall-clock number**
  (78.9s AGT005; 96.892s SQP001 against warm p50 0.841s). The
  container-start / weight-load / engine-init breakdown does not exist, because
  every benchmark launcher overrides the deployed class to `min_containers=1`
  before health-checking and every receipt is stamped `"warm_state": "warm"`.
- **No Modal invoice or billing-report API output was ever consulted.** Every
  Modal cost figure here is list-price arithmetic. The billing-report API (GA
  since v1.3.3) would give ground truth and was not used.
- **The 1,000-case qualification has never been executed**, and every measured
  arm FAILED the contract's absolute SLO gates (8.0 rps floor, 1.0s p95 ceiling).
  The contract targets 8 decisions/sec against a measured knee of 0.3724 — a
  ~21x miss.
- **`cpu=8.0` / `memory=65_536` are unjustified but NOT changed.** Zero host CPU
  or RAM measurement exists in the repo; the only monitor samples four GPU
  fields via `nvidia-smi`. Right-sizing to 4c/16GiB would break no test
  (the freeze test pins the keyword *names*, never `literal_eval`s the values).

## 4. UNKNOWN

- **Why the ~0.637s transport floor exists.** It is backend-independent
  (0.636815s FlashInfer, 0.643784s torch_reference — within 1%), engine queue
  time is ~0.00002s, and placement work explicitly ruled out scheduler queueing
  *and* simple region distance. The cause is not identified anywhere.
- **Whether 8 cores / 64 GiB is actually needed.** Not measured either way.
  Recorded as unknown, not as "over-provisioned" — the sibling generation
  workers run full vLLM on 4c/16GiB, which is suggestive but not evidence about
  *this* container.
- **Modal's minimum billable duration per container start**, and whether the
  boot/init phase is billed. Not documented; not inferred.
- **GCP H100 spot $/GPU-hr.** Google's own pricing pages would not render for
  automated fetch across repeated attempts; third-party trackers disagree by 3x
  ($1.21–$6.92 depending on region and source). Treated as UNRESOLVED in the
  calculator rather than picked.
- **What emptied `.agent-harness/rayline-kv-cache`.** The container directory
  has mtime 2026-08-08 12:30 with no child born that day — something was deleted
  or moved out, leaving no artifact.

## 5. Corrections to earlier records

Sources below are **NOT yet fixed** unless stated. Readers follow whichever they
find first, so these matter.

| Record | Correction | Fixed? |
|---|---|---|
| `~/.agent/diagrams/memex-gpu-serving-cost-calculator.html` (this session, earlier) | Models a 30B **generation** model — the wrong workload entirely. Superseded by `rayline-arc-encoder-serving-cost.html`. | Superseded, not deleted |
| `modal_canary_runtime.py:115-122` | Cost model has **no region multiplier term**. While the service was pinned, every `.agent-harness/` budget receipt understated spend by ~75%, and `COST_CEILING_USD` gates were checked against a rate that did not exist. Now moot for future runs (pin removed) but historical receipts remain wrong. | **NOT fixed** |
| `docs/benchmarks/rayline-vllm-performance-contract.md` | A full GDN generation stale: zero hits for FlashInfer/GDN/PERF02/PERF03, still `rayline-vllm-perf.v1` frozen 2026-07-30, still pins baseline `davidvgilmore/vllm@162bcefe` against deployed `atlasfutures/vllm@9f5ea81c+gdn-flashinfer-eager`, still names L40S primary though every run since PERF015 is H100. Its own rule requires a new contract version before a measured run. | **NOT fixed** |
| `docs/agent/tech-debt/README.md:86` | Still lists TD051 as open; TD051 is closed in its own body by commit `005670ff`. | **NOT fixed** |
| `~/.agent/diagrams/rayline-router-gpu-cost-and-scale.html` (2026-08-03) | Its `$8.47/hr` H100 figure assumes the us-east pin, which no longer exists in code. Its KV model is correct and was vindicated. | **NOT fixed** |

**My own errors during this session**, corrected in the current calculator but
possibly quoted elsewhere:

- `11,088 tok/s` for torch_reference mixed measurement stages
  (`backend_mean_seconds` vs `engine_inference_mean_seconds`), inflating the
  implied speedup to 10.44x. Correct engine-inference pair is **11,117 /
  115,764**; the frozen gate is **9.8586x**.
- A **×3.0 non-preemptible multiplier** was modelled. It does not exist for GPU
  functions — *"The `nonpreemptible` parameter is not supported for GPU
  Functions"* — and where it applies at all it multiplies CPU/memory only.
- The **~50.5s first-shape Torch compile is native-path only**, not the vLLM
  remote path (`enforce_eager=True`, plus a persistent `/root/.cache/vllm`
  Volume). Do not include it in a remote cold-start budget.
- **"71.27s" appears nowhere in the repository.** I repeated it; it is
  unsourced. Do not cite it.

## 6. Time-boxed facts

| Fact | Expires | Consequence when it does |
|---|---|---|
| BigQuery traffic pull is a rolling 30-day window ending 2026-08-09 | Continuously | Re-running the SQL later returns different numbers; the 1.87x diurnal ratio and 48-gap count are as-of, not stable constants |
| `.agent-harness/` receipts, **gitignored** | On disk loss / clean checkout | **Every measurement in §2 becomes unreproducible.** But see §7 — the irreplaceable part is only a few MB of JSON and is cheap to archive today. Off-repo copies are referenced as `rayline-ai/router-artifacts@<sha>`, which was not accessible from this session |
| R2 share links (§7) | No stated TTL; frozen at upload | They are **static snapshots** — editing the local HTML does not update them. A reader may be looking at a superseded model |
| Claude artifact URL (§7) | No stated TTL | Updates in place on republish, so it tracks the current file |
| Sonnet 5 API pricing changes 2026-09-01 ($2→$3 in, $10→$15 out per MTok) | 2026-09-01 | Only affects API-baseline comparisons; the current calculator has no API lane |

## 7. Where things live

- **Branch**: `codex/rayline-remote-mvp`, HEAD `902c4ab4`,
  upstream `atlasfutures/codex/rayline-remote-mvp`, **ahead 1 — NOT PUSHED**.
- **PR**: none exists for this branch.
- **Working tree**: clean.
- **Second worktree**: `.claude/worktrees/rayline-reconcile` on
  `codex/rayline-main-reconcile` @ `bfd0cc0e` — **still contains the old region
  pin** at `modal_session_service.py:83,186` and the two freeze assertions. It
  will need the same change when it reconciles. Left untouched deliberately.
- **Disk-only, outside the repo** — these are the only local copies:
  - `~/.agent/diagrams/rayline-arc-encoder-serving-cost.html` (155 KB) — the
    current calculator
  - `~/.agent/diagrams/memex-gpu-serving-cost-calculator.html` (79 KB) —
    superseded, wrong workload
- **Disk-only, inside the repo but gitignored**: `.agent-harness/` (2.5 GB) —
  the evidence base for every measured claim in §2. The size is misleading and
  the distinction matters if you are deciding what to preserve:

  | Path | Size | Contents |
  |---|---|---|
  | `rayline-parity/` | 2.4 G | 21 runs — PERF015–027, DYN006 |
  | ↳ `rayline-parity/private-source/` | **2.3 G** | **not measurements** — a 1.4 GB `qwen3.5-0.8b-bf16.gguf`, two `model.safetensors`, and compiled `rayline-c82-encoder` binaries (x86_64-linux, aarch64-darwin) |
  | `helm/` | 36 M | 3 runs |
  | `rayline-kv-cache/` | 4.7 M | 17 runs — AGT017/018/019 |
  | `rayline-modal-native/` | 528 K | 1 run — AGT014, the **only** non-H100 datapoint in existence |
  | `rayline-vllm-profile/` | 20 K | 3 runs — PERF030, source of both throughput constants |

  **The irreplaceable evidence is 250 `.json` + 28 `.jsonl` — a few MB.**
  PERF030, which supplies the 11,117 / 115,764 tok/s figures the whole cost
  model rests on, is a *single* `report.json`. The 2.3 GB that makes this
  directory look expensive to preserve is regenerable model/runtime material.
  Archiving the JSON is cheap and has not been done.

  Directly confirmed: **zero `.log` / `.txt` / stdout files anywhere** in the
  tree, which is why the vLLM startup values in §3 are derived rather than
  observed.
- **Published copies of the calculator**:
  - Artifact (updates in place): `https://claude.ai/code/artifact/767f0d29-f595-4ba2-8a06-f93365bc5d20`
  - R2 snapshot (frozen): `https://pub-20e8892d7637481fa429a3869df5a15f.r2.dev/shared/d4de3dea-4e7e-4324-b8e2-0169a3ea6e6c/rayline-arc-encoder-serving-cost.html`
  - Earlier R2 snapshots exist and show superseded models — prefer the above.

## 8. Exact resume commands

All of these were run and succeeded in this session, copied from the shell.

```bash
# state
cd /Users/chilang/code/semantic-router
git log --oneline -1                      # expect 902c4ab4
git status --short                        # expect clean

# the change is present and correct
cd src/vllm-plugins/rayline_arc_io
grep -n "^    region=" modal_session_service.py          # expect NO output (exit 1)
grep -c 'assert "region" not in function_keywords' tests/test_modal_session_service.py   # expect 1

# tests — note the package-local venv; system python3 has no pytest
.venv/bin/python -m pytest tests/test_modal_session_service.py -q        # expect 6 passed
.venv/bin/python -m pytest tests/ -q \
  --ignore=tests/test_rayline_parity_arc_config.py \
  --ignore=tests/test_rayline_dynamic_stop_contract.py                   # expect 261 passed, 3 failed (pre-existing)
```

Non-obvious flags and gotchas:

- **`.venv/bin/python`, not `python3`.** The package has a local venv;
  system python has no pytest and `uv` is not installed on this machine.
- **The two `--ignore`d modules fail to COLLECT**, not to assert:
  `ModuleNotFoundError: No module named 'yaml'`. Pre-existing environment gap in
  this checkout, unrelated to any change here.
- **`make agent-lint` does not run in this checkout.** It dies with
  ``tools/make/deps.mk:14: *** Recursive variable `PATH' references itself``.
  Pre-existing. Lint was run directly via `.venv/bin/python -m ruff check`.

```bash
# UNVERIFIED — assembled from a subagent's report, not run in this shell.
# Re-verify before trusting. Auth via a zsh login shell so the helper resolves.
zsh -lic 'gcp-adc && bq query --use_legacy_sql=false "
WITH spine AS (SELECT h AS hour_utc FROM UNNEST(GENERATE_TIMESTAMP_ARRAY(
    TIMESTAMP(\"2026-07-11\"), TIMESTAMP(\"2026-08-09 23:00:00\"), INTERVAL 1 HOUR)) h),
agg AS (SELECT hour_utc, SUM(total_turns) AS turns
  FROM \`memex-desktop.dbt_prod_marts.model_usage_hourly\`
  WHERE date_utc >= \"2026-07-10\" AND date_utc <= \"2026-08-10\"
  GROUP BY hour_utc)
SELECT EXTRACT(HOUR FROM s.hour_utc) hod, AVG(IFNULL(a.turns,0)/60.0) tpm
FROM spine s LEFT JOIN agg a USING (hour_utc) GROUP BY hod ORDER BY hod"'
```

**The spine join is required, not cosmetic** — `model_usage_hourly` stores only
non-zero hours, so a naive `GROUP BY hour_utc` silently omits idle hours and
will overstate the trough.

## 9. Cost

- **This session spent no GPU or provider money.** No Modal app was deployed, no
  benchmark was launched, no inference was run. All work was source analysis,
  BigQuery reads, and web research.
- **Prior program accounting** (context, not spent here): $151.95 of $154.31
  authorized cumulative; AGT019d provider spend $0.025316 for 78 logical
  requests.
- **What a resume would spend**: deploying the unpinned encoder and re-running a
  PERF021-style load ladder is the main cost. A single-container H100 at the
  31-minute timeout envelope is ~$2.50 at the pinned snapshot rate, ~$1.43
  unpinned. Capturing the vLLM startup log is one container start — pennies.
- **Cost model correction with real money attached**: at the deployed 8c/64 GiB
  request, unpinning is $8.466 → $4.838/hr per H100. At the calculator's default
  working point (5 provisioned GPUs) that is **$30,819 → $17,611/month**.

## 10. Open decisions for the human

1. **Push `902c4ab4`?** — Not pushed, per the standing instruction never to
   auto-push. The handoff skill correctly warns that an unpushed handoff
   describing unpushed work is two things that don't exist, so this is the
   decision that gates everything else. *Recommendation: push.* The change is
   small, green, and reversible, and the evidence against the pin is this repo's
   own.
2. **Right-size `cpu` / `memory`?** — Would take $4.838 → $4.266/hr (4c/16GiB) or
   $4.171 (2c/16GiB). Breaks no test. *Recommendation: do it, but measure first* —
   §4 records that host usage is genuinely unknown, and the honest fix is to add
   a host sampler to `_GpuMonitor` and read one qualification run rather than
   guess a smaller number.
3. **Stay on Modal at all?** — Measured duty is 99.73%, so every
   serverless-vs-dedicated crossover is exceeded: RunPod Secure H100 Pod at
   $2.99/hr beats a perfectly-tuned Modal config at $4.17. Modal's advantage is
   elasticity this traffic shape never uses. *Recommendation: evaluate, but not
   urgently* — the region and sizing fixes capture most of the saving with far
   less risk.
4. **Fix the historical cost receipts?** — Every `.agent-harness/` budget receipt
   understates spend by ~75% for the pinned period, and the `COST_CEILING_USD`
   gates were evaluated against a rate that did not exist. This is a governance
   question about whether past authorizations were valid, not an engineering one.
5. **Authorize the two cheap experiments?** — (a) capture the vLLM startup log to
   convert the whole 544-token block derivation from derived to observed; (b) run
   one PERF021-style ladder on a FlashInfer app, since every saturation number in
   existence is `torch_reference`-only. Both are blocked behind TD050's explicit
   parking decision, which is why they are listed here rather than as agent work.
