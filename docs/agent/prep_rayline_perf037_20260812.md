# Prep: PERF037 — the 32-lane burst-absorption packet, prepared and unlaunchable

Written 2026-08-12. Read this with
`docs/agent/handoff_rayline_perf036_20260811.md`, whose §3 records burst
absorption as unproven and whose §10.2 recommends measuring it before any
production commitment. Nothing here has run. Nothing here can run.

## TL;DR

- PERF037 is **prepared, not authorized**. `LAUNCHABLE_CONTRACT = None`, the
  Pathfinder pin is the literal `PENDING`, and the budget fails closed by
  arithmetic. No GPU second was bought preparing it.
- It asks: **does the deployment absorb the 2.33 dec/s worst recorded
  production burst, and at what lane count?**
- The preregistered answer is **no, at every lane count this corpus can
  express**. What the run converts the measurement into is the number the
  decision needs: how long such a burst may last before recovery exceeds
  budget.
- It needs a **$5.6186208 raise** — the whole envelope, because PERF036 closed
  on the reserve floor and left exactly $0.00 of spendable headroom.
- One design question is genuinely open and a human should answer it before
  authorizing. See §6.

## 1. What PERF037 asks, and why the shape is what it is

The worst recorded production burst is 2.33 decisions per second. PERF036
measured this card completing 0.8877 at eight lanes. The gap is why single-
instance Cloud Run cannot be committed to yet.

Eight lanes cannot even pose the question. The probe runs one thread per
episode, so at most `max_episode_lanes` requests are ever in flight; anything
offered past eight lands in client-side lateness rather than on the encoder.
So PERF037 runs at 32 lanes — every episode the frozen 128-case corpus has,
and therefore the widest rig this corpus can ever express.

It runs **PERF034's packet byte for byte on PERF036's card**. Every digest in
the contract is imported from the PERF034 contract object rather than retyped,
so byte-identity is structural. That makes all four rungs paired cross-GPU
comparisons rather than only the anchor, and it means no packet had to be
generated: the packet is already on disk at
`.agent-harness/rayline-parity/packet-perf034`, and its eleven digests were
re-verified during preparation.

### How it avoids the failure that voided three plateaus

PERF032, PERF034 and PERF036 all ended with a plateau verdict voided or
subsumed by the finite-corpus drain clause. The mechanism is known:
`completion_throughput = completed / (span + drain)` rolls off with rate even
under zero queueing, and the **marginal gain between rungs** inherits that
artifact.

PERF037's verdict does not use marginal gain. Absorption is decided on
quantities a single cell measures directly — realized arrival rate, completion
throughput, peak backlog, and the integrity counters — so the drain arithmetic
has nothing to corrupt. The plateau verdict is carried from PERF034 unchanged
as non-voting corroboration, and the contract preregisters that the drain
clause may void that and nothing else
(`DRAIN_CLAUSE_VOIDS_ABSORPTION_VERDICT = False`).

For this question the finite corpus is the instrument rather than the flaw. A
production burst *is* a finite pulse: each cell offers 128 decisions at a
Poisson rate and then stops, which is a burst of `128 / rate` seconds.

## 2. The prediction and its band

| Quantity | Value |
| --- | ---: |
| Anchor (PERF034, measured 32-lane H100 ceiling, arc `r645`) | `2.3061533124360074` dps |
| Scaling | dense-FP16 TFLOPS, `480 / 989` |
| **Point prediction** | **`1.1193` dps** |
| Band, `±30%` | `0.7835` – `1.4550` dps |
| Production burst | `2.33` dps |
| Burst as a multiple of the prediction | `2.08x` |
| Burst as a multiple of the band top | `1.60x` |

**Falsification, stated as a number.** A measured ceiling of `2.33` dps or
better falsifies the "not absorbed" prediction. That is more than double the
point prediction and well outside the band. A measured ceiling outside
`0.7835 – 1.4550` falsifies naive TFLOPS scaling at a second lane count, which
is a result rather than a voided arm — the same standing the band had in
PERF036.

**The cross-check the method already carries on this card.** Applying the same
ratio to PERF033's measured eight-lane H100 ceiling predicts
`1.7651 × 480/989 = 0.8567` for the eight-lane RTX PRO 6000. PERF036 measured
`0.8877` — `3.5%` low. That is a same-corpus, same-lane-count validation
PERF036 itself did not have when it ran.

**A second route, and why it is not a second witness.** Scaling PERF036's
measured eight-lane RTX ceiling by the H100's own 8-to-32-lane gain gives
`1.1598`. It sits inside the band, so the choice of route cannot flip the
outcome. But it is *not* independent: it differs from the TFLOPS route by
exactly the `3.5%` residual above, algebraically. The contract's test pins that
identity so nobody reads it as corroboration.

**What the measurement becomes.** `absorbable_burst_seconds` inverts the
queueing arithmetic: backlog accumulates at `burst − ceiling` and clears at
`ceiling`, so a burst of `T` seconds needs `(burst − ceiling)·T / ceiling` to
recover. Against a 30-second recovery budget:

| Ceiling | Absorbable burst |
| --- | ---: |
| Band floor `0.7835` | `15.2` s |
| Prediction `1.1193` | `27.7` s |
| Band top `1.4550` | `49.9` s |

That is the sentence a deployment decision can use: *this deployment absorbs a
2.33 dec/s burst lasting roughly half a minute, and queues beyond it.*

## 3. Pass, fail and void, preregistered

**A cell absorbs** (`absorbs_burst`) when all three hold:

1. Realized arrival rate `≥ 2.33` dps — it actually offered a burst worth the
   name.
2. Completion throughput `≥ 0.95 ×` realized arrivals — it kept up.
3. Peak backlog `≤ 32` — the rig never went over-full, so nothing waited behind
   a saturated encoder.

**The arm passes the absorption claim** if any cell absorbs. It fails — the
predicted outcome — if none does. The predicate is calibrated against a real
receipt: PERF034's `r240` (realized `2.5036`, completed `1.3382`, peak backlog
`33`) does **not** absorb, so the predicate is not trivially satisfiable.

**The arm is void** if any of these hold, unchanged from the family's standing
integrity gates:

- any cell records `failed > 0`, non-zero `cache_miss_tokens`, non-zero
  `session_actions.rebuilt`, or non-zero `provider_calls`;
- the cross-cell selected-worker trace digests differ;
- any packet digest differs from the contract's — byte-identity to PERF034 is
  the whole cross-GPU claim;
- the paid wall is exceeded, or any encoder container survives teardown.

**The drain clause may void the plateau verdict and nothing else.** Recorded in
the contract as data, not argued at analysis time.

**One inherited check cannot run.** The probe hashes the selected-worker trace
without persisting its entries, so the PERF020 32-case prefix digest is not
recomputable. PERF034 recorded that gap; PERF037 changes nothing about the
probe and inherits it. `TRACE_PREFIX_CHECK_IS_RUNNABLE = False`.

**Teardown** must show all five of `TEARDOWN_REQUIREMENTS`, including the one
an agent is most likely to skip: an independent `modal container list` in the
`dev` environment returning empty, separate from the launcher's own report.

## 4. The envelope, and the raise it needs

Same pricing basis PERF036 used, unchanged: Modal on-demand RTX PRO 6000 at
`$0.000842/s` (snapshot `modal-on-demand-2026-08-11-rtx6000-cpu-memory`), plus
the unchanged 8-core / 64 GiB container.

```text
rate     = 0.000842 + 8 × 0.0000131 + 64 × 0.00000222 = 0.00108888 USD/s
seconds  = 2400 paid wall + 2460 orphan + 300 scaledown = 5160
envelope = 5160 × 0.00108888 = $5.6186208   (conservative upper bound)
```

The authority position, unchanged since PERF036 closed:

```text
authorized ceiling   $197.459850266383
conservative to date $194.459850266383
difference           $3.00  -- all of it the required reserve
spendable headroom   $0.00
```

So the raise needed is the **whole envelope: `$5.6186208`**, taking the ceiling
to **`$203.078471066383`**, which leaves the reserve at exactly `$3.00` again.
This is the family's minimum-viable-grant precedent, and it is now computed
rather than typed — `minimum_viable_grant_usd(PERF037.budget)` returns it, and
returns `$0.00` for all three closed packets.

Until that grant lands, `budget_receipt` raises `BudgetError`: the reserve is
short by exactly the envelope. That is the fail-closed gate, and it is
arithmetic rather than a flag.

**Actual spend will be far lower.** PERF036 used `$0.9561` of the same
envelope. PERF034, the closest shape, ran a paid window near 1,000 s of its
2,400 s wall. A comparable outcome here is roughly `$1.6`. The envelope is the
bound, not the estimate.

## 5. The exact authorization steps a human must take

Preparation may move none of these. Each one is a separate, reviewed act.

1. **Grant the raise.** Say the figure out loud: raise the cumulative authority
   from `$197.459850266383` to `$203.078471066383`, a `$5.6186208` grant, for
   run id `rayline-rtx6000-burst-perf037-20260812`.
2. **Register it in Pathfinder.** New registry ID in
   `/Users/chilang/code/pathfinder-rayline-vsr-mvp` (branch
   `codex/rayline-vsr-mvp`) recording the confirmed grant. Never touch
   `~/code/pathfinder` itself — it holds uncommitted user work. Validate with
   `scripts/validate_experiment_registry.py`, and remember the YAML gotcha:
   plain scalars must not contain `": "`; use `" -- "`.
3. **Push Pathfinder**, and take the resulting head sha.
4. **Authorize commit** in this repo: move `AUTHORIZED_CUMULATIVE_USD` to
   `203.078471066383` and record the human's words and the minimum-grant
   interpretation in the commit message.
5. **Bind commit**: set `PATHFINDER_AUTHORIZATION_COMMIT` to the pushed
   Pathfinder head, and `LAUNCHABLE_CONTRACT = PERF037`.
6. **Run once.** One execution, no retries — a launched failure closes the ID
   for good under the registry's no-retry clause. Pass the PERF034 packet dir:
   `--packet-dir .agent-harness/rayline-parity/packet-perf034`.
7. **Close the gate**: `LAUNCHABLE_CONTRACT = None` again, keeping the real pin
   as the record of what was measured.
8. **Record** the result in pl-0041, complete the Pathfinder registry entry,
   and verify teardown with both instruments.

Steps 1–3 are the human's. An agent may not perform step 4 or 5 without a
fresh grant in the current conversation.

## 6. Open questions for the human, before authorizing

1. **Is a ~28-second burst budget enough to decide with?** The predicted
   outcome is "not absorbed", which is already implied by arithmetic: PERF034
   measured 2.31 dps at 32 lanes on an H100, and this card is roughly half an
   H100. If the decision only needs "no", it is already answerable for free.
   What the run adds is the *degradation shape* — the actual queueing delay,
   the recovery time, and whether integrity holds under a 2x overload on a card
   whose memory pool differs from the H100's. Fund it only if the shape, not
   the verdict, is what the decision turns on.
2. **Accept the thinner wall margin, or pay for a wider wall?** Calibrated on
   PERF034's own schedule (845 recorded seconds against a 708.5 s formula
   estimate, so the formula runs `1.19x` optimistic), the estimate here is
   ~1,445 s at the prediction and ~1,942 s at the band floor, against the
   2,400 s wall. That is ~19% margin at the floor, thinner than PERF036 had.
   The wall binds only below ~0.61 dps — under PERF036's measured eight-lane
   figure, which would mean 32 lanes ran slower than eight. A 50-minute wall
   would cost `$6.2719488` instead and need a `packet_ceiling_usd` of `$7.00`.
   Recommendation: keep the 40-minute wall; the breach case is close to
   impossible and a breach aborts rather than overspends.
3. **Is the anchor good enough?** The prediction scales PERF034's figure, and
   PERF034's own anchor gate is recorded as **not held** — it came in 7.15%
   under PERF033's recorded throughput with no tolerance preregistered. Every
   absolute PERF034 number inherits that qualification, and so does this
   prediction. The `±30%` band is wide enough to absorb it, but a reader should
   know the anchor is qualified.
4. **The rig cannot produce a baseline-plus-spike profile.** See §7.

## 7. What this packet cannot measure, stated plainly

The rayline-arc load generator produces **one homogeneous seeded-Poisson pulse
per cell** and nothing else. `arrival_process` is a compile-time constant
(`seeded_poisson`), the workload schema is exact-key validated so no burst
field can be added without a schema v2, and the schedule is generated at run
time from `(rate, seed)`. There is no ramp, no on/off pattern, no piecewise
rate schedule, and no explicit arrival list anywhere in the tree.

So PERF037 measures a **finite Poisson pulse at a burst-grade rate**, which is
the conservative approximation of a production burst: it starts against an
empty rig, with no warm baseline and no pre-existing backlog. It does **not**
measure a baseline → spike → baseline profile. Building that would need a new
`arrival_process` value, an `open-loop-workload.v2` schema, a second schedule
function beside `poisson_schedule`, and a new packet-builder path. None of that
is built here, deliberately: it is new measurement machinery, and the pulse
already answers the sizing question the deployment decision is waiting on.

If a reviewer decides the baseline-plus-spike shape is what the decision
actually needs, PERF037 as designed is the wrong packet and should not be
funded — the harness work comes first.

## 8. Where things are

- **semantic-router**: this worktree, branch `worktree-agent-a99d6770882aaee16`
  off `codex/rayline-remote-mvp` at `96d614f5`. Five commits, local, unpushed.
- **Contract**: `e2e/testing/rayline-arc/rayline_rtx6000_burst_contract.py`
- **Tests**: `src/vllm-plugins/rayline_arc_io/tests/test_rayline_rtx6000_burst_contract.py`
  (18 tests) and `.../test_rayline_three_arm_budget.py` (5 tests)
- **Encoder app**: `rayline-arc-session-encoder-flashinfer-perf037-rtx6000-32lane`,
  the only app in both `RTX6000_APP_PROFILES` and `CAP_RAISED_APP_PROFILES`
- **Packet**: `.agent-harness/rayline-parity/packet-perf034`, unchanged, all
  eleven digests re-verified 2026-08-12
- **Pathfinder**: untouched. The registry entry belongs to authorization time.

Resume command:

```bash
cd /Users/chilang/code/semantic-router/src/vllm-plugins/rayline_arc_io
~/.local/bin/uv run --extra test --with pyyaml python -m pytest \
  tests/test_rayline_rtx6000_burst_contract.py \
  tests/test_rayline_three_arm_budget.py -q
```
