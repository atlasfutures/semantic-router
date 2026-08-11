# Live `claude` CLI end-to-end through Rayline ARC — 2026-08-11

**Result: passed.** A real Claude Code CLI session drove three turns through the
full Rayline ARC path and built working code. No stage was mocked.

Total spend: **$0.01401404** of a $1.00 hard-capped OpenRouter key, plus a few
minutes of H100 time. All resources torn down and verified.

## What ran

```
claude CLI --> Envoy :18888 --> router :50051 (ExtProc)
                                   |--> Redis (episode store)
                                   |--> Modal H100 ARC session encoder
                                   `--> OpenRouter (3 real models)
```

Three turns, one session, `--resume` for turns 2 and 3:

1. "Create a file reverse.py containing a single function reverse_string(s)..."
2. "Now add a test file test_reverse.py with two assert-based tests..."
3. "Add a tiny CLI wrapper cli.py that reads one argument..."

All three exited 0. The built code runs: `test_reverse.py` passes, and
`python3 cli.py hello` prints `olleh`.

## The evidence that matters

**Episode continuity held across turns.** Six ARC decisions were logged, all
carrying the *same* episode hash `a37c8bd0c9443c66`. `--resume` keeps
`X-Claude-Code-Session-Id` stable, which is what makes the retained-KV path
reachable at all. Had each turn opened a fresh episode, the run would have
proved far less.

**Retained KV worked perfectly.** At end of session:

```
resident_sessions       1
resident_tokens         3185
llm_rayline_arc_cache_miss_tokens_sum    0
llm_rayline_arc_encoder_latency_seconds_count   6
```

`cache_miss_tokens_sum = 0` means every encoder call after the first reused the
retained session. Zero full replays across three turns.

**Readiness gates all passed** (the ARC ones are the load-bearing ones — router
`/health` and `/ready` can be green while ARC is dead):

| Gate | Result |
|---|---|
| Router `/health` | healthy |
| Router `/ready` | 200 |
| `llm_rayline_arc_component_ready{component="artifact_head_encoder"}` | 1 |
| `llm_rayline_arc_component_ready{component="episode_store"}` | 1 |
| Encoder via proxy token | `status: ok`, 0 resident at start, max_sessions 8 |

ARC readiness took ~70s after container start. A check at 35s reported the
metric family *absent*, which is a different state from `0` and is easy to
misread as "never configured".

## What this does NOT prove

**Routing quality.** The public artifact head is synthetic. Every one of the six
decisions selected arm 2, and the raw scores were evenly spaced by exactly
0.006 on every call:

```
[-0.02256021, -0.016560212, -0.010560212]
[-0.022460897, -0.016460897, -0.010460897]
```

That constant spacing is the signature of hand-built sparse tensors, not a
learned policy. **Arm switches in this configuration would be structurally
determined, not intelligent.** Anyone shown this run should be told that
explicitly. Swapping in the real C82 artifact is Topology B in the deployment
plan — though note C82 was itself NOT-PROMOTED, and the most recent registry
status keeps fixed GLM as the serving default.

Also unproven: concurrency beyond one session (the encoder caps at 8, and the
9th episode 429s into a 503), failover or HA, the Kubernetes deployment shape,
and durable decision audit — the Q-value line is stderr-only and is lost on
rotation.

## Fixes this run depended on

Four blockers had to be cleared first; three were found only by trying:

1. **Worker-b pricing mismatch.** The config had moved to `openai/gpt-5.6-luna`
   while the artifact fixture still declared the retired `xiaomi/mimo-v2.5` at
   different prices. Readiness fails closed on that, so the packet could not
   have reached readiness at all. Fixed, with a parity test at the Go code's own
   1e-9 tolerance.
2. **The 96-token completion cap.** `minimum_completion_tokens ==
   max_completion_tokens == 96` pins *every* reply to exactly 96 tokens,
   because the minimum is a floor on the budget rather than the output. The
   ceiling moved to 4096 and the floor dropped to 16 — not zero, because at zero
   the dispatcher deletes `max_tokens` entirely and uncaps provider spend.
3. **Stale router image.** Exactly one Go commit landed after it — `6dcaa3e0`
   "support bounded provider orders" — which is the feature worker-a's realigned
   `provider_order` depends on.
4. **Encoder not deployed.** The measurement launchers stop the app at teardown,
   so PERF032's run had removed it. Redeploy took 131s and kept the frozen host.

## Operational notes

- **The warm floor was deliberately not pinned.** Pinning bills an H100
  continuously and only stops when something unpins it. Unpinned, the container
  scales to zero five minutes after the last request with no action from anyone.
  The cost is one ~90s cold start on turn 1. For an unattended run that is the
  right trade.
- **`--resume`, not `--session-id`**, for turns 2+. The CLI hard-errors if the
  session file already exists. `--fork-session` mints a new UUID and must not be
  used.
- **`make` needs a sanitised PATH on this machine.** A literal `${PATH}` entry
  in the shell PATH makes `export PATH := $(TOOLS_BIN_DIR):$(PATH)`
  self-reference. This is environmental, not a repo defect — an earlier handoff
  recorded it as a broken `deps.mk`, which was wrong.

## Teardown

Verified clean: zero Modal containers, zero compose services, Modal proxy token
deleted, OpenRouter ephemeral key deleted, compose volumes removed.
