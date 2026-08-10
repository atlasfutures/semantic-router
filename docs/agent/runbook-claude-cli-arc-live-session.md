# Runbook: a live `claude` CLI session through Rayline ARC

Take the operator from nothing to a working
`claude` → Envoy → Semantic Router → Modal H100 ARC encoder → OpenRouter stack,
run a real multi-turn session against three real models, capture the only
record of each routing decision, and tear the whole thing down with proof.

**This runbook spends real money.** Read [§0](#0-cost-and-authority) before
anything else. Companion script for every cost-bearing action:
`e2e/testing/rayline-arc/live_session_ops.sh`.

Written 2026-08-10 against branch `codex/rayline-remote-mvp` @ `553ddf6f`.
Facts marked **CONFIRM AT RUN TIME** were not verifiable statically; §10 lists
them all in one place.

---

## 0. Cost and authority

Three things bill. Two of them keep billing if you walk away.

| Source | Rate | Stops when |
|---|---|---|
| Modal H100 encoder, pinned warm | **~$4.838/hr** (~$0.081/min) | `min_containers` back to 0 **and** containers stopped |
| OpenRouter inference | metered per token | ephemeral key deleted, or its hard limit is hit |
| Local Docker | $0 | `docker compose down` |

The $4.838/hr figure is the unpinned-region rate at the deployed 8-core /
64 GiB request, derived in `docs/agent/handoff_rayline_serving_cost_20260810.md`
§2.4. Region pinning was removed in `902c4ab4`; the older `$8.466/hr` figure
assumes the pin and no longer applies.

**Expected blast radius for a 30-minute session: ~$2.42 Modal + ≤$1.00
OpenRouter ≈ $3.42.**

> **Authority gate — read this.** The Rayline programme's own accounting
> (`handoff_rayline_serving_cost_20260810.md` §10.5) records **$2.56 of headroom
> against a $3.00 required reserve**. A ~$3.42 session is *not* funded by the
> existing cumulative authorization. This run needs a fresh human authorization
> before it starts. Nothing in this runbook creates that authorization.

Also note this is **not** a preregistered benchmark and must not be reported as
one. `openrouter_launch_authority.py:31-49` pins `agentic` at `("", "")`, so
`run_openrouter_fullstack.py --mode agentic` is source-closed and will exit with
`agentic launch authority is source-closed`. That is why this is a manual
runbook rather than a launcher invocation.

---

## 1. Preconditions

### 1.1 Repo state

```bash
cd /Users/chilang/code/semantic-router
git rev-parse --abbrev-ref HEAD          # expect codex/rayline-remote-mvp
git status --porcelain                   # expect clean before you start
```

You do **not** need the checkout to be remote-visible for this runbook (that
gate only exists inside the frozen launchers), but note that committing local
changes will break `verify_source_authority` for any source-closed launcher you
run afterwards.

### 1.2 Docker and the router image

```bash
docker version --format '{{.Server.Version}}'    # 29.4.0 confirmed working
docker image inspect ghcr.io/vllm-project/semantic-router/vllm-sr:latest \
  --format '{{.Created}}'
```

`compose.yaml:154-155` pins `ghcr.io/vllm-project/semantic-router/vllm-sr:latest`
with `pull_policy: never` — the image must exist locally and there is no
fallback.

**The image on this machine is stale.** It was built `2026-08-03T20:57`, and
commit `6dcaa3e0` ("feat(rayline): support bounded provider orders") touched
`src/semantic-router/pkg/selection/raylinearc/provider_contract.go` afterwards.
Rebuild:

```bash
make vllm-sr-build          # -> ghcr.io/vllm-project/semantic-router/vllm-sr:latest
```

Verify the rebuild is newer than the last Go change:

```bash
git log -1 --format=%cI -- src/semantic-router candle-binding ml-binding nlp-binding
docker image inspect ghcr.io/vllm-project/semantic-router/vllm-sr:latest --format '{{.Created}}'
```

### 1.3 Modal CLI

`modal` is **not** installed in any venv inside this checkout. Every launcher in
`e2e/testing/rayline-arc/` is run with the Pathfinder venv:

```bash
~/code/pathfinder-rayline-vsr-mvp/.venv/bin/python -c "import modal; print(modal.__version__)"
# expect 1.5.1  (run_openrouter_fullstack.py:97 requires exactly this)
grep -E '^\[|^active|^environment' ~/.modal.toml
# expect [atlasfutures] / environment = "dev" / active = true
```

`live_session_ops.sh` uses that interpreter by default; override with
`LIVE_SESSION_PYTHON`.

> **Colour escapes corrupt `modal … --json` parsing.** Every launcher, and this
> runbook's script, wraps Modal calls in
> `env -u FORCE_COLOR -u COLORTERM TERM=dumb`. Do not drop it.

### 1.4 The encoder app is NOT currently deployed

Checked 2026-08-10:

```bash
bash e2e/testing/rayline-arc/live_session_ops.sh encoder-app
# -> NOT DEPLOYED: rayline-arc-session-encoder
```

`rayline-arc-session-encoder` is absent from the `dev` environment; the frozen
`ENCODER_APP_ID = "ap-XtsWCBEWdw1ncu9Kv12Chj"` in
`run_openrouter_fullstack.py:75` is dead. A deploy is a **required** step
([§3.1](#31-deploy-the-encoder)), not an optional one, and it will mint a new
app id. Match on app *name*, never on that id.

### 1.5 `claude` CLI

```bash
ls -l ~/.local/bin/claude && ~/.local/bin/claude --version    # 2.1.226 confirmed
```

Not on `PATH` in non-login shells — use the absolute path or a login shell.

### 1.6 The two config/fixture facts you must verify first

Both are **currently broken on this branch** and another agent is fixing them in
parallel. Do not bring the stack up until both pass.

**(a) Pricing identity.** `raylineARCPriceIdentityMatches`
(`rayline_arc_readiness.go:299-333`) requires the four price legs in
`deploy/compose/rayline-arc/config-openrouter-agentic.yaml` to equal the
artifact manifest's per-token costs × 1e6, to a relative tolerance of 1e-9. As
of `553ddf6f` they do not:

| worker | config `provider_model_id` / prices | fixture `model` / prices |
|---|---|---|
| worker-a | `deepseek/deepseek-v4-flash` 0.09 / 0.09 / 0.09 / 0.18 | `deepseek/deepseek-v4-flash` — **match** |
| worker-b | `openai/gpt-5.6-luna` 0.2 / 0.02 / 0.25 / 1.2 | `xiaomi/mimo-v2.5` 0.168 / 0.168 / 0.168 / 0.336 — **MISMATCH** |
| worker-c | `tencent/hy3` 0.14 / 0.14 / 0.14 / 0.58 | `tencent/hy3` — **match** |

A mismatch is `failure_class="artifact_dispatch_contract"` and the router comes
up with ARC dead. Verify:

```bash
grep -n -A6 'name: worker-b' deploy/compose/rayline-arc/config-openrouter-agentic.yaml
grep -n -A12 '"id": "worker-b"' e2e/testing/rayline-arc/openrouter_agentic_artifact_fixture.py
```

The model id must match too — `raylineARCEndpointIdentityMatches` compares
`cfg.ResolveExternalModelID(worker.ID, endpoint.Name)` against `worker.Model`.

**(b) The 96-token completion cap.**
`openrouter_agentic_artifact_fixture.py:15` sets `MAX_COMPLETION_TOKENS = 96`,
and `_worker_contract` defaults `minimum_completion_tokens` to the same value.
`applyARCExecutionLimits` (`rayline_arc_dispatch.go:150-173`) then computes
`max(client max_tokens, minimum)` and clamps to the maximum — so **every reply
is pinned to exactly 96 tokens**, regardless of what `claude` asks for. That is
correct for a frozen benchmark and useless for an interactive session.

```bash
grep -n 'MAX_COMPLETION_TOKENS' e2e/testing/rayline-arc/openrouter_agentic_artifact_fixture.py
```

> **This is the moment the cost model changes.** The 96-token cap is what has
> kept every historical run under $0.026. Once it is lifted, **the OpenRouter
> key's hard server-side limit is the only cap on provider spend.** Set it
> deliberately ([§2.2](#22-openrouter-ephemeral-key--the-only-real-cost-cap)).
>
> When raising it, prefer raising `max_completion_tokens` while leaving
> `minimum_completion_tokens` low — the fixture already supports this
> (`openrouter_artifact_fixture.py:170-185`, "callers that must honour a real
> client's own budget pass a lower floor"). If the floor stays equal to the
> ceiling, every request — including one-word ones — is issued with the full
> budget, and some providers reject a `max_tokens` above their own output limit.

### 1.7 Credentials you will need (do not mint yet)

| What | Source |
|---|---|
| OpenRouter management key | 1Password `j4lg7ndsoyj7dnma6fb66bjadu`, vault `MacCli`, field `credential` |
| Modal proxy token | minted per run ([§2.1](#21-modal-proxy-token)) |
| OpenRouter ephemeral key | minted per run ([§2.2](#22-openrouter-ephemeral-key--the-only-real-cost-cap)) |
| HF token | **not needed** — see below |

**No HF token is required for this runbook.** The OpenRouter launchers never
read `HF_TOKEN` (unlike `rayline_three_arm_launcher.py:477`, which needs it for
a private checkpoint). The Modal encoder image uses a persistent Modal Volume
cache (`rayline-hf-cache`, `modal_session_service.py:195`) and carries no HF
credential. If you nonetheless need one for an unrelated step, the read-only
token is:

```bash
op item get wbq2jsrb2ne7zgyqi5ifhsjmb4 --vault MacCli --reveal --fields credential
```

---

## 2. Credentials — how to mint each, and how to scope it

### 2.1 Modal proxy token

The encoder's web app is declared `@modal.asgi_app(requires_proxy_auth=True)`
(`src/vllm-plugins/rayline_arc_io/modal_session_service.py:303`). Modal's edge
terminates the auth; the FastAPI app never sees it. The router sends the pair as
`Modal-Key` / `Modal-Secret` headers
(`src/semantic-router/pkg/selection/raylinearc/encoder_client.go:380-384`).

The only in-repo minting idiom, from `run_openrouter_fullstack.py:550-552` and
`:400`:

```python
manager = modal.Workspace.from_context().proxy_tokens
token = manager.create()          # -> token.token_id, token.token_secret
...
manager.delete(token.token_id)    # deletion takes the ID, not the object
```

Wrapped:

```bash
bash e2e/testing/rayline-arc/live_session_ops.sh token-mint
# line 1 = token_id, line 2 = token_secret
```

**Scoping.** Modal proxy tokens are workspace-scoped; there is no per-app
restriction. Scope them in *time* instead: mint immediately before the run,
delete immediately after ([§7](#7-teardown-and-cost-stop)). One token per run,
never reused.

Router side (`rayline_arc_readiness.go:415-470`):

```
RAYLINE_ARC_E2E_MODAL_KEY    -> compose -> RAYLINE_ARC_MODAL_KEY    -> config modal_key_env
RAYLINE_ARC_E2E_MODAL_SECRET -> compose -> RAYLINE_ARC_MODAL_SECRET -> config modal_secret_env
```

`raylineARCOptionalSecret` treats **unset and empty-string identically** and
returns an error, which `createRaylineARCEncoder`
(`rayline_arc_encoder_membership.go:38`) collapses to
`failure_class="encoder_auth"`. Fails closed: the router starts, but
`llm_rayline_arc_component_ready{component="artifact_head_encoder"}` is `0` and
every request 503s. There is also a paired-credential guard
(`encoder_client.go:230-232`) — supplying only one of the two yields
`failure_class="encoder_config"`.

### 2.2 OpenRouter ephemeral key — the only real cost cap

`e2e/testing/rayline-arc/openrouter_key_management.py:13,56-64` mints it:

```python
OPENROUTER_MANAGEMENT_URL = "https://openrouter.ai/api/v1/keys"

def create_ephemeral_key(management_key, run_id, key_limit_usd) -> tuple[str, str]:
    response = _management_request(
        method="POST", management_key=management_key,
        payload={
            "name": f"rayline-arc-{run_id}",
            "limit": key_limit_usd,          # USD, enforced server-side
            "include_byok_in_limit": True,
        },
        expected_status=201,
    )
    return response["key"], response["data"]["hash"]
```

and `:81-89` reads settled usage back (`GET /api/v1/keys/{hash}` → `data.usage`).
`:92-98` is a **hard delete** — there is no disable or expire path.

```bash
export OPENROUTER_MANAGEMENT_KEY="$(op item get j4lg7ndsoyj7dnma6fb66bjadu \
  --vault MacCli --reveal --fields credential)"

RUN_ID="claude-live-$(date -u +%Y%m%dT%H%M%SZ)"
bash e2e/testing/rayline-arc/live_session_ops.sh key-mint "$RUN_ID" 1.00
# line 1 = the key (goes to RAYLINE_ARC_E2E_PROVIDER_KEY)
# line 2 = the hash (needed for usage read-back and deletion — SAVE IT)
```

#### Recommended limit: **$1.00** for a ~30-minute interactive session

Justification, from this repo's own receipts:

- The in-repo precedent for a *30-minute* agentic packet is **$0.75**
  (`openrouter_fullstack_packets.py:76`, `maximum_seconds=30*60`). Same three
  models, same config, same wall-clock envelope — but with the 96-token cap on.
- Settled spend has **never exceeded $0.026** on any historical arm
  (`.agent-harness/rayline-kv-cache/` run dirs, `*-key-usage.json`; largest is
  `$0.025302124`). Those runs emit 96 tokens per reply.
- An interactive Claude Code session is a different shape: the CLI resends the
  whole conversation plus its tool schemas every turn, so input dominates.
  Order-of-magnitude for 40 turns averaging ~25k prompt tokens and ~1.5k
  completion tokens, at the most expensive arm's rates ($0.20/$1.20 per 1M):
  1M × $0.20 + 60k × $1.20 ≈ **$0.27**. A tool-heavy session that pumps file
  contents into context could plausibly reach 3M prompt tokens ≈ **$0.60**.
- **OpenRouter's limit check counts in-flight pre-authorization holds, not
  settled cost.** This is the failure that matters: AGT018 died with HTTP 402 at
  request 35 of 36 with only `$0.025` settled against a `$0.05` limit — a ~2×
  effective inflation (`openrouter_kv_cache_successor_contract.py:44-51`). The
  proven-comfortable headroom multiples are 7–11× ($0.15 limit) and 17–25×
  ($0.25 limit).

$1.00 gives ~4× headroom over the realistic $0.25 mid-point after the 2× hold
inflation, and is only 33% above the repo's own 30-minute precedent. A 402
mid-session is strictly worse than a small overspend here: it kills the run
while the H100 keeps billing at $0.081/min.

Do not go below $0.50. Do not go above $1.50 without re-reading §0.

**Scoping.** The key is named `rayline-arc-<run-id>` so it is identifiable in
the OpenRouter dashboard, it is passed only to the router container (never to
`claude`), and it is deleted at teardown. The management key never enters the
container environment.

---

## 3. Encoder warm floor

`modal_session_service.py:199-213` deploys with `scaledown_window=300` and
`max_containers=1`. Every think-gap longer than 5 minutes therefore drops the
container and the next turn pays a cold start measured at **78.9s (AGT005) to
96.892s (SQP001)** against a warm p50 of 0.841s. In an interactive session that
is a guaranteed client timeout ([§8.5](#85-cold-start-blows-the-client-timeout)).

Pin it. `openrouter_encoder_runtime.py:216-232`:

```python
instance = modal.Cls.from_name(app_name, class_name, environment_name="dev")()
instance.update_autoscaler(
    min_containers=1, max_containers=1, buffer_containers=0, scaledown_window=300,
)
```

and the unpin at `:235-244` is the same call with `min_containers=0`.

```bash
bash e2e/testing/rayline-arc/live_session_ops.sh encoder-pin     # BEGINS BILLING
bash e2e/testing/rayline-arc/live_session_ops.sh encoder-unpin   # ENDS IT
```

> **A pinned container bills continuously at ~$4.838/hr whether or not a single
> request is sent.** Unpinning at teardown is a cost control, not hygiene. Note
> also the latent bug at `openrouter_encoder_runtime.py:226-232`: the
> `encoder_autoscaler_pinned` flag is set *after* `update_autoscaler` returns,
> so a partially-applied pin that then raises leaves the automated cleanup
> silently skipping the unpin. Always verify with §7.

### 3.1 Deploy the encoder

Required — the app is not currently deployed (§1.4).

```bash
env -u FORCE_COLOR -u COLORTERM TERM=dumb MODAL_ENVIRONMENT=dev \
  ~/code/pathfinder-rayline-vsr-mvp/.venv/bin/python -m modal deploy \
  src/vllm-plugins/rayline_arc_io/modal_session_service.py
```

Deploy with **no** `RAYLINE_ARC_SESSION_APP_NAME` override. The default app name
is what keeps `ENGINE_BUILD_ID` at the bare
`vllm@9f5ea81ca0aa570aea46baf82311a1139c1267ca`
(`modal_session_service.py:64-73`), which is what
`RAYLINE_ARC_E2E_ENCODER_BUILD_ID` must equal. Any `-flashinfer-*` profile
stamps `+gdn-flashinfer-eager` and readiness fails on `encoder_probe` /
`engine_build`.

The image builds vLLM from source. Launchers allow `timeout=15*60`; budget
5–20 minutes on a cold layer cache. **The image build does not use a GPU and
does not bill H100 time** — only running containers do.

Then:

```bash
bash e2e/testing/rayline-arc/live_session_ops.sh encoder-app   # id, state=deployed, tasks
bash e2e/testing/rayline-arc/live_session_ops.sh encoder-url
# expect https://atlasfutures-dev--rayline-arc-session-encoder-sessionenc-2d82ac.modal.run
```

`verify_encoder_deployment` (`openrouter_encoder_runtime.py:118-126`) also
asserts that an **unauthenticated** `GET /health` returns **401** — proof the
proxy-auth route is actually registered:

```bash
curl -s -o /dev/null -w '%{http_code}\n' \
  "https://atlasfutures-dev--rayline-arc-session-encoder-sessionenc-2d82ac.modal.run/health"
# expect 401
```

---

## 4. Bring-up sequence

### 4.0 Config for the live session — episode identity

This is the single change that makes a Claude Code conversation into an ARC
episode. It requires a run-local config copy; **do not edit
`config-openrouter-agentic.yaml` in place.**

`buildRaylineARCSelectionContext` (`rayline_arc_context.go:45-52`) reads the
episode id from exactly one place — the configured header — and nothing maps
`x-claude-code-session-id` into it:

```go
rawEpisodeID := strings.TrimSpace(reqCtx.Headers[algorithm.RaylineARC.Episode.IDHeader])
if rawEpisodeID == "" {
    result.PreparationFailure = "missing_episode_id"
    return result
}
```

The shipped config sets `id_header: x-rayline-episode-id`
(`config-openrouter-agentic.yaml:120`). The `claude` CLI cannot send that
header by default, but it *always* sends `X-Claude-Code-Session-Id`, stable for
the whole session ([§5](#5-pointing-claude-at-it)).

**Preferred: point the router at the header the CLI already sends.**

```bash
mkdir -p /tmp/rayline-live
sed 's/^\( *id_header: \).*/\1x-claude-code-session-id/' \
  deploy/compose/rayline-arc/config-openrouter-agentic.yaml \
  > /tmp/rayline-live/config-live.yaml
grep -n 'id_header' /tmp/rayline-live/config-live.yaml   # expect x-claude-code-session-id
export RAYLINE_ARC_E2E_CONFIG_PATH=/tmp/rayline-live/config-live.yaml
```

Validation only requires a nonempty lowercase HTTP field name
(`rayline_arc_config.go:309-311`), and `ctx.Headers` is populated lowercased
(`processor_req_header.go:94`), so this is legal and exact. One conversation =
one episode, automatically, with no operator bookkeeping.

**Fallback, if you must leave the config untouched:** `claude` supports
`ANTHROPIC_CUSTOM_HEADERS` (newline-separated `Name: value`), so you can inject
the header the shipped config expects:

```bash
export ANTHROPIC_CUSTOM_HEADERS="x-rayline-episode-id: $(uuidgen | tr 'A-Z' 'a-z')"
export RAYLINE_ARC_E2E_CONFIG_PATH="$PWD/deploy/compose/rayline-arc/config-openrouter-agentic.yaml"
```

This works but decouples the episode from the CLI's own session id — you must
re-export a fresh UUID for every new conversation or episodes will collide.

### 4.1 Environment

```bash
cd /Users/chilang/code/semantic-router

export LIVE_PROJECT=rayline-arc-live-claude
export RAYLINE_ARC_E2E_ENVOY_CONFIG_PATH="$PWD/deploy/compose/rayline-arc/envoy-openrouter.yaml"
# RAYLINE_ARC_E2E_CONFIG_PATH set in §4.0

# Provider credential -> SYNTHETIC_API_KEY in the router container
export RAYLINE_ARC_E2E_PROVIDER_KEY="<ephemeral key from §2.2>"
export OPENROUTER_EPHEMERAL_KEY_HASH="<hash from §2.2>"

# Real Modal encoder, replacing the hermetic fake
export RAYLINE_ARC_E2E_ENCODER_BASE_URL="$(bash e2e/testing/rayline-arc/live_session_ops.sh encoder-url)"
export RAYLINE_ARC_E2E_ENCODER_BUILD_ID="vllm@9f5ea81ca0aa570aea46baf82311a1139c1267ca"
export RAYLINE_ARC_E2E_MODAL_KEY="<token_id from §2.1>"
export RAYLINE_ARC_E2E_MODAL_SECRET="<token_secret from §2.1>"

# Ports (defaults from compose.yaml; listed so you can grep them later)
export RAYLINE_ARC_E2E_ENVOY_PORT=18888
export RAYLINE_ARC_E2E_ROUTER_API_PORT=18082
export RAYLINE_ARC_E2E_METRICS_PORT=19190
export RAYLINE_ARC_E2E_REDIS_PORT=16379
```

Names follow `e2e/testing/rayline-arc/run.sh:23-30` and
`run_openrouter_fullstack.py:284-327` exactly. `RAYLINE_ARC_E2E_ARTIFACT_REVISION`
is deliberately **not** set: it only feeds the membership controller, while the
router's revision is hardcoded in the config as
`public-rayline-arc-openrouter-agentic-v1`, which the fixture also emits.

### 4.2 Stage 1 — build and start

```bash
compose() {
  docker compose --project-name "$LIVE_PROJECT" \
    --file deploy/compose/rayline-arc/compose.yaml \
    --file deploy/compose/rayline-arc/compose-openrouter-agentic.yaml "$@"
}

compose down --volumes --remove-orphans      # idempotent clean slate
compose up --build --detach
```

`--build` is required: the artifact-init image bakes
`openrouter_agentic_artifact_fixture.py` (`e2e/testing/rayline-arc/Dockerfile:4`),
so any fixture edit from §1.6 only lands on a rebuild.

The fake encoder/provider services still start — the router's `depends_on`
(`compose.yaml:178-190`) requires them healthy even though the real Modal
encoder is what serves. They are free.

**Start the log capture immediately** — see
[§6](#6-observability-capture--set-this-up-before-the-run).

### 4.3 Stage 2 — health checks, in order

Each check has a definite "healthy" answer. Do not proceed past a failure.

**(a) Router liveness** — `/health` on the API port:

```bash
curl -sS http://127.0.0.1:18082/health
# {"status": "healthy", "service": "classification-api"}
```

**(b) Router startup readiness** — `/ready`:

```bash
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:18082/ready   # 200
```

> **`/health` and `/ready` do NOT reflect ARC readiness.** Both can be green
> while ARC is completely dead. This is the single biggest gotcha in the stack.

**(c) ARC readiness — the check that actually matters:**

```bash
curl -sS http://127.0.0.1:19190/metrics | grep '^llm_rayline_arc_component_ready'
# llm_rayline_arc_component_ready{component="artifact_head_encoder"} 1
# llm_rayline_arc_component_ready{component="episode_store"} 1
```

Both must be `1`. There are exactly two `component` label values
(`rayline_arc_metrics.go:64-70`, `router_selection.go:50-54`). An *absent*
metric family means ARC was never configured — different from `0`.

If either is `0`, the failure class is in the startup log, emitted once:

```bash
compose logs --no-color router | grep rayline_arc_component_readiness
# {"level":"error", ... "msg":"rayline_arc_component_readiness","ready":false,
#  "failure_class":"artifact_dispatch_contract", "component":"extproc"}
```

Failure-class → cause map (`rayline_arc_readiness.go`,
`rayline_arc_encoder_membership.go`):

| `failure_class` | Cause |
|---|---|
| `artifact` | artifact dir unreadable / malformed |
| `artifact_revision` | `runtime.ArtifactID() != config artifact_revision` |
| `artifact_encoder_contract` | model / revision / dim 1024 / serializer disagree |
| `artifact_arm_mapping` | `modelRefs` not equal, in order, to manifest worker ids |
| `artifact_dispatch_contract` | **pricing, model id, endpoint, auth shape, or thinking mode mismatch** (§1.6a) |
| `encoder_auth` | `RAYLINE_ARC_MODAL_KEY`/`_SECRET` unset or empty |
| `encoder_config` | only one of the Modal credentials supplied; bad timeouts/rung |
| `encoder_probe` | live encoder unreachable, or its identity (engine build id, tokenizer sha, io plugin) disagrees |
| `episode_store` | Redis unreachable or password env unset |
| `conflicting_config` | two ARC decisions with differing `rayline_arc` blocks |

**(d) The encoder is reachable through the proxy token:**

```bash
bash e2e/testing/rayline-arc/live_session_ops.sh encoder-health
# {"status":"ok","resident_sessions":0,"resident_tokens":0,
#  "max_sessions":8,"max_resident_tokens":2097152,"pooling_capabilities":[...]}
```

`wait_protected_encoder` (`openrouter_encoder_runtime.py:184-200`) requires
exactly `status == "ok"` and `resident_sessions == 0`. A 401 here means the
proxy token is wrong or already deleted; a 403/404 means the app moved.

**(e) Envoy is listening:**

```bash
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:18888/nope   # 503
```

A `503` from the catch-all `direct_response` (`envoy-openrouter.yaml:67-70`) is
the *correct* answer here and proves Envoy is up.

**(f) One end-to-end smoke request** (costs a few thousandths of a cent):

```bash
curl -sS http://127.0.0.1:18888/v1/chat/completions \
  -H 'content-type: application/json' \
  -H "x-claude-code-session-id: $(uuidgen | tr 'A-Z' 'a-z')" \
  -D /tmp/rayline-live/smoke.headers \
  -d '{"model":"auto","messages":[{"role":"user","content":"ping"}],"max_tokens":16}' \
  | head -c 400
grep -i 'x-vsr-selected-model\|x-envoy-attempt-count' /tmp/rayline-live/smoke.headers
```

Expect HTTP 200, a real completion, and `x-vsr-selected-model: worker-a|b|c`.
Then confirm the decision record exists:

```bash
grep -c rayline_arc_selection /tmp/rayline-live/router.log   # expect >= 1
```

If you used the §4.0 fallback instead, send `x-rayline-episode-id` here rather
than `x-claude-code-session-id`.

---

## 5. Pointing `claude` at it

### 5.1 What the CLI sends

Verified empirically against a local header-recording server (no paid calls),
`claude` 2.1.226:

```
POST /v1/messages?beta=true
x-claude-code-session-id: 11111111-2222-4333-8444-555555555555
x-app: cli
user-agent: claude-cli/2.1.226 (external, sdk-cli)
anthropic-version: 2023-06-01
anthropic-beta: claude-code-20250219,context-1m-2025-08-07,...
authorization: Bearer <ANTHROPIC_AUTH_TOKEN>
```

The stack accepts this. `detectClientProtocol`
(`processor_req_header_endpoints.go:141`) tags the request Anthropic purely from
the `/v1/messages` path prefix; `normalizeRequestPath` strips `?beta=true`;
`raylinearc.NormalizeTurns` handles `ProtocolAnthropicMessages`
(`rayline_arc_context.go:161-162`); the router rewrites `:path` to
`/api/v1/chat/completions` from the profile
(`processor_req_body_routing.go:538-545` + `config/helper.go:769-780`) and sets
`x-selected-model`, then clears the route cache (`ClearRouteCache` defaults
`true`, `canonical_defaults.go:21`) so Envoy re-routes onto the matching
`prefix: /api/v1/` route. Responses are translated back to Anthropic JSON
(`processor_res_body_pipeline.go:79-86`) and Anthropic SSE
(`processor_res_body_streaming_anthropic_client.go`).

### 5.2 The model name must be an auto alias

`handleSpecifiedModelRouting` rejects unconfigured model names with
**400 `model "…" is not available`** (`processor_req_body.go:428-435`), and ARC
selection is skipped entirely for any non-auto name
(`req_filter_classification_runtime.go:212-219`). The accepted set is
`{"vllm-sr/auto", "auto", "MoM"}` (`config/helper.go:11-15,88-107`); the agentic
config sets no `auto_model_names` override.

So `ANTHROPIC_MODEL=auto` is **mandatory**, and the small/fast tier must be
overridden too or a background haiku call will 400.

### 5.3 Multi-turn session continuity — the crux

`X-Claude-Code-Session-Id` is read from a **process-global getter**, not
per-request, and is only mutated by the setter on `startup_custom_id`, `resume`,
`remote_attach`, `hydrate`, `cd`, `spare_claim`, and `clear`. It is therefore
identical on every request a session makes, including subagent and small-model
calls.

`--session-id` is **not** the flag for turns 2+. The binary refuses to reuse it:
`yPn(id)` is a `statSync` on `<projectDir>/<id>.jsonl`, and turn 1 creates that
file, so a second `--session-id` with the same UUID exits with
`Error: Session ID … is already in use.` **before sending any request**.

Empirically confirmed:

| Invocation | header sent |
|---|---|
| `-p --session-id 1111…5555` | `1111…5555` |
| `-p --resume 1111…5555` | **`1111…5555` — same** |
| `-p --continue` (same cwd) | **`1111…5555` — same** |
| `-p --session-id 1111…5555` again | hard error, zero requests |
| `-p --resume 1111…5555 --fork-session` | **new UUID** — do not use |

**The working multi-turn recipe:**

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:18888
export ANTHROPIC_AUTH_TOKEN=rayline-live-local        # value is ignored by the router
export ANTHROPIC_MODEL=auto
export ANTHROPIC_DEFAULT_HAIKU_MODEL=auto
export ANTHROPIC_SMALL_FAST_MODEL=auto                # legacy name; takes precedence
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export DISABLE_TELEMETRY=1 DISABLE_ERROR_REPORTING=1
export DISABLE_AUTOUPDATER=1 DISABLE_BUG_COMMAND=1

EPISODE=$(uuidgen | tr 'A-Z' 'a-z')
echo "episode/session uuid: $EPISODE" | tee -a /tmp/rayline-live/session.txt

run_turn() {  # run_turn <first|next> <prompt>
  local mode="$1"; shift
  local sess=(--resume "$EPISODE")
  [ "$mode" = first ] && sess=(--session-id "$EPISODE")
  env -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT -u CLAUDE_PID \
      -u CLAUDE_CODE_SESSION_ID -u CLAUDE_CODE_CHILD_SESSION \
      -u ANTHROPIC_API_KEY -u CLAUDE_CODE_OAUTH_TOKEN \
      CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=1 \
    ~/.local/bin/claude -p "$*" "${sess[@]}" --output-format json
}

run_turn first "Explain what a Rayline ARC episode is in two sentences."
run_turn next  "Now contrast it with an HTTP session."
run_turn next  "Which of those two is stateful on the encoder, and why?"
```

> **Without a stable session header across turns, the retained-KV path is never
> exercised and the run proves almost nothing.** Every turn would open a fresh
> episode, the encoder would re-serialize the whole history from scratch, and
> `session_action` in the decision log would never show a resume.

Two traps that silently break persistence:

- **Nesting.** If `CLAUDE_CODE_CHILD_SESSION` is inherited (i.e. you launch
  `claude` from inside another `claude`), session persistence is disabled
  entirely — no JSONL, so no `--resume`. The `env -u` list above scrubs it;
  `CLAUDE_CODE_FORCE_SESSION_PERSISTENCE=1` is the belt-and-braces.
- **`--no-session-persistence`** makes resume impossible. Never combine it.

`--continue` also preserves the id but resolves the most recent session in the
current directory. Prefer explicit `--resume "$EPISODE"`.

### 5.4 Proving continuity at run time

```bash
# every turn must carry the same episode hash
grep -o '"episode_id_hash":"[^"]*"' /tmp/rayline-live/router.log | sort | uniq -c
# expect ONE hash with a count equal to the number of turns

# and the encoder must report a resumed session, not a fresh one
grep -o '"session_action":"[^"]*"' /tmp/rayline-live/router.log | sort | uniq -c
```

The on-disk session file is also named exactly the UUID and every line carries
`sessionId`:

```bash
ls ~/.claude/projects/*/${EPISODE}.jsonl
```

---

## 6. Observability capture — set this up BEFORE the run

Per-decision Q-values appear in exactly **one** place:
`logging.ComponentEvent("extproc", "rayline_arc_selection", …)` at
`req_filter_classification.go:322-349`, carrying `raw_scores`,
`adjusted_scores`, `selected_arm`, `previous_arm`, `episode_id_hash`,
`switch_cost_usd`, `cache_miss_tokens`, `stayed`, the six token counts,
`session_action`, `session_revision`, and the encoder replica/latency fields.

It is a zap **info**-level JSON line on **stderr**
(`zap.NewProductionConfig()`, `logging/logging.go:54`; level default `info` from
`SR_LOG_LEVEL`, `logging.go:136`). No flag or config field enables it — but
setting `SR_LOG_LEVEL=warn` or higher would silence it.

> **This log line is the only decision record.** The selected arm is
> deliberately absent from Prometheus and Grafana — ARC skips the generic
> `RecordSelection` metric because model ids as label values would leak artifact
> arm identity (`req_filter_classification.go:157-159`). Router Replay is
> config-forbidden for ARC (`rayline_arc_config.go:393-399`), so there is no
> request archive either. **If the container's log rotates, the run's decisions
> are gone.** Tee from t=0.

```bash
mkdir -p /tmp/rayline-live

# 1. Full router stderr, from the first line, for the whole run.
#    `logs -f` replays from the start, so launching it right after `up -d`
#    loses nothing.
compose logs --follow --no-color --timestamps router \
  > /tmp/rayline-live/router.log 2>&1 &
LOG_PID=$!

# 2. A live view of just the decisions.
tail -F /tmp/rayline-live/router.log | grep --line-buffered rayline_arc_selection &

# 3. Metrics, sampled every 15s, appended with a timestamp.
( while sleep 15; do
    printf '\n# %s\n' "$(date -u +%FT%TZ)"
    curl -sS http://127.0.0.1:19190/metrics | grep '^llm_rayline_arc'
  done ) >> /tmp/rayline-live/metrics.log 2>&1 &
METRICS_PID=$!
```

Metrics worth watching (all from
`pkg/observability/metrics/rayline_arc_metrics.go` and the compose README):

```
llm_rayline_arc_component_ready{component="artifact_head_encoder"|"episode_store"}
llm_rayline_arc_selection_failures_total{class}
llm_rayline_arc_encoder_latency_seconds
llm_rayline_arc_tokens{kind}
llm_rayline_arc_switch_cost_usd
llm_rayline_arc_cache_miss_tokens
llm_rayline_arc_episode_transactions_total{outcome,failure_class}
llm_rayline_arc_session_actions_total{action}
llm_rayline_arc_encoder_session_closes_total{outcome}
llm_rayline_arc_provider_logical_requests_total{outcome}
llm_rayline_arc_provider_attempts_total{outcome}
llm_rayline_arc_provider_retries_total{outcome}
llm_rayline_arc_provider_retry_exhaustions_total{status}
```

After the run, take a final snapshot **before** `compose down` (the metrics
endpoint dies with the container) and read the real spend:

```bash
curl -sS http://127.0.0.1:19190/metrics > /tmp/rayline-live/metrics-final.txt
compose logs --no-color --timestamps router > /tmp/rayline-live/router-final.log

bash e2e/testing/rayline-arc/live_session_ops.sh key-usage "$OPENROUTER_EPHEMERAL_KEY_HASH"
# -> settled USD, e.g. 0.18342100
```

Extract the decision table:

```bash
grep rayline_arc_selection /tmp/rayline-live/router-final.log \
  | sed 's/^[^{]*//' \
  | ~/code/pathfinder-rayline-vsr-mvp/.venv/bin/python -c '
import json,sys
for line in sys.stdin:
    try: r = json.loads(line)
    except ValueError: continue
    print(r["episode_id_hash"][:12], r["selected_arm"], r["previous_arm"],
          r["session_action"], r["appended_tokens"], r["retained_prefix_tokens"],
          [round(s,4) for s in r["adjusted_scores"]])
'
```

---

## 7. Teardown and cost stop

Ordered by burn rate. Each step is independent — a failure in one must not skip
the next.

```bash
kill $LOG_PID $METRICS_PID 2>/dev/null || true

# 1. Release the H100 floor. THIS IS THE ONE THAT MATTERS.
bash e2e/testing/rayline-arc/live_session_ops.sh encoder-unpin

# 2. Stop the running container so it does not idle out the 300s window on you.
bash e2e/testing/rayline-arc/live_session_ops.sh encoder-stop

# 3. Delete the Modal proxy token.
bash e2e/testing/rayline-arc/live_session_ops.sh token-delete "$RAYLINE_ARC_E2E_MODAL_KEY"

# 4. Read spend, THEN delete the OpenRouter key (deletion loses the usage read).
bash e2e/testing/rayline-arc/live_session_ops.sh key-usage  "$OPENROUTER_EPHEMERAL_KEY_HASH"
bash e2e/testing/rayline-arc/live_session_ops.sh key-delete "$OPENROUTER_EPHEMERAL_KEY_HASH"

# 5. Local stack.
compose down --volumes --remove-orphans
```

> **Do not run `modal app stop rayline-arc-session-encoder`.** The default app
> is the shared, non-ephemeral deployment the frozen benchmarks resolve by name;
> stopping it undeploys it. `cleanup_encoder`
> (`openrouter_encoder_runtime.py:307-337`) only calls `app stop` when
> `encoder.ephemeral` is true, which the default deployment is not. Unpin plus
> container stop is the correct stop for this runbook.

### 7.1 Verification — prove each one actually stopped

```bash
bash e2e/testing/rayline-arc/live_session_ops.sh verify-stopped
```

which is equivalent to, and can be done by hand as:

```bash
# encoder containers: expect no output
bash e2e/testing/rayline-arc/live_session_ops.sh encoder-containers

# autoscaler floor: expect min_containers=0 echoed by a repeat unpin (idempotent)
bash e2e/testing/rayline-arc/live_session_ops.sh encoder-unpin

# proxy token: expect the encoder to now refuse the old credentials
bash e2e/testing/rayline-arc/live_session_ops.sh encoder-health   # expect failure / 401

# OpenRouter key: expect the usage read to fail (key no longer exists)
bash e2e/testing/rayline-arc/live_session_ops.sh key-usage "$OPENROUTER_EPHEMERAL_KEY_HASH"
# expect: RuntimeError: OpenRouter management request failed

# compose: expect no output
compose ps --quiet
```

### 7.2 Panic stop

If anything goes wrong mid-run, this halts all spend in cost order, best-effort,
never blocking on a failure:

```bash
bash e2e/testing/rayline-arc/live_session_ops.sh panic-stop \
  "$OPENROUTER_EPHEMERAL_KEY_HASH" "$RAYLINE_ARC_E2E_MODAL_KEY"
```

Both arguments default to `$OPENROUTER_EPHEMERAL_KEY_HASH` and
`$RAYLINE_ARC_E2E_MODAL_KEY`, so bare `panic-stop` works if those are exported.
Follow it with `verify-stopped`. If the unpin step reports a failure, open
<https://modal.com/apps> immediately and stop the container from the UI — that
is the only cost still accruing.

---

## 8. Failure modes: symptom → cause → fix

### 8.1 `missing_episode_id` fail-closed

**Symptom.** Every request returns HTTP 503 with

```json
{"error":{"message":"Rayline ARC routing unavailable","type":"invalid_request_error","code":503}}
```

(note `type` is `invalid_request_error` even on a 503 — `router.go:268-285`),
and the log shows

```json
{"level":"error","msg":"rayline_arc_selection_failed","failure_class":"missing_episode_id"}
```

with `llm_rayline_arc_selection_failures_total{class="missing_episode_id"}`
incrementing.

**Cause.** `rayline_arc_context.go:45-52` found the configured `id_header`
absent or blank. With `claude`, this is almost always §4.0 not being applied —
the CLI sends `x-claude-code-session-id`, the shipped config wants
`x-rayline-episode-id`, and nothing bridges them.

**Fix.** Apply §4.0 (config copy) or the `ANTHROPIC_CUSTOM_HEADERS` fallback, and
recreate the router:

```bash
compose up --detach --no-deps --force-recreate router
```

The same 503 covers the whole bounded failure-class set — always read
`failure_class` before acting: `invalid_close_signal`, `episode_store`,
`episode_timeout`, `episode_capacity`, `turns_*`, `encoder_*`, `not_ready`,
`candidate_count`, `candidate_order`, `policy`, `artifact_result`.

### 8.2 Envoy `direct_response: 503` on an unknown `x-selected-model`

**Symptom.** An immediate, bodiless 503 with **no** router log line for the
request and **no** `x-envoy-upstream-service-time` header. Nothing appears in
`rayline_arc_selection`.

**Cause.** `envoy-openrouter.yaml:29-70` has exactly three routes, matching
`prefix: /api/v1/` with `x-selected-model` exactly `worker-a`, `worker-b`, or
`worker-c`; everything else hits the catch-all
`direct_response: {status: 503}` at `:67-70`. So this fires when the router
selected a model name that is not one of those three, or when the router never
ran at all (ext_proc down) and the original `/v1/messages` path fell through.

**Fix.** Confirm the decision's `modelRefs` names are exactly
`worker-a/b/c` in the live config, and that the router container is up and
reachable from Envoy on `router:50051`. Distinguish the two cases by whether
`compose logs router` shows *any* activity for the request. Note a bare
`curl http://127.0.0.1:18888/nope` returning 503 is expected and healthy (§4.3e).

### 8.3 Encoder 429 `session_capacity` on the 9th concurrent episode

**Symptom.** HTTP 503 to the client, `failure_class="encoder_status"`, and
`llm_rayline_arc_selection_failures_total{class="encoder_status"}` incrementing.
The encoder's own 429 is **not** propagated.

**Cause.** `modal_session_service.py:96` sets `MAX_SESSIONS = 8`;
`session_coordinator.py:158-159` raises `SessionCapacityError` at the 9th
resident session, which `session_api.py:294-298` maps to
`HTTP 429 detail="session_capacity"`. On the Go side there is no special 429
handling: any non-2xx becomes `encoderStatusFailure("http_status", …)`
(`encoder_client.go:575`), which is a `*EncoderFailure` and therefore
short-circuits the retry loop with `retry == false`
(`encoder_client.go:511-532`). And `429` is **not** in the failover set
`[404, 410, 502, 503, 504]`, so no replica remap happens either — correctly, but
it means the request simply dies as a 503.

**Fix.** Run one conversation at a time. Sessions are released by the encoder's
`IDLE_TTL_SECONDS = 300`. To force a release, stop the encoder container and let
the pin restart it (costs one cold start). Do not raise `MAX_SESSIONS` — it is
also `max_num_seqs` on the vLLM engine (`modal_session_service.py:257`) and the
resident-token budget is derived from it.

Note: with the agentic config there is exactly one encoder and no `failover`
block at all, so the failover status set is inert here. It only applies to the
replica topology in `deploy/compose/rayline-arc/config.yaml:87`.

### 8.4 Readiness fails closed on a pricing mismatch

**Symptom.** Router starts, `/health` and `/ready` are 200, but
`llm_rayline_arc_component_ready{component="artifact_head_encoder"} 0` and the
startup log shows `"failure_class":"artifact_dispatch_contract"`. Every request
503s with `failure_class="not_ready"`.

**Cause.** `raylineARCPriceIdentityMatches` (`rayline_arc_readiness.go:299-333`)
compares the config's four `pricing.*_per_1m` values against the manifest's
per-token costs × 1e6 with tolerance `max(1e-12, |artifact| × 1e-9)`. Any leg
off — including `cached_input_per_1m` and `cache_write_per_1m`, which are easy to
forget — fails. `artifact_dispatch_contract` also covers a `provider_model_id`
that disagrees with the manifest's `model`, a non-`openai` endpoint type, a
non-`Bearer`/non-`Authorization` auth shape, a non-empty `chat_path`, an
`api_key_env` that disagrees with the manifest's, and a `use_reasoning` that
disagrees with `thinking_mode`.

**Fix.** Reconcile config and fixture per §1.6a, then
`compose up --build --detach --force-recreate` (the `--build` is needed to
re-bake the fixture) and re-check the gauge. Readiness is computed **once at
startup** — editing the config without recreating the router changes nothing.

### 8.5 Cold start blows the client timeout

**Symptom.** The first request after a think-gap longer than ~5 minutes hangs
and then fails. `llm_rayline_arc_encoder_latency_seconds` shows a bucket at
60-120s.

**Cause.** `scaledown_window=300` (`modal_session_service.py:204`). A cold H100
boot is measured at **78.9s–96.9s**. The config allows
`total_timeout_seconds: 180` and `max_retries: 0`
(`config-openrouter-agentic.yaml:116-118`), and Envoy's route timeout is 180s
(`envoy-openrouter.yaml:38`), so a slow cold start plus real generation can
exceed the budget. The client-visible class is `encoder_timeout` (or
`encoder_transport`).

**Fix.** Keep the pin on for the whole session (§3) — that is precisely what it
buys. If you must recover mid-session, warm it explicitly before the next turn:

```bash
time bash e2e/testing/rayline-arc/live_session_ops.sh encoder-health
```

and only send the turn once that returns in well under a second. Do not raise
`total_timeout_seconds` in the live config: it is part of the encoder client
contract and drifting it makes the run non-comparable to every recorded arm.

### 8.6 HTTP 402 from OpenRouter mid-session

**Symptom.** Requests start failing with a provider 402 while
`key-usage` reports far less than the limit.

**Cause.** OpenRouter's limit check counts **in-flight pre-authorization holds**,
not settled cost (`openrouter_kv_cache_successor_contract.py:44-51`). AGT018 hit
this at $0.025 settled against a $0.05 limit.

**Fix.** There is no mid-run remedy — the key's limit is immutable. Stop, mint a
new key with a higher limit, and restart the router with the new
`RAYLINE_ARC_E2E_PROVIDER_KEY`. Prevent it by sizing the limit per §2.2.

### 8.7 `400 model "…" is not available`

**Symptom.** `claude` reports an API error on the very first turn; the router
returns 400.

**Cause.** `ANTHROPIC_MODEL` was not set to an auto alias, so the CLI sent a real
Claude model name, which `processor_req_body.go:428-435` rejects.

**Fix.** Export `ANTHROPIC_MODEL=auto` **and**
`ANTHROPIC_DEFAULT_HAIKU_MODEL=auto` / `ANTHROPIC_SMALL_FAST_MODEL=auto` (§5.2).
A 400 that appears only occasionally, mid-session, is the small/fast tier.

### 8.8 `404` on `/v1/messages/count_tokens`

**Symptom.** Occasional 404s in the Envoy/router log for
`/v1/messages/count_tokens`.

**Cause.** `processor_req_header_validation.go:9-40` admits only the exact
`/v1/messages`; anything else under `/v1/` returns
`404 endpoint not found`.

**Fix.** Nothing to fix in the stack. **CONFIRM AT RUN TIME** whether 2.1.226
calls it and whether it degrades gracefully (it is expected to fall back to
local token estimation). If it turns out to be fatal, front the stack with a
tiny shim that answers `count_tokens` locally.

---

## 9. Quick reference

```
Envoy ingress          http://127.0.0.1:18888
Router API             http://127.0.0.1:18082/health  /ready  /startup-status
Router metrics         http://127.0.0.1:19190/metrics
Redis                  127.0.0.1:16379
Encoder                https://atlasfutures-dev--rayline-arc-session-encoder-sessionenc-2d82ac.modal.run
  GET  /health                                   (Modal-Key / Modal-Secret)
  GET  /v1/rayline/arc/session/metrics
  POST /v1/rayline/arc/session/pooling
  DEL  /v1/rayline/arc/session/{episode_id_hash}
Compose project        rayline-arc-live-claude
Decision log event     rayline_arc_selection            (info, stderr, JSON)
Readiness log event    rayline_arc_component_readiness  (info ok / error not-ok)
Failure log event      rayline_arc_selection_failed     (error)
Panic stop             live_session_ops.sh panic-stop
```

---

## 10. CONFIRM AT RUN TIME

Everything below could not be settled from source and must be checked live.

1. **The encoder deploy succeeds and reports the bare build id.** The app is not
   currently deployed; the image builds vLLM from source. Confirm
   `encoder-health` returns `status: ok` and that readiness does not fail with
   `encoder_probe`, which would mean the reported `EngineBuildID`,
   tokenizer sha, or io-plugin version diverged from the config's expectations.
2. **Whether the redeployed web URL keeps the frozen host.** Modal derives it
   from workspace+environment+app+class so it should be unchanged, but the new
   app id is definitely different. Always read it from `encoder-url`.
3. **Whether `claude` 2.1.226 calls `/v1/messages/count_tokens`, and whether a
   404 there is fatal or degrades to local estimation.** A single-turn `-p` run
   against a fake server showed only `/v1/messages?beta=true`, but that run had
   no tools and one turn.
4. **Whether a small/fast-model request appears at all.** None was observed in a
   minimal `-p` run even with non-essential traffic enabled; title generation
   looks interactive-only. Set the haiku overrides anyway.
5. **Whether the three OpenRouter models handle Claude Code's tool schemas.**
   `tools` / `tool_choice` are passed through `materializeARCTools`; whether
   `deepseek/deepseek-v4-flash`, the worker-b model, and `tencent/hy3` produce
   usable tool calls at these context sizes is untested here.
6. **Actual settled spend for an interactive session.** No receipt of this shape
   exists — every recorded arm ran with the 96-token cap. Read `key-usage` and
   record it; that number is what should replace the §2.2 estimate next time.
7. **Whether a single episode stays inside the encoder's 262,144-token resident
   budget** over a long Claude Code conversation. `MAX_SERIALIZED_TOKENS` is
   per-session; exceeding it raises a capacity error, surfacing as
   `encoder_status` → 503.
8. **Whether the router's `x-api-key`/`authorization` passthrough matters.** On
   the OpenAI-upstream path the client's inbound auth headers are not stripped,
   and the non-ARC injection uses Envoy's default append action. With ARC armed
   the credential is written with `OVERWRITE_IF_EXISTS_OR_ADD`
   (`processor_req_body_routing.go:408-437`), so it should be fine — verify no
   duplicate `authorization` reaches OpenRouter if you see 401s.
