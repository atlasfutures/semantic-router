# Rayline ARC

## Overview

`rayline_arc` is an experimental, artifact-verified selection algorithm for a
switch-aware orchestrator. A dedicated vLLM pooling deployment performs the
encoder inference; Semantic Router verifies and executes the small F32 policy
head, applies cache-loss and stay-margin policy, and owns transactional episode
state.

It aligns to `config/algorithm/selection/rayline-arc.yaml`.

The public implementation is schema-generic. Arm identities, provider/model
bindings, prices, and private goldens come only from an immutable mounted
runtime artifact and must not be copied into source configuration.

## Key Advantages

- Keeps the SLM encoder on vLLM while the router owns deterministic policy.
- Verifies artifact hashes, tensor shapes, arm order, and startup goldens.
- Prices switches using the artifact's immutable cache-aware cost snapshot.
- Serializes same-episode decisions and commits state only after successful
  upstream response headers.
- Fails closed on artifact, encoder, policy, or state errors.

## What Problem Does It Solve?

Ordinary prompt routers select each request independently. ARC routes a
multi-turn episode with a learned history representation while accounting for
the quality advantage of another arm, the cost of losing a warm provider KV
cache, and a stable stay margin. The split architecture keeps heavy encoder
inference in vLLM and keeps the small auditable decision policy in Semantic
Router.

## When to Use

Use ARC only for a frozen, compatible `rayline.mtrouter-runtime.v3` artifact
whose logical arms can be mapped exactly to configured providers. Keep it
experimental until CPU, Redis, CUDA, dispatch, privacy, and full-stack gates
pass for the exact artifact and vLLM build. Other selectors are a better fit
when requests are independent or an immutable orchestrator artifact is not
available.

## Contract

ARC is deliberately stricter than other selectors:

- `on_error` must be `fail_closed`; selection errors never choose the first
  candidate.
- `adaptations.mode` must be `bypass`; Router Learning cannot replace the ARC
  decision.
- `modelRefs` must be unique, remain in artifact arm order, and must not
  collide with an auto-routing alias. Startup rejects any mismatch with the
  mounted manifest, including a configured endpoint whose credential does not
  come from exactly the worker's declared `api_key_env`.
- Router Replay must be disabled for ARC decisions (a decision-level
  `router_replay` plugin with `enabled: false` when replay is globally on);
  episode requests are never persisted.
- The encoder model and revision are frozen. The vLLM build, IO plugin,
  serializer, and required serving capabilities are pinned and checked by
  readiness.
- Redis is the durable episode backend. `memory` requires
  `development_mode: true`, a positive bound, and sticky single-replica use.
- Redis passwords are named through `password_env`; the canonical config never
  contains the credential value.
- Retained-session deployments choose exactly one `encoder.base_url` or a
  static, versioned `encoder.replicas` set. Replica mode requires an explicit
  final-turn close header and disables per-client retries.

## Configuration

```yaml
routing:
  decisions:
    - name: arc-route
      rules:
        operator: AND
        conditions:
          - type: domain
            name: business
      modelRefs:
        # Replace these public placeholders with the artifact-declared logical
        # arms, in exact manifest order, in private deployment configuration.
        - model: public-arm-a
          use_reasoning: false
        - model: public-arm-b
          use_reasoning: true
      adaptations:
        mode: bypass
      algorithm:
        type: rayline_arc
        on_error: fail_closed
        rayline_arc:
          artifact_dir: /var/lib/vllm-sr/rayline-arc
          artifact_revision: ${RAYLINE_ARC_ARTIFACT_REVISION}
          encoder:
            base_url: http://rayline-arc-encoder:8000
            model: Qwen/Qwen3.5-0.8B
            model_revision: 2fc06364715b967f1860aea9cf38778875588b17
            expected_build_id: ${RAYLINE_ARC_VLLM_BUILD_ID}
            expected_io_plugin_version: rayline-arc-io@0.1.0
            serializer_version: mtrouter-token-blocks-v2
            serving_rung: B
            required_pooling_capabilities:
              - chunked_causal_mean
            modal_key_env: RAYLINE_ARC_MODAL_KEY
            modal_secret_env: RAYLINE_ARC_MODAL_SECRET
            connect_timeout_seconds: 5
            total_timeout_seconds: 180
            max_retries: 1
          episode:
            id_header: x-rayline-episode-id
            backend: redis
            key_prefix: "vsr:rayline-arc:"
            acquire_timeout_seconds: 30
            lease_ttl_seconds: 60
            idle_ttl_seconds: 900
            max_in_memory_episodes: 1024
            redis:
              address: redis:6379
              password_env: RAYLINE_ARC_REDIS_PASSWORD
              pool_size: 16
```

`total_timeout_seconds` is the complete encoder request budget across retries,
not a per-attempt timeout. Freeze the production value from the maximum-context
GPU canary; do not assume the example is an adequate production threshold.

`serving_rung: B` selects vLLM's in-engine causal MEAN path and requires
`chunked_causal_mean`. Rung A's `all_plugin_mean` remains a diagnostic
bootstrap and is not the production maximum-context serving shape.

The stateless comparison arm reports only `chunked_causal_mean` and uses
vLLM's standard `/pooling` wire. To select the explicit retained-session wire,
require both capabilities:

```yaml
required_pooling_capabilities:
  - chunked_causal_mean
  - resumable_causal_mean
```

That mode sends the complete reconstructible history directly to
`/v1/rayline/arc/session/pooling`. Exact token extensions compute only their
suffix; retry, mismatch, eviction, affinity loss, and restart remain correct
because the session is optional acceleration state. Automatic prefix caching
stays disabled: `resumable_causal_mean` describes a pinned live request, not a
vLLM prefix-cache hit.

### Concurrent turns on one episode

`episode.consistency` chooses how a cell handles two turns of one conversation
in flight at once. It is a per-cell choice made at deployment, never a request
header.

| Value | Behaviour | Use for |
|---|---|---|
| `strict` (default) | Turns on one episode take an exclusive lease and run one at a time. A turn that cannot take the lease within `acquire_timeout_seconds` is refused with 429 and `x-vsr-failure-class: session_busy`. Every turn is decided from the state the previous one committed. | Eval and training cells, where each reply must be on-policy and correctly attributed. |
| `relaxed` | A turn reads the episode without a lease and commits only if no other turn committed in between. A turn never waits and is never refused over episode state: a lost race or an unavailable store only drops that turn's state update, counted as `relaxed_dropped`. A relaxed cell also starts while its store is down, and decides statelessly until the store is back. | Serving cells, where availability matters more than exact continuity. |

In either mode, an identical resend of a turn still being decided joins that
turn's decision instead of competing with it.

A relaxed turn whose decision takes longer than `idle_ttl_seconds` also drops
its update: by then the episode it read could have expired and been recreated.
Relaxed episode state is best effort. In one narrow race at that boundary, a
stale first turn can still land on an episode that has just expired from
disuse; no live state is lost when it does.

`relaxed` does not yet accept `close_header` or the `resumable_causal_mean`
encoder capability (a retained encoder session serializes same-episode
encodes).

In the policy-service mode, a relaxed cell sends `episode_mode: relaxed` on
each decide call, so the service takes no session lock and keeps no session
state for it: concurrent turns on one episode are all decided, each with a
full encode. Not every policy service can do that (a package on the vLLM
encoder, or pinned to a runtime without unretained prediction, answers
`unsupported_request`). A relaxed policy cell therefore asks for one relaxed
decision before it arms, and stays not ready, logging
`rayline_arc_policy_relaxed_unsupported`, until the service serves one.

### Static retained-encoder replicas

For one to eight independent vLLM retained-session services, replace
`encoder.base_url` with a static membership block. All replicas share the same
model, build, plugin, serializer, capabilities, timeout, and optional Modal
credential shape:

```yaml
encoder:
  replicas:
    - id: encoder-a
      base_url: http://rayline-arc-encoder-a:8000
      state: active
    - id: encoder-b
      base_url: http://rayline-arc-encoder-b:8000
      state: active
  failover:
    schema_version: rayline.arc.encoder-failover.v1
    unavailable_status_codes: [404, 410, 502, 503, 504]
    unavailable_cooldown_seconds: 30
    max_remaps: 1
  model: Qwen/Qwen3.5-0.8B
  model_revision: 2fc06364715b967f1860aea9cf38778875588b17
  expected_build_id: ${RAYLINE_ARC_VLLM_BUILD_ID}
  expected_io_plugin_version: rayline-arc-io@0.1.0
  serializer_version: mtrouter-token-blocks-v2
  serving_rung: B
  required_pooling_capabilities:
    - chunked_causal_mean
    - resumable_causal_mean
  connect_timeout_seconds: 5
  total_timeout_seconds: 180
  max_retries: 0
episode:
  id_header: x-rayline-episode-id
  close_header: x-rayline-episode-close
  # Keep the remaining Redis lease/TTL fields from the complete example.
```

New episodes use deterministic rendezvous placement across `active` members.
An existing episode stays on its persisted owner even when that member is
`draining`. Only an explicitly configured HTTP status can trigger one remap;
transport, timeout, decode, and identity failures fail closed without calling
a peer. Treat the status list as a deployment assertion that those responses
occur before retained mutation, not as a generic retry list.

The configured close header accepts exact `true` on a final request. After the
provider returns 2xx, Semantic Router concurrently closes every visited
encoder owner. Clean close clears encoder affinity but retains ARC policy
history. For rolling replacement, first change a member from `active` to
`draining`, wait at least `episode.idle_ttl_seconds` after its last admitted
owner while watching close/session metrics, and only then remove it. Premature
removal fails closed.

### Dynamic retained-encoder membership

Dynamic membership replaces `encoder.replicas` with a reviewed Redis document;
it does not change the v1 request, affinity, remap, or close contract.

```yaml
encoder:
  membership:
    schema_version: rayline.arc.encoder-membership.v1
    source: redis
    refresh_seconds: 5
  failover:
    schema_version: rayline.arc.encoder-failover.v1
    unavailable_status_codes: [503]
    unavailable_cooldown_seconds: 30
    max_remaps: 1
```

The controller stores a revisioned membership document at
`<episode.key_prefix>encoder-membership`. It must publish `active` to
`draining` first, then wait one full `episode.idle_ttl_seconds` and prove that
the aggregate persisted owner/visited-owner count is zero before its CAS
removal revision. A router accepts only ordered revisions and pins a stable
replica ID to its original endpoint. If the source is missing at startup, the
router fails closed; if it fails after startup, the router retains the last
valid snapshot.

Run membership mutation in the separate controller image, never in the router
process. The controller reads the same canonical config and supports
privacy-safe `status`, one-shot `drain`/`reconcile`, and a continuous `run`
loop:

```bash
rayline-arc-controller status --config /app/config.yaml --decision arc-prod
rayline-arc-controller register --config /app/config.yaml \
  --decision arc-prod --replica-id encoder-c \
  --base-url https://encoder-c.internal
rayline-arc-controller drain --config /app/config.yaml \
  --decision arc-prod --replica-id encoder-a
rayline-arc-controller run --config /app/config.yaml \
  --decision arc-prod --interval 5s
```

`register` is idempotent only for the exact active ID/endpoint pair. It rejects
endpoint changes and draining-ID reactivation. Each router probes a newly
registered endpoint before adopting that membership revision; a failed probe
leaves its last valid snapshot in service.

Set `RAYLINE_ARC_CONTROLLER_PASSWORD_ENV` to the name of a controller-only
Redis password variable when the controller uses credentials distinct from
the router, and set `RAYLINE_ARC_CONTROLLER_REDIS_USERNAME` for a named Redis
ACL user. The router identity needs read-only access to the membership key;
the controller identity needs membership CAS authority plus read/scan access
to episode ownership fields.

See [Rayline ARC Retained-Encoder Replica Contract](../../../../../tools/agent/docs/architecture/rayline-arc-replica-membership.md)
for the full interaction, failure, observability, and rollout model.

Modal proxy authentication is configured by environment-variable name, never
by embedding credentials in YAML. Configure `modal_key_env` and
`modal_secret_env` together for a protected Modal web endpoint, or omit both
for an internal endpoint that does not use Modal proxy authentication.

## Tool names as a routing signal

`include_tool_names` folds the turn's available tool names into the first user
turn, where the opening system prompt would go:

```text
[available tools] Ledger, Scribe, Amend, Lantern, Anvil

rename the helper in the cache module
```

Names only, in the order the caller declared them, and never the schemas. A
2026-09-17 encoder probe measured all three renderings over 200 episodes
against the frozen head:

| rendering | tokens | cosine p10 | first-turn decisions changed |
|---|---|---|---|
| names | 42 | .996 | 7.6% |
| names with descriptions | 225 | .984 | 13.8% |
| full JSON schemas | 3,140 | .820 | 40.4% |

The schema block lands level with the bar that keeps `include_system_text`
off, and with the same signature: cross-episode similarity at the first
boundary rises from .75 to .98 as the shared prefix drowns the per-episode
signal. Tool definitions sit on the same dose-response curve as any other
shared prefix, so sending the contract destroys the signal it was meant to
add, while sending the names does not.

Off by default. The probe established that names are safe to send, not that
they improve routing: those 7.6% are decisions changing with no evidence they
changed for the better, and the selector has never been trained with tool
names. An eval against the current router decides that, not this probe.

Turning it on changes what the selector is asked on every routed turn, not
only on route lookups.

## Holding a tool loop on its model family

`hold_family_in_tool_loop` keeps a tool loop on the model family that opened
it. A turn is mid-loop when its last message returns the result of a tool
call that the episode's previous arm made: a Chat `tool` message, an
Anthropic `tool_result` block, or a Responses `function_call_output`. Such a
turn is scored only among the arms of the previous arm's family. The next user
turn may switch families again.

```yaml
algorithm:
  type: rayline_arc
  rayline_arc:
    hold_family_in_tool_loop: true
```

A switch mid-loop changes the reasoning the loop was built on, by target:

- to a Claude arm, over any wire format (Messages, or Chat and Responses
  through OpenRouter): only reasoning Claude provably wrote is carried, which
  is a Messages client's signed thinking block, a Responses reasoning item or
  `reasoning_details` item tagged `anthropic-claude-v1`, or an encrypted blob
  the episode records as issued by that worker. A mixed `reasoning_details`
  array keeps only Claude's items and its text is rebuilt from them. A
  reasoning-only assistant turn left empty is removed; a turn that called a
  tool keeps its call. Everything else (unsigned thinking,
  `reasoning_content`, `reasoning_details` of another format or of none) is
  dropped and counted as `foreign_dropped` in `reasoning_dropped`: Messages
  accepts thinking only with a signature, and over Chat, OpenRouter hands an
  assistant message's reasoning to Claude, which Anthropic refuses with
  `content_filter`;
- to any other arm (kimi, glm, DeepSeek, MiMo, ...) over Chat or Responses:
  Claude's thinking keeps its text as reasoning but loses its signature, its
  `redacted_thinking` is dropped, and unsigned reasoning and
  `reasoning_details` are carried unchanged. Without an episode record nothing
  proves who wrote unsigned reasoning, and DeepSeek and MiMo need their own
  `reasoning_content` back, so only a Claude arm drops what it cannot prove.

When Claude is served over Chat, the Messages output gives its thinking the
signature OpenRouter returned for it (`reasoning_details` tagged
`anthropic-claude-v1`), buffered or as a `signature_delta`, so a Messages
client replays Claude's thinking signed. A Claude arm reached over Chat is
sent that thinking back as the same signed `reasoning_details` item, so
Claude keeps its own reasoning across turns instead of losing it to the drop.

Which blocks are dropped depends only on each block and the arm's family, so
consecutive turns on one model send the same prefix and keep their prompt
cache. Claude is recognised by its family, so a Claude arm whose provider
model id has no `anthropic/` segment needs `publisher: anthropic` on its card.

So a loop that leaves Claude loses the thinking Claude can verify, and one
that enters Claude loses the other model's reasoning. Some providers (DeepSeek,
MiMo) also answer 400 when a tool history lacks their own reasoning content,
which a switch to Claude and back removes.

An arm's family is who made its model:

1. the model card's `publisher`, from the catalog or from
   `routing.modelCards[].publisher`;
2. otherwise, the first path segment of the provider model id the arm
   dispatches to (`deepseek/deepseek-v4-flash` is `deepseek`), without any
   `@variant` suffix.

The fallback only holds where a provider namespaces models by vendor, as
OpenRouter does. When a provider serves models under its own namespace
(`local/...`, `accounts/<name>/...`), set `publisher` on each model's card, or
every model behind that provider counts as one family.

The hold yields to every hard constraint (disabled, non-vision and incapable
arms, fallback exclusions). When it would leave no eligible arm, it is lifted
for that turn rather than refusing it. Both outcomes are logged as
`rayline_arc_tool_loop_family_hold`. In policy-service mode the hold narrows
the offered actions instead of the scored arms.

Off by default.

## Route lookup

A caller that owns its own provider keys and its own LLM bill can ask for the
selection without the execution:

```text
POST /v1/routes
```

The request body is the body that caller was going to send anyway -- the same
bytes `/v1/messages` or `/v1/responses` would have taken. Fields the selector
does not read, such as `max_tokens` or `temperature`, are ignored rather than
refused; `model` is ignored too, so neither auto-routing alias means anything
here.

One path serves every dialect, so the dialect has to be established from the
request. It is inferred from the body's own shape wherever the body says which
it is: a `system`, `stop_sequences` or `thinking` member, an `input_schema` on
a tool, a server tool declaring only its `type`, or a `tool_use` content block
make it Anthropic Messages; a `system`, `developer` or `tool` role inside
`messages[]`, a `tool_calls` member, a `function` on a tool or any of Chat's
own sampling fields make it Chat Completions. `max_tokens` decides nothing,
because both dialects accept it.

A body that carries none of those markers is read as Chat Completions and
answered with a `format_inferred` warning naming the choice. Send
`x-rayline-format` to settle it rather than be told:

| `x-rayline-format` | Dialect |
|---|---|
| `anthropic` | Anthropic Messages |
| `chat` | OpenAI Chat Completions |
| `responses` | OpenAI Responses |

The header wins over inference. An unrecognised value is ignored rather than
refused, because inference still has an answer and failing a well-formed
request over a header typo would be the worse of the two.

The answer names the model, the reasoning configuration the arm was scored
under, what it was chosen over, and both rate cards, so the caller can build
the provider call and compute its own savings without a reporting call back:

```json
{
  "route_id": "rte_7b929848",
  "object": "route",
  "model": "worker/model-id",
  "worker": "worker-id",
  "provider": "provider-slug",
  "thinking": { "mode": "off", "budget_tokens": null },
  "checkpoint": "arc-2026-09-12.c82a1f3e",
  "alternatives": [{ "model": "other/model-id", "worker": "other-worker-id", "provider": "provider-slug", "score": 0.62 }],
  "baseline": { "model": "reference/model-id", "input_per_mtok": 3.0, "output_per_mtok": 15.0, "cache_read_per_mtok": 0.3, "cache_write_per_mtok": 3.75 },
  "selected_pricing": { "input_per_mtok": 0.435, "output_per_mtok": 0.87, "cache_read_per_mtok": 0.0435, "cache_write_per_mtok": 0.544 },
  "warnings": [],
  "usage": { "encoded_input_tokens": 1840, "cache_read_tokens": 1200 },
  "latency_ms": 212
}
```

`thinking.budget_tokens` is the budget the arm was scored under, not a
suggestion. Executing a thinking-on arm with a different budget makes the
decision and the execution diverge on the axis the choice was made on.

`worker` and `provider` identify which arm was chosen. Two arms can serve the
same model through different providers at different prices -- the manifest
requires worker ids to be unique, not model names -- so `model` alone is not
enough to execute the decision that was made. `provider` is omitted when the
arm declares none.

`alternatives` explains the choice. It is not a failover list: those arms were
scored and rejected for this turn, and an arm a hard constraint removed before
scoring is not listed at all. Each entry names its arm the same way the
selected one does, because two arms serving one model would otherwise print as
two identical lines, and the list is sorted by score rather than left in
manifest order.

`warnings` is always present and usually empty. It names the ways a request
can produce a well-formed, plausible route while quietly not being the request
that was asked for -- tools dropped before encoding, a checkpoint pin this
cell cannot honour.

`baseline` is the artifact's declared reference worker. An artifact that
declares none omits the field rather than substituting a plausible model.

`checkpoint` pairs the release name an operator set in `checkpoint_label` with
the artifact's hash. The hash pins exactly one artifact but tells a reader
nothing; the label is readable but is not unique across deployments that reuse
a release name. Neither half alone is the identity.

There is deliberately no confidence score and no per-request explanation. The
selector has no source for either today, and a constant published under those
names reads as measured.

### Timing

One lookup is bounded by `deadline_ms`, default 1500, and a lookup that
exceeds it answers 504. The bound is the endpoint's own: the encoder's
`total_timeout_seconds` is sized for a dispatched turn that streams for
minutes and wraps whatever context it is handed, so a lookup that inherited it
would leave a waiting caller for the routed turn's timeout.

A lookup that collides with another on the same conversation, or with a busy
encoder, answers 429 with `Retry-After`. That is a healthy router saying when
to come back, not an unavailable one, and it is worth honouring: retrying
immediately lands on the lease or the queue that produced the contention.

### Episodes

Three optional request headers shape continuity:

| Header | Effect |
|---|---|
| `x-rayline-session` | the conversation this lookup belongs to. **Absent means stateless**, which is what a playground wants: experimenting must not advance a real conversation |
| `x-rayline-branch` | a subagent lane inside that conversation, so concurrent subagents are separate trajectories rather than each other's previous turn |
| `x-rayline-route-id` | a caller-minted id this router adopts and echoes, so one id spans both records |

A branch without a session names a lane in no conversation, so it is dropped
and reported as a `branch_ignored` warning rather than silently making the
lookup stateless.

Continuity also has to be switched on. With `episode_writes` off, which is the
default, every lookup is ephemeral: the episode identity is minted per call
and thrown away, nothing is leased and nothing is stored, so a lookup costs
the encode and nothing else. A caller that sends `x-rayline-session` to such a
cell gets an `episode_not_tracked` warning rather than silence, because a
route computed without the conversation is indistinguishable from one computed
with it.

Turn `episode_writes` on for a gateway that calls this endpoint on every turn
of an agentic run. A stable episode identity is also the encoder's prefix
cache key, so continuity and encoder efficiency arrive together, and neither
is reachable without the lease that serializes concurrent turns on one
conversation.

A tracked lookup commits its episode at decision time: there is no dispatch
phase to commit against, so the chosen arm becomes the previous arm on the
assumption that the caller ran it. A caller that routinely ignores the answer
will see `episode.stayed` stop making sense, which is the signal that its
episode ids are not stable per conversation. `episode` is omitted entirely
from an ephemeral lookup rather than reported as turn zero.

### Enabling it

```yaml
routing:
  decisions:
    - name: rayline_arc_reference_route
      algorithm:
        type: rayline_arc
        on_error: fail_closed
        rayline_arc:
          routes_api:
            enabled: true
            deadline_ms: 1500
            checkpoint_label: arc-2026-09-12
            episode_writes: false
```

Off by default, and deliberately not implied by configuring the algorithm. A
lookup drives the encoder with no paying turn behind it, and lands on the same
instance that serves routed traffic, so a cell acquires that load when an
operator says so. While it is off the path answers 404 for every method, so a
prober cannot tell a cell that has it switched off from one that never had it.

Errors use the Anthropic error envelope rather than this router's own, because
a caller of this endpoint is already parsing that envelope from the endpoint
it would otherwise have called. A contended lookup answers 429, not 503: the
router is healthy and briefly busy with that session.

### Describing the cell as one model

A gateway that fronts this cell as one model needs to say what that model is:
its context window, its output limit, whether it takes an image, whether it
takes a tool, what it costs. Those are per-arm facts, and the cell knows them,
so the model listing states them rather than leaving the gateway to hard-code
a copy. `GET /v1/models` lists the alias and, under its `routing` block, the
candidates the alias may resolve to:

```json
{
  "id": "rayline/arc-dev",
  "object": "model",
  "owned_by": "vllm-semantic-router",
  "routing": {
    "resolution": "virtual",
    "selectable": true,
    "default_route": true,
    "recipe": "default",
    "candidates": [
      {
        "model": "worker-id",
        "provider_model": "worker/model-id",
        "provider": "openrouter",
        "thinking": { "mode": "off" },
        "vision": false,
        "tools": true,
        "context_window": 1048576,
        "max_output_tokens": 943718,
        "pricing": { "currency": "USD", "input_per_mtok": 0.56, "output_per_mtok": 1.76, "cache_read_per_mtok": 0.104, "cache_write_per_mtok": 0.56 },
        "disabled": false
      }
    ]
  }
}
```

The candidates are listed under the alias and not as ids of their own,
because this cell refuses every id but the alias: they are what the selector
chooses among, not what a caller chooses. The gateway derives one model
document from them -- the smallest context window, the cheapest and dearest
rate, whether every selectable candidate takes a tool -- and addresses it by
the alias.

Every value is read off the loaded config and nothing else. On an ARC cell,
whose one decision is the arm set, the list is in artifact order, which is
the arm ordinal, and `model` names the arm the way a route's `worker` does.
`provider_model` is the id it dispatches as. A model ref that names a LoRA
adapter is listed as the adapter, with the card it is served under as
`base_model`. On the stock router an alias may resolve through several
decisions, chosen per request by their rules; the list holds each distinct
model and thinking mode once, in the order the decisions first declare them,
a route action's destination ahead of its decision's refs and the router's
`default_model` last, since an unmatched request resolves to it. Which
decision picks which is a per-turn fact the route lookup reports.
`thinking.mode` is the decision's `use_reasoning` for that arm. `vision` is the
card's verdict, which is capable when the card is silent; `tools` is the
card's positive `tools` claim, which an unmarked card does not make.
`context_window` and `max_output_tokens` are the card's `context_window_size`
and `max_output_tokens`, null where the card declares none. `pricing` is the
card's rate card in the route lookup's spelling, and `disabled` is the card's
out-of-service flag: a disabled arm is listed, so the set and the artifact
stay the same length, and marked, so the gateway leaves it out of what it
derives.

The list is not ARC's. Every virtual id on every router carries its
candidates. Nothing else about the decision is published: the gateway
derives its document from the candidates, and when the arm set changes the
candidates change with it, so there is no artifact pin to key on.

Two things about the numbers. `max_output_tokens` is also the limit the
router dispatches when a request states none of its own, so declaring it on a
card changes what such a request is sent. And `context_window` is enforced at
selection as a mask, not a filter: a turn longer than an arm's window excludes
that arm where it stands, the way an image turn excludes a text-only arm, and
a turn longer than every arm's window fails closed with `no_context_arm`, a
503 the gateway replays. The router's generic context filter, which removes a
plain decision's candidate from the list, is not applied to a `rayline_arc`
decision, because a shortened positional list does not steer a turn, it fails
it with `candidate_count`. Declare windows on ARC cards only on a build that
carries the mask; an earlier build applies the filter and fails every turn
above the smallest declared window.

## Deployment

The public Helm profile is
`deploy/helm/semantic-router/values-rayline-arc.yaml`. It mounts an existing
read-only `rayline-arc-artifact` PVC and reads credentials and private pins
from an existing `rayline-arc-runtime` Secret. Before deployment:

1. Populate the PVC from the immutable artifact revision. Do not use a mutable
   tag, and do not edit the mounted artifact in place.
2. Create the Secret keys named in the profile, including the protected Modal
   encoder endpoint, Modal proxy credentials, Redis address/password, and
   OpenRouter key.
3. Create a private values overlay containing every artifact-declared logical
   arm in exact manifest order. Its provider model, provider slug, thinking
   mode, and four cache-aware prices must match the manifest exactly.
4. Deploy the protected Rung B encoder from
   `src/vllm-plugins/rayline_arc_io/modal_service.py` only after its CUDA
   correctness gate passes.
5. For replica mode, provide stable IDs and independent endpoints in the
   private values overlay. Do not place a load balancer behind one replica ID;
   the ID is the retained-state owner.

Render and deploy a pinned chart release:

```bash
helm upgrade --install semantic-router \
  deploy/helm/semantic-router \
  --namespace semantic-router \
  --create-namespace \
  --values deploy/helm/semantic-router/values-rayline-arc.yaml \
  --values /private/rayline-arc-values.yaml \
  --atomic --wait
```

Readiness is intentionally fail closed. A missing credential, unavailable
encoder or Redis store, mutable/mismatched artifact, build/plugin/capability
drift, arm-order mismatch, provider mismatch, or pricing mismatch keeps ARC
unavailable rather than selecting a default arm.

For local CPU integration, `make rayline-arc-test-integration` builds and runs
real Envoy, Semantic Router, and Redis with a generated full-shape synthetic
artifact plus contract-faithful encoder and provider doubles. It covers both
arms, dispatch ownership, 2xx/non-2xx transactions, stream abort, client
cancel, same-episode fencing, cross-episode concurrency, two-replica affinity,
explicit-status failover, cooldown recovery, close fanout, stable-zero retained
sessions, router restart, and log privacy. It does not claim Qwen/CUDA
correctness; the Modal CUDA gate is separate.

## Rollback

Keep the prior Helm release, encoder deployment, artifact PVC, private values
overlay, and Secret version available as one immutable set. If readiness or
live canaries fail, stop admitting ARC traffic and roll back the router:

```bash
helm history semantic-router --namespace semantic-router
helm rollback semantic-router <LAST_GOOD_REVISION> \
  --namespace semantic-router --wait
```

Restore the matching previous encoder deployment and Secret/PVC references if
they changed. Do not point old router code at a new artifact or rewrite an
artifact in place. Verify `/health`, one public synthetic request per arm,
Redis episode advancement on 2xx, and absence of privacy canaries in logs
before restoring traffic. Delete or scale down the failed Modal deployment
after traffic is removed so it cannot continue incurring GPU cost.

## Status

This algorithm remains experimental. It is architecture-parity support, not a
model-quality or promotion claim. Production readiness additionally requires
the GPU numerical gates, Redis transaction tests, provider dispatch checks,
privacy canary, and full-stack acceptance described by the implementation
plan.
