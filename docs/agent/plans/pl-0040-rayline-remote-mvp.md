# PL-0040 Rayline Remote Router MVP

## Goal

Integrate the Pathfinder Rayline router as an authoritative, remote
per-decision model-selection algorithm in vLLM Semantic Router, using the
transaction, fail-closed, dispatch-validation, readiness, and privacy patterns
established by David's PL-0039 work.

The completion boundary is a deterministic end-to-end MVP: an OpenAI Chat
Completions request enters through Envoy, Semantic Router matches a decision,
Rayline selects one of that decision's allowed workers, Semantic Router invokes
the mapped provider, and Rayline advances or preserves episode state according
to the observed provider outcome.

Status: proposed on 2026-07-29. No implementation task is complete yet.

## Inputs and Dependency Boundary

The implementation starts from these reviewed inputs:

- Semantic Router
  [`davidvgilmore/semantic-router:rayline/pl-0039`](https://github.com/davidvgilmore/semantic-router/tree/rayline/pl-0039)
  at branch head `7a2d68ffcd788f356be3d052ddecb1914f51ef58`;
  the recorded code-bearing head is `4afa3361`.
- vLLM
  [`davidvgilmore/vllm:rayline/pl-0039-causal-mean`](https://github.com/davidvgilmore/vllm/tree/rayline/pl-0039-causal-mean)
  at `162bcefe1b41c5bb35eccc2f2219ea39e2c74bb7`.
- Pathfinder `origin/main` at
  `feec6409be249a045ef181711d48609a98f6cec6`, especially
  `src/rayline_router/serving/app.py`,
  `src/rayline_router/tracestate/`, and
  `tests/test_route_decision.py`.

These are integration inputs, not unreviewed merge instructions. RRM-001
re-resolves their heads and records the exact bases used by the implementation.

PL-0039 and PL-0040 remain separate modes:

- `rayline_arc` is the frozen, artifact-exact embedded selector. Semantic
  Router owns its episode store, F32 head/policy, and vLLM encoder contract.
- `rayline_remote` calls the Pathfinder service. Pathfinder is the only
  authoritative owner of its policy and episode state; Semantic Router retains
  only an opaque request-scoped receipt.
- The PL-0039 vLLM fork is required by `rayline_arc`. It is not a runtime
  dependency of `rayline_remote`.

Exactly one state owner is permitted in each mode. Router Learning adaptation,
Router Replay persistence, and any second session selector are bypassed for a
`rayline_remote` decision.

## Scope

### MVP In Scope

- Add experimental `algorithm.type: rayline_remote`.
- Support authoritative remote selection after Semantic Router has matched one
  decision.
- Support OpenAI Chat Completions requests, including tools and both streaming
  and non-streaming provider responses.
- Map Rayline worker IDs one-to-one to decision `modelRefs`; send only the
  mapped worker IDs as the request-scoped candidate allowlist.
- Require an explicit episode header and send Pathfinder an HMAC-derived opaque
  episode key, never the raw header value.
- Use one stable `decision_id` for every retry and lifecycle call belonging to
  a request.
- Add idempotent prepare, renew/check, commit, abort, and settle operations to
  the Pathfinder decision-plane API.
- Commit Rayline selection state synchronously on the first upstream 2xx
  response headers.
- Abort without advancing state on every terminal path before 2xx headers.
- Settle bounded actual status, token, cost, and latency facts after response
  completion when available. Settlement cannot rewrite the committed worker.
- Fail closed on remote timeout, invalid candidate, expired receipt, dispatch
  mismatch, or transaction failure.
- Validate the pinned Rayline bundle and worker-to-provider dispatch contract
  during readiness.
- Provide hermetic Semantic Router E2E coverage with a protocol fixture and a
  second cross-repository run against the actual Pathfinder service.

### MVP Non-Goals

- Replacing or collapsing `rayline_arc`.
- Anthropic Messages or OpenAI Responses API support.
- Multi-replica or restart-safe pending Rayline transactions.
- Redis, Firestore, or D1 transaction-journal support. The MVP pending journal
  is bounded, in-process, and explicitly single-replica; committed trace state
  may continue to use the existing supported Pathfinder store.
- Shadow-mode comparison, traffic splitting, or fail-open fallback.
- Online policy training or `UpdateFeedback`.
- Dashboard, DSL, CRD, or operator authoring for `rayline_remote`.
- Letting Rayline inject provider credentials, arbitrary headers, or arbitrary
  request-body mutations.
- Paid model evaluation, GPU evaluation, or changes to vLLM causal-MEAN.

Any production requirement for durable pending transactions or multi-replica
Pathfinder serving must be recorded as indexed technical debt before this plan
closes.

## Target Architecture

```text
OpenAI client
    |
    v
Envoy -> Semantic Router
            |
            | 1. signals + decision match
            | 2. prepare(decision_id, opaque episode, allowed workers, request)
            v
       Pathfinder Rayline
            |  reads committed episode state
            |  selects only from the allowlist
            |  creates a pending receipt; no state advance
            v
Semantic Router
    | 3. validate/renew receipt immediately before dispatch
    | 4. map selected worker -> configured ModelRef/provider
    | 5. execute provider with VSR-owned credentials
    |
    +-- first 2xx headers --> Rayline commit exactly once
    |                         then return/stream provider response
    |
    +-- any pre-2xx failure -> Rayline abort exactly once
    |
    `-- response complete --> Rayline settle actual bounded outcome
```

Semantic Router remains the data plane and credential owner. Pathfinder remains
the policy and state authority. The wire protocol carries selection and outcome
facts, not provider secrets or executable transport instructions.

## Configuration Contract

RRM-003 freezes exact field names before implementation. The target shape is:

```yaml
algorithm:
  type: rayline_remote
  on_error: fail_closed
  rayline_remote:
    base_url: http://rayline-router:8000
    bundle_version: mtrouter-example-immutable-revision
    api_key_env: RAYLINE_API_KEY
    episode_id_header: x-rayline-episode-id
    episode_hmac_key_env: RAYLINE_EPISODE_HMAC_KEY
    decision_id_header: x-rayline-route-id
    connect_timeout_ms: 250
    request_timeout_ms: 1000
    lease_ttl_seconds: 30
    max_retries: 1
    workers:
      - id: mock-a
        model: model-a
      - id: mock-b
        model: model-b
adaptations:
  mode: bypass
```

Validation requires:

- `on_error: fail_closed`;
- at least two unique workers and two unique decision `modelRefs`;
- a one-to-one worker-to-model mapping covering the decision's complete
  candidate set;
- an immutable bundle version;
- valid bounded HTTP(S) URL and timeouts;
- environment-variable references for both credentials and the episode HMAC
  key, with no secret material in canonical export;
- `adaptations.mode: bypass`;
- Router Replay disabled for the decision; and
- no worker mapping to an auto-routing alias or another virtual model.

The configuration values above are illustrative local-MVP values. RRM-003
freezes tested bounds rather than treating them as production defaults.

## Versioned Remote Contract

Pathfinder keeps the current `POST /v1/route` behavior for existing consumers.
The transactional integration uses new endpoints:

- `POST /v1/route/prepare`
- `POST /v1/route/renew`
- `POST /v1/route/commit`
- `POST /v1/route/abort`
- `POST /v1/route/settle`

The prepare request has the following logical envelope:

```json
{
  "schema_version": "rayline-router.selection-transaction.v1",
  "decision_id": "rt_123e4567-e89",
  "episode_key": "hmac-sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "bundle_version": "mtrouter-example-immutable-revision",
  "candidates": ["mock-a", "mock-b"],
  "request": {
    "protocol": "openai.chat.completions",
    "messages": [],
    "tools": []
  }
}
```

The response contains only bounded selection facts:

```json
{
  "schema_version": "rayline-router.selection-transaction.v1",
  "decision_id": "rt_123e4567-e89",
  "receipt": "opaque-server-token",
  "selected_worker": "mock-b",
  "route_call_index": 3,
  "bundle_version": "mtrouter-example-immutable-revision",
  "lease_expires_at": "2026-07-29T12:00:30Z"
}
```

All later operations carry the same `schema_version`, `decision_id`, and opaque
`receipt`.

- Semantic Router mints the decision ID unless a trusted ingress has stripped
  the client-supplied value and stamped a validated ID. An arbitrary client
  cannot choose another request's idempotency key.
- `renew` confirms ownership and extends the bounded pending lease. Semantic
  Router calls it immediately before dispatch and periodically until response
  headers.
- `commit` accepts only a 2xx upstream status and atomically advances the
  selected worker and turn index.
- `abort` records a bounded reason class and releases the pending turn without
  changing committed episode state.
- `settle` accepts bounded actual outcome fields such as status, input/output
  tokens, cost, and latency. It is idempotent and cannot select a different
  worker or advance the turn a second time.

Lifecycle idempotency rules:

- Repeated prepare with the same `decision_id` and request digest returns the
  same worker and receipt.
- Reusing a `decision_id` with different request, episode, bundle, or candidate
  data returns a conflict.
- Repeated identical renew, commit, abort, or settle calls return the original
  terminal result.
- Commit after abort, abort after commit, and settlement for an unknown receipt
  return a typed conflict.
- Expired pending receipts cannot dispatch or commit and do not advance state.

Requests and responses have explicit byte, collection, string, and numeric
bounds. Pathfinder authenticates Semantic Router independently of the client
request. The client's provider authorization header is never forwarded.

## Implementation Design

### Shared Semantic Router Lifecycle Seam

Refactor the PL-0039 ARC-specific request finalizer behind a narrow extproc
transaction interface:

```go
type SelectionTransaction interface {
    ValidateDispatch(context.Context) error
    CommitOnHeaders(context.Context, int) error
    Abort(context.Context, string) error
    Settle(context.Context, ActualOutcome) error
}
```

`RequestContext` owns at most one transaction. The existing deferred process
finalizer guarantees one pre-header terminal action. The ARC adapter preserves
PL-0039 behavior and uses a no-op settlement. The remote adapter owns only the
client and opaque receipt; it does not cache episode state.

Known lease loss must be consumed by `ValidateDispatch` immediately before the
provider call, closing PL-0039's open fail-closed-before-dispatch finding.
Reload holds apply only to requests that own a transaction, and their drain
budget derives from that transaction's configured request/lease bounds.

### Pathfinder Transaction Coordinator

Extract the pure policy calculation from the current eager `decide()` flow.
The existing `/v1/route` endpoint composes the helper with its current
optimistic behavior for compatibility. The new prepare endpoint:

1. validates authentication, bundle pin, request, episode, and candidate set
   before touching state;
2. reads a committed trace snapshot without advancing it;
3. applies the policy only to the allowed candidates;
4. writes a bounded pending record keyed by decision and episode;
5. returns an opaque receipt.

Only commit reserves the prepared turn and updates the previous worker. Commit
verifies that the trace version seen at prepare is still current. Pending
records serialize one in-flight decision per episode, expire automatically,
and retain a bounded terminal-result cache for idempotent retries.

The MVP journal is in-process and single-replica. Its interface must make a
durable implementation possible without changing the HTTP contract.

### Candidate and Dispatch Validation

Pathfinder may select only a worker listed in the prepare request. Semantic
Router rejects any response whose worker, bundle, decision ID, or receipt does
not match the prepared request.

At startup/readiness, Semantic Router queries Pathfinder's versioned capability
and worker catalog and verifies:

- protocol version and transaction capabilities;
- exact immutable bundle version;
- every configured Rayline worker exists;
- the worker maps to the expected provider model, reasoning mode, fallback
  policy, and pricing identity available in the VSR configuration; and
- no extra worker can become dispatchable through the decision mapping.

Semantic Router maps the chosen worker to its configured `ModelRef`, resolves
the provider endpoint, injects credentials, and owns request execution. Remote
execution hints are diagnostic only in the MVP.

### Failure and Response Semantics

- Prepare or pre-dispatch validation failure returns a typed JSON 503 and calls
  no provider.
- Provider transport error, 4xx/5xx headers, client cancellation, handler
  error, or panic before headers aborts the receipt.
- First 2xx headers commit synchronously before Semantic Router releases the
  extproc response. A commit failure becomes a typed 503.
- A streaming body failure after a successful 2xx commit does not undo state.
- Settlement failure after response completion is observable but does not
  replace a successful client response or re-run selection.
- There is no first-candidate fallback and no fail-open path in the MVP.

Errors expose bounded classes only. Metrics and traces use algorithm name,
candidate index or bounded hashes, outcome, and latency. They never contain raw
episode IDs, prompts, tool bodies, receipts, credentials, full worker IDs, or
provider response bodies.

## Planned Code Surfaces

Semantic Router, based on the PL-0039 branch:

- `src/semantic-router/pkg/config/rayline_remote_config.go`
- focused validation and canonical/reference contract tests under
  `src/semantic-router/pkg/config/`
- `src/semantic-router/pkg/selection/raylineremote/` for bounded wire types,
  client, readiness, and tests
- focused extproc files for remote selection, generic transaction ownership,
  dispatch validation, response-header commit, and response settlement
- `src/semantic-router/pkg/observability/metrics/`
- `config/algorithm/selection/rayline-remote.yaml`
- local reference config and user/operator documentation
- `e2e/testing/rayline-remote/`
- `tools/make/e2e.mk` and affected-profile mapping if required by the harness

Do not widen `config.go`, `processor_req_body.go`, or
`req_filter_classification.go`. Keep them as schema/orchestration entrypoints
and extract new responsibilities into adjacent focused files.

Pathfinder:

- `src/rayline_router/serving/app.py` only as a thin endpoint/service
  composition seam
- a focused serving module for the versioned selection contract
- a focused transaction coordinator and bounded pending journal
- the trace-store seam for a read-only snapshot/version contract
- policy helpers for request-scoped candidate masking
- unit tests alongside `tests/test_route_decision.py`
- a cross-repository integration harness using fake provider backends

Existing uncommitted files in either worktree are user-owned and must not be
overwritten or included accidentally.

## End-to-End Acceptance

The deterministic stack contains Envoy, Semantic Router, a real or
protocol-fixture Rayline service, and two fake OpenAI-compatible providers. It
uses the repository's local image flow:

```bash
make vllm-sr-dev
vllm-sr serve --image-pull-policy never --config <rayline-remote-config>
```

The MVP is accepted only when executable tests prove all of the following:

| Scenario | Required assertion |
| --- | --- |
| Happy path | Rayline selects worker B, only fake provider B is called, the client receives B's response, commit occurs once, and the next turn observes B as previous worker with the next index. |
| Candidate mask | A worker outside the decision mapping is never selected or dispatched, even when the underlying Rayline policy ranks it first. |
| Remote timeout | Semantic Router returns 503, calls no provider, and Rayline state is unchanged. |
| Invalid response | Wrong worker, bundle, decision ID, or receipt returns 503 with no provider call. |
| Lease loss | Expiry or renewal failure detected before dispatch returns 503 with no provider call and no state advance. |
| Provider failure | Transport failure or non-2xx headers abort once; retrying the episode reuses the same committed turn index. |
| Streaming success | State commits on first 2xx headers; a later body failure does not roll it back. |
| Idempotency | Duplicate prepare returns the same selection and receipt; duplicate commit and settle advance and account once. |
| Concurrency | Same-episode requests serialize or fail within the configured acquisition bound; different episodes proceed independently. |
| Settlement | Available usage/cost/latency reaches Rayline once; absent usage remains valid and does not fabricate zero-valued evidence. |
| Learning isolation | Router Learning and Router Replay cannot override or persist the remote selection path. |
| Privacy | Cross-service logs and metrics contain no prompt canary, tool canary, raw episode ID, receipt, API key, or authorization value. |
| Regression | Existing selectors and `rayline_arc` retain their prior behavior and tests. |

Two receipts are required:

1. A hermetic Semantic Router integration suite uses a contract-faithful fake
   Rayline service and runs in normal repository CI without Pathfinder source
   or private artifacts.
2. A Pathfinder-owned cross-repository suite starts the actual Rayline service,
   the locally built Semantic Router image, Envoy, and fake providers, then
   executes the same acceptance scenarios.

The second receipt, not the protocol fake alone, is the MVP end-to-end
completion proof.

## Exit Criteria

- The reviewed PL-0039 Semantic Router foundation is integrated and its
  `rayline_arc` tests remain green.
- PL-0039's reload-drain, pre-dispatch lease-loss, and private worker-label
  findings are fixed or isolated before their seams are reused.
- `rayline_remote` is typed, experimental, fail-closed, and cannot be
  post-selected by Router Learning.
- Pathfinder implements the versioned, idempotent transaction contract without
  breaking existing `/v1/route` consumers.
- Pathfinder owns remote episode state; Semantic Router owns only an opaque
  request-scoped receipt.
- Candidate masking and startup dispatch-contract validation prevent arbitrary
  worker or provider execution.
- Commit/abort/settle behavior matches the acceptance table on streaming and
  non-streaming paths.
- Hermetic Semantic Router and actual Pathfinder cross-repository E2E suites
  pass using fake providers and no GPU or paid model.
- Focused Go/Python tests, race-sensitive transaction tests, privacy checks,
  and all repo-native gates pass on exact final commits.
- Config fragment, reference docs, local runbook, failure behavior, readiness,
  observability, and rollback are documented.
- Any deferred durable-journal, HA, protocol, or authoring gap is indexed as
  technical debt rather than left only in this plan or a PR description.

## Task List

- [ ] RRM-001 Re-resolve and review the PL-0039 Semantic Router/vLLM heads and
      Pathfinder main; record exact implementation bases and a clean
      cross-repository branch strategy.
- [ ] RRM-002 Integrate the PL-0039 Semantic Router foundation and fix its three
      recorded open findings before extracting shared transaction behavior.
- [ ] RRM-003 Freeze `rayline_remote` configuration, wire schema, lifecycle
      state machine, error taxonomy, size limits, idempotency rules, and shared
      golden fixtures.
- [ ] RRM-004 Refactor Pathfinder policy evaluation into a pure selection
      helper and add exact request-scoped candidate masking without changing
      legacy `/v1/route` behavior.
- [ ] RRM-005 Implement Pathfinder prepare/renew/commit/abort/settle endpoints,
      bounded pending journal, receipt expiry, idempotency cache, and
      read-snapshot/version support.
- [ ] RRM-006 Add Semantic Router typed config, canonical validation, worker
      mapping, secret redaction, bounded HTTP client, and readiness/catalog
      validation.
- [ ] RRM-007 Generalize the PL-0039 request transaction finalizer and add the
      remote selector, pre-dispatch lease check, provider mapping, header-time
      commit, pre-header abort, and post-body settlement.
- [ ] RRM-008 Add focused Pathfinder and Semantic Router unit, concurrency,
      cancellation, timeout, malformed-contract, and privacy tests.
- [ ] RRM-009 Add the hermetic Semantic Router `rayline-remote` integration
      profile and explicit pass/fail assertions for the acceptance table.
- [ ] RRM-010 Add and run the Pathfinder-owned cross-repository E2E against the
      actual Rayline service, local Semantic Router image flow, Envoy, and fake
      providers.
- [ ] RRM-011 Add config fragment, reference docs, local runbook,
      readiness/metrics documentation, rollback steps, and indexed debt for
      every intentionally deferred production gap.
- [ ] RRM-012 Run final affected gates in both repositories, record exact
      commits and commands, rerun both E2E receipts on those heads, and close
      the plan only when every exit criterion has evidence.

## Next Action

Execute RRM-001. Re-resolve the three branch heads, review the complete PL-0039
diff rather than only its final plan, identify the smallest reviewable commit
stack that preserves `rayline_arc`, and record the exact Pathfinder base before
editing production code.

## Operating Rules

- Re-read this plan and continue from the first actionable unchecked task at
  the start of each loop.
- Keep one signed, reviewable branch per repository and never push directly to
  `main`.
- Preserve existing `/v1/route`, existing selector fallback behavior, and
  `rayline_arc` behavior; strict fail-closed semantics apply specifically to
  the new remote mode.
- Run the smallest focused tests after every task, then the repo-native gates
  reported for the exact changed-file set.
- Use the Semantic Router local image and serve flow for local behavior.
- Do not add a private Pathfinder source dependency, private bundle, deployment
  URL, provider credential, prompt, or golden to the public Semantic Router
  repository.
- Never weaken idempotency, candidate validation, privacy canaries, or the
  2xx-only commit point to obtain a passing test.
- Update this task list and append exact commands/results as each task closes.
- If implementation still diverges from the target architecture, create or
  update an indexed debt entry before handing off.

## Related Docs

- [Architecture guardrails](../architecture-guardrails.md)
- [Feature-complete checklist](../feature-complete-checklist.md)
- [Testing strategy](../testing-strategy.md)
- [Change surfaces](../change-surfaces.md)
- [PL-0039 branch plan](https://github.com/davidvgilmore/semantic-router/blob/rayline/pl-0039/docs/agent/plans/pl-0039-rayline-arc-orchestrator.md)
- [Pathfinder router API formalization](https://github.com/atlasfutures/pathfinder/blob/main/docs/system_one/router_api_formalization.md)
