# vLLM Semantic Router agent entry

vLLM Semantic Router is an Envoy ExtProc router for LLM inference. A public
entrypoint resolves to an isolated recipe; its signals and projections feed a
decision and algorithm, which select a backend and recipe-scoped plugins.

## Start here

- Read the nearest `AGENTS.md` only for directories you change.
- Use `tools/agent/docs/change-surfaces.md` when a user-visible router contract
  crosses configuration, runtime, deployment, or documentation.
- `tools/agent/domains.yaml` is the single changed-path registry for ownership,
  minimum checks, CI jobs, images, and E2E profiles.

```bash
make impact ENV=cpu CHANGED_FILES="path/one path/two"
make check CHANGED_FILES="path/one path/two"
make verify DOMAIN=<domain>       # explicit integration check
make verify PROFILE=<e2e-profile>
make ci-full                      # complete local PR baseline
```

Use `make harness-check` after changing the registry, workflows, harness code,
or these instructions. `impact` reports facts; it does not select a skill,
prescribe a work loop, or decide when the task is complete.

## Repository map

- `src/semantic-router/`: Go router, ExtProc runtime, routing, and APIs
- `src/vllm-sr/`: Python CLI and local stack orchestration
- `config/`: canonical configuration, fragments, schemas, and recipes
- `candle-binding/`, `ml-binding/`, `nlp-binding/`, `onnx-binding/`: inference bindings
- `dashboard/`: React frontend and Go management backend
- `deploy/`: deployment artifacts and operator
- `e2e/`: end-to-end framework and profiles
- `tools/`: build, development, release, security, and harness tooling
- `website/`: public documentation

## Durable constraints

- Use the existing `vllm-sr serve` local-image flow for local runtime behavior.
- Keep steady-state configuration canonical. Legacy layouts belong in explicit
  migration tooling, not the runtime parser.
- A behavior-visible routing, startup, config, Docker, CLI, or API change needs
  an appropriate integration or E2E assertion. Pure refactors do not.
- Generated artifacts and public docs change with their source contract.
- Numeric file, function, nesting, and interface limits are review signals, not
  architecture. Forbidden dependencies, new cycles, generated invariants, and
  unowned root files remain blocking checks.
- Prefer cohesive modules. Split or extract when ownership becomes mixed, not
  to satisfy a line-count target.
- Use an execution plan only for genuinely resumable multi-session work. Prefer
  GitHub issues for tracked debt; keep repository-only architectural risks in
  `tools/agent/docs/architecture-risks.md`.
- PR commits use `git commit -s`. Keep unrelated changes out of the branch.
- Never publish credentials, private hostnames, or private fleet details.

## Optional repository skills

Choose a skill by the semantics of the task, not by an automatic path match:

- `router-contract-change`: request-visible config/signal/decision/algorithm/plugin changes
- `routing-calibration`: live maintained-recipe probe and calibration work
- `maintainer-ops`: reviewed issue, PR, release, or GitHub mutation workflows

Canonical contributor and GitHub metadata remain in `CONTRIBUTING.md`, the
issue and PR templates, and `.prowlabels.yaml`.

## Fork working rules

These apply to work on this fork's branches; upstream-bound drafts follow the
upstream rules above and nothing here.

- Structure commits as a reviewable trajectory: one logical step per commit
  (refactor, then behaviour, then tests), every commit compiling and passing
  lint, messages that say why. Sign off every commit (`git commit -s`).
- Scope a PR to one subsystem where possible and title it with the module
  prefix: `[Router]`, `[CLI]`, `[Dashboard]`, `[Docs]`, `[CI/Build]`,
  `[Operator]`, `[E2E]`. Include a Test plan section that says what was run.
- Read the nearest `AGENTS.md` before editing a hotspot:
  `src/semantic-router/pkg/config/`, `src/semantic-router/pkg/extproc/`,
  `src/vllm-sr/cli/`, `deploy/operator/api/v1alpha1/`,
  `deploy/operator/controllers/`, `dashboard/frontend/src/`,
  `dashboard/backend/handlers/`. Hotspots are debt, not precedent; do not
  grow their responsibility.
- Layer model: `signal` -> `decision` -> `algorithm` -> `plugin` -> `global`.
  Interfaces belong only at true seams.
- Local runtime: `make vllm-sr-dev` then `vllm-sr serve --image-pull-policy
  never` (add `VLLM_SR_PLATFORM=amd` / `--platform amd` on ROCm). Focused
  suites: `make test-semantic-router`, `make test-binding`,
  `make test-{category,pii,jailbreak}-classifier`, `make e2e-test` (kind).
  Lint: `make go-lint`, `make check-go-mod-tidy`, `pre-commit run --all-files`,
  `markdownlint -c tools/linter/markdown/markdownlint.yaml "**/*.md"`.
