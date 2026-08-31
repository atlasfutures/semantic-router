# TD045: ARC Structured Builder Awaits Decision Adaptation Syntax

## Status

Open

## Owner Plan

[PL0039 Rayline ARC Orchestrator](../plans/pl-0039-rayline-arc-orchestrator.md)

## Release Relevance

The canonical YAML and CLI config surfaces support experimental ARC
configuration. The Dashboard raw config editor preserves the algorithm block,
but the structured DSL builder intentionally does not offer ARC yet.

## Scope

- `dashboard/frontend/src/lib/dslAlgorithmSchemas.ts`
- `src/semantic-router/pkg/dsl/compiler_algorithms.go`
- `src/semantic-router/pkg/dsl/decompiler_algorithms.go`
- decision-level `adaptations.mode`

## Summary

ARC requires `routing.decisions[].adaptations.mode: bypass` so Router Learning
cannot override the artifact-owned arm. The current Signal DSL has no syntax
for decision adaptations. Adding `rayline_arc` to the Dashboard structured
algorithm inventory without that field would generate a route that canonical
validation must reject.

The Dashboard canonical config surface stores `DecisionConfig.algorithm` as a
generic record and therefore preserves valid ARC YAML today. Only the
structured DSL builder is intentionally narrower.

## Evidence

- `config.AlgorithmConfig` and the Python CLI expose typed `rayline_arc`
  blocks.
- `validateRaylineARCDecisionContract` requires top-level
  `adaptations.mode=bypass`.
- `rawRouteItem` and `rawDecisionTreeItem` in `pkg/dsl/ast.go` have no
  adaptations clause, and the compiler cannot emit
  `DecisionAdaptationsConfig`.
- The Dashboard builder derives its algorithm choices from
  `dslAlgorithmSchemas.ts`; exposing ARC there would imply a round trip the DSL
  cannot perform.

## Why It Matters

A structured control that silently omits the mandatory learning exclusion
would be worse than an explicit YAML-only surface: the UI would appear to
configure a hard policy while producing an invalid or overridable route.

## Desired End State

Add a typed decision-adaptations clause to the Signal DSL, round-trip it through
the Go compiler/decompiler, then add nested ARC artifact/encoder/episode fields
to the Dashboard structured builder. The generated route must include
`adaptations.mode: bypass` and must never expose a plaintext Redis password
field.

## Exit Criteria

- Signal DSL compile/decompile round-trips decision-level
  `adaptations.mode: bypass`.
- Dashboard structured fields can author the complete typed ARC block and
  mandatory bypass without a raw JSON control.
- Dashboard, DSL, Go config, and Python CLI fixtures produce the same canonical
  YAML.
- This debt entry is removed once those tests pass.
