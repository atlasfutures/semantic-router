# One ARC cell serves every tier

Status: Proposed (acceptance is Chi Lang's call). 2026-10-29.

Scope: semantic-router's Rayline ARC cell (`pkg/extproc`), the `POST /v1/routes`
lookup and generation (`/v1/messages`, `/v1/chat/completions`, `/v1/responses`).
Code: `pkg/extproc/rayline_arc_tier.go`. Tests:
`pkg/extproc/rayline_arc_one_cell_tiers_test.go`.

## Why the split caused the wrong-tier bug

ARC has three public tiers: configurable `arc-1.0`, `arc-swift-1.0` and
`arc-sage-1.0`. The prod main cell holds them as three recipes: `default`
(top-level `routing:`) and `arc-swift` / `arc-sage` (`recipes:`), each selected
by a versioned entrypoint name.

Generation resolved the request model through the entrypoint table, so a Sage
turn ran on the Sage recipe. The lookup did not. `decisionOnlyRoutingTarget`
scanned only the top-level decisions, which are the default recipe's, and the
request model only gated id admission. So an `arc-sage` lookup on the main
cell was decided by arc-1.0's policy and operating point, and answered 200
with nothing in the response to say so.

The tier cells (`pf-arc073-sage`, `pf-arc073-swift`) existed to work around
this: a cell with one recipe cannot pick the wrong one. That fix moved the
tier from config into infrastructure. Every new tier needed a new cell, a new
gateway map entry and a new deploy, and the gateway had to send each tier to
the right cell. Nothing on the wire said which tier had decided, so a misroute
was as silent as the original bug.

The root cause was two code paths resolving "which recipe decides this
request" in two different ways, one of them implicitly. The fix is one
resolution, used by every endpoint, and a response that names its result.

## The invariant

Any ARC tier, in any public spelling, resolves on any cell to the same answer:

- the recipe that serves the tier decides it, on `/v1/routes` and on
  generation, priced or not;
- a cell that does not serve the tier refuses it (`404 model_not_served` on a
  lookup); it never decides it on another tier's recipe;
- every response the cell answers for an ARC request states what decided it
  (the header contract below), errors included once the tier is known.

Spellings (`raylineARCCanonicalTierIDs`): `arc-1.0`, `arc-1`, `arc-swift`,
`arc-swift-1.0`, `arc-sage`, `arc-sage-1.0`, each with or without `rayline/`.
Resolution (`raylineARCRecipeForModel`), in order:

1. a name the config selects (auto name or entrypoint) selects its recipe,
   exactly as generation always has;
2. any other tier spelling selects the recipe its canonical versioned id
   selects;
3. `arc-1.0` on a cell whose default recipe serves no other tier is that
   default recipe;
4. any other ARC-shaped name (`arc`, `arc-*`, `arc@*`) is refused;
5. a name outside the ARC namespace is not a tier: the default recipe, as a
   lookup has always read it.

A recipe's tier is the canonical id among the names that select it
(`raylineARCTierOfRecipe`): its entrypoints, plus the auto names for the
default recipe. That is how a single-recipe tier cell (`auto_model_names:
[rayline/arc-dev, arc-sage-1.0]`) reports `arc-sage-1.0` without a new config
field.

All lookup settings come from the resolved recipe's routes-enabled ARC
decision: deadline, routing controls, tool-name encoding, the models
`routing.prices` may name, and, at decision time, the recipe's own selector,
episode store and context-token classifier.

## Header contract

On every ARC response (lookup and generation; success, upstream error and the
cell's own refusals), replacing any upstream copy, stripped from non-ARC
responses:

| Header | Value |
| --- | --- |
| `x-rayline-tier` | canonical tier id with version: `arc-1.0`, `arc-swift-1.0`, `arc-sage-1.0` |
| `x-rayline-recipe` | the cell-local recipe: `default`, `arc-swift`, `arc-sage` |
| `x-rayline-arm` | the chosen arm (worker id); absent before selection |
| `x-rayline-model` | the arm's provider model id: the dispatched model on generation, the chosen worker's model on a lookup |
| `x-rayline-price-profile` | `catalog` or `inline`; a gateway that built inline prices from a saved profile restates it as `profile:<id>@<version>` |

`x-rayline-*`, not `x-vsr-*`: the ARC ingress scrubs `x-vsr-*` responses
except a keep-set, so these need no router-infra change. Client docs:
`website/docs/troubleshooting/vsr-headers.md`.

A caller, gateway or eval asserts on `x-rayline-tier` instead of trusting
which cell it reached.

## Adding a tier, recipe or model without a new cell

1. Add a recipe under `recipes:` with its ARC decision (policy, operating
   point, arms, `routes_api`).
2. Add its versioned entrypoint (`entrypoints: [{model_names: [arc-<tier>-<ver>],
   recipe: <name>}]`) and the same name to the ingress model gate.
3. Add the public spellings to `raylineARCCanonicalTierIDs` and, for a fixed
   tier, its value to `routesPublicModels` (and the fixed presets if it is one).
4. Add the tier to the test matrix (below); it must pass on the main cell and
   be refused on a cell that lacks it.

A new model is an arm in a recipe's `ModelRefs`; a new version of a tier is a
new entrypoint name on the new recipe, with the old one kept until callers
move. None of these needs a new cell or gateway routing.

Do not list a versioned tier id as an auto model name on a multi-recipe cell:
`RecipeForRoutingModel` checks auto names first, so its turns would go to the
default recipe (why #336 was closed).

## Test matrix

`TestOneCellResolvesEveryTierSpellingOnEveryEndpoint` runs cell shape (main,
Sage tier cell) × tier × every spelling × priced/unpriced × endpoint
(`/v1/routes`, generation): 96 cases. Each asserts the deciding recipe (only
that recipe's selector ran, and the answer is its arm) and all five headers;
on generation also the identity on a pre-selection refusal and on the
success and upstream-error header mutations. A tier the cell does not serve
must be 404 on a lookup with no selector run, and passthrough with no identity
on generation. Beside it: non-tier names, configured generation names
unchanged, priced lookups ephemeral, and the price refusals.

Reverting the lookup to the top-level decisions fails 16 cases (every Swift
and Sage spelling on the main cell).

## Priced lookups

- A lookup with `routing.prices` or `exclusive` never joins an episode, whatever
  `episode_writes` says, and warns `prices_untracked` when it would have. A
  committed pick would become the previous arm that later paid turns hold to.
- The policy service's `caller_prices_unsupported` and
  `caller_prices_inconsistent` answer 400 with those codes.
- Both `unknown_price_model` forms carry `error.models` (sorted; empty when the
  service named none).
- Price legs are bounded at `[0, 1e9]` USD per Mtok, as the contract and saved
  profiles are.

## Migration: retiring the tier cells

Retirement itself is a later Chi Lang decision. The path:

1. Roll this image to the main cell through the 0% gate. Check each tier and
   spelling by `x-rayline-tier` and `x-rayline-recipe`.
2. Roll it to the tier cells too. A cross-tier id there now gets 404, not a
   silent own-tier answer.
3. Shadow period: the gateway sends tier lookups to both the tier cell and the
   main cell and compares `x-rayline-tier`, `x-rayline-arm` and the presets.
4. Flip the gateway (`RAYLINE_ARC_TIER_CELLS` empty, tiers on standing), so
   lookups and turns for every tier go to the main cell.
5. Once the tier cells take no traffic, scale them to zero, then delete them.

## Open points

- The pathfinder public contract fixture (`model_ids.json`) still lists
  `arc-swift-1.0` under `refused_unknown_model`. The cell now admits the
  versioned ids, so the contract needs the matching update.
- Generation admits the short spellings in the cell, but the ingress model
  gate still admits only the configured names. Widening it is optional.
- A generation turn's `routing` block (`value_usd`, `temperature`) is still
  validated against the one top-level decision's `generation_routing`
  (`raylineGenerationRoutingPolicy`), not the resolved recipe's. Prod sets it
  the same on every recipe, so nothing differs today. It should follow the
  recipe in a later change.
