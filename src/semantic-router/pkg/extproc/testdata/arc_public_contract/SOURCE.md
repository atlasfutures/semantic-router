# ARC public contract fixtures

Copied unchanged from atlasfutures/pathfinder `tests/fixtures/arc_public_contract/`
at `2678755348696411e815b5b988dfe80802301a56` (pathfinder#5664, the ARC public
contract: `docs/contracts/arc-public-api.openapi.yaml`, generated from
`src/rayline_router/serving/arc_public_contract.py`). Spellings marked
`x-status: proposed` there await Marcin's and Chi's review. Refresh by copying
the directory from a newer pin; do not edit these files here.

The predictions fixtures (`request_*response_fields*`, `request_valid_decisions_predictions.json`,
`response_predictions*.json`) and `decision_response_predictions.v1.json` (from
`tests/fixtures/arc_policy_service/`) are copied unchanged from pathfinder#5717
(`claude/arc-public-predictions`) at `b7b52ad1abadff1f2e0892fb18d604b77fb211d1`.

`execution_vectors.json` and `request_*execution*.json` are copied unchanged
from pathfinder#5875 at `cefe836a48` (`routing.execution`, proposed;
reference `src/rayline_router/serving/arc_execution_context.py`).

The hybrid routing v2 fixtures (`execution_v2_vectors.json`,
`request_*execution_v2*.json`, `request_*policy*.json`) are copied unchanged
from pathfinder main at `0b5131db24` (#5904, #5907, #5915 and #5913,
prefer-local).

`selection_refusals_v2.v1.json` (from `tests/fixtures/arc_policy_service/`)
is copied unchanged from pathfinder#5906 at `87b156e001`: the policy
service's v2 refusal strings and their public codes.
