package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// The public predictions of a Decisions lookup (pathfinder#5717,
// arc_public_contract.project_predictions): the offered stage-one candidates
// with each one's predicted pass probability, rest-of-episode tokens and
// cost, and each result's own row (model_choice) with its provenance. Every
// value is the policy service's, copied, never recomputed; null stays null.

type raylineRoutesTokens struct {
	Input      *float64 `json:"input"`
	CacheRead  *float64 `json:"cache_read"`
	CacheWrite *float64 `json:"cache_write"`
	Output     *float64 `json:"output"`
}

type raylineRoutesPrediction struct {
	Success     *float64             `json:"success"`
	SuccessHead string               `json:"success_head"`
	Tokens      *raylineRoutesTokens `json:"tokens"`
	CostUSD     *float64             `json:"cost_usd"`
	Horizon     string               `json:"horizon"`
}

type raylineRoutesCandidate struct {
	Model      string                  `json:"model"`
	Effort     *string                 `json:"effort"`
	Prediction raylineRoutesPrediction `json:"prediction"`
}

type raylineRoutesCandidatesContext struct {
	Boundary   string            `json:"boundary"`
	SelectorID string            `json:"selector_id"`
	Provenance map[string]string `json:"provenance"`
}

type raylineRoutesResultProvenance struct {
	PackageSHA256 string `json:"package_sha256"`
	PointID       string `json:"point_id"`
}

// raylineRoutesProjection is the projection's output: Candidates and Context
// nil off a boundary; per result name, its row (nil off a boundary) and its
// provenance.
type raylineRoutesProjection struct {
	Candidates []raylineRoutesCandidate
	Context    *raylineRoutesCandidatesContext
	Choices    map[string]*raylineRoutesCandidate
	Provenance map[string]raylineRoutesResultProvenance
}

// raylineRoutesPoint is one result point: its name and the cost weight it
// was asked at.
type raylineRoutesPoint struct {
	name       string
	costWeight float64
}

// projectRaylineRoutesPredictions follows project_predictions exactly: the
// offered rows only, in the selector's order; each result's row is its
// tier pick's stage_one_candidate remapped to the offered-only index; a pick
// that names no offered row is refused.
func projectRaylineRoutesPredictions(
	predictions *raylinearc.PolicyDecisionPredictions,
	packageSHA256 string,
	picks map[string]*int,
	points []raylineRoutesPoint,
) (raylineRoutesProjection, error) {
	projection := raylineRoutesProjection{
		Choices:    map[string]*raylineRoutesCandidate{},
		Provenance: map[string]raylineRoutesResultProvenance{},
	}
	publicIndex := map[int]int{}
	if predictions != nil && predictions.StageOne != nil {
		stage := predictions.StageOne
		projection.Candidates = []raylineRoutesCandidate{}
		for index, row := range stage.Candidates {
			if !row.Offered {
				continue
			}
			publicIndex[index] = len(projection.Candidates)
			projection.Candidates = append(projection.Candidates, raylineRoutesCandidate{
				Model: row.Model, Effort: row.Effort,
				Prediction: raylineRoutesPrediction{
					Success: row.PPass, SuccessHead: "stage_one_p_pass",
					Tokens: raylineRoutesTokensOf(row.PredictedUsage), CostUSD: row.PredictedCostUSD,
					Horizon: "rest_of_episode",
				},
			})
		}
		projection.Context = &raylineRoutesCandidatesContext{
			Boundary: stage.Boundary, SelectorID: stage.SelectorID,
			Provenance: map[string]string{"package_sha256": packageSHA256},
		}
	}
	for _, point := range points {
		if projection.Candidates != nil {
			index := picks[point.name]
			if index == nil {
				return raylineRoutesProjection{}, fmt.Errorf("result %q chose no offered stage-one candidate", point.name)
			}
			public, ok := publicIndex[*index]
			if !ok {
				return raylineRoutesProjection{}, fmt.Errorf("result %q chose no offered stage-one candidate", point.name)
			}
			choice := projection.Candidates[public]
			projection.Choices[point.name] = &choice
		} else {
			projection.Choices[point.name] = nil
		}
		projection.Provenance[point.name] = raylineRoutesResultProvenance{
			PackageSHA256: packageSHA256, PointID: raylineRoutesPointID(point.costWeight),
		}
	}
	return projection, nil
}

// raylineRoutesTokensOf maps the internal usage legs to the public price
// legs (uncached_input is input); nil stays nil.
func raylineRoutesTokensOf(usage *raylinearc.PolicyPredictedUsage) *raylineRoutesTokens {
	if usage == nil {
		return nil
	}
	return &raylineRoutesTokens{
		Input: usage.UncachedInput, CacheRead: usage.CacheRead, CacheWrite: usage.CacheWrite, Output: usage.Output,
	}
}

// raylineRoutesPointID is the policy's OperatingPoint.point_id of a weight:
// sha256 of the canonical JSON {"cost_weight": W}, written as Python's
// json.dumps(sort_keys, separators=(",", ":")) writes it (pathfinder
// content_identity.digest), the float in Python's repr.
func raylineRoutesPointID(costWeight float64) string {
	digest := sha256.Sum256([]byte(`{"cost_weight":` + pythonFloatRepr(costWeight) + `}`))
	return hex.EncodeToString(digest[:])
}

// pythonFloatRepr is Python's repr of a finite float: the shortest
// round-trip digits, fixed notation for decimal exponents in [-4, 16) with a
// trailing ".0" for an integral value, and d.ddde+XX outside.
func pythonFloatRepr(value float64) string {
	if math.IsInf(value, 0) || math.IsNaN(value) {
		return "null"
	}
	if value == 0 {
		if math.Signbit(value) {
			return "-0.0"
		}
		return "0.0"
	}
	sign := ""
	if value < 0 {
		sign, value = "-", -value
	}
	scientific := strconv.FormatFloat(value, 'e', -1, 64)
	mantissa, exponentText, _ := strings.Cut(scientific, "e")
	exponent, _ := strconv.Atoi(exponentText)
	digits := strings.Replace(mantissa, ".", "", 1)
	if exponent < -4 || exponent >= 16 {
		text := digits[:1]
		if len(digits) > 1 {
			text += "." + digits[1:]
		}
		expSign := "+"
		if exponent < 0 {
			expSign, exponent = "-", -exponent
		}
		return fmt.Sprintf("%s%se%s%02d", sign, text, expSign, exponent)
	}
	var text string
	switch {
	case exponent < 0:
		text = "0." + strings.Repeat("0", -exponent-1) + digits
	case len(digits) <= exponent+1:
		text = digits + strings.Repeat("0", exponent+1-len(digits)) + ".0"
	default:
		text = digits[:exponent+1] + "." + digits[exponent+1:]
	}
	return sign + text
}

// marshalRaylineRoutesJSON writes a value as a RawMessage; nil writes JSON
// null (a key the answer states as null, not one it leaves out).
func marshalRaylineRoutesJSON(value any) json.RawMessage {
	encoded, err := json.Marshal(value)
	if err != nil {
		return json.RawMessage("null")
	}
	return encoded
}
