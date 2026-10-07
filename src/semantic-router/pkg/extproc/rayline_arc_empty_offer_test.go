package extproc

import (
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// An empty offer names every constraint that could have emptied it, so a
// policy_no_available_action 503 can be attributed from the log alone.
func TestEmptyOfferLogsEveryConstraint(t *testing.T) {
	logs := captureLogs(t)
	scorer := &policyServiceScorer{
		workerIDs:   []string{"glm-5.3-flash", "kimi-k3", "glm-5.3"},
		actionOrder: []string{"a-flash", "b-kimi", "c-glm", "d-glm"},
		bindings: map[string]policyBinding{
			"a-flash": {arm: 0}, "b-kimi": {arm: 1}, "c-glm": {arm: 2, model: "glm-5.3"}, "d-glm": {arm: 2},
		},
	}
	logRaylineARCEmptyOffer(&selection.RaylineARCSelectionContext{
		EpisodeIDHash: "episode", RequestFormat: policyFormatResponses, ImageBearing: true,
	}, scorer, emptyOffer{
		stage: "offer", held: 2, excluded: []bool{false, true, true}, hard: []bool{false, false, true},
		cellOut: map[int]string{1: turnFailureRateLimited},
	})
	fields := findLogEvent(t, logs, "rayline_arc_empty_offer")
	want := map[string]string{
		"empty_at": "offer", "held_arm": "2", "held_model": "glm-5.3", "request_format": policyFormatResponses,
		"image_bearing": "true", "excluded_arms": "[false true true]", "hard_excluded_arms": "[false false true]",
		"actions_per_arm": "[1 1 2]", "excluded_routes": "map[kimi-k3:rate_limited]",
	}
	for key, value := range want {
		if got := fmt.Sprint(fields[key]); got != value {
			t.Errorf("%s = %s, want %s", key, got, value)
		}
	}
}
