package extproc

import (
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// Gateway arm exclusion. A gateway may take arms out of one request's offer
// whatever its data mode: the arms it has ruled out for that caller. Unlike
// the privacy allowed set (rayline_arc_privacy.go), which is read only on a
// private request, this is read on every request, so it also masks the
// standard turns that explore. Like the other turn-signal headers it is read
// only with trust_turn_signal_headers, so a client cannot shape its own
// offer, and it is never forwarded upstream.
const (
	// raylineARCExcludedArmsHeader is the comma-separated arm ids (the
	// candidates' model names, as the decision lists them) the request may
	// not be served by. Absent or empty excludes none.
	raylineARCExcludedArmsHeader = "x-rayline-excluded-arms"
)

// arcFailureNoGatewayArm is the bounded class for a turn every arm of which
// that could serve it is one its gateway excluded.
const arcFailureNoGatewayArm = "no_gateway_arm"

// raylineARCExcludedArms reads the gateway's excluded arms off the trusted
// signal headers, sorted and deduplicated; nil when it names none.
func raylineARCExcludedArms(signalHeaders map[string]string) []string {
	var excluded []string
	for _, arm := range strings.Split(signalHeaders[raylineARCExcludedArmsHeader], ",") {
		if arm = strings.TrimSpace(arm); arm != "" {
			excluded = append(excluded, arm)
		}
	}
	slices.Sort(excluded)
	return slices.Compact(excluded)
}

// gatewayExcludedArms marks, by arm ordinal, the candidates the gateway
// excluded. It returns nil when it excludes none of them, which leaves
// selection exactly as it was.
func gatewayExcludedArms(excludedArms []string, modelRefs []config.ModelRef) []bool {
	if len(excludedArms) == 0 || len(modelRefs) == 0 {
		return nil
	}
	marked := false
	arms := make([]bool, len(modelRefs))
	for index, ref := range modelRefs {
		if slices.Contains(excludedArms, strings.TrimSpace(ref.Model)) {
			arms[index] = true
			marked = true
		}
	}
	if !marked {
		return nil
	}
	return arms
}

// withGatewayExcludedArms folds the gateway's exclusion into the mask. An
// excluded arm is out of the offer, so it is neither picked, held nor
// explored; a turn it leaves no arm for is a refusal.
func withGatewayExcludedArms(
	arcContext *selection.RaylineARCSelectionContext,
	armCount int,
	excluded []bool,
) ([]bool, error) {
	if len(arcContext.GatewayExcludedArms) == 0 {
		return excluded, nil
	}
	if len(arcContext.GatewayExcludedArms) != armCount {
		return nil, arcSelectionFailure("gateway_arm_mapping")
	}
	combined := make([]bool, armCount)
	eligible := 0
	for index := range combined {
		combined[index] = arcContext.GatewayExcludedArms[index] ||
			(len(excluded) == armCount && excluded[index])
		if !combined[index] {
			eligible++
		}
	}
	if eligible == 0 {
		return nil, arcSelectionFailure(arcFailureNoGatewayArm)
	}
	return combined, nil
}
