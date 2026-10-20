package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// Privacy routing (memex-desktop#7455). The gateway decides a request's data
// mode and which arms may serve it, from its model catalog; the router cannot
// know either, so it reads both from the gateway's headers. Like the other
// turn-signal headers they are read only with trust_turn_signal_headers, so a
// client cannot widen its own allowed set, and never forwarded upstream.
//
// x-rayline-privacy-mode keeps its exploration meaning (rayline_arc_exploration.go):
// "false" is a standard request. "privacy" and "zdr" are the two private
// modes, and any other stated value is read as "privacy", failing closed.
// An absent or empty header is a standard request, as before the mask.
const (
	// raylineARCPrivacyAllowedArmsHeader is the comma-separated arm ids (the
	// candidates' model names, as the decision lists them) a private request
	// may be served by. A private request without it may be served by none.
	raylineARCPrivacyAllowedArmsHeader = "x-rayline-privacy-allowed-arms"

	raylineARCPrivacyModePrivacy = "privacy"
	raylineARCPrivacyModeZDR     = "zdr"
)

// arcFailureNoPrivateArm is the bounded class for a private turn none of whose
// allowed arms can serve it. It answers 422: the same request is refused
// identically every time, and a 503 would invite the gateway to replay it.
const arcFailureNoPrivateArm = "no_private_arm"

// raylineARCPrivacyMode reads the request's data mode off the trusted signal
// headers: "" for a standard request, else privacy or zdr.
func raylineARCPrivacyMode(signalHeaders map[string]string) string {
	switch mode := strings.ToLower(strings.TrimSpace(signalHeaders[raylineARCPrivacyModeHeader])); mode {
	case "", "false":
		return ""
	case raylineARCPrivacyModeZDR:
		return raylineARCPrivacyModeZDR
	default:
		return raylineARCPrivacyModePrivacy
	}
}

// raylineARCPrivacyAllowedArms reads a private request's allowed set off the
// trusted signal headers, sorted and deduplicated: nil for a standard
// request, empty for a private one that names none.
func raylineARCPrivacyAllowedArms(mode string, signalHeaders map[string]string) []string {
	if mode == "" {
		return nil
	}
	allowed := []string{}
	for _, arm := range strings.Split(signalHeaders[raylineARCPrivacyAllowedArmsHeader], ",") {
		if arm = strings.TrimSpace(arm); arm != "" {
			allowed = append(allowed, arm)
		}
	}
	slices.Sort(allowed)
	return slices.Compact(allowed)
}

// privacyExcludedArms marks, by arm ordinal, the candidates outside a private
// request's allowed set. It returns nil for a standard request, which leaves
// selection exactly as it was.
func privacyExcludedArms(mode string, allowed []string, modelRefs []config.ModelRef) []bool {
	if mode == "" || len(modelRefs) == 0 {
		return nil
	}
	arms := make([]bool, len(modelRefs))
	for index, ref := range modelRefs {
		arms[index] = !slices.Contains(allowed, strings.TrimSpace(ref.Model))
	}
	return arms
}

// privacyModeLabel is the mode the cell applied, as logged and as
// x-vsr-privacy-mode states it.
func privacyModeLabel(mode string) string {
	if mode == "" {
		return "standard"
	}
	return mode
}

// privacyAllowedArmsHash is a short stable digest of a sorted allowed set:
// the set is the gateway's and unbounded in length, so records carry its
// count and this digest rather than the list. Empty for a standard request.
func privacyAllowedArmsHash(allowed []string) string {
	if allowed == nil {
		return ""
	}
	sum := sha256.Sum256([]byte(strings.Join(allowed, "\n")))
	return hex.EncodeToString(sum[:8])
}

// addPrivacyFields writes the request's privacy facts into a log record: the
// applied mode, and for a private request its allowed set's size and digest.
func addPrivacyFields(fields map[string]interface{}, ctx *RequestContext) {
	fields["privacy_mode"] = privacyModeLabel(ctx.RaylineARCPrivacyMode)
	if ctx.RaylineARCPrivacyMode == "" {
		return
	}
	fields["privacy_allowed_arms_count"] = len(ctx.RaylineARCPrivacyAllowedArms)
	fields["privacy_allowed_arms_hash"] = privacyAllowedArmsHash(ctx.RaylineARCPrivacyAllowedArms)
}

// withPrivacyExcludedArms folds the privacy constraint into the mask. An arm
// outside the allowed set is not a worse arm: serving there sends the
// request to a provider the user's data mode rules out, so it is a refusal.
func withPrivacyExcludedArms(
	arcContext *selection.RaylineARCSelectionContext,
	armCount int,
	excluded []bool,
) ([]bool, error) {
	if len(arcContext.PrivacyExcludedArms) == 0 {
		return excluded, nil
	}
	if len(arcContext.PrivacyExcludedArms) != armCount {
		return nil, arcSelectionFailure("privacy_arm_mapping")
	}
	combined := make([]bool, armCount)
	eligible := 0
	for index := range combined {
		combined[index] = arcContext.PrivacyExcludedArms[index] ||
			(len(excluded) == armCount && excluded[index])
		if !combined[index] {
			eligible++
		}
	}
	if eligible == 0 {
		return nil, arcSelectionFailure(arcFailureNoPrivateArm)
	}
	return combined, nil
}

// privacyProviderPreferences is the provider object a private turn sends: the
// arm's pin, or none, with OpenRouter told to use only providers that do not
// collect the data and not to fall back past them; a zdr turn also asks for
// zero-data-retention endpoints only. A standard turn's pin is unchanged.
func privacyProviderPreferences(
	pin *config.OpenRouterProviderPreferences,
	mode string,
) *config.OpenRouterProviderPreferences {
	if mode == "" {
		return pin
	}
	private := config.OpenRouterProviderPreferences{}
	if pin != nil {
		private = *pin
	}
	noFallbacks := false
	private.AllowFallbacks = &noFallbacks
	private.DataCollection = config.ProviderDataCollectionDeny
	if mode == raylineARCPrivacyModeZDR {
		zdr := true
		private.ZDR = &zdr
	}
	return &private
}
