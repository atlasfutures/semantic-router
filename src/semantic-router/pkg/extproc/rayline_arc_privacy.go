package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
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
	// raylineARCPrivacyAllowedEndpointsHeader is the OpenRouter endpoint
	// tags each allowed arm may be served by, <arm>=<tag>|<tag>,<arm>=<tag>,
	// read only beside the allowed arms. When present it lists every allowed
	// arm; an arm it leaves out is excluded. Absent, every endpoint an
	// allowed arm reaches may serve it, as before the header.
	raylineARCPrivacyAllowedEndpointsHeader = "x-rayline-privacy-allowed-endpoints"

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

// privacyAllowedEndpoints is a private request's per-arm endpoint allow list
// as its trusted header states it. A header that does not parse is
// malformed and allows no arm.
type privacyAllowedEndpoints struct {
	present   bool
	malformed bool
	// arms maps an arm id, as the allowed arms name it, to its allowed
	// endpoint tags, lower-cased.
	arms map[string][]string
}

// raylineARCPrivacyAllowedEndpoints parses the per-arm endpoint allow list of
// a private request that states its allowed arms; the zero value (absent)
// otherwise. The format is the gateway's (memex-desktop
// buildRaylineDataUseHeaders): arm=tag|tag entries joined by ",", an arm id
// possibly holding "@", and arm= for an allowed arm with no endpoints. Tags
// never contain ",", "=" or "|", so an entry is malformed when it has no "=",
// an empty arm, an empty tag in a non-empty list, a tag with "=", or repeats
// an arm.
func raylineARCPrivacyAllowedEndpoints(mode string, signalHeaders map[string]string) privacyAllowedEndpoints {
	raw := strings.TrimSpace(signalHeaders[raylineARCPrivacyAllowedEndpointsHeader])
	if mode == "" || raw == "" || strings.TrimSpace(signalHeaders[raylineARCPrivacyAllowedArmsHeader]) == "" {
		return privacyAllowedEndpoints{}
	}
	malformed := privacyAllowedEndpoints{present: true, malformed: true}
	parsed := privacyAllowedEndpoints{present: true, arms: map[string][]string{}}
	for _, entry := range strings.Split(raw, ",") {
		arm, list, ok := strings.Cut(entry, "=")
		arm = strings.TrimSpace(arm)
		if _, repeated := parsed.arms[arm]; !ok || arm == "" || repeated {
			return malformed
		}
		tags := []string{}
		if strings.TrimSpace(list) == "" {
			// arm= is an allowed arm with no endpoints to pick among (the
			// gateway writes it for a direct arm with no pin): its pin goes
			// out as it is.
			parsed.arms[arm] = tags
			continue
		}
		for _, tag := range strings.Split(list, "|") {
			tag = strings.ToLower(strings.TrimSpace(tag))
			if tag == "" || strings.Contains(tag, "=") {
				return malformed
			}
			if !slices.Contains(tags, tag) {
				tags = append(tags, tag)
			}
		}
		parsed.arms[arm] = tags
	}
	return parsed
}

// withPrivacyEndpointConstraints folds the endpoint allow list into a private
// request's arm mask: a malformed list excludes every arm and an allowed arm
// it leaves out is excluded, both failing closed. Each remaining OpenRouter
// arm's constraint is returned by ordinal for the selector, which masks it
// when none of its reachable endpoints is allowed; a non-OpenRouter arm, or
// one listed with no endpoints (arm=), has none to narrow and is judged by
// its presence alone.
func (r *OpenAIRouter) withPrivacyEndpointConstraints(
	excluded []bool,
	endpoints privacyAllowedEndpoints,
	modelRefs []config.ModelRef,
) ([]bool, []*raylinearc.ArmEndpoints) {
	if !endpoints.present || len(excluded) != len(modelRefs) {
		return excluded, nil
	}
	constraints := make([]*raylinearc.ArmEndpoints, len(modelRefs))
	for index, ref := range modelRefs {
		if excluded[index] {
			continue
		}
		allowed, listed := endpoints.arms[strings.TrimSpace(ref.Model)]
		if endpoints.malformed || !listed {
			excluded[index] = true
			continue
		}
		if len(allowed) == 0 {
			// Listed as arm=: allowed, with no endpoints to narrow by.
			continue
		}
		constraints[index] = r.armEndpointConstraint(ref.Model, allowed)
	}
	return excluded, constraints
}

// armEndpointConstraint is an OpenRouter arm's endpoint constraint, or nil for
// an arm dispatched elsewhere.
func (r *OpenAIRouter) armEndpointConstraint(model string, allowed []string) *raylinearc.ArmEndpoints {
	if r == nil || r.Config == nil {
		return nil
	}
	_, backend, _, _ := r.Config.ResolvePrimaryBackendForModel(model)
	if profile, err := r.Config.GetProviderProfileForEndpoint(backend); err != nil || !providerIsOpenRouter(profile) {
		return nil
	}
	constraint := &raylinearc.ArmEndpoints{Model: r.Config.ResolveExternalModelID(model, backend), Allowed: allowed}
	// Only a private turn narrows by endpoint, so the constraint is the arm's
	// private pin when it declares one: the endpoints the turn can reach.
	if pin := armProviderPin(r.Config, model, raylineARCPrivacyModePrivacy); pin != nil {
		constraint.Only, constraint.Order, constraint.Ignore = pin.Only, pin.Order, pin.Ignore
	}
	return constraint
}

// withPrivacyEndpointExclusions masks, once the routing-price feed is known,
// each arm none of whose reachable endpoints the gateway allows. Without a
// feed the cell cannot say what an arm reaches, so arms are judged by their
// presence in the list alone and their turns are sent unpicked.
func withPrivacyEndpointExclusions(
	arcContext *selection.RaylineARCSelectionContext,
	settlement *raylinearc.RoutingPriceSettlement,
) {
	arcContext.EndpointFeed = settlement
	if settlement == nil || len(arcContext.PrivacyArmEndpoints) != len(arcContext.PrivacyExcludedArms) {
		return
	}
	for index, constraint := range arcContext.PrivacyArmEndpoints {
		if constraint != nil && len(constraint.AllowedReach(settlement)) == 0 {
			arcContext.PrivacyExcludedArms[index] = true
		}
	}
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

// armProviderPin is the pin a turn in the given privacy mode sends for the
// model before its data-use members are added: on a private turn (privacy or
// zdr) the card's private_provider_preferences when it declares one, otherwise
// its provider_preferences. A standard turn ("") always gets the latter.
func armProviderPin(
	cfg *config.RouterConfig,
	model string,
	privacyMode string,
) *config.OpenRouterProviderPreferences {
	if cfg == nil {
		return nil
	}
	return cfg.ProviderPreferencesForTurn(model, privacyMode != "")
}

// privacyProviderPreferences is the provider object a private turn sends: the
// arm's pin (armProviderPin), or on an unpinned arm nothing but the data-use members, with
// OpenRouter told to use only providers that do not collect the data and not
// to fall back past them; a zdr turn also asks for zero-data-retention
// endpoints only. It replaces any provider object the request carries: the
// cell does not rely on the gateway to have kept a caller's provider choice
// within the request's privacy mode. A standard turn's pin is unchanged.
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
