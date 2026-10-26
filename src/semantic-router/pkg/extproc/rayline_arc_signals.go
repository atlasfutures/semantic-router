package extproc

import (
	"encoding/json"
	"errors"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// Caller-facing content signals (pathfinder #5943 and
// docs/plans/2026-10-25-decisions-topic-signal-contract.md): routing.signals
// names signals such as topic.operation; the policy service admits or refuses
// them and answers a signals block keyed by name. They are metadata for the
// caller and change no route, so they are accepted on /v1/routes and on
// generation turns alike.
//
// The router is signal-name generic. It checks only the member's shape and
// forwards the names as the decide request's signals; it keeps no list of
// signal names, passes the block through as opaque JSON, and answers a
// service refusal with the service's own reason as the public code.

// maxRoutingSignals bounds routing.signals. It is a shape limit, not a list
// of admitted names.
const maxRoutingSignals = 32

// arcFailureSignalRefusedPrefix prefixes a signal refusal's selection
// failure class; the rest is the service's bounded reason.
const arcFailureSignalRefusedPrefix = "signal_refused."

// arcFailureRoutesSignalsMissing: a lookup asked for signals and the
// service's answer carries no signals block.
const arcFailureRoutesSignalsMissing = "routes_signals_missing"

// readRoutingSignals reads routing.signals: an array of distinct non-empty
// strings. An empty array is no signals (nil): the answer is today's.
func readRoutingSignals(raw json.RawMessage) ([]string, *routesContractError) {
	var names []string
	if json.Unmarshal(raw, &names) != nil || names == nil || len(names) > maxRoutingSignals {
		return nil, routesInvalid("signals_invalid",
			"routing.signals must be an array of at most 32 distinct, non-empty signal names.")
	}
	if len(names) == 0 {
		return nil, nil
	}
	seen := make(map[string]bool, len(names))
	for _, name := range names {
		if name == "" || len(name) > 128 || seen[name] {
			return nil, routesInvalid("signals_invalid",
				"routing.signals must be an array of at most 32 distinct, non-empty signal names.")
		}
		seen[name] = true
	}
	return names, nil
}

// turnSignals are the signals a decide request names: a lookup's
// routing.signals, or a generation turn's.
func turnSignals(arcContext *selection.RaylineARCSelectionContext) []string {
	if arcContext == nil {
		return nil
	}
	if controls := arcContext.RouteControls; controls != nil && len(controls.Signals) > 0 {
		return append([]string(nil), controls.Signals...)
	}
	if len(arcContext.TurnSignals) > 0 {
		return append([]string(nil), arcContext.TurnSignals...)
	}
	return nil
}

// signalRefusalClass is the selection failure class of the service's
// refusal of a request's signals, or "".
func signalRefusalClass(err error) string {
	var failure *raylinearc.PolicyServiceError
	if !errors.As(err, &failure) || failure.Class != raylinearc.PolicySignalRefusedClass {
		return ""
	}
	reason, _ := failure.Detail["reason"].(string)
	if reason == "" {
		return ""
	}
	return arcFailureSignalRefusedPrefix + reason
}

// routeSignalRefusal is a signal refusal on its way to the caller, which it
// answers with the service's reason as the code.
type routeSignalRefusal struct {
	reason string
	err    error
}

func (refusal *routeSignalRefusal) Error() string {
	if refusal.err == nil {
		return arcFailureSignalRefusedPrefix + refusal.reason
	}
	return arcFailureSignalRefusedPrefix + refusal.reason + ": " + refusal.err.Error()
}

func (refusal *routeSignalRefusal) Unwrap() error { return refusal.err }

// signalRefusalOf is the refusal a selection failure class names, or nil.
func signalRefusalOf(class string) *routeSignalRefusal {
	reason, ok := strings.CutPrefix(class, arcFailureSignalRefusedPrefix)
	if !ok || reason == "" {
		return nil
	}
	return &routeSignalRefusal{reason: reason}
}

// contract is the refusal as the public error. A name ARC does not define,
// or one this cell does not serve, is the caller's 400; anything else (the
// cell's heads failed to load) is the cell's 503. The code is the service's
// reason, unchanged.
func (refusal *routeSignalRefusal) contract() *routesContractError {
	switch refusal.reason {
	case "signal_not_supported":
		return routesInvalid(refusal.reason, "routing.signals names a signal ARC does not define.")
	case "signal_not_admitted":
		return routesInvalid(refusal.reason, "routing.signals names a signal this ARC cell does not serve.")
	}
	return &routesContractError{
		status: 503, errType: "api_error", code: refusal.reason,
		message: "This ARC cell cannot serve routing.signals right now.",
	}
}
