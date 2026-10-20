package extproc

import (
	"context"
	"errors"
	"reflect"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

var privacyTestArms = []config.ModelRef{{Model: "arm-0"}, {Model: "arm-1"}, {Model: "arm-2"}}

func privacyHeaders(mode string, allowed string) map[string]string {
	return map[string]string{
		raylineARCPrivacyModeHeader:        mode,
		raylineARCPrivacyAllowedArmsHeader: allowed,
	}
}

// The mask reads the gateway's two headers: a standard request is unmasked,
// a private one may use only the arms it names, and a private one that names
// none may use none.
func TestPrivacyExcludedArmsReadTheGatewayHeaders(t *testing.T) {
	t.Parallel()
	tests := map[string]struct {
		headers  map[string]string
		wantMode string
		want     []bool
	}{
		"standard":                  {privacyHeaders("false", "arm-0"), "", nil},
		"unstated":                  {map[string]string{}, "", nil},
		"privacy":                   {privacyHeaders("privacy", "arm-0, arm-2"), "privacy", []bool{false, true, false}},
		"zdr":                       {privacyHeaders("ZDR", "arm-1"), "zdr", []bool{true, false, true}},
		"unknown mode fails closed": {privacyHeaders("true", "arm-1"), "privacy", []bool{true, false, true}},
		"no allowed arms":           {privacyHeaders("privacy", ""), "privacy", []bool{true, true, true}},
		"unknown arm only":          {privacyHeaders("privacy", "other"), "privacy", []bool{true, true, true}},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			mode := raylineARCPrivacyMode(test.headers)
			if mode != test.wantMode {
				t.Fatalf("mode = %q, want %q", mode, test.wantMode)
			}
			if got := privacyExcludedArms(mode, test.headers, privacyTestArms); !reflect.DeepEqual(got, test.want) {
				t.Fatalf("privacyExcludedArms() = %v, want %v", got, test.want)
			}
		})
	}
}

// A cell whose gateway is not trusted reads neither header, so a client
// cannot claim a mode or an allowed set; and neither is forwarded upstream
// whatever the trust.
func TestPrivacyHeadersAreIgnoredWhenUntrusted(t *testing.T) {
	t.Parallel()
	request := privacyHeaders("privacy", "arm-0")
	untrusted := raylineARCTrustedSignalHeaders(&config.RaylineARCPolicyServiceConfig{}, request)
	if mode := raylineARCPrivacyMode(untrusted); mode != "" {
		t.Fatalf("untrusted mode = %q, want standard", mode)
	}
	if got := privacyExcludedArms(raylineARCPrivacyMode(untrusted), untrusted, privacyTestArms); got != nil {
		t.Fatalf("untrusted mask = %v, want nil", got)
	}
	trusted := raylineARCTrustedSignalHeaders(
		&config.RaylineARCPolicyServiceConfig{TrustTurnSignalHeaders: true}, request,
	)
	if got := privacyExcludedArms(raylineARCPrivacyMode(trusted), trusted, privacyTestArms); !reflect.DeepEqual(got, []bool{false, true, true}) {
		t.Fatalf("trusted mask = %v", got)
	}
	if !slices.Contains(raylineARCTurnSignalHeadersForRemoval(), raylineARCPrivacyAllowedArmsHeader) {
		t.Fatal("the allowed-arms header would be forwarded upstream")
	}
}

// Two private requests that differ only in their allowed set are different
// turns: one must never inherit a decision made over the other's arms.
func TestPrivacyAllowedArmsSeparateCoalescedTurns(t *testing.T) {
	t.Parallel()
	arc := &config.RaylineARCAlgorithmConfig{}
	first := &RequestContext{Headers: privacyHeaders("privacy", "arm-0")}
	second := &RequestContext{Headers: privacyHeaders("privacy", "arm-1")}
	if raylineARCTurnInputs(arc, first, nil) == raylineARCTurnInputs(arc, second, nil) {
		t.Fatal("requests with different allowed arms share a coalescing key")
	}
}

// Arms outside the allowed set reach the artifact excluded, beside the
// other hard constraints.
func TestRaylineARCSelectorExcludesArmsOutsideThePrivacySet(t *testing.T) {
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	scorer := visionScorer()
	selector := armedVisionSelector(scorer, visionEncoder())
	selectionContext := validARCSelectionContext(state)
	selectionContext.RaylineARC.PrivacyExcludedArms = []bool{true, false}

	if _, err := selector.Select(context.Background(), selectionContext); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(scorer.excluded, []bool{true, false}) {
		t.Fatalf("scorer exclusion = %v, want the arm outside the privacy set excluded", scorer.excluded)
	}
}

// A private turn whose previous arm is outside its allowed set is pulled off
// it, not held there by the stay margin.
func TestRaylineARCSelectorExcludesAPreviousArmOutsideThePrivacySet(t *testing.T) {
	previous := 1
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	state.PreviousArm = &previous
	scorer := visionScorer()
	selector := armedVisionSelector(scorer, visionEncoder())
	selectionContext := validARCSelectionContext(state)
	selectionContext.RaylineARC.PrivacyExcludedArms = []bool{false, true}

	if _, err := selector.Select(context.Background(), selectionContext); err != nil {
		t.Fatal(err)
	}
	if len(scorer.excluded) != 2 || !scorer.excluded[previous] {
		t.Fatalf("scorer exclusion = %v, want the previous arm excluded", scorer.excluded)
	}
}

// No allowed arm leaves nothing the request may be served by. The router
// refuses with its own class before the encoder, and never as a contended or
// replayable failure.
func TestRaylineARCSelectorRefusesNoPrivateArm(t *testing.T) {
	tests := map[string]struct {
		privacy      []bool
		imageBearing bool
		nonVision    []bool
	}{
		"every arm excluded":                    {[]bool{true, true}, false, nil},
		"the allowed arm cannot take the image": {[]bool{true, false}, true, []bool{false, true}},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			state, err := raylinearc.NewEpisodeState(2)
			if err != nil {
				t.Fatal(err)
			}
			encoder := visionEncoder()
			selector := armedVisionSelector(visionScorer(), encoder)
			selectionContext := armConstraintContext(state, test.imageBearing, test.nonVision, nil)
			selectionContext.RaylineARC.PrivacyExcludedArms = test.privacy

			_, err = selector.Select(context.Background(), selectionContext)
			var failure *raylineARCSelectionFailure
			if !errors.As(err, &failure) || failure.class != arcFailureNoPrivateArm {
				t.Fatalf("Select() error = %v, want class %s", err, arcFailureNoPrivateArm)
			}
			if encoder.calls != 0 {
				t.Fatalf("encoder calls = %d, want 0", encoder.calls)
			}
			if failure.contended() {
				t.Fatal("no_private_arm must not answer 429")
			}
			// The decision-only API answers it as ExtProc does: 422.
			if !errors.Is(selectionFailureError(err), routerruntime.ErrRouteDecisionUndecidable) {
				t.Fatal("the decision-only API would not answer no_private_arm 422")
			}
		})
	}
}
