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

// The gateway's excluded arms are read in every data mode, standard
// included, and only from a trusted gateway.
func TestGatewayExcludedArmsReadInEveryMode(t *testing.T) {
	t.Parallel()
	trust := &config.RaylineARCPolicyServiceConfig{TrustTurnSignalHeaders: true}
	tests := map[string]struct {
		policy  *config.RaylineARCPolicyServiceConfig
		headers map[string]string
		want    []bool
	}{
		"standard": {trust, map[string]string{
			raylineARCPrivacyModeHeader: "false", raylineARCExcludedArmsHeader: "arm-1",
		}, []bool{false, true, false}},
		"unstated mode": {trust, map[string]string{raylineARCExcludedArmsHeader: " arm-2 , arm-0,arm-2"}, []bool{true, false, true}},
		"privacy": {trust, map[string]string{
			raylineARCPrivacyModeHeader: "privacy", raylineARCExcludedArmsHeader: "arm-0",
		}, []bool{true, false, false}},
		"none":              {trust, map[string]string{raylineARCExcludedArmsHeader: ""}, nil},
		"unknown arm only":  {trust, map[string]string{raylineARCExcludedArmsHeader: "other"}, nil},
		"untrusted gateway": {&config.RaylineARCPolicyServiceConfig{}, map[string]string{raylineARCExcludedArmsHeader: "arm-1"}, nil},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			signals := raylineARCTrustedSignalHeaders(test.policy, test.headers)
			if got := gatewayExcludedArms(raylineARCExcludedArms(signals), privacyTestArms); !reflect.DeepEqual(got, test.want) {
				t.Fatalf("gatewayExcludedArms() = %v, want %v", got, test.want)
			}
		})
	}
	if !slices.Contains(raylineARCTurnSignalHeadersForRemoval(), raylineARCExcludedArmsHeader) {
		t.Fatal("the excluded-arms header would be forwarded upstream")
	}
}

// Two requests that differ only in their excluded arms are different turns.
func TestGatewayExcludedArmsSeparateCoalescedTurns(t *testing.T) {
	t.Parallel()
	arc := &config.RaylineARCAlgorithmConfig{}
	first := &RequestContext{Headers: map[string]string{raylineARCExcludedArmsHeader: "arm-0"}}
	second := &RequestContext{Headers: map[string]string{}}
	if raylineARCTurnInputs(arc, first, nil) == raylineARCTurnInputs(arc, second, nil) {
		t.Fatal("requests with different excluded arms share a coalescing key")
	}
}

// An excluded arm reaches the artifact excluded on a standard turn, and a
// previous arm the gateway excluded is pulled off rather than held.
func TestRaylineARCSelectorExcludesGatewayExcludedArms(t *testing.T) {
	previous := 1
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	state.PreviousArm = &previous
	scorer := visionScorer()
	selector := armedVisionSelector(scorer, visionEncoder())
	selectionContext := validARCSelectionContext(state)
	selectionContext.RaylineARC.GatewayExcludedArms = []bool{false, true}

	if _, err := selector.Select(context.Background(), selectionContext); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(scorer.excluded, []bool{false, true}) {
		t.Fatalf("scorer exclusion = %v, want the gateway-excluded arm excluded", scorer.excluded)
	}
}

// A turn only gateway-excluded arms could serve is refused with its own
// class before the encoder; one the privacy set then empties says
// no_private_arm.
func TestRaylineARCSelectorRefusesNoGatewayArm(t *testing.T) {
	tests := map[string]struct {
		gateway, privacy []bool
		want             string
	}{
		"every arm excluded":           {[]bool{true, true}, nil, arcFailureNoGatewayArm},
		"the remaining arm is private": {[]bool{true, false}, []bool{false, true}, arcFailureNoPrivateArm},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			state, err := raylinearc.NewEpisodeState(2)
			if err != nil {
				t.Fatal(err)
			}
			encoder := visionEncoder()
			selector := armedVisionSelector(visionScorer(), encoder)
			selectionContext := validARCSelectionContext(state)
			selectionContext.RaylineARC.GatewayExcludedArms = test.gateway
			selectionContext.RaylineARC.PrivacyExcludedArms = test.privacy

			_, err = selector.Select(context.Background(), selectionContext)
			var failure *raylineARCSelectionFailure
			if !errors.As(err, &failure) || failure.class != test.want {
				t.Fatalf("Select() error = %v, want class %s", err, test.want)
			}
			if encoder.calls != 0 {
				t.Fatalf("encoder calls = %d, want 0", encoder.calls)
			}
		})
	}
}

// A decision-only consult is masked as the routed turn would be.
func TestDecisionOnlyConsultCarriesTheExcludedArms(t *testing.T) {
	selector := &stubARCSelector{arm: 0, workers: decisionServiceWorkers()}
	service := &raylineARCDecisionService{router: decisionServiceRouter(t, selector)}
	algorithm := decisionServiceConfig().Decisions[0].Algorithm

	requestContext, err := service.decisionOnlyRequestContext(
		context.Background(),
		algorithm,
		routerruntime.RouteDecisionRequest{
			Body: anthropicBody("hello"), DecisionID: "rt_0a1b", SessionID: "session-1",
			Headers: map[string]string{raylineARCExcludedArmsHeader: "arm-1"},
		},
	)
	if err != nil {
		t.Fatalf("decisionOnlyRequestContext() error = %v", err)
	}
	if requestContext.Headers[raylineARCExcludedArmsHeader] != "arm-1" {
		t.Fatalf("Headers = %v, want the excluded arms carried", requestContext.Headers)
	}
}
