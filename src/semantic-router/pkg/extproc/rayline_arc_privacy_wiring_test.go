//go:build !windows && cgo

package extproc

import (
	"bytes"
	"context"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// privacyWiringTurn carries one request's privacy headers through the real
// stages in order -- the ARC context the routed path builds from the request
// headers, the selector's hard mask, and the provider object the dispatch
// writes -- so a break between header capture, selection and dispatch fails
// here rather than only in a unit that was handed the mode.
func privacyWiringTurn(t *testing.T, trusted bool, requestHeaders map[string]string) (
	*RequestContext, []bool, error, []byte,
) {
	t.Helper()
	router := providerPinRouter("https://openrouter.ai/api/v1", providerPinPreferences())
	algorithm := &config.AlgorithmConfig{
		Type: config.RaylineARCAlgorithmType,
		RaylineARC: &config.RaylineARCAlgorithmConfig{
			Episode:       config.RaylineARCEpisodeConfig{IDHeader: "x-rayline-session"},
			PolicyService: &config.RaylineARCPolicyServiceConfig{TrustTurnSignalHeaders: trusted},
		},
	}
	requestContext := &RequestContext{
		Headers:           requestHeaders,
		SourceFormat:      llmprotocol.AnthropicMessagesV1,
		RaylineARCRawBody: []byte(`{"messages":[{"role":"user","content":"hello"}]}`),
		SemanticRequest:   testNeutralRequest("auto", "hello"),
	}
	built := router.buildRaylineARCSelectionContext(
		algorithm, requestContext, []config.ModelRef{{Model: "arm-0"}, {Model: providerPinArm}}, raylineARCEpisodeEphemeral,
	)
	if built.PreparationFailure != "" {
		t.Fatalf("preparation failed: %s", built.PreparationFailure)
	}

	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	scorer := visionScorer()
	selectionContext := validARCSelectionContext(state)
	selectionContext.RaylineARC.PrivacyExcludedArms = built.PrivacyExcludedArms
	_, selectErr := armedVisionSelector(scorer, visionEncoder()).Select(context.Background(), selectionContext)

	dispatch := &providerDispatch{
		logicalModel: providerPinArm, decisionName: providerPinDecision,
		targetFormat: llmprotocol.OpenAIChatV1, profile: openRouterProviderProfile(),
	}
	body, err := router.adaptProviderRequest([]byte(`{"model":"m"}`), dispatch, requestContext)
	if err != nil {
		t.Fatalf("adaptProviderRequest: %v", err)
	}
	return requestContext, scorer.excluded, selectErr, body
}

func TestPrivacyHeadersReachSelectionAndDispatch(t *testing.T) {
	// A trusted zdr request allowing only the pinned arm: the other arm is
	// masked, and the pinned arm's provider object carries the data mode.
	requestContext, excluded, err, body := privacyWiringTurn(t, true, map[string]string{
		raylineARCPrivacyModeHeader:        "zdr",
		raylineARCPrivacyAllowedArmsHeader: providerPinArm,
	})
	if err != nil {
		t.Fatalf("Select() error = %v", err)
	}
	if requestContext.RaylineARCPrivacyMode != raylineARCPrivacyModeZDR || !reflect.DeepEqual(excluded, []bool{true, false}) {
		t.Fatalf("mode %q, scorer exclusion %v, want zdr and the arm outside the set masked", requestContext.RaylineARCPrivacyMode, excluded)
	}
	want := `"provider":{"order":["deepinfra","together"],"allow_fallbacks":false,"data_collection":"deny","zdr":true}`
	if !bytes.Contains(body, []byte(want)) {
		t.Fatalf("dispatched body lacks the private pin, want %s:\n%s", want, body)
	}

	// The same headers on a cell that does not trust them change nothing.
	requestContext, excluded, err, body = privacyWiringTurn(t, false, map[string]string{
		raylineARCPrivacyModeHeader:        "zdr",
		raylineARCPrivacyAllowedArmsHeader: providerPinArm,
	})
	if err != nil || requestContext.RaylineARCPrivacyMode != "" || excluded != nil ||
		!bytes.Contains(body, []byte(providerPinWireBytes+"}")) {
		t.Fatalf("an untrusted request was treated as private: err %v, mode %q, excluded %v:\n%s",
			err, requestContext.RaylineARCPrivacyMode, excluded, body)
	}

	// A trusted private request that allows no arm of the basket is refused
	// 422 no_private_arm on the routed path's admission response.
	_, _, err, _ = privacyWiringTurn(t, true, map[string]string{
		raylineARCPrivacyModeHeader:        "privacy",
		raylineARCPrivacyAllowedArmsHeader: "some-other-arm",
	})
	response := (&OpenAIRouter{}).authoritativeSelectionFailureResponse(
		&modelSelectionFailure{algorithm: configRaylineARC, class: selectionFailureClassOf(t, err)},
		&RequestContext{},
	)
	if code := int(response.GetImmediateResponse().GetStatus().GetCode()); code != 422 ||
		immediateHeaderValue(response, headers.VSRFailureClass) != selectionFailureNoPrivateArm {
		t.Fatalf("status %d class %q, want 422 no_private_arm", code, immediateHeaderValue(response, headers.VSRFailureClass))
	}
}
