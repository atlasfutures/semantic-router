//go:build !windows && cgo

package extproc

import (
	"bytes"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/reasoning"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func privateProviderPinRouter(standard, private *config.OpenRouterProviderPreferences) *OpenAIRouter {
	router := providerPinRouter("https://openrouter.ai/api/v1", standard)
	arm := router.Config.ModelConfig[providerPinArm]
	arm.PrivateProviderPreferences = private
	router.Config.ModelConfig[providerPinArm] = arm
	return router
}

func privateProviderPinPreferences() *config.OpenRouterProviderPreferences {
	fallbacks := true
	return &config.OpenRouterProviderPreferences{
		Order: []string{"phala", "novita"}, Only: []string{"phala", "novita"}, AllowFallbacks: &fallbacks,
		MaxPrice: map[string]float64{"completion": 3},
	}
}

// A private turn on an arm that declares a private pin sends that pin, with
// the data mode added as it is to the standard pin; a standard turn on the
// same arm, and any turn on an arm without one, sends the bytes it sent before
// the key existed.
func TestPrivateProviderPinReplacesTheStandardPinOnPrivateTurnsOnly(t *testing.T) {
	tests := map[string]struct {
		mode    string
		private *config.OpenRouterProviderPreferences
		want    string
	}{
		"standard turn, arm with a private pin": {
			"", privateProviderPinPreferences(), providerPinWireBytes,
		},
		"privacy turn, arm with a private pin": {
			raylineARCPrivacyModePrivacy, privateProviderPinPreferences(),
			`"provider":{"order":["phala","novita"],"allow_fallbacks":false,"only":["phala","novita"],"data_collection":"deny","max_price":{"completion":3}}`,
		},
		"zdr turn, arm with a private pin": {
			raylineARCPrivacyModeZDR, privateProviderPinPreferences(),
			`"provider":{"order":["phala","novita"],"allow_fallbacks":false,"only":["phala","novita"],"data_collection":"deny","zdr":true,"max_price":{"completion":3}}`,
		},
		"standard turn, arm without a private pin": {
			"", nil, providerPinWireBytes,
		},
		"zdr turn, arm without a private pin": {
			raylineARCPrivacyModeZDR, nil,
			`"provider":{"order":["deepinfra","together"],"allow_fallbacks":false,"data_collection":"deny","zdr":true}`,
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			router := privateProviderPinRouter(providerPinPreferences(), test.private)
			request := testNeutralRequest("auto", "route this arm")
			ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
			ctx.RaylineARCPrivacyMode = test.mode
			raw := providerPinRoute(t, router, request, ctx)
			if !bytes.Contains(raw, []byte(test.want)) {
				t.Fatalf("provider object is wrong, want %s:\n%s", test.want, raw)
			}
		})
	}
}

// A private pin leaves the card untouched: a standard turn after a private
// one sends the standard pin.
func TestPrivateProviderPinLeavesTheNextStandardTurnUnchanged(t *testing.T) {
	router := privateProviderPinRouter(providerPinPreferences(), privateProviderPinPreferences())
	request := testNeutralRequest("auto", "route this arm")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.RaylineARCPrivacyMode = raylineARCPrivacyModeZDR
	providerPinRoute(t, router, request, ctx)

	if raw := providerPinRouteBytes(t, router); !bytes.Contains(raw, []byte(providerPinWireBytes)) {
		t.Fatalf("a standard turn after a private one changed its pin:\n%s", raw)
	}
}

// The endpoint constraint judged against the gateway's allowed endpoints is
// built only for private turns, so it lists the private pin's providers.
func TestArmEndpointConstraintReadsThePrivatePin(t *testing.T) {
	router := privateProviderPinRouter(providerPinPreferences(), privateProviderPinPreferences())
	constraint := router.armEndpointConstraint(providerPinArm, []string{"phala"})
	if constraint == nil || len(constraint.Order) != 2 || constraint.Order[0] != "phala" || constraint.Only[1] != "novita" {
		t.Fatalf("constraint = %#v, want the private pin's providers", constraint)
	}

	standardOnly := privateProviderPinRouter(providerPinPreferences(), nil)
	constraint = standardOnly.armEndpointConstraint(providerPinArm, []string{"deepinfra"})
	if constraint == nil || len(constraint.Order) != 2 || constraint.Order[0] != "deepinfra" {
		t.Fatalf("constraint = %#v, want the standard pin's providers on an arm without a private pin", constraint)
	}
}

// The issuer of a turn's opaque reasoning is the provider its own pin fixes:
// a private turn's private pin, a standard turn's standard pin.
func TestOpaqueReasoningIssuerReadsTheTurnsPin(t *testing.T) {
	noFallbacks := false
	private := &config.OpenRouterProviderPreferences{Order: []string{"azure"}, AllowFallbacks: &noFallbacks}
	router := privateProviderPinRouter(providerPinPreferences(), private)
	dispatch, err := router.resolveProviderDispatch(providerPinArm, "", false, llmprotocol.OpenAIChatV1)
	if err != nil {
		t.Fatal(err)
	}
	standard := router.opaqueReasoningIssuerFor(dispatch, "")
	zdr := router.opaqueReasoningIssuerFor(dispatch, raylineARCPrivacyModeZDR)
	if zdr == reasoning.ReasoningIssuerUnknown || zdr == standard {
		t.Fatalf("issuer standard=%q zdr=%q: the zdr turn should be issued by its private pin's single provider", standard, zdr)
	}
}

// A private turn never falls back, so a private pin that orders one provider
// fixes the issuer even when the card leaves fallbacks allowed.
func TestOpaqueReasoningIssuerAppliesThePrivateDataMode(t *testing.T) {
	fallbacks := true
	private := &config.OpenRouterProviderPreferences{Order: []string{"azure"}, AllowFallbacks: &fallbacks}
	router := privateProviderPinRouter(providerPinPreferences(), private)
	dispatch, err := router.resolveProviderDispatch(providerPinArm, "", false, llmprotocol.OpenAIChatV1)
	if err != nil {
		t.Fatal(err)
	}
	if issuer := router.opaqueReasoningIssuerFor(dispatch, raylineARCPrivacyModePrivacy); issuer == reasoning.ReasoningIssuerUnknown {
		t.Fatal("a private turn pinned to one provider has no known issuer")
	}
}

// A private turn's failure is recorded against its own route: an arm with a
// private pin has a separate private route, one without shares the standard.
func TestPolicyWorkerRouteSeparatesThePrivatePin(t *testing.T) {
	router := privateProviderPinRouter(providerPinPreferences(), privateProviderPinPreferences())
	ref := config.ModelRef{Model: providerPinArm}
	standard, private := policyWorkerRoute(router.Config, ref, false), policyWorkerRoute(router.Config, ref, true)
	if standard == "" || private == "" || standard == private {
		t.Fatalf("routes standard=%q private=%q, want two distinct routes", standard, private)
	}
	shared := privateProviderPinRouter(providerPinPreferences(), nil)
	if policyWorkerRoute(shared.Config, ref, false) != policyWorkerRoute(shared.Config, ref, true) {
		t.Fatal("an arm without a private pin has a separate private route")
	}
	scorer := &policyServiceScorer{routes: []string{"standard"}, privateRoutes: []string{"private"}}
	if scorer.routeFor(0, false) != "standard" || scorer.routeFor(0, true) != "private" {
		t.Fatal("routeFor did not pick the turn's route")
	}
}
