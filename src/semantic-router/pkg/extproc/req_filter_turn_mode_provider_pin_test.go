//go:build !windows && cgo

package extproc

import (
	"bytes"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/reasoning"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// turnModeProviderPinRouter is the provider-pin router with the arm's privacy
// and zdr overrides set (nil leaves one undeclared).
func turnModeProviderPinRouter(standard, privacy, zdr *config.OpenRouterProviderPreferences) *OpenAIRouter {
	router := providerPinRouter("https://openrouter.ai/api/v1", standard)
	arm := router.Config.ModelConfig[providerPinArm]
	arm.StandardProviderPreferences = privacy
	arm.ZDRProviderPreferences = zdr
	router.Config.ModelConfig[providerPinArm] = arm
	return router
}

func zdrProviderPinPreferences() *config.OpenRouterProviderPreferences {
	fallbacks := true
	return &config.OpenRouterProviderPreferences{
		Order: []string{"phala", "novita"}, Only: []string{"phala", "novita"}, AllowFallbacks: &fallbacks,
		MaxPrice: map[string]float64{"completion": 3},
	}
}

func privacyProviderPinPreferences() *config.OpenRouterProviderPreferences {
	return &config.OpenRouterProviderPreferences{Order: []string{"groq"}, Only: []string{"groq"}}
}

const (
	zdrPinWireBytes         = `"provider":{"order":["phala","novita"],"allow_fallbacks":false,"only":["phala","novita"],"data_collection":"deny","zdr":true,"max_price":{"completion":3}}`
	privacyPinWireBytes     = `"provider":{"order":["groq"],"allow_fallbacks":false,"only":["groq"],"data_collection":"deny"}`
	regularPrivacyWireBytes = `"provider":{"order":["deepinfra","together"],"allow_fallbacks":false,"data_collection":"deny"}`
	regularZDRTurnWireBytes = `"provider":{"order":["deepinfra","together"],"allow_fallbacks":false,"data_collection":"deny","zdr":true}`
)

// Each turn mode sends its own pin, with its data mode added: a privacy turn
// the privacy override when the arm declares one and the regular pin when it
// does not, never the zdr pin; a zdr turn the zdr pin when declared and the
// regular pin when not, never the privacy override. A standard turn sends the
// bytes it sent before either key existed.
func TestTurnModeProviderPinsApplyToTheirOwnModeOnly(t *testing.T) {
	tests := map[string]struct {
		mode         string
		privacy, zdr *config.OpenRouterProviderPreferences
		want, never  string
	}{
		"standard turn, arm with both overrides": {
			"", privacyProviderPinPreferences(), zdrProviderPinPreferences(), providerPinWireBytes, "",
		},
		"privacy turn, arm with both overrides": {
			raylineARCPrivacyModePrivacy, privacyProviderPinPreferences(), zdrProviderPinPreferences(),
			privacyPinWireBytes, `"phala"`,
		},
		"privacy turn, arm with only a zdr pin": {
			raylineARCPrivacyModePrivacy, nil, zdrProviderPinPreferences(), regularPrivacyWireBytes, `"phala"`,
		},
		"zdr turn, arm with both overrides": {
			raylineARCPrivacyModeZDR, privacyProviderPinPreferences(), zdrProviderPinPreferences(),
			zdrPinWireBytes, `"groq"`,
		},
		"zdr turn, arm with only a privacy override": {
			raylineARCPrivacyModeZDR, privacyProviderPinPreferences(), nil, regularZDRTurnWireBytes, `"groq"`,
		},
		"standard turn, arm without overrides": {
			"", nil, nil, providerPinWireBytes, "",
		},
		"zdr turn, arm without overrides": {
			raylineARCPrivacyModeZDR, nil, nil, regularZDRTurnWireBytes, "",
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			router := turnModeProviderPinRouter(providerPinPreferences(), test.privacy, test.zdr)
			request := testNeutralRequest("auto", "route this arm")
			ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
			ctx.RaylineARCPrivacyMode = test.mode
			raw := providerPinRoute(t, router, request, ctx)
			if !bytes.Contains(raw, []byte(test.want)) {
				t.Fatalf("provider object is wrong, want %s:\n%s", test.want, raw)
			}
			if test.never != "" && bytes.Contains(raw, []byte(test.never)) {
				t.Fatalf("the other mode's pin leaked into the provider object (%s):\n%s", test.never, raw)
			}
		})
	}
}

// An override leaves the card untouched: a standard turn after a zdr one
// sends the regular pin.
func TestZDRProviderPinLeavesTheNextStandardTurnUnchanged(t *testing.T) {
	router := turnModeProviderPinRouter(providerPinPreferences(), privacyProviderPinPreferences(), zdrProviderPinPreferences())
	request := testNeutralRequest("auto", "route this arm")
	ctx := routingTestContext(llmprotocol.OpenAIChatV1, request)
	ctx.RaylineARCPrivacyMode = raylineARCPrivacyModeZDR
	providerPinRoute(t, router, request, ctx)

	if raw := providerPinRouteBytes(t, router); !bytes.Contains(raw, []byte(providerPinWireBytes)) {
		t.Fatalf("a standard turn after a zdr one changed its pin:\n%s", raw)
	}
}

// The endpoint constraint judged against the gateway's allowed endpoints is
// the pin the turn's own mode sends.
func TestArmEndpointConstraintReadsTheTurnModesPin(t *testing.T) {
	router := turnModeProviderPinRouter(providerPinPreferences(), nil, zdrProviderPinPreferences())
	constraint := router.armEndpointConstraint(providerPinArm, []string{"phala"}, raylineARCPrivacyModeZDR)
	if constraint == nil || len(constraint.Order) != 2 || constraint.Order[0] != "phala" || constraint.Only[1] != "novita" {
		t.Fatalf("zdr constraint = %#v, want the zdr pin's providers", constraint)
	}
	constraint = router.armEndpointConstraint(providerPinArm, []string{"deepinfra"}, raylineARCPrivacyModePrivacy)
	if constraint == nil || len(constraint.Order) != 2 || constraint.Order[0] != "deepinfra" {
		t.Fatalf("privacy constraint = %#v, want the regular pin's providers, not the zdr pin's", constraint)
	}

	withPrivacy := turnModeProviderPinRouter(providerPinPreferences(), privacyProviderPinPreferences(), zdrProviderPinPreferences())
	constraint = withPrivacy.armEndpointConstraint(providerPinArm, []string{"groq"}, raylineARCPrivacyModePrivacy)
	if constraint == nil || len(constraint.Order) != 1 || constraint.Order[0] != "groq" {
		t.Fatalf("privacy constraint = %#v, want the privacy override's providers", constraint)
	}
}

// The issuer of a turn's opaque reasoning is the provider its own pin fixes:
// a zdr turn's zdr pin, a privacy turn's regular pin when only a zdr pin is
// declared, a standard turn's regular pin.
func TestOpaqueReasoningIssuerReadsTheTurnsPin(t *testing.T) {
	noFallbacks := false
	zdr := &config.OpenRouterProviderPreferences{Order: []string{"azure"}, AllowFallbacks: &noFallbacks}
	router := turnModeProviderPinRouter(providerPinPreferences(), nil, zdr)
	dispatch, err := router.resolveProviderDispatch(providerPinArm, "", false, llmprotocol.OpenAIChatV1)
	if err != nil {
		t.Fatal(err)
	}
	zdrIssuer := router.opaqueReasoningIssuerFor(dispatch, raylineARCPrivacyModeZDR)
	privacyIssuer := router.opaqueReasoningIssuerFor(dispatch, raylineARCPrivacyModePrivacy)
	if zdrIssuer == reasoning.ReasoningIssuerUnknown || zdrIssuer == privacyIssuer {
		t.Fatalf("issuer privacy=%q zdr=%q: the zdr turn should be issued by its zdr pin's single provider", privacyIssuer, zdrIssuer)
	}
}

// A private turn never falls back, so a privacy override that orders one
// provider fixes the issuer even when the card leaves fallbacks allowed.
func TestOpaqueReasoningIssuerAppliesThePrivateDataMode(t *testing.T) {
	fallbacks := true
	privacy := &config.OpenRouterProviderPreferences{Order: []string{"azure"}, AllowFallbacks: &fallbacks}
	router := turnModeProviderPinRouter(providerPinPreferences(), privacy, nil)
	dispatch, err := router.resolveProviderDispatch(providerPinArm, "", false, llmprotocol.OpenAIChatV1)
	if err != nil {
		t.Fatal(err)
	}
	if issuer := router.opaqueReasoningIssuerFor(dispatch, raylineARCPrivacyModePrivacy); issuer == reasoning.ReasoningIssuerUnknown {
		t.Fatal("a privacy turn pinned to one provider has no known issuer")
	}
}

// A privacy or zdr turn's failure is recorded against its own mode's route:
// an arm with an override for the mode has a separate route for it, and a
// mode without one shares the standard route.
func TestPolicyWorkerRouteSeparatesTheTurnModePins(t *testing.T) {
	ref := config.ModelRef{Model: providerPinArm}
	router := turnModeProviderPinRouter(providerPinPreferences(), privacyProviderPinPreferences(), zdrProviderPinPreferences())
	standard := policyWorkerRoute(router.Config, ref, "")
	privacy := policyWorkerRoute(router.Config, ref, raylineARCPrivacyModePrivacy)
	zdr := policyWorkerRoute(router.Config, ref, raylineARCPrivacyModeZDR)
	if standard == "" || privacy == "" || zdr == "" || standard == privacy || standard == zdr || privacy == zdr {
		t.Fatalf("routes standard=%q privacy=%q zdr=%q, want three distinct routes", standard, privacy, zdr)
	}
	zdrOnly := turnModeProviderPinRouter(providerPinPreferences(), nil, zdrProviderPinPreferences())
	if policyWorkerRoute(zdrOnly.Config, ref, raylineARCPrivacyModePrivacy) != policyWorkerRoute(zdrOnly.Config, ref, "") {
		t.Fatal("an arm without a privacy override has a separate privacy route")
	}
	if policyWorkerRoute(zdrOnly.Config, ref, raylineARCPrivacyModeZDR) == policyWorkerRoute(zdrOnly.Config, ref, "") {
		t.Fatal("an arm with a zdr pin shares its zdr route with the standard one")
	}
	scorer := &policyServiceScorer{routes: []string{"standard"}, privacyRoutes: []string{"privacy"}, zdrRoutes: []string{"zdr"}}
	for mode, want := range map[string]string{
		"": "standard", raylineARCPrivacyModePrivacy: "privacy", raylineARCPrivacyModeZDR: "zdr", "unknown": "zdr",
	} {
		if got := scorer.routeFor(0, mode); got != want {
			t.Fatalf("routeFor(%q) = %q, want %q", mode, got, want)
		}
	}
}
