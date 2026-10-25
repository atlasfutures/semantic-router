package protocolcodec

import (
	"strings"
	"testing"
)

const testMarkerDigest = "AbCdEfGhIjKlMnOpQrStUvWxYz012-_9"

func TestRouterSignatureRecognisesOnlyTheExactMarkerShape(t *testing.T) {
	cases := []struct {
		name      string
		signature string
		family    string
		reserved  bool
	}{
		{"exact marker", "vsr.thinking.v1.moonshotai." + testMarkerDigest, "moonshotai", true},
		{"family with digit and dash", "vsr.thinking.v1.z-ai2." + testMarkerDigest, "z-ai2", true},
		{"anthropic signature", "EqQBCkgIARABGAIiQL2xRnd3", "", false},
		{"empty", "", "", false},
		{"openclaw sentinel", "reasoning_content", "", false},
		{"prefix lookalike without dot", "vsrthinking.v1.moonshotai." + testMarkerDigest, "", false},
		{"bare namespace", "vsr.", "", true},
		{"other router blob", "vsr.reasoning_details.v1.e30", "", true},
		{"future version", "vsr.thinking.v2.moonshotai." + testMarkerDigest, "", true},
		{"uppercase family", "vsr.thinking.v1.MoonShot." + testMarkerDigest, "", true},
		{"empty family", "vsr.thinking.v1.." + testMarkerDigest, "", true},
		{"long family", "vsr.thinking.v1." + strings.Repeat("a", 65) + "." + testMarkerDigest, "", true},
		{"short digest", "vsr.thinking.v1.moonshotai.abc", "", true},
		{"long digest", "vsr.thinking.v1.moonshotai." + testMarkerDigest + "x", "", true},
		{"digest outside base64url", "vsr.thinking.v1.moonshotai." + strings.Repeat("+", 32), "", true},
		{"extra segment", "vsr.thinking.v1.moonshotai.x." + testMarkerDigest, "", true},
		{"no digest", "vsr.thinking.v1.moonshotai", "", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			family, reserved := routerSignature(tc.signature)
			if family != tc.family || reserved != tc.reserved {
				t.Fatalf("routerSignature(%q) = (%q, %v), want (%q, %v)", tc.signature, family, reserved, tc.family, tc.reserved)
			}
		})
	}
}

func TestThinkingMarkerFamilyLabelNamesMalformedValues(t *testing.T) {
	if got := thinkingMarkerFamilyLabel("deepseek"); got != "deepseek" {
		t.Fatalf("label = %q", got)
	}
	if got := thinkingMarkerFamilyLabel(""); got != "malformed" {
		t.Fatalf("label = %q", got)
	}
}
