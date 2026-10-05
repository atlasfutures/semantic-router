package protocolcodec

import "strings"

// A thinking marker is a signature the Router gives thinking that no provider
// signed, so a Messages client keeps the block as thinking when it resends
// history instead of turning it into text (atlasfutures/semantic-router#191):
//
//	vsr.thinking.v1.<family>.<digest>
//	  family = the worker family, [a-z0-9-]+
//	  digest = 32 characters of base64url
//
// A marker proves nothing. It is not Anthropic's, and no provider may be sent
// one: the request decoders strip it, so the block is ordinary unsigned
// reasoning again and the reasoning carry rules decide where it goes, and
// every request encoder refuses one as a second guard.
//
// The contract that keeps this sound is stripRouterSignatures
// (thinking_marker_strip.go): one pre-pass per source format, run on the
// client's body before any decoding or classification, so nothing downstream
// ever reads a Router signature as provenance. The request encoders refuse
// one again (thinking_marker_egress.go) only as defence in depth. Those two are
// the only callers of routerSignature.
//
// The whole "vsr." namespace is the Router's. No provider signature starts
// with it -- the dot is outside the base64 alphabet Anthropic's and every
// other carried blob use, the argument mintedReasoningDetailsPrefix also
// rests on -- so a value there that is not an exact marker is a forgery or a
// corruption, and it is refused the same way. Only an exact marker names a
// family.
const (
	routerSignatureNamespace = "vsr."
	thinkingMarkerPrefix     = routerSignatureNamespace + "thinking.v1."
	thinkingMarkerDigestLen  = 32
	thinkingMarkerFamilyMax  = 64
)

// routerSignature classifies a thinking signature. reserved is true for any
// value in the Router's namespace, which must never be treated as a provider's
// signature or sent to one. family is the producing family of an exact
// marker, and empty for every other value, a malformed reserved one included.
func routerSignature(signature string) (family string, reserved bool) {
	if !strings.HasPrefix(signature, routerSignatureNamespace) {
		return "", false
	}
	rest, isMarker := strings.CutPrefix(signature, thinkingMarkerPrefix)
	if !isMarker {
		return "", true
	}
	family, digest, found := strings.Cut(rest, ".")
	if !found || !validMarkerFamily(family) || !validMarkerDigest(digest) {
		return "", true
	}
	return family, true
}

func validMarkerFamily(family string) bool {
	if family == "" || len(family) > thinkingMarkerFamilyMax {
		return false
	}
	for _, char := range family {
		if (char < 'a' || char > 'z') && (char < '0' || char > '9') && char != '-' {
			return false
		}
	}
	return true
}

func validMarkerDigest(digest string) bool {
	if len(digest) != thinkingMarkerDigestLen {
		return false
	}
	for _, char := range digest {
		if (char < 'a' || char > 'z') && (char < 'A' || char > 'Z') && (char < '0' || char > '9') &&
			char != '-' && char != '_' {
			return false
		}
	}
	return true
}

// thinkingMarkerFamilyLabel is the family a log line names for a reserved
// signature: the marker's family, or "malformed" for any other value.
func thinkingMarkerFamilyLabel(family string) string {
	if family == "" {
		return "malformed"
	}
	return family
}
