package protocolcodec

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

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
// The contract that keeps this sound: every client-supplied place a signature
// can arrive is stripped at decode, before anything reads provenance from
// it. Those places are a Messages thinking block (decodeAnthropicContentBlock),
// a Responses anthropic-claude-v1 reasoning item (responsesAnthropicReasoning
// .applyTo), and a reasoning_details item, whether a Chat message's or one
// held in a minted Responses blob (clientReasoningDetails). What reads
// provenance -- DropReasoningNotFromAnthropic (a signature, or an item's
// format), reasoningProvenance (content.thinking.signed) and
// SignedThinkingAsReasoningDetails -- runs after decode, so it only ever sees
// stripped data. A new decode path that carries a signature or a
// reasoning_details array must strip here too; source replay is refused
// separately (holdsRouterSignatureValue), since it skips decoding altogether.
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

// clientThinkingSignature is the signature a decoded thinking block keeps: the
// client's own value, or none for a value in the Router's namespace, so the
// block is unsigned reasoning and never counts as a provider's proof. Each
// strip is logged with the marker's family; nothing is gated on it, and the
// thinking text is never logged.
func clientThinkingSignature(signature string, source llmprotocol.WireFormat) string {
	family, reserved := routerSignature(signature)
	if !reserved {
		return signature
	}
	logging.ComponentEvent("protocolcodec", "thinking_marker_stripped", map[string]interface{}{
		"family": thinkingMarkerFamilyLabel(family),
		"source": string(source),
	})
	return ""
}

// holdsRouterSignatureValue reports whether a request body holds any JSON
// string in the Router's namespace. Such a body is never kept for source
// replay (requestEnvelope): its bytes still carry the value the decoders strip
// and the encoders refuse, so the request is encoded instead. Any string
// counts, not only a signature member, so a text that starts with "vsr." also
// gives up replay; that costs at most a prompt-cache miss. A body that cannot
// be scanned is treated as holding one.
func holdsRouterSignatureValue(body []byte) bool {
	// A decoded string holds "vsr." only if the bytes do, or spell part of it
	// as a \u escape, the only escape JSON allows for those characters.
	if !bytes.Contains(body, []byte(routerSignatureNamespace)) && !bytes.Contains(body, []byte(`\u`)) {
		return false
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	for {
		token, err := decoder.Token()
		if errors.Is(err, io.EOF) {
			return false
		}
		if err != nil {
			return true
		}
		if value, ok := token.(string); ok {
			if _, reserved := routerSignature(value); reserved {
				return true
			}
		}
	}
}

// clientReasoningDetails is the reasoning_details a decoded client message
// keeps: every item but one signed with a Router signature. The format an item
// names is the client's claim, so an anthropic-claude-v1 item under a marker
// would otherwise pass for Claude's (DropReasoningNotFromAnthropic). Without
// it, the message's reasoning text is unsigned reasoning like any other. Each
// strip is logged as clientThinkingSignature logs one.
func clientReasoningDetails(details []byte, source llmprotocol.WireFormat) json.RawMessage {
	if details == nil {
		return nil
	}
	kept, err := withoutRouterSignedDetails(details, func(family string) {
		logging.ComponentEvent("protocolcodec", "thinking_marker_stripped", map[string]interface{}{
			"family": thinkingMarkerFamilyLabel(family),
			"source": string(source),
		})
	})
	if err != nil {
		// decodeReasoningDetailsArray already proved it an array of objects.
		return nil
	}
	return kept
}
