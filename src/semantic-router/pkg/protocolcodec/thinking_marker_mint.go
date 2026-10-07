package protocolcodec

import (
	"crypto/sha256"
	"encoding/base64"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Minting (Phase 2 of #191). Thinking a non-Anthropic upstream wrote reaches
// a Messages client unsigned, and pi and OpenClaw turn an unsigned thinking
// block into visible text when they resend history; Claude, sent another
// model's reasoning as the assistant's own words, refuses. Signed with a
// Router marker, the block is resent as thinking, the request pre-pass strips
// the marker (thinking_marker_strip.go), and the reasoning carry rules drop it
// for Claude and carry it to a Chat worker.
//
// A marker is minted only where the encoder would otherwise send the block
// unsigned: Claude's own signature, carried or recovered from
// reasoning_details, always wins. It is deterministic in the family and the
// thinking text, so a replay is byte-identical and the prompt cache holds.

// ThinkingMarkerFamily is the marker family for a served model: the
// provider's organisation when the model id names one ("moonshotai/kimi-k3" is
// moonshotai), else the model's own name, folded to the family alphabet. It is
// telemetry: the request pre-pass logs it and gates nothing on it.
func ThinkingMarkerFamily(model string) string {
	model = strings.ToLower(strings.TrimSpace(model))
	if org, _, found := strings.Cut(model, "/"); found && org != "" {
		model = org
	}
	model, _, _ = strings.Cut(model, "@")
	var family strings.Builder
	for _, char := range model {
		switch {
		case char >= 'a' && char <= 'z', char >= '0' && char <= '9', char == '-':
			family.WriteRune(char)
		default:
			family.WriteByte('-')
		}
	}
	folded := strings.Trim(family.String(), "-")
	if len(folded) > thinkingMarkerFamilyMax {
		folded = strings.TrimRight(folded[:thinkingMarkerFamilyMax], "-")
	}
	if folded == "" {
		return "unknown"
	}
	return folded
}

// ThinkingMarker is the marker for thinking text from a family. The digest is
// the first 24 bytes of sha256 over the version, the family and the text,
// base64url without padding: 32 characters, the exact shape routerSignature
// recognises.
func ThinkingMarker(family, text string) string {
	sum := sha256.Sum256([]byte(thinkingMarkerPrefix + "\x00" + family + "\x00" + text))
	return thinkingMarkerPrefix + family + "." + base64.RawURLEncoding.EncodeToString(sum[:24])
}

// withThinkingMarkers signs, on a fresh slice, each reasoning content that
// is still unsigned and holds text with family's marker. Empty family mints
// nothing.
func withThinkingMarkers(output []llmprotocol.OutputItem, family string) []llmprotocol.OutputItem {
	if family == "" {
		return output
	}
	signed := make([]llmprotocol.OutputItem, len(output))
	for index, item := range output {
		signed[index] = item
		copied := false
		for contentIndex, content := range item.Content {
			if !mintableThinking(content) {
				continue
			}
			if !copied {
				signed[index].Content = append([]llmprotocol.Content(nil), item.Content...)
				copied = true
			}
			signed[index].Content[contentIndex].Signature = ThinkingMarker(family, content.Text)
		}
	}
	return signed
}

func mintableThinking(content llmprotocol.Content) bool {
	return content.Kind == llmprotocol.ContentReasoning && content.Signature == "" && content.Text != ""
}

// ResponseMintsThinkingMarkers reports whether a Messages encoding of the
// response would mint a marker: some reasoning content holds text and is
// still unsigned once Claude's carried signatures are applied. A same-format
// body the Router would pass through as the upstream wrote it is re-encoded
// when this holds.
func ResponseMintsThinkingMarkers(response llmprotocol.Response) bool {
	if response.ThinkingMarkerFamily == "" {
		return false
	}
	for _, item := range withClaudeThinkingSignatures(withoutResponsesOnlyOutput(response.Output)) {
		for _, content := range item.Content {
			if mintableThinking(content) {
				return true
			}
		}
	}
	return false
}
