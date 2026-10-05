package protocolcodec

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// stripRouterSignatures is the one place a client request loses its Router
// signatures. Each request codec runs it on the client's body before it
// decodes or classifies anything, so every later step -- the decoder's
// classification of items as modelled or carried whole, source-replay
// eligibility, the provenance filters (DropReasoningNotFromAnthropic,
// reasoningProvenance, SignedThinkingAsReasoningDetails) -- sees only
// stripped input, by construction rather than by each step remembering to
// strip. What a client can put a Router signature in, per source format:
//
//   - Messages: a thinking block's signature, which becomes "" (unsigned
//     thinking).
//   - Chat: a reasoning_details item's signature; the item is removed, so its
//     format tag cannot pass for Claude's provenance.
//   - Responses: a reasoning item's signature member, which is removed; and a
//     Router-minted encrypted_content (mintedReasoningDetailsPrefix), whose
//     signed items are removed -- and the blob itself when none remains, so
//     the item is plain reasoning rather than a blob to classify.
//
// A body that held one is decoded from the stripped bytes and is never kept
// for source replay, so the client's original bytes cannot reach a provider
// either. A body this pass cannot read is left as it is for the decoder to
// refuse.
//
// It replaces the per-site strips that review of #207 found one at a time,
// each a step that classified input before its strip ran: source replay
// (round 1), Chat reasoning_details provenance (round 2), minted-blob
// identity (round 3) and carried-whole Responses items (round 4). The request
// encoders' refusals (thinking_marker_egress.go) remain as a second guard.
func stripRouterSignatures(format llmprotocol.WireFormat, body []byte, policy llmprotocol.Policy) ([]byte, bool) {
	// A decoded string holds "vsr." only if the bytes do, or spell part of it
	// as a \u escape, the only escape JSON allows for those characters.
	if !bytes.Contains(body, []byte(routerSignatureNamespace)) && !bytes.Contains(body, []byte(`\u`)) {
		return body, false
	}
	if validateClientJSONDocument(body, policy, true) != nil {
		return body, false
	}
	var request map[string]json.RawMessage
	if json.Unmarshal(body, &request) != nil {
		return body, false
	}
	strip := routerSignatureStrip{families: map[string]int{}}
	var changed bool
	switch format {
	case llmprotocol.AnthropicMessagesV1:
		changed = strip.eachObject(request, "messages", func(message map[string]json.RawMessage) bool {
			return strip.eachObject(message, "content", strip.anthropicThinking)
		})
	case llmprotocol.OpenAIChatV1:
		changed = strip.eachObject(request, "messages", strip.chatReasoningDetails)
	case llmprotocol.OpenAIResponsesV1:
		changed = strip.eachObject(request, "input", strip.responsesReasoningItem)
	}
	if !changed {
		return body, false
	}
	stripped, err := json.Marshal(request)
	if err != nil {
		return body, false
	}
	strip.report(format)
	return stripped, true
}

// routerSignatureStrip counts the Router signatures one request lost, by
// family.
type routerSignatureStrip struct {
	families map[string]int
}

// signed reports whether value is a JSON string in the Router's namespace,
// and counts it.
func (strip routerSignatureStrip) signed(value json.RawMessage) bool {
	var signature string
	if json.Unmarshal(value, &signature) != nil {
		return false
	}
	family, reserved := routerSignature(signature)
	if reserved {
		strip.families[thinkingMarkerFamilyLabel(family)]++
	}
	return reserved
}

// eachObject applies edit to each object of the array under key, writing
// back the array when any edit changed its object. A member that is not an
// array of objects (Messages string content, for one) is left alone.
func (strip routerSignatureStrip) eachObject(
	parent map[string]json.RawMessage,
	key string,
	edit func(map[string]json.RawMessage) bool,
) bool {
	var elements []json.RawMessage
	if json.Unmarshal(parent[key], &elements) != nil {
		return false
	}
	changed := false
	for index, element := range elements {
		var object map[string]json.RawMessage
		if json.Unmarshal(element, &object) != nil || object == nil || !edit(object) {
			continue
		}
		encoded, err := json.Marshal(object)
		if err != nil {
			continue
		}
		elements[index] = encoded
		changed = true
	}
	if !changed {
		return false
	}
	encoded, err := json.Marshal(elements)
	if err != nil {
		return false
	}
	parent[key] = encoded
	return true
}

func (strip routerSignatureStrip) anthropicThinking(block map[string]json.RawMessage) bool {
	var blockType string
	if json.Unmarshal(block["type"], &blockType) != nil || blockType != "thinking" || !strip.signed(block["signature"]) {
		return false
	}
	block["signature"] = json.RawMessage(`""`)
	return true
}

func (strip routerSignatureStrip) chatReasoningDetails(message map[string]json.RawMessage) bool {
	details, ok := message[chatReasoningDetailsMember]
	if !ok {
		return false
	}
	kept, changed := strip.details(details)
	if !changed {
		return false
	}
	if kept == nil {
		delete(message, chatReasoningDetailsMember)
	} else {
		message[chatReasoningDetailsMember] = kept
	}
	return true
}

func (strip routerSignatureStrip) responsesReasoningItem(item map[string]json.RawMessage) bool {
	changed := false
	if signature, ok := item["signature"]; ok && strip.signed(signature) {
		delete(item, "signature")
		changed = true
	}
	var encrypted string
	if json.Unmarshal(item["encrypted_content"], &encrypted) != nil ||
		!strings.HasPrefix(encrypted, mintedReasoningDetailsPrefix) {
		return changed
	}
	decoded, err := base64.RawURLEncoding.DecodeString(strings.TrimPrefix(encrypted, mintedReasoningDetailsPrefix))
	if err != nil {
		return changed
	}
	kept, stripped := strip.details(decoded)
	if !stripped {
		return changed
	}
	if kept == nil {
		delete(item, "encrypted_content")
	} else if blob := mintReasoningDetails(kept); blob != nil {
		item["encrypted_content"] = blob
	} else {
		delete(item, "encrypted_content")
	}
	return true
}

// details removes each reasoning_details item signed with a Router
// signature. kept is nil when no item remains.
func (strip routerSignatureStrip) details(details json.RawMessage) (json.RawMessage, bool) {
	var items []json.RawMessage
	if json.Unmarshal(details, &items) != nil {
		return nil, false
	}
	kept := make([]json.RawMessage, 0, len(items))
	for _, item := range items {
		var signed struct {
			Signature json.RawMessage `json:"signature"`
		}
		if json.Unmarshal(item, &signed) == nil && strip.signed(signed.Signature) {
			continue
		}
		kept = append(kept, item)
	}
	if len(kept) == len(items) {
		return nil, false
	}
	if len(kept) == 0 {
		return nil, true
	}
	encoded, err := json.Marshal(kept)
	if err != nil {
		return nil, true
	}
	return encoded, true
}

// report logs one line per family stripped, with the count and the source
// format. It never logs thinking text, and nothing is gated on the family.
func (strip routerSignatureStrip) report(source llmprotocol.WireFormat) {
	families := make([]string, 0, len(strip.families))
	for family := range strip.families {
		families = append(families, family)
	}
	sort.Strings(families)
	for _, family := range families {
		logging.ComponentEvent("protocolcodec", "thinking_marker_stripped", map[string]interface{}{
			"family": family,
			"count":  strip.families[family],
			"source": string(source),
		})
	}
}
