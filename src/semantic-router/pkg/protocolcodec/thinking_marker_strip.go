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
// The pass never re-encodes the body. It finds the byte span of each value to
// remove or replace and splices it out of the client's bytes, so the output
// is the input with marker spans removed (a Messages signature value becomes
// "", a minted blob a shorter blob); no other byte changes, and len(out) <=
// len(in). A body within the size limit therefore stays within it, whatever
// its keys and strings hold. A body that held one is decoded from the
// stripped bytes and is never kept for source replay, so the client's
// original bytes cannot reach a provider either. A body this pass cannot read is left as it is for the decoder to
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
	// The scanner below trusts the document's shape, so it runs only on a
	// body the codecs' own validation accepts; any other body is left for
	// the decoder to refuse.
	if validateClientJSONDocument(body, policy, true) != nil {
		return body, false
	}
	strip := routerSignatureStrip{body: body, families: map[string]int{}, depth: policy.Limits.JSONDepth}
	top := strip.value(skipJSONSpace(body, 0))
	switch format {
	case llmprotocol.AnthropicMessagesV1:
		strip.eachNamedObject(top, "messages", func(message jsonSpan) {
			strip.eachNamedObject(message, "content", strip.anthropicThinking)
		})
	case llmprotocol.OpenAIChatV1:
		strip.eachNamedObject(top, "messages", strip.chatReasoningDetails)
	case llmprotocol.OpenAIResponsesV1:
		strip.eachNamedObject(top, "input", strip.responsesReasoningItem)
	}
	if len(strip.edits) == 0 {
		return body, false
	}
	stripped := strip.splice()
	if len(stripped) > len(body) {
		// Every edit removes bytes or replaces a value with a shorter one, so
		// this cannot happen; if it did, the request is refused rather than
		// sent with a signature or grown past the limit it met.
		return nil, true
	}
	strip.report(format)
	return stripped, true
}

// routerSignatureStrip collects, over one client body, the byte edits that
// remove its Router signatures, and counts them by family.
type routerSignatureStrip struct {
	body     []byte
	edits    []jsonEdit
	families map[string]int
	depth    int
}

// jsonSpan is a JSON value's bytes, body[start:end]. A zero span is absent.
type jsonSpan struct{ start, end int }

func (span jsonSpan) present() bool { return span.end > span.start }

// jsonEdit replaces body[start:end] with with; an empty with removes it.
type jsonEdit struct {
	start, end int
	with       []byte
}

// jsonMember is an object member: its key, decoded, and the span from the
// key's opening quote to the value's end, and the value's own span.
type jsonMember struct {
	key   string
	whole jsonSpan
	value jsonSpan
}

func skipJSONSpace(body []byte, at int) int {
	for at < len(body) && (body[at] == ' ' || body[at] == '\t' || body[at] == '\n' || body[at] == '\r') {
		at++
	}
	return at
}

// value returns the span of the JSON value starting at start.
func (strip *routerSignatureStrip) value(start int) jsonSpan {
	body := strip.body
	if start >= len(body) {
		return jsonSpan{}
	}
	switch body[start] {
	case '"':
		at := start + 1
		for at < len(body) && body[at] != '"' {
			if body[at] == '\\' {
				at++
			}
			at++
		}
		return jsonSpan{start, at + 1}
	case '{', '[':
		closing := byte('}')
		if body[start] == '[' {
			closing = ']'
		}
		at := skipJSONSpace(body, start+1)
		for at < len(body) && body[at] != closing {
			if body[at] == ',' || body[at] == ':' {
				at = skipJSONSpace(body, at+1)
				continue
			}
			at = skipJSONSpace(body, strip.value(at).end)
		}
		return jsonSpan{start, at + 1}
	default:
		at := start
		for at < len(body) && !strings.ContainsRune(",}] \t\n\r", rune(body[at])) {
			at++
		}
		return jsonSpan{start, at}
	}
}

// elements returns the spans of an array's elements, or nil for any other
// value.
func (strip *routerSignatureStrip) elements(array jsonSpan) []jsonSpan {
	if !array.present() || strip.body[array.start] != '[' {
		return nil
	}
	var spans []jsonSpan
	at := skipJSONSpace(strip.body, array.start+1)
	for at < array.end-1 {
		element := strip.value(at)
		spans = append(spans, element)
		at = skipJSONSpace(strip.body, element.end)
		if at < array.end-1 && strip.body[at] == ',' {
			at = skipJSONSpace(strip.body, at+1)
		}
	}
	return spans
}

// members returns an object's members, or nil for any other value.
func (strip *routerSignatureStrip) members(object jsonSpan) []jsonMember {
	if !object.present() || strip.body[object.start] != '{' {
		return nil
	}
	var members []jsonMember
	at := skipJSONSpace(strip.body, object.start+1)
	for at < object.end-1 {
		key := strip.value(at)
		var name string
		_ = json.Unmarshal(strip.body[key.start:key.end], &name)
		value := strip.value(skipJSONSpace(strip.body, skipJSONSpace(strip.body, key.end)+1))
		members = append(members, jsonMember{key: name, whole: jsonSpan{key.start, value.end}, value: value})
		at = skipJSONSpace(strip.body, value.end)
		if at < object.end-1 && strip.body[at] == ',' {
			at = skipJSONSpace(strip.body, at+1)
		}
	}
	return members
}

// named returns the value of every member of an object whose name matches
// name under Unicode case folding, as encoding/json matches a field. All of
// them are examined, not the first: the duplicate-key check lower-cases names
// and so does not pair "signature" with an alias such as "ſignature" (long s),
// though the decoder reads either.
func (strip *routerSignatureStrip) named(object jsonSpan, name string) []jsonSpan {
	var values []jsonSpan
	for _, member := range strip.members(object) {
		if strings.EqualFold(member.key, name) {
			values = append(values, member.value)
		}
	}
	return values
}

// anySigned reports whether any of the values is a Router signature,
// counting each.
func (strip *routerSignatureStrip) anySigned(values []jsonSpan) bool {
	signed := false
	for _, value := range values {
		if strip.signed(value) {
			signed = true
		}
	}
	return signed
}

// eachNamedObject visits the objects of every array an object holds under a
// name, by the folding rule of named.
func (strip *routerSignatureStrip) eachNamedObject(object jsonSpan, name string, visit func(jsonSpan)) {
	for _, array := range strip.named(object, name) {
		strip.eachObject(array, visit)
	}
}

func (strip *routerSignatureStrip) eachObject(array jsonSpan, visit func(jsonSpan)) {
	for _, element := range strip.elements(array) {
		if strip.body[element.start] == '{' {
			visit(element)
		}
	}
}

// remove removes the listed entries of a container -- array elements, or
// object members whole -- with the commas that separate them, leaving the
// rest of the container byte for byte.
func (strip *routerSignatureStrip) remove(container jsonSpan, entries []jsonSpan, removed []bool) {
	for index, entry := range entries {
		if !removed[index] {
			continue
		}
		if index+1 < len(entries) {
			// Up to the next entry: this one and the comma after it.
			strip.edits = append(strip.edits, jsonEdit{start: entry.start, end: entries[index+1].start})
			continue
		}
		// The last entry: from the end of the last entry kept before it, so
		// its preceding comma goes too, or from the opening bracket.
		from := container.start + 1
		for previous := index - 1; previous >= 0; previous-- {
			if !removed[previous] {
				from = entries[previous].end
				break
			}
		}
		strip.edits = append(strip.edits, jsonEdit{start: from, end: entry.end})
	}
}

// signed reports whether a value is a JSON string in the Router's namespace,
// and counts it.
func (strip *routerSignatureStrip) signed(value jsonSpan) bool {
	if !value.present() || strip.body[value.start] != '"' {
		return false
	}
	var signature string
	if json.Unmarshal(strip.body[value.start:value.end], &signature) != nil {
		return false
	}
	family, reserved := routerSignature(signature)
	if reserved {
		strip.families[thinkingMarkerFamilyLabel(family)]++
	}
	return reserved
}

func (strip *routerSignatureStrip) anthropicThinking(block jsonSpan) {
	thinking := false
	for _, kind := range strip.named(block, "type") {
		var blockType string
		if json.Unmarshal(strip.body[kind.start:kind.end], &blockType) == nil && blockType == "thinking" {
			thinking = true
		}
	}
	if !thinking {
		return
	}
	for _, signature := range strip.named(block, "signature") {
		if strip.signed(signature) {
			strip.edits = append(strip.edits, jsonEdit{start: signature.start, end: signature.end, with: []byte(`""`)})
		}
	}
}

func (strip *routerSignatureStrip) chatReasoningDetails(message jsonSpan) {
	members := strip.members(message)
	for index, member := range members {
		if !strings.EqualFold(member.key, chatReasoningDetailsMember) {
			continue
		}
		items := strip.elements(member.value)
		removed := make([]bool, len(items))
		count := 0
		for item, span := range items {
			if strip.anySigned(strip.named(span, "signature")) {
				removed[item], count = true, count+1
			}
		}
		switch {
		case count == 0:
		case count == len(items):
			// No item is left: the member goes, so no empty array is sent.
			strip.removeMember(message, members, index)
		default:
			strip.remove(member.value, items, removed)
		}
	}
}

func (strip *routerSignatureStrip) removeMember(object jsonSpan, members []jsonMember, index int) {
	wholes := make([]jsonSpan, len(members))
	removed := make([]bool, len(members))
	for position, member := range members {
		wholes[position] = member.whole
	}
	removed[index] = true
	strip.remove(object, wholes, removed)
}

func (strip *routerSignatureStrip) responsesReasoningItem(item jsonSpan) {
	members := strip.members(item)
	wholes := make([]jsonSpan, len(members))
	removed := make([]bool, len(members))
	removing := false
	for index, member := range members {
		wholes[index] = member.whole
		switch {
		case strings.EqualFold(member.key, "signature") && strip.signed(member.value):
			removed[index], removing = true, true
		case strings.EqualFold(member.key, "encrypted_content"):
			blob, drop := strip.mintedBlob(member.value)
			switch {
			case drop:
				removed[index], removing = true, true
			case blob != nil:
				strip.edits = append(strip.edits, jsonEdit{start: member.value.start, end: member.value.end, with: blob})
			}
		}
	}
	if removing {
		strip.remove(item, wholes, removed)
	}
}

// mintedBlob strips the Router-signed items of a Router-minted
// encrypted_content. It returns the replacement blob, or drop when the blob
// goes: every item was stripped, its items repeat a member (the body's
// duplicate-key check cannot see inside the base64, and a decoder would read
// one of the values only), or the re-minted blob would not be shorter than
// the original. Neither is returned for any other value.
func (strip *routerSignatureStrip) mintedBlob(value jsonSpan) (blob []byte, drop bool) {
	var encrypted string
	if !value.present() || json.Unmarshal(strip.body[value.start:value.end], &encrypted) != nil ||
		!strings.HasPrefix(encrypted, mintedReasoningDetailsPrefix) {
		return nil, false
	}
	decoded, err := base64.RawURLEncoding.DecodeString(strings.TrimPrefix(encrypted, mintedReasoningDetailsPrefix))
	if err != nil {
		return nil, false
	}
	if validateNoDuplicateKeys(decoded, strip.depth) != nil {
		strip.families[thinkingMarkerFamilyLabel("")]++
		return nil, true
	}
	kept, stripped := strip.details(decoded)
	if !stripped {
		return nil, false
	}
	if kept == nil {
		return nil, true
	}
	blob = mintReasoningDetails(kept)
	if blob == nil || len(blob) > value.end-value.start {
		return nil, true
	}
	return blob, false
}

// details removes each reasoning_details item signed with a Router
// signature from a decoded minted array. The items kept are joined as they
// were written; kept is nil when none remains.
func (strip *routerSignatureStrip) details(details []byte) (json.RawMessage, bool) {
	inner := routerSignatureStrip{body: details, families: strip.families, depth: strip.depth}
	items := inner.elements(inner.value(skipJSONSpace(details, 0)))
	kept := make([][]byte, 0, len(items))
	for _, span := range items {
		if details[span.start] == '{' && inner.anySigned(inner.named(span, "signature")) {
			continue
		}
		kept = append(kept, details[span.start:span.end])
	}
	if len(kept) == len(items) {
		return nil, false
	}
	if len(kept) == 0 {
		return nil, true
	}
	return append(append([]byte{'['}, bytes.Join(kept, []byte{','})...), ']'), true
}

// splice applies the edits to the client's bytes. Removals that overlap --
// two adjacent entries of one container -- are merged.
func (strip *routerSignatureStrip) splice() []byte {
	edits := append([]jsonEdit(nil), strip.edits...)
	sort.Slice(edits, func(i, j int) bool { return edits[i].start < edits[j].start })
	out := make([]byte, 0, len(strip.body))
	at := 0
	for _, edit := range edits {
		if edit.end <= at {
			continue
		}
		if edit.start < at {
			edit.start = at
		}
		out = append(out, strip.body[at:edit.start]...)
		out = append(out, edit.with...)
		at = edit.end
	}
	return append(out, strip.body[at:]...)
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
