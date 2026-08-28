package conformance

import "fmt"

// This file gives cases.yaml's expectation.invariants teeth.
//
// An invariant names an equivalence or a property the case's contract permits or
// requires. Three of them describe encodings the public API contracts declare
// interchangeable, so a byte-level or structural diff would report a difference
// that is not one. Those three are enforced here: the comparator normalizes both
// sides before diffing, and only for a case that declares the invariant.
//
// The rest are held by the authored artifacts under the case's declared
// comparison modes. They still have to be spelled correctly, so the loader
// validates every name against the vocabulary below.

// Invariant is one declared expectation.invariants entry.
type Invariant string

// The enforced invariants. Each one names an encoding difference a public API
// contract permits, which the comparator normalizes away before diffing.
const (
	// InvariantArgumentJSON says a tool call's arguments string is JSON, so it is
	// compared as parsed JSON rather than as bytes. OpenAI does not promise a
	// particular spacing or key order inside that string.
	InvariantArgumentJSON Invariant = "argument-json-equivalence"
	// InvariantContentEncoding says an OpenAI Chat message content written as a
	// bare string and as a single text part are the same content.
	InvariantContentEncoding Invariant = "content-encoding-equivalence"
	// InvariantNullVsOmitted says a field carrying an explicit null and the same
	// field omitted are the same absence. OpenAI accepts and emits both.
	InvariantNullVsOmitted Invariant = "null-vs-omitted-equivalence"
)

// InvariantSupport says how a declared invariant is held.
type InvariantSupport string

const (
	// SupportEnforced means the comparator normalizes for it, in this file.
	SupportEnforced InvariantSupport = "enforced"
	// SupportCovered means the case's authored artifacts already assert it under
	// its declared comparison modes, so no extra normalization is needed. The
	// table below names the machinery for each one.
	SupportCovered InvariantSupport = "covered"
	// SupportDeferred means nothing asserts it yet. Only a case outside the
	// promoted tranche may declare one, so a promoted case can never carry an
	// invariant that does nothing.
	SupportDeferred InvariantSupport = "deferred"
)

// invariantVocabulary is the closed set of invariant names v1 accepts. A name
// outside it is a load error; before this table existed an unknown name was
// silently inert.
//
// The covered entries are held by the artifacts the case already authors:
//   - stream shape (no-chat-chunk-leak, response-completed-exactly-once,
//     ordered-text, ordered-blocks, no-message-stop-success, no-sse-success-frame,
//     partial-text-preserved) is held by the expected-client-response event
//     sequence, which is compared by name, order, and count.
//   - payload content (usage-preserved, incomplete-usage, no-usage,
//     tool-id-pairing) is held by the expected artifact's JSON, which the
//     structural diff compares field by field, absent fields included.
//   - hop metadata (status-429, retry-after-bounded) is held by the replay
//     script's committed status step and the preserved fidelity ledger entries
//     the runner relays against.
//   - no-second-dispatch is held by expectation.dispatch_attempts.
var invariantVocabulary = map[Invariant]InvariantSupport{
	InvariantArgumentJSON:    SupportEnforced,
	InvariantContentEncoding: SupportEnforced,
	InvariantNullVsOmitted:   SupportEnforced,

	"incomplete-usage":                SupportCovered,
	"no-chat-chunk-leak":              SupportCovered,
	"no-message-stop-success":         SupportCovered,
	"no-second-dispatch":              SupportCovered,
	"no-sse-success-frame":            SupportCovered,
	"no-usage":                        SupportCovered,
	"ordered-blocks":                  SupportCovered,
	"ordered-text":                    SupportCovered,
	"partial-text-preserved":          SupportCovered,
	"response-completed-exactly-once": SupportCovered,
	"retry-after-bounded":             SupportCovered,
	"status-429":                      SupportCovered,
	"tool-id-pairing":                 SupportCovered,
	"usage-preserved":                 SupportCovered,

	"cache-read-preserved":         SupportDeferred,
	"cache-write-preserved":        SupportDeferred,
	"carrier-before-tool":          SupportDeferred,
	"error-event-present":          SupportDeferred,
	"no-clean-message-stop":        SupportDeferred,
	"no-terminal-success":          SupportDeferred,
	"no-visible-reasoning":         SupportDeferred,
	"opaque-bytes-preserved":       SupportDeferred,
	"original-history-immutable":   SupportDeferred,
	"provider-a-carrier-restored":  SupportDeferred,
	"start-delta-usage-consistent": SupportDeferred,
	"tool-pairing-preserved":       SupportDeferred,
	"upstream-context-cancelled":   SupportDeferred,
}

// Support reports how the invariant is held, and whether it is a known name.
func (i Invariant) Support() (InvariantSupport, bool) {
	support, ok := invariantVocabulary[i]
	return support, ok
}

// Enforced reports whether the comparator normalizes for this invariant.
func (i Invariant) Enforced() bool {
	support, ok := invariantVocabulary[i]
	return ok && support == SupportEnforced
}

// enforcedInvariants keeps only the entries the comparator acts on.
func enforcedInvariants(declared []Invariant) []Invariant {
	var out []Invariant
	for _, invariant := range declared {
		if invariant.Enforced() {
			out = append(out, invariant)
		}
	}
	return out
}

// applyInvariants rewrites the decoded want and got pair in place so the
// equivalences the case declares stop reading as differences.
//
// It runs before the diff and never after it, so every relaxation is visible as
// one named, opt-in rule rather than as a special case buried in the walk.
func applyInvariants(declared []Invariant, want, got any) {
	for _, invariant := range declared {
		switch invariant {
		case InvariantContentEncoding:
			expandChatContent(want)
			expandChatContent(got)
		case InvariantArgumentJSON:
			parseToolCallArguments(want)
			parseToolCallArguments(got)
		case InvariantNullVsOmitted:
			reconcileNullAndOmitted(want, got)
		}
	}
}

// expandChatContent rewrites an OpenAI Chat message content written as a bare
// string into the single text part it is equivalent to. Only the two positions
// the Chat schema puts a message in are touched, so an Anthropic body, whose
// content is always a block list, is left alone.
//
// Expanding rather than collapsing keeps the comparison strict: a part carrying
// anything beyond type and text is not equivalent to a string, and still differs.
func expandChatContent(root any) {
	document, ok := root.(map[string]any)
	if !ok {
		return
	}
	for _, message := range elements(document["messages"]) {
		expandContentField(message)
	}
	for _, choice := range elements(document["choices"]) {
		if message, ok := choice["message"].(map[string]any); ok {
			expandContentField(message)
		}
	}
}

func expandContentField(message map[string]any) {
	text, ok := message["content"].(string)
	if !ok {
		return
	}
	message["content"] = []any{map[string]any{"type": "text", "text": text}}
}

// parseToolCallArguments replaces every tool-call arguments string with its
// parsed JSON, so the comparison is structural. The carrier is the shape rather
// than a fixed pointer: an OpenAI tool call is a "function" object with a string
// "arguments", wherever a request, a response, or a stream delta puts it.
//
// A string that is not JSON is left as a string. The two sides then differ by
// type, which is the failure it should be.
func parseToolCallArguments(node any) {
	switch typed := node.(type) {
	case map[string]any:
		if function, ok := typed["function"].(map[string]any); ok {
			if raw, ok := function["arguments"].(string); ok {
				if decoded, err := decodeJSON([]byte(raw)); err == nil {
					function["arguments"] = decoded
				}
			}
		}
		for _, child := range typed {
			parseToolCallArguments(child)
		}
	case []any:
		for _, child := range typed {
			parseToolCallArguments(child)
		}
	}
}

// reconcileNullAndOmitted deletes a field that is explicitly null on one side and
// absent on the other. Only a null is ever removed: a field carrying a value is
// never reconciled away, so a dropped value is still a failure.
func reconcileNullAndOmitted(want, got any) {
	switch wantNode := want.(type) {
	case map[string]any:
		if gotNode, ok := got.(map[string]any); ok {
			reconcileObject(wantNode, gotNode)
		}
	case []any:
		if gotNode, ok := got.([]any); ok && len(gotNode) == len(wantNode) {
			for i := range wantNode {
				reconcileNullAndOmitted(wantNode[i], gotNode[i])
			}
		}
	}
}

// reconcileObject drops each side's explicitly null field that the other side
// omits, then recurses into the fields both sides carry.
func reconcileObject(want, got map[string]any) {
	dropNullsAbsentFrom(want, got)
	dropNullsAbsentFrom(got, want)
	for key, wantChild := range want {
		if gotChild, present := got[key]; present {
			reconcileNullAndOmitted(wantChild, gotChild)
		}
	}
}

// dropNullsAbsentFrom deletes every null-valued field of from that other omits.
func dropNullsAbsentFrom(from, other map[string]any) {
	for key, value := range from {
		if value != nil {
			continue
		}
		if _, present := other[key]; !present {
			delete(from, key)
		}
	}
}

// elements returns the objects in a JSON array, skipping anything that is not one.
func elements(node any) []map[string]any {
	array, ok := node.([]any)
	if !ok {
		return nil
	}

	out := make([]map[string]any, 0, len(array))
	for _, entry := range array {
		if object, ok := entry.(map[string]any); ok {
			out = append(out, object)
		}
	}
	return out
}

// validateInvariants checks every declared name against the vocabulary and keeps a
// promoted case from carrying one nothing asserts.
func validateInvariants(c *Case) []error {
	var problems []error
	for _, invariant := range c.Expectation.Invariants {
		support, known := invariant.Support()
		if !known {
			problems = append(problems, fmt.Errorf(
				"case %q: expectation.invariants declares unknown invariant %q", c.ID, invariant))
			continue
		}
		if support == SupportDeferred && c.Tranche == PromotedTranche {
			problems = append(problems, fmt.Errorf(
				"case %q: invariant %q is recognized but not asserted yet, so a promoted case may not declare it; enforce it or drop it",
				c.ID, invariant))
		}
	}
	return problems
}
