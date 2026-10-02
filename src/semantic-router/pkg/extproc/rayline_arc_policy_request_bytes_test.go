//go:build !windows && cgo

package extproc

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// The decide request forwards the client's system, tools and messages
// exactly as received: for captured Claude Code bodies, each field of the
// decide request's request is byte-identical to the client's, not merely
// equal as JSON. The service hashes and projects what it receives, so any
// rewrite on the way (escaping, whitespace, key order) would be invisible to
// a JSON comparison and visible to the service.
func TestPolicyDecideRequestCarriesTheClientBytes(t *testing.T) {
	clients, err := filepath.Glob(filepath.Join(v5Golden, "messages", "*", "client-*.json"))
	if err != nil || len(clients) == 0 {
		t.Fatalf("no captured Messages bodies: %v", err)
	}
	router, fake := v5Router(t, "anthropic")
	for index, path := range clients {
		client, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		// The captures name their own model; the route is the auto alias. Only
		// the model value changes, so every other byte is the capture's.
		client = clientModel.ReplaceAll(client, []byte(`{"model":"auto"`))
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(client, &fields); err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		before := len(fake.received())
		v5Turn(t, router, fake, v5GLMUp, "request-bytes-"+strings.ReplaceAll(path, "/", "-"), string(client))
		received := fake.received()
		if len(received) != before+1 {
			t.Fatalf("%s: %d decide calls, want 1", path, len(received)-before)
		}
		request := received[len(received)-1].Request
		for name, got := range map[string]json.RawMessage{
			"system": request.System, "tools": request.Tools, "messages": request.Messages,
		} {
			want, present := fields[name]
			if !present {
				// An absent field is sent as null, never invented.
				if string(got) != "null" && got != nil {
					t.Errorf("%s: %s absent from the client but sent as %.80s", path, name, got)
				}
				continue
			}
			if !bytes.Equal(got, want) {
				t.Errorf("%s (%d): decide request.%s differs from the client's bytes at offset %d (%d vs %d bytes)",
					path, index, name, firstDifference(got, want), len(got), len(want))
			}
		}
	}
}

// clientModel is a capture's top-level model; a nested "model" key never
// appears in these Messages bodies.
var clientModel = regexp.MustCompile(`^\{"model":"[^"]*"`)

func firstDifference(a, b []byte) int {
	for i := 0; i < len(a) && i < len(b); i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return min(len(a), len(b))
}

// Insignificant whitespace is the client's too: a pretty-printed body reaches
// the service with its system and messages unchanged, not compacted.
func TestPolicyDecideRequestKeepsTheClientsWhitespace(t *testing.T) {
	router, fake := v5Router(t, "anthropic")
	system := "[ {\n  \"type\" : \"text\",\n  \"text\" : \"<system-reminder> a & b </system-reminder>\"\n} ]"
	messages := "[\n  { \"role\" : \"user\", \"content\" : \"fix <this> & that\" }\n]"
	client := "{\"model\":\"auto\",\n \"max_tokens\" : 1024,\n \"system\" : " + system + ",\n \"messages\" : " + messages + "\n}"
	v5Turn(t, router, fake, v5GLMUp, "request-bytes-whitespace", client)
	received := fake.received()
	request := received[len(received)-1].Request
	if string(request.System) != system || string(request.Messages) != messages {
		t.Fatalf("decide request differs from the client's bytes:\nsystem   %s\nmessages %s", request.System, request.Messages)
	}
}

// A Responses decide body carries each materialized item and the
// instructions in one encoding: neither is HTML-escaped.
func TestPolicyResponsesDecideBodyIsUnescaped(t *testing.T) {
	router := &OpenAIRouter{}
	stored := storeRoundTrip(t, &responseapi.StoredResponse{
		ID: "resp_1", Object: "response",
		Input:      []responseapi.InputItem{responsesItem(t, `{"type":"message","role":"user","content":"read <a> & <b>"}`)},
		OutputText: "saw <a> & <b>",
	})
	turn := &RequestContext{ResponseObjectState: &ResponseObjectState{
		ConversationHistory: []*responseapi.StoredResponse{stored},
		Input:               []responseapi.InputItem{responsesItem(t, `{"type":"message","role":"user","content":"now <c>"}`)},
		Instructions:        "You are <Codex> & careful.",
	}}
	items, instructions, err := router.raylineARCPolicyResponsesInput(turn)
	if err != nil {
		t.Fatal(err)
	}
	// The stored item, read back from an escaping store, is byte-identical to
	// the same item on the turn that sent it, so the prefix holds.
	first, _, err := router.raylineARCPolicyResponsesInput(&RequestContext{ResponseObjectState: &ResponseObjectState{
		Input: []responseapi.InputItem{responsesItem(t, `{"type":"message","role":"user","content":"read <a> & <b>"}`)},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first[0], items[0]) {
		t.Fatalf("a stored item's bytes changed between turns:\n%s\n%s", first[0], items[0])
	}
	body, err := raylinearc.EncodePolicyDecisionRequest(raylinearc.PolicyDecisionRequest{
		SchemaVersion: raylinearc.PolicyDecisionRequestSchema, RequestFormat: policyFormatResponses,
		Request: raylinearc.PolicyClientRequest{Input: items, Instructions: instructions},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"read <a> & <b>", "saw <a> & <b>", "now <c>", "You are <Codex> & careful."} {
		if !strings.Contains(string(body), want) {
			t.Errorf("decide body lacks %q", want)
		}
	}
	if strings.Contains(string(body), `\u003c`) || strings.Contains(string(body), `\u0026`) {
		t.Fatalf("decide body mixes in escaped text: %s", body)
	}
	var decoded map[string]any
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("decide body is not JSON: %v", err)
	}
}
