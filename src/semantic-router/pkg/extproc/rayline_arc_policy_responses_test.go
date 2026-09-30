package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func responsesItem(t *testing.T, raw string) responseapi.InputItem {
	t.Helper()
	var item responseapi.InputItem
	if err := json.Unmarshal([]byte(raw), &item); err != nil {
		t.Fatal(err)
	}
	return item
}

// storeRoundTrip mimics the object store: a stored response is encoded and
// decoded before a later turn reads it back.
func storeRoundTrip(t *testing.T, stored *responseapi.StoredResponse) *responseapi.StoredResponse {
	t.Helper()
	encoded, err := json.Marshal(stored)
	if err != nil {
		t.Fatal(err)
	}
	var restored responseapi.StoredResponse
	if err := json.Unmarshal(encoded, &restored); err != nil {
		t.Fatal(err)
	}
	return &restored
}

// A turn's materialized items are a byte prefix of the next turn's, so the
// attribution ledger keeps one epoch across a previous_response_id chain, and
// the reply's first output item -- reasoning here, with no assistant message
// -- is attributable.
func TestPolicyResponsesInputIsAByteStablePrefix(t *testing.T) {
	router := &OpenAIRouter{}
	firstInput := []responseapi.InputItem{responsesItem(t, `{"type":"message","role":"user","content":"List the files."}`)}
	turn1 := &RequestContext{ResponseObjectState: &ResponseObjectState{Input: firstInput, Instructions: "You are Codex."}}
	items1, instructions, err := router.raylineARCPolicyResponsesInput(turn1)
	if err != nil {
		t.Fatal(err)
	}
	if instructions == nil || *instructions != "You are Codex." || len(items1) != 1 {
		t.Fatalf("turn 1: %d items, instructions %v", len(items1), instructions)
	}

	stored := storeRoundTrip(t, &responseapi.StoredResponse{
		ID: "resp_1", Object: "response", Input: cloneResponseInputItems(firstInput),
		Output: []responseapi.OutputItem{
			{Type: "reasoning", Summary: []responseapi.ContentPart{{Type: "summary_text", Text: "Use ls."}}},
			{Type: "function_call", CallID: "call_1", Name: "shell", Arguments: `{"command":["ls"]}`},
		},
	})
	turn2 := &RequestContext{ResponseObjectState: &ResponseObjectState{
		ConversationHistory: []*responseapi.StoredResponse{stored},
		Input:               []responseapi.InputItem{responsesItem(t, `{"type":"function_call_output","call_id":"call_1","output":"README.md"}`)},
	}}
	items2, _, err := router.raylineARCPolicyResponsesInput(turn2)
	if err != nil {
		t.Fatal(err)
	}
	if len(items2) != 4 {
		t.Fatalf("turn 2 materialized %d items, want input, reasoning, call, output", len(items2))
	}
	for index := range items1 {
		if !bytes.Equal(items1[index], items2[index]) {
			t.Fatalf("item %d changed between turns:\n%s\n%s", index, items1[index], items2[index])
		}
	}
	roles1, _ := policyResponsesRoles(items1)
	roles2, err := policyResponsesRoles(items2)
	if err != nil {
		t.Fatal(err)
	}
	committed := (&raylinearc.PolicyEpisodeState{}).Next(items1, strings.Repeat("a", 64), "arm")
	_, attributionBefore := raylinearc.PolicyTurn(nil, items1, roles1, 0)
	turn, attribution := raylinearc.PolicyTurn(committed, items2, roles2, 1)
	if len(attributionBefore) != 0 || turn.Epoch != 0 || len(attribution) != 1 || attribution[0].Message != len(items1) {
		t.Fatalf("epoch %d attribution %+v", turn.Epoch, attribution)
	}
	if roles2[len(items1)] != "assistant" || roles2[len(items1)+1] != "assistant" || roles2[3] != "" {
		t.Fatalf("roles = %v", roles2)
	}
}

// Without a stored object state (a decision-only lookup), the body's input is
// read through the same snapshot, so the bytes match what a stored turn sends.
func TestPolicyResponsesInputWithoutObjectState(t *testing.T) {
	body := []byte(`{"model":"auto","instructions":"Be brief.","input":"hello"}`)
	request, err := decodeResponsesForTest(body)
	if err != nil {
		t.Fatal(err)
	}
	items, instructions, err := (&OpenAIRouter{}).raylineARCPolicyResponsesInput(
		&RequestContext{RaylineARCRawBody: body, SemanticRequest: request})
	if err != nil {
		t.Fatal(err)
	}
	again, _, err := (&OpenAIRouter{}).raylineARCPolicyResponsesInput(
		&RequestContext{RaylineARCRawBody: body, SemanticRequest: request})
	if err != nil {
		t.Fatal(err)
	}
	// The snapshot gives an id-less item a fresh random id each time; the
	// policy items leave it out, so the same body gives the same bytes.
	if len(items) != 1 || !bytes.Equal(items[0], again[0]) || strings.Contains(string(items[0]), `"id"`) ||
		instructions == nil || *instructions != "Be brief." {
		t.Fatalf("items %s / %s, instructions %v", items, again, instructions)
	}
}

// A client that resends its whole history without item ids (no stored
// object) still sends a byte-stable prefix turn to turn.
func TestPolicyResponsesInputIsStableForResentHistory(t *testing.T) {
	turn := func(body string) []json.RawMessage {
		request, err := decodeResponsesForTest([]byte(body))
		if err != nil {
			t.Fatal(err)
		}
		items, _, err := (&OpenAIRouter{}).raylineARCPolicyResponsesInput(
			&RequestContext{RaylineARCRawBody: []byte(body), SemanticRequest: request})
		if err != nil {
			t.Fatal(err)
		}
		return items
	}
	first := turn(`{"model":"auto","store":false,"input":[{"type":"message","role":"user","content":"List the files."}]}`)
	second := turn(`{"model":"auto","store":false,"input":[{"type":"message","role":"user","content":"List the files."},` +
		`{"type":"function_call","call_id":"call_1","name":"shell","arguments":"{}"},` +
		`{"type":"function_call_output","call_id":"call_1","output":"README.md"}]}`)
	if len(second) != 3 || !bytes.Equal(first[0], second[0]) {
		t.Fatalf("resent history changed bytes:\n%s\n%s", first[0], second[0])
	}
}

// The selector sends a Responses request as input items and instructions,
// with no messages, and attributes the reply's reasoning item.
func TestPolicySelectorSendsResponsesInput(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	bindings := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return bindings[0].ActionID })
	items := []json.RawMessage{
		json.RawMessage(`{"type":"message","role":"user","content":"List the files."}`),
		json.RawMessage(`{"type":"reasoning","summary":[{"type":"summary_text","text":"Use ls."}]}`),
		json.RawMessage(`{"type":"function_call","call_id":"call_1","name":"shell","arguments":"{}"}`),
		json.RawMessage(`{"type":"function_call_output","call_id":"call_1","output":"README.md"}`),
	}
	instructions := "You are Codex."
	state, _ := raylinearc.NewEpisodeState(2)
	state.Policy = (&raylinearc.PolicyEpisodeState{}).Next(items[:1], bindings[0].ActionID, "arm-a")
	state.TurnIndex = 1
	_, err := fixture.selector.Select(context.Background(), &selection.SelectionContext{
		DecisionName: fixture.decision.Name, CandidateModels: fixture.decision.ModelRefs,
		RaylineARC: &selection.RaylineARCSelectionContext{
			EpisodeIDHash: strings.Repeat("e", 64), State: state,
			RequestFormat: policyFormatResponses, PolicyInput: items, PolicyInstructions: &instructions,
		},
	})
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	sent := fixture.fake.received()[0]
	if sent.RequestFormat != policyFormatResponses || len(sent.Request.Input) != 4 || sent.Request.Messages != nil ||
		sent.Request.Instructions == nil || *sent.Request.Instructions != instructions {
		t.Fatalf("decide request = %+v", sent.Request)
	}
	if len(sent.Attribution) != 1 || sent.Attribution[0].Message != 1 || sent.ContextEpoch != "0" {
		t.Fatalf("attribution %+v epoch %s", sent.Attribution, sent.ContextEpoch)
	}
}

func decodeResponsesForTest(body []byte) (*llmprotocol.Request, error) {
	request, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIResponsesV1, body)
	return &request, err
}

// A path with no stored object state cannot load the earlier turns a request
// names, so it refuses rather than deciding on the current input alone.
func TestPolicyResponsesInputRefusesUnresolvableHistory(t *testing.T) {
	body := []byte(`{"model":"auto","previous_response_id":"resp_1","input":"and then?"}`)
	request, err := decodeResponsesForTest(body)
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = (&OpenAIRouter{}).raylineARCPolicyResponsesInput(&RequestContext{RaylineARCRawBody: body, SemanticRequest: request})
	if !errors.Is(err, errPolicyResponsesHistoryUnavailable) {
		t.Fatalf("error = %v, want errPolicyResponsesHistoryUnavailable", err)
	}
}

// canonicalResponsesItem keeps exactly what pathfinder's canonicalizer reads
// from a Responses item (curation/canonical_conversation.py::_responses_item):
// a message's role and content parts (type, text, non-empty annotations,
// refusal), a call's call_id, name and arguments, an output's call_id and
// output, and reasoning's summary and content.
func canonicalResponsesItem(t *testing.T, raw json.RawMessage) map[string]any {
	t.Helper()
	var item map[string]any
	if err := json.Unmarshal(raw, &item); err != nil {
		t.Fatal(err)
	}
	parts := func(value any) []any {
		list, _ := value.([]any)
		kept := make([]any, 0, len(list))
		for _, rawPart := range list {
			part, _ := rawPart.(map[string]any)
			entry := map[string]any{"type": part["type"], "text": part["text"], "refusal": part["refusal"]}
			if annotations, ok := part["annotations"].([]any); ok && len(annotations) > 0 {
				entry["annotations"] = annotations
			}
			kept = append(kept, entry)
		}
		return kept
	}
	switch item["type"] {
	case "function_call":
		return map[string]any{"type": item["type"], "call_id": item["call_id"], "name": item["name"], "arguments": item["arguments"]}
	case "function_call_output":
		return map[string]any{"type": item["type"], "call_id": item["call_id"], "output": item["output"]}
	case "reasoning":
		return map[string]any{"type": item["type"], "summary": parts(item["summary"]), "content": parts(item["content"])}
	default:
		return map[string]any{"type": item["type"], "role": item["role"], "content": parts(item["content"])}
	}
}

// A reply read back from the object store gives the policy service every
// field the canonicalizer reads, exactly as the Responses encoder wrote it:
// reasoning summary and text, output text with its citation, a refusal, and
// a tool call.
func TestPolicyResponsesStoredOutputKeepsWhatTheCanonicalizerReads(t *testing.T) {
	providerBody := []byte(`{"id":"resp_1","object":"response","created_at":1,"model":"m","status":"completed","output":[` +
		`{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Use ls."}],"content":[{"type":"reasoning_text","text":"Listing is safe."}]},` +
		`{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[` +
		`{"type":"output_text","text":"See the docs.","annotations":[{"type":"url_citation","url":"https://example.com","title":"Docs","start_index":4,"end_index":8}]},` +
		`{"type":"refusal","refusal":"Not that one."}]},` +
		`{"type":"function_call","id":"fc_1","call_id":"call_1","name":"shell","arguments":"{\"command\":[\"ls\"]}","status":"completed"}]}`)
	response, _, _, err := (protocolcodec.OpenAIResponsesCodec{}).DecodeResponse(providerBody, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := protocolcodec.NewBuiltinEngine().EncodeResponse(llmprotocol.OpenAIResponsesV1, response, llmprotocol.Envelope{})
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Output []json.RawMessage `json:"output"`
	}
	var persisted responseapi.ResponseAPIResponse
	if err := json.Unmarshal(encoded.Body, &wire); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(encoded.Body, &persisted); err != nil {
		t.Fatal(err)
	}
	stored := storeRoundTrip(t, &responseapi.StoredResponse{ID: "resp_1", Output: persisted.Output})
	items, _, err := (&OpenAIRouter{}).raylineARCPolicyResponsesInput(&RequestContext{ResponseObjectState: &ResponseObjectState{
		ConversationHistory: []*responseapi.StoredResponse{stored},
		Input:               []responseapi.InputItem{responsesItem(t, `{"type":"message","role":"user","content":"next"}`)},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != len(wire.Output)+1 {
		t.Fatalf("%d items for %d outputs", len(items), len(wire.Output))
	}
	for index, output := range wire.Output {
		want, got := canonicalResponsesItem(t, output), canonicalResponsesItem(t, items[index])
		if !reflect.DeepEqual(want, got) {
			t.Fatalf("output %d changed what the canonicalizer reads:\nencoder %v\npolicy  %v", index, want, got)
		}
	}
}

// An item type the canonicalizer does not read (image_generation_call,
// web_search_call) is refused by the service; the turn fails closed with a
// class that names it, not a generic error.
func TestPolicyUnsupportedResponsesItemFailsWithANamedClass(t *testing.T) {
	fixture := newPolicySelectorFixture(t, "")
	fixture.fake.failNext("unsupported_request")
	state, _ := raylinearc.NewEpisodeState(2)
	_, err := fixture.selector.Select(context.Background(), &selection.SelectionContext{
		DecisionName: fixture.decision.Name, CandidateModels: fixture.decision.ModelRefs,
		RaylineARC: &selection.RaylineARCSelectionContext{
			EpisodeIDHash: strings.Repeat("e", 64), State: state, RequestFormat: policyFormatResponses,
			PolicyInput: []json.RawMessage{json.RawMessage(`{"type":"image_generation_call","result":"aGk="}`)},
		},
	})
	var failure *raylineARCSelectionFailure
	if !errors.As(err, &failure) || failure.class != "policy_service_unsupported_request" || selectionFailureIsContended(failure.class) {
		t.Fatalf("error = %v, want class policy_service_unsupported_request (not contended)", err)
	}
}
