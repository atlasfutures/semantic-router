//go:build !windows && cgo

package extproc

import (
	"encoding/json"
	"net/http"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A picked endpoint's history carries reasoning encrypted for the pick's host
// family. Re-sent on an endpoint of another family (gpt on azure, then on
// openai), that reasoning is refused there with a 400, and the turn is lost;
// so the re-send strips it, as the thinking-signature re-send does, and the
// run starts again at the re-send's reply. Within one family it goes as sent.

// withHeldReasoning puts an assistant message holding encrypted
// reasoning_details at the head of the staged re-send's history.
func withHeldReasoning(t *testing.T, ctx *RequestContext) {
	t.Helper()
	plan := ctx.RaylineARCEndpointResend
	if plan == nil {
		t.Fatal("no endpoint re-send was staged")
	}
	var members map[string]json.RawMessage
	var messages []json.RawMessage
	if json.Unmarshal(plan.dispatched, &members) != nil || json.Unmarshal(members["messages"], &messages) != nil {
		t.Fatalf("staged body is not a Chat body:\n%s", plan.dispatched)
	}
	held := json.RawMessage(`{"role":"assistant","content":"earlier answer",` +
		`"reasoning_details":[{"type":"reasoning.encrypted","data":"opaque-for-the-pick","format":"openai-responses-v1"}]}`)
	messages = append([]json.RawMessage{json.RawMessage(`{"role":"user","content":"earlier question"}`), held}, messages...)
	members["messages"] = mustJSONMessage(t, messages)
	plan.dispatched = mustJSONMessage(t, members)
}

func mustJSONMessage(t *testing.T, value interface{}) json.RawMessage {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}

// resentHistory is the assistant messages of the body the re-send carried.
func resentHistory(t *testing.T, body []byte) []map[string]json.RawMessage {
	t.Helper()
	var wire struct {
		Messages []map[string]json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode re-sent body: %v\n%s", err, body)
	}
	var assistant []map[string]json.RawMessage
	for _, message := range wire.Messages {
		if wireRole(message) == "assistant" {
			assistant = append(assistant, message)
		}
	}
	return assistant
}

func TestCrossFamilyEndpointResendStripsTheEncryptedReasoning(t *testing.T) {
	logs := captureLogs(t)
	tags := []string{"azure", "openai"}
	order := raylinearc.PickEndpoint(endpointPickKey, tags)
	upstream := newEndpointUpstream(t, map[string]int{})
	router := endpointPickRouter(upstream.server.URL, tags...)
	ctx, _ := routeEndpointTurn(t, router, endpointTurn{key: endpointPickKey, feed: endpointFeed(t, tags...)})
	withHeldReasoning(t, ctx)
	transaction := &raylineARCEpisodeTransaction{thinkingWorker: "run", thinkingAdopted: "adopted"}
	ctx.RaylineARCTransaction = transaction

	reply := answerHeaders(t, router, ctx, http.StatusServiceUnavailable)

	if code := int(reply.GetImmediateResponse().GetStatus().GetCode()); code != http.StatusOK {
		t.Fatalf("client status = %d, want the next endpoint's 200", code)
	}
	if got := upstream.calls(); !slices.Equal(got, order[1:2]) {
		t.Fatalf("re-sent on %v, want %v", got, order[1:2])
	}
	history := resentHistory(t, upstream.bodies[0])
	if len(history) != 1 {
		t.Fatalf("re-sent %d assistant messages, want the one held", len(history))
	}
	if _, kept := history[0]["reasoning_details"]; kept {
		t.Fatalf("the re-send to %s carried %s's encrypted reasoning: %s", order[1], order[0], history[0]["reasoning_details"])
	}
	if string(history[0]["content"]) != `"earlier answer"` {
		t.Fatalf("the re-send lost the assistant's answer: %s", history[0]["content"])
	}
	if !transaction.thinkingRestart || transaction.thinkingAdopted != "" {
		t.Fatalf("the run did not restart at the stripped re-send: restart=%v adopted=%q",
			transaction.thinkingRestart, transaction.thinkingAdopted)
	}
	strip := findLogEvent(t, logs, "encrypted_reasoning_strip")
	assertFields(t, "strip", strip, map[string]interface{}{
		"from_endpoint": order[0], "to_endpoint": order[1],
		"from_host_family": signatureHostFamily(order[0]), "to_host_family": signatureHostFamily(order[1]),
	})
	if stripped := mustJSONRaw(t, strip["stripped"]); !sameJSON(t, stripped, "1") {
		t.Fatalf("stripped = %s, want 1", stripped)
	}
	resend := findLogEvent(t, logs, "endpoint_resend")
	if stripped := mustJSONRaw(t, resend["reasoning_stripped"]); !sameJSON(t, stripped, "1") {
		t.Fatalf("endpoint_resend reasoning_stripped = %s, want 1", stripped)
	}
}

// Within one host family the reasoning verifies on the next endpoint: it goes
// as the pick was sent it, and the run is left as it was. So it does with the
// guard off.
func TestSameFamilyOrUnguardedEndpointResendKeepsTheReasoning(t *testing.T) {
	cases := []struct {
		name  string
		tags  []string
		guard bool
	}{
		{"same family", []string{"anthropic", "azure"}, true},
		{"guard off", []string{"azure", "openai"}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			logs := captureLogs(t)
			upstream := newEndpointUpstream(t, map[string]int{})
			router := endpointPickRouter(upstream.server.URL, tc.tags...)
			router.Config.KeepUnservedClaudeThinking = !tc.guard
			ctx, _ := routeEndpointTurn(t, router, endpointTurn{key: endpointPickKey, feed: endpointFeed(t, tc.tags...)})
			withHeldReasoning(t, ctx)
			transaction := &raylineARCEpisodeTransaction{thinkingWorker: "run"}
			ctx.RaylineARCTransaction = transaction

			reply := answerHeaders(t, router, ctx, http.StatusServiceUnavailable)

			if code := int(reply.GetImmediateResponse().GetStatus().GetCode()); code != http.StatusOK {
				t.Fatalf("client status = %d, want the next endpoint's 200", code)
			}
			history := resentHistory(t, upstream.bodies[0])
			if len(history) != 1 || history[0]["reasoning_details"] == nil {
				t.Fatalf("the re-send lost the pick's reasoning: %v", history)
			}
			if transaction.thinkingRestart {
				t.Fatal("the run restarted though nothing was stripped")
			}
			for _, entry := range logs.All() {
				if name, _ := entry.ContextMap()["event"].(string); name == "encrypted_reasoning_strip" {
					t.Fatalf("a strip was logged: %v", entry.ContextMap())
				}
			}
		})
	}
}
