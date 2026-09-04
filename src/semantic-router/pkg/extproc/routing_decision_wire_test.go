package extproc

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// routing_decision recomputed the arm's configured effort and logged it as
// reasoning_effort, whatever the request actually carried. On build 16 that
// read as if the dial had been sent while the wire carried a bound alone, so
// the record and the request disagreed and the record was the one being
// believed.
//
// The record now names the controls the rendered body carries, read back off
// that body at the provider boundary. Reading the wire rather than recomputing
// the configuration is what makes the two unable to disagree.
func TestDispatchedReasoningControlsAreReadOffTheWire(t *testing.T) {
	tests := []struct {
		name       string
		body       string
		wantEffort string
		wantBound  *int64
	}{
		{
			name:       "a bound and the effort beside it",
			body:       `{"model":"m","reasoning":{"max_tokens":1024},"reasoning_effort":"low"}`,
			wantEffort: "low",
			wantBound:  int64Ptr(1024),
		},
		{
			name:       "an effort with no bound",
			body:       `{"model":"m","reasoning_effort":"high"}`,
			wantEffort: "high",
		},
		{
			// vLLM-compatible backends take the effort through the chat
			// template rather than as a request field.
			name:       "the effort a chat template carries",
			body:       `{"model":"m","chat_template_kwargs":{"reasoning_effort":"medium"}}`,
			wantEffort: "medium",
		},
		{
			name: "a turn that reasons about nothing",
			body: `{"model":"m","messages":[]}`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx := &RequestContext{}
			recordDispatchedReasoningControls(ctx, []byte(test.body))
			assert.Equal(t, test.wantEffort, ctx.DispatchedReasoningEffort)
			if test.wantBound == nil {
				assert.Nil(t, ctx.DispatchedReasoningBound)
				return
			}
			require.NotNil(t, ctx.DispatchedReasoningBound)
			assert.Equal(t, *test.wantBound, *ctx.DispatchedReasoningBound)
		})
	}
}

// The record is staged where the decision is made and written where the wire
// is known, because the body does not exist yet at the point the decision is
// taken. Staging alone must log nothing: a line written there could only
// report the configuration again.
func TestRoutingDecisionWaitsForTheWire(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := &RequestContext{RequestID: "rt_stage", ProcessingStartTime: time.Now()}

	router.logRoutingDecision(ctx, "entrypoint_routing", "auto", "gpt-5-mini", "arc", true)

	for _, entry := range logs.All() {
		if name, _ := entry.ContextMap()["event"].(string); name == "routing_decision" {
			t.Fatalf("the record was written before the wire existed: %v", entry.ContextMap())
		}
	}

	ctx.DispatchedReasoningEffort = "low"
	ctx.DispatchedReasoningBound = int64Ptr(1024)
	router.emitRoutingDecision(ctx)

	fields := findLogEvent(t, logs, "routing_decision")
	assert.Equal(t, "low", fields["reasoning_effort"])
	assert.Equal(t, int64(1024), fields["reasoning_max_tokens"])
	assert.Equal(t, "gpt-5-mini", fields["selected_model"])
}

// A turn whose body carries no reasoning control says so by carrying neither
// field, rather than by naming a level nothing was told to use.
func TestRoutingDecisionOmitsControlsTheWireDoesNotCarry(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := &RequestContext{RequestID: "rt_none", ProcessingStartTime: time.Now()}

	router.logRoutingDecision(ctx, "entrypoint_routing", "auto", "gpt-5-mini", "arc", true)
	router.emitRoutingDecision(ctx)

	fields := findLogEvent(t, logs, "routing_decision")
	for _, forbidden := range []string{"reasoning_effort", "reasoning_max_tokens"} {
		if _, present := fields[forbidden]; present {
			t.Fatalf("the record named a control the request does not carry: %v", fields)
		}
	}
	// Whether the arm reasons at all is the decision's own fact, and it stays.
	assert.Equal(t, true, fields["reasoning_enabled"])
}

func int64Ptr(value int64) *int64 {
	return &value
}
