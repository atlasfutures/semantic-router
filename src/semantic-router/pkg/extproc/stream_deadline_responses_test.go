package extproc

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A turn the Router cut ends on the Responses lane as the resource the protocol
// defines for a response that stopped early.
//
// Responses has no [DONE] and no error frame that carries usage; what it does
// have is response.incomplete with incomplete_details, and that resource states
// usage like any other. So the count reaches the client without a member being
// invented, and the gateway's own reconstruction of a clean terminal can be
// deleted rather than extended (memex-desktop#7214).
func TestTruncatedResponsesStreamEndsIncomplete(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := overrunningStreamContext(t)
	ctx.SourceFormat = llmprotocol.OpenAIResponsesV1

	body := streamDeadlineResponseBody(t, router, ctx, endlessUpstreamChunk(1))

	require.Contains(t, string(body), "event: response.incomplete",
		"the turn did not end as the resource Responses defines for stopping early")
	terminal := truncatedResponsesTerminal(t, body)

	assert.Equal(t, "response.incomplete", terminal["type"])
	response, _ := terminal["response"].(map[string]any)
	require.NotNil(t, response, "the terminal event carried no response resource")
	assert.Equal(t, "incomplete", response["status"])

	// The reason is one of the values this repo's provider validation accepts
	// and the OpenAI SDKs narrow the field to. There is no enum value for a
	// router deadline, and coining one would be a string no SDK contains.
	details, _ := response["incomplete_details"].(map[string]any)
	require.NotNil(t, details, "an incomplete resource arrived without its details")
	assert.Equal(t, "max_output_tokens", details["reason"])

	usage, _ := response["usage"].(map[string]any)
	require.NotNil(t, usage, "the turn was billed as nothing")
	assert.Equal(t, float64(4), usage["output_tokens"],
		"the count was not what the stream carried")

	// The failure shapes the lane used to end on are both gone.
	assert.NotContains(t, string(body), "event: error")
	assert.NotContains(t, string(body), "response.failed")
}

// A Responses turn cut before anything was generated has no count to state, so
// it keeps the error event rather than publishing an incomplete resource whose
// usage is absent or zero. The client still receives a terminal.
func TestTruncatedResponsesStreamWithNoOutputStillTerminates(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := overrunningStreamContext(t)
	ctx.SourceFormat = llmprotocol.OpenAIResponsesV1

	body := streamDeadlineResponseBody(t, router, ctx, "")

	assert.Contains(t, string(body), `"code":"stream_truncated"`,
		"an uncounted cut did not name itself")
	assert.NotContains(t, string(body), "response.incomplete",
		"a turn that generated nothing published an incomplete resource")
	assert.NotContains(t, string(body), `"usage"`,
		"a turn that generated nothing was billed")
}

// truncatedResponsesTerminal is the data object of the stream's last SSE frame.
func truncatedResponsesTerminal(t *testing.T, body []byte) map[string]any {
	t.Helper()
	var last []byte
	for _, frame := range bytes.Split(body, []byte("\n\n")) {
		for _, line := range bytes.Split(frame, []byte("\n")) {
			if data, found := bytes.CutPrefix(line, []byte("data: ")); found {
				last = data
			}
		}
	}
	require.NotEmpty(t, last, "the stream carried no terminal frame")
	terminal := map[string]any{}
	require.NoError(t, json.Unmarshal(last, &terminal))
	return terminal
}
