//go:build !windows && cgo

package extproc

import (
	"bytes"
	"context"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// An upstream that keeps a quiet stream alive reaches the client as one
// keepalive per upstream keepalive, and a keepalive is not content: a stream
// cut after nothing but keepalives sent no content before its failure.
func TestKeepalivesReachTheClientAndAreNotContent(t *testing.T) {
	const anthropicStart = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"m\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":8}}}\n\n"
	const anthropicPing = "event: ping\ndata: {\"type\":\"ping\"}\n\n"
	const processing = ": OPENROUTER PROCESSING\n\n"
	cases := []struct {
		name           string
		source, target llmprotocol.WireFormat
		chunks         []string
		keepalive      string
	}{
		{
			name:   "anthropic pings after message_start",
			source: llmprotocol.AnthropicMessagesV1, target: llmprotocol.AnthropicMessagesV1,
			chunks:    []string{anthropicStart, anthropicPing, anthropicPing + anthropicPing},
			keepalive: anthropicPing,
		},
		{
			name:   "anthropic pings to a chat client",
			source: llmprotocol.AnthropicMessagesV1, target: llmprotocol.OpenAIChatV1,
			chunks:    []string{anthropicStart, anthropicPing, anthropicPing},
			keepalive: ": keepalive\n\n",
		},
		{
			name:   "chat processing comments to a messages client",
			source: llmprotocol.OpenAIChatV1, target: llmprotocol.AnthropicMessagesV1,
			chunks:    []string{processing, processing, processing},
			keepalive: ": keepalive\n\n",
		},
		{
			name:   "responses processing comments to a responses client",
			source: llmprotocol.OpenAIResponsesV1, target: llmprotocol.OpenAIResponsesV1,
			chunks:    []string{processing, processing},
			keepalive: ": keepalive\n\n",
		},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			logs := captureLogs(t)
			// The client speaks the target format; the upstream speaks the source.
			stream, err := protocolcodec.NewBuiltinEngine().NewStream(test.source, test.target,
				llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
			if err != nil {
				t.Fatal(err)
			}
			ctx := &RequestContext{
				RequestID: "req-keepalive", RequestModel: "test", StartTime: time.Now(), UpstreamStatusCode: 200,
				IsStreamingResponse: true, SourceFormat: test.target, TargetFormat: test.source,
				ProtocolResponseStream: stream,
				SemanticStreamState:    &semanticResponseStreamState{items: map[int]*semanticStreamItem{}},
			}
			buffers := &semanticStreamBuffers{}
			upstreamKeepalives := 0
			for _, chunk := range test.chunks {
				upstreamKeepalives += bytes.Count([]byte(chunk), []byte("event: ping\n")) +
					bytes.Count([]byte(chunk), []byte(processing))
				buffers.push([]byte(chunk), ctx)
			}
			// The upstream goes away before any content.
			buffers.finalize(ctx)
			(&OpenAIRouter{}).finalizeSemanticStreamingResponse(ctx, buffers.streamErr)

			if got := bytes.Count(buffers.translated, []byte(test.keepalive)); got != upstreamKeepalives {
				t.Fatalf("client got %d keepalives for %d upstream ones:\n%s", got, upstreamKeepalives, buffers.translated)
			}
			if len(ctx.SemanticStreamState.items) != 0 {
				t.Fatalf("a keepalive was observed as content: %d items", len(ctx.SemanticStreamState.items))
			}
			usage := findLogEvent(t, logs, "llm_usage")
			if usage["failure_class"] == nil || usage["content_sent_before_failure"] != false {
				t.Fatalf("llm_usage = %#v, want a classed failure with no content sent before it", usage)
			}
			if failed := findLogEvent(t, logs, "turn_failed"); failed["content_sent_before_failure"] != false {
				t.Fatalf("turn_failed = %#v, want no content sent before the failure", failed)
			}
		})
	}
}
