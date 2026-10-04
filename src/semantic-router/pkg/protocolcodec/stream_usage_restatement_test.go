package protocolcodec

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func anthropicUsageStream(startUsage, deltaUsage string) string {
	return "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"provider-model\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":" + startUsage + "}}\n\n" +
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"the answer\"}}\n\n" +
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":" + deltaUsage + "}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
}

// A provider that re-divides a total between its splits mid-stream -- here
// input first stated all uncached, then restated as mostly cache reads -- has
// not contradicted itself: the input total stands. The stream completes, the
// published usage is the provider's final statement, and a diagnostic names
// the split that went down. A total that goes backwards is still a broken
// stream, and its error names the counter.
func TestStreamUsageSplitRestatementCompletesAndTotalsStillFail(t *testing.T) {
	restated := anthropicUsageStream(
		`{"input_tokens":100,"cache_read_input_tokens":0,"output_tokens":1}`,
		`{"input_tokens":10,"cache_read_input_tokens":90,"output_tokens":9}`)
	for _, target := range []llmprotocol.WireFormat{llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1} {
		stream, err := NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, target,
			llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
		if err != nil {
			t.Fatal(err)
		}
		_, events, diagnostics, err := stream.Push([]byte(restated))
		if err != nil {
			t.Fatalf("-> %s: a split restated lower failed the stream: %v", target, err)
		}
		_, finalEvents, finalDiagnostics, err := stream.Finalize(nil)
		if err != nil {
			t.Fatalf("-> %s: finalize: %v", target, err)
		}
		events = append(events, finalEvents...)
		diagnostics = append(diagnostics, finalDiagnostics...)
		var final *llmprotocol.Usage
		for index := range events {
			if events[index].Type == llmprotocol.EventResponseCompleted {
				final = events[index].Usage
			}
		}
		if final == nil || final.InputUncached.Value == nil || *final.InputUncached.Value != 10 ||
			final.InputCacheRead.Value == nil || *final.InputCacheRead.Value != 90 ||
			final.InputTotal.Value == nil || *final.InputTotal.Value != 100 {
			t.Fatalf("-> %s: completed usage = %+v, want the final statement (10 uncached, 90 cache read, 100 input)", target, final)
		}
		named := false
		for _, diagnostic := range diagnostics {
			named = named || diagnostic.Field == "usage.input_uncached"
		}
		if !named {
			t.Fatalf("-> %s: no diagnostic named the restated split: %+v", target, diagnostics)
		}
	}

	// Control: the input total itself goes backwards.
	shrunk := anthropicUsageStream(
		`{"input_tokens":100,"cache_read_input_tokens":0,"output_tokens":1}`,
		`{"input_tokens":10,"cache_read_input_tokens":80,"output_tokens":9}`)
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1,
		llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	if err != nil {
		t.Fatal(err)
	}
	_, _, _, err = stream.Push([]byte(shrunk))
	if err == nil {
		_, _, _, err = stream.Finalize(nil)
	}
	if err == nil || !strings.Contains(err.Error(), "usage counter decreased: input_total") {
		t.Fatalf("a decreasing input total = %v, want a usage_decreased error naming input_total", err)
	}
}

// A split restated lower on weaker evidence than it was stated on is not a
// restatement the stream can take: the provider has not said the new value
// with the authority of the old one.
func TestUsageSplitRestatementNeedsEvidenceAsStrong(t *testing.T) {
	current := llmprotocol.Usage{State: llmprotocol.UsageAvailable,
		InputCacheRead: llmprotocol.TokenCount{Value: llmprotocol.Int64(90), Provenance: llmprotocol.UsageAuthoritative}}
	update := llmprotocol.Usage{State: llmprotocol.UsageAvailable,
		InputCacheRead: llmprotocol.TokenCount{Value: llmprotocol.Int64(80), Provenance: llmprotocol.UsageDerived}}
	if _, _, err := mergeMonotonicUsage(current, update); err == nil {
		t.Fatal("a split restated lower on weaker evidence was accepted")
	}
}

// A caller of the exported decoder, without the stream engine, sees the same
// diagnostic.
func TestUsageSplitRestatementDiagnosticFromTheExportedDecoder(t *testing.T) {
	decoder := AnthropicMessagesCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"},
		NewBuiltinEngine().providerStreamPolicy())
	_, diagnostics, err := decoder.Push([]byte(anthropicUsageStream(
		`{"input_tokens":100,"cache_read_input_tokens":0,"output_tokens":1}`,
		`{"input_tokens":10,"cache_read_input_tokens":90,"output_tokens":9}`)))
	if err != nil {
		t.Fatal(err)
	}
	// The Push that carried the restatement reports it.
	for _, diagnostic := range diagnostics {
		if diagnostic.Field == "usage.input_uncached" {
			return
		}
	}
	t.Fatalf("the exported decoder's Push did not report the restated split: %+v", diagnostics)
}

// More restatements in one Push than the policy's diagnostic limit end in the
// truncation marker, not a list that looks complete.
func TestUsageSplitRestatementsBeyondTheLimitAreMarkedTruncated(t *testing.T) {
	policy := NewBuiltinEngine().providerStreamPolicy()
	policy.Limits.Diagnostics = 3
	decoder := AnthropicMessagesCodec{}.NewDecoder(llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"}, policy)
	stream := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"provider-model\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":100,\"output_tokens\":1}}}\n\n"
	for uncached := 90; uncached >= 40; uncached -= 10 {
		stream += fmt.Sprintf("event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":null,\"stop_sequence\":null},\"usage\":{\"input_tokens\":%d,\"cache_read_input_tokens\":%d,\"output_tokens\":1}}\n\n", uncached, 100-uncached)
	}
	_, diagnostics, err := decoder.Push([]byte(stream))
	if err != nil {
		t.Fatal(err)
	}
	if len(diagnostics) != 3 || diagnostics[2].Action != llmprotocol.DiagnosticTruncated {
		t.Fatalf("diagnostics = %+v, want 3 ending in the truncation marker", diagnostics)
	}
}
