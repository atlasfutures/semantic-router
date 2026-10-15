package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A cache leg the upstream did not report is unknown, never a reported 0
// (memex-desktop #7417, #6822): a router that reads it as 0 believes the
// prefix was neither read nor written.
func TestUnreportedCacheLegsStayUnknown(t *testing.T) {
	known := func(count llmprotocol.TokenCount) bool { return count.Value != nil }
	value := func(count llmprotocol.TokenCount) int64 { return *count.Value }

	var responses responsesUsageWire
	if err := json.Unmarshal([]byte(`{"input_tokens":100,"output_tokens":5,"total_tokens":105}`), &responses); err != nil {
		t.Fatal(err)
	}
	usage := decodeResponsesUsage(responses)
	if known(usage.InputCacheRead) || known(usage.InputCacheWrite) || known(usage.InputUncached) {
		t.Fatalf("a Responses usage without input_tokens_details reported cache legs: %+v", usage)
	}

	var chat chatUsageWire
	if err := json.Unmarshal([]byte(`{"prompt_tokens":100,"completion_tokens":5,"total_tokens":105,"prompt_tokens_details":{"audio_tokens":0}}`), &chat); err != nil {
		t.Fatal(err)
	}
	usage = decodeChatUsage(chat)
	if known(usage.InputCacheRead) || known(usage.InputCacheWrite) {
		t.Fatalf("a Chat usage whose details omit cached_tokens reported cache legs: %+v", usage)
	}

	// A reported read with no write field: neither schema bills a write it does not state.
	if err := json.Unmarshal([]byte(`{"prompt_tokens":100,"completion_tokens":5,"total_tokens":105,"prompt_tokens_details":{"cached_tokens":80}}`), &chat); err != nil {
		t.Fatal(err)
	}
	usage = decodeChatUsage(chat)
	if !known(usage.InputCacheRead) || value(usage.InputCacheRead) != 80 || value(usage.InputCacheWrite) != 0 ||
		value(usage.InputUncached) != 20 {
		t.Fatalf("a reported read decoded wrong: %+v", usage)
	}

	if err := json.Unmarshal([]byte(`{"input_tokens":100,"output_tokens":5,"total_tokens":105,"input_tokens_details":{"cached_tokens":60,"cache_write_tokens":30}}`), &responses); err != nil {
		t.Fatal(err)
	}
	usage = decodeResponsesUsage(responses)
	if value(usage.InputCacheRead) != 60 || value(usage.InputCacheWrite) != 30 || value(usage.InputUncached) != 10 {
		t.Fatalf("reported Responses legs decoded wrong: %+v", usage)
	}
}

// The encoded client bytes are unchanged: Chat states cached_tokens whenever a
// cache leg is known and cache_write_tokens only when nonzero.
func TestEncodedCacheLegsKeepTheirBytes(t *testing.T) {
	read := authoritative(7)
	wire := encodeChatUsage(llmprotocol.Usage{State: llmprotocol.UsageAvailable, InputTotal: authoritative(10),
		OutputTotal: authoritative(1), Total: authoritative(11), InputCacheRead: read, InputCacheWrite: authoritative(0)})
	body, err := json.Marshal(wire.PromptTokensDetails)
	if err != nil {
		t.Fatal(err)
	}
	if string(body) != `{"cached_tokens":7}` {
		t.Fatalf("chat prompt_tokens_details bytes changed: %s", body)
	}
}
