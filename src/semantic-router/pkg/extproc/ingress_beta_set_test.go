package extproc

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Rule 5. The base Anthropic version has not moved since 2023-06-01; the
// protocol moves through anthropic-beta instead. The cell neither read nor
// recorded that header, so the effective protocol version of a request had no
// representation anywhere, and every member that appeared cost an
// investigation to attribute.
//
// Reading it is not permission to act on it. Nothing may refuse on a beta, and
// nothing may route on one: the gateway does not replay a body-level 400, so a
// refusal on an unrecognised beta would end a user's turn over a header the
// cell has no opinion about.

func TestAnthropicBetaSetIsSortedAndDeduplicated(t *testing.T) {
	got := anthropicBetaSet(map[string]string{
		"anthropic-beta": "effort-2025-11-24, extended-cache-ttl-2025-04-11 ,,effort-2025-11-24",
	})
	want := []string{"effort-2025-11-24", "extended-cache-ttl-2025-04-11"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("beta set = %v, want %v", got, want)
	}
}

// Envoy lowercases header names, but the cell also builds a RequestContext in
// tests and in the looper path, so the lookup does not depend on which.
func TestAnthropicBetaSetReadsTheHeaderWhateverItsCase(t *testing.T) {
	got := anthropicBetaSet(map[string]string{"Anthropic-Beta": "context-management-2025-06-27"})
	want := []string{"context-management-2025-06-27"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("beta set = %v, want %v", got, want)
	}
}

func TestAnthropicBetaSetIsEmptyWithoutTheHeader(t *testing.T) {
	for name, headers := range map[string]map[string]string{
		"no header":    {":path": "/v1/messages"},
		"empty header": {"anthropic-beta": "  ,  "},
		"nil headers":  nil,
	} {
		if got := anthropicBetaSet(headers); len(got) != 0 {
			t.Fatalf("%s: beta set = %v, want empty", name, got)
		}
	}
}

// The line is per request, and it carries the request id, because attributing
// a member to a beta means joining one turn to one header.
func TestIngressLogsTheBetaSetPerRequest(t *testing.T) {
	logs := captureLogs(t)
	logIngressBetaSet(&RequestContext{
		RequestID:    "rt_ae5e3e62-4d9",
		SourceFormat: llmprotocol.AnthropicMessagesV1,
		Headers: map[string]string{
			"anthropic-beta": "interleaved-thinking-2025-05-14,effort-2025-11-24",
		},
	})
	entries := logs.All()
	if len(entries) != 1 {
		t.Fatalf("a declared beta set wrote %d log lines, want exactly 1", len(entries))
	}
	fields := entries[0].ContextMap()
	if event, _ := fields["event"].(string); event != "ingress_beta_set" {
		t.Fatalf("event = %q, want ingress_beta_set", event)
	}
	if id, _ := fields["request_id"].(string); id != "rt_ae5e3e62-4d9" {
		t.Fatalf("request_id = %q, want rt_ae5e3e62-4d9", id)
	}
	if format, _ := fields["format"].(string); format != string(llmprotocol.AnthropicMessagesV1) {
		t.Fatalf("format = %q, want the Messages format", format)
	}
	betas, _ := fields["anthropic_betas"].(string)
	if want := "effort-2025-11-24,interleaved-thinking-2025-05-14"; betas != want {
		t.Fatalf("anthropic_betas = %q, want %q", betas, want)
	}
}

// A request that declares nothing says nothing. Most traffic on the OpenAI
// formats carries no such header, and a line per request stating "none" would
// be the noise the cell already avoids on the provider field report.
func TestIngressWritesNoBetaLineWhenNoneIsDeclared(t *testing.T) {
	logs := captureLogs(t)
	logIngressBetaSet(&RequestContext{RequestID: "rt_1", SourceFormat: llmprotocol.OpenAIChatV1})
	if entries := logs.All(); len(entries) != 0 {
		t.Fatalf("a request declaring no beta wrote %d log lines, want 0", len(entries))
	}
}

// A refusal is the moment the attribution is worth most: it is the one record
// of the turn that survives, and the body it came from is never stored.
func TestIngressRefusalNamesTheBetaSet(t *testing.T) {
	logs := captureLogs(t)
	recordIngressProtocolError(&RequestContext{
		RequestID:    "rt_ffae7c99-26e",
		SourceFormat: llmprotocol.AnthropicMessagesV1,
		Headers:      map[string]string{"anthropic-beta": "prompt-caching-evict-2026-05-12"},
	}, fmt.Errorf("boom"))

	entries := logs.All()
	if len(entries) != 1 {
		t.Fatalf("an ingress rejection wrote %d log lines, want exactly 1", len(entries))
	}
	betas, _ := entries[0].ContextMap()["anthropic_betas"].(string)
	if want := "prompt-caching-evict-2026-05-12"; betas != want {
		t.Fatalf("anthropic_betas = %q, want %q", betas, want)
	}
}

// A refused request that declared no beta must not grow an empty field: an
// absent header and a header declaring nothing are the same thing, and a
// present-but-empty field reads as neither.
func TestIngressRefusalOmitsAnUndeclaredBetaSet(t *testing.T) {
	logs := captureLogs(t)
	recordIngressProtocolError(&RequestContext{RequestID: "rt_1"}, fmt.Errorf("boom"))
	entries := logs.All()
	if len(entries) != 1 {
		t.Fatalf("an ingress rejection wrote %d log lines, want exactly 1", len(entries))
	}
	if _, present := entries[0].ContextMap()["anthropic_betas"]; present {
		t.Fatal("a request declaring no beta carried an anthropic_betas field")
	}
}

// The rule the other tests do not state: reading the header changes nothing.
// A beta value the cell has never heard of routes exactly as one it knows.
func TestIngressNeverRefusesOnTheBetaHeader(t *testing.T) {
	router, _ := routingTestRouterForFormat(llmprotocol.AnthropicMessagesV1)
	body := []byte(`{"model":"virtual","max_tokens":32,` +
		`"messages":[{"role":"user","content":"hello"}]}`)
	for _, beta := range []string{
		"",
		"effort-2025-11-24",
		"a-beta-nobody-has-published-2099-01-01",
	} {
		ctx := &RequestContext{
			Headers:      map[string]string{"anthropic-beta": beta},
			SourceFormat: llmprotocol.AnthropicMessagesV1,
			RequestID:    "beta-" + beta,
			TraceContext: context.Background(),
		}
		request, immediate := router.prepareProtocolRequest(body, ctx)
		if immediate != nil {
			t.Fatalf("anthropic-beta %q was refused at ingress", beta)
		}
		if request == nil {
			t.Fatalf("anthropic-beta %q produced no request", beta)
		}
	}
}
