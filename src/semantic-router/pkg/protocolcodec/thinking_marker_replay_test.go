package protocolcodec

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A request that reaches its own format unchanged is sent as the client's
// bytes (Envelope.CanReplay). Those bytes still hold any Router signature the
// decoder stripped or the encoder would refuse, so a request holding one is
// never replayed: it is encoded, and the guards apply.
func TestSourceReplayNeverSendsARouterSignature(t *testing.T) {
	marker := "vsr.thinking.v1.moonshotai." + testMarkerDigest
	// The same marker with its first letters written as JSON escapes: the
	// decoded value is the marker, the bytes do not contain it literally.
	escaped := `\u0076\u0073r.thinking.v1.moonshotai.` + testMarkerDigest
	bodies := map[llmprotocol.WireFormat][]string{
		llmprotocol.OpenAIChatV1: {
			`{"model":"m","messages":[{"role":"user","content":"hi"},` +
				`{"role":"assistant","content":"ok","reasoning_content":"kimi reasoning","reasoning_details":[` +
				`{"type":"reasoning.text","text":"kimi reasoning","signature":"SIG","format":"anthropic-claude-v1","index":0}]},` +
				`{"role":"user","content":"go on"}]}`,
		},
		llmprotocol.AnthropicMessagesV1: {
			`{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"hi"},` +
				`{"role":"assistant","content":[{"type":"thinking","thinking":"kimi reasoning","signature":"SIG"},{"type":"text","text":"ok"}]},` +
				`{"role":"user","content":"go on"}]}`,
		},
		llmprotocol.OpenAIResponsesV1: {
			`{"model":"m","input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]},` +
				`{"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"kimi reasoning"}],"signature":"SIG","format":"anthropic-claude-v1"},` +
				`{"role":"user","content":[{"type":"input_text","text":"go on"}]}]}`,
		},
	}
	engine := NewBuiltinEngine()
	for format, templates := range bodies {
		for _, template := range templates {
			for name, signature := range map[string]string{"literal": marker, "escaped": escaped} {
				t.Run(string(format)+"/"+name, func(t *testing.T) {
					body := strings.Replace(template, "SIG", signature, 1)
					request, envelope, _, err := engine.DecodeRequest(format, []byte(body))
					if err != nil {
						t.Fatalf("decode: %v", err)
					}
					result, err := engine.EncodeRequest(format, request, envelope)
					if err != nil {
						t.Fatalf("encode: %v", err)
					}
					sent := string(result.Body)
					if strings.Contains(sent, "vsr.") || strings.Contains(sent, `vsr.`) {
						t.Fatalf("a Router signature was replayed to the provider: %s", sent)
					}
				})
			}
		}
	}
}

func TestSourceReplayStillReplaysRequestsWithoutARouterSignature(t *testing.T) {
	body := `{"model":"m","messages":[{"role":"user","content":"hi"}],"temperature":1}`
	request, envelope, _, err := NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIChatV1, []byte(body))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !envelope.CanReplay(llmprotocol.OpenAIChatV1, request.Generation, NewBuiltinEngine().requestEncodePolicy(), false) {
		t.Fatal("a request without a Router signature lost source replay")
	}
}

// A stripped body is not replayed, but what the client asked for is still
// read from its bytes: here, reasoning {summary: auto} with no effort, which
// a same-format dispatch keeps. Only the replay decision may differ from an
// unstripped body.
func TestStrippedBodyKeepsTheClientsReasoningIntent(t *testing.T) {
	marker := "vsr.thinking.v1.moonshotai." + testMarkerDigest
	for name, signature := range map[string]string{"stripped": marker, "unstripped": "EqQBreal"} {
		t.Run(name, func(t *testing.T) {
			body := `{"model":"m","reasoning":{"summary":"auto"},"input":[` +
				`{"role":"user","content":[{"type":"input_text","text":"hi"}]},` +
				`{"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"kimi reasoning"}],"signature":"` + signature + `","format":"anthropic-claude-v1"},` +
				`{"role":"assistant","content":[{"type":"output_text","text":"ok"}]},` +
				`{"role":"user","content":[{"type":"input_text","text":"go on"}]}]}`
			result, err := NewBuiltinEngine().TranslateRequest(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1, []byte(body),
				func(request *llmprotocol.Request) error { request.Model = "routed-model"; return nil })
			if err != nil {
				t.Fatalf("translate: %v", err)
			}
			sent := string(result.Body)
			if strings.Contains(sent, "vsr.") || !strings.Contains(sent, `"summary":"auto"`) {
				t.Fatalf("Responses target request = %s", sent)
			}
		})
	}
}
