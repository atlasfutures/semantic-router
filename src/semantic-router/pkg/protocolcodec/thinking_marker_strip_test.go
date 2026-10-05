package protocolcodec

import (
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestStripRouterSignaturesLeavesOtherBodiesByteForByte(t *testing.T) {
	policy := NewBuiltinEngine().policy
	for format, body := range map[llmprotocol.WireFormat]string{
		llmprotocol.AnthropicMessagesV1: `{"model":"m","max_tokens":8,"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"t","signature":"EqQB"}]},{"role":"user","content":"vsr. in text is text"}]}`,
		llmprotocol.OpenAIChatV1:        `{"model":"m","messages":[{"role":"assistant","content":"vsr.x","reasoning_details":[{"type":"reasoning.text","signature":"EqQB"}]}]}`,
		llmprotocol.OpenAIResponsesV1:   `{"model":"m","input":"vsr.thinking.v1.moonshotai.x"}`,
	} {
		got, stripped := stripRouterSignatures(format, []byte(body), policy)
		if stripped || string(got) != body {
			t.Fatalf("%s: stripped=%v body=%s", format, stripped, got)
		}
	}
	// A body the decoder will refuse is left for it to refuse.
	duplicate := `{"model":"m","messages":[],"messages":[{"role":"assistant","reasoning_details":[{"signature":"vsr.x"}]}]}`
	if got, stripped := stripRouterSignatures(llmprotocol.OpenAIChatV1, []byte(duplicate), policy); stripped || string(got) != duplicate {
		t.Fatalf("a duplicate-key body was rewritten: %s", got)
	}
}

func TestStripRouterSignaturesRemovesEverySignaturePlace(t *testing.T) {
	policy := NewBuiltinEngine().policy
	marker := "vsr.thinking.v1.moonshotai." + testMarkerDigest
	minted := func(details string) string {
		blob := mintReasoningDetails(json.RawMessage(details))
		return string(blob)
	}
	cases := []struct {
		name   string
		format llmprotocol.WireFormat
		body   string
		keep   []string
	}{
		{
			"messages thinking", llmprotocol.AnthropicMessagesV1,
			`{"model":"m","max_tokens":8,"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"kimi","signature":"` + marker + `"}]}]}`,
			[]string{`"thinking":"kimi"`, `"signature":""`},
		},
		{
			"messages escaped", llmprotocol.AnthropicMessagesV1,
			`{"model":"m","max_tokens":8,"messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"kimi","signature":"\u0076sr.x"}]}]}`,
			[]string{`"signature":""`},
		},
		{
			"chat details", llmprotocol.OpenAIChatV1,
			`{"model":"m","messages":[{"role":"assistant","content":"ok","reasoning_details":[{"type":"reasoning.text","signature":"` + marker + `","format":"anthropic-claude-v1"},{"type":"reasoning.encrypted","data":"gAAA"}]}]}`,
			[]string{`"data":"gAAA"`},
		},
		{
			"responses signature", llmprotocol.OpenAIResponsesV1,
			`{"model":"m","input":[{"type":"reasoning","summary":[],"signature":"` + marker + `","format":"anthropic-claude-v1","x":1}]}`,
			[]string{`"format":"anthropic-claude-v1"`, `"x":1`},
		},
		{
			"responses minted, emptied", llmprotocol.OpenAIResponsesV1,
			`{"model":"m","input":[{"type":"reasoning","summary":[],"encrypted_content":` + minted(`[{"signature":"`+marker+`"}]`) + `}]}`,
			[]string{`"type":"reasoning"`},
		},
		{
			"responses minted, mixed", llmprotocol.OpenAIResponsesV1,
			`{"model":"m","input":[{"type":"reasoning","summary":[],"encrypted_content":` + minted(`[{"signature":"`+marker+`"},{"data":"gAAA"}]`) + `}]}`,
			[]string{`"encrypted_content":"vsr.reasoning_details.v1.`},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, stripped := stripRouterSignatures(tc.format, []byte(tc.body), policy)
			if !stripped {
				t.Fatalf("not stripped: %s", got)
			}
			if strings.Contains(string(got), "vsr.thinking") || strings.Contains(string(got), `\u0076sr`) {
				t.Fatalf("a Router signature survived: %s", got)
			}
			for _, want := range tc.keep {
				if !strings.Contains(string(got), want) {
					t.Fatalf("%s lost %s", got, want)
				}
			}
		})
	}
	// The mixed blob keeps its other item, still minted.
	body := `{"model":"m","input":[{"type":"reasoning","summary":[],"encrypted_content":` + minted(`[{"signature":"`+marker+`"},{"data":"gAAA"}]`) + `}]}`
	got, _ := stripRouterSignatures(llmprotocol.OpenAIResponsesV1, []byte(body), policy)
	var request struct {
		Input []struct {
			EncryptedContent json.RawMessage `json:"encrypted_content"`
		} `json:"input"`
	}
	if err := json.Unmarshal(got, &request); err != nil || len(request.Input) != 1 {
		t.Fatalf("stripped body = %s", got)
	}
	details, ok := mintedReasoningDetails(request.Input[0].EncryptedContent)
	if !ok || strings.Contains(string(details), "vsr.") || !strings.Contains(string(details), "gAAA") {
		t.Fatalf("re-minted blob holds %s, %v", details, ok)
	}
}

// A minted blob is base64, so the body's duplicate-key check cannot see
// inside it. An item that names its signature twice -- a Router one, then
// another -- must not survive with the Router one in its bytes.
func TestStripRouterSignaturesDropsAMintedBlobWithDuplicateKeys(t *testing.T) {
	policy := NewBuiltinEngine().policy
	marker := "vsr.thinking.v1.moonshotai." + testMarkerDigest
	for name, item := range map[string]string{
		"router then provider": `{"type":"reasoning.text","text":"kimi","signature":"` + marker + `","signature":"EqQB","format":"anthropic-claude-v1"}`,
		"case-folded":          `{"type":"reasoning.text","text":"kimi","signature":"EqQB","Signature":"` + marker + `"}`,
	} {
		t.Run(name, func(t *testing.T) {
			blob := "vsr.reasoning_details.v1." + base64.RawURLEncoding.EncodeToString([]byte(`[`+item+`]`))
			body := `{"model":"m","input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]},` +
				`{"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"kimi"}],"encrypted_content":"` + blob + `"},` +
				`{"role":"user","content":[{"type":"input_text","text":"go on"}]}]}`
			got, stripped := stripRouterSignatures(llmprotocol.OpenAIResponsesV1, []byte(body), policy)
			if !stripped || strings.Contains(string(got), "encrypted_content") {
				t.Fatalf("the duplicate-key blob was kept (stripped=%v): %s", stripped, got)
			}
			engine := NewBuiltinEngine()
			for _, target := range goldenFormats {
				result, err := engine.TranslateRequest(llmprotocol.OpenAIResponsesV1, target.format, []byte(body),
					func(*llmprotocol.Request) error { return nil })
				if err != nil {
					t.Fatalf("%s: %v", target.name, err)
				}
				if strings.Contains(string(result.Body), "vsr.") || strings.Contains(string(result.Body), "EqQB") {
					t.Fatalf("%s request kept the forged item: %s", target.name, result.Body)
				}
			}
		})
	}
}

// Rebuilding a stripped body must not HTML-escape the client's text: a body
// within the size limit stays within it.
func TestStripRouterSignaturesDoesNotGrowTheBody(t *testing.T) {
	policy := NewBuiltinEngine().policy
	text := strings.Repeat("<a & b>", 1000)
	marker := "vsr.thinking.v1.moonshotai." + testMarkerDigest
	body := `{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"` + text + `"},` +
		`{"role":"assistant","content":[{"type":"thinking","thinking":"` + text + `","signature":"` + marker + `"}]}]}`
	got, stripped := stripRouterSignatures(llmprotocol.AnthropicMessagesV1, []byte(body), policy)
	if !stripped {
		t.Fatal("not stripped")
	}
	if strings.Count(string(got), text) != 2 {
		t.Fatalf("the text was re-escaped: %.200s", got)
	}
	if removed := len(marker); len(got) > len(body)-removed {
		t.Fatalf("stripped body is %d bytes, want at most %d", len(got), len(body)-removed)
	}
}
