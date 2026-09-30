package extproc

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A Claude Code turn in which the model searched streams through the router to
// the client with its search blocks and citations, and the settled response
// keeps them. Before this, the router refused the stream.
func TestAnthropicSearchedAnswerStreamsThroughTheRouter(t *testing.T) {
	raw, err := os.ReadFile("../protocolcodec/testdata/golden/stream/039-anthropic-web-search-in.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Chunks []string `json:"chunks"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	for _, client := range []llmprotocol.WireFormat{llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1} {
		t.Run(string(client), func(t *testing.T) {
			router := &OpenAIRouter{}
			ctx := &RequestContext{
				SourceFormat: client, TargetFormat: llmprotocol.AnthropicMessagesV1,
				RequestModel: "public-model", TraceContext: t.Context(),
			}
			if err := router.ensureSemanticResponseStream(ctx); err != nil {
				t.Fatal(err)
			}
			wire := pushExtProcStreamFixture(t, ctx, []byte(strings.Join(fixture.Chunks, "")))
			body := wire.Bytes()
			want := `"type":"server_tool_use"`
			if client == llmprotocol.OpenAIResponsesV1 {
				want = `"type":"web_search_call"`
			}
			if !bytes.Contains(body, []byte(want)) || !bytes.Contains(body, []byte("Go releases twice a year.")) {
				t.Fatalf("the searched answer did not reach the client:\n%s", body)
			}
			settled, err := ctx.SemanticStreamState.response()
			if err != nil {
				t.Fatalf("settle: %v", err)
			}
			if len(settled.Output) != 3 || settled.Output[0].Content[0].Unmodeled == nil {
				t.Fatalf("the settled response lost the search blocks: %+v", settled.Output)
			}
			if text := settled.Output[2].Content[0]; len(text.CitationsRaw) == 0 || len(text.Citations) != 1 {
				t.Fatalf("the settled answer lost its citations: %+v", text)
			}
		})
	}
}
