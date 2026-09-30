package extproc

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// The provider-bound body carries a client's web_search declaration only when
// the dispatched model's card admits it (DispatchHostedTools).
func TestDispatchSendsWebSearchOnlyToAnAdmittedModel(t *testing.T) {
	body := []byte(`{"model":"m","input":"what shipped in go 1.30?","tools":[{"type":"web_search"}]}`)
	for name, tc := range map[string]struct {
		hosted []string
		want   bool
	}{
		"admitted":     {[]string{"web_search"}, true},
		"not admitted": {nil, false},
	} {
		t.Run(name, func(t *testing.T) {
			request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, body)
			if err != nil {
				t.Fatal(err)
			}
			request.Model = "provider-model"
			request.Generation++
			ctx := &RequestContext{
				SourceFormat: llmprotocol.OpenAIResponsesV1, TargetFormat: llmprotocol.OpenAIResponsesV1,
				SemanticRequest: &request, ProtocolEnvelope: envelope, DispatchHostedTools: tc.hosted,
			}
			encoded, err := (&OpenAIRouter{}).encodeDispatchRequest(ctx)
			if err != nil {
				t.Fatal(err)
			}
			if got := bytes.Contains(encoded, []byte(`"type":"web_search"`)); got != tc.want {
				t.Fatalf("web_search sent = %v, want %v: %s", got, tc.want, encoded)
			}
		})
	}
}

// A searched answer streams through the router to a Responses client with its
// web_search_call item, and the settled response keeps the answer.
func TestSearchedAnswerStreamsThroughTheRouter(t *testing.T) {
	stream, err := os.ReadFile("../protocolcodec/testdata/golden/stream/038-responses-web-search-in.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Chunks []string `json:"chunks"`
	}
	if err = json.Unmarshal(stream, &fixture); err != nil {
		t.Fatal(err)
	}
	router := &OpenAIRouter{}
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIResponsesV1, TargetFormat: llmprotocol.OpenAIResponsesV1,
		RequestModel: "public-model", TraceContext: t.Context(),
	}
	if err = router.ensureSemanticResponseStream(ctx); err != nil {
		t.Fatal(err)
	}
	wire := pushExtProcStreamFixture(t, ctx, []byte(strings.Join(fixture.Chunks, "")))
	client := wire.Bytes()
	if !bytes.Contains(client, []byte(`"type":"web_search_call"`)) || !bytes.Contains(client, []byte(`"query":"go release history"`)) {
		t.Fatalf("the search item did not reach the client:\n%s", client)
	}
	settled, err := ctx.SemanticStreamState.response()
	if err != nil {
		t.Fatalf("settle: %v", err)
	}
	if len(settled.Output) != 2 || len(settled.Output[0].Content) != 1 ||
		settled.Output[0].Content[0].Unmodeled == nil ||
		!bytes.Contains(settled.Output[0].Content[0].Unmodeled.Raw, []byte(`"query":"go release history"`)) {
		t.Fatalf("the settled response lost the search item: %+v", settled.Output)
	}
}
