package protocolcodec

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// responsesInputIDs encodes a Chat history as a Responses request and returns
// its input item ids in order.
func responsesInputIDs(t *testing.T, chat string) []string {
	t.Helper()
	engine := NewBuiltinEngine()
	request, envelope, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, []byte(chat))
	if err != nil {
		t.Fatal(err)
	}
	request.Model = "target-model"
	request.Generation++
	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Input []struct {
			ID string `json:"id"`
		} `json:"input"`
	}
	if err := json.Unmarshal(encoded.Body, &wire); err != nil {
		t.Fatal(err)
	}
	ids := make([]string, 0, len(wire.Input))
	for _, item := range wire.Input {
		ids = append(ids, item.ID)
	}
	return ids
}

// Every item of a request translated into Responses has its own id, though
// none of its messages had one; before, every id-less message's first item
// shared one. Appending a turn keeps the earlier items' ids, so a resent
// history repeats them.
func TestResponsesItemIDsAreUniqueAndStable(t *testing.T) {
	const turns = `{"role":"user","content":"one"},{"role":"assistant","content":"two"},{"role":"user","content":"three"}`
	ids := responsesInputIDs(t, `{"model":"m","messages":[`+turns+`]}`)
	seen := map[string]bool{}
	for _, id := range ids {
		if id == "" || seen[id] {
			t.Fatalf("item ids are not unique: %v", ids)
		}
		seen[id] = true
	}
	longer := responsesInputIDs(t, `{"model":"m","messages":[`+turns+`,{"role":"assistant","content":"four"},{"role":"user","content":"five"}]}`)
	for index, id := range ids {
		if longer[index] != id {
			t.Fatalf("item %d changed id when a turn was appended: %v then %v", index, ids, longer)
		}
	}
}
