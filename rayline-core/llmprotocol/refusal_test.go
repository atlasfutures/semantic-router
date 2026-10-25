package llmprotocol

import "testing"

func TestResponseRefusalIncludesEmptyRefusalParts(t *testing.T) {
	for _, response := range []*Response{
		{StopReason: StopContentFilter},
		{StopReason: StopEndTurn, Output: []OutputItem{{Content: []Content{{Kind: ContentRefusal}}}}},
	} {
		if !ResponseRefused(response) {
			t.Fatalf("refusal counted as a completed generation: %+v", response)
		}
	}
	if ResponseRefused(nil) || ResponseRefused(&Response{StopReason: StopEndTurn, Output: []OutputItem{{Content: []Content{{Kind: ContentText, Text: "answer"}}}}}) {
		t.Fatal("ordinary response counted as a refusal")
	}
}
