package llmprotocol

import (
	"encoding/json"
	"strings"
	"testing"
)

// The cache identity hashes a request's JSON, so a router-only flag must not
// appear on the messages that do not carry it: every cache entry written
// before the flag existed stays reachable.
func TestJoinsToolMediaIsAbsentFromAnUnmarkedMessage(t *testing.T) {
	plain, err := json.Marshal(Message{Role: RoleUser, Content: []Content{{Kind: ContentText, Text: "go"}}})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(plain), "JoinsToolMedia") {
		t.Fatalf("an unmarked message serializes the flag: %s", plain)
	}
	marked, err := json.Marshal(Message{Role: RoleUser, JoinsToolMedia: true})
	if err != nil {
		t.Fatal(err)
	}
	var back Message
	if err := json.Unmarshal(marked, &back); err != nil || !back.JoinsToolMedia {
		t.Fatalf("a marked message does not round-trip: %s (%v)", marked, err)
	}
}
