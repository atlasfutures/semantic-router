package raylinearc

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strconv"
	"testing"
)

// referenceMessagesDigest is MessagesDigest as it was before the memo: every
// call canonicalizes every message it hashes.
func referenceMessagesDigest(messages []json.RawMessage, n int) string {
	hash := sha256.New()
	for _, message := range messages[:n] {
		canonical := canonicalPrefixMessage(message)
		hash.Write([]byte(strconv.Itoa(len(canonical))))
		hash.Write([]byte{':'})
		hash.Write(canonical)
	}
	return hex.EncodeToString(hash.Sum(nil))
}

// One request's digests, asked in any order and any number of times, are the
// digests an unmemoized hash gives: the memo changes what is computed, never
// what is recorded or compared.
func TestPrefixDigestsMatchTheUnmemoizedDigest(t *testing.T) {
	messages := []json.RawMessage{
		json.RawMessage(`{"role":"user","content":[{"type":"text","text":"a <b> & c","cache_control":{"type":"ephemeral"}}]}`),
		json.RawMessage(`{ "content" : "Looking.", "role":"assistant" }`),
		json.RawMessage(`{"role":"system","content":[{"type":"text","text":"mid-conversation"}]}`),
		json.RawMessage(`not json`),
		json.RawMessage(`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBOR"}}]}]}`),
	}
	digests := NewPrefixDigests(messages)
	for _, n := range []int{5, 0, 2, 5, 3, 1, 4, 2} {
		if got, want := digests.Digest(n), referenceMessagesDigest(messages, n); got != want {
			t.Fatalf("Digest(%d) = %s, want %s", n, got, want)
		}
		if got, want := MessagesDigest(messages, n), referenceMessagesDigest(messages, n); got != want {
			t.Fatalf("MessagesDigest(%d) = %s, want %s", n, got, want)
		}
	}
	for index, message := range messages {
		if got, want := string(digests.Canonical(index)), string(canonicalPrefixMessage(message)); got != want {
			t.Fatalf("Canonical(%d) = %s, want %s", index, got, want)
		}
	}
	shape := NewPolicyPrefixShapeFrom(NewPrefixDigests(messages), json.RawMessage(`"system"`), nil)
	fresh := NewPolicyPrefixShape(messages, json.RawMessage(`"system"`), nil)
	if shape.Digest != fresh.Digest || shape.Len != fresh.Len || shape.System != fresh.System || shape.Tools != fresh.Tools {
		t.Fatalf("shape = %+v, want %+v", shape, fresh)
	}
	for index := range fresh.Messages {
		if shape.Messages[index] != fresh.Messages[index] {
			t.Fatalf("shape message %d = %d, want %d", index, shape.Messages[index], fresh.Messages[index])
		}
	}
}
