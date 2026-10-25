//go:build !windows && cgo

package extproc

import (
	"encoding/json"
	"testing"
)

// Setting session_id and provider in one rewrite writes the bytes the two
// one-member rewrites wrote: the encoding is canonical (sorted members,
// compact and HTML-escaped values), so re-encoding an encoded body changes
// nothing.
func TestOneRewriteOfTwoMembersIsTwoRewrites(t *testing.T) {
	bodies := []string{
		`{"model":"m","messages":[ {"role":"user","content":"a <b> & c   é"} ],"stream":true}`,
		`{"z":1,"a":{"nested":[1, 2,3]},"model":"m","model":"dup"}`,
		`{"model":"m","provider":{"order":["old"]},"session_id":"old"}`,
	}
	session := json.RawMessage(`"0123abcd"`)
	provider := json.RawMessage(`{"only":["together"],"order":["together"],"allow_fallbacks":false}`)
	for _, body := range bodies {
		first, err := setTopLevelMembers([]byte(body), map[string]json.RawMessage{upstreamSessionIDField: session})
		if err != nil {
			t.Fatal(err)
		}
		twice, err := setTopLevelMembers(first, map[string]json.RawMessage{providerPreferencesField: provider})
		if err != nil {
			t.Fatal(err)
		}
		once, err := setTopLevelMembers([]byte(body), map[string]json.RawMessage{
			upstreamSessionIDField: session, providerPreferencesField: provider,
		})
		if err != nil {
			t.Fatal(err)
		}
		if string(once) != string(twice) {
			t.Fatalf("one rewrite:\n%s\ntwo rewrites:\n%s", once, twice)
		}
	}
	if _, err := setTopLevelMembers([]byte(`[1]`), map[string]json.RawMessage{"a": json.RawMessage(`1`)}); err == nil {
		t.Fatal("a body that is not an object was rewritten")
	}
}
