//go:build !windows && cgo

package extproc

import (
	"encoding/json"
	"fmt"
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

// The joint read-back records what the two read-backs recorded, including
// for a body where one group's member has the wrong type: that group records
// nothing, and the other still records its own.
func TestJointReadBackRecordsWhatTheTwoReadBacksRecorded(t *testing.T) {
	for _, body := range []string{
		`{"reasoning_effort":"high","provider":{"order":["deepinfra"],"zdr":true,"data_collection":"deny","allow_fallbacks":false}}`,
		`{"reasoning_effort":"high","provider":{"order":123}}`,
		`{"reasoning":{"max_tokens":"1024"},"provider":{"order":["deepinfra"],"zdr":true}}`,
		`{"output_config":{"effort":"medium"},"reasoning":{"max_tokens":2048}}`,
		`not json`,
	} {
		separate := &RequestContext{DispatchedReasoningEffort: "stale", DispatchedProviderOrder: []string{"stale"}}
		recordDispatchedReasoningControls(separate, []byte(body))
		recordDispatchedProviderPin(separate, []byte(body))
		joint := &RequestContext{DispatchedReasoningEffort: "stale", DispatchedProviderOrder: []string{"stale"}}
		recordDispatchedControls(joint, []byte(body))
		if got, want := dispatchedRecord(joint), dispatchedRecord(separate); got != want {
			t.Fatalf("%s: joint %s, separate %s", body, got, want)
		}
	}
}

func dispatchedRecord(ctx *RequestContext) string {
	deref := func(value any) string {
		switch typed := value.(type) {
		case *int64:
			if typed != nil {
				return fmt.Sprint(*typed)
			}
		case *bool:
			if typed != nil {
				return fmt.Sprint(*typed)
			}
		}
		return "nil"
	}
	return fmt.Sprintf("effort=%q bound=%s order=%v data=%q zdr=%s fallbacks=%s",
		ctx.DispatchedReasoningEffort, deref(ctx.DispatchedReasoningBound), ctx.DispatchedProviderOrder,
		ctx.DispatchedProviderPrivacy.dataCollection, deref(ctx.DispatchedProviderPrivacy.zdr),
		deref(ctx.DispatchedProviderPrivacy.allowFallbacks))
}
