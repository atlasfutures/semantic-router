package raylinearc

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"
)

var cacheContextNow = time.Date(2026, 10, 16, 12, 0, 0, 0, time.UTC)

func actionIDFor(arm int) string {
	return fmt.Sprintf("%064x", arm+1)
}

func intPtr(value int) *int { return &value }

// identityArms binds actionIDFor(arm) to arm.
func identityArms(actionID string) (int, bool) {
	for arm := 0; arm < 64; arm++ {
		if actionIDFor(arm) == actionID {
			return arm, true
		}
	}
	return 0, false
}

func servedWarmth(arm int, ago time.Duration, call CallCache) *WorkerWarmth {
	call.ActionID = actionIDFor(arm)
	return &WorkerWarmth{LastUsed: cacheContextNow.Add(-ago), LastInputTokens: 900, Call: &call}
}

func encodeCacheContext(t *testing.T, context *PolicyCacheContext) map[string]interface{} {
	t.Helper()
	encoded, err := json.Marshal(context)
	if err != nil {
		t.Fatal(err)
	}
	if reason := validCacheContext(encoded); reason != "" {
		t.Fatalf("service would refuse %s: %s", encoded, reason)
	}
	var decoded map[string]interface{}
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	return decoded
}

// validCacheContext ports the policy service's check of cache_context
// (pathfinder deploy/arc-policy-service/service.py _valid_cache_context at
// 50ca637f52): "" when valid, otherwise why it is refused.
func validCacheContext(body []byte) string {
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	var value interface{}
	if err := decoder.Decode(&value); err != nil {
		return "json"
	}
	if value == nil {
		return ""
	}
	object, ok := value.(map[string]interface{})
	if !ok {
		return "not an object"
	}
	for key := range object {
		if key != "actions" {
			return "unknown key " + key
		}
	}
	entries, ok := object["actions"].([]interface{})
	if object["actions"] != nil && !ok {
		return "actions not a list"
	}
	actionPattern := regexp.MustCompile(`^[a-f0-9]{64}$`)
	allowed := map[string]bool{"action_id": true, "gap_s": true, "ttl_s": true, "ttl_source": true, "context_epoch": true,
		"prompt_tokens": true, "cache_read_tokens": true, "cache_write_tokens": true}
	seen := map[string]bool{}
	for _, raw := range entries {
		entry, ok := raw.(map[string]interface{})
		if !ok {
			return "entry not an object"
		}
		for key := range entry {
			if !allowed[key] {
				return "unknown entry key " + key
			}
		}
		if epoch, present := entry["context_epoch"]; present && epoch != nil {
			if text, ok := epoch.(string); !ok || text == "" {
				return "context_epoch"
			}
		}
		action, ok := entry["action_id"].(string)
		if !ok || !actionPattern.MatchString(action) {
			return "malformed action id"
		}
		counts := map[string]*int64{}
		for _, name := range []string{"prompt_tokens", "cache_read_tokens", "cache_write_tokens"} {
			if entry[name] == nil {
				continue
			}
			number, ok := entry[name].(json.Number)
			if !ok || strings.ContainsAny(number.String(), ".eE") {
				return name + " not an int"
			}
			parsed, err := number.Int64()
			if err != nil || parsed < 0 {
				return name + " negative"
			}
			counts[name] = &parsed
		}
		for _, name := range []string{"gap_s", "ttl_s"} {
			if entry[name] == nil {
				continue
			}
			number, ok := entry[name].(json.Number)
			if !ok {
				return name + " not a number"
			}
			parsed, err := number.Float64()
			if err != nil || math.IsInf(parsed, 0) || math.IsNaN(parsed) || parsed < 0 || name == "ttl_s" && parsed == 0 {
				return name + " out of range"
			}
		}
		source, sourcePresent := entry["ttl_source"], entry["ttl_source"] != nil
		if (entry["ttl_s"] == nil) == sourcePresent {
			return "ttl without its source"
		}
		if sourcePresent {
			if text, ok := source.(string); !ok || text == "" {
				return "ttl_source"
			}
		}
		if prompt := counts["prompt_tokens"]; prompt != nil {
			var legs int64
			for _, name := range []string{"cache_read_tokens", "cache_write_tokens"} {
				if leg := counts[name]; leg != nil {
					legs += *leg
				}
			}
			if legs > *prompt {
				return "cache legs exceed the prompt"
			}
		}
		if seen[action] {
			return "repeated action"
		}
		seen[action] = true
	}
	return ""
}

func TestCacheContextReportsTheLastCallOfEachRecentArm(t *testing.T) {
	warmth := []*WorkerWarmth{
		servedWarmth(0, 42500*time.Millisecond, CallCache{
			TTLSeconds: 300, TTLSource: wireTTLSourceEphemeral5m,
			PromptTokens: intPtr(1800), CacheReadTokens: intPtr(1500), CacheWriteTokens: intPtr(250),
		}),
		servedWarmth(1, 900*time.Second, CallCache{
			TTLSeconds: 3600, TTLSource: wireTTLSourceEphemeral1h,
			PromptTokens: intPtr(1200), CacheReadTokens: intPtr(1100), CacheWriteTokens: intPtr(0),
		}),
		nil,
	}
	context := BuildPolicyCacheContext(warmth, cacheContextNow, identityArms)
	encoded, err := json.Marshal(context)
	if err != nil {
		t.Fatal(err)
	}
	want := fmt.Sprintf(`{"actions":[`+
		`{"action_id":%q,"gap_s":42.5,"ttl_s":300,"ttl_source":"anthropic cache_control ephemeral 5m","prompt_tokens":1800,"cache_read_tokens":1500,"cache_write_tokens":250},`+
		`{"action_id":%q,"gap_s":900,"ttl_s":3600,"ttl_source":"anthropic cache_control ephemeral 1h","prompt_tokens":1200,"cache_read_tokens":1100,"cache_write_tokens":0}]}`,
		actionIDFor(0), actionIDFor(1))
	if string(encoded) != want {
		t.Fatalf("cache_context\n got %s\nwant %s", encoded, want)
	}
	encodeCacheContext(t, context)
}

func TestCacheContextUnknownIsNullNeverZero(t *testing.T) {
	cases := map[string]struct {
		call CallCache
		want map[string]interface{}
	}{
		// OpenAI's automatic caching states no TTL on the wire.
		"unstated ttl": {
			call: CallCache{PromptTokens: intPtr(500), CacheReadTokens: intPtr(400), CacheWriteTokens: nil},
			want: map[string]interface{}{"ttl_s": nil, "ttl_source": nil, "prompt_tokens": 500.0, "cache_read_tokens": 400.0, "cache_write_tokens": nil},
		},
		"ttl without its source": {
			call: CallCache{TTLSeconds: 300},
			want: map[string]interface{}{"ttl_s": nil, "ttl_source": nil},
		},
		"source without its ttl": {
			call: CallCache{TTLSource: wireTTLSourceEphemeral5m},
			want: map[string]interface{}{"ttl_s": nil, "ttl_source": nil},
		},
		"no usage reported": {
			call: CallCache{TTLSeconds: 300, TTLSource: wireTTLSourceEphemeral5m},
			want: map[string]interface{}{"ttl_s": 300.0, "prompt_tokens": nil, "cache_read_tokens": nil, "cache_write_tokens": nil},
		},
		"negative count": {
			call: CallCache{PromptTokens: intPtr(-1), CacheReadTokens: intPtr(-5)},
			want: map[string]interface{}{"prompt_tokens": nil, "cache_read_tokens": nil},
		},
		"legs exceed the prompt": {
			call: CallCache{PromptTokens: intPtr(100), CacheReadTokens: intPtr(80), CacheWriteTokens: intPtr(30)},
			want: map[string]interface{}{"prompt_tokens": 100.0, "cache_read_tokens": nil, "cache_write_tokens": nil},
		},
		"one leg exceeds the prompt": {
			call: CallCache{PromptTokens: intPtr(100), CacheWriteTokens: intPtr(101)},
			want: map[string]interface{}{"prompt_tokens": 100.0, "cache_read_tokens": nil, "cache_write_tokens": nil},
		},
		"legs without a prompt": {
			call: CallCache{CacheReadTokens: intPtr(80), CacheWriteTokens: intPtr(30)},
			want: map[string]interface{}{"prompt_tokens": nil, "cache_read_tokens": 80.0, "cache_write_tokens": 30.0},
		},
	}
	for name, test := range cases {
		t.Run(name, func(t *testing.T) {
			context := BuildPolicyCacheContext([]*WorkerWarmth{servedWarmth(0, time.Minute, test.call)}, cacheContextNow, identityArms)
			decoded := encodeCacheContext(t, context)
			entry := decoded["actions"].([]interface{})[0].(map[string]interface{})
			if len(entry) != 7 {
				t.Fatalf("entry has %d members, want every contract member: %v", len(entry), entry)
			}
			for key, want := range test.want {
				if got, present := entry[key]; !present || got != want {
					t.Fatalf("%s = %v (present %v), want %v", key, got, present, want)
				}
			}
		})
	}
}

func TestCacheContextGapFromTheFutureIsUnknown(t *testing.T) {
	warmth := []*WorkerWarmth{servedWarmth(0, -2*time.Second, CallCache{})}
	decoded := encodeCacheContext(t, BuildPolicyCacheContext(warmth, cacheContextNow, identityArms))
	entry := decoded["actions"].([]interface{})[0].(map[string]interface{})
	if entry["gap_s"] != nil {
		t.Fatalf("gap_s = %v, want null", entry["gap_s"])
	}
}

func TestCacheContextKeepsTheLastHourOnly(t *testing.T) {
	warmth := []*WorkerWarmth{
		servedWarmth(0, time.Hour, CallCache{}),
		servedWarmth(1, time.Hour+time.Millisecond, CallCache{}),
		servedWarmth(2, 0, CallCache{}),
	}
	context := BuildPolicyCacheContext(warmth, cacheContextNow, identityArms)
	got := []string{}
	for _, entry := range context.Actions {
		got = append(got, entry.ActionID)
	}
	want := []string{actionIDFor(2), actionIDFor(0)}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("actions %v, want %v", got, want)
	}
	encodeCacheContext(t, context)
}

func TestCacheContextIsCappedMostRecentFirst(t *testing.T) {
	warmth := make([]*WorkerWarmth, 20)
	order := rand.New(rand.NewSource(7)).Perm(20)
	for arm := range warmth {
		// Arm a was served order[a] minutes ago.
		warmth[arm] = servedWarmth(arm, time.Duration(order[arm])*time.Minute, CallCache{})
	}
	context := BuildPolicyCacheContext(warmth, cacheContextNow, identityArms)
	if len(context.Actions) != MaxCacheContextActions {
		t.Fatalf("%d actions, want %d", len(context.Actions), MaxCacheContextActions)
	}
	for index, entry := range context.Actions {
		if *entry.GapS != float64(index*60) {
			t.Fatalf("entry %d gap %v, want %v", index, *entry.GapS, index*60)
		}
	}
	encodeCacheContext(t, context)
}

func TestCacheContextLeavesOutWhatItCannotName(t *testing.T) {
	malformed := servedWarmth(2, time.Minute, CallCache{})
	malformed.Call.ActionID = "A" + actionIDFor(2)[1:]
	otherArm := servedWarmth(3, time.Minute, CallCache{})
	otherArm.Call.ActionID = actionIDFor(0) // bound to arm 0, recorded on arm 3
	unbound := servedWarmth(4, time.Minute, CallCache{})
	unbound.Call.ActionID = strings.Repeat("f", 64)
	warmth := []*WorkerWarmth{
		// A record from before the call was kept: no action, no entry.
		{LastUsed: cacheContextNow.Add(-time.Minute), LastInputTokens: 10},
		{LastUsed: time.Time{}, Call: &CallCache{ActionID: actionIDFor(1)}},
		malformed, otherArm, unbound,
		servedWarmth(5, time.Minute, CallCache{}),
	}
	context := BuildPolicyCacheContext(warmth, cacheContextNow, identityArms)
	if len(context.Actions) != 1 || context.Actions[0].ActionID != actionIDFor(5) {
		t.Fatalf("actions %+v, want only arm 5's", context.Actions)
	}
	if empty := BuildPolicyCacheContext(nil, cacheContextNow, identityArms); empty == nil || len(empty.Actions) != 0 {
		t.Fatalf("no warmth = %+v, want an empty list", empty)
	}
	encoded, _ := json.Marshal(BuildPolicyCacheContext(nil, cacheContextNow, identityArms))
	if string(encoded) != `{"actions":[]}` {
		t.Fatalf("empty cache_context = %s", encoded)
	}
}

// Whatever the warmth holds, the service accepts the cache_context built
// from it.
func TestCacheContextIsAlwaysValid(t *testing.T) {
	random := rand.New(rand.NewSource(44))
	count := func() *int {
		switch random.Intn(4) {
		case 0:
			return nil
		case 1:
			return intPtr(-random.Intn(100))
		default:
			return intPtr(random.Intn(5000))
		}
	}
	for trial := 0; trial < 2000; trial++ {
		warmth := make([]*WorkerWarmth, random.Intn(24))
		for arm := range warmth {
			if random.Intn(5) == 0 {
				continue
			}
			ago := time.Duration(random.Int63n(int64(2*time.Hour))) - time.Minute
			value := servedWarmth(arm, ago, CallCache{
				TTLSeconds:   []int{0, 300, 3600, -5}[random.Intn(4)],
				TTLSource:    []string{"", wireTTLSourceEphemeral5m}[random.Intn(2)],
				PromptTokens: count(), CacheReadTokens: count(), CacheWriteTokens: count(),
			})
			if random.Intn(6) == 0 {
				value.Call.ActionID = actionIDFor(random.Intn(len(warmth)))
			}
			warmth[arm] = value
		}
		encodeCacheContext(t, BuildPolicyCacheContext(warmth, cacheContextNow, identityArms))
	}
}

func TestWireCacheTTLReadsWhatTheBodyStates(t *testing.T) {
	cases := map[string]struct {
		body    string
		seconds int
		source  string
	}{
		"none":            {`{"messages":[{"role":"user","content":"hi"}]}`, 0, ""},
		"ephemeral":       {`{"system":[{"type":"text","text":"s","cache_control":{"type":"ephemeral"}}],"messages":[]}`, 300, wireTTLSourceEphemeral5m},
		"explicit 5m":     {`{"messages":[{"role":"user","content":[{"type":"text","text":"x","cache_control":{"type":"ephemeral","ttl":"5m"}}]}]}`, 300, wireTTLSourceEphemeral5m},
		"1h":              {`{"tools":[{"name":"Bash","cache_control":{"type":"ephemeral","ttl":"1h"}}],"messages":[]}`, 3600, wireTTLSourceEphemeral1h},
		"top level 1h":    {`{"cache_control":{"type":"ephemeral","ttl":"1h"},"messages":[]}`, 3600, wireTTLSourceEphemeral1h},
		"1h then 5m":      {`{"system":[{"type":"text","text":"s","cache_control":{"type":"ephemeral","ttl":"1h"}}],"messages":[{"role":"user","content":[{"type":"text","text":"x","cache_control":{"type":"ephemeral"}}]}]}`, 300, wireTTLSourceEphemeral5m},
		"chat parts":      {`{"messages":[{"role":"user","content":[{"type":"text","text":"x","cache_control":{"type":"ephemeral","ttl":"1h"}}]}]}`, 3600, wireTTLSourceEphemeral1h},
		"unknown ttl":     {`{"messages":[{"role":"user","content":[{"type":"text","text":"x","cache_control":{"type":"ephemeral","ttl":"24h"}}]}]}`, 0, ""},
		"unknown type":    {`{"messages":[{"role":"user","content":[{"type":"text","text":"x","cache_control":{"type":"persistent"}}]}]}`, 0, ""},
		"in a tool input": {`{"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"t","name":"x","input":{"cache_control":{"type":"ephemeral","ttl":"1h"}}}]}]}`, 0, ""},
		"in text":         {`{"messages":[{"role":"user","content":"\"cache_control\": {\"type\":\"ephemeral\"}"}]}`, 0, ""},
		"not json":        {`{"cache_control":`, 0, ""},
	}
	for name, test := range cases {
		t.Run(name, func(t *testing.T) {
			seconds, source := WireCacheTTL([]byte(test.body))
			if seconds != test.seconds || source != test.source {
				t.Fatalf("WireCacheTTL = %d %q, want %d %q", seconds, source, test.seconds, test.source)
			}
		})
	}
}

// The call evidence survives the episode record, and a record from before it
// (no call member) reads as unknown.
func TestEpisodeRecordKeepsTheCallEvidence(t *testing.T) {
	state, err := NewEpisodeState(3)
	if err != nil {
		t.Fatal(err)
	}
	if err := state.Commit(0, 900, cacheContextNow.Add(-time.Minute)); err != nil {
		t.Fatal(err)
	}
	state.Warmth[0].Call = &CallCache{
		ActionID: actionIDFor(0), TTLSeconds: 3600, TTLSource: wireTTLSourceEphemeral1h,
		PromptTokens: intPtr(1800), CacheReadTokens: intPtr(0), CacheWriteTokens: nil,
	}
	if err := state.Commit(1, 500, cacheContextNow.Add(-time.Second)); err != nil {
		t.Fatal(err)
	}
	state.Warmth[1].Call = &CallCache{ActionID: actionIDFor(1)}
	payload, err := marshalEpisodeState(state, 3, cacheContextNow)
	if err != nil {
		t.Fatal(err)
	}
	restored, _, err := unmarshalEpisodeState(payload, 3, cacheContextNow)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := json.Marshal(BuildPolicyCacheContext(restored.Warmth, cacheContextNow, identityArms))
	want, _ := json.Marshal(BuildPolicyCacheContext(state.Warmth, cacheContextNow, identityArms))
	if string(got) != string(want) {
		t.Fatalf("restored\n got %s\nwant %s", got, want)
	}
	if !strings.Contains(string(got), `"cache_read_tokens":0,"cache_write_tokens":null`) {
		t.Fatalf("a reported zero must stay zero and an unreported leg null: %s", got)
	}
	if !strings.Contains(string(got), `"ttl_s":null,"ttl_source":null,"prompt_tokens":null`) {
		t.Fatalf("arm 1's unknowns must be null: %s", got)
	}

	// The same record as a router before this change wrote it.
	var old map[string]interface{}
	if err := json.Unmarshal(payload, &old); err != nil {
		t.Fatal(err)
	}
	for _, warmth := range old["warmth"].([]interface{}) {
		if warmth != nil {
			delete(warmth.(map[string]interface{}), "call")
		}
	}
	oldPayload, _ := json.Marshal(old)
	oldState, _, err := unmarshalEpisodeState(oldPayload, 3, cacheContextNow)
	if err != nil {
		t.Fatal(err)
	}
	if oldState.Warmth[0] == nil || oldState.Warmth[0].Call != nil {
		t.Fatalf("an old record's warmth = %+v, want no call", oldState.Warmth[0])
	}
	if context := BuildPolicyCacheContext(oldState.Warmth, cacheContextNow, identityArms); len(context.Actions) != 0 {
		t.Fatalf("an old record names no action, so it reports none: %+v", context.Actions)
	}
}

// Pathfinder's cache_context fixture decodes strictly into the router's
// request, and the decide body writes it back as it is.
func TestPathfinderGoldenCacheContextEncodesAsTheDecideBody(t *testing.T) {
	body, err := os.ReadFile(filepath.Join("testdata", "cache_context", "pathfinder_decision_request_cache_context.v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	var request PolicyDecisionRequest
	if err := decodeStrict(body, &request); err != nil {
		t.Fatal(err)
	}
	if request.CacheContext == nil || len(request.CacheContext.Actions) != 2 {
		t.Fatalf("decoded cache_context %+v", request.CacheContext)
	}
	encoded, err := EncodePolicyDecisionRequest(request)
	if err != nil {
		t.Fatal(err)
	}
	requireSameJSON(t, "golden decide body", body, encoded)
	var fixture map[string]json.RawMessage
	if err := json.Unmarshal(body, &fixture); err != nil {
		t.Fatal(err)
	}
	if reason := validCacheContext(fixture["cache_context"]); reason != "" {
		t.Fatalf("the ported validator refuses the fixture: %s", reason)
	}
}

// A request without cache_context leaves it out, as before the field.
func TestPolicyDecisionRequestOmitsAbsentCacheContext(t *testing.T) {
	encoded, err := EncodePolicyDecisionRequest(PolicyDecisionRequest{SchemaVersion: PolicyDecisionRequestSchema})
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(encoded, []byte("cache_context")) {
		t.Fatalf("absent cache_context was written: %s", encoded)
	}
}
