package raylinearc

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"slices"
	"testing"
)

// endpointPickFixture is vendored verbatim from memex-desktop
// firebase/shared/tests/usage/fixtures/endpoint-pick-hrw.json at commit
// 2cabd83b89 (PR #7643), the order the gateway's pickEndpoint produces. The
// digest pins the copy: a refreshed fixture changes it here on purpose.
const (
	endpointPickFixture       = "testdata/endpoint_pick/endpoint-pick-hrw.json"
	endpointPickFixtureSHA256 = "4874db7f0c798ba48d9d8948106c0b4f84ae3435655c92d8b1a0d7f427819b5e"
)

// The cell orders every fixture case exactly as the gateway does, so a
// conversation lands on the same endpoint whichever side picks.
func TestPickEndpointMatchesTheGatewayFixture(t *testing.T) {
	raw, err := os.ReadFile(endpointPickFixture)
	if err != nil {
		t.Fatal(err)
	}
	if sum := sha256.Sum256(raw); hex.EncodeToString(sum[:]) != endpointPickFixtureSHA256 {
		t.Fatalf("%s is not the vendored copy: sha256 %x", endpointPickFixture, sum)
	}
	var fixture struct {
		Cases []struct {
			Key   string   `json:"key"`
			Tags  []string `json:"tags"`
			Order []string `json:"order"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) != 35 {
		t.Fatalf("fixture has %d cases, want 35", len(fixture.Cases))
	}
	for index, c := range fixture.Cases {
		if got := PickEndpoint(c.Key, c.Tags); !slices.Equal(got, c.Order) {
			t.Errorf("case %d (key %q, tags %v): order %v, want %v", index, c.Key, c.Tags, got, c.Order)
		}
	}
}

// The order depends on the set of tags, not on how the config lists them,
// and a repeated tag is one endpoint.
func TestPickEndpointIgnoresListOrderAndRepeats(t *testing.T) {
	key := SessionKeyDigest("uid-1", "conv-1")
	want := PickEndpoint(key, []string{"anthropic", "amazon-bedrock", "google-vertex/global"})
	got := PickEndpoint(key, []string{"google-vertex/global", "anthropic", "amazon-bedrock", "anthropic"})
	if !slices.Equal(got, want) {
		t.Fatalf("order %v, want %v", got, want)
	}
	if len(PickEndpoint(key, nil)) != 0 {
		t.Fatal("no tags must give no order")
	}
}

// The session key is the gateway's sessionKeyDigest: hex sha256 of
// uid:conversation. memex's test (endpointPick.test.ts) states the rule for
// uid-1 and conv-7 rather than a literal; this literal was computed with
// `shasum -a 256` over "uid-1:conv-7".
func TestSessionKeyDigestIsTheGatewaysDigest(t *testing.T) {
	const want = "eda4777e0ee8f325dde334879e8f9f4125fe99aef897a8eb180134b039f14963"
	if got := SessionKeyDigest("uid-1", "conv-7"); got != want {
		t.Fatalf("session key %q, want %q", got, want)
	}
}

// slugReachFixture is vendored verbatim from memex-desktop
// firebase/shared/tests/usage/fixtures/openrouter-slug-reach.json at commit
// d6220f7c29 (branch claude/optc-m5-arc-mask): the slug rule memex pinReach
// and VSR slugMatches share.
const (
	slugReachFixture       = "testdata/endpoint_pick/openrouter-slug-reach.json"
	slugReachFixtureSHA256 = "d55b592f47933870ba4c33b16472496df7d5e94f46e01715036b4a1381c14769"
)

// Every slug the gateway's fixture names reaches the same endpoints here, in
// the endpoints' order.
func TestPinRangeReachesWhatTheGatewayFixtureReaches(t *testing.T) {
	raw, err := os.ReadFile(slugReachFixture)
	if err != nil {
		t.Fatal(err)
	}
	if sum := sha256.Sum256(raw); hex.EncodeToString(sum[:]) != slugReachFixtureSHA256 {
		t.Fatalf("%s is not the vendored copy: sha256 %x", slugReachFixture, sum)
	}
	var fixture struct {
		Endpoints []struct {
			Tag string `json:"tag"`
		} `json:"endpoints"`
		Cases []struct {
			Name  string   `json:"name"`
			Slugs []string `json:"slugs"`
			Reach []string `json:"reach"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	tags := make([]string, len(fixture.Endpoints))
	for index, endpoint := range fixture.Endpoints {
		tags[index] = endpoint.Tag
	}
	if len(fixture.Cases) != 11 {
		t.Fatalf("fixture has %d cases, want 11", len(fixture.Cases))
	}
	for _, c := range fixture.Cases {
		if got := PinRange(c.Slugs, nil, nil, tags); !slices.Equal(got, c.Reach) {
			t.Errorf("%s: %v reaches %v, want %v", c.Name, c.Slugs, got, c.Reach)
		}
	}
}

// only and order both narrow; ignore removes what it reaches, children
// included.
func TestPinRangeIntersectsOnlyAndOrderLessIgnore(t *testing.T) {
	tags := []string{"anthropic", "google-vertex", "google-vertex/global", "xai", "xai/zdr", "xai/zdr/us"}
	got := PinRange([]string{"Google-Vertex", "xai"}, []string{"google-vertex/global", "xai/zdr"}, []string{"xai/zdr/us"}, tags)
	if want := []string{"google-vertex/global", "xai/zdr"}; !slices.Equal(got, want) {
		t.Fatalf("reach %v, want %v", got, want)
	}
	if got := PinRange(nil, nil, []string{"xai"}, tags); !slices.Equal(got, tags[:3]) {
		t.Fatalf("ignore xai left %v", got)
	}
}
