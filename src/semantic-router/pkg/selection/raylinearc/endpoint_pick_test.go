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

// The session key is the gateway's sessionKeyDigest: hex sha256 of uid:conversation.
func TestSessionKeyDigestIsTheGatewaysDigest(t *testing.T) {
	sum := sha256.Sum256([]byte("uid-1:conv-1"))
	if got := SessionKeyDigest("uid-1", "conv-1"); got != hex.EncodeToString(sum[:]) || len(got) != 64 {
		t.Fatalf("session key %q", got)
	}
}
