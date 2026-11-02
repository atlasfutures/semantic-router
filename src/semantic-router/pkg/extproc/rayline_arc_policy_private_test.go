package extproc

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// releaseInline runs every private release on the turn's own goroutine, so
// a test reads what it sent as soon as the turn returns.
func releaseInline(t *testing.T) {
	t.Helper()
	previous := runPolicyPrivateRelease
	runPolicyPrivateRelease = func(release func()) { release() }
	t.Cleanup(func() { runPolicyPrivateRelease = previous })
}

func decodeReleases(t *testing.T, bodies [][]byte) []raylinearc.PolicyReleaseRequest {
	t.Helper()
	releases := make([]raylinearc.PolicyReleaseRequest, 0, len(bodies))
	for _, body := range bodies {
		var release raylinearc.PolicyReleaseRequest
		if err := json.Unmarshal(body, &release); err != nil {
			t.Fatal(err)
		}
		releases = append(releases, release)
	}
	return releases
}

// On a delta cell a private turn is never staged and its reply is never
// appended: every private turn goes out in full, the service holds no
// staged turn for the episode, and each turn ends with a release that names
// the episode and the package and carries no content.
func TestPrivateTurnsNeverStageOrAppend(t *testing.T) {
	releaseInline(t)
	fixture := newDeltaFixture(t, config.RaylineARCDecideModeDelta, nil)
	fixture.private = true
	fixture.fake.servesRelease = true

	first, ctx := fixture.turn(t, policyTestRequest(t, deltaUser))
	if first.RaylineARC.PolicyDecisionMode != "" || first.RaylineARC.PolicyStaging != nil {
		t.Fatalf("private turn 1 staged: mode %q staging %v", first.RaylineARC.PolicyDecisionMode, first.RaylineARC.PolicyStaging)
	}
	complete(t, ctx, deltaReply(t))
	second, ctx := fixture.turn(t, policyTestRequest(t, deltaUser, deltaAssistant, deltaToolResult))
	if second.RaylineARC.PolicyDecisionMode != "" || second.RaylineARC.PolicyStaging != nil {
		t.Fatalf("private turn 2 staged: mode %q", second.RaylineARC.PolicyDecisionMode)
	}
	complete(t, ctx, deltaReply(t))

	if appends := fixture.fake.receivedAppends(); len(appends) != 0 {
		t.Fatalf("a private reply was appended: %+v", appends)
	}
	sent := fixture.fake.received()
	if len(sent) != 2 {
		t.Fatalf("decides = %d, want 2", len(sent))
	}
	for index, decide := range sent {
		if decide.Stage != nil || decide.TurnDelta != nil {
			t.Fatalf("private decide %d staged or sent a delta: %+v", index, decide)
		}
	}
	if body := fixture.fake.receivedBodies()[1]; !bytes.Contains(body, []byte("list the files")) {
		t.Fatal("private turn 2 was not sent in full")
	}
	if staged := fixture.fake.stagedEpisodes(); staged != 0 {
		t.Fatalf("the service holds %d staged episodes after private turns", staged)
	}
	bodies := fixture.fake.receivedReleases()
	releases := decodeReleases(t, bodies)
	if len(releases) != 2 {
		t.Fatalf("releases = %d, want one per private turn", len(releases))
	}
	for index, release := range releases {
		if release.SchemaVersion != raylinearc.PolicyReleaseRequestSchema || release.EpisodeIDHash != fixture.episode ||
			release.Package.Alias != policyTestAlias || release.Package.PackageSHA256 != policyTestPackage {
			t.Fatalf("release %d = %+v", index, release)
		}
		if bytes.Contains(bodies[index], []byte("list the files")) || bytes.Contains(bodies[index], []byte("Listing them")) {
			t.Fatalf("release %d carried content: %s", index, bodies[index])
		}
	}
}

// A standard turn on the same cell still stages, appends and releases
// nothing: the release is the private turn's alone.
func TestStandardTurnsSendNoRelease(t *testing.T) {
	releaseInline(t)
	fixture := newDeltaFixture(t, config.RaylineARCDecideModeDelta, nil)
	fixture.fake.servesRelease = true
	fixture.stagedFirstTurn(t)
	if len(fixture.fake.receivedAppends()) != 1 {
		t.Fatal("the standard turn was not appended")
	}
	if releases := fixture.fake.receivedReleases(); len(releases) != 0 {
		t.Fatalf("a standard turn sent %d releases", len(releases))
	}
}

// A service that predates the release answers it 404: the private turn is
// served all the same.
func TestPrivateTurnServedByAServiceWithoutRelease(t *testing.T) {
	releaseInline(t)
	fixture := newDeltaFixture(t, config.RaylineARCDecideModeDelta, nil)
	fixture.private = true
	result, ctx := fixture.turn(t, policyTestRequest(t, deltaUser))
	if result.RaylineARC.PolicyActionID != fixture.decided {
		t.Fatalf("served %q", result.RaylineARC.PolicyActionID)
	}
	complete(t, ctx, deltaReply(t))
	if len(fixture.fake.receivedReleases()) != 1 {
		t.Fatal("the release was not attempted")
	}
	if got := raylinearc.PolicyReleaseOutcomeOf(&raylinearc.PolicyServiceError{Class: raylinearc.PolicyReleaseUnsupportedClass}); got != raylinearc.PolicyReleaseUnsupported {
		t.Fatalf("a 404 release reads %q", got)
	}
}

// On a sessioned Server a private turn never decides on the episode's
// session: it starts one of its own, releases the episode on it and ends
// it, and records none. The episode's session is left as it was.
func TestPrivateTurnDecidesOnASessionOfItsOwn(t *testing.T) {
	releaseInline(t)
	fixture, front := newPolicySessionFixture(t)
	for _, container := range front.containers {
		container.servesRelease = true
	}
	state, _ := raylinearc.NewEpisodeState(2)
	body := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})
	first, err := fixture.selectOn(t, state, body)
	if err != nil {
		t.Fatal(err)
	}
	persisted := first.RaylineARC.PolicySession
	if persisted == nil {
		t.Fatal("the standard turn recorded no session")
	}
	state.PolicySession = persisted
	owner := front.containerOf(persisted.Token)

	private, err := fixture.selectOnWith(t, state, body, func(arcContext *selection.RaylineARCSelectionContext) {
		arcContext.PrivateTurn = true
	})
	if err != nil {
		t.Fatal(err)
	}
	if private.RaylineARC.PolicySession != nil {
		t.Fatalf("the private turn recorded a session: %+v", private.RaylineARC.PolicySession)
	}
	front.mu.Lock()
	starts, live := front.starts, len(front.sessions)
	front.mu.Unlock()
	if starts != 2 || live != 1 || front.containerOf(persisted.Token) != owner {
		t.Fatalf("starts %d, live sessions %d: the private turn's session was not its own or not ended", starts, live)
	}
	privateContainer := 2 % len(front.containers) // the fake places the n-th session on container n mod 2
	releases := decodeReleases(t, front.containers[privateContainer].receivedReleases())
	if len(releases) != 1 || releases[0].EpisodeIDHash != strings.Repeat("e", 64) {
		t.Fatalf("private container releases %+v", releases)
	}
	if other := front.containers[1-privateContainer].receivedReleases(); len(other) != 0 {
		t.Fatalf("a release reached the episode's own container: %d", len(other))
	}
}
