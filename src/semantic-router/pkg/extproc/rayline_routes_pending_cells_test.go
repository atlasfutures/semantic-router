package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// withDelta makes the fixture's cell a delta one (decide_mode delta, the
// service staging turns), as deployed cells run.
func withDelta(fixture *pendingRoutesFixture) {
	policy := fixture.policy.decision.Algorithm.RaylineARC.PolicyService
	policy.DecideMode = config.RaylineARCDecideModeDelta
	scorer := newPolicyServiceScorer(&config.RouterConfig{}, fixture.policy.decision)
	scorer.armDelta([]string{raylinearc.PolicyCapabilityStagedTurns})
	fixture.policy.selector.arm(&raylineARCArmedComponents{
		scorer:    scorer,
		admission: raylinearc.NewAdmissionGate(0),
		policy: raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{
			BaseURL: fixture.policy.fake.URL(), TotalTimeout: 5 * time.Second,
		}),
	})
}

// On a delta cell a generation turn after a lookup goes out in full (the
// service's staged turn is not the conversation's, as the lookup dropped it)
// with the inferred turn attributed, commits both entries, and the next turn
// goes out as a delta on the turn it staged.
func TestDeltaCellGenerationTurnAfterALookup(t *testing.T) {
	fixture := newPendingRoutesFixture(t, true, "")
	withDelta(fixture)
	const conversation = "conversation-delta"
	fixture.picks = []string{fixture.actions[2]} // vendor/off
	fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))
	turn := fixture.generationTurn(t, conversation, pendingRoutesBody(t, pendingTurnTask, pendingTurnReply, pendingTurnNext))
	bodies := fixture.policy.fake.receivedBodies()
	sent := fixture.policy.fake.received()
	last := sent[len(sent)-1]
	if bytes.Contains(bodies[len(bodies)-1], []byte(`"turn_delta"`)) {
		t.Fatal("the turn after a lookup went out as a delta")
	}
	if len(last.Attribution) != 1 || last.Attribution[0].Message != 1 || last.Attribution[0].ActionID != fixture.actions[2] {
		t.Fatalf("attribution %+v, want the inferred turn", last.Attribution)
	}
	stageRaylineARCAppendSource(turn, []byte(`{"id":"m","type":"message","role":"assistant","content":[{"type":"text","text":"test added"}],"model":"x","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	if err := turn.RaylineARCTransaction.commit(context.Background(), turn); err != nil {
		t.Fatal(err)
	}
	state := fixture.stored(t, conversation)
	if state.TurnIndex != 2 || len(state.Policy.Ledger) != 2 || state.Policy.Ledger[0].Basis != raylinearc.PolicyLedgerBasisInferred ||
		state.Policy.Ledger[1].Basis != "" || state.Pending != nil {
		t.Fatalf("after the generation turn the record is %+v", state.Policy)
	}
}

// A generation turn after a lookup, through selection, staging and the
// transaction's own finish: committed (two entries, the served model of the
// turn it served), aborted (the inferred turn stays, staged as soon as it was
// settled), and refused while it still holds its lease (the exclusion is
// written with it).
func TestGenerationTurnAfterALookupFinishes(t *testing.T) {
	for _, finish := range []string{"commit", "abort", "refusal_with_lease"} {
		fixture := newPendingRoutesFixture(t, true, "")
		withFallback(fixture)
		conversation := "conversation-finish-" + finish
		fixture.picks = []string{fixture.actions[2], fixture.actions[0]} // lookup: off; generation: think
		fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))
		turn := fixture.generationTurn(t, conversation, pendingRoutesBody(t, pendingTurnTask, pendingTurnReply, pendingTurnNext))
		switch finish {
		case "commit":
			if err := turn.RaylineARCTransaction.commit(context.Background(), turn); err != nil {
				t.Fatal(err)
			}
		case "abort":
			if err := turn.RaylineARCTransaction.abort(context.Background(), "upstream_error"); err != nil {
				t.Fatal(err)
			}
		case "refusal_with_lease":
			declineRefusedTurn(turn)
			if err := turn.RaylineARCTransaction.abort(context.Background(), "refusal"); err != nil {
				t.Fatal(err)
			}
		}
		state := fixture.stored(t, conversation)
		inferred := state.Policy != nil && len(state.Policy.Ledger) >= 1 && state.Policy.Ledger[0].Basis == raylinearc.PolicyLedgerBasisInferred
		if !inferred || state.Pending != nil {
			t.Fatalf("%s: the inferred turn is not stored: %+v", finish, state.Policy)
		}
		served := fixture.servedRecord(t, conversation)
		switch finish {
		case "commit":
			if state.TurnIndex != 2 || len(state.Policy.Ledger) != 2 || served != "vendor/think" {
				t.Fatalf("commit: turn %d ledger %d served %q", state.TurnIndex, len(state.Policy.Ledger), served)
			}
		case "abort":
			if state.TurnIndex != 1 || served != "vendor/off" {
				t.Fatalf("abort: turn %d served %q, want the inferred turn only", state.TurnIndex, served)
			}
		case "refusal_with_lease":
			if state.TurnIndex != 1 || !state.Policy.Excludes("vendor/think") {
				t.Fatalf("refusal: turn %d exclusions %+v", state.TurnIndex, state.Policy.Exclusions)
			}
		}
	}
}

// sharedRedisStore is a Redis episode store under a key prefix of its own,
// as two instances of one cell share it; the test is skipped without a Redis.
func sharedRedisStore(t *testing.T) (*raylinearc.RedisEpisodeStore, *redis.Client, string) {
	t.Helper()
	address := os.Getenv("RAYLINE_ARC_TEST_REDIS_ADDR")
	if address == "" {
		t.Skip("RAYLINE_ARC_TEST_REDIS_ADDR is not set")
	}
	prefix := "vsr:pending-test:" + strings.ReplaceAll(t.Name(), "/", "_") + ":" + time.Now().Format("150405.000000") + ":"
	store, err := raylinearc.NewRedisEpisodeStore(raylinearc.RedisEpisodeStoreConfig{
		Address: address, KeyPrefix: prefix, LeaseTTL: 30 * time.Second, IdleTTL: time.Minute,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	client := redis.NewClient(&redis.Options{Addr: address})
	t.Cleanup(func() { _ = client.Close() })
	return store, client, prefix
}

// onStore points the fixture's cell at store, as a second instance of the
// same cell would be.
func (fixture *pendingRoutesFixture) onStore(store *raylinearc.RedisEpisodeStore) *pendingRoutesFixture {
	fixture.router.RaylineARCEpisodeStore = store
	fixture.store, fixture.records = nil, store
	return fixture
}

// Two instances of one cell on one Redis: a lookup on one, the next turn's
// lookup on the other, which settles the first's decision.
func TestTwoInstancesShareThePendingDecision(t *testing.T) {
	store, _, _ := sharedRedisStore(t)
	first := newPendingRoutesFixture(t, true, "").onStore(store)
	second := newPendingRoutesFixture(t, true, "").onStore(store)
	first.picks, second.picks = []string{first.actions[2]}, []string{second.actions[0]}
	const conversation = "conversation-two-instances"
	first.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))
	second.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask, pendingTurnReply, pendingTurnNext))
	state := second.stored(t, conversation)
	if state.TurnIndex != 1 || len(state.Policy.Ledger) != 1 || state.Policy.Ledger[0].ActionID != first.actions[2] ||
		state.Pending == nil || state.Pending.Policy.Entry.ActionID != second.actions[0] {
		t.Fatalf("the second instance did not settle the first's decision: %+v pending %+v", state.Policy, state.Pending)
	}
	if sent := second.policy.fake.received(); len(sent[0].Attribution) != 1 || sent[0].Attribution[0].ActionID != first.actions[2] {
		t.Fatalf("the second instance's decide attribution %+v", sent[0].Attribution)
	}
}

// Relaxed cells: two instances race lookups on one conversation. Every
// lookup answers; the record holds one of their decisions, intact.
func TestRelaxedInstancesRacingLookupsKeepTheRecordIntact(t *testing.T) {
	store, _, _ := sharedRedisStore(t)
	cells := []*pendingRoutesFixture{
		newPendingRoutesFixture(t, true, config.RaylineARCConsistencyRelaxed).onStore(store),
		newPendingRoutesFixture(t, true, config.RaylineARCConsistencyRelaxed).onStore(store),
	}
	const conversation = "conversation-relaxed-race"
	var wait sync.WaitGroup
	routes := make(chan string, 8)
	for index := range 8 {
		wait.Add(1)
		body := pendingRoutesBody(t, pendingTurnTask)
		go func() {
			defer wait.Done()
			response := cells[index%2].router.handleRaylineRoutesAPI(body, routesContext(map[string]string{
				"x-rayline-format": "anthropic", raylineRoutesSessionHeader: conversation,
			}))
			var payload raylineRoutesResponse
			if response.GetImmediateResponse().GetStatus().GetCode() == 200 &&
				json.Unmarshal(response.GetImmediateResponse().Body, &payload) == nil {
				routes <- payload.RouteID
				return
			}
			routes <- "failed: " + string(response.GetImmediateResponse().Body)
		}()
	}
	wait.Wait()
	for _, cell := range cells {
		cell.router.background.Wait()
	}
	close(routes)
	answered := map[string]bool{}
	for route := range routes {
		if strings.HasPrefix(route, "failed: ") {
			t.Fatalf("a racing lookup failed: %s", route)
		}
		answered[route] = true
	}
	state := cells[0].stored(t, conversation)
	if state == nil || state.Pending == nil || !answered[state.Pending.RouteID] || state.TurnIndex != 0 {
		t.Fatalf("after racing lookups the record is %+v", state)
	}
}

// A lookup whose lease is lost before its pending write fails, writes
// nothing, and leaves the record as it was.
func TestLookupThatLosesItsLeaseWritesNothing(t *testing.T) {
	store, client, prefix := sharedRedisStore(t)
	fixture := newPendingRoutesFixture(t, true, "").onStore(store)
	const conversation = "conversation-lease-lost"
	fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))
	before := fixture.stored(t, conversation)
	hash := raylinearc.HashEpisodeID(conversation)
	fixture.policy.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string {
		// Another owner takes the episode while this lookup decides.
		client.Del(context.Background(), prefix+hash+":lease")
		client.Set(context.Background(), prefix+hash+":lease", "someone-else", time.Minute)
		return fixture.actions[2]
	})
	headers := map[string]string{"x-rayline-format": "anthropic", raylineRoutesSessionHeader: conversation}
	response := fixture.router.handleRaylineRoutesAPI(pendingRoutesBody(t, pendingTurnTask), routesContext(headers))
	fixture.router.background.Wait()
	if code := immediateStatusCode(t, response); code == 200 {
		t.Fatal("a lookup that lost its lease answered 200")
	}
	client.Del(context.Background(), prefix+hash+":lease")
	after := fixture.stored(t, conversation)
	if after.Pending == nil || after.Pending.RouteID != before.Pending.RouteID {
		t.Fatalf("the record changed: before %+v after %+v", before.Pending, after.Pending)
	}
}
