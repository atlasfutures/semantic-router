package extproc

import (
	"context"
	"errors"
	"net/http"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

const (
	snapshotFirst        = "arc-1.0-20261022"
	snapshotSecond       = "arc-1.0-20261105"
	snapshotFirstRelease = "rayline-ai/arc-1.0@20261022"
	snapshotSecondAlias  = "rayline/arc-later"
)

var snapshotSecondPackage = strings.Repeat("b", 64)

// snapshotTable is a two-snapshot table for policyTestDecision: the default
// is the block's package at V 10, the second another package with the same
// catalog at V 0.2.
func snapshotTable() *config.RaylineARCSnapshotsConfig {
	swift, sage := 10.0, 0.2
	return &config.RaylineARCSnapshotsConfig{
		Default: snapshotFirst,
		Entries: map[string]config.RaylineARCSnapshotEntry{
			snapshotFirst: {
				Release: snapshotFirstRelease, PackageAlias: policyTestAlias, PackageSHA256: policyTestPackage,
				CostWeight: &swift,
			},
			snapshotSecond: {
				Release: "rayline-ai/arc-1.0@20261105", PackageAlias: snapshotSecondAlias,
				PackageSHA256: snapshotSecondPackage, CostWeight: &sage,
			},
		},
	}
}

// newSnapshotFixture is a snapshot cell over the policy fixture, whose
// service serves both snapshots' packages.
func newSnapshotFixture(t *testing.T) *policySelectorFixture {
	t.Helper()
	return newSnapshotFixtureWith(t, nil)
}

// newSnapshotFixtureWith edits the policy block before any scorer reads it:
// a block's snapshot views are built once.
func newSnapshotFixtureWith(t *testing.T, edit func(*config.RaylineARCPolicyServiceConfig)) *policySelectorFixture {
	t.Helper()
	fixture := newPolicySelectorFixture(t, "")
	// The base fixture armed a scorer over the plain block; start again.
	fixture.decision = policyTestDecision("")
	policy := fixture.decision.Algorithm.RaylineARC.PolicyService
	policy.Snapshots = snapshotTable()
	if edit != nil {
		edit(policy)
	}
	fixture.fake.others = map[string]string{snapshotSecondAlias: snapshotSecondPackage}
	fixture.selector.arm(&raylineARCArmedComponents{
		scorer:    newPolicyServiceScorer(&config.RouterConfig{}, fixture.decision),
		admission: raylinearc.NewAdmissionGate(0),
		policy: raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{
			BaseURL: fixture.fake.URL(), TotalTimeout: 5 * time.Second,
		}),
	})
	actionID := fixture.decision.Algorithm.RaylineARC.PolicyService.Bindings[0].ActionID
	fixture.fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actionID })
	return fixture
}

func (fixture *policySelectorFixture) selectUnder(
	t *testing.T,
	snapshot string,
	state *raylinearc.EpisodeState,
	body []byte,
) (*selection.SelectionResult, error) {
	t.Helper()
	return fixture.selector.Select(context.Background(), &selection.SelectionContext{
		DecisionName:    fixture.decision.Name,
		CandidateModels: fixture.decision.ModelRefs,
		RaylineARC: &selection.RaylineARCSelectionContext{
			EpisodeIDHash: strings.Repeat("e", 64),
			RequestID:     "req-snapshot-test",
			State:         state,
			RawRequest:    body,
			RequestFormat: policyFormatAnthropic,
			Snapshot:      snapshot,
		},
	})
}

// Two snapshots interleaved on one cell each decide with their own package
// and V, and each decision records the snapshot, its release and package.
func TestSnapshotsInterleavedDecideWithTheirOwnPackages(t *testing.T) {
	fixture := newSnapshotFixture(t)
	body := policyTestRequest(t, map[string]any{"role": "user", "content": "go"})
	want := []struct {
		snapshot, alias, sha256, release string
		weight                           float64
	}{
		{snapshotFirst, policyTestAlias, policyTestPackage, snapshotFirstRelease, 10},
		{snapshotSecond, snapshotSecondAlias, snapshotSecondPackage, "rayline-ai/arc-1.0@20261105", 0.2},
		{snapshotFirst, policyTestAlias, policyTestPackage, snapshotFirstRelease, 10},
		{snapshotSecond, snapshotSecondAlias, snapshotSecondPackage, "rayline-ai/arc-1.0@20261105", 0.2},
	}
	for index, turn := range want {
		state, _ := raylinearc.NewEpisodeState(2)
		result, err := fixture.selectUnder(t, turn.snapshot, state, body)
		if err != nil {
			t.Fatalf("turn %d (%s): %v", index, turn.snapshot, err)
		}
		sent := fixture.fake.received()[index]
		if sent.Package.Alias != turn.alias || sent.Package.PackageSHA256 != turn.sha256 {
			t.Fatalf("turn %d sent package %+v, want %s", index, sent.Package, turn.alias)
		}
		if weight := sent.Selection.OperatingPoint.CostWeight; weight == nil || *weight != turn.weight {
			t.Fatalf("turn %d sent cost weight %v, want %g", index, weight, turn.weight)
		}
		trace := result.RaylineARC
		if trace.Snapshot != turn.snapshot || trace.Release != turn.release || trace.PolicyPackageSHA256 != turn.sha256 ||
			trace.PolicyCostWeight == nil || *trace.PolicyCostWeight != turn.weight {
			t.Fatalf("turn %d trace = snapshot %q release %q package %q weight %v", index,
				trace.Snapshot, trace.Release, trace.PolicyPackageSHA256, trace.PolicyCostWeight)
		}
		if trace.ArtifactRevision != hashedARCIdentity(turn.sha256) {
			t.Fatalf("turn %d artifact revision is not its package's", index)
		}
	}
}

// A decision answered for another package than the pinned snapshot's fails
// the turn as the service's error (502), never served.
func TestSnapshotPackageMismatchFailsTheTurn(t *testing.T) {
	fixture := newSnapshotFixture(t)
	fixture.fake.answerAs = &raylinearc.PolicyPackageRef{Alias: policyTestAlias, PackageSHA256: policyTestPackage}
	state, _ := raylinearc.NewEpisodeState(2)
	_, err := fixture.selectUnder(t, snapshotSecond, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
	var failure *raylineARCSelectionFailure
	if !errors.As(err, &failure) || failure.class != arcFailureSnapshotPackageMismatch {
		t.Fatalf("err = %v, want %s", err, arcFailureSnapshotPackageMismatch)
	}
	response := (&OpenAIRouter{}).authoritativeSelectionFailureResponse(
		&modelSelectionFailure{algorithm: configRaylineARC, class: failure.class}, &RequestContext{},
	)
	if got := int(response.GetImmediateResponse().GetStatus().GetCode()); got != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502", got)
	}
	if got := immediateHeaderValue(response, selectionFailureHeader); got != selectionFailureUnavailable {
		t.Fatalf("failure class = %q", got)
	}
}

// A snapshot the scorer does not hold is refused, never decided by another.
func TestSnapshotScorerRefusesAnUnknownSnapshot(t *testing.T) {
	fixture := newSnapshotFixture(t)
	state, _ := raylinearc.NewEpisodeState(2)
	_, err := fixture.selectUnder(t, "arc-1.0-20991231", state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"}))
	var failure *raylineARCSelectionFailure
	if !errors.As(err, &failure) || failure.class != arcFailureUnknownSnapshot {
		t.Fatalf("err = %v, want %s", err, arcFailureUnknownSnapshot)
	}
	if len(fixture.fake.received()) != 0 {
		t.Fatal("an unknown snapshot reached the service")
	}
}

// The header pins the snapshot; absent, the default serves; an unknown one is
// refused before the lease with a 400 and the public class unknown_snapshot.
func TestSnapshotHeaderResolvesBeforeTheLease(t *testing.T) {
	decision := policyTestDecision("")
	decision.Algorithm.RaylineARC.PolicyService.Snapshots = snapshotTable()
	decision.Algorithm.RaylineARC.Episode.IDHeader = testEpisodeIDHeader
	for header, want := range map[string]string{"": snapshotFirst, snapshotSecond: snapshotSecond, " " + snapshotSecond + " ": snapshotSecond} {
		ctx := snapshotRequestContext(header)
		got := (&OpenAIRouter{}).buildRaylineARCSelectionContext(decision.Algorithm, ctx, decision.ModelRefs, raylineARCEpisodeEphemeral)
		if got.Snapshot != want || got.PreparationFailure == arcFailureUnknownSnapshot {
			t.Fatalf("header %q: snapshot %q failure %q, want %q", header, got.Snapshot, got.PreparationFailure, want)
		}
	}
	ctx := snapshotRequestContext("arc-1.0-20991231")
	got := (&OpenAIRouter{}).buildRaylineARCSelectionContext(decision.Algorithm, ctx, decision.ModelRefs, raylineARCEpisodeEphemeral)
	if got.PreparationFailure != arcFailureUnknownSnapshot || got.Snapshot != "" {
		t.Fatalf("unknown snapshot: failure %q snapshot %q", got.PreparationFailure, got.Snapshot)
	}
	requestContext := &RequestContext{}
	response := (&OpenAIRouter{}).authoritativeSelectionFailureResponse(
		&modelSelectionFailure{algorithm: configRaylineARC, class: arcFailureUnknownSnapshot}, requestContext,
	)
	if code := int(response.GetImmediateResponse().GetStatus().GetCode()); code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", code)
	}
	if class := immediateHeaderValue(response, selectionFailureHeader); class != selectionFailureUnknownSnapshot {
		t.Fatalf("failure class = %q", class)
	}
	if requestContext.ImmediateProtocolError == nil {
		t.Fatal("the refusal carries no protocol error")
	}

	// A cell without a table reads no header: the snapshot stays empty.
	plain := policyTestDecision("")
	plain.Algorithm.RaylineARC.Episode.IDHeader = testEpisodeIDHeader
	got = (&OpenAIRouter{}).buildRaylineARCSelectionContext(plain.Algorithm, snapshotRequestContext("arc-1.0-20991231"), plain.ModelRefs, raylineARCEpisodeEphemeral)
	if got.Snapshot != "" || got.PreparationFailure == arcFailureUnknownSnapshot {
		t.Fatalf("a cell without a table: snapshot %q failure %q", got.Snapshot, got.PreparationFailure)
	}
}

func snapshotRequestContext(pin string) *RequestContext {
	ctx := coalesceRequestContext(`{"turn":1}`)
	ctx.Headers = map[string]string{}
	if pin != "" {
		ctx.Headers[config.RaylineARCSnapshotHeader] = pin
	}
	ctx.SourceFormat = llmprotocol.AnthropicMessagesV1
	return ctx
}

// The pin is never forwarded upstream, and two requests that differ only in
// it are different turns to the coalescer.
func TestSnapshotHeaderIsStrippedAndKeysTheTurn(t *testing.T) {
	if !slices.Contains(raylineARCTurnSignalHeadersForRemoval(), config.RaylineARCSnapshotHeader) {
		t.Fatal("x-rayline-arc-snapshot is forwarded upstream")
	}
	arc := policyTestDecision("").Algorithm.RaylineARC
	first, second := snapshotRequestContext(snapshotFirst), snapshotRequestContext(snapshotSecond)
	if raylineARCTurnInputs(arc, first, nil) == raylineARCTurnInputs(arc, second, nil) {
		t.Fatal("requests pinned to two snapshots share a coalescing key")
	}
}

// A served snapshot turn names its snapshot and release beside its package;
// a turn without a snapshot names neither.
func TestSnapshotServedIdentityHeaders(t *testing.T) {
	ctx := servedIdentityContext()
	ctx.VSRRaylineARC.Snapshot, ctx.VSRRaylineARC.Release = snapshotFirst, snapshotFirstRelease
	set := servedIdentityHeaders(ctx, true)
	if set[headers.VSRARCSnapshot] != snapshotFirst || set[headers.VSRARCRelease] != snapshotFirstRelease ||
		set[headers.VSRPolicyPackage] != "package-digest" {
		t.Fatalf("headers = %v", set)
	}
	plain := servedIdentityHeaders(servedIdentityContext(), true)
	if _, ok := plain[headers.VSRARCSnapshot]; ok {
		t.Fatalf("a turn without a snapshot names one: %v", plain)
	}
	if _, ok := plain[headers.VSRARCRelease]; ok {
		t.Fatalf("a turn without a snapshot names a release: %v", plain)
	}
}

// A conversation whose earlier turn another snapshot's package decided sends
// only the attributions this package knows, and releases a held arm this
// package has no action on.
func TestSnapshotKeepsOnlyItsOwnAttributionsAndArms(t *testing.T) {
	fixture := newSnapshotFixtureWith(t, func(policy *config.RaylineARCPolicyServiceConfig) {
		// The second snapshot's package serves only the thinking worker.
		entry := policy.Snapshots.Entries[snapshotSecond]
		entry.Bindings = policy.Bindings[:2]
		policy.Snapshots.Entries[snapshotSecond] = entry
		policy.ModelSchedule = config.RaylineARCModelScheduleTaskTurnCompaction
	})
	policy := fixture.decision.Algorithm.RaylineARC.PolicyService
	cell := newPolicyServiceScorer(&config.RouterConfig{}, fixture.decision)
	second, ok := cell.forSnapshot(snapshotSecond)
	if !ok {
		t.Fatal("no scorer for the second snapshot")
	}
	offAction := policy.Bindings[2].ActionID
	kept := second.snapshotAttribution([]raylinearc.PolicyAttribution{
		{Message: 1, ActionID: policy.Bindings[0].ActionID}, {Message: 3, ActionID: offAction},
	})
	if len(kept) != 1 || kept[0].Message != 1 {
		t.Fatalf("kept %+v", kept)
	}
	// Trained arm ids are per package: one this package's manifest does not
	// list (here, a v4 package's, which VSR cannot know) is sent as null.
	otherArm := "arm-of-another-package"
	nulled := second.snapshotAttribution([]raylinearc.PolicyAttribution{
		{Message: 1, ActionID: policy.Bindings[0].ActionID, ArmID: &otherArm},
	})
	if len(nulled) != 1 || nulled[0].ArmID != nil {
		t.Fatalf("attribution %+v keeps another package's arm", nulled)
	}
	known := policyServiceScorer{bindings: map[string]policyBinding{"a": {trainedArms: []string{otherArm}}}}
	if got := known.snapshotAttribution([]raylinearc.PolicyAttribution{{ActionID: "a", ArmID: &otherArm}}); got[0].ArmID == nil {
		t.Fatal("an arm the package lists for the action was dropped")
	}
	// A served-model record naming a worker this package binds no action on
	// says nothing to this snapshot.
	if model := second.servedModelOf("off"); model != "" {
		t.Fatalf("servedModelOf(off) = %q under a package without it", model)
	}
	if model := second.servedModelOf("think"); model == "" {
		t.Fatal("servedModelOf(think) says nothing under a package that binds it")
	}
	if !second.armBound(0) || second.armBound(1) {
		t.Fatal("arm binding is not the package's")
	}
	if _, ok := cell.forSnapshot(""); ok {
		t.Fatal("a snapshot cell decided with no snapshot")
	}
	if plain := newPolicyServiceScorer(&config.RouterConfig{}, policyTestDecision("")); plain.snapshots != nil {
		t.Fatal("a cell without a table built snapshot scorers")
	} else if served, ok := plain.forSnapshot(""); !ok || served != plain {
		t.Fatal("a cell without a table does not decide with its own scorer")
	}

	// A held turn on the off arm, served under the second snapshot, is
	// released and decided among the package's own actions.
	fixture.selector.arm(&raylineARCArmedComponents{
		scorer:    cell,
		admission: raylinearc.NewAdmissionGate(0),
		policy: raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{
			BaseURL: fixture.fake.URL(), TotalTimeout: 5 * time.Second,
		}),
	})
	// The service scores the second package's own catalog.
	fixture.fake.catalog = policyTestActionIDs(policy.Bindings[:2])
	state, _ := raylinearc.NewEpisodeState(2)
	off := 1
	state.PreviousArm = &off
	state.TurnIndex = 1
	if _, err := fixture.selectUnder(t, snapshotSecond, state, policyTestRequest(t, map[string]any{"role": "user", "content": "go"})); err != nil {
		t.Fatalf("select: %v", err)
	}
	sent := fixture.fake.received()[0]
	if slices.Contains(sent.Selection.AvailableActionIDs, offAction) || len(sent.Selection.AvailableActionIDs) != 2 {
		t.Fatalf("offered %v", sent.Selection.AvailableActionIDs)
	}
}

// A snapshot cell is ready only once the service serves every snapshot's
// package, each asked for by its own pin.
func TestSnapshotCellReadinessCoversEverySnapshot(t *testing.T) {
	fixture := newSnapshotFixture(t)
	policy := fixture.decision.Algorithm.RaylineARC.PolicyService
	cell := newPolicyServiceScorer(&config.RouterConfig{}, fixture.decision)
	client := raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{BaseURL: fixture.fake.URL(), TotalTimeout: 5 * time.Second})
	if err := probeRaylineARCPolicyCell(context.Background(), client, policy, nil, cell); err != nil {
		t.Fatalf("a service serving both packages: %v", err)
	}
	asked := map[string]bool{}
	for _, query := range fixture.fake.listingQueries {
		asked[query.Get("alias")+"@"+query.Get("package_sha256")] = true
	}
	if !asked[policyTestAlias+"@"+policyTestPackage] || !asked[snapshotSecondAlias+"@"+snapshotSecondPackage] {
		t.Fatalf("listings asked for %v", asked)
	}
	fixture.fake.mu.Lock()
	fixture.fake.others = nil
	fixture.fake.mu.Unlock()
	var failure *raylinearc.PolicyServiceError
	err := probeRaylineARCPolicyCell(context.Background(), client, policy, nil, cell)
	if !errors.As(err, &failure) || failure.Class != "package_not_loaded" {
		t.Fatalf("a service without the second package: err = %v", err)
	}
}

// POST /v1/routes refuses an unserved pin as the routed path does: an
// invalid request (400) whose message names the snapshot header.
func TestSnapshotUnknownPinOnTheRoutesAPI(t *testing.T) {
	err := prepareFailureError(arcFailureUnknownSnapshot)
	if !errors.Is(err, routerruntime.ErrRouteDecisionInvalidBody) || !errors.Is(err, errRaylineRoutesUnknownSnapshot) {
		t.Fatalf("err = %v", err)
	}
}
