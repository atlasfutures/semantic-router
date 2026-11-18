package extproc

import (
	"context"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// generationTurn builds, selects and stages a generation turn of conversation
// on the fixture's cell, as the routed path does before dispatch.
func (fixture *pendingRoutesFixture) generationTurn(t *testing.T, conversation string, body []byte) *RequestContext {
	t.Helper()
	decision := fixture.router.Config.Decisions[0]
	reqCtx := &RequestContext{
		Headers: map[string]string{decisionServiceEpisodeHeader: conversation}, RaylineARCRawBody: body,
		SourceFormat: llmprotocol.AnthropicMessagesV1, RequestID: "generation-" + conversation,
		TraceContext: context.Background(),
	}
	arcContext := fixture.router.buildRaylineARCSelectionContext(decision.Algorithm, reqCtx, decision.ModelRefs, raylineARCEpisodeRequired)
	if arcContext.PreparationFailure != "" {
		t.Fatalf("preparation failed: %s", arcContext.PreparationFailure)
	}
	result, err := fixture.policy.selector.Select(context.Background(), &selection.SelectionContext{
		DecisionName: decision.Name, CandidateModels: decision.ModelRefs, RaylineARC: arcContext,
	})
	if err != nil {
		t.Fatalf("select: %v", err)
	}
	reqCtx.VSRRaylineARC, reqCtx.VSRSelectedDecision, reqCtx.VSRSelectedModel = result.RaylineARC, &decision, result.SelectedModel
	reqCtx.RaylineARCTransaction.markSelection(result.RaylineARC.SelectedArm, result.RaylineARC.SerializedTokens)
	stageRaylineARCPolicySelection(reqCtx, result.RaylineARC, result.SelectedModel)
	return reqCtx
}

func withFallback(fixture *pendingRoutesFixture) {
	fixture.router.Config.Decisions[0].Algorithm.RaylineARC.PolicyService.Fallback = &config.RaylineARCPolicyFallbackConfig{Enabled: true}
}

// A generation turn that records a lookup's decision as a turn that ran and
// is then refused by the provider after its lease was given back (a late
// refusal) still excludes the refusing model: the store holds the episode
// the turn was decided on, so the refusal matches it.
func TestLateRefusalAfterAnInferredTurnExcludesTheModel(t *testing.T) {
	fixture := newPendingRoutesFixture(t, true, "")
	withFallback(fixture)
	const conversation = "conversation-late-refusal"
	fixture.picks = []string{fixture.actions[0]}
	fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))
	reqCtx := fixture.generationTurn(t, conversation, pendingRoutesBody(t, pendingTurnTask, pendingTurnReply, pendingTurnNext))
	if reqCtx.RaylineARCTransaction.state.TurnIndex != 1 {
		t.Fatalf("the turn did not infer the lookup's decision: turn %d", reqCtx.RaylineARCTransaction.state.TurnIndex)
	}
	refused := refusedModel(reqCtx)
	if err := reqCtx.RaylineARCTransaction.abort(context.Background(), "upstream_error"); err != nil {
		t.Fatal(err)
	}
	declineRefusedTurn(reqCtx)
	stored := fixture.stored(t, conversation)
	if stored.Policy == nil || !stored.Policy.Excludes(refused) {
		t.Fatalf("after a late refusal the store does not exclude %q: %+v", refused, stored.Policy)
	}
}
