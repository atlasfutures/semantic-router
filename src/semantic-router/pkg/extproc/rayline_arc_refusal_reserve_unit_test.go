//go:build !windows && cgo

package extproc

import (
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func reservePolicy(enabled, eval bool) *config.RaylineARCPolicyServiceConfig {
	return &config.RaylineARCPolicyServiceConfig{
		EvalCell: eval,
		Fallback: &config.RaylineARCPolicyFallbackConfig{
			Enabled: true, Reserve: &config.RaylineARCRefusalReserveConfig{Enabled: enabled},
		},
	}
}

// An eval cell never re-serves, whatever its config says; off by default.
func TestRefusalReserveIsOffOnEvalCellsAndByDefault(t *testing.T) {
	assert.True(t, reservePolicy(true, false).RefusalReserveEnabled())
	assert.False(t, reservePolicy(true, true).RefusalReserveEnabled(), "an eval cell re-serves")
	assert.False(t, reservePolicy(false, false).RefusalReserveEnabled())
	assert.False(t, (&config.RaylineARCPolicyServiceConfig{Fallback: &config.RaylineARCPolicyFallbackConfig{Enabled: true}}).RefusalReserveEnabled())
	maxReserves, budget, hold, families := reservePolicy(true, false).RefusalReserveLimits()
	assert.Equal(t, 2, maxReserves)
	assert.Equal(t, 45.0, budget.Seconds())
	assert.Equal(t, 60.0, hold.Seconds())
	assert.Equal(t, []string{"claude"}, families)
}

// The gateway's trusted opt-out, and only "off".
func TestRefusalFallbackHeaderOptsOut(t *testing.T) {
	assert.True(t, raylineARCRefusalFallbackOff(map[string]string{raylineARCRefusalFallbackHeader: "off"}))
	assert.True(t, raylineARCRefusalFallbackOff(map[string]string{raylineARCRefusalFallbackHeader: " OFF "}))
	assert.False(t, raylineARCRefusalFallbackOff(map[string]string{raylineARCRefusalFallbackHeader: "structured"}))
	assert.False(t, raylineARCRefusalFallbackOff(map[string]string{}))
	assert.Contains(t, raylineARCTurnSignalHeadersForRemoval(), raylineARCRefusalFallbackHeader, "the opt-out header would reach the provider")
	untrusted := raylineARCTrustedSignalHeaders(&config.RaylineARCPolicyServiceConfig{},
		map[string]string{raylineARCRefusalFallbackHeader: "off"})
	assert.False(t, raylineARCRefusalFallbackOff(untrusted), "an untrusted client opted out")
}

// Only a streamed, lease-holding main turn on a held family, not opted out,
// is held.
func TestStageRefusalReserveHoldsOnlyEligibleTurns(t *testing.T) {
	algorithm := &config.AlgorithmConfig{
		Type: config.RaylineARCAlgorithmType, OnError: "fail_closed",
		RaylineARC: &config.RaylineARCAlgorithmConfig{PolicyService: reservePolicy(true, false)},
	}
	base := func(model string) *RequestContext {
		ctx := &RequestContext{ExpectStreamingResponse: true, VSRSelectedModel: model}
		ctx.VSRRaylineARC = &selection.RaylineARCTrace{PolicyActionModel: model}
		ctx.RaylineARCTransaction = &raylineARCEpisodeTransaction{}
		keepRaylineARCReselect(ctx, &selection.SelectionContext{RaylineARC: &selection.RaylineARCSelectionContext{}}, algorithm, "d", 1)
		return ctx
	}
	router := &OpenAIRouter{}
	dispatch := &providerDispatch{logicalModel: "claude"}
	held := base("claude-opus-5.5")
	router.stageRefusalReserve(held, dispatch, []byte(`{}`))
	assert.NotNil(t, held.RefusalReserve, "an opus turn is not held")
	assert.NotNil(t, held.HostRetry, "a held turn has no plan to hold its frames in")
	for name, ctx := range map[string]*RequestContext{
		"glm":       base("glm-5.3"),
		"opted out": func() *RequestContext { ctx := base("claude-opus-5.5"); ctx.RefusalFallbackOptOut = true; return ctx }(),
		"unstreamed": func() *RequestContext {
			ctx := base("claude-opus-5.5")
			ctx.ExpectStreamingResponse = false
			return ctx
		}(),
		"side call": func() *RequestContext {
			ctx := base("claude-opus-5.5")
			ctx.RaylineARCTransaction.sideCall = true
			return ctx
		}(),
	} {
		router.stageRefusalReserve(ctx, dispatch, []byte(`{}`))
		assert.Nil(t, ctx.RefusalReserve, name)
	}
	evalAlgorithm := *algorithm
	evalAlgorithm.RaylineARC = &config.RaylineARCAlgorithmConfig{PolicyService: reservePolicy(true, true)}
	evalCtx := &RequestContext{ExpectStreamingResponse: true}
	keepRaylineARCReselect(evalCtx, &selection.SelectionContext{RaylineARC: &selection.RaylineARCSelectionContext{}}, &evalAlgorithm, "d", 1)
	assert.Nil(t, evalCtx.raylineARCReselect, "an eval cell kept a re-serve")
}

func TestRefusalSignalsByShape(t *testing.T) {
	state := func(mutate func(*semanticResponseStreamState)) *semanticResponseStreamState {
		s := &semanticResponseStreamState{items: map[int]*semanticStreamItem{0: {reasoning: "thinking"}}}
		mutate(s)
		return s
	}
	assert.Equal(t, "", streamRefusalSignal(state(func(*semanticResponseStreamState) {})))
	assert.Equal(t, "content_filter", streamRefusalSignal(state(func(s *semanticResponseStreamState) { s.stop = "content_filter" })))
	assert.Equal(t, "refusal_part", streamRefusalSignal(state(func(s *semanticResponseStreamState) { s.items[1] = &semanticStreamItem{refused: true} })))
	assert.False(t, streamShowsVisibleContent(state(func(*semanticResponseStreamState) {})), "thinking counts as visible")
	assert.True(t, streamShowsVisibleContent(state(func(s *semanticResponseStreamState) { s.items[1] = &semanticStreamItem{text: "hi"} })))
	assert.Equal(t, "claude", raylinearc.ModelFamily("anthropic/claude-opus-5.5"))
}

// A client cannot turn Anthropic's server-side fallback on through the cell.
func TestServerSideFallbackBetaIsStrippedOnARCTurns(t *testing.T) {
	stripped, removed := withoutServerSideFallbackBetas("context-1m-2025-08-07, server-side-fallback-2026-06-01,interleaved-thinking")
	assert.True(t, removed)
	assert.Equal(t, "context-1m-2025-08-07,interleaved-thinking", stripped)
	_, removed = withoutServerSideFallbackBetas("context-1m-2025-08-07")
	assert.False(t, removed)

	ctx := &RequestContext{Headers: map[string]string{"anthropic-beta": "server-side-fallback-2026-07-01"}, VSRRaylineARC: &selection.RaylineARCTrace{}}
	mutation := &ext_proc.HeaderMutation{}
	stripServerSideFallbackBeta(mutation, ctx)
	assert.Equal(t, []string{"anthropic-beta"}, mutation.RemoveHeaders)
	assert.Equal(t, "", ctx.Headers["anthropic-beta"])

	ctx = &RequestContext{Headers: map[string]string{"anthropic-beta": "a,server-side-fallback-2026-07-01"}, VSRRaylineARC: &selection.RaylineARCTrace{}}
	mutation = &ext_proc.HeaderMutation{SetHeaders: []*core.HeaderValueOption{overwriteRequestHeader("anthropic-beta", "a,server-side-fallback-2026-07-01,extended-cache-ttl-2025-04-11")}}
	stripServerSideFallbackBeta(mutation, ctx)
	assert.Equal(t, "a,extended-cache-ttl-2025-04-11", string(mutation.SetHeaders[0].Header.RawValue))

	plain := &RequestContext{Headers: map[string]string{"anthropic-beta": "server-side-fallback-2026-07-01"}}
	mutation = &ext_proc.HeaderMutation{}
	stripServerSideFallbackBeta(mutation, plain)
	assert.Empty(t, mutation.RemoveHeaders, "a turn outside ARC was touched")
}
