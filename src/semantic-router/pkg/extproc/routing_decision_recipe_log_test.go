package extproc

import (
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A generation turn's routing_decision names the recipe that decided it, as a
// /v1/routes lookup's does, so monitoring can split generation by tier on the
// one cell that serves them all.
func TestRoutingDecisionNamesTheDecidingRecipe(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := &RequestContext{RequestID: "rt_recipe", ProcessingStartTime: time.Now()}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: "arc-sage"})

	router.logRoutingDecision(ctx, "entrypoint_routing", "arc-sage", "gpt-5-mini", "arc", false)
	router.emitRoutingDecision(ctx)

	fields := findLogEvent(t, logs, "routing_decision")
	if fields["recipe"] != "arc-sage" {
		t.Fatalf("recipe = %v, want arc-sage: %v", fields["recipe"], fields)
	}
}

// A concrete model routes by no recipe, and its record says none.
func TestRoutingDecisionOfAPassthroughTurnNamesNoRecipe(t *testing.T) {
	logs := captureLogs(t)
	router := &OpenAIRouter{}
	ctx := &RequestContext{RequestID: "rt_passthrough", ProcessingStartTime: time.Now()}
	ctx.Routing.SelectPassthrough()

	router.logRoutingDecision(ctx, "passthrough", "gpt-5-mini", "gpt-5-mini", "", false)
	router.emitRoutingDecision(ctx)

	fields := findLogEvent(t, logs, "routing_decision")
	if recipe, present := fields["recipe"]; present {
		t.Fatalf("a passthrough turn named recipe %v: %v", recipe, fields)
	}
}
