//go:build !windows && cgo

package extproc

import (
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// feedScorer is remapScorer with each worker's OpenRouter provider order, as
// the cell config pins them: flash and flash-1017 prefer deepinfra, opus
// anthropic.
func feedScorer() *policyServiceScorer {
	scorer := remapScorer()
	scorer.workers[0].OpenRouterProviderOrder = []string{"cohere", "deepinfra/fp8"}
	scorer.workers[1].OpenRouterProviderOrder = []string{"deepinfra/fp8", "morph/fp8"}
	scorer.workers[2].OpenRouterProviderOrder = []string{"anthropic", "amazon-bedrock"}
	return scorer
}

func feedWorld(rates map[string]raylinearc.PolicyRoutingRate) *raylinearc.PolicyRoutingPrices {
	return &raylinearc.PolicyRoutingPrices{
		Schema: raylinearc.RoutingPricesSchema, StoreVersion: "217", AsOf: "2026-11-13T00:00:00Z", Rates: rates,
	}
}

// The feed as memex-desktop serves it for these models: 1017 and opus at
// several endpoints (tagged keys only), 0910 not listed.
func taggedFeed() *raylinearc.PolicyRoutingPrices {
	return feedWorld(map[string]raylinearc.PolicyRoutingRate{
		"deepseek/deepseek-v4.1-flash-1017@deepinfra/fp8": callerRate(0.098, 0.294),
		"deepseek/deepseek-v4.1-flash-1017@morph/fp8":     callerRate(0.022, 0.4),
		"anthropic/claude-opus-5@anthropic":               callerRate(5, 25),
		"anthropic/claude-opus-5@amazon-bedrock":          callerRate(5.5, 27.5),
	})
}

// A remapped arm reads its served twin's rate at the dispatch worker's first
// endpoint under its own id, and an arm whose model the feed lists only at
// tagged endpoints reads its first endpoint's rate under its bare id. Every
// feed entry still goes out, and the cache's feed is not changed.
func TestRouteFeedPricesKeyEveryArmByTheIDTheServiceReads(t *testing.T) {
	feed := taggedFeed()
	sent := routeFeedPrices(feedScorer(), feed)
	want := []string{
		"anthropic/claude-opus-5", "anthropic/claude-opus-5@amazon-bedrock", "anthropic/claude-opus-5@anthropic",
		"deepseek/deepseek-v4.1-flash", "deepseek/deepseek-v4.1-flash-1017@deepinfra/fp8",
		"deepseek/deepseek-v4.1-flash-1017@morph/fp8",
	}
	if got := rateKeys(sent); !slices.Equal(got, want) {
		t.Fatalf("sent keys = %v, want %v", got, want)
	}
	for key, legs := range map[string][4]float64{
		"deepseek/deepseek-v4.1-flash": {0.098, 0, 0.098, 0.294},
		"anthropic/claude-opus-5":      {5, 0, 5, 25},
	} {
		if got := routeRateLegs(sent.Rates[key]); got != legs {
			t.Fatalf("%s = %v, want %v", key, got, legs)
		}
	}
	if len(feed.Rates) != 4 {
		t.Fatalf("the cached feed was changed in place: %v", rateKeys(feed))
	}
	if sent.StoreVersion != feed.StoreVersion || sent.AsOf != feed.AsOf || sent.Schema != feed.Schema || sent.Basis != "" {
		t.Fatalf("sent %+v lost the feed's identity", sent)
	}
}

// A feed that already keys every arm as the service reads it goes out as
// served, the same object.
func TestRouteFeedPricesLeaveAFeedTheServiceReadsAsServed(t *testing.T) {
	feed := feedWorld(map[string]raylinearc.PolicyRoutingRate{
		"deepseek/deepseek-v4.1-flash": callerRate(0.098, 0.294),
		"anthropic/claude-opus-5":      callerRate(5, 25),
	})
	if sent := routeFeedPrices(feedScorer(), feed); sent != feed {
		t.Fatalf("sent %v, want the feed itself", rateKeys(sent))
	}
}

// A remapped arm is priced at the model it is sent to, over an entry the feed
// lists under its own id (a retired snapshot still in the catalog); an
// unremapped arm's own entry is never replaced.
func TestRouteFeedPricesPreferTheDispatchedRateForARemappedArm(t *testing.T) {
	feed := taggedFeed()
	feed.Rates["deepseek/deepseek-v4.1-flash"] = callerRate(0.3, 1.2)
	feed.Rates["anthropic/claude-opus-5"] = callerRate(4, 20)
	sent := routeFeedPrices(feedScorer(), feed)
	if got := routeRateLegs(sent.Rates["deepseek/deepseek-v4.1-flash"]); got != [4]float64{0.098, 0, 0.098, 0.294} {
		t.Fatalf("remapped arm = %v, want the 1017 deepinfra rate", got)
	}
	if got := routeRateLegs(sent.Rates["anthropic/claude-opus-5"]); got != [4]float64{4, 0, 4, 20} {
		t.Fatalf("opus = %v, want its own listed entry", got)
	}
	if got := routeRateLegs(feed.Rates["deepseek/deepseek-v4.1-flash"]); got != [4]float64{0.3, 0, 0.3, 1.2} {
		t.Fatal("the cached feed was changed in place")
	}
}

// Without an entry at the worker's first-preference endpoint, the arm is not
// priced from another endpoint: it keeps the package rate.
func TestRouteFeedPricesNeverPriceAnArmAtAnotherEndpoint(t *testing.T) {
	feed := feedWorld(map[string]raylinearc.PolicyRoutingRate{
		"deepseek/deepseek-v4.1-flash-1017@morph/fp8": callerRate(0.022, 0.4),
		"anthropic/claude-opus-5@amazon-bedrock":      callerRate(5.5, 27.5),
	})
	if sent := routeFeedPrices(feedScorer(), feed); sent != feed {
		t.Fatalf("sent %v, want the feed unchanged", rateKeys(sent))
	}
	scorer := feedScorer()
	scorer.workers[1].OpenRouterProviderOrder = nil
	if sent := routeFeedPrices(scorer, taggedFeed()); sent.Rates["deepseek/deepseek-v4.1-flash"].Input != nil {
		t.Fatal("a worker with no provider order was priced at an endpoint")
	}
}

// A caller's world, an absent feed and a cell without a scorer pass through.
func TestRouteFeedPricesPassCallerWorldsAndAbsentFeedsThrough(t *testing.T) {
	caller := callerWorld(map[string]raylinearc.PolicyRoutingRate{
		"deepseek/deepseek-v4.1-flash-1017@deepinfra/fp8": callerRate(0.2, 0.8),
	})
	if sent := routeFeedPrices(feedScorer(), caller); sent != caller {
		t.Fatal("a caller's world was translated as a feed")
	}
	if sent := routeFeedPrices(feedScorer(), nil); sent != nil {
		t.Fatal("an absent feed came back as prices")
	}
	feed := taggedFeed()
	if sent := routeFeedPrices(nil, feed); sent != feed {
		t.Fatal("a cell without a scorer changed the feed")
	}
}

// A remapped arm takes its dispatched rate even when an unremapped action
// reading the same id is bound before it.
func TestRouteFeedPricesRemappedArmWinsWhateverTheActionOrder(t *testing.T) {
	scorer := feedScorer()
	scorer.actionOrder = append([]string{"flash-own"}, scorer.actionOrder...)
	scorer.bindings["flash-own"] = policyBinding{arm: 0, dispatchArm: 0, model: "deepseek-v4.1-flash"}
	feed := taggedFeed()
	feed.Rates["deepseek/deepseek-v4.1-flash@cohere"] = callerRate(0.5, 2)
	sent := routeFeedPrices(scorer, feed)
	if got := routeRateLegs(sent.Rates["deepseek/deepseek-v4.1-flash"]); got != [4]float64{0.098, 0, 0.098, 0.294} {
		t.Fatalf("flash = %v, want the dispatched 1017 deepinfra rate", got)
	}
}

// An own-id entry with a null leg is not the served rate with that leg 0:
// the translated rate goes out.
func TestRouteFeedPricesTellANullLegFromZero(t *testing.T) {
	feed := taggedFeed()
	own := callerRate(0.098, 0.294)
	own.CacheRead = nil
	feed.Rates["deepseek/deepseek-v4.1-flash"] = own
	sent := routeFeedPrices(feedScorer(), feed)
	if sent == feed || sent.Rates["deepseek/deepseek-v4.1-flash"].CacheRead == nil {
		t.Fatal("a null cache_read leg was taken for the served rate's 0")
	}
}
