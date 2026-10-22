package raylinearc

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"slices"
	"strings"
)

// Which OpenRouter endpoint of an arm serves a conversation's turn. The
// gateway (memex-desktop firebase/shared/src/usage/endpointPick.ts) and this
// cell pick the same way, so one conversation sticks to one endpoint's prompt
// cache whichever side picks, and the gateway can bill the endpoint that
// served. testdata/endpoint-pick-hrw.json pins the order for both.

// PickEndpoint orders an arm's endpoint tags for one session: rendezvous
// (highest random weight) hashing, each tag scored by the SHA-256 of
// sessionKey + "\n" + tag (UTF-8), highest digest first as unsigned bytes,
// ties by tag. The first is the pick and the rest are the retry order.
// Removing a tag moves only the sessions that had picked it. A repeated tag
// counts once; no tags is no order.
func PickEndpoint(sessionKey string, tags []string) []string {
	type scored struct {
		tag    string
		digest [sha256.Size]byte
	}
	candidates := make([]scored, 0, len(tags))
	for _, tag := range tags {
		if slices.ContainsFunc(candidates, func(c scored) bool { return c.tag == tag }) {
			continue
		}
		candidates = append(candidates, scored{tag: tag, digest: sha256.Sum256([]byte(sessionKey + "\n" + tag))})
	}
	slices.SortFunc(candidates, func(a, b scored) int {
		if order := bytes.Compare(b.digest[:], a.digest[:]); order != 0 {
			return order
		}
		return strings.Compare(a.tag, b.tag)
	})
	order := make([]string, len(candidates))
	for index, candidate := range candidates {
		order[index] = candidate.tag
	}
	return order
}

// SessionKeyDigest is a conversation's session key as the gateway sends it in
// x-rayline-session-key: the hex SHA-256 of userID + ":" + conversationID
// (UTF-8), so it never carries either id.
func SessionKeyDigest(userID string, conversationID string) string {
	digest := sha256.Sum256([]byte(userID + ":" + conversationID))
	return hex.EncodeToString(digest[:])
}

// EndpointTags are the endpoint tags the feed carries rows for under an
// OpenRouter model id, lower-cased, sorted and each once: the endpoints a
// pin can reach for that model. Nil for a model the feed does not carry.
func (settlement *RoutingPriceSettlement) EndpointTags(model string) []string {
	if settlement == nil {
		return nil
	}
	var tags []string
	for _, row := range settlement.rows[model] {
		if row.endpointTag != "" && !slices.Contains(tags, row.endpointTag) {
			tags = append(tags, row.endpointTag)
		}
	}
	slices.Sort(tags)
	return tags
}

// PinRange is the tags, in their order, an OpenRouter provider object can be
// served by: what both only and order reach (a list left unset does not
// narrow), less what ignore reaches, each slug reaching as OpenRouter matches
// endpoint slugs (slugMatches: xai reaches xai/zdr and xai/zdr/us, a tier
// endpoint only its full slug). It is memex's pinRange
// (firebase/router/src/rayline-data-use.ts) over the feed's tags rather than
// catalog records.
func PinRange(only, order, ignore, tags []string) []string {
	only, order, ignore = uniqueLowerSlugs(only), uniqueLowerSlugs(order), uniqueLowerSlugs(ignore)
	reached := []string{}
	for _, tag := range tags {
		if (len(only) == 0 || pinAdmits(only, tag)) && (len(order) == 0 || pinAdmits(order, tag)) &&
			!pinAdmits(ignore, tag) {
			reached = append(reached, tag)
		}
	}
	return reached
}

// ArmEndpoints is what a private turn's endpoint constraint needs of one
// OpenRouter arm: the model id and pin its reach is computed from, and the
// endpoint tags the gateway allows it (x-rayline-privacy-allowed-endpoints).
type ArmEndpoints struct {
	Model               string
	Only, Order, Ignore []string
	Allowed             []string
}

// AllowedReach is the arm's reachable endpoints that the gateway allows, in
// the feed's order: empty when none is, and then the arm cannot serve the
// turn.
func (arm ArmEndpoints) AllowedReach(settlement *RoutingPriceSettlement) []string {
	reach := PinRange(arm.Only, arm.Order, arm.Ignore, settlement.EndpointTags(arm.Model))
	return AllowedEndpoints(reach, arm.Allowed)
}

// AllowedEndpoints keeps the tags, in their order, that allowed lists.
func AllowedEndpoints(tags, allowed []string) []string {
	return slices.DeleteFunc(slices.Clone(tags), func(tag string) bool { return !slices.Contains(allowed, tag) })
}
