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
