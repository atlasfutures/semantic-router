package raylinearc

import (
	"errors"
	"regexp"
	"slices"
)

// Encrypted reasoning issuers (atlasfutures/semantic-router#109).
//
// A Responses client running with store:false (Codex) resends every reasoning
// item it has received, each with its encrypted_content. Only the provider
// account and model that issued a blob can read it, and the blob itself does
// not say who that was. So the episode records the set of targets that issued
// the blobs its client can still be holding, and a turn forwards them only
// when that set is exactly the turn's own target. Two entries are enough: a
// set with two is already "some blob is foreign", and it stays so until a
// turn arrives that resends none.

// MaxReasoningIssuers bounds the issuer set an episode keeps.
const MaxReasoningIssuers = 2

// ReasoningIssuerUnknown stands for blobs the episode did not see issued: the
// client resent encrypted reasoning before the episode recorded any issuer
// (an episode older than the record, or blobs from outside the route). It
// never equals a target, so those blobs are never forwarded.
const ReasoningIssuerUnknown = "unknown"

var reasoningIssuerPattern = regexp.MustCompile(`^(unknown|[0-9a-f]{16})$`)

// ReasoningIssuersAre reports whether the set is exactly issuer, the only case
// in which every blob the client resends was issued by that target.
func ReasoningIssuersAre(issuers []string, issuer string) bool {
	return len(issuers) == 1 && issuers[0] == issuer
}

// NextReasoningIssuers is the issuer set after a turn commits.
//
//   - held: the request resent encrypted reasoning.
//   - forwarded: it was sent on to the target, which only happens when the
//     set was exactly that target.
//   - issues: the target speaks Responses, so its reply can carry new blobs.
//
// A turn that resends none tells the router the client holds none, so the set
// restarts from what this turn's reply can add. Otherwise the blobs the client
// still holds keep their issuers, and the target's new ones join them.
func NextReasoningIssuers(previous []string, held, forwarded, issues bool, issuer string) []string {
	switch {
	case !held:
		if issues {
			return []string{issuer}
		}
		return nil
	case forwarded:
		return []string{issuer}
	}
	next := append([]string(nil), previous...)
	if len(next) == 0 {
		next = []string{ReasoningIssuerUnknown}
	}
	if issues && !slices.Contains(next, issuer) {
		next = append(next, issuer)
	}
	if len(next) > MaxReasoningIssuers {
		// Any two entries already mean "foreign"; which two is immaterial.
		next = next[:MaxReasoningIssuers]
	}
	return next
}

func validateReasoningIssuers(issuers []string) error {
	if len(issuers) > MaxReasoningIssuers {
		return errors.New("reasoning issuer count exceeds limit")
	}
	for index, issuer := range issuers {
		if !reasoningIssuerPattern.MatchString(issuer) {
			return errors.New("reasoning issuer is invalid")
		}
		if slices.Contains(issuers[:index], issuer) {
			return errors.New("reasoning issuer is duplicated")
		}
	}
	return nil
}
