package reasoning

// A thinking run is the stretch of a conversation the Router served, turn
// after turn, on one Claude model. Claude's thinking signature verifies only
// on the model (and the provider family) that signed it, so a Claude-bound
// request keeps the thinking of the turns in the current run and strips the
// rest: thinking another model signed, thinking signed before the
// conversation reached the Router (a client's own key), and thinking from a
// run the episode no longer records.
//
// The record is the run's anchor: the digest of the first thinking signature
// the Router served in it, kept in the episode's reasoning provenance under
// the run's worker (ThinkingRunWorker). Every assistant message from the one
// holding the anchor to the end of the history is the run's, so the kept
// stretch is a suffix of the history that only grows while the run lasts: a
// block's fate does not change from turn to turn, and the prompt cache keeps
// matching. A turn on another Claude model ends the run (WithThinkingRun
// drops its anchors), and a history that no longer holds the anchor (a
// compaction, a client rewrite) starts a new one.

// MaxThinkingRunAnchors bounds the anchors one run worker keeps: one per
// transcript the episode serves on that model (a client that sends two
// transcripts under one episode re-anchors on each), the most recent kept.
const MaxThinkingRunAnchors = 4

const (
	thinkingRunPrefix = "thinking-run\x00"
	// thinkingRunAnchorPrefix keeps an anchor's block digest apart from an
	// opaque block's: an anchor at a redacted_thinking block must never read
	// as that block's issuer record (ReasoningIssuedElsewhere).
	thinkingRunAnchorPrefix = "thinking-run-anchor\x00"
)

// ThinkingRunAnchorDigest is the block digest an anchor at key (a thinking
// signature, or a redacted_thinking block's data) is recorded under.
func ThinkingRunAnchorDigest(key string) string {
	return ReasoningBlockDigest(thinkingRunAnchorPrefix + key)
}

// ThinkingRunWorker names the provenance worker of a thinking run on a
// Claude target: the thinking issuer, in a namespace of its own so a run's
// entries never read as an opaque block's.
func ThinkingRunWorker(issuer string) string {
	return thinkingRunPrefix + issuer
}

// ThinkingRunAnchors lists the block digests recorded as anchors of worker's
// run, oldest first.
func ThinkingRunAnchors(provenance []ReasoningProvenance, worker string) []string {
	digest := ReasoningWorkerDigest(worker)
	var anchors []string
	for _, entry := range provenance {
		if entry.Worker == digest {
			anchors = append(anchors, entry.Block)
		}
	}
	return anchors
}

// WithThinkingRun records a committed turn served by worker's run. Every
// anchor of the others (the run workers of the cell's other Claude targets)
// is dropped: their runs ended with this turn. worker's own anchors are kept
// and moved to the newest end, so opaque blocks recorded meanwhile do not
// push them out; when the turn's history held none of them (anchored false)
// and the turn signed thinking, its first signature becomes a new anchor.
// The whole list keeps the MaxReasoningProvenance bound.
func WithThinkingRun(
	provenance []ReasoningProvenance,
	worker string,
	others []string,
	anchored bool,
	signature string,
) []ReasoningProvenance {
	own := ReasoningWorkerDigest(worker)
	ended := make(map[string]bool, len(others))
	for _, other := range others {
		if digest := ReasoningWorkerDigest(other); digest != own {
			ended[digest] = true
		}
	}
	next := make([]ReasoningProvenance, 0, len(provenance)+1)
	var anchors []ReasoningProvenance
	for _, entry := range provenance {
		switch {
		case ended[entry.Worker]:
		case entry.Worker == own:
			anchors = append(anchors, entry)
		default:
			next = append(next, entry)
		}
	}
	if !anchored && signature != "" {
		block := ThinkingRunAnchorDigest(signature)
		kept := anchors[:0]
		for _, anchor := range anchors {
			if anchor.Block != block {
				kept = append(kept, anchor)
			}
		}
		anchors = append(kept, ReasoningProvenance{Block: block, Worker: own})
	}
	if len(anchors) > MaxThinkingRunAnchors {
		anchors = anchors[len(anchors)-MaxThinkingRunAnchors:]
	}
	next = append(next, anchors...)
	if len(next) > MaxReasoningProvenance {
		next = next[len(next)-MaxReasoningProvenance:]
	}
	return next
}

// WithoutThinkingRun drops every anchor of worker's run: a turn whose call
// was refused for a signature and re-sent with the run's thinking stripped
// starts the run again.
func WithoutThinkingRun(provenance []ReasoningProvenance, worker string) []ReasoningProvenance {
	digest := ReasoningWorkerDigest(worker)
	next := make([]ReasoningProvenance, 0, len(provenance))
	for _, entry := range provenance {
		if entry.Worker != digest {
			next = append(next, entry)
		}
	}
	return next
}

// SplitThinkingRun separates worker's anchors from the rest of the
// provenance, so opaque blocks recorded with a turn are trimmed against the
// rest alone and never push the live run's anchors out.
func SplitThinkingRun(provenance []ReasoningProvenance, worker string) (own, rest []ReasoningProvenance) {
	digest := ReasoningWorkerDigest(worker)
	for _, entry := range provenance {
		if entry.Worker == digest {
			own = append(own, entry)
		} else {
			rest = append(rest, entry)
		}
	}
	return own, rest
}
