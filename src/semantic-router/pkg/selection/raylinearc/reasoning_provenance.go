package raylinearc

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"regexp"
)

var reasoningProvenancePattern = regexp.MustCompile(`^[0-9a-f]{16}$`)

// MaxReasoningProvenance bounds the opaque reasoning blocks an episode
// remembers the issuer of; older blocks fall off and are treated as unknown.
const MaxReasoningProvenance = 32

// ReasoningProvenance names the worker that produced one opaque reasoning
// block (an Anthropic-format redacted_thinking). The block is readable only by
// the model that issued it: resent to another worker it fails the turn, so a
// later turn drops the blocks whose issuer is known to be someone else. Both
// fields are truncated digests, so the record holds neither reasoning nor
// configuration names.
type ReasoningProvenance struct {
	Block  string `json:"block"`
	Worker string `json:"worker"`
}

// ReasoningBlockDigest names an opaque reasoning block by its data.
func ReasoningBlockDigest(data string) string {
	return shortDigest(data)
}

// ReasoningWorkerDigest names the worker a block was issued by.
func ReasoningWorkerDigest(worker string) string {
	return shortDigest(worker)
}

func shortDigest(value string) string {
	digest := sha256.Sum256([]byte(value))
	return hex.EncodeToString(digest[:8])
}

// WithReasoningProvenance records that worker issued blocks, newest last,
// keeping each block once and at most MaxReasoningProvenance in all.
func WithReasoningProvenance(existing []ReasoningProvenance, worker string, blocks []string) []ReasoningProvenance {
	if len(blocks) == 0 {
		return existing
	}
	workerDigest := ReasoningWorkerDigest(worker)
	next := make([]ReasoningProvenance, 0, len(existing)+len(blocks))
	fresh := make(map[string]bool, len(blocks))
	for _, data := range blocks {
		fresh[ReasoningBlockDigest(data)] = true
	}
	for _, entry := range existing {
		if !fresh[entry.Block] {
			next = append(next, entry)
		}
	}
	for _, data := range blocks {
		block := ReasoningBlockDigest(data)
		if !fresh[block] {
			continue
		}
		delete(fresh, block)
		next = append(next, ReasoningProvenance{Block: block, Worker: workerDigest})
	}
	if len(next) > MaxReasoningProvenance {
		next = next[len(next)-MaxReasoningProvenance:]
	}
	return next
}

// ReasoningIssuedElsewhere reports whether an opaque reasoning block is known
// not to be readable by target: it was issued by another issuer, or by one
// whose serving provider could vary (ReasoningIssuerUnknown), or target's own
// provider could vary. A block the episode has no record of is not: it is
// left as the client sent it.
func ReasoningIssuedElsewhere(provenance []ReasoningProvenance, data, target string) bool {
	block := ReasoningBlockDigest(data)
	unknown := ReasoningWorkerDigest(ReasoningIssuerUnknown)
	for index := len(provenance) - 1; index >= 0; index-- {
		if provenance[index].Block == block {
			issuer := provenance[index].Worker
			return issuer == unknown || target == ReasoningIssuerUnknown || issuer != ReasoningWorkerDigest(target)
		}
	}
	return false
}

func validateReasoningProvenance(provenance []ReasoningProvenance) error {
	if len(provenance) > MaxReasoningProvenance {
		return errors.New("reasoning provenance exceeds its bound")
	}
	for _, entry := range provenance {
		if !reasoningProvenancePattern.MatchString(entry.Block) || !reasoningProvenancePattern.MatchString(entry.Worker) {
			return errors.New("reasoning provenance is malformed")
		}
	}
	return nil
}
