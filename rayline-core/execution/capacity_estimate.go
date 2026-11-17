package execution

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"

	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"
)

// A local target without a registered runtime counter gets a capacity fact
// from the exact prepared request, so an over-window turn is handed off before
// the runtime can silently truncate it: the UTF-8 byte count of a worst-case
// re-serialization of the prepared request, an upper bound for a byte-level
// BPE tokenizer (every token covers at least one byte) provided the template
// renders each part of the request at most once and adds at most the margin
// below (README): values re-encoded with indent=4, Python separators, floats
// by repr, non-ASCII and <, >, & escaped, and each tool call's arguments
// string reloaded and re-encoded the same way beside it. It is raised by
// headroom_ratio and estimateTemplateMarginTokens for a preamble the body
// does not carry; provenance caller_bounded, which the policy contract admits
// as a fit (a bound that fits holds for the exact count too).
//
// An inline base64 image is charged media_tokens instead of its bytes, and a
// fact with images is caller_estimated: the runtime's image token count
// depends on the resolution. So is any byte bound over a body the core does
// not fully understand (boundedBody, capacity_bounded.go).
const (
	defaultEstimateHeadroomRatio = 0.05
	// A Retina screenshot is about 6.6k Qwen-VL tokens; Claude caps an image
	// near 1.6k.
	defaultEstimateMediaTokens = 8192
	// A shorter base64 run is counted as text.
	mediaBlobMinBytes = 1024
	// The longest system preamble a served template adds (Qwen3.8's tools
	// instructions are about 330 tokens).
	estimateTemplateMarginTokens = 512
)

func (estimate *CapacityEstimate) settings() (headroom float64, mediaTokens int64) {
	headroom, mediaTokens = defaultEstimateHeadroomRatio, defaultEstimateMediaTokens
	if estimate == nil {
		return
	}
	if estimate.HeadroomRatio != nil {
		headroom = *estimate.HeadroomRatio
	}
	if estimate.MediaTokens != nil {
		mediaTokens = *estimate.MediaTokens
	}
	return
}

func (estimate *CapacityEstimate) validate() error {
	headroom, mediaTokens := estimate.settings()
	if !(headroom >= 0 && headroom <= 10) || mediaTokens < 0 {
		return fmt.Errorf("capacity_estimate needs headroom_ratio in [0, 10] and nonnegative media_tokens")
	}
	return nil
}

// localPromptSize bounds a prepared request by bytes for a local target
// without a registered counter. An estimated size comes with what made it
// one: the charged images, or the first member unboundedMember reports.
func localPromptSize(deployment Deployment, body []byte) (tokens int64, provenance, units, estimatedBy string) {
	units = deployment.TokenUnits
	tokens, images := byteBoundTokens(body, deployment.CapacityEstimate)
	if images > 0 {
		return tokens, capacityProvenanceEstimated, units, fmt.Sprintf("inline images: %d", images)
	}
	if member := unboundedMember(deployment.Protocol, body); member != "" {
		return tokens, capacityProvenanceEstimated, units, member
	}
	return tokens, capacityProvenanceBounded, units, ""
}

func sizedCapacityFact(deployment Deployment, prompt int64, provenance, units string, outputLimit int64) *CapacityFact {
	required := prompt + outputLimit + deployment.CapacityMarginTokens
	available := deployment.ContextWindow
	source := capacitySourceConfigured
	fact := &CapacityFact{Fits: required <= available, Required: &required, Available: &available, AvailableSource: &source, Provenance: provenance, prompt: prompt}
	if units != "" {
		fact.Units = &units
	}
	if !fact.Fits {
		reason := CapacityContext
		fact.Reason = &reason
	}
	return fact
}

// byteBoundTokens is ceil(worst-case bytes * (1 + headroom_ratio)) + images *
// media_tokens + estimateTemplateMarginTokens over the prepared body.
// byteBoundTokens returns the bound and the number of inline images charged
// media_tokens; with none, the bound is an upper bound on the exact count.
func byteBoundTokens(body []byte, estimate *CapacityEstimate) (int64, int) {
	headroom, mediaTokens := estimate.settings()
	text := worstCaseBytes(body) + namespaceFlatteningBytes(body)
	blobs := inlineImages(body)
	for _, blob := range blobs {
		text -= int64(blob[1] - blob[0])
	}
	return int64(math.Ceil(float64(max(text, 0))*(1+headroom))) + int64(len(blobs))*mediaTokens + estimateTemplateMarginTokens, len(blobs)
}

// worstCaseBytes is the longest spelling a template can give the request's
// JSON (thinkingcontrol.PythonWorstCaseBytes, one parse), with each tool
// call's arguments string reloaded and re-encoded beside it, plus what a
// tokenizer's Unicode normalizer can add. A body that does not parse is
// charged three times its bytes.
func worstCaseBytes(body []byte) int64 {
	size, err := thinkingcontrol.PythonWorstCaseBytes(body, "arguments")
	if err != nil {
		return 3 * int64(len(body))
	}
	return size
}

// namespaceFlatteningBytes is what a runtime can add by flattening each
// Responses namespace tool into one function per member. Runtimes repeat the
// namespace's name in every member's name in different spellings: vLLM
// <namespace>__<name>, Ollama <namespace>.<name>, llama.cpp
// ns_<namespace>___<name>. A runtime that flattens as the core does for Chat
// and Messages (llmprotocol.FlattenedToolDescription) also puts the
// namespace's description and a blank line before each member's own. Each
// member is charged the longest of these, "ns_" + name + "___" + description
// + "\n\n", in its worst-case spelling, so the bound does not rely on how
// a particular runtime flattens.
func namespaceFlatteningBytes(body []byte) int64 {
	if !bytes.Contains(body, []byte(`"namespace"`)) {
		return 0
	}
	var request struct {
		Tools []map[string]any `json:"tools"`
	}
	if json.Unmarshal(body, &request) != nil {
		return 0
	}
	var total int64
	for _, tool := range request.Tools {
		members, _ := tool["tools"].([]any)
		name, _ := tool["name"].(string)
		if tool["type"] != "namespace" || len(members) == 0 {
			continue
		}
		description, _ := tool["description"].(string)
		prefix, _ := json.Marshal("ns_" + name + "___" + description + "\n\n")
		total += int64(len(members)) * worstCaseBytes(prefix)
	}
	return total
}

// inlineImages finds base64 images that are JSON structure, not text: an
// Anthropic image source "data" member and a "data:image/...;base64," URL
// value, as [start, end) byte ranges in order. The same bytes quoted inside a
// string (a tool result) are escaped and stay text, and other media (a PDF)
// is charged as text, which over-counts it.
func inlineImages(body []byte) [][2]int {
	var blobs [][2]int
	for _, marker := range [][]byte{[]byte(`"data":"`), []byte(";base64,")} {
		for offset := 0; ; {
			index := bytes.Index(body[offset:], marker)
			if index < 0 {
				break
			}
			at := offset + index
			start := at + len(marker)
			run := base64Run(body[start:])
			offset = start + run
			if run < mediaBlobMinBytes {
				continue
			}
			if marker[0] == ';' {
				quote := bytes.LastIndexByte(body[:at], '"')
				if quote <= 0 || body[quote-1] == '\\' || !bytes.HasPrefix(body[quote:], []byte(`"data:image/`)) {
					continue
				}
			} else if !bytes.Contains(body[max(0, at-256):at], []byte(`"media_type":"image/`)) {
				continue
			}
			blobs = append(blobs, [2]int{start, start + run})
		}
	}
	// The two markers never overlap; order the ranges for one pass.
	for i := 1; i < len(blobs); i++ {
		for j := i; j > 0 && blobs[j][0] < blobs[j-1][0]; j-- {
			blobs[j], blobs[j-1] = blobs[j-1], blobs[j]
		}
	}
	return blobs
}

func base64Run(body []byte) int {
	for index, char := range body {
		if !(char >= 'A' && char <= 'Z' || char >= 'a' && char <= 'z' || char >= '0' && char <= '9' || char == '+' || char == '/' || char == '=' || char == '\\' && index+1 < len(body) && body[index+1] == '/') {
			return index
		}
	}
	return len(body)
}
