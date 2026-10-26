package execution

import (
	"bytes"
	"fmt"
	"log/slog"
	"math"

	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"
)

// A local target without a registered runtime counter gets a capacity fact
// from the exact prepared request, so an over-window turn is handed off before
// the runtime can silently truncate it:
//
//   - Qwen3.8-27B (registry_model qwen/qwen3.8-27b, Chat wire): the request
//     is rendered with the model's chat template and counted with the
//     embedded tokenizer (qwen_capacity.go), provenance local_measured.
//     A segment the counter can only bound by bytes (text x/text cannot
//     normalize as the reference does) makes the fact caller_estimated.
//   - Any other model: the UTF-8 byte count of a worst-case re-serialization
//     of the prepared request, a proven upper bound for a byte-level BPE
//     tokenizer (every token covers at least one byte) whatever a template
//     does to the request's JSON: values re-encoded with indent=4, Python
//     separators, floats by repr, non-ASCII and <, >, & escaped, and each
//     tool call's arguments string reloaded and re-encoded the same way
//     beside it. It is raised by headroom_ratio and
//     estimateTemplateMarginTokens for a preamble the body does not carry;
//     provenance caller_estimated.
//
// An inline base64 image is charged media_tokens instead of its bytes in both
// cases, and a fact with images is caller_estimated: the runtime's image
// token count depends on the resolution.
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
	qwenTokenUnits               = "hf:Qwen/Qwen3.8-27B"
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

// localPromptSize sizes a prepared request for a local target without a
// registered counter: counted for Qwen3.8-27B, otherwise bounded by bytes.
func localPromptSize(deployment Deployment, body []byte) (int64, string, string) {
	_, mediaTokens := deployment.CapacityEstimate.settings()
	units := deployment.TokenUnits
	if isQwen38Deployment(deployment) {
		tokens, images, exact, err := qwenPromptTokens(body)
		if err == nil {
			provenance := capacityProvenanceEstimated
			if images == 0 && exact {
				provenance = capacityProvenanceMeasured
			}
			if units == "" {
				units = qwenTokenUnits
			}
			return tokens + images*mediaTokens, provenance, units
		}
		slog.Warn("qwen capacity count failed; using the byte bound", "deployment", deployment.ID, "error", err.Error())
	}
	return byteBoundTokens(body, deployment.CapacityEstimate), capacityProvenanceEstimated, units
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
func byteBoundTokens(body []byte, estimate *CapacityEstimate) int64 {
	headroom, mediaTokens := estimate.settings()
	text := worstCaseBytes(body)
	blobs := inlineImages(body)
	for _, blob := range blobs {
		text -= int64(blob[1] - blob[0])
	}
	return int64(math.Ceil(float64(max(text, 0))*(1+headroom))) + int64(len(blobs))*mediaTokens + estimateTemplateMarginTokens
}

// worstCaseBytes is the longest spelling a template can give the request's
// JSON (thinkingcontrol.PythonWorstCaseBytes, one parse), with each tool
// call's arguments string reloaded and re-encoded beside it. A body that does
// not parse is charged three times its bytes.
func worstCaseBytes(body []byte) int64 {
	size, err := thinkingcontrol.PythonWorstCaseBytes(body, "arguments")
	if err != nil {
		return 3 * int64(len(body))
	}
	return size
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
