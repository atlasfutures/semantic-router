package execution

import (
	"bytes"
	"fmt"
	"log/slog"
	"math"
)

// A local target without a registered runtime counter gets a capacity fact
// from the exact prepared request, so an over-window turn is handed off before
// the runtime can silently truncate it:
//
//   - Qwen3.8-27B (registry_model qwen/qwen3.8-27b, Chat wire): the request
//     is rendered with the model's chat template and counted with the
//     embedded tokenizer (qwen_capacity.go), provenance local_measured.
//   - Any other model: the UTF-8 byte count of the prepared request, a proven
//     upper bound for a byte-level BPE tokenizer (every token covers at
//     least one byte; the JSON framing of each message exceeds its template
//     tokens), raised by headroom_ratio and estimateTemplateMarginTokens for a
//     template preamble the body does not carry; provenance caller_estimated.
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

// estimatedCapacityFact is capacityFact for a local target without a
// registered counter: counted for Qwen3.8-27B, otherwise bounded by bytes.
func estimatedCapacityFact(deployment Deployment, body []byte, outputLimit int64) *CapacityFact {
	if deployment.ContextWindow <= 0 {
		return nil
	}
	_, mediaTokens := deployment.CapacityEstimate.settings()
	provenance, units := capacityProvenanceEstimated, deployment.TokenUnits
	var prompt int64
	counted := false
	if isQwen38Deployment(deployment) {
		tokens, images, err := qwenPromptTokens(body)
		if err == nil {
			prompt, counted = tokens+images*mediaTokens, true
			if images == 0 {
				provenance = capacityProvenanceMeasured
			}
			if units == "" {
				units = qwenTokenUnits
			}
		} else {
			slog.Warn("qwen capacity count failed; using the byte bound", "deployment", deployment.ID, "error", err.Error())
		}
	}
	if !counted {
		prompt = byteBoundTokens(body, deployment.CapacityEstimate)
	}
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

// byteBoundTokens is ceil(text bytes * (1 + headroom_ratio)) + images *
// media_tokens + estimateTemplateMarginTokens over the prepared body.
func byteBoundTokens(body []byte, estimate *CapacityEstimate) int64 {
	headroom, mediaTokens := estimate.settings()
	text := int64(len(body))
	blobs := inlineImages(body)
	for _, blob := range blobs {
		text -= int64(blob[1] - blob[0])
	}
	return int64(math.Ceil(float64(text)*(1+headroom))) + int64(len(blobs))*mediaTokens + estimateTemplateMarginTokens
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
