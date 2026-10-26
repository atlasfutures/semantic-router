package execution

import (
	"bytes"
	"context"
	"fmt"
	"log/slog"
	"math"
	"sort"

	"github.com/atlasfutures/semantic-router/rayline-core/cachecontext"
	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

const (
	CapacityContext            = "context_capacity"
	CapacityRuntimeUnavailable = "runtime_unavailable"

	// Allocation is stated as configured until a runtime adapter observes it.
	// llama.cpp's per-slot n_ctx is not yet observed, so nothing here is ever
	// labelled runtime_observed.
	capacitySourceConfigured = "configured"

	// A fact counted by the target's own tokenizer, versus one estimated from
	// the prepared request's bytes. The policy contract (pathfinder
	// arc_execution_v2.decide_inputs_v2) admits an estimated fits:false as
	// exclusion and reads an estimated fits:true as unknown.
	capacityProvenanceMeasured  = "local_measured"
	capacityProvenanceEstimated = "caller_estimated"

	// CodeContextExceedsAllDeployments refuses a turn whose prompt fits no
	// admitted deployment, before any runtime is contacted.
	CodeContextExceedsAllDeployments = "context_exceeds_all_deployments"
)

// Estimator defaults, overridable per deployment with capacity_estimate.
//
// BPE tokenizers of the served families (Qwen, Claude, Kimi) average about
// 3.5-4.5 bytes per token on English prose and 3-3.5 on code, and the encoded
// JSON request carries escapes and structure the chat template does not
// tokenize as text, so bytes/3 over-counts typical agent traffic. The 15%
// headroom covers denser content (Qwen splits digits one per token, so logs
// and tables tokenize near 2 bytes per token) and template tokens. The
// estimate only has to err high: an over-count costs an early handoff to a
// larger window; an under-count lets Ollama or MLX silently truncate.
const (
	defaultEstimateBytesPerToken = 3.0
	defaultEstimateHeadroomRatio = 0.15
	// Inline base64 media is not tokenized as text. A 1080p screenshot is
	// about 1.6k tokens for Claude and 2.7k for Qwen-VL at 28px patches.
	defaultEstimateMediaTokens = 4096
	// A shorter base64 run is counted as text.
	mediaBlobMinBytes = 1024
)

// CapacityFact is this exact request's verdict on one target action. Nil means
// unknown: never zero and never "fits".
type CapacityFact struct {
	Fits            bool    `json:"fits"`
	Reason          *string `json:"reason"`
	Required        *int64  `json:"required"`
	Available       *int64  `json:"available"`
	AvailableSource *string `json:"available_source"`
	Units           *string `json:"units"`
	Provenance      string  `json:"provenance"`
	// prompt is the prompt's own size within Required, for refusal messages.
	prompt int64
}

// capacityFact compares the exact prepared prompt plus the actual output limit
// and configured margin with the deployment's configured context, in the
// target's own tokenizer units. Windows are never compared across targets.
func capacityFact(deployment Deployment, prompt *cachecontext.MeasuredPrompt, outputLimit int64) *CapacityFact {
	if prompt == nil || deployment.ContextWindow <= 0 || prompt.Units() == "" {
		return nil
	}
	required := int64(prompt.PromptTokens()) + outputLimit + deployment.CapacityMarginTokens
	available := deployment.ContextWindow
	source, units := capacitySourceConfigured, prompt.Units()
	fact := &CapacityFact{Fits: required <= available, Required: &required, Available: &available, AvailableSource: &source, Units: &units, Provenance: capacityProvenanceMeasured, prompt: int64(prompt.PromptTokens())}
	if !fact.Fits {
		reason := CapacityContext
		fact.Reason = &reason
	}
	return fact
}

// runtimeContextRejection is the runtime's own refusal before any output. It
// is admitted as fits:false whatever the allocation turns out to be.
func runtimeContextRejection(deployment Deployment, previous *CapacityFact) *CapacityFact {
	reason := CapacityContext
	fact := &CapacityFact{Fits: false, Reason: &reason, Provenance: "provider_reported"}
	if previous != nil {
		fact.Required, fact.Units, fact.prompt = previous.Required, previous.Units, previous.prompt
	} else if deployment.TokenUnits != "" {
		units := deployment.TokenUnits
		fact.Units = &units
	}
	return fact
}

// estimatedCapacityFact is capacityFact for a target without a counter: the
// prompt is estimated from the exact prepared request bytes, never measured.
func estimatedCapacityFact(deployment Deployment, body []byte, outputLimit int64) *CapacityFact {
	if deployment.ContextWindow <= 0 {
		return nil
	}
	prompt := estimatePromptTokens(body, deployment.CapacityEstimate)
	required := prompt + outputLimit + deployment.CapacityMarginTokens
	available := deployment.ContextWindow
	source := capacitySourceConfigured
	fact := &CapacityFact{Fits: required <= available, Required: &required, Available: &available, AvailableSource: &source, Provenance: capacityProvenanceEstimated, prompt: prompt}
	if deployment.TokenUnits != "" {
		units := deployment.TokenUnits
		fact.Units = &units
	}
	if !fact.Fits {
		reason := CapacityContext
		fact.Reason = &reason
	}
	return fact
}

func (estimate *CapacityEstimate) settings() (bytesPerToken, headroom float64, mediaTokens int64) {
	bytesPerToken, headroom, mediaTokens = defaultEstimateBytesPerToken, defaultEstimateHeadroomRatio, defaultEstimateMediaTokens
	if estimate == nil {
		return
	}
	if estimate.BytesPerToken != nil {
		bytesPerToken = *estimate.BytesPerToken
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
	bytesPerToken, headroom, mediaTokens := estimate.settings()
	if !(bytesPerToken >= 0.5 && bytesPerToken <= 16) || !(headroom >= 0 && headroom <= 10) || mediaTokens < 0 {
		return fmt.Errorf("capacity_estimate needs bytes_per_token in [0.5, 16], headroom_ratio in [0, 10] and nonnegative media_tokens")
	}
	return nil
}

// estimatePromptTokens is ceil((text bytes / bytes_per_token + media blobs *
// media_tokens) * (1 + headroom_ratio)) over the prepared request body.
func estimatePromptTokens(body []byte, estimate *CapacityEstimate) int64 {
	bytesPerToken, headroom, mediaTokens := estimate.settings()
	mediaBytes, blobs := inlineMediaBytes(body)
	tokens := (float64(int64(len(body))-mediaBytes)/bytesPerToken + float64(blobs*mediaTokens)) * (1 + headroom)
	return int64(math.Ceil(tokens))
}

// inlineMediaBytes finds base64 images that are JSON structure, not text: an
// Anthropic image source "data" member and a "data:image/...;base64," URL
// value. The same bytes quoted inside a string (a tool result) are escaped and
// stay text, and other media (a PDF) keeps the conservative byte count.
func inlineMediaBytes(body []byte) (total, count int64) {
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
			total += int64(run)
			count++
		}
	}
	return total, count
}

func base64Run(body []byte) int {
	for index, char := range body {
		if !(char >= 'A' && char <= 'Z' || char >= 'a' && char <= 'z' || char >= '0' && char <= '9' || char == '+' || char == '/' || char == '=' || char == '\\' && index+1 < len(body) && body[index+1] == '/') {
			return index
		}
	}
	return len(body)
}

// measureCapacity prepares every action exactly as dispatch would. A target
// with a qualified counter is counted with its own runtime tokenizer
// (local_measured); any other is estimated from the prepared bytes
// (caller_estimated). Nothing is committed.
func (server *Server) measureCapacity(ctx context.Context, body []byte, format llmprotocol.WireFormat, state *branchState) (map[string]*CapacityFact, map[string]*cachecontext.MeasuredPrompt, error) {
	facts := make(map[string]*CapacityFact)
	measured := make(map[string]*cachecontext.MeasuredPrompt)
	ids := make([]string, 0, len(server.actions))
	for id := range server.actions {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		selected := server.actions[id]
		if selected.deployment.ID == "" {
			continue
		}
		request, envelope, _, err := server.engine.DecodeRequestForMutation(format, body)
		if err != nil {
			return nil, nil, err
		}
		prepared, err := server.prepare(request, envelope, selected, state)
		if err != nil {
			// This action cannot carry the request at all (for example an
			// output allowance above its limit); that is not a context fact.
			continue
		}
		if server.counters[selected.deployment.ID] == nil {
			facts[id] = estimatedCapacityFact(selected.deployment, prepared.body, prepared.outputLimit)
			continue
		}
		prompt, err := server.measurePreparedPrompt(ctx, selected, prepared.body, state, nil)
		if err != nil {
			// A runtime that cannot count is a fact about that target only:
			// it is unavailable, never a reason to refuse the whole turn.
			slog.Warn("capacity measurement failed", "action_id", id, "deployment", selected.deployment.ID, "error", err.Error())
			reason := CapacityRuntimeUnavailable
			facts[id] = &CapacityFact{Fits: false, Reason: &reason, Provenance: capacityProvenanceMeasured}
			continue
		}
		measured[id] = prompt
		facts[id] = capacityFact(selected.deployment, prompt, prepared.outputLimit)
	}
	return facts, measured, nil
}

// isRuntimeContextRejection recognizes llama.cpp's documented refusal of a
// prompt that exceeds its slot context. It applies only to local targets and
// only before any output was delivered.
func isRuntimeContextRejection(deployment Deployment, status int, body []byte) bool {
	if deployment.Kind != "local" || status != 400 {
		return false
	}
	for _, marker := range [][]byte{[]byte("exceed_context_size_error"), []byte("exceeds the available context size")} {
		if bytes.Contains(body, marker) {
			return true
		}
	}
	return false
}

func int64OrNil(value *int64) any {
	if value == nil {
		return nil
	}
	return *value
}

// contextExceededMessage explains a turn no deployment can hold, from the
// largest window among the targets reported not to fit for context. It is
// empty unless such a fact exists and no fact says a target fits.
func (server *Server) contextExceededMessage(facts map[string]*CapacityFact) string {
	ids := make([]string, 0, len(facts))
	for id, fact := range facts {
		if fact != nil && fact.Fits {
			return ""
		}
		ids = append(ids, id)
	}
	sort.Strings(ids)
	var largest *CapacityFact
	model := ""
	for _, id := range ids {
		fact := facts[id]
		if fact == nil || fact.Fits || fact.Reason == nil || *fact.Reason != CapacityContext || fact.Available == nil || fact.Required == nil {
			continue
		}
		if largest == nil || *fact.Available > *largest.Available {
			largest, model = fact, server.actions[id].action.Model
		}
	}
	if largest == nil {
		return ""
	}
	basis := "measured"
	if largest.Provenance == capacityProvenanceEstimated {
		basis = "estimated"
	}
	size := fmt.Sprintf("needs about %d tokens", *largest.Required)
	if largest.prompt > 0 {
		size = fmt.Sprintf("prompt is about %d tokens (%s); with the output allowance it needs %d tokens", largest.prompt, basis, *largest.Required)
	}
	return fmt.Sprintf("this conversation's %s, more than any available deployment's context window: the largest is %d tokens (%s). Compact or shorten the conversation, then retry.", size, *largest.Available, model)
}

// exceedsAllDeployments reports whether every executable action has a known
// context shortfall for this request, so nothing can be dispatched.
func (server *Server) exceedsAllDeployments(facts map[string]*CapacityFact) bool {
	executable := 0
	for id, selected := range server.actions {
		if selected.deployment.ID == "" {
			continue
		}
		executable++
		fact := facts[id]
		if fact == nil || fact.Fits || fact.Reason == nil || *fact.Reason != CapacityContext {
			return false
		}
	}
	return executable > 0
}
