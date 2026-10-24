package execution

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/reasoning"
	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"
)

// branchState is serialized for the entire decision/dispatch/settlement. Only
// completed executions replace placements and cache observations. A failed
// dispatch keeps its decision for an identical retry, but never marks it warm.
type branchState struct {
	lease      chan struct{}
	epoch      uint64
	completed  uint64
	held       string
	heldWorker string
	placements map[string]thinkingcontrol.PlacerState
	workers    map[string]WorkerEvidence
	prefix     []string
	pending    *pendingTurn
	issuers    []string
	provenance []reasoning.ReasoningProvenance
	ledger     []attributedTurn
	// wire fingerprints the last committed request's wire messages, so an
	// explicit compaction can carry retained attribution to new positions.
	wire []string
	// carried keeps the last committed transcript's ledger and wire across an
	// epoch reset until the next commit, so a failed compaction request can be
	// retried without losing retained attribution.
	carried *carriedTranscript
	// explicitEvents records that this branch's harness adapter states events.
	// A later request without one (such as Pi's compaction summary call) is
	// then a side call, never an advance of the main transcript.
	explicitEvents bool
	// seen fingerprints the wire messages of the last non-sidecall request
	// this branch saw, committed or not: a compaction is detected against it.
	seen []string
}

type carriedTranscript struct {
	ledger []attributedTurn
	wire   []string
}

// attributedTurn uses the original client wire's message index, as VSR's
// policy ledger does. Neutral messages can split a tool-result wire message.
type attributedTurn struct {
	format llmprotocol.WireFormat
	entry  Attribution
}

type pendingTurn struct {
	fingerprint string
	requestID   string
	decision    Decision
	delivered   bool
	completed   bool
	// input is the turn's original decision input; a re-consult keeps its
	// event and history and narrows only the stated capacity fact.
	input    DecisionInput
	consults int
}

type stateStore struct {
	mu       sync.Mutex
	branches map[string]*branchState
}

func (store *stateStore) acquire(ctx context.Context, session, branch string) (*branchState, error) {
	encoded, err := json.Marshal([]string{session, branch})
	if err != nil {
		return nil, err
	}
	key := string(encoded)
	store.mu.Lock()
	state := store.branches[key]
	if state == nil {
		// Evicting a live hidden-instruction ledger would silently change the
		// provider transcript. At capacity reject new sessions explicitly.
		if len(store.branches) >= 256 {
			store.mu.Unlock()
			return nil, fmt.Errorf("execution session capacity reached; restart with fresh sessions")
		}
		state = newBranchState()
		store.branches[key] = state
	}
	store.mu.Unlock()
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-state.lease:
		return state, nil
	}
}

func newBranchState() *branchState {
	state := &branchState{lease: make(chan struct{}, 1), placements: make(map[string]thinkingcontrol.PlacerState), workers: make(map[string]WorkerEvidence)}
	state.lease <- struct{}{}
	return state
}

// hasOtherBranch reports whether this session already has another branch: a
// new branch of it with no assistant turn is a child session or subagent.
func (store *stateStore) hasOtherBranch(session, branch string) bool {
	store.mu.Lock()
	defer store.mu.Unlock()
	for key := range store.branches {
		var identity []string
		if json.Unmarshal([]byte(key), &identity) == nil && len(identity) == 2 && identity[0] == session && identity[1] != branch {
			return true
		}
	}
	return false
}

func (state *branchState) release() { state.lease <- struct{}{} }

func (state *branchState) evidence(instance string) ExecutionContext {
	result := ExecutionContext{InstanceID: instance, ContextEpoch: instance + ":" + strconv.FormatUint(state.epoch, 10), HeldActionID: state.held, HeldWorker: state.heldWorker, CompletedTurns: state.completed, Workers: make([]WorkerEvidence, 0, len(state.workers))}
	for _, worker := range state.workers {
		result.Workers = append(result.Workers, worker)
	}
	return result
}

func (state *branchState) evidenceForRequest(instance string, format llmprotocol.WireFormat, roles []string) ExecutionContext {
	evidence := state.evidence(instance)
	for _, turn := range state.ledger {
		if turn.format == format && turn.entry.Message < len(roles) && roles[turn.entry.Message] == "assistant" {
			evidence.Attribution = append(evidence.Attribution, turn.entry)
		}
	}
	return evidence
}

// The exact index is len(the original wire array), then a later assistant-item
// check at that index. Responses attributes the first item of the output run.
// A string input has no known array index and remains explicitly unattributed.
func (state *branchState) recordAttribution(format llmprotocol.WireFormat, roles []string, decision Decision) {
	if roles == nil {
		return
	}
	routeID, result := decision.RouteID, decision.Result
	entry := Attribution{Message: len(roles), ActionID: decision.ActionID, ArmID: decision.ArmID,
		Worker: decision.Worker, RouteID: &routeID, Result: &result}
	state.ledger = append(state.ledger, attributedTurn{format: format, entry: entry})
	const maxAttributions = 256 // VSR MaxPolicyLedgerEntries
	if len(state.ledger) > maxAttributions {
		state.ledger = state.ledger[len(state.ledger)-maxAttributions:]
	}
}

func wireMessageRoles(format llmprotocol.WireFormat, body []byte) ([]string, error) {
	if format == llmprotocol.OpenAIResponsesV1 {
		var wire struct {
			Input json.RawMessage `json:"input"`
		}
		if err := json.Unmarshal(body, &wire); err != nil {
			return nil, err
		}
		var items []struct {
			Type string `json:"type"`
			Role string `json:"role"`
		}
		if len(wire.Input) == 0 || wire.Input[0] != '[' {
			return nil, nil
		}
		if err := json.Unmarshal(wire.Input, &items); err != nil {
			return nil, err
		}
		roles := make([]string, len(items))
		for index, item := range items {
			switch item.Type {
			case "function_call", "reasoning":
				roles[index] = "assistant"
			case "message", "":
				roles[index] = item.Role
			case "function_call_output":
				roles[index] = "tool"
			}
		}
		return roles, nil
	}
	var wire struct {
		Messages []struct {
			Role string `json:"role"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		return nil, err
	}
	roles := make([]string, len(wire.Messages))
	for index, message := range wire.Messages {
		roles[index] = message.Role
	}
	return roles, nil
}

func fingerprint(raw []byte) string {
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

func requestPrefix(request llmprotocol.Request) ([]string, error) {
	// Include all shared-prefix semantic inputs, not just messages. Per-turn
	// sampling/output allowances and routing/private instructions are absent.
	header, err := json.Marshal(struct {
		Instructions []llmprotocol.InstructionBlock
		Tools        []llmprotocol.Tool
	}{prefixInstructions(request.Instructions), prefixTools(request.Tools)})
	if err != nil {
		return nil, err
	}
	result := []string{fingerprint(header)}
	for _, message := range request.Messages {
		message.Content = prefixContents(message.Content)
		encoded, err := json.Marshal(message)
		if err != nil {
			return nil, err
		}
		result = append(result, fingerprint(encoded))
	}
	return result, nil
}

// Cache breakpoints move as client history grows. Compare typed semantic
// content without those directives; never edit provider-bound request bytes.
func prefixContents(contents []llmprotocol.Content) []llmprotocol.Content {
	if contents == nil {
		return nil
	}
	result := append([]llmprotocol.Content(nil), contents...)
	for index := range result {
		result[index].Cache = nil
		if result[index].ToolResult != nil {
			toolResult := *result[index].ToolResult
			toolResult.Content = prefixContents(toolResult.Content)
			result[index].ToolResult = &toolResult
		}
	}
	return result
}

func prefixInstructions(instructions []llmprotocol.InstructionBlock) []llmprotocol.InstructionBlock {
	if instructions == nil {
		return nil
	}
	result := append([]llmprotocol.InstructionBlock(nil), instructions...)
	for index := range result {
		result[index].Content = prefixContents(result[index].Content)
	}
	return result
}

func prefixTools(tools []llmprotocol.Tool) []llmprotocol.Tool {
	if tools == nil {
		return nil
	}
	result := append([]llmprotocol.Tool(nil), tools...)
	for index := range result {
		result[index].Cache = nil
	}
	return result
}

func extendsPrefix(previous, current []string) bool {
	if len(previous) > len(current) {
		return false
	}
	for index, item := range previous {
		if current[index] != item {
			return false
		}
	}
	return true
}

// resetEpoch invalidates transcript-dependent evidence after a prefix change.
// keepServed (execution_context_v2 only) preserves the last successfully
// executed action so last_served stays truthful; the v2 policy decides from
// the event whether this is a switch boundary. Current and v1 modes clear it,
// exactly as before v2: there a compaction is a model-switch boundary.
func (state *branchState) resetEpoch(keepServed bool) {
	if state.carried == nil {
		state.carried = &carriedTranscript{ledger: state.ledger, wire: state.seen}
	}
	state.epoch++
	if !keepServed {
		state.held, state.heldWorker = "", ""
	}
	state.placements = make(map[string]thinkingcontrol.PlacerState)
	state.workers = make(map[string]WorkerEvidence)
	state.pending = nil
	state.prefix = nil
	state.issuers = nil
	state.provenance = nil
	state.ledger = nil
	state.wire = nil
}

var legacySession = regexp.MustCompile(`(?i)_session_([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$`)

func requestIdentity(header http.Header, body []byte) (string, string, error) {
	var wire struct {
		Metadata struct {
			UserID string `json:"user_id"`
		} `json:"metadata"`
		PromptCacheKey string `json:"prompt_cache_key"`
		SessionID      string `json:"session_id"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		return "", "", fmt.Errorf("invalid identity metadata")
	}
	first := func(names ...string) string {
		for _, name := range names {
			if value := strings.TrimSpace(header.Get(name)); value != "" {
				return value
			}
		}
		return ""
	}
	session := first("x-rayline-session-id", "x-conversation-id", "x-metrics-conversation-id", "x-claude-code-session-id", "x-parent-session-id", "x-session-affinity", "x-session-id")
	if session == "" {
		session = wire.PromptCacheKey
	}
	if session == "" {
		session = wire.SessionID
	}
	if session == "" && wire.Metadata.UserID != "" {
		var user struct {
			SessionID string `json:"session_id"`
		}
		if json.Unmarshal([]byte(wire.Metadata.UserID), &user) == nil {
			session = user.SessionID
		}
		if session == "" {
			if match := legacySession.FindStringSubmatch(wire.Metadata.UserID); match != nil {
				session = match[1]
			}
		}
	}
	branch := first("x-rayline-branch-id", "thread-id", "x-claude-code-agent-id")
	if branch == "" {
		branch = "main"
	}
	if session == "" {
		return "", "", fmt.Errorf("hybrid execution requires a stable session identity")
	}
	for _, value := range []string{session, branch} {
		if len(value) > 1024 || strings.ContainsAny(value, "\r\n\x00") {
			return "", "", fmt.Errorf("invalid execution session or branch identity")
		}
	}
	return session, branch, nil
}
