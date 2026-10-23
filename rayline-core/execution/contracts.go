// Package execution owns standalone HTTP execution using the same protocol and
// thinking-control implementation as VSR. Policy selection remains external.
package execution

import (
	"context"
	"encoding/json"

	"github.com/atlasfutures/semantic-router/rayline-core/cachecontext"
	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"
)

const ContractVersion = 1

// Deployment identifies an actual serving target, independently of the ARC
// model/action identity. Credentials are supplied by the parent through an
// environment variable; they never belong in a manifest or process argument.
type Deployment struct {
	ID                    string                 `json:"id"`
	Kind                  string                 `json:"kind"`
	URL                   string                 `json:"url"`
	Protocol              llmprotocol.WireFormat `json:"protocol"`
	WireModel             string                 `json:"wire_model"`
	AuthEnv               string                 `json:"auth_env,omitempty"`
	DispatchProfileSHA256 string                 `json:"dispatch_profile_sha256,omitempty"`
	IssuerSHA256          string                 `json:"issuer_sha256,omitempty"`
	ProviderPin           *ProviderPin           `json:"provider_pin,omitempty"`
	RegistryModel         string                 `json:"registry_model"`
	RegistryProvider      string                 `json:"registry_provider"`
	// RuntimeIdentity pins the operator-qualified model/template/runtime
	// profile. It is not evidence of cache residency or exclusive use.
	RuntimeIdentity string          `json:"runtime_identity"`
	MaxOutputTokens int64           `json:"max_output_tokens"`
	TokenUnits      string          `json:"token_units"`
	PromptCounter   *LlamaCPPConfig `json:"prompt_counter,omitempty"`
	// InbandSystem is an explicitly qualified role-changing compatibility
	// rule, separate from the action's exact thinking-control identity.
	InbandSystem string `json:"inband_system,omitempty"`
}

// ProviderPin is part of the approved cloud dispatch profile. Exact dispatch
// requires one named OpenRouter provider and never permits provider fallback.
type ProviderPin struct {
	Only           []string `json:"only"`
	AllowFallbacks bool     `json:"allow_fallbacks"`
	DataCollection string   `json:"data_collection,omitempty"`
	ZDR            *bool    `json:"zdr,omitempty"`
}

// Action is an exact package action bound to one qualified deployment. Control
// is the package's v5 JSON object, parsed by the shared renderer, not rebuilt
// from a display label such as "high".
type Action struct {
	ID           string          `json:"action_id"`
	Worker       string          `json:"worker"`
	Model        string          `json:"model"`
	DeploymentID string          `json:"deployment_id"`
	ControlID    string          `json:"control_id"`
	Control      json.RawMessage `json:"control"`
}

type Decision struct {
	RouteID       string
	ActionID      string
	Worker        string
	Model         string
	PackageSHA256 string
	ControlID     string
	ArmID         *string
	Result        string
}

// DecisionInput contains the original public request and execution evidence.
// Private-append bytes are never sent back as the user's decision transcript.
type DecisionInput struct {
	Body      []byte
	Format    llmprotocol.WireFormat
	SessionID string
	BranchID  string
	RequestID string
	State     ExecutionContext
}

// DecisionProvider is implemented by the hosted public-contract adapter. It
// must reject an unapplied context/price world and return the selected action;
// the executor does not rank public marginal predictions itself.
type DecisionProvider interface {
	Decide(context.Context, DecisionInput) (Decision, error)
}

// ExecutionContext is process-local evidence. Its public projection belongs
// to the versioned Decisions adapter, rather than HTTP transport or codecs.
type ExecutionContext struct {
	InstanceID     string
	ContextEpoch   string
	HeldActionID   string
	CompletedTurns uint64
	Workers        []WorkerEvidence
	Attribution    []Attribution
}

type Attribution struct {
	Message  int     `json:"message"`
	ActionID string  `json:"action_id"`
	ArmID    *string `json:"arm_id"`
	Worker   string  `json:"worker"`
	RouteID  *string `json:"route_id"`
	Result   *string `json:"result"`
}

type WorkerEvidence struct {
	DeploymentID string
	Call         cachecontext.CallCache
	// PrefixTokens is unknown until a qualified worker tokenizer or provider
	// count establishes current compatible overlap; ARC encoder tokens are
	// not a substitute.
	PrefixTokens     *int
	OverlapTokens    *int
	TokenUnits       string
	PrefixProvenance string
	Residency        *Residency
	prompt           *cachecontext.MeasuredPrompt
}

type Residency struct {
	State      string `json:"state"`
	Provenance string `json:"provenance"`
}

type ControlReceipt = thinkingcontrol.Receipt
