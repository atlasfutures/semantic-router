package selection

import (
	"encoding/json"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// RaylineARCSelectionContext carries only the structured, privacy-safe inputs
// required by the artifact-owned ARC selector. It is nil for every other
// algorithm.
type RaylineARCSelectionContext struct {
	EpisodeIDHash string
	// Coalesced is the decision an identical in-flight request on this
	// episode already made. When set, the selector returns it instead of
	// deciding again: the request is a resend of a turn being decided, not a
	// new turn.
	Coalesced *SelectionResult
	// EncoderVisitedReplicaIDs names the replicas this turn's encode touched,
	// recorded as soon as the encode returns rather than when a result is
	// built.
	//
	// It exists because closing a retained session needs those identities and
	// every step after the encode can fail: scoring, policy, validation, the
	// manifest mapping. A cleanup that reads them from the finished trace
	// therefore finds nothing on exactly the paths where the session was
	// retained but the lookup did not complete.
	EncoderVisitedReplicaIDs []string
	Turns                    []raylinearc.Turn
	State                    *raylinearc.EpisodeState
	InputTokens              int
	// ImageBearing reports that this turn carries image input. The turn
	// projection drops image blocks, so the encoder never sees them, but the
	// provider request does: an arm that rejects image input answers 404 and
	// not a degraded completion.
	ImageBearing bool
	// NonVisionArms marks, by arm ordinal, the candidates whose model card
	// declares no image input. It is indexed like CandidateModels and is nil
	// when every arm is vision-capable, which is the unmarked default.
	NonVisionArms []bool
	// DisabledArms marks, by arm ordinal, the candidates an operator has taken
	// out of service. Unlike NonVisionArms it is not a fact about this turn:
	// it holds for every turn until the card changes. Nil when no arm is
	// marked, which is the unmarked default.
	DisabledArms []bool
	// RequiredCapabilities names what this turn needs an arm to hold, from
	// llmprotocol.RequiredRoutingCapabilities. It is empty for almost every
	// turn: only a server tool or an image inside a tool result puts a name
	// here.
	RequiredCapabilities []string
	// IncapableArms marks, by arm ordinal, the candidates whose model card
	// does not claim every capability this turn requires. It is nil when the
	// turn requires none, which leaves selection exactly as it was.
	IncapableArms      []bool
	PreparationFailure string
	// RawRequest and RequestFormat are the client body as received and its
	// policy-service request format (anthropic_messages or openai_chat), set
	// only for the policy-service mode.
	RawRequest    []byte
	RequestFormat string
	// PolicyInput and PolicyInstructions are a Responses request materialized
	// for the policy service; set only for request format openai_responses.
	PolicyInput        []json.RawMessage
	PolicyInstructions *string
}

// RaylineARCTrace records bounded, privacy-safe artifact policy diagnostics.
type RaylineARCTrace struct {
	// PolicyActionID, PolicyArmID and ThinkingLevel are the policy-service
	// decision: the package action, the trained arm the service scored, and
	// the level the action binds. Empty outside the policy-service mode.
	PolicyActionID string
	PolicyArmID    string
	ThinkingLevel  string
	// PolicyActionModel is the chosen action's trained model and
	// WorkerProviderModel the provider model its worker serves; they differ
	// when an action is remapped to another provider.
	PolicyActionModel   string
	WorkerProviderModel string
	// PolicyLatency is the decide call's round trip as the router timed it;
	// EncoderLatency stays the encode alone.
	PolicyLatency time.Duration
	// EncoderLatencyUnknown marks an encode time the policy service did not
	// report: EncoderLatency is then not a measurement, and is neither
	// logged nor observed.
	EncoderLatencyUnknown bool
	// PolicyNextState is the ledger and epoch to commit with this turn.
	PolicyNextState *raylinearc.PolicyEpisodeState
	// ArtifactID and ArtifactRevision hold SHA256-derived hashes of the
	// deployment-private artifact identity, never the raw pins.
	ArtifactID          string
	ArtifactRevision    string
	EncoderRevision     string
	EpisodeIDHash       string
	SelectedArm         int
	PreviousArm         *int
	RawScores           []float32
	AdjustedScores      []float32
	SwitchCostUSD       []float64
	CacheMissTokens     []int
	Stayed              bool
	UpgradeExemptions   []bool
	StayUpgradeExempted bool
	// ExcludedArms marks the arms a hard constraint removed before scoring.
	// It is what explains a switch the scores alone do not.
	ExcludedArms         []bool
	SerializedTokens     int
	FullHistoryTokens    int
	TruncatedTokens      int
	CachedPrefixTokens   int
	RetainedPrefixTokens int
	AppendedTokens       int
	SessionAction        string
	SessionRevision      int
	EncoderLatency       time.Duration
	EncoderReplicaIndex  int
	EncoderAttempts      int
	EncoderFailover      bool
	// EncoderReplicaID and EncoderVisitedReplicaIDs are internal transaction
	// state. They must never be logged or exported as metric labels.
	EncoderReplicaID         string
	EncoderVisitedReplicaIDs []string
	// TurnIndex is the episode position this selection was made at: the
	// number of turns the episode had committed before it.
	TurnIndex uint64
}
