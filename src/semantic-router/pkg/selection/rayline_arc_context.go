package selection

import (
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// RaylineARCSelectionContext carries only the structured, privacy-safe inputs
// required by the artifact-owned ARC selector. It is nil for every other
// algorithm.
type RaylineARCSelectionContext struct {
	EpisodeIDHash string
	Turns         []raylinearc.Turn
	State         *raylinearc.EpisodeState
	InputTokens   int
	// ImageBearing reports that this turn carries image input. The turn
	// projection drops image blocks, so the encoder never sees them, but the
	// provider request does: an arm that rejects image input answers 404 and
	// not a degraded completion.
	ImageBearing bool
	// NonVisionArms marks, by arm ordinal, the candidates whose model card
	// declares no image input. It is indexed like CandidateModels and is nil
	// when every arm is vision-capable, which is the unmarked default.
	NonVisionArms      []bool
	PreparationFailure string
}

// RaylineARCTrace records bounded, privacy-safe artifact policy diagnostics.
type RaylineARCTrace struct {
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
}
