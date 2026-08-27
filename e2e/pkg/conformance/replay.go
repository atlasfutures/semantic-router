package conformance

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"sigs.k8s.io/yaml"
)

// ReplaySchemaVersion is the only replay.yaml schema_version this loader accepts.
const ReplaySchemaVersion = "protocol-conformance-replay-v1"

// StepKind is one instruction the provider fixture executes, in order.
type StepKind string

const (
	// StepStatus writes the response status line and headers. It must come first and
	// may appear only once; everything after it is a committed response.
	StepStatus StepKind = "status"
	// StepBody writes a buffered body read from File, then ends the response.
	StepBody StepKind = "body"
	// StepSSE writes an SSE stream read from File. ChunkBytes splits it at arbitrary
	// byte boundaries so a case can exercise UTF-8 and event-boundary splits.
	StepSSE StepKind = "sse"
	// StepDelay sleeps for Millis before the next step.
	StepDelay StepKind = "delay"
	// StepDisconnect closes the connection without a terminal event.
	StepDisconnect StepKind = "disconnect"
)

// ReplayScript is the provider-side program for one case. DPC-101's programmable
// fixture executes it; this package only parses and validates it.
type ReplayScript struct {
	SchemaVersion string       `json:"schema_version"`
	Expect        ExpectSpec   `json:"expect"`
	Steps         []ReplayStep `json:"steps"`
}

// ExpectSpec is what the fixture asserts about the inbound provider request before
// it replays anything. Body identity is asserted by the comparators, not here.
type ExpectSpec struct {
	Method string `json:"method"`
	Path   string `json:"path"`
	// Headers are header names and exact values the provider request must carry.
	Headers map[string]string `json:"headers"`
}

// ReplayStep is one instruction. Fields not used by Kind must be unset.
type ReplayStep struct {
	Kind StepKind `json:"kind"`

	// Status and Headers apply to StepStatus.
	Status  int               `json:"status"`
	Headers map[string]string `json:"headers"`

	// File is the artifact StepBody and StepSSE replay, relative to the case directory.
	File string `json:"file"`
	// ChunkBytes optionally splits a StepSSE file into fixed-size writes. Zero writes it whole.
	ChunkBytes int `json:"chunk_bytes"`

	// Millis is the StepDelay duration.
	Millis int `json:"millis"`
}

func parseReplayScript(raw []byte, dir string) (*ReplayScript, error) {
	var script ReplayScript
	if err := yaml.UnmarshalStrict(raw, &script); err != nil {
		return nil, fmt.Errorf("parse %s: %w", fileReplayScript, err)
	}
	if script.SchemaVersion != ReplaySchemaVersion {
		return nil, fmt.Errorf("%s: unsupported schema_version %q, want %q", fileReplayScript, script.SchemaVersion, ReplaySchemaVersion)
	}
	if script.Expect.Method == "" || script.Expect.Path == "" {
		return nil, fmt.Errorf("%s: expect.method and expect.path are required", fileReplayScript)
	}
	if len(script.Steps) == 0 {
		return nil, fmt.Errorf("%s: at least one step is required", fileReplayScript)
	}

	for i, step := range script.Steps {
		if err := validateStep(step, dir); err != nil {
			return nil, fmt.Errorf("%s step %d: %w", fileReplayScript, i, err)
		}
	}
	if script.Steps[0].Kind != StepStatus {
		return nil, fmt.Errorf("%s: the first step must be %q so the commit point is explicit", fileReplayScript, StepStatus)
	}
	for _, step := range script.Steps[1:] {
		if step.Kind == StepStatus {
			return nil, fmt.Errorf("%s: only the first step may be %q", fileReplayScript, StepStatus)
		}
	}
	return &script, nil
}

func validateStep(step ReplayStep, dir string) error {
	switch step.Kind {
	case StepStatus:
		if step.Status < 100 || step.Status > 599 {
			return fmt.Errorf("status %d is not an HTTP status", step.Status)
		}
	case StepBody, StepSSE:
		return validateReplaySource(step, dir)
	case StepDelay:
		if step.Millis <= 0 {
			return fmt.Errorf("kind %q requires a positive millis", step.Kind)
		}
	case StepDisconnect:
		// No fields.
	default:
		return fmt.Errorf("unknown kind %q", step.Kind)
	}
	return nil
}

func validateReplaySource(step ReplayStep, dir string) error {
	if step.File == "" {
		return fmt.Errorf("kind %q requires file", step.Kind)
	}
	if err := requireArtifact(dir, step.File); err != nil {
		return err
	}
	if step.ChunkBytes < 0 {
		return fmt.Errorf("chunk_bytes %d is negative", step.ChunkBytes)
	}
	if step.Kind == StepBody && step.ChunkBytes != 0 {
		return errors.New("chunk_bytes applies to sse steps only")
	}
	return nil
}

func requireArtifact(dir, name string) error {
	// A replay step may only name a sibling artifact. Anything with a separator or a
	// parent reference would let a fixture read outside its own case directory.
	if name != filepath.Base(name) || name == "." || name == ".." {
		return fmt.Errorf("file %q must be a plain name inside the case directory", name)
	}
	if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
		return fmt.Errorf("file %q is not readable: %w", name, err)
	}
	return nil
}
