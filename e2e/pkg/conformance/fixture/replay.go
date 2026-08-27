package fixture

import (
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/conformance"
)

// replay executes a validated script's steps in order against one response.
//
// Step 0 is always status, which the schema calls the commit point: the status and
// headers reach the wire before any later step runs, so a case that truncates or
// errors after step 0 truncates a response the client has already committed to.
// Flushing there means a buffered body is framed chunked rather than with a
// Content-Length, which is the price of making the commit point real.
func replay(w http.ResponseWriter, script *conformance.ReplayScript, dir string) {
	controller := http.NewResponseController(w)
	for _, step := range script.Steps {
		if done := runStep(w, controller, step, dir); done {
			return
		}
	}
}

// runStep executes one step and reports whether the response is finished.
func runStep(w http.ResponseWriter, controller *http.ResponseController, step conformance.ReplayStep, dir string) bool {
	switch step.Kind {
	case conformance.StepStatus:
		writeStatus(w, controller, step)
		return false

	case conformance.StepBody:
		// A buffered body ends the response, so nothing after it can run.
		writeArtifact(w, controller, dir, step.File, 0)
		return true

	case conformance.StepSSE:
		// A stream continues: another sse step, a delay, or a disconnect may follow.
		delivered := writeArtifact(w, controller, dir, step.File, step.ChunkBytes)
		return !delivered

	case conformance.StepDelay:
		time.Sleep(time.Duration(step.Millis) * time.Millisecond)
		return false

	case conformance.StepDisconnect:
		disconnect(controller)
		return true
	}
	// The loader rejects an unknown kind, so reaching here means the script and the
	// executor disagree. End the response rather than replay a half-understood script.
	return true
}

func writeStatus(w http.ResponseWriter, controller *http.ResponseController, step conformance.ReplayStep) {
	for name, value := range step.Headers {
		w.Header().Set(name, value)
	}
	w.WriteHeader(step.Status)
	_ = controller.Flush()
}

// writeArtifact writes a sibling artifact and reports whether all of it reached the
// client. chunk splits the file into fixed-size writes, each flushed on its own, so
// a case can put an event boundary or a UTF-8 sequence across two network reads.
// A chunk of zero or less writes the file whole.
//
// A write that fails means the client is gone; the caller stops there rather than
// finishing a stream nobody is reading.
func writeArtifact(w http.ResponseWriter, controller *http.ResponseController, dir, name string, chunk int) bool {
	raw, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		// The loader proved the file exists and is readable, so this is a fixture the
		// run cannot trust. Cut the connection instead of sending a short response
		// that a comparator might read as a legitimate truncation case.
		disconnect(controller)
		return false
	}

	if chunk <= 0 {
		chunk = len(raw)
	}
	for start := 0; start < len(raw); start += chunk {
		end := min(start+chunk, len(raw))
		if _, err := w.Write(raw[start:end]); err != nil {
			return false
		}
		if err := controller.Flush(); err != nil {
			return false
		}
	}
	return true
}

// disconnect closes the connection with no terminal event, leaving a chunked
// response without its terminating chunk. That is what a client sees when a
// provider dies mid-stream.
func disconnect(controller *http.ResponseController) {
	_ = controller.Flush()

	conn, _, err := controller.Hijack()
	if err != nil {
		// This writer cannot surrender its connection. Aborting the handler ends the
		// response the same way, without a terminal event and without a stack trace.
		panic(http.ErrAbortHandler)
	}
	_ = conn.Close()
}
