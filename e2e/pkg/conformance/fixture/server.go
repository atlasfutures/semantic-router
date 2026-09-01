// Package fixture runs the hermetic programmable provider fixture PL-0042 requires.
//
// The server stands in for a real upstream provider. It accepts the Chat
// Completions, Responses, and Anthropic Messages paths, records every inbound
// request byte for byte, and replays the scripted response a case declares in its
// replay.yaml. The script format, its types, and its validation belong to the
// parent conformance package; this package only executes a parsed script over
// HTTP.
//
// A case driver starts the server in-process, calls Reset with the case's script
// and directory, points the router at URL, then reads Observed to assert what the
// provider actually saw. The same lifecycle is available over the control
// endpoints so the fixture can also run out of process.
package fixture

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/conformance"
)

// Provider wire endpoints the fixture serves. A case points the router's provider
// backend at URL plus one of these paths.
const (
	PathChatCompletions = "/v1/chat/completions"
	PathResponses       = "/v1/responses"
	PathMessages        = "/v1/messages"
)

// Control endpoints. They are the out-of-process form of Reset and Observed.
const (
	// PathReset clears the recorded requests and installs a replay script. The body
	// is the raw replay.yaml, and the "dir" query parameter is the case directory
	// its file references resolve against. An empty body only clears state.
	PathReset = "/reset"
	// PathObserved returns the recorded requests as JSON, newest last.
	PathObserved = "/observed"
)

// StatusExpectationFailed is the distinctive status the fixture returns when an
// inbound request does not match the script's expect block, or when no script is
// loaded. It is outside the range any provider or router uses, so a conformance
// run cannot silently pass against the wrong route.
const StatusExpectationFailed = 597

// readHeaderTimeout bounds a client that opens a connection and never completes
// its request headers. It never applies to a scripted delay, which happens after
// the request is fully read.
const readHeaderTimeout = 10 * time.Second

// ProviderPaths are the three protocol paths a case may route to.
func ProviderPaths() []string {
	return []string{PathChatCompletions, PathResponses, PathMessages}
}

// Server is a running provider fixture. The zero value is not usable; call Start.
type Server struct {
	http     *http.Server
	listener net.Listener

	// mu guards everything a request handler and a control call share.
	mu       sync.Mutex
	script   *conformance.ReplayScript
	dir      string
	observed []ObservedRequest
}

// Start listens on addr and serves until Shutdown. Pass "127.0.0.1:0" to take any
// free port and read the result from URL.
func Start(addr string) (*Server, error) {
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, fmt.Errorf("fixture: listen on %s: %w", addr, err)
	}

	s := &Server{listener: listener}
	s.http = &http.Server{Handler: s.routes(), ReadHeaderTimeout: readHeaderTimeout}
	go func() { _ = s.http.Serve(listener) }()
	return s, nil
}

// URL is the base URL a router or client dials, with no trailing slash.
func (s *Server) URL() string {
	addr, ok := s.listener.Addr().(*net.TCPAddr)
	if !ok {
		return "http://" + s.listener.Addr().String()
	}
	host := addr.IP.String()
	if addr.IP == nil || addr.IP.IsUnspecified() {
		host = "127.0.0.1"
	}
	return "http://" + net.JoinHostPort(host, strconv.Itoa(addr.Port))
}

// Shutdown stops serving. Connections a disconnect step hijacked are already
// closed and are not waited on.
func (s *Server) Shutdown(ctx context.Context) error {
	if err := s.http.Shutdown(ctx); err != nil {
		return fmt.Errorf("fixture: shutdown: %w", err)
	}
	return nil
}

// Reset clears the recorded requests and installs the script the fixture replays
// next. dir is the case directory the script's file references resolve against.
// A nil script leaves the fixture loaded with nothing, so the next request is
// recorded and refused rather than answered.
func (s *Server) Reset(script *conformance.ReplayScript, dir string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.script, s.dir, s.observed = script, dir, nil
}

// Observed returns a copy of the requests recorded since the last Reset, in
// arrival order. It is safe to call while requests are in flight.
func (s *Server) Observed() []ObservedRequest {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]ObservedRequest, len(s.observed))
	copy(out, s.observed)
	return out
}

func (s *Server) routes() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc(PathReset, s.handleReset)
	mux.HandleFunc(PathObserved, s.handleObserved)
	// Every other path reaches the provider handler, including one no case expects.
	// A misrouted request must be recorded and refused, never answered with a 404
	// that a run could read as "the provider was simply not called".
	mux.HandleFunc("/", s.handleProvider)
	return mux
}

func (s *Server) handleReset(w http.ResponseWriter, r *http.Request) {
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, fmt.Sprintf("read replay script: %v", err), http.StatusBadRequest)
		return
	}

	dir := r.URL.Query().Get("dir")
	if len(bytes.TrimSpace(raw)) == 0 {
		s.Reset(nil, dir)
		w.WriteHeader(http.StatusNoContent)
		return
	}

	script, err := conformance.ParseReplayScript(raw, dir)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	s.Reset(script, dir)
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) handleObserved(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("content-type", "application/json")
	if err := json.NewEncoder(w).Encode(s.Observed()); err != nil {
		http.Error(w, fmt.Sprintf("encode observed: %v", err), http.StatusInternalServerError)
	}
}

func (s *Server) handleProvider(w http.ResponseWriter, r *http.Request) {
	body, readErr := io.ReadAll(r.Body)
	observed := capture(r, body)

	script, dir := s.snapshot()
	observed.Mismatch = matchExpectation(script, observed)
	if readErr != nil && observed.Mismatch == "" {
		observed.Mismatch = fmt.Sprintf("read request body: %v", readErr)
	}
	s.record(observed)

	if observed.Mismatch != "" {
		w.Header().Set("content-type", "text/plain; charset=utf-8")
		w.WriteHeader(StatusExpectationFailed)
		_, _ = io.WriteString(w, observed.Mismatch+"\n")
		return
	}
	replay(w, script, dir)
}

func (s *Server) snapshot() (*conformance.ReplayScript, string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.script, s.dir
}

func (s *Server) record(observed ObservedRequest) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.observed = append(s.observed, observed)
}
