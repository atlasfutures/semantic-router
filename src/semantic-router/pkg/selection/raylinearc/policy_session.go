/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package raylinearc

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

// A policy service deployed as a Modal Server with sticky sessions
// (@modal.sessioned) routes every request carrying a session's token to the
// one container that holds the session. The policy service keeps each
// episode's retained encoder session and strict episode memory in its
// process, so binding an episode to one Modal session keeps its turns
// appending on one container while the Server autoscales.
//
// The contract, as Modal's SDK speaks it (modal/_server.py) and as measured
// on a dev Server:
//   - POST {url}/_modal/sessions/start with the proxy key starts a session
//     and answers {session_id, token}; without proxy auth it is 401.
//   - A request carrying "Modal-Authorization: Bearer <token>" reaches the
//     session's container; it needs no proxy key.
//   - A token whose container is gone (stopped, preempted) answers 502 with
//     an empty body; an idle-expired session answers 404 "Unknown or expired
//     session". Neither is ever routed elsewhere by Modal.
//   - POST {url}/_modal/sessions/terminate needs the proxy key and the
//     token; the token alone is 401.
const (
	policySessionStartPath     = "/_modal/sessions/start"
	policySessionTerminatePath = "/_modal/sessions/terminate"

	// PolicySessionStartClass is the failure class of a session that could
	// not be started: a timeout, a refusal or a malformed answer. The turn
	// fails closed (policy_service_session_start) and commits nothing.
	PolicySessionStartClass = "session_start"
	// PolicySessionGoneClass and PolicySessionExpiredClass are Modal's own
	// answers for a token, recognised by their verified signatures only: a
	// 502 with an empty body (the session's container is gone) and a 404
	// "Unknown or expired session". Neither reached a container that holds
	// the episode, so the turn may move to a new session, once. A 401 is not
	// among them: no token-rejection signature is verified, so it fails the
	// turn closed like any other refusal.
	PolicySessionGoneClass    = "session_gone"
	PolicySessionExpiredClass = "session_expired"

	maxPolicySessionIDBytes    = 128
	maxPolicySessionTokenBytes = 8 * 1024
	maxPolicySessionStartBytes = 64 * 1024

	// DefaultPolicySessionIdleTimeout matches the episode store's shipped
	// idle TTL, so a session lives as long as the episode it serves.
	DefaultPolicySessionIdleTimeout = 900 * time.Second
	// DefaultPolicySessionStartTimeout bounds a session start. Modal queues
	// a start for capacity for minutes; a turn must not.
	DefaultPolicySessionStartTimeout = 10 * time.Second
	// policySessionTerminateTimeout bounds a best-effort terminate.
	policySessionTerminateTimeout = 5 * time.Second
	// policySessionTransportRetryBudget bounds the one move a turn makes when
	// a decide on the episode's session fails in transport: the new session's
	// start and its decide together.
	policySessionTransportRetryBudget = 2 * time.Second
	// DefaultPolicySessionReplacedGrace is how long a session the episode
	// moved off is kept after the commit that replaced it: a turn on another
	// router instance that read the episode before that commit (a side call,
	// a relaxed turn) may still send its decides there. It is never shorter
	// than two decide timeouts (a decide and a re-ask); a decide in flight at
	// the terminate finishes anyway.
	DefaultPolicySessionReplacedGrace = 60 * time.Second
)

// PolicySession is one Modal sticky session: the episode's owner on a
// sessioned policy Server. The token routes and authorizes requests, so it
// is a credential: it is persisted only in the episode state and never
// logged; String and GoString redact it.
type PolicySession struct {
	ID    string
	Token string
}

func (PolicySession) String() string   { return "policy-session(redacted)" }
func (PolicySession) GoString() string { return "policy-session(redacted)" }

// Format redacts every verb, so %v, %+v and %#v never print the token.
func (session PolicySession) Format(state fmt.State, _ rune) {
	_, _ = io.WriteString(state, session.String())
}

// MarshalJSON redacts the session wherever a value holding it is encoded; the
// episode store persists it through its own wire type.
func (PolicySession) MarshalJSON() ([]byte, error) { return []byte(`"redacted"`), nil }

// ValidatePolicySession checks a session's shape: a bounded id and token of
// visible ASCII, safe to put in a header and to persist.
func ValidatePolicySession(session PolicySession) error {
	if !visibleASCII(session.ID, maxPolicySessionIDBytes) {
		return errors.New("ARC policy session id is invalid")
	}
	if !visibleASCII(session.Token, maxPolicySessionTokenBytes) {
		return errors.New("ARC policy session token is invalid")
	}
	return nil
}

func visibleASCII(value string, limit int) bool {
	if value == "" || len(value) > limit {
		return false
	}
	for index := 0; index < len(value); index++ {
		if value[index] <= ' ' || value[index] > '~' {
			return false
		}
	}
	return true
}

func clonePolicySession(session *PolicySession) *PolicySession {
	if session == nil {
		return nil
	}
	cloned := *session
	return &cloned
}

// PolicySessionConfig is the resolved sessions configuration.
type PolicySessionConfig struct {
	IdleTimeout   time.Duration
	StartTimeout  time.Duration
	ReplacedGrace time.Duration
}

// PolicySessionBackend starts and ends the Modal sessions of a sessioned
// policy Server, over the cell's one client (its URL and proxy key).
type PolicySessionBackend struct {
	client *PolicyServiceClient
	// controlHTTP sends session starts and terminates; it follows no
	// redirect, so the proxy key and a token stay on the Server's origin.
	controlHTTP *http.Client
	config      PolicySessionConfig
	now         func() time.Time
	newKey      func() (string, error)
	// suspect holds the sessions whose last decide timed out or failed in
	// transport (#358), so the episode's next turn does not wait on them again.
	suspect *suspectSessions
	// transportRetryBudget bounds a turn's move after a transport failure.
	transportRetryBudget time.Duration
}

// suspectSessions remembers, by a hash of the token and until the session's
// idle timeout, each session whose decide timed out or failed in transport.
// That turn still fails closed: the container may have advanced its session,
// so it is never asked again within the turn. But a stopped container's
// sessions often time out for minutes before Modal answers 502 or 404 for them
// (a 10-26 dev fault: about 7 min), and every turn of an episode on one would
// fail the same way. Its next turn starts a new session instead, which is the
// turn's one move. Process-local: each router instance learns on its own.
type suspectSessions struct {
	mu    sync.Mutex
	until map[[sha256.Size]byte]time.Time
	ttl   time.Duration
	now   func() time.Time
}

func newSuspectSessions(ttl time.Duration, now func() time.Time) *suspectSessions {
	return &suspectSessions{until: map[[sha256.Size]byte]time.Time{}, ttl: ttl, now: now}
}

// mark records a suspect session. It also forgets the expired ones, so the set
// stays bounded by the sessions that timed out within one idle timeout.
func (suspect *suspectSessions) mark(session PolicySession) {
	if suspect == nil {
		return
	}
	suspect.mu.Lock()
	defer suspect.mu.Unlock()
	now := suspect.now()
	for key, until := range suspect.until {
		if !now.Before(until) {
			delete(suspect.until, key)
		}
	}
	suspect.until[sha256.Sum256([]byte(session.Token))] = now.Add(suspect.ttl)
}

// has reports a suspect session whose idle timeout has not passed (after it,
// Modal has ended the session anyway: a turn on it gets the 404 and moves). A
// lookup touches one entry; mark does the pruning.
func (suspect *suspectSessions) has(session PolicySession) bool {
	if suspect == nil {
		return false
	}
	suspect.mu.Lock()
	defer suspect.mu.Unlock()
	key := sha256.Sum256([]byte(session.Token))
	until, found := suspect.until[key]
	if found && !suspect.now().Before(until) {
		delete(suspect.until, key)
		return false
	}
	return found
}

// NewPolicySessionBackend validates the configuration. The client must carry
// the proxy key: a session start is refused without proxy auth.
func NewPolicySessionBackend(client *PolicyServiceClient, config PolicySessionConfig) (*PolicySessionBackend, error) {
	if client == nil || client.config.ModalKey == "" || client.config.ModalSecret == "" {
		return nil, errors.New("ARC policy sessions need the Modal proxy key")
	}
	if config.IdleTimeout <= 0 {
		config.IdleTimeout = DefaultPolicySessionIdleTimeout
	}
	if config.StartTimeout <= 0 {
		config.StartTimeout = DefaultPolicySessionStartTimeout
	}
	if config.ReplacedGrace <= 0 {
		config.ReplacedGrace = max(DefaultPolicySessionReplacedGrace, 2*client.config.TotalTimeout)
	}
	return &PolicySessionBackend{
		client: client, controlHTTP: noRedirectClient(client.http), config: config, now: time.Now, newKey: idempotencyKey,
		suspect: newSuspectSessions(config.IdleTimeout, time.Now), transportRetryBudget: policySessionTransportRetryBudget,
	}, nil
}

func idempotencyKey() (string, error) {
	raw := make([]byte, 16)
	if _, err := rand.Read(raw); err != nil {
		return "", err
	}
	raw[6] = raw[6]&0x0f | 0x40 // version 4
	raw[8] = raw[8]&0x3f | 0x80 // RFC 4122 variant
	encoded := hex.EncodeToString(raw)
	return encoded[0:8] + "-" + encoded[8:12] + "-" + encoded[12:16] + "-" + encoded[16:20] + "-" + encoded[20:], nil
}

// Start starts one session, bounded by the start timeout. Any failure is a
// PolicyServiceError of class session_start.
func (backend *PolicySessionBackend) Start(ctx context.Context) (PolicySession, error) {
	key, err := backend.newKey()
	if err != nil {
		return PolicySession{}, &PolicyServiceError{Class: PolicySessionStartClass}
	}
	header := http.Header{}
	header.Set("x-modal-server-session-idle-timeout", strconv.Itoa(int(backend.config.IdleTimeout/time.Second)))
	header.Set("x-idempotency-key", key)
	header.Set("x-modal-timestamp", strconv.FormatFloat(float64(backend.now().UnixNano())/1e9, 'f', 6, 64))
	status, body, err := backend.control(ctx, policySessionStartPath, header, backend.config.StartTimeout)
	if err != nil {
		return PolicySession{}, &PolicyServiceError{Class: PolicySessionStartClass}
	}
	if status != http.StatusOK {
		return PolicySession{}, &PolicyServiceError{Class: PolicySessionStartClass, Status: status}
	}
	var started struct {
		SessionID string `json:"session_id"`
		Token     string `json:"token"`
	}
	if json.Unmarshal(body, &started) != nil {
		return PolicySession{}, &PolicyServiceError{Class: PolicySessionStartClass, Status: status}
	}
	session := PolicySession{ID: started.SessionID, Token: started.Token}
	if ValidatePolicySession(session) != nil {
		return PolicySession{}, &PolicyServiceError{Class: PolicySessionStartClass, Status: status}
	}
	return session, nil
}

// Terminate ends a session: its in-flight requests finish and new ones are
// refused. It needs the proxy key and the token.
func (backend *PolicySessionBackend) Terminate(ctx context.Context, session PolicySession) error {
	if ValidatePolicySession(session) != nil {
		return errors.New("ARC policy session is invalid")
	}
	header := http.Header{}
	// Modal's terminate reads the session token from x-modal-server-session-token; with the token only in
	// Modal-Authorization it answers 401 and the session keeps its slot until its idle timeout (prod 2026-10-27).
	header.Set("x-modal-server-session-token", session.Token)
	status, _, err := backend.control(ctx, policySessionTerminatePath, header, policySessionTerminateTimeout)
	if err != nil {
		return err
	}
	if status != http.StatusOK && status != http.StatusNoContent {
		return &PolicyServiceError{Class: "status", Status: status}
	}
	return nil
}

// control posts one session-control request with the proxy key.
func (backend *PolicySessionBackend) control(
	ctx context.Context,
	path string,
	header http.Header,
	timeout time.Duration,
) (int, []byte, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	config := backend.client.config
	endpoint := strings.TrimRight(config.BaseURL, "/") + path
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(nil))
	if err != nil {
		return 0, nil, err
	}
	for name, values := range header {
		for _, value := range values {
			request.Header.Add(name, value)
		}
	}
	request.Header.Set("Modal-Key", config.ModalKey)
	request.Header.Set("Modal-Secret", config.ModalSecret)
	response, err := backend.controlHTTP.Do(request)
	if err != nil {
		return 0, nil, err
	}
	defer func() { _ = response.Body.Close() }()
	body, err := io.ReadAll(io.LimitReader(response.Body, maxPolicySessionStartBytes+1))
	if err != nil || len(body) > maxPolicySessionStartBytes {
		return response.StatusCode, nil, errors.New("ARC policy session control response is unreadable")
	}
	return response.StatusCode, body, nil
}

// Probe checks the Server through a fresh session, then ends it: every
// request to a sessioned Server needs a token, readiness included. The start
// also warms a container, which a new deployment needs before it takes turns.
func (backend *PolicySessionBackend) Probe(
	ctx context.Context,
	check func(context.Context, *PolicyServiceClient) error,
) error {
	session, err := backend.Start(ctx)
	if err != nil {
		return err
	}
	checkErr := check(ctx, backend.client.WithSessionToken(session.Token))
	terminateCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), policySessionTerminateTimeout)
	defer cancel()
	_ = backend.Terminate(terminateCtx, session)
	return checkErr
}

// Bind starts one turn's routing for an episode. persisted is the session the
// episode state holds (nil for none). A turn that may not change the episode
// (a side call, which commits nothing) is ephemeral: it uses the persisted
// session when there is one, and otherwise, or once that session is gone,
// one of its own that Release ends.
func (backend *PolicySessionBackend) Bind(persisted *PolicySession, ephemeral bool) *PolicySessionBinding {
	return &PolicySessionBinding{backend: backend, persisted: clonePolicySession(persisted), ephemeral: ephemeral}
}

// PolicySessionBinding routes every call of one turn to one session: the
// persisted one, else one it starts. A call whose session is gone (Modal's
// 502 or 404 for the token, so it never reached a container holding the
// episode)
// starts a new session and is retried there once; the turn moves at most once,
// so a strict episode is never in flight on two containers within a turn. A
// transport failure on the episode's session, which never got an answer, is
// that move too, bounded by a small budget. A timeout, or a transport failure
// on a session the turn started, is returned as is: the container may have
// advanced its session, so the turn fails closed. That session is then
// suspect (#358): the episode's next turn starts a new one rather than wait on
// it again.
//
// A session the episode state records is never ended by the turn that moves
// off it, only once that turn's commit has recorded the new one
// (TerminateReplaced): the state is shared by every router instance, so a turn
// that ended it and then failed before its commit would leave the episode on a
// session that is gone, and the next turn on any other instance would fail on
// it (prod 2026-10-28). It is not safe for concurrent use.
type PolicySessionBinding struct {
	backend   *PolicySessionBackend
	persisted *PolicySession
	ephemeral bool
	current   *PolicySession
	// owned are the sessions this turn started; an ephemeral turn ends them.
	owned    []PolicySession
	sessions int
	moved    bool
	// replaced is the persisted session this turn moved off, ended only after
	// the commit that records its replacement. An ephemeral turn records no
	// replacement, so it never sets one.
	replaced *PolicySession
}

// Decide asks the bound session for one decision.
func (binding *PolicySessionBinding) Decide(
	ctx context.Context,
	request PolicyDecisionRequest,
) (*PolicyDecisionResponse, error) {
	if binding.current == nil {
		switch {
		case binding.persisted != nil && binding.backend.suspect.has(*binding.persisted):
			// The persisted session timed out on an earlier turn: this turn's
			// one move is to a new session, before any call reaches the old one.
			binding.moved = true
			if err := binding.moveOff(ctx); err != nil {
				return nil, err
			}
		case binding.persisted != nil:
			binding.use(*binding.persisted, false)
		default:
			if err := binding.start(ctx); err != nil {
				return nil, err
			}
		}
	}
	response, err := binding.decide(ctx, request)
	stale := err == nil && staleOwner(request, response)
	transport := binding.transportRetryable(ctx, err)
	if !stale && !transport && !policySessionLost(err) {
		return response, err
	}
	if stale && !binding.onPersisted() {
		// A rollover can leave the session on a container still serving the
		// previous build. End one this turn started; the episode's own is
		// replaced by the move below, and ended once the turn commits. The
		// caller refuses the stale answer itself (policy_package_mismatch) if
		// no new session replaces it. Ended off the turn's path: a stalled
		// terminate must not use up the deadline the new session needs.
		go binding.terminate(ctx, *binding.current)
	}
	if binding.moved {
		return response, err
	}
	binding.moved = true
	moveCtx := ctx
	if transport {
		var cancel context.CancelFunc
		moveCtx, cancel = context.WithTimeout(ctx, binding.backend.transportRetryBudget)
		defer cancel()
	}
	if startErr := binding.moveOff(moveCtx); startErr != nil {
		if stale || transport {
			// The turn fails (or is refused) on its first answer, as it would
			// have without the move.
			return response, err
		}
		return nil, startErr
	}
	response, err = binding.decide(moveCtx, request)
	if err == nil && staleOwner(request, response) {
		go binding.terminate(ctx, *binding.current)
	}
	return response, err
}

// decide sends one decide to the current session. A decide that failed
// without an answer makes the session suspect, unless the caller went away
// (a client disconnect), which says nothing about the container.
func (binding *PolicySessionBinding) decide(
	ctx context.Context,
	request PolicyDecisionRequest,
) (*PolicyDecisionResponse, error) {
	response, err := binding.backend.client.WithSessionToken(binding.current.Token).Decide(ctx, request)
	if policySessionUncertain(err) && !errors.Is(ctx.Err(), context.Canceled) {
		binding.backend.suspect.mark(*binding.current)
	}
	return response, err
}

// transportRetryable reports a transport failure the turn may move on: one
// on the episode's existing session, before the turn's one move, that got no
// decision (no answer, or an answer other than a 200 whose body could not be
// read) from a caller still waiting. A brand-new session's failure, or a
// timeout, is not: the container may be working on it.
func (binding *PolicySessionBinding) transportRetryable(ctx context.Context, err error) bool {
	var failure *PolicyServiceError
	if binding.moved || !binding.onPersisted() || ctx.Err() != nil || !errors.As(err, &failure) {
		return false
	}
	return failure.Class == "transport" && failure.Status != http.StatusOK
}

// onPersisted reports whether the turn is on the session the episode records.
func (binding *PolicySessionBinding) onPersisted() bool {
	return binding.persisted != nil && binding.current != nil && *binding.current == *binding.persisted
}

// moveOff starts the session the turn moves to. The persisted session it
// leaves is the replaced one, ended once the turn commits; an ephemeral turn
// commits nothing, so the episode keeps it and it is not ended at all.
func (binding *PolicySessionBinding) moveOff(ctx context.Context) error {
	if err := binding.start(ctx); err != nil {
		return err
	}
	if binding.persisted != nil && !binding.ephemeral {
		binding.replaced = clonePolicySession(binding.persisted)
	}
	return nil
}

// TerminateReplaced ends, best effort and off the caller's path, the session
// this turn moved the episode off, after the replaced grace. The caller calls
// it only once the episode commit has recorded the new session: before it, the
// old one is still the one every router instance would decide the episode's
// next turn on. The grace covers a turn that read the episode just before the
// commit; one still later moves off it (a transport failure or Modal's 404).
func (binding *PolicySessionBinding) TerminateReplaced(ctx context.Context) {
	if binding == nil || binding.replaced == nil {
		return
	}
	replaced := *binding.replaced
	binding.replaced = nil
	detached := context.WithoutCancel(ctx)
	time.AfterFunc(binding.backend.config.ReplacedGrace, func() { binding.terminate(detached, replaced) })
}

// Replaces reports whether the turn moved off a session the episode records,
// which its commit then ends (TerminateReplaced).
func (binding *PolicySessionBinding) Replaces() bool {
	return binding != nil && binding.replaced != nil
}

func (binding *PolicySessionBinding) start(ctx context.Context) error {
	session, err := binding.backend.Start(ctx)
	if err != nil {
		return err
	}
	binding.use(session, true)
	return nil
}

func (binding *PolicySessionBinding) use(session PolicySession, started bool) {
	binding.current = &session
	binding.sessions++
	if started {
		binding.owned = append(binding.owned, session)
	}
}

func (binding *PolicySessionBinding) terminate(ctx context.Context, session PolicySession) {
	terminateCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), policySessionTerminateTimeout)
	defer cancel()
	_ = binding.backend.Terminate(terminateCtx, session)
}

// policySessionFailureClass classifies Modal's own answer for a session
// token by its verified signature, "" for anything else: the service's JSON
// refusals (a 404 package_not_loaded among them), any other body, and any
// other status keep their usual class.
func policySessionFailureClass(status int, body []byte) string {
	switch {
	case status == http.StatusBadGateway && len(bytes.TrimSpace(body)) == 0:
		return PolicySessionGoneClass
	case status == http.StatusNotFound && string(bytes.TrimSpace(body)) == "Unknown or expired session":
		return PolicySessionExpiredClass
	default:
		return ""
	}
}

// policySessionLost reports a call Modal answered for its session token
// without reaching a container that holds the episode. A timeout, a
// transport failure or any other answer is not: the container may have
// advanced its session, so the turn fails closed.
//
// One residual: Modal's 502 for a container that is gone is also what a
// gateway failure in front of a live container would look like. A decide in
// flight when its container stops answers it too (measured on a dev Server),
// so it is taken as gone.
func policySessionLost(err error) bool {
	var failure *PolicyServiceError
	if !errors.As(err, &failure) {
		return false
	}
	switch failure.Class {
	case PolicySessionGoneClass, PolicySessionExpiredClass:
		return true
	default:
		return false
	}
}

// policySessionUncertain reports a decide that failed without an answer: a
// timeout or a transport failure. The container may or may not have advanced
// its session, so the turn fails closed and the session becomes suspect.
func policySessionUncertain(err error) bool {
	var failure *PolicyServiceError
	if !errors.As(err, &failure) {
		return false
	}
	return failure.Class == "timeout" || failure.Class == "transport"
}

// PolicySessionUnanswered reports a failure of the session layer itself (a
// start that failed, or a session lost again after the turn's one move): the
// policy never answered it, so a caller must not ask again on its account.
func PolicySessionUnanswered(err error) bool {
	var failure *PolicyServiceError
	if !errors.As(err, &failure) {
		return false
	}
	return failure.Class == PolicySessionStartClass || policySessionLost(err)
}

// Session is the session the episode should persist after this turn and
// whether it differs from the one it held. An ephemeral turn persists none.
func (binding *PolicySessionBinding) Session() (*PolicySession, bool) {
	if binding == nil || binding.ephemeral || binding.current == nil {
		return nil, false
	}
	changed := binding.persisted == nil || *binding.persisted != *binding.current
	return clonePolicySession(binding.current), changed
}

// Sessions is how many sessions the turn's calls used, and whether it moved.
func (binding *PolicySessionBinding) Sessions() (int, bool) {
	if binding == nil {
		return 0, false
	}
	return binding.sessions, binding.moved || binding.persisted != nil && binding.sessions > 1
}

// Append extends the staged turn on the session this turn decided on (ADR
// 0155). It never moves the turn: the append only prepares the next one, so
// a session that is gone fails it.
func (binding *PolicySessionBinding) Append(
	ctx context.Context,
	request PolicyAppendRequest,
) (*PolicyAppendResponse, error) {
	if binding == nil || binding.current == nil {
		return nil, &PolicyServiceError{Class: PolicySessionGoneClass}
	}
	return binding.backend.client.WithSessionToken(binding.current.Token).Append(ctx, request)
}

// Release ends the sessions an ephemeral turn started; the episode keeps no
// record of them. A persisting turn keeps its session for the next turn.
func (binding *PolicySessionBinding) Release(ctx context.Context) {
	if binding == nil || !binding.ephemeral {
		return
	}
	for _, session := range binding.owned {
		binding.terminate(ctx, session)
	}
	binding.owned = nil
}
