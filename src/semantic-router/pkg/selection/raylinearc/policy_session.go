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
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
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
	IdleTimeout  time.Duration
	StartTimeout time.Duration
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
	return &PolicySessionBackend{
		client: client, controlHTTP: noRedirectClient(client.http), config: config, now: time.Now, newKey: idempotencyKey,
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
	header.Set("Modal-Authorization", "Bearer "+session.Token)
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
// timeout or transport failure is returned as is: the container may have
// advanced its session, so the turn fails closed. It is not safe for
// concurrent use.
type PolicySessionBinding struct {
	backend   *PolicySessionBackend
	persisted *PolicySession
	ephemeral bool
	current   *PolicySession
	// owned are the sessions this turn started; an ephemeral turn ends them.
	owned    []PolicySession
	sessions int
	moved    bool
}

// Decide asks the bound session for one decision.
func (binding *PolicySessionBinding) Decide(
	ctx context.Context,
	request PolicyDecisionRequest,
) (*PolicyDecisionResponse, error) {
	if binding.current == nil {
		if binding.persisted != nil {
			binding.use(*binding.persisted, false)
		} else if err := binding.start(ctx); err != nil {
			return nil, err
		}
	}
	response, err := binding.backend.client.WithSessionToken(binding.current.Token).Decide(ctx, request)
	stale := err == nil && staleOwner(request, response)
	if !stale && !policySessionLost(err) {
		return response, err
	}
	if stale {
		// A rollover can leave the session on a container still serving the
		// previous build. End it; the caller refuses the stale answer itself
		// (policy_package_mismatch) if no new session replaces it.
		binding.terminate(ctx, *binding.current)
	}
	if binding.moved {
		return response, err
	}
	binding.moved = true
	if startErr := binding.start(ctx); startErr != nil {
		if stale {
			return response, nil
		}
		return nil, startErr
	}
	response, err = binding.backend.client.WithSessionToken(binding.current.Token).Decide(ctx, request)
	if err == nil && staleOwner(request, response) {
		binding.terminate(ctx, *binding.current)
	}
	return response, err
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
