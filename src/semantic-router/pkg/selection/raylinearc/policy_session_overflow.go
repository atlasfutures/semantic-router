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
	"context"
	"errors"
	"sync"
	"time"
)

// A turn past its caller's new-session budget (policy_session_budget.go) is
// decided without a session of its own. Every request to a sessioned Server
// needs a session token, so "without a session" means on the backend's one
// shared overflow session: started on the first such turn, kept for every
// later one, and replaced only when Modal answers it lost. The turn is decided
// one-shot (relaxed, the way a side call is): the service takes no lock and
// keeps no episode state on it, so concurrent overflow turns of any caller
// share it, and the episode records no session. A cell therefore holds at
// most one overflow session per router instance, however many callers are
// past their budget.

// policyOverflowSession is the backend's shared overflow session.
type policyOverflowSession struct {
	// starting admits one start at a time; a turn waiting on it gives up
	// with its context.
	starting chan struct{}
	mu       sync.Mutex
	session  *PolicySession
}

func newPolicyOverflowSession() *policyOverflowSession {
	return &policyOverflowSession{starting: make(chan struct{}, 1)}
}

func (overflow *policyOverflowSession) load() *PolicySession {
	overflow.mu.Lock()
	defer overflow.mu.Unlock()
	return clonePolicySession(overflow.session)
}

// overflowSession answers the shared overflow session, starting it when there
// is none. Concurrent turns wait for the one start rather than each starting
// a session.
func (backend *PolicySessionBackend) overflowSession(ctx context.Context) (PolicySession, error) {
	overflow := backend.overflow
	if session := overflow.load(); session != nil {
		return *session, nil
	}
	select {
	case overflow.starting <- struct{}{}:
	case <-ctx.Done():
		return PolicySession{}, &PolicyServiceError{Class: PolicySessionStartClass}
	}
	defer func() { <-overflow.starting }()
	if session := overflow.load(); session != nil {
		return *session, nil
	}
	session, err := backend.Start(ctx)
	if err != nil {
		return PolicySession{}, err
	}
	overflow.mu.Lock()
	overflow.session = clonePolicySession(&session)
	overflow.mu.Unlock()
	return session, nil
}

// retireOverflow forgets the overflow session when it is still session, so
// the next overflow turn starts a new one. A session that answered for a stale
// package is still alive: it is ended after the replaced grace, which lets the
// overflow turns already on it finish.
func (backend *PolicySessionBackend) retireOverflow(ctx context.Context, session PolicySession, alive bool) {
	overflow := backend.overflow
	overflow.mu.Lock()
	current := overflow.session != nil && *overflow.session == session
	if current {
		overflow.session = nil
	}
	overflow.mu.Unlock()
	if !current || !alive {
		return
	}
	detached := context.WithoutCancel(ctx)
	time.AfterFunc(backend.config.ReplacedGrace, func() {
		terminateCtx, cancel := context.WithTimeout(detached, policySessionTerminateTimeout)
		defer cancel()
		_ = backend.Terminate(terminateCtx, session)
	})
}

// BindOverflow starts the routing of a turn past its caller's new-session
// budget: ephemeral (the episode records no session, so its next turn is
// counted again), on the shared overflow session, never on one it starts for
// itself. The caller sends the turn relaxed.
func (backend *PolicySessionBackend) BindOverflow() *PolicySessionBinding {
	return &PolicySessionBinding{backend: backend, ephemeral: true, overflow: true}
}

// Overflow reports whether the turn is routed on the shared overflow session.
func (binding *PolicySessionBinding) Overflow() bool {
	return binding != nil && binding.overflow
}

// decideOverflow is Decide on the overflow session. It moves at most once, as
// any turn does: when Modal answers the session lost, or it answers for a
// stale package, the session is retired and the decide is sent once on its
// replacement. The overflow session is never the binding's own, so Release
// and EndUnrecorded leave it for the next overflow turn.
func (binding *PolicySessionBinding) decideOverflow(
	ctx context.Context,
	request PolicyDecisionRequest,
) (*PolicyDecisionResponse, error) {
	if binding.current == nil {
		session, err := binding.backend.overflowSession(ctx)
		if err != nil {
			return nil, err
		}
		binding.use(session, false)
	}
	response, err := binding.decide(ctx, request)
	if policySessionUncertain(err) && !errors.Is(ctx.Err(), context.Canceled) {
		// No answer: the turn fails as any timed-out turn does, and later
		// overflow turns start a new session rather than wait on this one.
		binding.backend.retireOverflow(ctx, *binding.current, true)
		return response, err
	}
	stale := err == nil && staleOwner(request, response)
	if !stale && !policySessionLost(err) || binding.moved {
		return response, err
	}
	binding.moved = true
	binding.moveReason, binding.moveError = policySessionMoveReason(stale, false, err)
	binding.backend.retireOverflow(ctx, *binding.current, stale || policySessionUnrecognised(err))
	session, startErr := binding.backend.overflowSession(ctx)
	if startErr != nil {
		if stale {
			// Refused on its first answer, as it would have been without
			// the move (policy_package_mismatch).
			return response, err
		}
		return nil, startErr
	}
	binding.use(session, false)
	return binding.decide(ctx, request)
}
