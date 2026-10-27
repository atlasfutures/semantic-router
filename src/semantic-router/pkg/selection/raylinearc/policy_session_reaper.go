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
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// A session start the cell gives up on (the caller went away or the start
// timeout lapsed) is not withdrawn by closing its request: Modal keeps placing
// the session, and it holds one of the Server's slots until the idle timeout
// with no one holding its token. On a saturated Server every retried turn
// orphans another slot, so the overload sustains itself (a 10-27 dev fault: a
// 160-slot Server took no session for about 15 min with no real load).
//
// Nor can it be recovered later: a repeated start with the same
// x-idempotency-key and x-modal-timestamp starts another session (measured on
// a dev Server, 10-27). So the start's own request is never abandoned: it runs
// on a context detached from the caller, and when the caller gives up first, a
// reaper keeps waiting for that same response, bounded by the reap timeout,
// and ends the session it brings. Reaps are bounded; a start beyond the bound
// is cancelled as before, and counted.
const (
	// DefaultPolicySessionReapTimeout bounds a start's request, the wait of a
	// reap included. Modal answers once the session is placed, which on a
	// saturated Server takes as long as the queue.
	DefaultPolicySessionReapTimeout = 2 * time.Minute
	// DefaultPolicySessionMaxReaps bounds the reaps in flight per backend.
	DefaultPolicySessionMaxReaps = 8

	policySessionReapEvent = "rayline_arc_policy_session_start_reap"

	policySessionReaped     = "reaped"
	policySessionNotCreated = "not_created"
	policySessionReapFailed = "failed"
	policySessionReapDrop   = "dropped"
)

type policySessionStartAnswer struct {
	status int
	body   []byte
	err    error
}

// awaitStart sends a start and waits for its answer up to the start timeout
// or the caller's end. A start given up on is handed to a reaper.
func (backend *PolicySessionBackend) awaitStart(ctx context.Context, header http.Header) (int, []byte, error) {
	if err := ctx.Err(); err != nil {
		return 0, nil, err
	}
	timeout := backend.config.ReapTimeout
	if timeout <= 0 {
		timeout = DefaultPolicySessionReapTimeout
	}
	// The request outlives the caller's wait, never the other way round.
	timeout = max(timeout, backend.config.StartTimeout)
	startCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	answers := make(chan policySessionStartAnswer, 1)
	go func() {
		status, body, err := backend.control(startCtx, policySessionStartPath, header, timeout)
		answers <- policySessionStartAnswer{status: status, body: body, err: err}
	}()
	wait := time.NewTimer(backend.config.StartTimeout)
	defer wait.Stop()
	select {
	case answer := <-answers:
		cancel()
		return answer.status, answer.body, answer.err
	case <-ctx.Done():
	case <-wait.C:
	}
	backend.reap(answers, cancel)
	return 0, nil, errors.New("ARC policy session start was given up on")
}

// reap waits, in the background, for the answer to a start the caller gave up
// on, and ends the session it brings.
func (backend *PolicySessionBackend) reap(answers <-chan policySessionStartAnswer, cancel context.CancelFunc) {
	backend.reapInit.Do(func() {
		limit := backend.config.MaxReaps
		if limit <= 0 {
			limit = DefaultPolicySessionMaxReaps
		}
		backend.reapSlots = make(chan struct{}, limit)
	})
	select {
	case backend.reapSlots <- struct{}{}:
	default:
		cancel()
		backend.reapDone(policySessionReapDrop, map[string]interface{}{"dropped": backend.reapDropped.Add(1)})
		return
	}
	go func() {
		defer func() { <-backend.reapSlots }()
		defer cancel()
		began := time.Now()
		answer := <-answers
		outcome := backend.reapAnswer(answer)
		backend.reapDone(outcome, map[string]interface{}{"status": answer.status, "elapsed_ms": time.Since(began).Milliseconds()})
	}()
}

// reapAnswer ends the session a start's answer brings. A refusal started none;
// no answer, or a 200 whose session cannot be read, leaves it unknown.
func (backend *PolicySessionBackend) reapAnswer(answer policySessionStartAnswer) string {
	switch {
	case answer.status != 0 && answer.status != http.StatusOK:
		return policySessionNotCreated
	case answer.err != nil:
		return policySessionReapFailed
	}
	session, ok := decodePolicySessionStart(answer.body)
	if !ok {
		return policySessionReapFailed
	}
	// Terminate bounds itself.
	if backend.Terminate(context.Background(), session) != nil {
		return policySessionReapFailed
	}
	return policySessionReaped
}

// reapDone logs a reap's outcome; never the session or its token.
func (backend *PolicySessionBackend) reapDone(outcome string, fields map[string]interface{}) {
	fields["outcome"] = outcome
	if outcome == policySessionReaped || outcome == policySessionNotCreated {
		logging.ComponentEvent("selection", policySessionReapEvent, fields)
	} else {
		logging.ComponentWarnEvent("selection", policySessionReapEvent, fields)
	}
	if backend.onReap != nil {
		backend.onReap(outcome)
	}
}
