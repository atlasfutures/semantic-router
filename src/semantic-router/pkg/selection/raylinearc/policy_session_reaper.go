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
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// A start that the cell gives up on (the caller went away or the start
// timeout lapsed) is not withdrawn: Modal keeps placing the session and holds
// one of the Server's slots for it until the idle timeout, with no one holding
// its token. On a saturated Server every retried turn orphans another slot, so
// the overload sustains itself (a 10-27 dev fault: a 160-slot Server took no
// session for about 15 min with no real load).
//
// The start is therefore handed to a reaper, off the turn's path: it repeats
// the start with the same x-idempotency-key and x-modal-timestamp, which is
// what Modal's SDK does when it retries a start (modal/_server.py,
// _post_session_start: one key and timestamp per start, kept across up to 5
// attempts), so the answer is the session the first attempt created, and ends
// it at once. Should Modal not dedupe on the key, the repeat starts a session
// of its own and ends it: a slot held briefly, never for the idle timeout.
// Reaps are bounded; one beyond the bound is dropped and logged.
const (
	// DefaultPolicySessionReapTimeout bounds a reap's repeated start. Modal
	// answers it once the session is placed, which on a saturated Server
	// takes as long as the queue.
	DefaultPolicySessionReapTimeout = 2 * time.Minute
	// DefaultPolicySessionMaxReaps bounds the reaps in flight per backend.
	DefaultPolicySessionMaxReaps = 8

	policySessionReapEvent = "rayline_arc_policy_session_start_reap"

	policySessionReaped     = "reaped"
	policySessionNotCreated = "not_created"
	policySessionReapFailed = "failed"
	policySessionReapDrop   = "dropped"
)

// reap ends, best effort and in the background, the session a start that got
// no answer may have created. header is that start's, idempotency key
// included.
func (backend *PolicySessionBackend) reap(ctx context.Context, header http.Header) {
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
		backend.reapDone(policySessionReapDrop, map[string]interface{}{"dropped": backend.reapDropped.Add(1)})
		return
	}
	go func() {
		defer func() { <-backend.reapSlots }()
		began := time.Now()
		outcome, status := backend.reapStart(context.WithoutCancel(ctx), header)
		backend.reapDone(outcome, map[string]interface{}{"status": status, "elapsed_ms": time.Since(began).Milliseconds()})
	}()
}

// reapStart repeats the start and ends the session it answers. A 4xx means
// the key started no session; no answer, or a 5xx (which Modal's SDK retries),
// leaves it unknown.
func (backend *PolicySessionBackend) reapStart(ctx context.Context, header http.Header) (string, int) {
	timeout := backend.config.ReapTimeout
	if timeout <= 0 {
		timeout = DefaultPolicySessionReapTimeout
	}
	status, body, err := backend.control(ctx, policySessionStartPath, header, timeout)
	switch {
	case err != nil, status >= http.StatusInternalServerError:
		return policySessionReapFailed, status
	case status != http.StatusOK:
		return policySessionNotCreated, status
	}
	session, ok := decodePolicySessionStart(body)
	if !ok {
		return policySessionReapFailed, status
	}
	// Terminate bounds itself; ctx is detached, so a slow start leaves it its
	// full bound.
	if backend.Terminate(ctx, session) != nil {
		return policySessionReapFailed, status
	}
	return policySessionReaped, status
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
