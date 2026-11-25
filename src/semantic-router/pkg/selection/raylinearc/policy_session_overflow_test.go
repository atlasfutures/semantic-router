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
	"sync"
	"testing"
	"time"
)

// Every overflow turn, concurrent ones included, is decided on one shared
// session: one start for all of them, none recorded, none ended by a turn.
func TestPolicyOverflowTurnsShareOneSession(t *testing.T) {
	fake := newFakeModalServer(t, 3)
	backend := testSessionBackend(t, fake, time.Second)
	var wg sync.WaitGroup
	errs := make(chan error, 8)
	for turn := 0; turn < 8; turn++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			binding := backend.BindOverflow()
			if _, err := binding.Decide(context.Background(), sessionDecisionRequest(t)); err != nil {
				errs <- err
				return
			}
			if session, changed := binding.Session(); session != nil || changed || !binding.Ephemeral() || !binding.Overflow() {
				t.Error("an overflow turn records a session")
			}
			binding.Release(context.Background())
			binding.EndUnrecorded(context.Background())
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	starts, terminates, violations := fake.counts()
	if starts != 1 || terminates != 0 || len(violations) != 0 {
		t.Fatalf("starts %d terminates %d violations %v, want one shared session kept", starts, terminates, violations)
	}
}

// A lost overflow session is replaced once within the turn, and the next
// overflow turn uses the replacement.
func TestPolicyOverflowSessionLostIsReplaced(t *testing.T) {
	fake := newFakeModalServer(t, 3)
	backend := testSessionBackend(t, fake, time.Second)
	if _, err := sessionDecide(t, backend.BindOverflow()); err != nil {
		t.Fatal(err)
	}
	first := backend.overflow.load()
	fake.kill(fake.containerOf(*first))

	turn := backend.BindOverflow()
	if _, err := sessionDecide(t, turn); err != nil {
		t.Fatalf("the moved overflow turn failed: %v", err)
	}
	if used, moved := turn.Sessions(); used != 2 || !moved {
		t.Fatalf("sessions %d moved %t", used, moved)
	}
	second := backend.overflow.load()
	if second == nil || *second == *first {
		t.Fatal("the lost overflow session was kept")
	}
	if _, err := sessionDecide(t, backend.BindOverflow()); err != nil {
		t.Fatal(err)
	}
	if starts, _, _ := fake.counts(); starts != 2 {
		t.Fatalf("starts = %d, want 2", starts)
	}
}

// An overflow turn never touches a session an episode records.
func TestPolicyOverflowLeavesEpisodeSessionsAlone(t *testing.T) {
	fake := newFakeModalServer(t, 2)
	backend := testSessionBackend(t, fake, time.Second)
	main := backend.Bind(nil, false)
	if _, err := sessionDecide(t, main); err != nil {
		t.Fatal(err)
	}
	persisted, _ := main.Session()
	overflow := backend.BindOverflow()
	if _, err := sessionDecide(t, overflow); err != nil {
		t.Fatal(err)
	}
	overflow.Release(context.Background())
	if fake.containerOf(*persisted) == -1 || *backend.overflow.load() == *persisted {
		t.Fatal("the overflow turn used or ended the episode's session")
	}
}
