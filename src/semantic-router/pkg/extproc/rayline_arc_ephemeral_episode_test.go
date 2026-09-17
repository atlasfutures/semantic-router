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

package extproc

import "testing"

// A lookup that joins no conversation must touch no episode state. The two
// observable halves of that are a nil State, which is what makes the selector
// build a fresh in-memory episode, and a nil transaction, which is what says
// no lease was taken and nothing was written.
func TestEphemeralEpisodePreparesNothing(t *testing.T) {
	router, requestContext, algorithm := missingSessionRequestContext(t, "")
	arcContext := router.buildRaylineARCSelectionContext(
		algorithm,
		requestContext,
		missingSessionModelRefs(),
		raylineARCEpisodeEphemeral,
	)
	if arcContext == nil {
		t.Fatal("no ARC selection context was built")
	}
	if arcContext.PreparationFailure != "" {
		t.Fatalf("preparation failure = %q, want none", arcContext.PreparationFailure)
	}
	if arcContext.State != nil {
		t.Fatal("an ephemeral lookup carries episode state, want none")
	}
	if requestContext.RaylineARCTransaction != nil {
		t.Fatal("an ephemeral lookup took an episode lease, want none")
	}
	if requestContext.SelectionTransaction != nil {
		t.Fatal("an ephemeral lookup bound a selection transaction, want none")
	}
	// The encoder still needs a session key, and it is the encoder's prefix
	// cache key. A shared or derived one would braid unrelated callers into a
	// single encoder session.
	if arcContext.EpisodeIDHash == "" {
		t.Fatal("no episode hash was minted, but the encoder needs a session key")
	}
	if len(arcContext.Turns) == 0 {
		t.Fatal("no turns were projected, so the lookup would score nothing")
	}
}

func TestEphemeralEpisodeIdentitiesDoNotCollide(t *testing.T) {
	seen := map[string]bool{}
	for attempt := 0; attempt < 4; attempt++ {
		router, requestContext, algorithm := missingSessionRequestContext(t, "")
		arcContext := router.buildRaylineARCSelectionContext(
			algorithm,
			requestContext,
			missingSessionModelRefs(),
			raylineARCEpisodeEphemeral,
		)
		if seen[arcContext.EpisodeIDHash] {
			t.Fatalf("episode hash %q was minted twice", arcContext.EpisodeIDHash)
		}
		seen[arcContext.EpisodeIDHash] = true
	}
}

// A named conversation still takes the lease, because that is what serializes
// concurrent turns and what gives the encoder a stable prefix cache key.
func TestEphemeralModeStillPreparesANamedEpisode(t *testing.T) {
	router, requestContext, algorithm := missingSessionRequestContext(t, "conv-1")
	arcContext := router.buildRaylineARCSelectionContext(
		algorithm,
		requestContext,
		missingSessionModelRefs(),
		raylineARCEpisodeEphemeral,
	)
	if arcContext.PreparationFailure != "" {
		t.Fatalf("preparation failure = %q, want none", arcContext.PreparationFailure)
	}
	if arcContext.State == nil {
		t.Fatal("a named episode carries no state, want the prepared episode")
	}
	if requestContext.RaylineARCTransaction == nil {
		t.Fatal("a named episode took no lease, want one")
	}
}
