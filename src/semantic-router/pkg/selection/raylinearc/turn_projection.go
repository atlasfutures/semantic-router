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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// ProjectTurns renders the neutral request the codec produced into the turn
// list the encoder was trained on.
//
// The router decodes every public wire format exactly once, so this is a
// projection and not a parser. It is deliberately lossy: the encoder reads
// role-tagged text and nothing else.
func ProjectTurns(
	request *llmprotocol.Request,
	options TurnOptions,
) ([]Turn, error) {
	return nil, nil
}
