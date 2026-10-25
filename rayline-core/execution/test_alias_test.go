package execution

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"
)

func aliasTestConfig(t *testing.T) (Config, *llamaEndpointFixture) {
	t.Helper()
	local, fixture := llamaCounterFixture(t)
	config := currentTestConfig(t)
	config.AllowExperimentalControls = true
	target := config.Actions[0]
	source := target
	source.ID = strings.Repeat("d", 64)
	source.Worker = "kimi-k3"
	source.Model = "kimi-k3"
	source.DeploymentID = "selected-cloud"
	source.ControlID = "605a61ada934f704248cb99cc84179bf15747302120d11e67ba723db0d8d955b"
	source.Control = publicControl(t, source.ControlID)
	cloud := local
	cloud.ID = source.DeploymentID
	cloud.Kind = "cloud"
	cloud.CloudTransport = cloudModelPin
	cloud.URL = "http://127.0.0.1:1/v1/chat/completions"
	cloud.WireModel = "moonshotai/kimi-k3"
	cloud.RegistryModel = cloud.WireModel
	cloud.RegistryProvider = "openrouter"
	cloud.PromptCounter = nil
	cloud.AuthEnv = "ALIAS_CLOUD_KEY"
	cloud.ContextWindow = 32768
	cloud.MaxOutputTokens = 4096
	cloud.InputModalities = []string{"text", "image"}
	t.Setenv(cloud.AuthEnv, "test-cloud-key")
	config.Actions = []Action{source, target}
	config.Deployments = []Deployment{cloud, local}
	config.TestAlias = &TestAlias{SelectedModel: source.Model, ExecutionActionID: target.ID}
	return config, fixture
}

func TestExplicitAliasExecutesQualifiedNeutralQwenAndSeparatesEvidence(t *testing.T) {
	config, fixture := aliasTestConfig(t)
	// No transport is claimed for the selected source. Only the actual local
	// target is qualified; the package/control identity still binds selection.
	config.Actions[0].DeploymentID = ""
	config.Actions[0].ReportedModel = "moonshotai/kimi-k3"
	config.Deployments = config.Deployments[1:]
	source, target := config.Actions[0], config.Actions[1]
	provider := currentDecisionFixture(t, config, map[string]any{"route_id": "route-alias", "model": "moonshotai/kimi-k3", "worker": source.Worker, "policy_action_id": source.ID})
	registry, err := thinkingcontrol.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	core, err := NewServer(config, registry, provider, testCoreToken)
	if err != nil {
		t.Fatal(err)
	}
	if err := core.RegisterConfiguredPromptCounters(t.Context()); err != nil {
		t.Fatal(err)
	}
	httpServer := httptest.NewServer(core)
	defer httpServer.Close()
	status, body := executeHTTP(t, httpServer.URL, "/v1/chat/completions", `{"model":"arc-1.0","messages":[{"role":"user","content":"The original task."}]}`, "alias-session")
	if status != 200 {
		t.Fatalf("alias execution %d: %s", status, body)
	}
	fixture.mu.Lock()
	defer fixture.mu.Unlock()
	if len(fixture.generated) != 1 {
		t.Fatalf("generation count %d", len(fixture.generated))
	}
	var wire struct {
		Model    string `json:"model"`
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(fixture.generated[0], &wire); err != nil {
		t.Fatal(err)
	}
	if wire.Model != "qwen-fixture" || len(wire.Messages) != 1 || wire.Messages[0].Content != "The original task." {
		t.Fatalf("source-model steering leaked into local Qwen: %s", fixture.generated[0])
	}
	state, err := core.states.acquire(t.Context(), "alias-session", "main")
	if err != nil {
		t.Fatal(err)
	}
	defer state.release()
	evidence := state.workers[target.DeploymentID]
	if evidence.SelectedActionID != source.ID || evidence.SelectedModel != source.Model || evidence.ExecutedModel != target.Model || evidence.Call.ActionID != target.ID || state.held != target.ID {
		t.Fatalf("alias evidence conflated models: %+v", evidence)
	}
	if core.capabilities.ContextWindow != 16384 || core.capabilities.MaxOutputTokens != 8192 || strings.Join(core.capabilities.InputModalities, ",") != "text" {
		t.Fatalf("capabilities included a source-only deployment: %+v", core.capabilities)
	}
}

func TestAliasIsOptInAndCannotReuseSourceControl(t *testing.T) {
	config, _ := aliasTestConfig(t)
	registry, _ := thinkingcontrol.Embedded()
	for _, kind := range []string{"advanced mode", "source model missing", "nonlocal target", "unqualified target", "steered target", "implicit substitution"} {
		t.Run(kind, func(t *testing.T) {
			copy := config
			copy.Actions = append([]Action(nil), config.Actions...)
			copy.Deployments = append([]Deployment(nil), config.Deployments...)
			alias := *config.TestAlias
			copy.TestAlias = &alias
			switch kind {
			case "advanced mode":
				copy.Decisions.APIMode = DecisionsExecutionContextV1
			case "source model missing":
				alias.SelectedModel = "absent"
			case "nonlocal target":
				alias.ExecutionActionID = copy.Actions[0].ID
				alias.SelectedModel = copy.Actions[1].Model
			case "unqualified target":
				copy.Deployments[1].PromptCounter = nil
			case "steered target":
				copy.Actions[1].Control = copy.Actions[0].Control
				copy.Actions[1].ControlID = copy.Actions[0].ControlID
			case "implicit substitution":
				copy.TestAlias = nil
				copy.Actions[0].DeploymentID = copy.Deployments[1].ID
			}
			if _, err := NewServer(copy, registry, &HTTPDecisions{}, testCoreToken); err == nil {
				t.Fatal("unsafe or implicit alias accepted")
			}
		})
	}
	config.TestAlias = nil
	core, err := NewServer(config, registry, &HTTPDecisions{}, testCoreToken)
	if err != nil {
		t.Fatal(err)
	}
	source := config.Actions[0]
	selected, execution, reason, err := core.executionFor(Decision{ActionID: source.ID, Worker: source.Worker, Model: source.Model, ControlID: source.ControlID, PackageSHA256: config.PackageSHA256})
	if err != nil || reason != "" || execution.ActionID != source.ID || selected.deployment.Kind != "cloud" {
		t.Fatal("omitted alias silently substituted a model")
	}
}

func TestCapabilitiesRequireExplicitQualifiedCommonLimits(t *testing.T) {
	config := currentTestConfig(t)
	registry, _ := thinkingcontrol.Embedded()
	for _, kind := range []string{"missing context", "output exceeds context", "missing text", "unknown modality", "duplicate modality"} {
		t.Run(kind, func(t *testing.T) {
			copy := config
			copy.Deployments = append([]Deployment(nil), config.Deployments...)
			d := &copy.Deployments[0]
			switch kind {
			case "missing context":
				d.ContextWindow = 0
			case "output exceeds context":
				d.ContextWindow = 1024
			case "missing text":
				d.InputModalities = []string{"image"}
			case "unknown modality":
				d.InputModalities = []string{"text", "audio"}
			case "duplicate modality":
				d.InputModalities = []string{"text", "text"}
			}
			if _, err := NewServer(copy, registry, &HTTPDecisions{}, testCoreToken); err == nil {
				t.Fatal("unqualified capabilities accepted")
			}
		})
	}
}

func TestSelectionOnlyAliasSourceRequiresExactBinding(t *testing.T) {
	config, _ := aliasTestConfig(t)
	config.Actions[0].DeploymentID = ""
	config.Actions[0].ReportedModel = "moonshotai/kimi-k3"
	config.Deployments = config.Deployments[1:]
	registry, _ := thinkingcontrol.Embedded()
	for _, kind := range []string{"missing alias", "missing reported model", "wrong model alias", "wrong control hash"} {
		t.Run(kind, func(t *testing.T) {
			copy := config
			copy.Actions = append([]Action(nil), config.Actions...)
			switch kind {
			case "missing alias":
				copy.TestAlias = nil
			case "missing reported model":
				copy.Actions[0].ReportedModel = ""
			case "wrong model alias":
				copy.Actions[0].ReportedModel = "openai/gpt-6-luna"
			case "wrong control hash":
				copy.Actions[0].ControlID = strings.Repeat("f", 64)
			}
			if _, err := NewServer(copy, registry, &HTTPDecisions{}, testCoreToken); err == nil {
				t.Fatal("invalid selection-only binding accepted")
			}
		})
	}
}
