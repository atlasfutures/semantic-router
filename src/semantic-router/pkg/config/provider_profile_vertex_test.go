package config

import "testing"

func TestResolveRequestPathVertexAI(t *testing.T) {
	const location = "https://aiplatform.googleapis.com/v1/projects/p/locations/global"
	chat := &ProviderProfile{Type: "vertex-ai", BaseURL: location + "/endpoints/openapi"}
	messages := &ProviderProfile{Type: "vertex-ai", BaseURL: location}

	cases := []struct {
		name     string
		profile  *ProviderProfile
		protocol string
		model    string
		stream   bool
		want     string
	}{
		{"chat", chat, "openai/chat-completions@1", "xai/grok", false,
			"/v1/projects/p/locations/global/endpoints/openapi/chat/completions"},
		{"chat stream keeps the path", chat, "openai/chat-completions@1", "xai/grok", true,
			"/v1/projects/p/locations/global/endpoints/openapi/chat/completions"},
		{"messages", messages, "anthropic/messages@1", "claude-x-1", false,
			"/v1/projects/p/locations/global/publishers/anthropic/models/claude-x-1:rawPredict"},
		{"messages stream", messages, "anthropic/messages@1", "claude-x-1", true,
			"/v1/projects/p/locations/global/publishers/anthropic/models/claude-x-1:streamRawPredict"},
		{"model id is path-escaped", messages, "anthropic/messages@1", "a/b c@v1", false,
			"/v1/projects/p/locations/global/publishers/anthropic/models/a%2Fb%20c@v1:rawPredict"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.profile.ResolveRequestPath(tc.protocol, tc.model, tc.stream)
			if err != nil {
				t.Fatalf("ResolveRequestPath: %v", err)
			}
			if got != tc.want {
				t.Fatalf("path = %q, want %q", got, tc.want)
			}
		})
	}

	if _, err := messages.ResolveRequestPath("anthropic/messages@1", " ", false); err == nil {
		t.Fatal("a {model} path with no model id resolved; want error")
	}
}

func TestResolveRequestPathOtherProvidersUnchanged(t *testing.T) {
	cases := []*ProviderProfile{
		{Type: "openrouter", BaseURL: "https://openrouter.ai/api/v1"},
		{Type: "anthropic", BaseURL: "https://api.anthropic.com"},
	}
	for _, profile := range cases {
		for _, stream := range []bool{false, true} {
			want, err := profile.ResolveCreatePath("anthropic/messages@1")
			if err != nil {
				t.Fatalf("%s: ResolveCreatePath: %v", profile.Type, err)
			}
			got, err := profile.ResolveRequestPath("anthropic/messages@1", "some-model", stream)
			if err != nil || got != want {
				t.Fatalf("%s stream=%v: path = %q, %v; want %q", profile.Type, stream, got, err, want)
			}
		}
	}
}

func TestVertexStreamPath(t *testing.T) {
	cases := map[string]string{
		"/m/x:rawPredict":         "/m/x:streamRawPredict",
		"/m/x:rawPredict?a=b":     "/m/x:streamRawPredict?a=b",
		"/m/x:streamRawPredict":   "/m/x:streamRawPredict",
		"/endpoints/openapi/chat": "/endpoints/openapi/chat",
	}
	for in, want := range cases {
		if got := vertexStreamPath(in); got != want {
			t.Errorf("vertexStreamPath(%q) = %q, want %q", in, got, want)
		}
	}
}
