package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A provider credential written under a custom auth_header (one no naming
// convention marks as secret) is still redacted from the debug dump of the
// header mutation.
func TestCredentialUnderACustomAuthHeaderIsRedacted(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(`
version: v0.3
providers:
  models:
    - name: private
      provider_model_id: private
      api_format: openai
      backend_refs:
        - provider: vllm
          endpoint: https://private.example/v1
          auth_header: x-credential-carrier
          auth_prefix: ""
          api_key: static-secret-value
routing: {}
`))
	if err != nil {
		t.Fatal(err)
	}
	if isSensitiveHeaderKey("x-credential-carrier") {
		t.Fatal("the fixture's header is already sensitive by name; pick another")
	}
	router := &OpenAIRouter{Config: cfg, CredentialResolver: authz.NewCredentialResolver(authz.NewStaticConfigProvider(cfg))}
	profile, err := cfg.GetProviderProfileForEndpoint("private_primary")
	if err != nil {
		t.Fatal(err)
	}
	state := &routeHeaderState{profile: profile}
	if response := router.appendProviderCredential(
		state, "private", "private_primary", &RequestContext{Headers: map[string]string{}},
	); response != nil {
		t.Fatalf("appendProviderCredential() returned error response: %#v", response)
	}
	dump := logFormat(redactResponseForLog(requestBodyResponseWithHeaders(state.setHeaders...)))
	if strings.Contains(dump, "static-secret-value") {
		t.Fatalf("the credential reached the log dump: %s", dump)
	}
	if !isSensitiveHeaderKey("X-Credential-Carrier") {
		t.Fatal("the credential header is not marked sensitive")
	}
}
