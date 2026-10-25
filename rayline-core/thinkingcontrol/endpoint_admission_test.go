package thinkingcontrol

import "testing"

func TestEndpointAdmissionUsesSelectedNativeBase(t *testing.T) {
	registry, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	control, ok := registry.Control("605a61ada934f704248cb99cc84179bf15747302120d11e67ba723db0d8d955b")
	if !ok {
		t.Fatal("public private-append control missing")
	}
	cell, err := registry.Admit("moonshotai/kimi-k3", "openrouter", FormatChat, control, true)
	if err != nil {
		t.Fatal(err)
	}
	for _, endpoint := range []string{"together", "moonshotai/mxfp4"} {
		if err := cell.AdmitEndpoint(control, endpoint); err != nil {
			t.Fatal(err)
		}
	}
	for _, endpoint := range []string{"deepinfra/bf16", "unknown", ""} {
		if err := cell.AdmitEndpoint(control, endpoint); err == nil {
			t.Fatalf("unqualified endpoint %q admitted", endpoint)
		}
	}
}
