package reasoning

import "testing"

func TestIssuerIdentitySeparatesProviderModelAndCredential(t *testing.T) {
	original := EncryptedIssuer("worker", "backend", "model", "key", true, false)
	for _, changed := range []string{
		EncryptedIssuer("other", "backend", "model", "key", true, false),
		EncryptedIssuer("worker", "other", "model", "key", true, false),
		EncryptedIssuer("worker", "backend", "other", "key", true, false),
		EncryptedIssuer("worker", "backend", "model", "rotated", true, false),
	} {
		if changed == original || ReasoningIssuersAre([]string{original}, changed) {
			t.Fatal("a different encrypted-reasoning issuer was considered readable")
		}
	}
	if EncryptedIssuer("worker", "backend", "model", "key", false, false) != ReasoningIssuerUnknown || EncryptedIssuer("worker", "backend", "model", "key", true, true) != ReasoningIssuerUnknown {
		t.Fatal("an unresolved account or varying provider has a known issuer")
	}
	if OpaqueIssuer("worker", "backend", "model", true, "") != ReasoningIssuerUnknown {
		t.Fatal("an unpinned opaque issuer is known")
	}
	if OpaqueIssuer("worker", "backend", "model", true, "one") == OpaqueIssuer("worker", "backend", "model", true, "two") {
		t.Fatal("different OpenRouter providers share an opaque issuer")
	}
}
