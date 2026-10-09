package config

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func gzipBytes(t *testing.T, plain []byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	writer := gzip.NewWriter(&buf)
	if _, err := writer.Write(plain); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// policyV5GzipDecision serves the v5 fixture from a gzip-compressed
// package.json, pinned by the sha256 of the decompressed manifest.
// It is mounted where the lab mounts it, as package.json.gz.
func policyV5GzipDecision(t *testing.T, plain []byte) (*RouterConfig, Decision) {
	t.Helper()
	cfg, decision := policyV5Decision(t, gzipBytes(t, plain))
	policy := decision.Algorithm.RaylineARC.PolicyService
	gz := policy.PackageManifest + ".gz"
	if err := os.Rename(policy.PackageManifest, gz); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(plain)
	policy.PackageManifest, policy.PackageSHA256 = gz, hex.EncodeToString(sum[:])
	return cfg, decision
}

// A Secret Manager version holds at most 64 KiB, so a larger manifest is
// mounted gzip-compressed under the same name. It loads exactly as the plain
// one does, and package_sha256 stays the sha256 of the plain manifest.
func TestRaylineARCPolicyPackageV5LoadsGzip(t *testing.T) {
	plain := readPolicyV5Fixture(t)
	for name, build := range map[string]func(*testing.T, []byte) (*RouterConfig, Decision){
		"plain": policyV5Decision,
		"gzip":  policyV5GzipDecision,
	} {
		t.Run(name, func(t *testing.T) {
			cfg, decision := build(t, plain)
			if err := validatePolicyDispatch(cfg, decision); err != nil {
				t.Fatalf("the v5 fixture refused: %v", err)
			}
			action, ok := decision.Algorithm.RaylineARC.PolicyService.PackageV5Action(policyV5GLMUp)
			if !ok || action.Model != "glm-5.3-flash" {
				t.Fatalf("PackageV5Action = %+v, %v", action, ok)
			}
		})
	}
}

func TestRaylineARCPolicyPackageV5GzipRefusals(t *testing.T) {
	plain := readPolicyV5Fixture(t)

	t.Run("sha256 of the compressed bytes", func(t *testing.T) {
		// The pin is over the decompressed manifest: a sha256 taken over the
		// compressed file does not match it.
		cfg, decision := policyV5Decision(t, gzipBytes(t, plain))
		err := validatePolicyDispatch(cfg, decision)
		if err == nil || !strings.Contains(err.Error(), "not package_sha256") {
			t.Fatalf("err = %v, want a sha256 refusal", err)
		}
	})

	t.Run("another manifest decompressed", func(t *testing.T) {
		cfg, decision := policyV5GzipDecision(t, plain)
		other := append(append([]byte{}, plain...), '\n')
		path := decision.Algorithm.RaylineARC.PolicyService.PackageManifest
		if err := os.WriteFile(path, gzipBytes(t, other), 0o600); err != nil {
			t.Fatal(err)
		}
		err := validatePolicyDispatch(cfg, decision)
		if err == nil || !strings.Contains(err.Error(), "not package_sha256") {
			t.Fatalf("err = %v, want a sha256 refusal", err)
		}
	})

	t.Run("a .gz path holding plain JSON", func(t *testing.T) {
		cfg, decision := policyV5GzipDecision(t, plain)
		path := decision.Algorithm.RaylineARC.PolicyService.PackageManifest
		if err := os.WriteFile(path, plain, 0o600); err != nil {
			t.Fatal(err)
		}
		err := validatePolicyDispatch(cfg, decision)
		if err == nil || !strings.Contains(err.Error(), "not gzip-compressed") {
			t.Fatalf("err = %v, want a refusal", err)
		}
	})

	t.Run("corrupt gzip", func(t *testing.T) {
		compressed := gzipBytes(t, plain)
		cfg, decision := policyV5GzipDecision(t, plain)
		path := decision.Algorithm.RaylineARC.PolicyService.PackageManifest
		if err := os.WriteFile(path, compressed[:len(compressed)/2], 0o600); err != nil {
			t.Fatal(err)
		}
		err := validatePolicyDispatch(cfg, decision)
		if err == nil || !strings.Contains(err.Error(), "gzip") {
			t.Fatalf("err = %v, want a gzip refusal", err)
		}
	})
}

func TestReadPackageManifestBoundsDecompression(t *testing.T) {
	dir := t.TempDir()
	write := func(name string, size int) string {
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, gzipBytes(t, bytes.Repeat([]byte{' '}, size)), 0o600); err != nil {
			t.Fatal(err)
		}
		return path
	}
	if got, err := readPackageManifest(write("at-cap.json", maxPackageManifestBytes)); err != nil ||
		len(got) != maxPackageManifestBytes {
		t.Fatalf("a manifest at the cap: %d bytes, %v", len(got), err)
	}
	if _, err := readPackageManifest(write("over-cap.json", maxPackageManifestBytes+1)); err == nil ||
		!strings.Contains(err.Error(), "4 MiB") {
		t.Fatalf("err = %v, want the size cap", err)
	}
}
