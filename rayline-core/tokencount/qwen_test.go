package tokencount

import (
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

func TestQwen38MergesMatchTheRecordedSnapshot(t *testing.T) {
	reader, err := gzip.NewReader(bytesReader(qwen38Merges))
	if err != nil {
		t.Fatal(err)
	}
	raw, err := io.ReadAll(reader)
	if err != nil {
		t.Fatal(err)
	}
	if sum := sha256.Sum256(raw); hex.EncodeToString(sum[:]) != Qwen38MergesSHA256 {
		t.Fatalf("embedded merges sha256 %x", sum)
	}
}

// TestQwen38CountsEqualTheReferenceTokenizer compares Count with the real
// Qwen3.8-27B tokenizer's counts for every fixture in
// ../execution/testdata/capacity (counts.json).
func TestQwen38CountsEqualTheReferenceTokenizer(t *testing.T) {
	counter, err := Qwen38()
	if err != nil {
		t.Fatal(err)
	}
	directory := "../execution/testdata/capacity"
	raw, err := os.ReadFile(filepath.Join(directory, "counts.json"))
	if err != nil {
		t.Fatal(err)
	}
	var manifest struct {
		Fixtures map[string]struct {
			Tokens int `json:"qwen_tokens"`
		} `json:"fixtures"`
	}
	if err := json.Unmarshal(raw, &manifest); err != nil {
		t.Fatal(err)
	}
	names := make([]string, 0, len(manifest.Fixtures))
	for name := range manifest.Fixtures {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		file, err := os.Open(filepath.Join(directory, name+".txt.gz"))
		if err != nil {
			t.Fatal(err)
		}
		reader, err := gzip.NewReader(file)
		if err != nil {
			t.Fatal(err)
		}
		text, err := io.ReadAll(reader)
		file.Close()
		if err != nil {
			t.Fatal(err)
		}
		got, want := counter.Count(string(text)), manifest.Fixtures[name].Tokens
		t.Logf("%-28s reference=%6d core=%6d", name, want, got)
		if got != want {
			t.Errorf("%s: core %d, reference %d", name, got, want)
		}
	}
}
