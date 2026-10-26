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
	"strings"
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
		got, exact := counter.CountExact(string(text))
		want := manifest.Fixtures[name].Tokens
		t.Logf("%-34s reference=%6d core=%6d exact=%v", name, want, got, exact)
		if got != want || !exact {
			t.Errorf("%s: core %d (exact=%v), reference %d", name, got, exact, want)
		}
	}
}

// TestQwen38NeverUndercountsRandomText runs seeded random strings over the
// review's character pools (testdata/make_random.py): an exact count equals
// the reference, and no count is below it.
func TestQwen38NeverUndercountsRandomText(t *testing.T) {
	counter, err := Qwen38()
	if err != nil {
		t.Fatal(err)
	}
	file, err := os.Open("testdata/random.json.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	reader, err := gzip.NewReader(file)
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Text   string `json:"text"`
		Tokens int    `json:"qwen_tokens"`
	}
	if err := json.NewDecoder(reader).Decode(&cases); err != nil {
		t.Fatal(err)
	}
	exactCount := 0
	for index, c := range cases {
		got, exact := counter.CountExact(c.Text)
		if got < c.Tokens || exact && got != c.Tokens {
			t.Errorf("case %d (%.40q): core %d exact=%v, reference %d", index, c.Text, got, exact, c.Tokens)
		}
		if exact {
			exactCount++
		}
	}
	t.Logf("%d strings, %d counted exactly, none under", len(cases), exactCount)
}

func TestQwen38NeverTrustsXTextComposition(t *testing.T) {
	counter, err := Qwen38()
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		text string
		real int
	}{
		// Unicode 16 composition the reference does not apply.
		{strings.Repeat("\U00011390\U000113C9", 20000), 160000},
		// x/text's truncated key composes U+10041 + U+0301 to U+00C1.
		{strings.Repeat("a\U00010041́", 1000), 6000},
	} {
		if got := counter.Count(tc.text); got < tc.real {
			t.Errorf("%.12q: %d below the reference %d", tc.text, got, tc.real)
		}
	}
	// Decomposed accents compose exactly.
	if got, exact := counter.CountExact("café résumé"); !exact || got != counter.Count("café résumé") {
		t.Errorf("NFD text counted %d exact=%v", got, exact)
	}
}
