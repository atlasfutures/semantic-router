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
	"strconv"
	"strings"
	"testing"
	"unicode"
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

// qwenExactnessIsSound reports whether Go's Unicode tables classify every
// code point as the reference tokenizer does, which holds on Unicode 17, the
// edition the reference uses. Go ships Unicode 17 from 1.27; 1.25 and 1.26
// carry Unicode 15.
//
// Under an older edition isSymbol's negation reads a code point assigned
// since Go's tables as a symbol where the reference reads a letter, so a
// symbol run takes a character that would have begun a word and the count
// comes out a few tokens high while CountExact still reports exact. Four of
// the random strings below hit that on Unicode 15. The count is never low on
// any edition, so capacity admission holds regardless; what a wrong exact
// costs is the provenance label, local_measured where caller_estimated was
// due (see localPromptSize).
//
// A newer edition than the reference's is not sound either: the same
// mechanism runs in reverse, Go reading a letter where the reference reads a
// symbol, so asserting exactness there would report the reference's own
// staleness as a defect in this package. Revisit the upper bound when Go
// carries Unicode 18, against the edition the reference has by then.
func qwenExactnessIsSound() bool { return unicodeMajor() == 17 }

// qwenUnsoundDrift is how far the random strings below overcount in total
// while still reporting exact on an edition that is not sound: ten tokens
// over four strings, on Unicode 15 (+1, +1, +4, +4).
//
// The test holds an unsound edition to this budget rather than excusing it,
// so the known drift passes while a regression that widens the exact claim
// still fails on a toolchain the gate would otherwise excuse --
// Dockerfile.extproc-rocm builds on 1.25, so that path ships. Budgeting the
// excess tokens rather than the number of drifting strings also catches a
// worsening overcount on a string that already drifts. A trade that
// withdraws exact from one string and overcounts another by as much keeps
// the total and passes here; holding each string to its own delta would
// catch it, at the cost of recording per-edition indices, and the
// reference's own edition catches it anyway.
//
// Pinning the drifting indices instead would record a property of Go's
// tables, and which strings drift differs per edition; the total a wrong
// exact can cost is what this package can hold itself to.
const qwenUnsoundDrift = 10

func unicodeMajor() int {
	major, _, _ := strings.Cut(unicode.Version, ".")
	value, err := strconv.Atoi(major)
	if err != nil {
		return 0
	}
	return value
}

// TestQwen38NeverUndercountsRandomText runs seeded random strings over the
// review's character pools (testdata/make_random.py): no count is below the
// reference, which is the property capacity admission rests on, and on a
// toolchain whose tables match the reference a count reporting exact equals
// it (qwenExactnessIsSound).
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
	sound := qwenExactnessIsSound()
	exactCount, drifted, excess := 0, 0, 0
	for index, c := range cases {
		got, exact := counter.CountExact(c.Text)
		if got < c.Tokens {
			t.Errorf("case %d (%.40q): core %d is below the reference %d", index, c.Text, got, c.Tokens)
		}
		if exact && got != c.Tokens {
			drifted++
			excess += got - c.Tokens
			if sound {
				t.Errorf("case %d (%.40q): core %d reports exact, reference %d", index, c.Text, got, c.Tokens)
			}
		}
		if exact {
			exactCount++
		}
	}
	if !sound {
		switch {
		case excess > qwenUnsoundDrift:
			t.Errorf("Unicode %s overcounts by %d tokens across %d strings that report exact, more than the %d this edition is known to drift: CountExact has widened the exact claim", unicode.Version, excess, drifted, qwenUnsoundDrift)
		case excess < qwenUnsoundDrift:
			t.Logf("Unicode %s overcounts by %d tokens, under the %d recorded; if CountExact now withdraws exact for those strings, lower qwenUnsoundDrift", unicode.Version, excess, qwenUnsoundDrift)
		default:
			t.Logf("Unicode %s overcounts by %d tokens across %d of %d strings that report exact; expected on this toolchain, not a regression (go1.27 carries Unicode 17 and counts all of them exactly)", unicode.Version, excess, drifted, len(cases))
		}
	}
	t.Logf("Unicode %s: %d strings, %d counted exactly, none under, %d wrongly exact by %d tokens", unicode.Version, len(cases), exactCount, drifted, excess)
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
