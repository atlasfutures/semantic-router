package execution

import (
	"bytes"
	"fmt"
	"math"
	"unicode/utf16"
	"unicode/utf8"
)

// The capacity estimate sizes a prepared request for a local target that has
// no prompt counter (a custom endpoint, Ollama, MLX), so an over-window turn is
// handed off before the runtime can silently truncate it. It must err high: an
// over-count costs an early handoff, an under-count a truncation.
//
// A flat bytes-per-token ratio cannot do this. Byte-level BPE vocabularies
// (Qwen splits every digit into its own token) spend about one token per
// character on numbers, hashes, base64 and random letters, but 3.5-4 bytes per
// token on prose and code. Byte-level BPE never spends more tokens than UTF-8
// bytes, so the byte count is the fallback charge; cheaper rates apply only to
// classes whose measured worst case (random text) stays below them. Over the
// decoded JSON text:
//
//   - an ASCII alphanumeric run with a digit, or long and of mixed case
//     (base64-like): 1 token per character;
//   - a letter run that is longer than a word (> estimateWordMax), all
//     upper case (>= 4), or vowel-poor (>= 4, under a quarter vowels; random
//     slugs, DNA): estimateDenseLetter per character;
//   - any other letter run (a word): 1 + len/estimateWordChars;
//   - a punctuation/symbol run: 1, then estimatePunctTail per character;
//   - a whitespace run: nothing for one space (it joins the next token),
//     otherwise 1 per estimateSpaceChars characters, at least 1;
//   - Greek, Cyrillic, Hebrew, Arabic: 1 per character (random worst 0.96);
//   - CJK unified ideographs, kana, full-width forms: 2 per character (random
//     worst 1.75);
//   - every other non-ASCII character (accented Latin, other scripts, rare
//     Han, Hangul, emoji, symbols): its UTF-8 byte count;
//   - an inline base64 image: media_tokens.
//
// The sum is raised by headroom_ratio. The calibration (real Qwen3.8-27B
// counts) is testdata/capacity and TestCapacityEstimateUndercountsNoFixture;
// content outside those fixtures is not proven.
const (
	defaultEstimateHeadroomRatio = 0.10
	// A Retina screenshot is about 6.6k Qwen-VL tokens; Claude caps an image
	// near 1.6k.
	defaultEstimateMediaTokens = 8192
	// A shorter base64 run is counted as text.
	mediaBlobMinBytes = 1024

	estimateWordChars    = 8.0
	estimateWordMax      = 16
	estimateDenseLetter  = 0.7
	estimatePunctTail    = 0.5
	estimateSpaceChars   = 4.0
	estimateMixedCaseLen = 12
)

func (estimate *CapacityEstimate) settings() (headroom float64, mediaTokens int64) {
	headroom, mediaTokens = defaultEstimateHeadroomRatio, defaultEstimateMediaTokens
	if estimate == nil {
		return
	}
	if estimate.HeadroomRatio != nil {
		headroom = *estimate.HeadroomRatio
	}
	if estimate.MediaTokens != nil {
		mediaTokens = *estimate.MediaTokens
	}
	return
}

func (estimate *CapacityEstimate) validate() error {
	headroom, mediaTokens := estimate.settings()
	if !(headroom >= 0 && headroom <= 10) || mediaTokens < 0 {
		return fmt.Errorf("capacity_estimate needs headroom_ratio in [0, 10] and nonnegative media_tokens")
	}
	return nil
}

// estimatedCapacityFact is capacityFact for a local target without a counter:
// the prompt is estimated from the exact prepared request, never measured.
func estimatedCapacityFact(deployment Deployment, body []byte, outputLimit int64) *CapacityFact {
	if deployment.ContextWindow <= 0 {
		return nil
	}
	prompt := estimatePromptTokens(body, deployment.CapacityEstimate)
	required := prompt + outputLimit + deployment.CapacityMarginTokens
	available := deployment.ContextWindow
	source := capacitySourceConfigured
	fact := &CapacityFact{Fits: required <= available, Required: &required, Available: &available, AvailableSource: &source, Provenance: capacityProvenanceEstimated, prompt: prompt}
	if deployment.TokenUnits != "" {
		units := deployment.TokenUnits
		fact.Units = &units
	}
	if !fact.Fits {
		reason := CapacityContext
		fact.Reason = &reason
	}
	return fact
}

// estimatePromptTokens is ceil((text tokens + images * media_tokens) *
// (1 + headroom_ratio)) over the prepared request body.
func estimatePromptTokens(body []byte, estimate *CapacityEstimate) int64 {
	headroom, mediaTokens := estimate.settings()
	blobs := inlineImages(body)
	tokens := float64(int64(len(blobs)) * mediaTokens)
	start := 0
	for _, blob := range blobs {
		tokens += estimateTextTokens(body[start:blob[0]])
		start = blob[1]
	}
	tokens += estimateTextTokens(body[start:])
	return int64(math.Ceil(tokens * (1 + headroom)))
}

type charClass int

const (
	classNone charClass = iota
	classAlnum
	classPunct
	classSpace
)

// estimateTextTokens charges the character classes described above. JSON
// escapes (\n, \", <) count as the one character they stand for.
func estimateTextTokens(body []byte) float64 {
	var tokens float64
	class, length, digits, upper, lower, spaces, vowels := classNone, 0, 0, 0, 0, 0, 0
	flush := func() {
		switch class {
		case classAlnum:
			switch {
			case digits > 0 || length >= estimateMixedCaseLen && upper*4 >= length && lower*4 >= length:
				tokens += float64(length)
			case length > estimateWordMax || length >= 4 && (upper == length || vowels*4 < length):
				tokens += math.Max(1, estimateDenseLetter*float64(length))
			default:
				tokens += 1 + float64(length)/estimateWordChars
			}
		case classPunct:
			tokens += 1 + float64(length-1)*estimatePunctTail
		case classSpace:
			if !(length == 1 && spaces == 1) {
				tokens += math.Max(1, float64(length)/estimateSpaceChars)
			}
		}
		class, length, digits, upper, lower, spaces, vowels = classNone, 0, 0, 0, 0, 0, 0
	}
	for index := 0; index < len(body); {
		char, size := decodeJSONChar(body[index:])
		index += size
		var next charClass
		switch {
		case char >= utf8.RuneSelf:
			flush()
			tokens += nonASCIITokens(char)
			continue
		case char >= '0' && char <= '9' || char >= 'a' && char <= 'z' || char >= 'A' && char <= 'Z':
			next = classAlnum
		case char == ' ' || char == '\n' || char == '\t' || char == '\r':
			next = classSpace
		default:
			next = classPunct
		}
		if next != class {
			flush()
			class = next
		}
		length++
		switch {
		case char >= '0' && char <= '9':
			digits++
		case char >= 'A' && char <= 'Z':
			upper++
		case char >= 'a' && char <= 'z':
			lower++
		case char == ' ':
			spaces++
		}
		switch char | 0x20 {
		case 'a', 'e', 'i', 'o', 'u':
			vowels++
		}
	}
	flush()
	return tokens
}

// nonASCIITokens charges one non-ASCII character: 1 for Greek, Cyrillic,
// Hebrew and Arabic, 2 for CJK unified ideographs, kana and full-width forms,
// and otherwise its UTF-8 byte count, which byte-level BPE never exceeds. A
// lone surrogate or invalid byte is charged as the 3-byte replacement.
func nonASCIITokens(char rune) float64 {
	switch {
	case char >= 0x0370 && char <= 0x052F || char >= 0x0590 && char <= 0x06FF:
		return 1
	case char >= 0x4E00 && char <= 0x9FFF || char >= 0x3040 && char <= 0x30FF || char >= 0xFF01 && char <= 0xFF5E:
		return 2
	}
	if size := utf8.RuneLen(char); size > 0 {
		return float64(size)
	}
	return 3
}

// decodeJSONChar reads one character, decoding a JSON string escape as the
// character it stands for. Invalid UTF-8 is one character per byte.
func decodeJSONChar(body []byte) (rune, int) {
	if body[0] == '\\' && len(body) > 1 {
		switch body[1] {
		case 'n':
			return '\n', 2
		case 't':
			return '\t', 2
		case 'r':
			return '\r', 2
		case 'u':
			value, ok := jsonHex4(body[2:])
			if !ok {
				return '\\', 1
			}
			// A surrogate pair is one character; a lone surrogate stays
			// invalid and is charged as such.
			if value >= 0xD800 && value <= 0xDBFF && len(body) >= 12 && body[6] == '\\' && body[7] == 'u' {
				if low, ok := jsonHex4(body[8:]); ok && low >= 0xDC00 && low <= 0xDFFF {
					return utf16.DecodeRune(value, low), 12
				}
			}
			return value, 6
		default:
			return rune(body[1]), 2
		}
	}
	if body[0] < utf8.RuneSelf {
		return rune(body[0]), 1
	}
	char, size := utf8.DecodeRune(body)
	return char, size
}

func jsonHex4(body []byte) (rune, bool) {
	if len(body) < 4 {
		return 0, false
	}
	var value rune
	for _, digit := range body[:4] {
		switch {
		case digit >= '0' && digit <= '9':
			value = value*16 + rune(digit-'0')
		case digit >= 'a' && digit <= 'f':
			value = value*16 + rune(digit-'a'+10)
		case digit >= 'A' && digit <= 'F':
			value = value*16 + rune(digit-'A'+10)
		default:
			return 0, false
		}
	}
	return value, true
}

// inlineImages finds base64 images that are JSON structure, not text: an
// Anthropic image source "data" member and a "data:image/...;base64," URL
// value, as [start, end) byte ranges in order. The same bytes quoted inside a
// string (a tool result) are escaped and stay text, and other media (a PDF)
// is charged as text, which over-counts it.
func inlineImages(body []byte) [][2]int {
	var blobs [][2]int
	for _, marker := range [][]byte{[]byte(`"data":"`), []byte(";base64,")} {
		for offset := 0; ; {
			index := bytes.Index(body[offset:], marker)
			if index < 0 {
				break
			}
			at := offset + index
			start := at + len(marker)
			run := base64Run(body[start:])
			offset = start + run
			if run < mediaBlobMinBytes {
				continue
			}
			if marker[0] == ';' {
				quote := bytes.LastIndexByte(body[:at], '"')
				if quote <= 0 || body[quote-1] == '\\' || !bytes.HasPrefix(body[quote:], []byte(`"data:image/`)) {
					continue
				}
			} else if !bytes.Contains(body[max(0, at-256):at], []byte(`"media_type":"image/`)) {
				continue
			}
			blobs = append(blobs, [2]int{start, start + run})
		}
	}
	// The two markers never overlap; order the ranges for one pass.
	for i := 1; i < len(blobs); i++ {
		for j := i; j > 0 && blobs[j][0] < blobs[j-1][0]; j-- {
			blobs[j], blobs[j-1] = blobs[j-1], blobs[j]
		}
	}
	return blobs
}

func base64Run(body []byte) int {
	for index, char := range body {
		if !(char >= 'A' && char <= 'Z' || char >= 'a' && char <= 'z' || char >= '0' && char <= '9' || char == '+' || char == '/' || char == '=' || char == '\\' && index+1 < len(body) && body[index+1] == '/') {
			return index
		}
	}
	return len(body)
}
