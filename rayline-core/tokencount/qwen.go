package tokencount

import (
	_ "embed"
	"sync"
	"unicode"
	"unicode/utf8"
)

// qwen38Merges is merges.txt of the Qwen3.8-27B tokenizer (Hugging Face
// snapshot 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0; tokenizer.json sha256
// 0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3),
// gzip-compressed. Its uncompressed sha256 is Qwen38MergesSHA256.
//
//go:embed qwen38_merges.txt.gz
var qwen38Merges []byte

// Licence: the Qwen model family publishes its tokenizers under Apache-2.0.
// The cached snapshot this file was taken from carries no LICENSE file, so
// the licence of this exact snapshot must be confirmed before a release that
// ships it.
//
// Qwen38MergesSHA256 identifies the embedded merges.txt.
const Qwen38MergesSHA256 = "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"

// qwen38Added are the tokenizer's added tokens, matched on the raw text
// before normalization; each counts as one token.
var qwen38Added = []string{
	"<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
	"<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
	"<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>", "<|fim_prefix|>", "<|fim_middle|>",
	"<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>", "<tool_response>", "</tool_response>",
	"<think>", "</think>", "<|audio_start|>", "<|audio_end|>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>",
	"<tts_text_bos_single>", "<|audio_pad|>",
}

var (
	qwen38Once  sync.Once
	qwen38      *BPE
	qwen38Error error
)

// Qwen38 returns the Qwen3.8-27B counter, built on first use (about 250k
// merges).
func Qwen38() (*BPE, error) {
	qwen38Once.Do(func() { qwen38, qwen38Error = newBPE(qwen38Merges, qwen38Added, qwenSplit) })
	return qwen38, qwen38Error
}

// qwenSplit is the Qwen pre-tokenizer regex, evaluated with ordered
// alternation as the reference (Oniguruma) engine does:
//
//	(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|
//	 ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
func qwenSplit(text string, emit func(string)) {
	runes := []rune(text)
	offsets := make([]int, len(runes)+1)
	position := 0
	for index, char := range runes {
		offsets[index] = position
		position += utf8.RuneLen(char)
	}
	offsets[len(runes)] = position
	for start := 0; start < len(runes); {
		end := qwenMatch(runes, start)
		emit(text[offsets[start]:offsets[end]])
		start = end
	}
}

func isLetterOrMark(char rune) bool { return unicode.IsLetter(char) || unicode.IsMark(char) }
func isNewline(char rune) bool      { return char == '\r' || char == '\n' }

// isSymbol is [^\s\p{L}\p{M}\p{N}].
func isSymbol(char rune) bool {
	return !unicode.IsSpace(char) && !isLetterOrMark(char) && !unicode.IsNumber(char)
}

func qwenMatch(runes []rune, start int) int {
	count := len(runes)
	// (?i:'s|'t|'re|'ve|'m|'ll|'d)
	if runes[start] == '\'' {
		for _, suffix := range []string{"s", "t", "re", "ve", "m", "ll", "d"} {
			end := start + 1 + len(suffix)
			if end > count {
				continue
			}
			matched := true
			for index, want := range suffix {
				if unicode.ToLower(runes[start+1+index]) != want {
					matched = false
					break
				}
			}
			if matched {
				return end
			}
		}
	}
	// [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
	first := runes[start]
	if !isNewline(first) && !unicode.IsLetter(first) && !unicode.IsNumber(first) && start+1 < count && isLetterOrMark(runes[start+1]) {
		end := start + 1
		for end < count && isLetterOrMark(runes[end]) {
			end++
		}
		return end
	}
	if isLetterOrMark(first) {
		end := start
		for end < count && isLetterOrMark(runes[end]) {
			end++
		}
		return end
	}
	// \p{N}
	if unicode.IsNumber(first) {
		return start + 1
	}
	//  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
	symbols := start
	if first == ' ' && start+1 < count && isSymbol(runes[start+1]) {
		symbols = start + 1
	}
	if isSymbol(runes[symbols]) {
		end := symbols
		for end < count && isSymbol(runes[end]) {
			end++
		}
		for end < count && isNewline(runes[end]) {
			end++
		}
		return end
	}
	// The whitespace alternatives.
	end := start
	lastNewline := -1
	for end < count && unicode.IsSpace(runes[end]) {
		if isNewline(runes[end]) {
			lastNewline = end
		}
		end++
	}
	if end == start {
		return start + 1
	}
	// \s*[\r\n]+
	if lastNewline >= 0 {
		return lastNewline + 1
	}
	// \s+(?!\S), then \s+
	if end < count && end-start >= 2 {
		return end - 1
	}
	return end
}
