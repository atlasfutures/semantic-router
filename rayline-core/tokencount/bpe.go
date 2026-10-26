// Package tokencount counts prompt tokens in-process with a byte-level BPE
// tokenizer reconstructed from a Hugging Face merges list. It reproduces the
// token count of the reference tokenizer (normalizer, added tokens,
// pre-tokenizer and merges); token ids are not needed and not kept.
package tokencount

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"container/heap"
	"fmt"
	"strings"
	"sync"

	"golang.org/x/text/unicode/norm"
)

// BPE counts tokens for one byte-level BPE vocabulary.
type BPE struct {
	ranks map[uint64]merge
	added []string
	split func(string, func(string))

	mu    sync.Mutex
	cache map[string]int32
}

type merge struct{ rank, id uint32 }

const pieceCacheLimit = 1 << 18

func pairKey(left, right uint32) uint64 { return uint64(left)<<32 | uint64(right) }

// newBPE builds the merge table from a gzip-compressed merges.txt whose
// tokens use the GPT-2 byte-to-unicode alphabet.
func newBPE(compressed []byte, added []string, split func(string, func(string))) (*BPE, error) {
	reader, err := gzip.NewReader(bytes.NewReader(compressed))
	if err != nil {
		return nil, err
	}
	decode := byteDecoder()
	ids := make(map[string]uint32, 1<<18)
	for value := 0; value < 256; value++ {
		ids[string([]byte{byte(value)})] = uint32(value)
	}
	ranks := make(map[uint64]merge, 1<<18)
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 1<<16), 1<<20)
	rank := uint32(0)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" || strings.HasPrefix(line, "#version") {
			continue
		}
		left, right, ok := strings.Cut(line, " ")
		if !ok {
			return nil, fmt.Errorf("merge %d is malformed", rank)
		}
		leftBytes, err := decodeToken(left, decode)
		if err != nil {
			return nil, err
		}
		rightBytes, err := decodeToken(right, decode)
		if err != nil {
			return nil, err
		}
		// The reference takes merge parts from its vocabulary, not from
		// earlier merges, so a part may first appear here. Only identity
		// matters for counting: one id per byte string.
		id := func(token string) uint32 {
			value, exists := ids[token]
			if !exists {
				value = uint32(len(ids))
				ids[token] = value
			}
			return value
		}
		leftID, rightID, mergedID := id(leftBytes), id(rightBytes), id(leftBytes+rightBytes)
		if _, duplicate := ranks[pairKey(leftID, rightID)]; !duplicate {
			ranks[pairKey(leftID, rightID)] = merge{rank: rank, id: mergedID}
		}
		rank++
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return &BPE{ranks: ranks, added: added, split: split, cache: make(map[string]int32)}, nil
}

// byteDecoder inverts GPT-2's bytes_to_unicode table.
func byteDecoder() map[rune]byte {
	decode := make(map[rune]byte, 256)
	next := rune(256)
	for value := 0; value < 256; value++ {
		if value >= '!' && value <= '~' || value >= 0xA1 && value <= 0xAC || value >= 0xAE && value <= 0xFF {
			decode[rune(value)] = byte(value)
			continue
		}
		decode[next] = byte(value)
		next++
	}
	return decode
}

func decodeToken(token string, decode map[rune]byte) (string, error) {
	out := make([]byte, 0, len(token))
	for _, char := range token {
		value, ok := decode[char]
		if !ok {
			return "", fmt.Errorf("merge token %q is outside the byte alphabet", token)
		}
		out = append(out, value)
	}
	return string(out), nil
}

// Count returns the number of tokens the reference tokenizer produces for
// text with no special tokens added: added tokens split first, then NFC, the
// pre-tokenizer and byte-level BPE on each piece.
func (bpe *BPE) Count(text string) int {
	total := 0
	for len(text) > 0 {
		at, length := bpe.nextAdded(text)
		segment := text
		if at >= 0 {
			segment = text[:at]
		}
		if segment != "" {
			if !norm.NFC.IsNormalString(segment) {
				segment = norm.NFC.String(segment)
			}
			bpe.split(segment, func(piece string) { total += bpe.countPiece(piece) })
		}
		if at < 0 {
			break
		}
		total++
		text = text[at+length:]
	}
	return total
}

// nextAdded finds the leftmost (then longest) added token in text.
func (bpe *BPE) nextAdded(text string) (int, int) {
	for offset := 0; ; {
		index := strings.IndexByte(text[offset:], '<')
		if index < 0 {
			return -1, 0
		}
		at := offset + index
		best := 0
		for _, token := range bpe.added {
			if len(token) > best && strings.HasPrefix(text[at:], token) {
				best = len(token)
			}
		}
		if best > 0 {
			return at, best
		}
		offset = at + 1
	}
}

func (bpe *BPE) countPiece(piece string) int {
	if len(piece) <= 1 {
		return len(piece)
	}
	cacheable := len(piece) <= 64
	if cacheable {
		bpe.mu.Lock()
		count, ok := bpe.cache[piece]
		bpe.mu.Unlock()
		if ok {
			return int(count)
		}
	}
	count := bpe.mergeCount(piece)
	if cacheable {
		bpe.mu.Lock()
		if len(bpe.cache) >= pieceCacheLimit {
			bpe.cache = make(map[string]int32)
		}
		bpe.cache[piece] = int32(count)
		bpe.mu.Unlock()
	}
	return count
}

type symbol struct {
	id         uint32
	prev, next int
	alive      bool
}

type candidate struct {
	rank, left, right uint32
	pos               int
}

type candidates []candidate

func (queue candidates) Len() int { return len(queue) }
func (queue candidates) Less(i, j int) bool {
	return queue[i].rank < queue[j].rank || queue[i].rank == queue[j].rank && queue[i].pos < queue[j].pos
}
func (queue candidates) Swap(i, j int)   { queue[i], queue[j] = queue[j], queue[i] }
func (queue *candidates) Push(value any) { *queue = append(*queue, value.(candidate)) }
func (queue *candidates) Pop() any {
	old := *queue
	last := old[len(old)-1]
	*queue = old[:len(old)-1]
	return last
}

// mergeCount applies merges lowest rank first, leftmost first among equal
// ranks (the reference Word::merge_all order), and returns the symbol count.
func (bpe *BPE) mergeCount(piece string) int {
	symbols := make([]symbol, len(piece))
	for index := range symbols {
		symbols[index] = symbol{id: uint32(piece[index]), prev: index - 1, next: index + 1, alive: true}
	}
	symbols[len(symbols)-1].next = -1
	queue := candidates{}
	push := func(pos int) {
		if pos < 0 || symbols[pos].next < 0 {
			return
		}
		left, right := symbols[pos].id, symbols[symbols[pos].next].id
		if found, ok := bpe.ranks[pairKey(left, right)]; ok {
			heap.Push(&queue, candidate{rank: found.rank, left: left, right: right, pos: pos})
		}
	}
	for index := range symbols {
		push(index)
	}
	count := len(symbols)
	for queue.Len() > 0 {
		top := heap.Pop(&queue).(candidate)
		current := &symbols[top.pos]
		if !current.alive || current.next < 0 || current.id != top.left || symbols[current.next].id != top.right {
			continue
		}
		next := current.next
		current.id = bpe.ranks[pairKey(top.left, top.right)].id
		symbols[next].alive = false
		current.next = symbols[next].next
		if current.next >= 0 {
			symbols[current.next].prev = top.pos
		}
		count--
		push(current.prev)
		push(top.pos)
	}
	return count
}
