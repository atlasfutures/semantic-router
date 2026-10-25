package cachecontext

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
)

// MessageChain returns the chained digest after each message: entry i
// covers the first i messages, entry 0 is the empty prefix. Digests are the
// first 16 bytes of sha256, in hex.
func MessageChain(body []byte) ([]string, bool) {
	var wire struct {
		Messages []json.RawMessage `json:"messages"`
	}
	if json.Unmarshal(body, &wire) != nil {
		return nil, false
	}
	chain := make([]string, 0, len(wire.Messages)+1)
	running := sha256.Sum256(nil)
	chain = append(chain, hex.EncodeToString(running[:16]))
	for _, raw := range wire.Messages {
		var message interface{}
		if json.Unmarshal(raw, &message) != nil {
			return nil, false
		}
		projected, _ := json.Marshal(projectUpstreamMessage(message))
		digest := sha256.Sum256(projected)
		running = sha256.Sum256(append(running[:], digest[:]...))
		chain = append(chain, hex.EncodeToString(running[:16]))
	}
	return chain, true
}

func projectUpstreamMessage(value interface{}) interface{} {
	switch typed := value.(type) {
	case map[string]interface{}:
		projected := make(map[string]interface{}, len(typed))
		for key, member := range typed {
			if key != "cache_control" {
				projected[key] = projectUpstreamMessage(member)
			}
		}
		if parts, ok := projected["content"].([]interface{}); ok && len(parts) == 1 {
			if part, ok := parts[0].(map[string]interface{}); ok && part["type"] == "text" && len(part) == 2 {
				if text, ok := part["text"].(string); ok {
					projected["content"] = text
				}
			}
		}
		return projected
	case []interface{}:
		projected := make([]interface{}, len(typed))
		for index, member := range typed {
			projected[index] = projectUpstreamMessage(member)
		}
		return projected
	default:
		return value
	}
}
