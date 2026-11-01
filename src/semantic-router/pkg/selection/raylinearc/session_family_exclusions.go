package raylinearc

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"
)

// Session family exclusions (pathfinder#6066 D3) keep a refusing model's
// vendor family out of the offer for the rest of a session, across the
// contexts its turns open. A context exclusion (PolicyEpisodeState.Exclusions)
// ends where its context ends: a fork, a retry that edits an earlier message
// or a compaction starts with the full offer. Claude Code's retry after a
// refusal is such an edit, so on 2026-10-29 a refused turn's retry went to the
// refusing model again and was refused again. A family exclusion is keyed by
// the session (the caller's session key: user and conversation), not by the
// episode, so it survives all of them.
//
// The record names families only, never content: a bounded set of short
// labels under the session key's digest, which expires on its own.

// MaxSessionFamilyExclusions bounds the families one session excludes.
const MaxSessionFamilyExclusions = 8

// maxSessionFamilyBytes bounds one family label.
const maxSessionFamilyBytes = 32

// SessionFamilyExclusionStore keeps the families a session excludes.
type SessionFamilyExclusionStore interface {
	// LoadSessionFamilyExclusions answers the families the session excludes
	// now, or none.
	LoadSessionFamilyExclusions(ctx context.Context, sessionKey string) ([]string, error)
	// AddSessionFamilyExclusion excludes family for the session for ttl from
	// now; a later exclusion of any family extends the whole record.
	AddSessionFamilyExclusion(ctx context.Context, sessionKey string, family string, ttl time.Duration) error
}

// familyPrefixes are the vendor families a refusal excludes as a whole.
// Refusals correlate within these families (pathfinder#6066 §4: the same
// tasks refused on Opus, Sonnet and Haiku alike). Another model's refusal
// excludes that model only, through its context exclusion.
var familyPrefixes = []string{"claude", "gpt"}

// ModelFamily is the vendor family of model ("claude" for claude-opus-5.5 or
// anthropic/claude-sonnet-5.5), or "" when it belongs to none the fallback
// excludes as a family.
func ModelFamily(model string) string {
	model = strings.ToLower(strings.TrimSpace(model))
	if slash := strings.LastIndexByte(model, '/'); slash >= 0 {
		model = model[slash+1:]
	}
	for _, family := range familyPrefixes {
		if model == family || strings.HasPrefix(model, family+"-") {
			return family
		}
	}
	return ""
}

// ExcludesFamily reports whether families holds model's family.
func ExcludesFamily(families []string, model string) bool {
	family := ModelFamily(model)
	if family == "" {
		return false
	}
	for _, excluded := range families {
		if excluded == family {
			return true
		}
	}
	return false
}

func validSessionFamilyRequest(sessionKey, family string) bool {
	return strings.TrimSpace(sessionKey) != "" &&
		family != "" && len(family) <= maxSessionFamilyBytes && !strings.ContainsAny(family, " \t\r\n")
}

// sessionFamilyKey is the record's key: the session key's digest, so a raw
// session key never becomes a Redis key.
func (store *RedisEpisodeStore) sessionFamilyKey(sessionKey string) string {
	return store.keyPrefix + "famx:" + HashEpisodeID(sessionKey)
}

var redisSessionFamilyAddScript = redis.NewScript(`
if redis.call("SISMEMBER", KEYS[1], ARGV[1]) == 0 and redis.call("SCARD", KEYS[1]) >= tonumber(ARGV[3]) then
  return 0
end
redis.call("SADD", KEYS[1], ARGV[1])
redis.call("PEXPIRE", KEYS[1], ARGV[2])
return 1
`)

// LoadSessionFamilyExclusions reads the session's families.
func (store *RedisEpisodeStore) LoadSessionFamilyExclusions(ctx context.Context, sessionKey string) ([]string, error) {
	if store == nil || strings.TrimSpace(sessionKey) == "" {
		return nil, errors.New("invalid ARC Redis session family read")
	}
	families, err := store.client.SMembers(ctx, store.sessionFamilyKey(sessionKey)).Result()
	if errors.Is(err, redis.Nil) {
		return nil, nil
	}
	if err != nil {
		return nil, boundedRedisEpisodeError("session family read", err)
	}
	return boundedFamilies(families), nil
}

// AddSessionFamilyExclusion records family for the session.
func (store *RedisEpisodeStore) AddSessionFamilyExclusion(
	ctx context.Context, sessionKey string, family string, ttl time.Duration,
) error {
	if store == nil || !validSessionFamilyRequest(sessionKey, family) || ttl <= 0 {
		return errors.New("invalid ARC Redis session family write")
	}
	err := redisSessionFamilyAddScript.Run(ctx, store.client, []string{store.sessionFamilyKey(sessionKey)},
		family, ttl.Milliseconds(), MaxSessionFamilyExclusions).Err()
	if err != nil {
		return boundedRedisEpisodeError("session family write", err)
	}
	return nil
}

// boundedFamilies keeps at most MaxSessionFamilyExclusions well-formed
// labels, whoever wrote the record.
func boundedFamilies(families []string) []string {
	out := make([]string, 0, len(families))
	for _, family := range families {
		if len(out) == MaxSessionFamilyExclusions {
			break
		}
		if family != "" && len(family) <= maxSessionFamilyBytes {
			out = append(out, family)
		}
	}
	return out
}
