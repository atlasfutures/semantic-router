package raylinearc

import (
	"context"
	"errors"
	"strings"
	"time"
)

// memorySessionFamilies is a session's record in a memory store.
type memorySessionFamilies struct {
	families []string
	until    time.Time
}

// sessionFamilyTable is the memory store's records; the caller holds
// familyMu.
func (store *MemoryEpisodeStore) sessionFamilyTable() map[string]memorySessionFamilies {
	if store.families == nil {
		store.families = map[string]memorySessionFamilies{}
	}
	return store.families
}

// LoadSessionFamilyExclusions reads the session's families.
func (store *MemoryEpisodeStore) LoadSessionFamilyExclusions(_ context.Context, sessionKey string) ([]string, error) {
	if store == nil || strings.TrimSpace(sessionKey) == "" {
		return nil, errors.New("invalid ARC memory session family read")
	}
	store.familyMu.Lock()
	defer store.familyMu.Unlock()
	sessions := store.sessionFamilyTable()
	key := HashEpisodeID(sessionKey)
	record, ok := sessions[key]
	if !ok {
		return nil, nil
	}
	if !store.now().Before(record.until) {
		delete(sessions, key)
		return nil, nil
	}
	return append([]string(nil), record.families...), nil
}

// AddSessionFamilyExclusion records family for the session.
func (store *MemoryEpisodeStore) AddSessionFamilyExclusion(
	_ context.Context, sessionKey string, family string, ttl time.Duration,
) error {
	if store == nil || !validSessionFamilyRequest(sessionKey, family) || ttl <= 0 {
		return errors.New("invalid ARC memory session family write")
	}
	store.familyMu.Lock()
	defer store.familyMu.Unlock()
	sessions := store.sessionFamilyTable()
	now := store.now()
	for key, record := range sessions {
		if !now.Before(record.until) {
			delete(sessions, key)
		}
	}
	key := HashEpisodeID(sessionKey)
	record := sessions[key]
	present := false
	for _, existing := range record.families {
		present = present || existing == family
	}
	if !present {
		if len(record.families) >= MaxSessionFamilyExclusions {
			return nil
		}
		record.families = append(record.families, family)
	}
	record.until = now.Add(ttl)
	sessions[key] = record
	return nil
}
