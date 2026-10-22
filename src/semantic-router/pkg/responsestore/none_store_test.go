package responsestore

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// The none backend is the store for a cell that keeps the Responses surface on
// and retains nothing. It must read as enabled, or the endpoint is 404; it
// must read as not retaining, or previous_response_id is answered as a lookup
// miss; and every read must answer as not found, so the object endpoints
// behave as they do for an id that was never stored.
func TestNoneStoreIsEnabledAndRetainsNothing(t *testing.T) {
	store, err := NewStore(StoreConfig{Enabled: true, BackendType: NoneStoreType})
	if err != nil {
		t.Fatalf("NewStore(none) error = %v", err)
	}
	if !store.IsEnabled() {
		t.Fatal("the none store must be enabled, or the Responses surface is taken down")
	}
	if store.Retains() {
		t.Fatal("the none store must not read as retaining")
	}
	if err := store.CheckConnection(t.Context()); err != nil {
		t.Fatalf("CheckConnection() = %v, want nil: there is nothing to connect to", err)
	}

	stored := &responseapi.StoredResponse{ID: "resp_gone", Object: "response"}
	if err := store.StoreResponse(t.Context(), stored); err != nil {
		t.Fatalf("StoreResponse() = %v, want a silent no-op", err)
	}
	if _, err := store.GetResponse(t.Context(), "resp_gone"); !errors.Is(err, ErrNoRetention) || !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetResponse() after store = %v, want ErrNoRetention that is also ErrNotFound", err)
	}
	if _, err := store.GetConversationChain(t.Context(), "resp_gone"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("GetConversationChain() = %v, want not found", err)
	}
	if err := store.DeleteResponse(t.Context(), "resp_gone"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("DeleteResponse() = %v, want not found", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("Close() = %v", err)
	}
}

// Retains is part of the ResponseStore contract, not an optional capability
// with a default. Every store the factory can build has to answer for itself,
// and the answer has to match what the backend actually does with a write.
// The old helper assumed a store without the capability retained, which let a
// new backend that forgot the method read as retaining; the compiler now
// refuses that backend instead.
func TestEveryFactoryBackendDeclaresWhetherItRetains(t *testing.T) {
	for name, test := range map[string]struct {
		config      StoreConfig
		wantEnabled bool
		wantRetains bool
	}{
		"memory": {
			config:      StoreConfig{Enabled: true, BackendType: MemoryStoreType},
			wantEnabled: true, wantRetains: true,
		},
		"backend unset defaults to memory": {
			config:      StoreConfig{Enabled: true},
			wantEnabled: true, wantRetains: true,
		},
		"none": {
			config:      StoreConfig{Enabled: true, BackendType: NoneStoreType},
			wantEnabled: true, wantRetains: false,
		},
		// enabled: false builds the disabled memory store. It retains in the
		// sense this flag means - a write would be readable - and is simply
		// never reached, because the surface is 404 at the header phase. It
		// must not read as a cell that retains nothing on purpose, or a
		// disabled surface would start answering 400 instead of 404.
		"disabled": {
			config:      StoreConfig{Enabled: false},
			wantEnabled: false, wantRetains: true,
		},
	} {
		t.Run(name, func(t *testing.T) {
			store, err := NewStore(test.config)
			if err != nil {
				t.Fatalf("NewStore() error = %v", err)
			}
			defer func() { _ = store.Close() }()
			if got := store.IsEnabled(); got != test.wantEnabled {
				t.Fatalf("IsEnabled() = %v, want %v", got, test.wantEnabled)
			}
			if got := store.Retains(); got != test.wantRetains {
				t.Fatalf("Retains() = %v, want %v", got, test.wantRetains)
			}
		})
	}
}

// A store that says it retains has to mean it: a write is readable back.
// A store that says it does not must not produce a readable write. This is
// what makes Retains worth branching on rather than a label.
func TestRetainsMatchesWhatTheStoreDoesWithAWrite(t *testing.T) {
	for name, backend := range map[string]StoreBackendType{
		"memory": MemoryStoreType,
		"none":   NoneStoreType,
	} {
		t.Run(name, func(t *testing.T) {
			store, err := NewStore(StoreConfig{Enabled: true, BackendType: backend})
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = store.Close() }()
			if storeErr := store.StoreResponse(t.Context(), &responseapi.StoredResponse{
				ID: "resp_probe", Object: "response",
			}); storeErr != nil {
				t.Fatalf("StoreResponse() = %v", storeErr)
			}
			_, readErr := store.GetResponse(t.Context(), "resp_probe")
			if store.Retains() && readErr != nil {
				t.Fatalf("Retains() is true but the write could not be read back: %v", readErr)
			}
			if !store.Retains() && !errors.Is(readErr, ErrNotFound) {
				t.Fatalf("Retains() is false but GetResponse() = %v, want not found", readErr)
			}
		})
	}
}

// The redis store retains. It is declared on the type rather than exercised
// against a live server here: the integration suite covers the round trip,
// and this pins the declaration the gate reads.
func TestRedisStoreDeclaresThatItRetains(t *testing.T) {
	var store ResponseStore = &RedisStore{}
	if !store.Retains() {
		t.Fatal("the redis store must read as retaining")
	}
}

// Disabled wins over the backend name: enabled: false still builds the
// disabled memory store, whatever store_backend says.
func TestNoneBackendWithStoreDisabledIsStillDisabled(t *testing.T) {
	store, err := NewStore(StoreConfig{Enabled: false, BackendType: NoneStoreType})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = store.Close() }()
	if store.IsEnabled() {
		t.Fatal("enabled: false must disable the store regardless of backend")
	}
}
