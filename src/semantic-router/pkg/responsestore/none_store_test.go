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
	if Retains(store) {
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

// Retains is a capability, not a requirement: the memory and redis stores do
// not declare it and are read as retaining, and a nil store retains nothing.
func TestRetainsDefaultsToTrueForStoresWithoutTheCapability(t *testing.T) {
	memory, err := NewMemoryStore(StoreConfig{Enabled: true})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = memory.Close() }()
	if !Retains(memory) {
		t.Fatal("a store without the capability must be read as retaining")
	}
	if Retains(nil) {
		t.Fatal("a nil store retains nothing")
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
