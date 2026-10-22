package responsestore

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// NoneStore is the store for a cell that serves the Responses surface and
// retains nothing: zero data retention with the endpoint still on. It is
// enabled, so POST /v1/responses is admitted and a response id is minted for
// every turn, but every write is a no-op and every read answers
// ErrNoRetention, so GET and DELETE /v1/responses/{id} answer 404 exactly as
// they do for an id the store never held, and previous_response_id can never
// resolve. The difference from a disabled store is deliberate: a disabled
// store takes the whole endpoint down (404 at the header phase), while this
// one keeps stateless generation available.
type NoneStore struct{}

// NewNoneStore creates the retaining-nothing store.
func NewNoneStore() *NoneStore { return &NoneStore{} }

// IsEnabled is true: the Responses surface is on, retention is not.
func (*NoneStore) IsEnabled() bool { return true }

// Retains is false: nothing given to this store can be read back, so a
// previous_response_id naming any id is refused 400 stateless_path.
func (*NoneStore) Retains() bool { return false }

// CheckConnection always succeeds: there is nothing to connect to.
func (*NoneStore) CheckConnection(context.Context) error { return nil }

// Close releases nothing.
func (*NoneStore) Close() error { return nil }

func (*NoneStore) StoreResponse(context.Context, *responseapi.StoredResponse) error { return nil }

func (*NoneStore) GetResponse(context.Context, string) (*responseapi.StoredResponse, error) {
	return nil, ErrNoRetention
}

func (*NoneStore) UpdateResponse(context.Context, *responseapi.StoredResponse) error {
	return ErrNoRetention
}

func (*NoneStore) DeleteResponse(context.Context, string) error { return ErrNoRetention }

func (*NoneStore) GetConversationChain(context.Context, string) ([]*responseapi.StoredResponse, error) {
	return nil, ErrNoRetention
}

func (*NoneStore) ListResponsesByConversation(context.Context, string, ListOptions) ([]*responseapi.StoredResponse, error) {
	return nil, nil
}

func (*NoneStore) CreateConversation(context.Context, *responseapi.StoredConversation) error {
	return nil
}

func (*NoneStore) GetConversation(context.Context, string) (*responseapi.StoredConversation, error) {
	return nil, ErrNoRetention
}

func (*NoneStore) UpdateConversation(context.Context, *responseapi.StoredConversation) error {
	return ErrNoRetention
}

func (*NoneStore) DeleteConversation(context.Context, string, bool) error { return ErrNoRetention }

func (*NoneStore) ListConversations(context.Context, ListOptions) ([]*responseapi.StoredConversation, error) {
	return nil, nil
}

func (*NoneStore) AddResponseToConversation(context.Context, string, string) error {
	return ErrNoRetention
}
