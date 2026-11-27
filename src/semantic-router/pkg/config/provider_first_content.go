package config

// FirstContentSecForModel is how long a streamed turn on the arm may wait for
// its first content: the arm's own first_content_sec when it sets one, else
// the global response_stream.first_content_sec. Zero means no limit. It
// accepts the names the other model lookups accept.
func (c *RouterConfig) FirstContentSecForModel(modelName string) int {
	if c == nil {
		return 0
	}
	if params, found := c.resolveModelConfig(modelName); found && params.FirstContentSec > 0 {
		return params.FirstContentSec
	}
	return c.ResponseStreamFirstContentSec
}

// IdleSecForModel is how long a streamed turn on the arm may go between
// content once its first content has arrived: the arm's own idle_sec when it
// sets one, else the global response_stream.idle_sec. Zero means no limit. It
// accepts the names the other model lookups accept.
func (c *RouterConfig) IdleSecForModel(modelName string) int {
	if c == nil {
		return 0
	}
	if params, found := c.resolveModelConfig(modelName); found && params.IdleSec > 0 {
		return params.IdleSec
	}
	return c.ResponseStreamIdleSec
}

// HostRetryForModel reports whether a stalled streamed turn on the arm is
// re-sent on its next provider host: the arm's own host_retry when it sets
// one, else the global response_stream.host_retry.
func (c *RouterConfig) HostRetryForModel(modelName string) bool {
	if c == nil {
		return false
	}
	if params, found := c.resolveModelConfig(modelName); found && params.HostRetry != nil {
		return *params.HostRetry
	}
	return c.ResponseStreamHostRetry
}

// DefaultStreamHeadRetryTokens is the window an arm's head_retry holds back
// when it sets no head_retry_tokens of its own.
const DefaultStreamHeadRetryTokens = 8

// StreamHeadRetryTokensForModel is how many completion tokens of a streamed
// turn on the arm are held back so that a stream the upstream ends early can
// be re-sent once before any content reaches the client: the arm's own
// head_retry_tokens, else DefaultStreamHeadRetryTokens, when its head_retry is
// on. Zero means the arm holds nothing back, which is the default. It accepts
// the names the other model lookups accept.
func (c *RouterConfig) StreamHeadRetryTokensForModel(modelName string) int {
	if c == nil {
		return 0
	}
	params, found := c.resolveModelConfig(modelName)
	if !found || !params.HeadRetry {
		return 0
	}
	if params.HeadRetryTokens > 0 {
		return params.HeadRetryTokens
	}
	return DefaultStreamHeadRetryTokens
}
