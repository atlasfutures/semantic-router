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
