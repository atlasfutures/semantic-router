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
