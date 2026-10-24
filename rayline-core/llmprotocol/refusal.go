package llmprotocol

// ResponseRefused reports a model refusal: a content_filter stop, or refusal
// content whatever stop its source format gave it.
func ResponseRefused(response *Response) bool {
	if response == nil {
		return false
	}
	if response.StopReason == StopContentFilter {
		return true
	}
	for _, item := range response.Output {
		for _, content := range item.Content {
			if content.Kind == ContentRefusal {
				return true
			}
		}
	}
	return false
}
