package cachecontext

import "testing"

func TestMessageChainRecognizesCacheBreakpointMovesButRejectsRewrites(t *testing.T) {
	first, ok := MessageChain([]byte(`{"messages":[{"role":"user","content":[{"type":"text","text":"hello","cache_control":{"type":"ephemeral"}}]}]}`))
	if !ok {
		t.Fatal("initial message was not projected")
	}
	extended, ok := MessageChain([]byte(`{"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"answer"}]}`))
	if !ok || len(extended) != 3 || extended[1] != first[1] {
		t.Fatal("moving a cache breakpoint or flattening a text block broke the semantic prefix")
	}
	rewritten, ok := MessageChain([]byte(`{"messages":[{"role":"user","content":"summary"},{"role":"assistant","content":"answer"}]}`))
	if !ok || rewritten[1] == first[1] {
		t.Fatal("changed context retained the previous prefix")
	}
}
