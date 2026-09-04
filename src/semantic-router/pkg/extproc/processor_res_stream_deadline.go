package extproc

// The platform's timeout ladder above one routed turn, and the Router's own
// deadline at the bottom of it.
//
// A turn that reaches the top of the ladder is ended by the platform, and the
// Router never gets to speak: by the time the cut arrives the ext_proc stream
// is gone and the frames that would close the message honestly have nowhere to
// go. That is what three thinking-arm turns on the dev cell 2026-09-03 looked
// like from the client -- message_start, content_block_start, deltas, then a
// clean close, with curl exiting 0.
//
// So the Router ends the turn first, while the connection to the client is
// still open. 590 s leaves 20 s below the lowest rung for the closing frames
// to travel.
//
//	Cloud Run request timeout    630 s
//	Envoy stream_idle_timeout    620 s
//	ext_proc message_timeout     610 s
//	this deadline                590 s
//
// The cell's Envoy configuration is authoritative for the three rungs above;
// extProcMessageTimeoutSeconds mirrors it so the relationship can be asserted
// here rather than only remembered.
const (
	defaultResponseStreamDeadlineSeconds = 590
	extProcMessageTimeoutSeconds         = 610
)
