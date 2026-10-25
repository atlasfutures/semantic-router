//go:build !windows && cgo

package extproc

import (
	"slices"
	"testing"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// The allowed set is logged as a count and a digest of its sorted members,
// so two spellings of one set log the same digest and the list never does.
func TestPrivacyAllowedArmsDigestIsStable(t *testing.T) {
	t.Parallel()
	first := raylineARCPrivacyAllowedArms("privacy", privacyHeaders("privacy", "arm-1, arm-0,arm-1"))
	second := raylineARCPrivacyAllowedArms("privacy", privacyHeaders("privacy", "arm-0,arm-1"))
	if !slices.Equal(first, []string{"arm-0", "arm-1"}) {
		t.Fatalf("allowed = %v, want sorted and deduplicated", first)
	}
	if privacyAllowedArmsHash(first) != privacyAllowedArmsHash(second) || len(privacyAllowedArmsHash(first)) != 16 {
		t.Fatalf("digests %q and %q differ or are not 16 hex", privacyAllowedArmsHash(first), privacyAllowedArmsHash(second))
	}
	if privacyAllowedArmsHash(nil) != "" {
		t.Fatal("a standard request has a digest")
	}
}

// The selection line names the applied mode, and a private one its allowed
// set's size and digest.
func TestSelectionLogCarriesThePrivacyMode(t *testing.T) {
	tests := map[string]struct {
		ctx  *RequestContext
		want map[string]interface{}
	}{
		"standard": {
			&RequestContext{RequestID: "r"},
			map[string]interface{}{"privacy_mode": "standard"},
		},
		"zdr": {
			&RequestContext{
				RequestID: "r", RaylineARCPrivacyMode: raylineARCPrivacyModeZDR,
				RaylineARCPrivacyAllowedArms: []string{"arm-0", "arm-1"},
			},
			map[string]interface{}{
				"privacy_mode": "zdr", "privacy_allowed_arms_count": int64(2),
				"privacy_allowed_arms_hash": privacyAllowedArmsHash([]string{"arm-0", "arm-1"}),
			},
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			logs := captureLogs(t)
			observeRaylineARCSelection(test.ctx, &selection.RaylineARCTrace{})
			fields := findLogEvent(t, logs, "rayline_arc_selection")
			for key, value := range test.want {
				if fields[key] != value {
					t.Errorf("%s = %#v, want %#v", key, fields[key], value)
				}
			}
			if _, present := fields["privacy_allowed_arms_count"]; present && test.ctx.RaylineARCPrivacyMode == "" {
				t.Error("a standard turn logged an allowed set")
			}
		})
	}
}

// The usage line carries the mode and the provider object's data-use members
// as the dispatched body carried them, read back off the body.
func TestUsageRecordCarriesThePrivacyWire(t *testing.T) {
	logs := captureLogs(t)
	ctx := &RequestContext{
		RequestID: "rt_private", RequestModel: "priced-model",
		VSRRaylineARC:                &selection.RaylineARCTrace{},
		RaylineARCPrivacyMode:        raylineARCPrivacyModePrivacy,
		RaylineARCPrivacyAllowedArms: []string{"arm-0"},
	}
	recordDispatchedProviderPin(ctx, []byte(`{"provider":{"order":["z-ai"],"allow_fallbacks":false,"data_collection":"deny","zdr":true}}`))

	usageRecordRouter().reportNonStreamingUsage(ctx, time.Second, streamingFlagUsage())

	fields := findLogEvent(t, logs, "llm_usage")
	want := map[string]interface{}{
		"privacy_mode": "privacy", "privacy_allowed_arms_count": int64(1),
		"privacy_allowed_arms_hash": privacyAllowedArmsHash([]string{"arm-0"}),
		"provider_data_collection":  "deny", "provider_zdr": true, "provider_allow_fallbacks": false,
	}
	for key, value := range want {
		if fields[key] != value {
			t.Errorf("%s = %#v, want %#v", key, fields[key], value)
		}
	}
}

// A body that sent no provider object, outside ARC, leaves every privacy
// member null rather than claiming a mode or a default.
func TestUsageRecordWithoutPrivacyLeavesItNull(t *testing.T) {
	logs := captureLogs(t)
	ctx := &RequestContext{RequestID: "rt_plain", RequestModel: "priced-model"}
	recordDispatchedProviderPin(ctx, []byte(`{"model":"m"}`))

	usageRecordRouter().reportNonStreamingUsage(ctx, time.Second, streamingFlagUsage())

	fields := findLogEvent(t, logs, "llm_usage")
	for _, key := range []string{
		"privacy_mode", "privacy_allowed_arms_count", "privacy_allowed_arms_hash",
		"provider_data_collection", "provider_zdr", "provider_allow_fallbacks",
	} {
		if value, present := fields[key]; !present || value != nil {
			t.Errorf("%s = %v (present %v), want null", key, value, present)
		}
	}
}

// x-vsr-privacy-mode states what the cell applied on an ARC turn, replacing
// any upstream copy; every other response has it stripped, and so does every
// request, so neither a client nor a provider can state it.
func TestPrivacyModeHeaderIsTheCellsAlone(t *testing.T) {
	t.Parallel()
	arc := buildResponseHeaderMutation(&RequestContext{
		VSRRaylineARC: &selection.RaylineARCTrace{}, RaylineARCPrivacyMode: raylineARCPrivacyModeZDR,
	}, true)
	var stated *core.HeaderValueOption
	for _, option := range arc.GetSetHeaders() {
		if option.GetHeader().GetKey() == headers.VSRPrivacyMode {
			stated = option
		}
	}
	if stated == nil || string(stated.GetHeader().GetRawValue()) != "zdr" {
		t.Fatalf("ARC response %s = %v, want zdr", headers.VSRPrivacyMode, stated)
	}
	if stated.GetAppendAction() != core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD {
		t.Fatal("an upstream copy of the header would survive beside the cell's")
	}
	// An error response states it too: the harness reads the mode either way.
	standard := buildResponseHeaderMutation(&RequestContext{VSRRaylineARC: &selection.RaylineARCTrace{}}, false)
	value := ""
	for _, option := range standard.GetSetHeaders() {
		if option.GetHeader().GetKey() == headers.VSRPrivacyMode {
			value = string(option.GetHeader().GetRawValue())
		}
	}
	if value != "standard" {
		t.Fatalf("standard ARC turn %s = %q, want standard", headers.VSRPrivacyMode, value)
	}
	plain := buildResponseHeaderMutation(&RequestContext{}, true)
	if !slices.Contains(plain.GetRemoveHeaders(), headers.VSRPrivacyMode) {
		t.Fatal("a non-ARC response passes an upstream's copy through")
	}
	if !slices.Contains(requestHeaderPhaseRemovals(), headers.VSRPrivacyMode) {
		t.Fatal("a client's copy would be forwarded upstream")
	}
}

// A response the router passes through untouched -- an unprocessed request,
// a Looper call -- still has an upstream's x-vsr-privacy-mode stripped.
func TestPrivacyModeHeaderIsStrippedOnPassThroughResponses(t *testing.T) {
	t.Parallel()
	router := &OpenAIRouter{}
	for name, ctx := range map[string]*RequestContext{
		"skip processing": {SkipProcessing: true},
		"looper":          {LooperRequest: true},
	} {
		response, err := router.handleResponseHeaders(&ext_proc.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{
				Headers: []*core.HeaderValue{{Key: ":status", Value: "200"}},
			}},
		}, ctx)
		if err != nil {
			t.Fatalf("%s: handleResponseHeaders: %v", name, err)
		}
		removed := response.GetResponseHeaders().GetResponse().GetHeaderMutation().GetRemoveHeaders()
		if !slices.Contains(removed, headers.VSRPrivacyMode) {
			t.Fatalf("%s: an upstream's %s passes through (removed %v)", name, headers.VSRPrivacyMode, removed)
		}
	}
}

// A standard native Responses body is sent as the codec rendered it, and the
// provider object it carries -- the caller's own -- is still what is recorded.
func TestStandardResponsesBodyRecordsTheCallersProviderObject(t *testing.T) {
	router := providerPinRouter("https://openrouter.ai/api/v1", nil)
	dispatch := &providerDispatch{
		logicalModel: providerPinArm, decisionName: providerPinDecision,
		targetFormat: llmprotocol.OpenAIResponsesV1, profile: openRouterProviderProfile(),
	}
	ctx := &RequestContext{}
	body := []byte(`{"model":"m","input":"hi","provider":{"data_collection":"deny","zdr":true}}`)
	if _, err := router.adaptProviderRequest(body, dispatch, ctx); err != nil {
		t.Fatalf("adaptProviderRequest: %v", err)
	}
	recorded := ctx.DispatchedProviderPrivacy
	if recorded.dataCollection != "deny" || recorded.zdr == nil || !*recorded.zdr {
		t.Fatalf("recorded = %+v, want the caller's data_collection and zdr", recorded)
	}
}
