package raylinearc

import (
	"bytes"
	"encoding/json"
	"fmt"
	"regexp"
	"slices"
)

// A v6 package (pathfinder docs/arc_serving_contract.md, "Policy package
// v6"; ADR 0122, pathfinder#3495) is a v5 package whose encoder reads images
// natively: its encoding_profile is the canonical_v2 image profile
// (pathfinder ImageEncodingProfile) and every other member keeps its v5
// meaning and checks. A text profile stays v5, so a v6 manifest with one is
// refused, as a v4 or v5 manifest with an image member is.
//
// VSR checks these values only to refuse a profile it does not know; the
// policy service encodes by them, and profile_id stays opaque.

// The values a v6 image profile states.
const (
	PolicySerializerV2            = "arc-role-blocks-v2"
	PolicyConversationCanonicalV2 = "canonical_v2"
	policyProjectionV1            = "pathfinder.input-projection.v1"
	policyReadoutFinalLayer       = "final_layer_block_last_token"
	policyPositionsQwen3VLMRoPE   = "qwen3_vl_mrope"
	policyImageRuleV1             = "arc-image-v1"
	policyImageResampleBicubic    = "pillow_bicubic"
	policyImageDecodeFirstFrame   = "first_frame_exif_transpose_rgb_on_white"
	policyVisionDTypeFloat32      = "float32"
	policyVisionEmbedCastBF16     = "bfloat16"
)

var policyEncoderRevision = regexp.MustCompile(`^[a-f0-9]{40}$`)

// policyImageProcessor is encoding_profile.image_processor, decoded strictly.
// Every integer is required (absent decodes as 0, which is refused).
type policyImageProcessor struct {
	Rule           string `json:"rule"`
	Factor         int    `json:"factor"`
	PatchSize      int    `json:"patch_size"`
	MergeSize      int    `json:"merge_size"`
	MinPixels      int    `json:"min_pixels"`
	MaxPixels      int    `json:"max_pixels"`
	MaxLongEdge    int    `json:"max_long_edge"`
	MaxAspectRatio int    `json:"max_aspect_ratio"`
	Resample       string `json:"resample"`
	Decode         string `json:"decode"`
}

// policyVision is encoding_profile.vision, decoded strictly. PerImageCalls is
// raw so that only the JSON literal true passes.
type policyVision struct {
	DType         string          `json:"dtype"`
	EmbedCast     string          `json:"embed_cast"`
	PerImageCalls json.RawMessage `json:"per_image_calls"`
}

// DecodePolicyPackageManifestV6 decodes a v6 manifest into v5's shape. It
// refuses unknown fields at every level, another schema, live prices, a
// profile that is not the canonical_v2 image profile, and any action v5
// would refuse. Bindings, readiness and dispatch are then exactly v5's.
func DecodePolicyPackageManifestV6(body []byte) (*PolicyPackageManifestV5, error) {
	var manifest PolicyPackageManifestV5
	if err := decodeStrict(body, &manifest); err != nil {
		return nil, err
	}
	if manifest.SchemaVersion != PolicyPackageSchemaV6 {
		return nil, fmt.Errorf("policy package schema %q", manifest.SchemaVersion)
	}
	if manifest.Pricing.LivePricesAffectDecisions {
		return nil, fmt.Errorf("policy package lets live prices affect decisions")
	}
	if err := manifest.checkImageProfile(); err != nil {
		return nil, err
	}
	if err := manifest.checkControlActions(); err != nil {
		return nil, err
	}
	return &manifest, nil
}

// DecodePolicyPackageManifestV5OrLater decodes a v5 or v6 manifest, the
// schemas whose actions are thinking controls, and refuses any other.
func DecodePolicyPackageManifestV5OrLater(body []byte) (*PolicyPackageManifestV5, error) {
	schema, err := PolicyPackageSchemaOf(body)
	if err != nil {
		return nil, err
	}
	switch schema {
	case PolicyPackageSchemaV5:
		return DecodePolicyPackageManifestV5(body)
	case PolicyPackageSchemaV6:
		return DecodePolicyPackageManifestV6(body)
	default:
		return nil, fmt.Errorf("policy package schema %q is not %s or %s", schema, PolicyPackageSchemaV5, PolicyPackageSchemaV6)
	}
}

// checkImageProfile checks a v6 profile against the contract's table: the
// text members as a v5 profile states them, conversation required as
// canonical_v2, and the image members required and strict.
func (common *PolicyPackageCommon) checkImageProfile() error {
	profile := common.EncodingProfile
	if profile.Serializer != PolicySerializerV2 {
		return fmt.Errorf("policy package encoding_profile.serializer %q is not the image profile's %s", profile.Serializer, PolicySerializerV2)
	}
	if profile.Conversation == nil {
		return fmt.Errorf("policy package encoding_profile.conversation is required as %s", PolicyConversationCanonicalV2)
	}
	if err := common.checkProfileMembers(PolicyConversationCanonicalV2); err != nil {
		return err
	}
	if err := checkPolicyEncoder(profile.Projection, profile.ToolDefinitions, profile.EncoderModel,
		profile.EncoderRevision, profile.DType, profile.Readout, profile.MaxTokens, profile.Dimension); err != nil {
		return err
	}
	if !jsonEquals(profile.Modalities, `["text","image"]`) {
		return fmt.Errorf("policy package encoding_profile.modalities %s is not [\"text\",\"image\"]", orAbsent(profile.Modalities))
	}
	if !jsonEquals(profile.Positions, `"`+policyPositionsQwen3VLMRoPE+`"`) {
		return fmt.Errorf("policy package encoding_profile.positions %s is not %s", orAbsent(profile.Positions), policyPositionsQwen3VLMRoPE)
	}
	if err := checkPolicyImageProcessor(profile.ImageProcessor); err != nil {
		return err
	}
	return checkPolicyVision(profile.Vision)
}

func checkPolicyEncoder(projection, tools, model, revision, dtype, readout string, maxTokens, dimension int) error {
	switch {
	case projection != policyProjectionV1:
		return fmt.Errorf("policy package encoding_profile.projection %q is not %s", projection, policyProjectionV1)
	case tools != "include_recorded" && tools != "exclude":
		return fmt.Errorf("policy package encoding_profile.tool_definitions %q is not one the contract defines", tools)
	case model == "":
		return fmt.Errorf("policy package encoding_profile.encoder_model is required")
	case !policyEncoderRevision.MatchString(revision):
		return fmt.Errorf("policy package encoding_profile.encoder_revision %q is not a 40-hex commit", revision)
	case dtype != "bfloat16" && dtype != "float32":
		return fmt.Errorf("policy package encoding_profile.dtype %q is not one the contract defines", dtype)
	case readout != policyReadoutFinalLayer:
		return fmt.Errorf("policy package encoding_profile.readout %q is not %s", readout, policyReadoutFinalLayer)
	case maxTokens < 1 || dimension < 1:
		return fmt.Errorf("policy package encoding_profile.max_tokens and dimension must be at least 1")
	}
	return nil
}

func checkPolicyImageProcessor(raw json.RawMessage) error {
	if raw == nil {
		return fmt.Errorf("policy package encoding_profile.image_processor is required")
	}
	var processor policyImageProcessor
	if err := decodeStrict(raw, &processor); err != nil {
		return fmt.Errorf("policy package encoding_profile.image_processor: %w", err)
	}
	switch {
	case processor.Rule != policyImageRuleV1:
		return fmt.Errorf("policy package encoding_profile.image_processor.rule %q is not %s", processor.Rule, policyImageRuleV1)
	case processor.Resample != policyImageResampleBicubic:
		return fmt.Errorf("policy package encoding_profile.image_processor.resample %q is not %s", processor.Resample, policyImageResampleBicubic)
	case processor.Decode != policyImageDecodeFirstFrame:
		return fmt.Errorf("policy package encoding_profile.image_processor.decode %q is not %s", processor.Decode, policyImageDecodeFirstFrame)
	case slices.ContainsFunc([]int{
		processor.Factor, processor.PatchSize, processor.MergeSize, processor.MinPixels,
		processor.MaxPixels, processor.MaxLongEdge, processor.MaxAspectRatio,
	}, func(value int) bool { return value < 1 }):
		return fmt.Errorf("policy package encoding_profile.image_processor integers must each be at least 1")
	case processor.Factor != processor.PatchSize*processor.MergeSize:
		return fmt.Errorf("policy package encoding_profile.image_processor.factor %d is not patch_size x merge_size", processor.Factor)
	case processor.MinPixels > processor.MaxPixels:
		return fmt.Errorf("policy package encoding_profile.image_processor.min_pixels exceeds max_pixels")
	}
	return nil
}

func checkPolicyVision(raw json.RawMessage) error {
	if raw == nil {
		return fmt.Errorf("policy package encoding_profile.vision is required")
	}
	var vision policyVision
	if err := decodeStrict(raw, &vision); err != nil {
		return fmt.Errorf("policy package encoding_profile.vision: %w", err)
	}
	switch {
	case vision.DType != policyVisionDTypeFloat32:
		return fmt.Errorf("policy package encoding_profile.vision.dtype %q is not %s", vision.DType, policyVisionDTypeFloat32)
	case vision.EmbedCast != policyVisionEmbedCastBF16:
		return fmt.Errorf("policy package encoding_profile.vision.embed_cast %q is not %s", vision.EmbedCast, policyVisionEmbedCastBF16)
	case !jsonEquals(vision.PerImageCalls, "true"):
		return fmt.Errorf("policy package encoding_profile.vision.per_image_calls %s is not true", orAbsent(vision.PerImageCalls))
	}
	return nil
}

// jsonEquals reports whether raw is the JSON value want, compared compacted.
func jsonEquals(raw json.RawMessage, want string) bool {
	if raw == nil {
		return false
	}
	var compact bytes.Buffer
	if json.Compact(&compact, raw) != nil {
		return false
	}
	return compact.String() == want
}

func orAbsent(raw json.RawMessage) string {
	if raw == nil {
		return "absent"
	}
	return string(raw)
}
