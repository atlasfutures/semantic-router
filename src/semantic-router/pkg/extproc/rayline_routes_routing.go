package extproc

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf16"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// The public routing controls on POST /v1/routes, as pathfinder#5664 pins
// them (src/rayline_router/serving/arc_public_contract.py, its OpenAPI and
// golden fixtures; spellings x-status: proposed until Marcin and Chi sign
// off). The body is the caller's inference body plus a routing block:
// {value_usd, temperature, prices, exclusive}. Every answer of a cell that
// serves them carries presets.swift and presets.sage; value_usd adds
// custom. All three are alternatives from the lookup's one encode, asked as
// the policy service's tier points; none advances the episode.
const (
	routesValueUSDMin      = 0.1
	routesValueUSDMax      = 5.0
	routesMaxPricedModels  = 256
	routesCallerPrices     = "arc-caller-prices.v1"
	routesCustomResult     = "custom"
	routesBasisCaller      = "caller"
	routesBasisCatalog     = "catalog"
	routesUnsupportedCause = "cell_does_not_support_exploration"
)

// routesPresets are the fixed tier operating points, in response order.
// Their V is ours: a caller cannot move it.
var routesPresets = []struct {
	name     string
	valueUSD float64
}{{"swift", 0.1}, {"sage", 4.5}}

// routesPublicModels are the canonical ARC ids and their kept aliases, with
// the fixed V of a tier id (0 for configurable arc-1.0).
var routesPublicModels = map[string]float64{
	"arc-swift": 0.1, "rayline/arc-swift": 0.1,
	"arc-sage": 4.5, "rayline/arc-sage": 4.5,
	"arc-1.0": 0, "arc-1": 0, "rayline/arc-1": 0, "rayline/arc-1.0": 0,
}

var routesPriceLegNames = [4]string{"input", "cache_read", "cache_write", "output"}

// routesContractError is a lookup the public contract refuses, with its
// machine-readable code.
type routesContractError struct {
	status  int
	errType string
	code    string
	message string
}

func routesInvalid(code string, message string) *routesContractError {
	return &routesContractError{status: 400, errType: "invalid_request_error", code: code, message: message}
}

// raylineRoutesRouting is a validated routing block. A member the caller did
// not send is nil, never a default.
type raylineRoutesRouting struct {
	valueUSD    *float64
	temperature *float64
	prices      map[string][4]float64
	exclusive   bool
	// exclusiveSent tells an explicit false from absence.
	exclusiveSent bool
	priceSHA256   string
	// responseFields is routing.response_fields, nil when not sent.
	responseFields []string
}

func (routing *raylineRoutesRouting) wantsPredictions() bool {
	return routing != nil && slices.Contains(routing.responseFields, "predictions")
}

func (routing *raylineRoutesRouting) pricedModels() []string {
	models := make([]string, 0, len(routing.prices))
	for model := range routing.prices {
		models = append(models, model)
	}
	sort.Strings(models)
	return models
}

// readRaylineRoutesRouting reads and removes the body's routing block, in the
// contract's check order (the model, the block's shape and members, then
// value_usd, temperature, prices and exclusive), so one invalid body is
// refused with the same code by every implementation. The body member is
// the one transport (memex-desktop#7538): #7531's retired
// x-rayline-exploration-temperature header is not read. cellAliases are
// this cell's own model names (auto_model_names), accepted beside the
// public ids.
//
// No block returns a nil routing and the body unchanged.
func readRaylineRoutesRouting(
	body []byte,
	cellAliases []string,
) (*raylineRoutesRouting, []byte, *routesContractError) {
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(body, &envelope); err != nil {
		return nil, nil, routesInvalid("invalid_body", "The request body must be a JSON object.")
	}
	raw, sent := envelope["routing"]
	if !sent {
		return nil, body, nil
	}
	if failure := routesCheckModel(envelope["model"], cellAliases); failure != nil {
		return nil, nil, failure
	}
	var members map[string]json.RawMessage
	if json.Unmarshal(raw, &members) != nil || members == nil {
		return nil, nil, routesInvalid("routing_invalid", "routing must be an object.")
	}
	routing := &raylineRoutesRouting{}
	if failure := routesReadMembers(routing, members, envelope["model"]); failure != nil {
		return nil, nil, failure
	}
	stripped, refusal := stripRoutingMember(body)
	if refusal != nil {
		return nil, nil, refusal
	}
	return routing, stripped, nil
}

// routesCheckModel refuses a missing model and an ARC-shaped id that is not
// one of ours. A non-ARC model passes: refusing it (409 non_arc_lookup) is
// the gateway's, by the caller's settings.
func routesCheckModel(raw json.RawMessage, cellAliases []string) *routesContractError {
	var model string
	if json.Unmarshal(raw, &model) != nil || model == "" {
		return routesInvalid("invalid_body", "model must be a non-empty string.")
	}
	if _, public := routesPublicModels[model]; public {
		return nil
	}
	for _, alias := range cellAliases {
		if alias == model {
			return nil
		}
	}
	bare := strings.TrimPrefix(model, "rayline/")
	if bare == "arc" || strings.HasPrefix(bare, "arc-") || strings.HasPrefix(bare, "arc@") {
		return routesInvalid("unknown_model", "Unknown ARC model or pin: "+model+
			". ARC ids are arc-1.0, arc-sage, arc-swift.")
	}
	return nil
}

func routesReadMembers(
	routing *raylineRoutesRouting,
	members map[string]json.RawMessage,
	rawModel json.RawMessage,
) *routesContractError {
	var unknown []string
	for name := range members {
		switch name {
		case "value_usd", "temperature", "prices", "exclusive", "response_fields":
		default:
			unknown = append(unknown, name)
		}
	}
	if len(unknown) > 0 {
		sort.Strings(unknown)
		return routesInvalid("routing_unknown_member", "routing has unsupported member(s): "+strings.Join(unknown, ", ")+".")
	}
	if raw, ok := members["value_usd"]; ok {
		value, finite := routesFiniteNumber(raw)
		if !finite {
			return routesInvalid("value_invalid", "routing.value_usd must be a finite number of USD.")
		}
		if value < routesValueUSDMin || value > routesValueUSDMax {
			return routesInvalid("value_out_of_bounds", fmt.Sprintf(
				"routing.value_usd must lie in [%g, %g]; it is never clamped.", routesValueUSDMin, routesValueUSDMax))
		}
		var model string
		_ = json.Unmarshal(rawModel, &model)
		if tierV := routesPublicModels[model]; tierV > 0 && value != tierV {
			return routesInvalid("tier_value_immutable", fmt.Sprintf(
				"%s decides at a fixed value_usd of %g; use arc-1.0 for a custom value.", model, tierV))
		}
		routing.valueUSD = &value
	}
	if raw, ok := members["temperature"]; ok {
		value, finite := routesFiniteNumber(raw)
		if !finite || value < 0 {
			return routesInvalid("temperature_invalid", "routing.temperature must be a finite number >= 0 "+
				"(a softmax temperature over normalized scores; 0 is greedy).")
		}
		routing.temperature = &value
	}
	if raw, ok := members["prices"]; ok {
		prices, failure := routesReadPrices(raw)
		if failure != nil {
			return failure
		}
		routing.prices = prices
	}
	if raw, ok := members["exclusive"]; ok {
		var exclusive bool
		if json.Unmarshal(raw, &exclusive) != nil || !bytes.Equal(bytes.TrimSpace(raw), []byte(strconv.FormatBool(exclusive))) {
			return routesInvalid("exclusive_invalid", "routing.exclusive must be true or false.")
		}
		if exclusive && routing.prices == nil {
			return routesInvalid("exclusive_without_prices",
				"routing.exclusive needs routing.prices: the priced models are the only candidates.")
		}
		routing.exclusive, routing.exclusiveSent = exclusive, true
	}
	if raw, ok := members["response_fields"]; ok {
		fields, failure := routesReadResponseFields(raw)
		if failure != nil {
			return failure
		}
		routing.responseFields = fields
	}
	if routing.prices != nil {
		routing.priceSHA256 = routesPriceSHA256(routing.prices, routing.exclusive)
	}
	return nil
}

// routesPublicResponseFields are the routing.response_fields values a lookup
// may ask for (pathfinder#5717).
var routesPublicResponseFields = map[string]bool{"predictions": true}

// routesReadResponseFields reads routing.response_fields: a non-empty array
// of distinct known values.
func routesReadResponseFields(raw json.RawMessage) ([]string, *routesContractError) {
	invalid := routesInvalid("response_fields_invalid",
		`routing.response_fields must be a non-empty array of distinct values from ["predictions"].`)
	var fields []string
	if json.Unmarshal(raw, &fields) != nil || len(fields) == 0 {
		return nil, invalid
	}
	seen := map[string]bool{}
	for _, field := range fields {
		if !routesPublicResponseFields[field] || seen[field] {
			return nil, invalid
		}
		seen[field] = true
	}
	return fields, nil
}

func routesReadPrices(raw json.RawMessage) (map[string][4]float64, *routesContractError) {
	invalid := routesInvalid("prices_invalid", fmt.Sprintf(
		"routing.prices must map 1 to %d model ids to their four USD-per-million-token legs.", routesMaxPricedModels))
	var models map[string]json.RawMessage
	if json.Unmarshal(raw, &models) != nil || len(models) == 0 || len(models) > routesMaxPricedModels {
		return nil, invalid
	}
	names := make([]string, 0, len(models))
	for name := range models {
		names = append(names, name)
	}
	sort.Strings(names)
	prices := make(map[string][4]float64, len(models))
	for _, model := range names {
		if model == "" {
			return nil, routesInvalid("prices_invalid", "routing.prices keys are non-empty model ids.")
		}
		var legs map[string]json.RawMessage
		if json.Unmarshal(models[model], &legs) != nil || legs == nil {
			return nil, routesInvalid("prices_invalid", fmt.Sprintf("routing.prices[%q] must be an object of four legs.", model))
		}
		var extra, missing []string
		for leg := range legs {
			if !routesIsPriceLeg(leg) {
				extra = append(extra, leg)
			}
		}
		if len(extra) > 0 {
			sort.Strings(extra)
			return nil, routesInvalid("prices_invalid", fmt.Sprintf(
				"routing.prices[%q] has unsupported member(s): %s.", model, strings.Join(extra, ", ")))
		}
		for _, leg := range routesPriceLegNames {
			if _, ok := legs[leg]; !ok {
				missing = append(missing, leg)
			}
		}
		if len(missing) > 0 {
			return nil, routesInvalid("price_leg_missing", fmt.Sprintf(
				"routing.prices[%q] is missing %s: all four legs are required (0 is a price; absent is not).",
				model, strings.Join(missing, ", ")))
		}
		var parsed [4]float64
		for index, leg := range routesPriceLegNames {
			value, finite := routesFiniteNumber(legs[leg])
			if !finite || value < 0 {
				return nil, routesInvalid("price_leg_invalid", fmt.Sprintf(
					"routing.prices[%q].%s must be a finite number >= 0 USD per million tokens.", model, leg))
			}
			parsed[index] = value
		}
		prices[model] = parsed
	}
	return prices, nil
}

func routesIsPriceLeg(name string) bool {
	for _, leg := range routesPriceLegNames {
		if leg == name {
			return true
		}
	}
	return false
}

// routesFiniteNumber reads a JSON number, finite. A bool, a string, null or
// a number beyond float64 is not one.
func routesFiniteNumber(raw json.RawMessage) (float64, bool) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value interface{}
	if decoder.Decode(&value) != nil {
		return 0, false
	}
	number, ok := value.(json.Number)
	if !ok {
		return 0, false
	}
	parsed, err := strconv.ParseFloat(number.String(), 64)
	if err != nil || math.IsNaN(parsed) || math.IsInf(parsed, 0) {
		return 0, false
	}
	return parsed, true
}

// routesPriceSHA256 is the canonical hash of a caller's price world,
// sha256(JCS({schema, exclusive, prices})) (RFC 8785): key order, integer
// versus float spelling and whitespace do not move it; any leg, model or
// exclusive change does.
func routesPriceSHA256(prices map[string][4]float64, exclusive bool) string {
	digest := sha256.Sum256([]byte(routesPriceJCS(prices, exclusive)))
	return hex.EncodeToString(digest[:])
}

// routesPriceJCS is the hash's preimage, the RFC 8785 form of
// {schema, exclusive, prices}.
func routesPriceJCS(prices map[string][4]float64, exclusive bool) string {
	var out strings.Builder
	out.WriteString(`{"exclusive":`)
	out.WriteString(strconv.FormatBool(exclusive))
	out.WriteString(`,"prices":{`)
	models := make([]string, 0, len(prices))
	for model := range prices {
		models = append(models, model)
	}
	sort.Slice(models, func(i, j int) bool { return jcsLess(models[i], models[j]) })
	for index, model := range models {
		if index > 0 {
			out.WriteByte(',')
		}
		legs := prices[model]
		out.WriteString(jcsString(model))
		// The legs' JCS order: cache_read, cache_write, input, output.
		fmt.Fprintf(&out, `:{"cache_read":%s,"cache_write":%s,"input":%s,"output":%s}`,
			jcsNumber(legs[1]), jcsNumber(legs[2]), jcsNumber(legs[0]), jcsNumber(legs[3]))
	}
	out.WriteString(`},"schema":` + jcsString(routesCallerPrices) + `}`)
	return out.String()
}

// jcsLess orders object keys by their UTF-16 code units (RFC 8785 3.2.3).
func jcsLess(left string, right string) bool {
	a, b := utf16.Encode([]rune(left)), utf16.Encode([]rune(right))
	for index := 0; index < len(a) && index < len(b); index++ {
		if a[index] != b[index] {
			return a[index] < b[index]
		}
	}
	return len(a) < len(b)
}

// jcsString is a JSON string as RFC 8785 writes it: only the quote, the
// backslash and control characters escaped, short forms where they exist.
func jcsString(value string) string {
	var out strings.Builder
	out.WriteByte('"')
	for _, r := range value {
		switch r {
		case '"':
			out.WriteString(`\"`)
		case '\\':
			out.WriteString(`\\`)
		case '\b':
			out.WriteString(`\b`)
		case '\f':
			out.WriteString(`\f`)
		case '\n':
			out.WriteString(`\n`)
		case '\r':
			out.WriteString(`\r`)
		case '\t':
			out.WriteString(`\t`)
		default:
			if r < 0x20 {
				fmt.Fprintf(&out, `\u%04x`, r)
			} else {
				out.WriteRune(r)
			}
		}
	}
	out.WriteByte('"')
	return out.String()
}

// jcsNumber is ECMAScript Number::toString of a finite value (RFC 8785
// 3.2.2.3), what JSON.stringify writes: the shortest round-trip digits, in
// plain notation for exponents in [-6, 21) and exponent notation outside.
func jcsNumber(value float64) string {
	if value == 0 {
		return "0"
	}
	sign := ""
	if value < 0 {
		sign, value = "-", -value
	}
	scientific := strconv.FormatFloat(value, 'e', -1, 64)
	mantissa, exponentText, _ := strings.Cut(scientific, "e")
	digits := strings.Replace(mantissa, ".", "", 1)
	exponent, _ := strconv.Atoi(exponentText)
	k := len(digits)
	n := exponent + 1 // value = 0.digits x 10^n
	var text string
	switch {
	case k <= n && n <= 21:
		text = digits + strings.Repeat("0", n-k)
	case 0 < n && n <= 21:
		text = digits[:n] + "." + digits[n:]
	case -6 < n && n <= 0:
		text = "0." + strings.Repeat("0", -n) + digits
	default:
		e := n - 1
		text = digits[:1]
		if k > 1 {
			text += "." + digits[1:]
		}
		if e >= 0 {
			text += "e+" + strconv.Itoa(e)
		} else {
			text += "e-" + strconv.Itoa(-e)
		}
	}
	return sign + text
}

// raylineRoutesControls turns a validated routing block into the lookup's
// decide-request members: swift and sage always, custom at the caller's V,
// each at cost_weight = 1 / V; the caller's exploration (surface decisions,
// never training data, the cell's salt) when it asked for T > 0 and the
// cell seeds draws; and its prices in place of the live feed's.
func raylineRoutesControls(
	routing *raylineRoutesRouting,
	settings *config.RaylineARCRoutesRoutingConfig,
	now time.Time,
) *selection.RaylineARCRouteControls {
	controls := &selection.RaylineARCRouteControls{}
	for _, preset := range routesPresets {
		controls.Points = append(controls.Points, raylinearc.PolicyTierPoint{Name: preset.name, CostWeight: 1 / preset.valueUSD})
	}
	if routing == nil {
		return controls
	}
	if routing.valueUSD != nil {
		controls.Points = append(controls.Points, raylinearc.PolicyTierPoint{Name: routesCustomResult, CostWeight: 1 / *routing.valueUSD})
	}
	if routing.temperature != nil && *routing.temperature > 0 && settings.ExplorationSalt != "" {
		controls.Exploration = &raylinearc.PolicyExplorationRequest{
			Salt: settings.ExplorationSalt,
			Temperature: &raylinearc.PolicyExplorationTemperature{
				Temperature:  *routing.temperature,
				Surface:      raylinearc.PolicyExplorationSurfaceDecisions,
				Source:       raylinearc.PolicyExplorationSourceCaller,
				TrainingData: raylinearc.PolicyTrainingDataNotAllowed,
			},
		}
	}
	if routing.prices != nil {
		rates := make(map[string]raylinearc.PolicyRoutingRate, len(routing.prices))
		for model, legs := range routing.prices {
			rates[model] = raylinearc.PolicyRoutingRate{
				Input: &legs[0], CacheRead: &legs[1], CacheWrite: &legs[2], Output: &legs[3],
			}
		}
		// routing_prices.basis caller (pathfinder#5668): one caller price
		// world for the decision and every tier pick, exactly four legs per
		// model, never filled from the package.
		controls.Prices = &raylinearc.PolicyRoutingPrices{
			Schema:       raylinearc.RoutingPricesSchema,
			StoreVersion: "caller:" + routing.priceSHA256,
			AsOf:         now.UTC().Format(time.RFC3339),
			Rates:        rates,
			Basis:        raylinearc.RoutingPricesBasisCaller,
			Exclusive:    routing.exclusive,
		}
		controls.PricedModels = routing.pricedModels()
		controls.Exclusive = routing.exclusive
		controls.PriceSHA256 = routing.priceSHA256
	}
	controls.Predictions = routing.wantsPredictions()
	return controls
}

// raylineRoutesRoutingControlsKey names a lookup's controls for its
// coalescing key: two lookups that differ in one never coalesce.
func raylineRoutesRoutingControlsKey(controls *selection.RaylineARCRouteControls) string {
	if controls == nil {
		return ""
	}
	encoded, err := json.Marshal(struct {
		Points      []raylinearc.PolicyTierPoint         `json:"points"`
		Exploration *raylinearc.PolicyExplorationRequest `json:"exploration"`
		Store       string                               `json:"store"`
		Exclusive   bool                                 `json:"exclusive"`
		Predictions bool                                 `json:"predictions"`
	}{controls.Points, controls.Exploration, routesStoreVersion(controls), controls.Exclusive, controls.Predictions})
	if err != nil {
		return "unencodable"
	}
	return string(encoded)
}

func routesStoreVersion(controls *selection.RaylineARCRouteControls) string {
	if controls.Prices == nil {
		return ""
	}
	return controls.Prices.StoreVersion
}

// --- the answer -------------------------------------------------------------

type raylineRoutesResultPrices struct {
	Basis       string  `json:"basis"`
	PriceSHA256 *string `json:"price_sha256"`
}

type raylineRoutesCandidatePi struct {
	Model string  `json:"model"`
	Pi    float64 `json:"pi"`
}

type raylineRoutesResultTemperature struct {
	Requested   float64                    `json:"requested"`
	Applied     *float64                   `json:"applied"`
	Status      string                     `json:"status"`
	Reason      *string                    `json:"reason"`
	Propensity  *float64                   `json:"propensity"`
	Explored    *bool                      `json:"explored"`
	Pi          []raylineRoutesCandidatePi `json:"pi"`
	DecisionKey *string                    `json:"decision_key"`
}

type raylineRoutesResult struct {
	Model          string                          `json:"model"`
	Worker         string                          `json:"worker"`
	Provider       string                          `json:"provider,omitempty"`
	Thinking       raylineRoutesThinking           `json:"thinking"`
	ThinkingLevel  string                          `json:"thinking_level,omitempty"`
	PolicyActionID string                          `json:"policy_action_id,omitempty"`
	ValueUSD       float64                         `json:"value_usd"`
	CostWeight     float64                         `json:"cost_weight"`
	Prices         raylineRoutesResultPrices       `json:"prices"`
	Temperature    *raylineRoutesResultTemperature `json:"temperature,omitempty"`
	// ModelChoice and Provenance are present together exactly when
	// predictions were asked for and applied: the result's stage-one row
	// (JSON null off a boundary) and the package and point it came from.
	ModelChoice json.RawMessage                `json:"model_choice,omitempty"`
	Provenance  *raylineRoutesResultProvenance `json:"provenance,omitempty"`
}

type raylineRoutesPresets struct {
	Swift raylineRoutesResult `json:"swift"`
	Sage  raylineRoutesResult `json:"sage"`
}

type raylineRoutesEchoStatus struct {
	Requested float64 `json:"requested"`
	Status    string  `json:"status"`
}

type raylineRoutesPricesEcho struct {
	Status      string   `json:"status"`
	PriceSHA256 string   `json:"price_sha256"`
	Exclusive   bool     `json:"exclusive"`
	Models      []string `json:"models"`
}

type raylineRoutesRoutingEcho struct {
	ValueUSD       *raylineRoutesEchoStatus `json:"value_usd,omitempty"`
	Temperature    *raylineRoutesEchoStatus `json:"temperature,omitempty"`
	Prices         *raylineRoutesPricesEcho `json:"prices,omitempty"`
	ResponseFields *raylineRoutesFieldsEcho `json:"response_fields,omitempty"`
	UnappliedKeys  []string                 `json:"unapplied_keys"`
}

type raylineRoutesFieldsEcho struct {
	Requested []string `json:"requested"`
	Status    string   `json:"status"`
}

// raylineRoutesAddResults writes the presets, the custom result, the routing
// echo and the lookup's own exploration record onto the answer. Every
// result decided in the request's price world; a temperature the serving
// path did not report applying is unsupported, never shown as applied.
func raylineRoutesAddResults(
	payload *raylineRoutesResponse,
	routing *raylineRoutesRouting,
	decision routerruntime.RouteDecision,
) error {
	results := map[string]raylineRoutesResult{}
	allSupported := true
	for _, result := range decision.Results {
		rendered, supported := raylineRoutesResultOf(result, routing)
		allSupported = allSupported && supported
		results[result.Name] = rendered
	}
	predictionsApplied := routing.wantsPredictions() && decision.PredictionsApplied
	if predictionsApplied {
		picks := make(map[string]*int, len(decision.Results))
		points := make([]raylineRoutesPoint, 0, len(decision.Results))
		for _, result := range decision.Results {
			picks[result.Name] = result.StageOneCandidate
			points = append(points, raylineRoutesPoint{name: result.Name, costWeight: result.CostWeight})
		}
		projection, err := projectRaylineRoutesPredictions(decision.Predictions, decision.PackageSHA256, picks, points)
		if err != nil {
			return err
		}
		payload.Candidates = marshalRaylineRoutesJSON(projection.Candidates)
		payload.CandidatesContext = marshalRaylineRoutesJSON(projection.Context)
		for name, rendered := range results {
			provenance := projection.Provenance[name]
			rendered.ModelChoice = marshalRaylineRoutesJSON(projection.Choices[name])
			rendered.Provenance = &provenance
			results[name] = rendered
		}
	}
	swift, haveSwift := results["swift"]
	sage, haveSage := results["sage"]
	if !haveSwift || !haveSage {
		return fmt.Errorf("the decision carries no preset result")
	}
	payload.Presets = &raylineRoutesPresets{Swift: swift, Sage: sage}
	if routing == nil {
		return nil
	}
	echo := &raylineRoutesRoutingEcho{UnappliedKeys: []string{}}
	if routing.valueUSD != nil {
		custom, ok := results[routesCustomResult]
		if !ok {
			return fmt.Errorf("the decision carries no custom result")
		}
		payload.Custom = &custom
		echo.ValueUSD = &raylineRoutesEchoStatus{Requested: *routing.valueUSD, Status: "applied"}
	}
	if routing.temperature != nil {
		status := "applied"
		if !allSupported {
			status = "unsupported"
			echo.UnappliedKeys = append(echo.UnappliedKeys, "temperature")
		}
		echo.Temperature = &raylineRoutesEchoStatus{Requested: *routing.temperature, Status: status}
		if decision.Exploration != nil {
			if encoded, err := json.Marshal(decision.Exploration); err == nil {
				payload.Exploration = encoded
			}
		}
	}
	if routing.responseFields != nil {
		status := "applied"
		if routing.wantsPredictions() && !predictionsApplied {
			status = "unsupported"
			echo.UnappliedKeys = append(echo.UnappliedKeys, "response_fields")
		}
		echo.ResponseFields = &raylineRoutesFieldsEcho{Requested: append([]string(nil), routing.responseFields...), Status: status}
	}
	if routing.prices != nil {
		echo.Prices = &raylineRoutesPricesEcho{
			Status: "applied", PriceSHA256: routing.priceSHA256, Exclusive: routing.exclusive, Models: routing.pricedModels(),
		}
	}
	sort.Strings(echo.UnappliedKeys)
	payload.Routing = echo
	return nil
}

// raylineRoutesResultOf renders one result, and reports whether its
// temperature (when one was asked) is supported.
func raylineRoutesResultOf(result routerruntime.RouteResult, routing *raylineRoutesRouting) (raylineRoutesResult, bool) {
	rendered := raylineRoutesResult{
		Model:          result.WorkerModel,
		Worker:         result.SelectedWorker,
		Provider:       result.Provider,
		Thinking:       raylineRoutesThinkingOf(result.Thinking),
		ThinkingLevel:  result.ThinkingLevel,
		PolicyActionID: result.PolicyActionID,
		ValueUSD:       routesValueOf(result, routing),
		CostWeight:     result.CostWeight,
		Prices:         raylineRoutesResultPrices{Basis: routesBasisCatalog},
	}
	if routing == nil {
		return rendered, true
	}
	if routing.prices != nil {
		hash := routing.priceSHA256
		rendered.Prices = raylineRoutesResultPrices{Basis: routesBasisCaller, PriceSHA256: &hash}
	}
	if routing.temperature == nil {
		return rendered, true
	}
	rendered.Temperature = raylineRoutesTemperatureOf(*routing.temperature, result.Exploration)
	return rendered, rendered.Temperature.Status != "unsupported"
}

// routesValueOf is the V a result was decided at: the preset's own, or the
// caller's for custom. Read from the request, never back from a weight.
func routesValueOf(result routerruntime.RouteResult, routing *raylineRoutesRouting) float64 {
	for _, preset := range routesPresets {
		if preset.name == result.Name {
			return preset.valueUSD
		}
	}
	if result.Name == routesCustomResult && routing != nil && routing.valueUSD != nil {
		return *routing.valueUSD
	}
	return 1 / result.CostWeight
}

func raylineRoutesTemperatureOf(requested float64, record *raylinearc.PolicyExplorationRecord) *raylineRoutesResultTemperature {
	rendered := &raylineRoutesResultTemperature{Requested: requested}
	switch {
	case requested == 0:
		zero, unexplored := 0.0, false
		rendered.Status, rendered.Applied, rendered.Explored = "greedy", &zero, &unexplored
	case record == nil:
		reason := routesUnsupportedCause
		rendered.Status, rendered.Reason = "unsupported", &reason
	default:
		rendered.Status = "not_applied"
		if record.Applied {
			rendered.Status = "applied"
		}
		rendered.Applied, rendered.Reason = record.Temperature, record.Reason
		rendered.Propensity, rendered.Explored, rendered.DecisionKey = record.Propensity, record.Explored, record.DecisionKey
		rendered.Pi = routesPiByModel(record)
	}
	return rendered
}

// routesPiByModel is the record's distribution per model, in catalog order:
// candidates of one model at several efforts add up.
func routesPiByModel(record *raylinearc.PolicyExplorationRecord) []raylineRoutesCandidatePi {
	if len(record.Offered) == 0 || len(record.Offered) != len(record.Pi) {
		return nil
	}
	var out []raylineRoutesCandidatePi
	index := map[string]int{}
	for position, candidate := range record.Offered {
		model := ""
		if name := candidate["model"]; name != nil {
			model = *name
		}
		if at, seen := index[model]; seen {
			out[at].Pi += record.Pi[position]
			continue
		}
		index[model] = len(out)
		out = append(out, raylineRoutesCandidatePi{Model: model, Pi: record.Pi[position]})
	}
	return out
}

// createRaylineRoutesCodedError answers a contract refusal in the Anthropic
// envelope with its machine-readable code (pathfinder#5664's error schema).
func (r *OpenAIRouter) createRaylineRoutesCodedError(
	ctx *RequestContext,
	failure *routesContractError,
) *ext_proc.ProcessingResponse {
	response := r.createJSONResponse(failure.status, map[string]interface{}{
		"type": "error",
		"error": map[string]interface{}{
			"type":    failure.errType,
			"message": failure.message,
			"code":    failure.code,
		},
	})
	if ctx != nil {
		ctx.ImmediateResponseEncoded = true
	}
	return response
}

// raylineRoutesDecisionWorkerModels are the provider model ids the lookup's
// decision dispatches: the ids routing.prices may name.
func (r *OpenAIRouter) raylineRoutesDecisionWorkerModels() map[string]bool {
	models := map[string]bool{}
	if r == nil || r.Config == nil {
		return models
	}
	for index := range r.Config.Decisions {
		decision := &r.Config.Decisions[index]
		if !raylineARCSelection(decision.Algorithm) || decision.Algorithm.RaylineARC == nil ||
			!decision.Algorithm.RaylineARC.RoutesAPI.Enabled {
			continue
		}
		for _, ref := range decision.ModelRefs {
			models[policyWorkerManifest(r.Config, ref).Model] = true
		}
		return models
	}
	return models
}

// raylineRoutesUnknownPriceModels are the priced models this cell does not
// offer, sorted.
func raylineRoutesUnknownPriceModels(routing *raylineRoutesRouting, offered map[string]bool) []string {
	if routing == nil {
		return nil
	}
	var unknown []string
	for _, model := range routing.pricedModels() {
		if !offered[model] {
			unknown = append(unknown, model)
		}
	}
	return unknown
}
