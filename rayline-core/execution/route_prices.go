package execution

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode/utf16"
)

// Routes API V1 price members (pathfinder#6159): the caller names candidates
// with routing.models / routing.excluded_models and prices them with
// routing.prices (any subset of the four legs per model) and/or
// routing.price_profile {id, version?}. The gateway resolves every candidate's
// four legs (caller, then profile, then catalog) and echoes that table under
// routing.prices with each leg's source and a digest over the table. There is
// no routing.exclusive: the resolved candidates are the only candidates.

// routePricesSchema tags the public routing.prices.price_sha256 preimage.
const routePricesSchema = "arc-route-prices.v1"

var priceLegs = [...]string{"input", "cache_read", "cache_write", "output"}

var priceProfileID = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$`)

// priceProfileRef is routing.price_profile as configured and sent.
type priceProfileRef struct {
	ID      string `json:"id"`
	Version int64  `json:"version,omitempty"`
}

// callerPrices is routing.prices as configured: model -> leg -> USD per
// million tokens, only the legs the profile set.
type callerPrices map[string]map[string]float64

// resolvedPrice is one candidate's four resolved legs and their sources.
type resolvedPrice struct {
	Input      *float64          `json:"input"`
	CacheRead  *float64          `json:"cache_read"`
	CacheWrite *float64          `json:"cache_write"`
	Output     *float64          `json:"output"`
	Source     map[string]string `json:"source"`
}

func (price resolvedPrice) leg(name string) *float64 {
	switch name {
	case "input":
		return price.Input
	case "cache_read":
		return price.CacheRead
	case "cache_write":
		return price.CacheWrite
	case "output":
		return price.Output
	}
	return nil
}

// pricesEcho is routing.prices in the answer.
type pricesEcho struct {
	Status string                   `json:"status"`
	Models map[string]resolvedPrice `json:"models"`
	Unused []string                 `json:"unused"`
	SHA256 string                   `json:"price_sha256"`
}

// priceProfileEcho is routing.price_profile in the answer. Only the identity
// members are read back: the profile's resolved rows and coverage lists are
// the gateway's record, and the price world the route was decided in is
// acknowledged from routing.prices leg by leg.
type priceProfileEcho struct {
	Status string `json:"status"`
	AppliedPriceProfile
	Schema string `json:"schema"`
}

// modelsEcho is routing.models in the answer.
type modelsEcho struct {
	Status   string   `json:"status"`
	Models   []string `json:"models"`
	Resolved []string `json:"resolved"`
}

// excludedModelsEcho is routing.excluded_models in the answer.
type excludedModelsEcho struct {
	Status         string   `json:"status"`
	ExcludedModels []string `json:"excluded_models"`
}

// resultPrices is the price world a result was decided in: the top-level
// identity's and each preset/custom result's `prices`.
type resultPrices struct {
	Basis  string  `json:"basis"`
	SHA256 *string `json:"price_sha256"`
}

func parseCallerPrices(raw json.RawMessage) (callerPrices, error) {
	var rows map[string]map[string]json.RawMessage
	if err := json.Unmarshal(raw, &rows); err != nil || len(rows) == 0 || len(rows) > 256 {
		return nil, fmt.Errorf("routing prices must be a nonempty object of at most 256 priced models")
	}
	prices := make(callerPrices, len(rows))
	for model, row := range rows {
		if model == "" || len(row) == 0 {
			return nil, fmt.Errorf("routing prices rows must name a model and at least one leg")
		}
		legs := make(map[string]float64, len(row))
		for leg, value := range row {
			// A null leg decodes into a nil pointer, never into a zero price.
			var price *float64
			if !isPriceLeg(leg) || json.Unmarshal(value, &price) != nil || price == nil || math.IsNaN(*price) || math.IsInf(*price, 0) || *price < 0 {
				return nil, fmt.Errorf("routing prices legs are input, cache_read, cache_write and output: nonnegative finite numbers, never null")
			}
			legs[leg] = *price
		}
		prices[model] = legs
	}
	return prices, nil
}

func parsePriceProfile(raw json.RawMessage) (*priceProfileRef, error) {
	var members map[string]json.RawMessage
	if err := json.Unmarshal(raw, &members); err != nil || members == nil {
		return nil, fmt.Errorf("routing price_profile must be an object {id, version?}")
	}
	var profile priceProfileRef
	for key, value := range members {
		switch key {
		case "id":
			if err := json.Unmarshal(value, &profile.ID); err != nil || !priceProfileID.MatchString(profile.ID) {
				return nil, fmt.Errorf("routing price_profile.id must be a profile ID")
			}
		case "version":
			if err := json.Unmarshal(value, &profile.Version); err != nil || profile.Version <= 0 || profile.Version > 9007199254740991 {
				return nil, fmt.Errorf("routing price_profile.version must be a positive safe integer")
			}
		default:
			return nil, fmt.Errorf("routing price_profile member %q is not id or version", key)
		}
	}
	if profile.ID == "" {
		return nil, fmt.Errorf("routing price_profile requires an id")
	}
	return &profile, nil
}

// parseModelPatterns reads routing.models or routing.excluded_models: distinct
// nonempty exact ids or `*` patterns, at most 256.
func parseModelPatterns(raw json.RawMessage, member string) ([]string, error) {
	var patterns []string
	if err := json.Unmarshal(raw, &patterns); err != nil || patterns == nil || len(patterns) > 256 {
		return nil, fmt.Errorf("routing %s must be an array of at most 256 model ids or patterns", member)
	}
	seen := make(map[string]bool, len(patterns))
	for _, pattern := range patterns {
		if pattern == "" || seen[pattern] {
			return nil, fmt.Errorf("routing %s entries must be distinct nonempty strings", member)
		}
		seen[pattern] = true
	}
	return patterns, nil
}

func isPriceLeg(name string) bool {
	for _, leg := range priceLegs {
		if leg == name {
			return true
		}
	}
	return false
}

// routePricesSHA256 is the public price digest, recomputed from the echoed
// table: sha256 of the RFC 8785 canonical form of
// {"schema": "arc-route-prices.v1", "prices": {id: {input, cache_read,
// cache_write, output}}}. Every row needs all four legs, each a nonnegative
// finite number with a known source.
func routePricesSHA256(models map[string]resolvedPrice) (string, error) {
	ids := make([]string, 0, len(models))
	for id := range models {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return lessUTF16(ids[i], ids[j]) })
	var canonical strings.Builder
	canonical.WriteString(`{"prices":{`)
	for index, id := range ids {
		row := models[id]
		if index > 0 {
			canonical.WriteByte(',')
		}
		canonical.WriteString(jcsString(id))
		canonical.WriteString(`:{`)
		for legIndex, leg := range [...]string{"cache_read", "cache_write", "input", "output"} {
			price := row.leg(leg)
			source := row.Source[leg]
			if price == nil || math.IsNaN(*price) || math.IsInf(*price, 0) || *price < 0 || (source != "caller" && source != "profile" && source != "catalog") {
				return "", fmt.Errorf("Decisions returned an incomplete resolved price for %s", id)
			}
			if legIndex > 0 {
				canonical.WriteByte(',')
			}
			canonical.WriteString(jcsString(leg))
			canonical.WriteByte(':')
			canonical.WriteString(jcsNumber(*price))
		}
		canonical.WriteByte('}')
	}
	canonical.WriteString(`},"schema":`)
	canonical.WriteString(jcsString(routePricesSchema))
	canonical.WriteByte('}')
	digest := sha256.Sum256([]byte(canonical.String()))
	return hex.EncodeToString(digest[:]), nil
}

// priceBasis names the sources a resolved table used: `catalog` when no
// caller or profile leg was used, that one source when only it set a leg,
// `mixed` when both did.
func priceBasis(models map[string]resolvedPrice) string {
	caller, profile := false, false
	for _, row := range models {
		for _, source := range row.Source {
			caller = caller || source == "caller"
			profile = profile || source == "profile"
		}
	}
	switch {
	case caller && profile:
		return "mixed"
	case caller:
		return "caller"
	case profile:
		return "profile"
	}
	return "catalog"
}

// lessUTF16 orders object keys by UTF-16 code units, as RFC 8785 requires.
func lessUTF16(a, b string) bool {
	ua, ub := utf16.Encode([]rune(a)), utf16.Encode([]rune(b))
	for index := 0; index < len(ua) && index < len(ub); index++ {
		if ua[index] != ub[index] {
			return ua[index] < ub[index]
		}
	}
	return len(ua) < len(ub)
}

// jcsString serializes a string as RFC 8785 does: only the quote, the
// backslash and control characters are escaped.
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

// jcsNumber serializes a finite number as ECMAScript Number::toString, which
// RFC 8785 adopts: the shortest round-trip digits, positional notation for
// decimal exponents in (-7, 21], exponential notation otherwise.
func jcsNumber(value float64) string {
	if value == 0 {
		return "0"
	}
	sign := ""
	if value < 0 {
		sign, value = "-", -value
	}
	// Shortest round-trip digits and a decimal exponent: d.ddddde±xx.
	scientific := strconv.FormatFloat(value, 'e', -1, 64)
	mantissa, exponent, _ := strings.Cut(scientific, "e")
	digits := strings.Replace(mantissa, ".", "", 1)
	power, err := strconv.Atoi(exponent)
	if err != nil {
		return sign + scientific
	}
	k, n := len(digits), power+1 // value = 0.digits × 10^n
	switch {
	case k <= n && n <= 21:
		return sign + digits + strings.Repeat("0", n-k)
	case 0 < n && n <= 21:
		return sign + digits[:n] + "." + digits[n:]
	case -6 < n && n <= 0:
		return sign + "0." + strings.Repeat("0", -n) + digits
	}
	exponentSign := "+"
	if n-1 < 0 {
		exponentSign = "-"
	}
	fraction := ""
	if k > 1 {
		fraction = "." + digits[1:]
	}
	return sign + digits[:1] + fraction + "e" + exponentSign + strconv.Itoa(absInt(n-1))
}

func absInt(value int) int {
	if value < 0 {
		return -value
	}
	return value
}

// acknowledgePrices checks the answer's candidate and price echoes against
// the members this client sent and the selected result's price receipt. With
// nothing priced, the result must have been decided at the catalog. With
// prices or a profile, routing.prices must be applied, its digest must be the
// digest of the table it echoes (and the configured price_sha256 when one is
// pinned), every caller-priced model must be resolved leg for leg at the
// caller's price or listed unused, the selected model must be one of the
// priced candidates, and the result's receipt must name that table's basis
// and digest. It returns the applied profile when one was sent.
func (provider *HTTPDecisions) acknowledgePrices(result decisionsResponse, selectedModel string, prices *resultPrices) (*AppliedPriceProfile, error) {
	routing := result.Routing
	if err := provider.acknowledgeCandidates(result, selectedModel); err != nil {
		return nil, err
	}
	if provider.prices == nil && provider.profile == nil {
		if routing.Prices != nil || routing.PriceProfile != nil || prices == nil || prices.Basis != "catalog" || prices.SHA256 != nil {
			return nil, fmt.Errorf("Decisions did not confirm the requested catalog price world")
		}
		return nil, nil
	}
	echo := routing.Prices
	if echo == nil || echo.Status != "applied" || len(echo.Models) == 0 || echo.Unused == nil {
		return nil, fmt.Errorf("Decisions did not apply the requested price world")
	}
	digest, err := routePricesSHA256(echo.Models)
	if err != nil {
		return nil, err
	}
	if echo.SHA256 != digest {
		return nil, fmt.Errorf("Decisions price digest does not match its resolved price table")
	}
	if pinned := provider.config.Decisions.PriceSHA256; pinned != "" && digest != pinned {
		return nil, fmt.Errorf("Decisions resolved price world does not match the configured price_sha256")
	}
	if routing.Models != nil && !sameSet(routing.Models.Resolved, echo.Models) {
		return nil, fmt.Errorf("Decisions priced a different candidate set than it resolved")
	}
	for _, id := range echo.Unused {
		if _, candidate := echo.Models[id]; candidate {
			return nil, fmt.Errorf("Decisions listed a candidate as an unused price")
		}
	}
	// The route must have picked one of the candidates it priced: a selected
	// model outside the table (for example a caller-priced one listed
	// unused) was not decided in the price world the receipt names. Unused
	// and resolved are already disjoint from and equal to the table.
	if _, candidate := echo.Models[selectedModel]; !candidate {
		return nil, fmt.Errorf("Decisions selected a model outside the priced candidates")
	}
	for model, legs := range provider.prices {
		row, candidate := echo.Models[model]
		if !candidate {
			if !containsString(echo.Unused, model) {
				return nil, fmt.Errorf("Decisions did not account for the caller price of %s", model)
			}
			continue
		}
		for leg, price := range legs {
			if resolved := row.leg(leg); resolved == nil || *resolved != price || row.Source[leg] != "caller" {
				return nil, fmt.Errorf("Decisions did not apply the caller %s price of %s", leg, model)
			}
		}
	}
	// The reverse: a leg may claim a caller or profile source only where
	// this client supplied one. Sources are outside the digest and set the
	// receipt's basis, so an unclaimed source would be a price world this
	// client never asked for, acknowledged as if it had.
	for model, row := range echo.Models {
		for leg, source := range row.Source {
			switch source {
			case "caller":
				if price, configured := provider.prices[model][leg]; !configured || row.leg(leg) == nil || *row.leg(leg) != price {
					return nil, fmt.Errorf("Decisions attributed the %s price of %s to the caller, who did not set it", leg, model)
				}
			case "profile":
				if provider.profile == nil {
					return nil, fmt.Errorf("Decisions attributed the %s price of %s to a profile that was not sent", leg, model)
				}
			}
		}
	}
	if prices == nil || prices.Basis != priceBasis(echo.Models) || prices.SHA256 == nil || *prices.SHA256 != digest {
		return nil, fmt.Errorf("Decisions selected result was not decided in the applied price world")
	}
	if provider.profile == nil {
		return nil, nil
	}
	profile := routing.PriceProfile
	if profile == nil || profile.Status != "applied" || profile.ID != provider.profile.ID || profile.Version <= 0 || profile.Version > 9007199254740991 || (provider.profile.Version != 0 && profile.Version != provider.profile.Version) || !strings.HasPrefix(profile.ContentHash, "sha256:") || !validSHA256(strings.TrimPrefix(profile.ContentHash, "sha256:")) || (profile.Schema != "price-profile-rules.v1" && profile.Schema != "legacy") {
		return nil, fmt.Errorf("Decisions did not apply the requested price profile")
	}
	return &profile.AppliedPriceProfile, nil
}

// acknowledgeCandidates checks the routing.models and
// routing.excluded_models echoes against the members this client sent:
// applied, the entries sent, distinct resolved ids, and the selected model
// among them. Applied candidates confine the pick, priced or not, so every
// result path runs this, including a top result without an identity.
func (provider *HTTPDecisions) acknowledgeCandidates(result decisionsResponse, selectedModel string) error {
	routing := result.Routing
	if provider.models != nil {
		echo := routing.Models
		if echo == nil || echo.Status != "applied" || !equalStrings(echo.Models, provider.models) || len(echo.Resolved) == 0 || !distinctStrings(echo.Resolved) {
			return fmt.Errorf("Decisions did not apply the requested candidate models")
		}
		if !containsString(echo.Resolved, selectedModel) {
			return fmt.Errorf("Decisions selected a model outside the resolved candidates")
		}
	}
	if provider.excludedModels != nil {
		echo := routing.ExcludedModels
		if echo == nil || echo.Status != "applied" || !equalStrings(echo.ExcludedModels, provider.excludedModels) {
			return fmt.Errorf("Decisions did not apply the requested excluded models")
		}
	}
	return nil
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for index := range a {
		if a[index] != b[index] {
			return false
		}
	}
	return true
}

func distinctStrings(list []string) bool {
	seen := make(map[string]bool, len(list))
	for _, entry := range list {
		if seen[entry] {
			return false
		}
		seen[entry] = true
	}
	return true
}

func containsString(list []string, value string) bool {
	for _, entry := range list {
		if entry == value {
			return true
		}
	}
	return false
}

// sameSet reports whether ids, without repeats, are exactly the keys of models.
func sameSet(ids []string, models map[string]resolvedPrice) bool {
	if len(ids) != len(models) {
		return false
	}
	seen := make(map[string]bool, len(ids))
	for _, id := range ids {
		if _, exists := models[id]; !exists || seen[id] {
			return false
		}
		seen[id] = true
	}
	return true
}

// callerPricesSchema tags the previous contract's digest preimage, which the
// serving cell still uses internally and the Routes API never returns.
const callerPricesSchema = "arc-caller-prices.v1"

// isCallerPricesDigest reports whether pin is the previous contract's digest
// of the configured prices: sha256 of the RFC 8785 form of
// {"schema": "arc-caller-prices.v1", "exclusive": bool, "prices": table},
// for either exclusivity. That digest needed every row's four legs; a
// partial table has none, so such a pin is just a wrong V1 digest.
func isCallerPricesDigest(pin string, prices callerPrices) bool {
	if len(prices) == 0 {
		return false
	}
	ids := make([]string, 0, len(prices))
	for id, legs := range prices {
		if len(legs) != len(priceLegs) {
			return false
		}
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return lessUTF16(ids[i], ids[j]) })
	for _, exclusive := range []bool{false, true} {
		var canonical strings.Builder
		fmt.Fprintf(&canonical, `{"exclusive":%t,"prices":{`, exclusive)
		for index, id := range ids {
			if index > 0 {
				canonical.WriteByte(',')
			}
			canonical.WriteString(jcsString(id))
			canonical.WriteString(`:{`)
			for legIndex, leg := range [...]string{"cache_read", "cache_write", "input", "output"} {
				if legIndex > 0 {
					canonical.WriteByte(',')
				}
				canonical.WriteString(jcsString(leg))
				canonical.WriteByte(':')
				canonical.WriteString(jcsNumber(prices[id][leg]))
			}
			canonical.WriteByte('}')
		}
		canonical.WriteString(`},"schema":`)
		canonical.WriteString(jcsString(callerPricesSchema))
		canonical.WriteByte('}')
		digest := sha256.Sum256([]byte(canonical.String()))
		if hex.EncodeToString(digest[:]) == pin {
			return true
		}
	}
	return false
}

// isRoutePriceKey names the V1 selection and price members of routing.
func isRoutePriceKey(key string) bool {
	return key == "models" || key == "excluded_models" || key == "prices" || key == "price_profile"
}

// routingKeyError refuses a routing member this client cannot acknowledge,
// naming the V1 replacement for the members the previous contract had.
func routingKeyError(mode, key string) error {
	switch key {
	case "exclusive":
		return fmt.Errorf("routing exclusive is not a Routes API V1 member: the resolved candidates are the only candidates; limit them with routing.models")
	case "price_profile_id", "price_profile_version":
		return fmt.Errorf("routing %s is not a Routes API V1 member: send routing.price_profile {id, version?}", key)
	}
	return fmt.Errorf("%s routing key %q has no supported applied acknowledgement", mode, key)
}
