package authz

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"net/url"
	"regexp"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"
)

const (
	testAccessKeyID     = "AKIDEXAMPLETESTONLY"
	testSecretAccessKey = "secret/EXAMPLE+test+only"
	testSessionToken    = "session-token-EXAMPLE-test-only"
)

// staticAWSCredentials loads fixed test credentials expiring at expires
// (zero: they do not expire) in region.
func staticAWSCredentials(region string, expires time.Time, retrieves *atomic.Int32) CredentialsLoader {
	return func(context.Context) (aws.CredentialsProvider, string, error) {
		provider := aws.CredentialsProviderFunc(func(context.Context) (aws.Credentials, error) {
			if retrieves != nil {
				retrieves.Add(1)
			}
			return aws.Credentials{
				AccessKeyID: testAccessKeyID, SecretAccessKey: testSecretAccessKey, SessionToken: testSessionToken,
				CanExpire: !expires.IsZero(), Expires: expires,
			}, nil
		})
		return provider, region, nil
	}
}

// decodeBedrockKey undoes the key encoding and parses the presigned request.
func decodeBedrockKey(t *testing.T, key string) *url.URL {
	t.Helper()
	if !strings.HasPrefix(key, "bedrock-api-key-") {
		t.Fatalf("key has no bedrock-api-key- prefix")
	}
	raw, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(key, "bedrock-api-key-"))
	if err != nil {
		t.Fatalf("key is not standard base64: %v", err)
	}
	text := string(raw)
	if !strings.HasSuffix(text, "&Version=1") {
		t.Fatalf("decoded key does not end in &Version=1")
	}
	if strings.HasPrefix(text, "https://") {
		t.Fatalf("decoded key keeps its scheme")
	}
	parsed, err := url.Parse("https://" + strings.TrimSuffix(text, "&Version=1"))
	if err != nil {
		t.Fatalf("decoded key is not a URL: %v", err)
	}
	return parsed
}

func TestBedrockShortTermKeyIsAPresignedCallWithBearerToken(t *testing.T) {
	issued := time.Date(2026, 11, 23, 15, 4, 5, 0, time.UTC)
	source := NewBedrockShortTermKeySourceWith(staticAWSCredentials("us-east-1", time.Time{}, nil), func() time.Time { return issued })

	tok, err := source.Token(context.Background())
	if err != nil {
		t.Fatalf("Token: %v", err)
	}
	presigned := decodeBedrockKey(t, tok.Value)
	if presigned.Host != "bedrock.amazonaws.com" || presigned.Path != "/" {
		t.Fatalf("presigned request is for %s%s, want bedrock.amazonaws.com/", presigned.Host, presigned.Path)
	}
	query := presigned.Query()
	want := map[string]string{
		"Action":               "CallWithBearerToken",
		"X-Amz-Algorithm":      "AWS4-HMAC-SHA256",
		"X-Amz-Credential":     testAccessKeyID + "/20261123/us-east-1/bedrock/aws4_request",
		"X-Amz-Date":           "20261123T150405Z",
		"X-Amz-Expires":        "43200",
		"X-Amz-SignedHeaders":  "host",
		"X-Amz-Security-Token": testSessionToken,
	}
	for name, value := range want {
		if got := query.Get(name); got != value {
			t.Errorf("%s = %q, want %q", name, got, value)
		}
	}
	if !regexp.MustCompile(`^[0-9a-f]{64}$`).MatchString(query.Get("X-Amz-Signature")) {
		t.Errorf("X-Amz-Signature is not a SigV4 signature")
	}
	// The signature botocore's SigV4QueryAuth gives the same request, as
	// aws-bedrock-token-generator-python builds it, for these test
	// credentials and time.
	if got, want := query.Get("X-Amz-Signature"), "f28b585a9803402dbdca3eb6c668e20db89beeb9f108e0810569be2eb88b1f10"; got != want {
		t.Errorf("X-Amz-Signature = %s, want botocore's %s", got, want)
	}
	if strings.Contains(tok.Value, testSecretAccessKey) {
		t.Fatal("the key carries the secret access key")
	}
	if !tok.Expiry.Equal(issued.Add(12*time.Hour)) || !tok.RefreshAt.Equal(issued.Add(11*time.Hour)) {
		t.Fatalf("expiry %v refresh %v, want 12h and 11h after issue", tok.Expiry, tok.RefreshAt)
	}

	again, err := source.Token(context.Background())
	if err != nil || again.Value != tok.Value {
		t.Fatalf("the same credentials and time signed a different key")
	}
}

func TestBedrockShortTermKeyLivesNoLongerThanItsCredentials(t *testing.T) {
	issued := time.Date(2026, 11, 23, 15, 0, 0, 0, time.UTC)
	now := func() time.Time { return issued }

	source := NewBedrockShortTermKeySourceWith(staticAWSCredentials("us-east-1", issued.Add(6*time.Hour), nil), now)
	tok, err := source.Token(context.Background())
	if err != nil {
		t.Fatalf("Token: %v", err)
	}
	if !tok.Expiry.Equal(issued.Add(6*time.Hour)) || !tok.RefreshAt.Equal(issued.Add(5*time.Hour+30*time.Minute)) {
		t.Fatalf("expiry %v refresh %v, want the credentials' 6h and 11/12 of it", tok.Expiry, tok.RefreshAt)
	}

	// Credentials about to expire still sign a key, refreshed no sooner
	// than once a minute.
	source = NewBedrockShortTermKeySourceWith(staticAWSCredentials("us-east-1", issued.Add(20*time.Second), nil), now)
	tok, err = source.Token(context.Background())
	if err != nil {
		t.Fatalf("Token: %v", err)
	}
	if !tok.RefreshAt.Equal(issued.Add(time.Minute)) {
		t.Fatalf("refresh %v, want a minute after issue", tok.RefreshAt)
	}

	source = NewBedrockShortTermKeySourceWith(staticAWSCredentials("us-east-1", issued.Add(-time.Second), nil), now)
	if _, err := source.Token(context.Background()); err == nil {
		t.Fatal("expired credentials signed a key")
	}
}

func TestBedrockShortTermKeySourceRefusals(t *testing.T) {
	now := time.Now
	cases := map[string]CredentialsLoader{
		"no region": staticAWSCredentials(" ", time.Time{}, nil),
		"load fails": func(context.Context) (aws.CredentialsProvider, string, error) {
			return nil, "", errors.New("no configuration")
		},
		"no provider": func(context.Context) (aws.CredentialsProvider, string, error) { return nil, "us-east-1", nil },
		"empty credentials": func(context.Context) (aws.CredentialsProvider, string, error) {
			return aws.CredentialsProviderFunc(func(context.Context) (aws.Credentials, error) {
				return aws.Credentials{}, nil
			}), "us-east-1", nil
		},
		"retrieve fails": func(context.Context) (aws.CredentialsProvider, string, error) {
			return aws.CredentialsProviderFunc(func(context.Context) (aws.Credentials, error) {
				return aws.Credentials{}, errors.New("pod identity agent unreachable")
			}), "us-east-1", nil
		},
	}
	for name, load := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := NewBedrockShortTermKeySourceWith(load, now).Token(context.Background()); err == nil {
				t.Fatal("Token succeeded")
			}
		})
	}
}

func TestBedrockKeyProviderServesOnlyBedrockAndFailsClosed(t *testing.T) {
	var fail atomic.Bool
	source := tokenSourceFunc(func(ctx context.Context) (AccessToken, error) {
		if fail.Load() {
			return AccessToken{}, errors.New("no credentials")
		}
		return AccessToken{Value: "bedrock-api-key-x", Expiry: time.Now().Add(time.Hour)}, nil
	})
	p := NewBedrockKeyProvider(source)
	if p.Name() != BedrockKeyProviderType {
		t.Fatalf("Name = %q", p.Name())
	}
	if got := p.GetKey(ProviderVertexAI, "m", nil); got != "" {
		t.Fatalf("GetKey(vertex-ai) = %q, want empty", got)
	}
	if got := p.GetKey(ProviderBedrock, "m", nil); got != "bedrock-api-key-x" {
		t.Fatalf("GetKey(bedrock) = %q", got)
	}

	fail.Store(true)
	resolver := NewCredentialResolver(NewBedrockKeyProvider(source))
	if _, err := resolver.KeyForProvider(ProviderBedrock, "m", nil); err == nil {
		t.Fatal("the chain resolved a Bedrock key with no credentials")
	}
}

// The key, the presigned request and the credentials never reach the log,
// whether minting succeeds or fails.
func TestBedrockKeyProviderNeverLogsTheKey(t *testing.T) {
	core, logs := observer.New(zapcore.DebugLevel)
	restore := zap.ReplaceGlobals(zap.New(core))
	defer restore()

	clock := &fakeClock{now: time.Date(2026, 11, 23, 15, 0, 0, 0, time.UTC)}
	var retrieves atomic.Int32
	var refuse atomic.Bool
	inner := NewBedrockShortTermKeySourceWith(staticAWSCredentials("us-east-1", time.Time{}, &retrieves), clock.Now)
	source := tokenSourceFunc(func(ctx context.Context) (AccessToken, error) {
		if refuse.Load() {
			// A failure whose cause names the credentials must not log them.
			return AccessToken{}, fmt.Errorf("retrieve AWS credentials: %w", errors.New("denied"))
		}
		return inner.Token(ctx)
	})
	p := NewBedrockKeyProvider(source, WithCloudTokenClock(clock.Now))

	key := p.GetKey(ProviderBedrock, "m", nil)
	if key == "" {
		t.Fatal("no key")
	}
	refuse.Store(true)
	clock.Advance(12 * time.Hour)
	if got := p.GetKey(ProviderBedrock, "m", nil); got != "" {
		t.Fatalf("an expired key was served after a failed refresh")
	}
	if logs.Len() == 0 {
		t.Fatal("the failed refresh was not logged")
	}
	signature := decodeBedrockKey(t, key).Query().Get("X-Amz-Signature")
	for _, entry := range logs.All() {
		text := entry.Message + fmt.Sprint(entry.ContextMap())
		for _, secret := range []string{key, strings.TrimPrefix(key, "bedrock-api-key-"), signature, testSecretAccessKey, testSessionToken, "X-Amz-"} {
			if strings.Contains(text, secret) {
				t.Fatalf("log entry %q carries a secret", entry.Message)
			}
		}
	}
}

// A key is replaced after 11/12 of its life, not at the provider's default
// five-minute margin.
func TestBedrockKeyProviderRefreshesAtElevenHours(t *testing.T) {
	clock := &fakeClock{now: time.Date(2026, 11, 23, 15, 0, 0, 0, time.UTC)}
	source := NewBedrockShortTermKeySourceWith(staticAWSCredentials("us-east-1", time.Time{}, nil), clock.Now)
	var calls atomic.Int32
	counting := tokenSourceFunc(func(ctx context.Context) (AccessToken, error) {
		calls.Add(1)
		return source.Token(ctx)
	})
	p := NewBedrockKeyProvider(counting, WithCloudTokenClock(clock.Now))

	first := p.GetKey(ProviderBedrock, "m", nil)
	clock.Advance(10*time.Hour + 59*time.Minute)
	if got := p.GetKey(ProviderBedrock, "m", nil); got != first || calls.Load() != 1 {
		t.Fatalf("refreshed before 11h (calls %d)", calls.Load())
	}
	clock.Advance(2 * time.Minute)
	if got := p.GetKey(ProviderBedrock, "m", nil); got != first {
		t.Fatal("inside the refresh window the cached key is still served while one refresh runs")
	}
	waitFor(t, func() bool { return p.GetKey(ProviderBedrock, "m", nil) != first })
	if calls.Load() != 2 {
		t.Fatalf("calls = %d, want 2", calls.Load())
	}
}

type tokenSourceFunc func(ctx context.Context) (AccessToken, error)

func (f tokenSourceFunc) Token(ctx context.Context) (AccessToken, error) { return f(ctx) }
