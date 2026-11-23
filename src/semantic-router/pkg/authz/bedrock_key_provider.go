package authz

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	v4 "github.com/aws/aws-sdk-go-v2/aws/signer/v4"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
)

// BedrockKeyProviderType is the authz.providers[].type that selects the
// Amazon Bedrock short-term API key provider.
const BedrockKeyProviderType = "bedrock-short-term-key"

const (
	// bedrockKeyHost is the host a short-term key's request is signed for. It
	// is the same in every Region; the key is valid only in the Region it is
	// signed in.
	bedrockKeyHost = "bedrock.amazonaws.com"
	// bedrockKeyService is the SigV4 service name.
	bedrockKeyService = "bedrock"
	// bedrockKeyPrefix and bedrockKeyVersion frame the encoded request.
	bedrockKeyPrefix  = "bedrock-api-key-"
	bedrockKeyVersion = "&Version=1"
	// bedrockKeyMaxLifetime is the longest a short-term key is valid. A key
	// also stops working when the session credentials that signed it expire.
	bedrockKeyMaxLifetime = 12 * time.Hour
	// bedrockKeyRefreshFraction of a key's life passes before it is replaced:
	// a 12-hour key is refreshed after 11 hours.
	bedrockKeyRefreshFraction = 11.0 / 12.0
	// bedrockKeyMinRefresh bounds how often keys are minted when the signing
	// credentials are about to expire and every new key is short-lived.
	bedrockKeyMinRefresh = time.Minute
	// bedrockCredentialsRenewBefore: session credentials with less life than
	// this left are renewed before a key is signed with them, so a key is not
	// cut short by credentials the SDK would otherwise keep serving until they
	// expire.
	bedrockCredentialsRenewBefore = 10 * time.Minute
	// bedrockKeyMinValidity: a key this close to expiry is no longer served.
	bedrockKeyMinValidity = 30 * time.Second
	// bedrockCredentialsRenewCooldown: credentials are invalidated and
	// fetched again at most this often. A key minted from near-expiry
	// credentials is unusable at once, so without it every request would
	// ask the credentials endpoint again until fresh credentials arrive.
	bedrockCredentialsRenewCooldown = 30 * time.Second
	// emptyPayloadSHA256 is the SHA-256 of an empty body: the request has
	// none, and AWS's token generators sign its hash (not UNSIGNED-PAYLOAD).
	emptyPayloadSHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)

// NewBedrockKeyProvider serves Amazon Bedrock short-term API keys from
// source, as Bearer credentials for the bedrock provider. Keys are cached and
// replaced at their RefreshAt (CloudTokenProvider): there is no fixed
// refresh-ahead margin, since a key's life follows its signing credentials'
// and a fixed margin would re-mint on every request once that life is short.
func NewBedrockKeyProvider(source TokenSource, opts ...CloudTokenOption) *CloudTokenProvider {
	base := []CloudTokenOption{
		WithCloudTokenProviders(ProviderBedrock),
		WithCloudTokenRefreshAhead(0, bedrockKeyMinValidity),
		func(p *CloudTokenProvider) { p.name = BedrockKeyProviderType },
	}
	return NewCloudTokenProvider(source, append(base, opts...)...)
}

// CredentialsLoader returns the AWS credentials provider and Region a key is
// signed with.
type CredentialsLoader func(ctx context.Context) (aws.CredentialsProvider, string, error)

// BedrockShortTermKeySource mints Amazon Bedrock short-term API keys: a SigV4
// presigned CallWithBearerToken request, signed with the workload's AWS
// credentials, encoded as the key. On EKS the credentials come from Pod
// Identity (or IRSA) through the SDK's default chain, so no key is stored
// anywhere. Neither the presigned request nor the key is ever logged or
// returned in an error.
type BedrockShortTermKeySource struct {
	load   CredentialsLoader
	now    func() time.Time
	signer *v4.Signer

	credentials aws.CredentialsProvider
	region      string
	// renewedAt is when near-expiry credentials were last renewed.
	renewedAt time.Time
	// held is the last credentials fetched; retryAt, after a failed fetch,
	// is when credentials are fetched again. Until then held is used (or,
	// with none, Token fails), so a credentials endpoint that is down is
	// asked at most once per cooldown.
	held    aws.Credentials
	retryAt time.Time
}

// NewBedrockShortTermKeySource returns a source that signs with the SDK's
// default credential chain and Region (AWS_REGION). The chain is resolved on
// first use, so a router without AWS credentials starts and fails closed.
func NewBedrockShortTermKeySource() *BedrockShortTermKeySource {
	return NewBedrockShortTermKeySourceWith(defaultAWSCredentials, time.Now)
}

// NewBedrockShortTermKeySourceWith returns a source with its credentials and
// clock supplied (tests).
func NewBedrockShortTermKeySourceWith(load CredentialsLoader, now func() time.Time) *BedrockShortTermKeySource {
	return &BedrockShortTermKeySource{load: load, now: now, signer: v4.NewSigner()}
}

func defaultAWSCredentials(ctx context.Context) (aws.CredentialsProvider, string, error) {
	// No credentials cache expiry window: a window shifts the Expires the
	// cache reports, which would cut every key short (or, for a session no
	// longer than the window, make fresh credentials look expired). The
	// container (Pod Identity) provider sets five minutes unless told
	// otherwise. Near-expiry credentials are renewed in Token instead.
	cfg, err := awsconfig.LoadDefaultConfig(ctx, awsconfig.WithCredentialsCacheOptions(func(options *aws.CredentialsCacheOptions) {
		options.ExpiryWindow = 0
		options.ExpiryWindowJitterFrac = 0
	}))
	if err != nil {
		return nil, "", fmt.Errorf("load AWS configuration: %w", err)
	}
	return cfg.Credentials, cfg.Region, nil
}

// Token mints one short-term key.
func (s *BedrockShortTermKeySource) Token(ctx context.Context) (AccessToken, error) {
	if s.credentials == nil {
		credentials, region, err := s.load(ctx)
		if err != nil {
			return AccessToken{}, err
		}
		if credentials == nil {
			return AccessToken{}, errors.New("no AWS credentials provider")
		}
		if strings.TrimSpace(region) == "" {
			return AccessToken{}, errors.New("no AWS Region (set AWS_REGION to the Bedrock Region)")
		}
		s.credentials, s.region = credentials, strings.TrimSpace(region)
	}
	issued := s.now()
	backoff := issued.Before(s.retryAt)
	creds := s.held
	if !backoff {
		fetched, err := s.retrieve(ctx)
		if err != nil {
			s.retryAt = issued.Add(bedrockCredentialsRenewCooldown)
			return AccessToken{}, err
		}
		creds, s.held = fetched, fetched
	} else if creds.AccessKeyID == "" {
		return AccessToken{}, errors.New("AWS credentials are unavailable; retrying after a cooldown")
	}
	if !backoff && creds.CanExpire && creds.Expires.Sub(issued) < bedrockCredentialsRenewBefore &&
		(s.renewedAt.IsZero() || issued.Sub(s.renewedAt) >= bedrockCredentialsRenewCooldown) {
		// Ask for fresh credentials; keep the old ones if that fails or
		// gives none that last longer.
		if renewer, ok := s.credentials.(interface{ Invalidate() }); ok {
			s.renewedAt = issued
			renewer.Invalidate()
			fresh, err := s.retrieve(ctx)
			switch {
			case err != nil:
				s.retryAt = issued.Add(bedrockCredentialsRenewCooldown)
			case !fresh.CanExpire || fresh.Expires.After(creds.Expires):
				creds, s.held = fresh, fresh
			}
		}
	}
	lifetime := bedrockKeyMaxLifetime
	if creds.CanExpire {
		if remaining := creds.Expires.Sub(issued); remaining < lifetime {
			lifetime = remaining
		}
	}
	if lifetime <= 0 {
		return AccessToken{}, errors.New("AWS credentials have expired")
	}
	key, err := s.sign(ctx, creds, issued)
	if err != nil {
		return AccessToken{}, err
	}
	refresh := time.Duration(float64(lifetime) * bedrockKeyRefreshFraction)
	if refresh < bedrockKeyMinRefresh {
		refresh = bedrockKeyMinRefresh
	}
	return AccessToken{Value: key, Expiry: issued.Add(lifetime), RefreshAt: issued.Add(refresh)}, nil
}

func (s *BedrockShortTermKeySource) retrieve(ctx context.Context) (aws.Credentials, error) {
	creds, err := s.credentials.Retrieve(ctx)
	if err != nil {
		return aws.Credentials{}, fmt.Errorf("retrieve AWS credentials: %w", err)
	}
	if creds.AccessKeyID == "" || creds.SecretAccessKey == "" {
		return aws.Credentials{}, errors.New("AWS credentials are empty")
	}
	return creds, nil
}

// sign presigns POST https://bedrock.amazonaws.com/?Action=CallWithBearerToken
// for 12 hours and encodes it as AWS's token generators do: the URL without
// its scheme, with &Version=1 appended, base64 (standard, padded), prefixed
// with bedrock-api-key-.
func (s *BedrockShortTermKeySource) sign(ctx context.Context, creds aws.Credentials, at time.Time) (string, error) {
	query := url.Values{}
	query.Set("Action", "CallWithBearerToken")
	query.Set("X-Amz-Expires", strconv.Itoa(int(bedrockKeyMaxLifetime.Seconds())))
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://"+bedrockKeyHost+"/?"+query.Encode(), nil)
	if err != nil {
		return "", errors.New("build the short-term key request")
	}
	presigned, _, err := s.signer.PresignHTTP(ctx, creds, request, emptyPayloadSHA256, bedrockKeyService, s.region, at)
	if err != nil {
		// The signer's error is not wrapped: it could quote the request.
		return "", errors.New("sign the short-term key request")
	}
	encoded := base64.StdEncoding.EncodeToString([]byte(strings.TrimPrefix(presigned, "https://") + bedrockKeyVersion))
	return bedrockKeyPrefix + encoded, nil
}
