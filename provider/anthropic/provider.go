/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Package anthropic implements the Anthropic Messages API chat provider. It
// owns the full native wire layer (request/response types, translation to and
// from the canonical schema, SSE event parsing) so those vendor concepts never
// leak into the root package. Importing this package registers the provider
// under Name.
//
// Anthropic Messages API reference: https://platform.claude.com/docs/en/api/messages
package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/vogo/aimodel/ais"
)

// Name is the registered provider name. Select it via the root package's
// WithProvider(anthropic.Name).
const Name = "anthropic"

func init() {
	ais.Register(Name, New)
}

// Options carries Anthropic-specific configuration. Pass it to the root
// package's client through WithProviderOptions; leaving it unset selects the
// documented defaults (API version 2023-06-01, no beta features, no profile).
type Options struct {
	// Beta enables one or more Anthropic beta features via the
	// "anthropic-beta" request header. Empty strings are ignored; on the wire
	// the values are joined with commas. Unset omits the header.
	Beta []string

	// Version overrides the "anthropic-version" request header. Empty keeps
	// the default (anthropicAPIVersion).
	Version string

	// UserProfileID sets the "anthropic-user-profile-id" request header,
	// associating requests with an end-user profile. Empty leaves it unset.
	UserProfileID string
}

// New constructs an Anthropic provider. The base URL is optional (it defaults
// to the public endpoint). cfg.Options, when set, must be Options.
func New(cfg ais.Config) (ais.ChatProvider, error) {
	p := &provider{
		apiKey:  cfg.APIKey,
		baseURL: cfg.BaseURL,
		version: anthropicAPIVersion,
	}

	switch o := cfg.Options.(type) {
	case nil:
	case Options:
		for _, v := range o.Beta {
			if v != "" {
				p.beta = append(p.beta, v)
			}
		}

		if o.Version != "" {
			p.version = o.Version
		}

		p.userProfileID = o.UserProfileID
	default:
		return nil, fmt.Errorf("aimodel/anthropic: unexpected provider options of type %T", cfg.Options)
	}

	return p, nil
}

type provider struct {
	apiKey        string
	baseURL       string
	beta          []string
	version       string
	userProfileID string
}

func (p *provider) endpoint() string {
	base := p.baseURL
	if base == "" {
		base = anthropicDefaultBaseURL
	}

	return base + "/v1/messages"
}

// NewChatRequest translates the canonical request into the Anthropic wire
// shape and builds the HTTP request with the Anthropic auth/version headers.
func (p *provider) NewChatRequest(ctx context.Context, req *ais.ChatRequest) (*http.Request, error) {
	ar, err := toAnthropicRequest(req)
	if err != nil {
		return nil, err
	}

	body, err := json.Marshal(ar)
	if err != nil {
		return nil, fmt.Errorf("aimodel: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.endpoint(), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("aimodel: create request: %w", err)
	}

	p.setHeaders(httpReq)

	return httpReq, nil
}

func (p *provider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.apiKey)
	req.Header.Set("anthropic-version", p.version)

	if beta := strings.Join(p.beta, ","); beta != "" {
		req.Header.Set("anthropic-beta", beta)
	}

	if p.userProfileID != "" {
		req.Header.Set("anthropic-user-profile-id", p.userProfileID)
	}
}

// ParseChatResponse decodes an Anthropic message response and converts it to
// the canonical ChatResponse.
func (p *provider) ParseChatResponse(body io.Reader) (*ais.ChatResponse, error) {
	var result MessagesResponse
	if err := json.NewDecoder(body).Decode(&result); err != nil {
		return nil, fmt.Errorf("aimodel: decode response: %w", err)
	}

	cr := fromAnthropicResponse(&result)
	if len(cr.Choices) == 0 {
		return nil, ais.ErrEmptyResponse
	}

	return cr, nil
}

// ParseErrorResponse maps a non-2xx Anthropic response body to an APIError,
// falling back to the raw body when it carries no recognizable error object.
func (p *provider) ParseErrorResponse(statusCode int, body []byte) error {
	var errResp MessagesErrorResponse
	if err := json.Unmarshal(body, &errResp); err != nil || errResp.Error.Message == "" {
		return &ais.APIError{
			StatusCode: statusCode,
			Message:    string(body),
		}
	}

	return &ais.APIError{
		StatusCode: statusCode,
		Type:       errResp.Error.Type,
		Message:    errResp.Error.Message,
	}
}
