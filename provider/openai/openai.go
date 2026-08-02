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

// Package openai implements the OpenAI-compatible chat provider. Importing this package registers the
// provider under Name; the root aimodel package imports it by default.
//
// OpenAI reference: https://platform.openai.com/docs/api-reference/chat
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/vogo/aimodel/ais"
)

// Name is the registered provider name and the default provider selected when
// a client names none.
//
// Deprecated: the canonical layer is removed in v0.6.0. Use the native
// client and wire types of provider/openai or provider/anthropic instead;
// see MIGRATION.md for the symbol-by-symbol migration table.
const Name = "openai"

func init() {
	ais.Register(Name, New)
}

// New constructs an OpenAI-compatible provider. It requires a non-empty base
// URL (OpenAI-compatible endpoints have no universal default) and accepts no
// vendor options.
//
// Deprecated: the canonical layer is removed in v0.6.0. Use the native
// client and wire types of provider/openai or provider/anthropic instead;
// see MIGRATION.md for the symbol-by-symbol migration table.
func New(cfg ais.Config) (ais.ChatProvider, error) {
	if cfg.BaseURL == "" {
		return nil, ais.ErrNoBaseURL
	}

	if cfg.Options != nil {
		return nil, fmt.Errorf("aimodel/openai: unexpected provider options of type %T", cfg.Options)
	}

	return &provider{
		apiKey:  cfg.APIKey,
		baseURL: cfg.BaseURL,
	}, nil
}

type provider struct {
	apiKey  string
	baseURL string
}

// NativeResponsesClient exposes this provider's credentials and base URL as a
// native client, so the unified client can delegate the Responses capability
// without duplicating transport configuration. It is the method set the root
// package type-asserts to decide whether the resolved provider supports
// Responses; providers without it report the capability as unsupported.
//
// Responses has no canonical form (see doc/adr/0006), so the returned client
// speaks native types end to end.
func (p *provider) NativeResponsesClient(httpClient *http.Client) *Client {
	return NewClient(p.apiKey, WithBaseURL(p.baseURL), WithHTTPClient(httpClient))
}

// NewChatRequest translates shared fields into the OpenAI wire body.
func (p *provider) NewChatRequest(ctx context.Context, req *ais.ChatRequest) (*http.Request, error) {
	body, err := json.Marshal(toOpenAIRequest(req))
	if err != nil {
		return nil, fmt.Errorf("aimodel: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("aimodel: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)

	return httpReq, nil
}

// ParseChatResponse decodes an OpenAI-shape completion. A body-level error
// object becomes an APIError; a response with no choices is ErrEmptyResponse.
func (p *provider) ParseChatResponse(body io.Reader) (*ais.ChatResponse, error) {
	var result ChatCompletionResponse
	if err := json.NewDecoder(body).Decode(&result); err != nil {
		return nil, fmt.Errorf("aimodel: decode response: %w", err)
	}

	if result.Error != nil {
		return nil, &ais.APIError{
			Code:    result.Error.Code,
			Message: result.Error.Message,
			Type:    result.Error.Type,
		}
	}

	if len(result.Choices) == 0 {
		return nil, ais.ErrEmptyResponse
	}

	return fromOpenAIResponse(&result), nil
}

// ParseErrorResponse maps a non-2xx OpenAI response body to an APIError,
// falling back to the raw body when it carries no recognizable error object.
func (p *provider) ParseErrorResponse(statusCode int, body []byte) error {
	var errResp struct {
		Error *Error `json:"error"`
	}

	if err := json.Unmarshal(body, &errResp); err != nil || errResp.Error == nil {
		return &ais.APIError{
			StatusCode: statusCode,
			Message:    string(body),
		}
	}

	return &ais.APIError{
		StatusCode: statusCode,
		Code:       errResp.Error.Code,
		Message:    errResp.Error.Message,
		Type:       errResp.Error.Type,
	}
}
