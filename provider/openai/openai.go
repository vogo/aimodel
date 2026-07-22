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

// Package openai implements the OpenAI-compatible chat provider. The canonical
// types are already the OpenAI wire shape, so this path serializes the request
// directly with no translation layer. Importing this package registers the
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

	"github.com/vogo/aimodel/core"
)

// Name is the registered provider name and the default provider selected when
// a client names none.
const Name = "openai"

func init() {
	core.Register(Name, New)
}

// New constructs an OpenAI-compatible provider. It requires a non-empty base
// URL (OpenAI-compatible endpoints have no universal default) and accepts no
// vendor options.
func New(cfg core.Config) (core.ChatProvider, error) {
	if cfg.BaseURL == "" {
		return nil, core.ErrNoBaseURL
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

// NewChatRequest serializes the canonical request as the OpenAI wire body and
// builds the HTTP request. On a streaming call it defaults stream_options to
// include usage in the final chunk, unless the caller already set it.
func (p *provider) NewChatRequest(ctx context.Context, req *core.ChatRequest) (*http.Request, error) {
	native := ChatCompletionRequest(*req)
	if native.Stream && native.StreamOptions == nil {
		native.StreamOptions = &core.StreamOptions{IncludeUsage: true}
	}

	body, err := json.Marshal(&native)
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
func (p *provider) ParseChatResponse(body io.Reader) (*core.ChatResponse, error) {
	var result ChatCompletionResponse
	if err := json.NewDecoder(body).Decode(&result); err != nil {
		return nil, fmt.Errorf("aimodel: decode response: %w", err)
	}

	if result.Error != nil {
		return nil, &core.APIError{
			Code:    result.Error.Code,
			Message: result.Error.Message,
			Type:    result.Error.Type,
		}
	}

	if len(result.Choices) == 0 {
		return nil, core.ErrEmptyResponse
	}

	canonical := core.ChatResponse(result)
	return &canonical, nil
}

// ParseErrorResponse maps a non-2xx OpenAI response body to an APIError,
// falling back to the raw body when it carries no recognizable error object.
func (p *provider) ParseErrorResponse(statusCode int, body []byte) error {
	var errResp struct {
		Error *core.Error `json:"error"`
	}

	if err := json.Unmarshal(body, &errResp); err != nil || errResp.Error == nil {
		return &core.APIError{
			StatusCode: statusCode,
			Message:    string(body),
		}
	}

	return &core.APIError{
		StatusCode: statusCode,
		Code:       errResp.Error.Code,
		Message:    errResp.Error.Message,
		Type:       errResp.Error.Type,
	}
}
