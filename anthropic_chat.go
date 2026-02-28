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

package aimodel

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// AnthropicChatCompletion sends a non-streaming request to the Anthropic Messages API.
func (c *Client) AnthropicChatCompletion(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	r := *req
	r.Stream = false

	ar, err := toAnthropicRequest(&r)
	if err != nil {
		return nil, err
	}

	body, err := json.Marshal(ar)
	if err != nil {
		return nil, fmt.Errorf("aimodel: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.anthropicBaseURL()+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("aimodel: create request: %w", err)
	}

	c.setAnthropicHeaders(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("aimodel: send request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, parseAnthropicErrorResponse(resp)
	}

	var result anthropicResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("aimodel: decode response: %w", err)
	}

	cr := fromAnthropicResponse(&result)
	if len(cr.Choices) == 0 {
		return nil, ErrEmptyResponse
	}

	return cr, nil
}

// AnthropicChatCompletionStream sends a streaming request to the Anthropic Messages API.
func (c *Client) AnthropicChatCompletionStream(ctx context.Context, req *ChatRequest) (*Stream, error) {
	r := *req
	r.Stream = true

	ar, err := toAnthropicRequest(&r)
	if err != nil {
		return nil, err
	}

	body, err := json.Marshal(ar)
	if err != nil {
		return nil, fmt.Errorf("aimodel: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.anthropicBaseURL()+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("aimodel: create request: %w", err)
	}

	c.setAnthropicHeaders(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("aimodel: send request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		defer func() { _ = resp.Body.Close() }()
		return nil, parseAnthropicErrorResponse(resp)
	}

	return newAnthropicStream(resp.Body), nil
}

func (c *Client) setAnthropicHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", c.apiKey)
	req.Header.Set("anthropic-version", anthropicAPIVersion)
}

func (c *Client) anthropicBaseURL() string {
	if c.baseURL != "" {
		return c.baseURL
	}

	return anthropicDefaultBaseURL
}

func parseAnthropicErrorResponse(resp *http.Response) error {
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxErrorBodySize))
	if err != nil {
		return &APIError{
			StatusCode: resp.StatusCode,
			Message:    "failed to read error response",
			Err:        err,
		}
	}

	var errResp anthropicErrorResponse
	if err := json.Unmarshal(body, &errResp); err != nil || errResp.Error.Message == "" {
		return &APIError{
			StatusCode: resp.StatusCode,
			Message:    string(body),
		}
	}

	return &APIError{
		StatusCode: resp.StatusCode,
		Type:       errResp.Error.Type,
		Message:    errResp.Error.Message,
	}
}
