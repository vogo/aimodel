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
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

// maxErrorBodySize limits the error response body read to 1 MB.
const maxErrorBodySize = 1 << 20

// applyDefaultModel sets the request model to the client default if empty.
func (c *Client) applyDefaultModel(r *ChatRequest) {
	if r.Model == "" {
		r.Model = c.model
	}
}

// ChatCompletion sends a non-streaming chat completion request.
// The client's protocol setting determines which API is used.
func (c *Client) ChatCompletion(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	switch c.protocol {
	case ProtocolAnthropic:
		return c.anthropicChatCompletion(ctx, req)
	default:
		return c.openaiChatCompletion(ctx, req)
	}
}

// ChatCompletionStream sends a streaming chat completion request
// and returns a Stream for reading chunks.
// The client's protocol setting determines which API is used.
func (c *Client) ChatCompletionStream(ctx context.Context, req *ChatRequest) (*Stream, error) {
	switch c.protocol {
	case ProtocolAnthropic:
		return c.anthropicChatCompletionStream(ctx, req)
	default:
		return c.openaiChatCompletionStream(ctx, req)
	}
}

// openaiChatCompletion sends a non-streaming request using the OpenAI-compatible API.
func (c *Client) openaiChatCompletion(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	r := req.clone()
	r.Stream = false

	c.applyDefaultModel(&r)

	resp, err := c.doRequest(ctx, &r)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		return nil, c.parseErrorResponse(resp)
	}

	var result ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("aimodel: decode response: %w", err)
	}

	if result.Error != nil {
		return nil, &APIError{
			StatusCode: resp.StatusCode,
			Code:       result.Error.Code,
			Message:    result.Error.Message,
			Type:       result.Error.Type,
		}
	}

	if len(result.Choices) == 0 {
		return nil, ErrEmptyResponse
	}

	return &result, nil
}

// openaiChatCompletionStream sends a streaming request using the OpenAI-compatible API.
func (c *Client) openaiChatCompletionStream(ctx context.Context, req *ChatRequest) (*Stream, error) {
	r := req.clone()
	r.Stream = true

	c.applyDefaultModel(&r)

	resp, err := c.doRequest(ctx, &r)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		defer func() { _ = resp.Body.Close() }()
		return nil, c.parseErrorResponse(resp)
	}

	return newStream(resp.Body), nil
}
