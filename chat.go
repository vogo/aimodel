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
	"fmt"
	"io"
	"net/http"

	"github.com/vogo/aimodel/ais"
)

// ChatCompleter is the chat capability contract. Capabilities are defined as
// small per-interaction-form interfaces so a new interaction form is added by
// introducing a new capability interface (and a matching client method),
// never by changing this one. Both *Client and composes.ComposeClient
// implement it.
type ChatCompleter interface {
	ChatCompletion(ctx context.Context, req *ais.ChatRequest) (*ais.ChatResponse, error)
	ChatCompletionStream(ctx context.Context, req *ais.ChatRequest) (*Stream, error)
}

// Compile-time check: *Client implements ChatCompleter.
var _ ChatCompleter = (*Client)(nil)

// maxErrorBodySize limits the error response body read to 1 MB.
const maxErrorBodySize = 1 << 20

// applyDefaultModel sets the request model to the client default if empty.
func (c *Client) applyDefaultModel(r *ais.ChatRequest) {
	if r.Model == "" {
		r.Model = c.model
	}
}

// ChatCompletion sends a non-streaming chat completion request, delegating the
// protocol-specific work to the client's resolved provider.
func (c *Client) ChatCompletion(ctx context.Context, req *ais.ChatRequest) (*ais.ChatResponse, error) {
	r := req.Clone()
	r.Stream = false

	c.applyDefaultModel(&r)

	resp, err := c.send(ctx, &r)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	if !isSuccess(resp.StatusCode) {
		return nil, c.parseError(resp)
	}

	return c.provider.ParseChatResponse(resp.Body)
}

// ChatCompletionStream sends a streaming chat completion request and returns a
// Stream backed by the provider's SSE decoder.
func (c *Client) ChatCompletionStream(ctx context.Context, req *ais.ChatRequest) (*Stream, error) {
	r := req.Clone()
	r.Stream = true

	c.applyDefaultModel(&r)

	resp, err := c.send(ctx, &r)
	if err != nil {
		return nil, err
	}

	if !isSuccess(resp.StatusCode) {
		defer func() { _ = resp.Body.Close() }()
		return nil, c.parseError(resp)
	}

	return newStream(resp.Body, c.provider.NewStreamDecoder(resp.Body)), nil
}

// send builds the provider request and issues the single HTTP call. On any
// build or transport failure the caller receives an error and no response body
// to close.
func (c *Client) send(ctx context.Context, r *ais.ChatRequest) (*http.Response, error) {
	httpReq, err := c.provider.NewChatRequest(ctx, r)
	if err != nil {
		return nil, err
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("aimodel: send request: %w", err)
	}

	return resp, nil
}

// parseError reads the error body under the shared size limit and hands it to
// the provider for conversion into the canonical error model.
func (c *Client) parseError(resp *http.Response) error {
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxErrorBodySize))
	if err != nil {
		return &ais.APIError{
			StatusCode: resp.StatusCode,
			Message:    "failed to read error response",
			Err:        err,
		}
	}

	return c.provider.ParseErrorResponse(resp.StatusCode, body)
}

// isSuccess reports whether an HTTP status is 2xx.
func isSuccess(statusCode int) bool {
	return statusCode >= http.StatusOK && statusCode < http.StatusMultipleChoices
}
