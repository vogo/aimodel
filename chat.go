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

// ChatCompletion sends a non-streaming chat completion request.
func (c *Client) ChatCompletion(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	r := *req
	r.Stream = false

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

// ChatCompletionStream sends a streaming chat completion request
// and returns a Stream for reading chunks.
func (c *Client) ChatCompletionStream(ctx context.Context, req *ChatRequest) (*Stream, error) {
	r := *req
	r.Stream = true

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
