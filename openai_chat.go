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

func (c *Client) doRequest(ctx context.Context, req *ChatRequest) (*http.Response, error) {
	if c.baseURL == "" {
		return nil, ErrNoBaseURL
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("aimodel: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("aimodel: create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("aimodel: send request: %w", err)
	}

	return resp, nil
}

func (c *Client) parseErrorResponse(resp *http.Response) error {
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxErrorBodySize))
	if err != nil {
		return &APIError{
			StatusCode: resp.StatusCode,
			Message:    "failed to read error response",
			Err:        err,
		}
	}

	var errResp struct {
		Error *Error `json:"error"`
	}

	if err := json.Unmarshal(body, &errResp); err != nil || errResp.Error == nil {
		return &APIError{
			StatusCode: resp.StatusCode,
			Message:    string(body),
		}
	}

	return &APIError{
		StatusCode: resp.StatusCode,
		Code:       errResp.Error.Code,
		Message:    errResp.Error.Message,
		Type:       errResp.Error.Type,
	}
}
