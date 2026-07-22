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
package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
)

const maxNativeBodySize = 1 << 20

type Client struct {
	apiKey, baseURL string
	httpClient      *http.Client
}
type ClientOption func(*Client)

func WithBaseURL(url string) ClientOption {
	return func(c *Client) { c.baseURL = strings.TrimRight(url, "/") }
}

func WithHTTPClient(client *http.Client) ClientOption {
	if client == nil {
		panic("aimodel/openai: nil HTTP client")
	}
	return func(c *Client) { c.httpClient = client }
}

func NewClient(apiKey string, options ...ClientOption) *Client {
	c := &Client{apiKey: apiKey, baseURL: "https://api.openai.com/v1", httpClient: http.DefaultClient}
	for _, option := range options {
		option(c)
	}
	return c
}

type HTTPError struct {
	StatusCode          int
	Code, Type, Message string
	Body                json.RawMessage
	Err                 error
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("openai: HTTP %d: %s", e.StatusCode, e.Message)
}
func (e *HTTPError) Unwrap() error { return e.Err }

func (c *Client) request(ctx context.Context, input *ChatCompletionRequest, stream bool) (*http.Response, error) {
	if input == nil {
		return nil, fmt.Errorf("openai: nil chat completions request")
	}
	body, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal chat completions request: %w", err)
	}
	var request ChatCompletionRequest
	if err = json.Unmarshal(body, &request); err != nil {
		return nil, fmt.Errorf("openai: copy chat completions request: %w", err)
	}
	request.Stream = stream
	body, err = json.Marshal(&request)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal chat completions request: %w", err)
	}
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: create chat completions request: %w", err)
	}
	httpRequest.Header.Set("Content-Type", "application/json")
	httpRequest.Header.Set("Authorization", "Bearer "+c.apiKey)
	response, err := c.httpClient.Do(httpRequest)
	if err != nil {
		return nil, fmt.Errorf("openai: send chat completions request: %w", err)
	}
	return response, nil
}

func parseNativeError(response *http.Response) error {
	body, err := io.ReadAll(io.LimitReader(response.Body, maxNativeBodySize))
	result := &HTTPError{StatusCode: response.StatusCode, Body: append(json.RawMessage(nil), body...), Message: string(body), Err: err}
	if err != nil {
		result.Message = "failed to read error response"
		return result
	}
	var wire struct {
		Error *Error `json:"error"`
	}
	if json.Unmarshal(body, &wire) == nil && wire.Error != nil {
		result.Code, result.Type, result.Message = wire.Error.Code, wire.Error.Type, wire.Error.Message
	}
	return result
}

func (c *Client) ChatCompletions(ctx context.Context, request *ChatCompletionRequest) (*ChatCompletionResponse, error) {
	response, err := c.request(ctx, request, false)
	if err != nil {
		return nil, err
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, parseNativeError(response)
	}
	var result ChatCompletionResponse
	if err = json.NewDecoder(response.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("openai: decode chat completions response: %w", err)
	}
	if result.Error != nil {
		return nil, &HTTPError{StatusCode: response.StatusCode, Code: result.Error.Code, Type: result.Error.Type, Message: result.Error.Message}
	}
	return &result, nil
}

type ChatCompletionStream struct {
	body io.ReadCloser
	scan *bufio.Scanner
	once sync.Once
}

func (c *Client) ChatCompletionsStream(ctx context.Context, request *ChatCompletionRequest) (*ChatCompletionStream, error) {
	response, err := c.request(ctx, request, true)
	if err != nil {
		return nil, err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		defer func() { _ = response.Body.Close() }()
		return nil, parseNativeError(response)
	}
	scan := bufio.NewScanner(response.Body)
	scan.Buffer(make([]byte, 0, 64<<10), maxNativeBodySize)
	return &ChatCompletionStream{body: response.Body, scan: scan}, nil
}

func (s *ChatCompletionStream) Recv() (*ChatCompletionChunk, error) {
	for s.scan.Scan() {
		line := s.scan.Text()
		if line == "" || strings.HasPrefix(line, ":") || !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			_ = s.Close()
			return nil, io.EOF
		}
		var chunk ChatCompletionChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			_ = s.Close()
			return nil, fmt.Errorf("openai: decode stream chunk: %w", err)
		}
		if chunk.Error != nil {
			_ = s.Close()
			return nil, &HTTPError{Code: chunk.Error.Code, Type: chunk.Error.Type, Message: chunk.Error.Message}
		}
		return &chunk, nil
	}
	if err := s.scan.Err(); err != nil {
		_ = s.Close()
		return nil, fmt.Errorf("openai: read stream: %w", err)
	}
	_ = s.Close()
	return nil, io.EOF
}

func (s *ChatCompletionStream) Close() error {
	var err error
	s.once.Do(func() { err = s.body.Close() })
	return err
}
