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

	"github.com/vogo/aimodel/core"
)

const maxNativeBodySize = 1 << 20

// ChatCompletionRequest is the native Chat Completions request at the documented 2026-06-02 baseline.
// It is a distinct public type even where its wire fields coincide with the canonical model.
type ChatCompletionRequest core.ChatRequest

// Native request component aliases let callers stay within the openai package.
type (
	Message       = core.Message
	Content       = core.Content
	ContentPart   = core.ContentPart
	Tool          = core.Tool
	StreamOptions = core.StreamOptions
)

// ChatCompletionResponse is the native non-streaming response.
type ChatCompletionResponse core.ChatResponse

// ChatCompletionChunk is one native streaming payload. Raw preserves unknown fields.
type ChatCompletionChunk struct {
	core.StreamChunk
	Error *core.Error     `json:"error,omitempty"`
	Raw   json.RawMessage `json:"-"`
}

type Client struct {
	apiKey, baseURL string
	httpClient      *http.Client
}
type ClientOption func(*Client)

func WithBaseURL(url string) ClientOption {
	return func(c *Client) { c.baseURL = strings.TrimRight(url, "/") }
}

func WithHTTPClient(h *http.Client) ClientOption {
	if h == nil {
		panic("aimodel/openai: nil HTTP client")
	}
	return func(c *Client) { c.httpClient = h }
}

// NewClient constructs a native Chat Completions client.
func NewClient(apiKey string, opts ...ClientOption) *Client {
	c := &Client{apiKey: apiKey, baseURL: "https://api.openai.com/v1", httpClient: http.DefaultClient}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

type HTTPError struct {
	StatusCode          int
	Code, Type, Message string
	Body                json.RawMessage
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("openai: HTTP %d: %s", e.StatusCode, e.Message)
}

func (c *Client) request(ctx context.Context, in *ChatCompletionRequest, stream bool) (*http.Response, error) {
	if in == nil {
		return nil, fmt.Errorf("openai: nil chat completion request")
	}
	body, err := json.Marshal(in)
	if err != nil {
		return nil, err
	}
	var req ChatCompletionRequest
	if err = json.Unmarshal(body, &req); err != nil {
		return nil, err
	}
	req.Stream = stream
	if stream && req.StreamOptions == nil {
		req.StreamOptions = &core.StreamOptions{IncludeUsage: true}
	}
	body, err = json.Marshal(&req)
	if err != nil {
		return nil, err
	}
	h, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	h.Header.Set("Content-Type", "application/json")
	h.Header.Set("Authorization", "Bearer "+c.apiKey)
	return c.httpClient.Do(h)
}

func parseHTTPError(resp *http.Response) error {
	b, readErr := io.ReadAll(io.LimitReader(resp.Body, maxNativeBodySize))
	if readErr != nil {
		return &HTTPError{StatusCode: resp.StatusCode, Message: readErr.Error()}
	}
	e := &HTTPError{StatusCode: resp.StatusCode, Body: append(json.RawMessage(nil), b...), Message: string(b)}
	var v struct {
		Error *core.Error `json:"error"`
	}
	if json.Unmarshal(b, &v) == nil && v.Error != nil {
		e.Code, e.Type, e.Message = v.Error.Code, v.Error.Type, v.Error.Message
	}
	return e
}

func (c *Client) ChatCompletions(ctx context.Context, req *ChatCompletionRequest) (*ChatCompletionResponse, error) {
	resp, err := c.request(ctx, req, false)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, parseHTTPError(resp)
	}
	var out ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("openai: decode completion: %w", err)
	}
	return &out, nil
}

type ChatCompletionStream struct {
	body io.ReadCloser
	scan *bufio.Scanner
	once sync.Once
}

func (c *Client) ChatCompletionsStream(ctx context.Context, req *ChatCompletionRequest) (*ChatCompletionStream, error) {
	resp, err := c.request(ctx, req, true)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer func() { _ = resp.Body.Close() }()
		return nil, parseHTTPError(resp)
	}
	s := bufio.NewScanner(resp.Body)
	s.Buffer(make([]byte, 0, 64<<10), maxNativeBodySize)
	return &ChatCompletionStream{body: resp.Body, scan: s}, nil
}

func (s *ChatCompletionStream) Recv() (*ChatCompletionChunk, error) {
	var data []string
	for s.scan.Scan() {
		line := s.scan.Text()
		if line == "" {
			if len(data) == 0 {
				continue
			}
			return decodeChunk(strings.Join(data, "\n"))
		}
		if after, ok := strings.CutPrefix(line, "data:"); ok {
			data = append(data, strings.TrimPrefix(after, " "))
		}
	}
	if err := s.scan.Err(); err != nil {
		_ = s.Close()
		return nil, err
	}
	if len(data) != 0 {
		return decodeChunk(strings.Join(data, "\n"))
	}
	_ = s.Close()
	return nil, io.EOF
}

func decodeChunk(data string) (*ChatCompletionChunk, error) {
	if data == "[DONE]" {
		return nil, io.EOF
	}
	var out ChatCompletionChunk
	if err := json.Unmarshal([]byte(data), &out); err != nil {
		return nil, fmt.Errorf("openai: decode stream event: %w", err)
	}
	out.Raw = append(json.RawMessage(nil), data...)
	return &out, nil
}

func (s *ChatCompletionStream) Close() error {
	var err error
	s.once.Do(func() { err = s.body.Close() })
	return err
}
