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

package anthropic

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

// Client calls Anthropic's Messages endpoint without canonical translation.
type Client struct {
	apiKey, baseURL, version, userProfileID string
	beta                                    []string
	httpClient                              *http.Client
}

// ClientOption configures a native Client.
type ClientOption func(*Client)

func WithBaseURL(url string) ClientOption {
	return func(c *Client) { c.baseURL = strings.TrimRight(url, "/") }
}

func WithHTTPClient(h *http.Client) ClientOption {
	if h == nil {
		panic("aimodel/anthropic: nil HTTP client")
	}
	return func(c *Client) { c.httpClient = h }
}

func WithVersion(v string) ClientOption {
	return func(c *Client) {
		if v != "" {
			c.version = v
		}
	}
}

func WithBeta(v ...string) ClientOption {
	return func(c *Client) { c.beta = append([]string(nil), v...) }
}
func WithUserProfileID(v string) ClientOption { return func(c *Client) { c.userProfileID = v } }

// NewClient constructs a native Messages client.
func NewClient(apiKey string, opts ...ClientOption) *Client {
	c := &Client{apiKey: apiKey, baseURL: anthropicDefaultBaseURL, version: anthropicAPIVersion, httpClient: http.DefaultClient}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// HTTPError is a non-success Anthropic response.
type HTTPError struct {
	StatusCode    int
	Type, Message string
	Body          json.RawMessage
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("anthropic: HTTP %d: %s", e.StatusCode, e.Message)
}

func (c *Client) request(ctx context.Context, in *MessageRequest, stream bool) (*http.Response, error) {
	if in == nil {
		return nil, fmt.Errorf("anthropic: nil message request")
	}
	body, err := json.Marshal(in)
	if err != nil {
		return nil, fmt.Errorf("anthropic: marshal request: %w", err)
	}
	var req MessageRequest
	if err = json.Unmarshal(body, &req); err != nil {
		return nil, err
	}
	req.Stream = stream
	body, err = json.Marshal(&req)
	if err != nil {
		return nil, err
	}
	h, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	h.Header.Set("Content-Type", "application/json")
	h.Header.Set("x-api-key", c.apiKey)
	h.Header.Set("anthropic-version", c.version)
	if b := strings.Join(c.beta, ","); b != "" {
		h.Header.Set("anthropic-beta", b)
	}
	if c.userProfileID != "" {
		h.Header.Set("anthropic-user-profile-id", c.userProfileID)
	}
	return c.httpClient.Do(h)
}

func parseHTTPError(resp *http.Response) error {
	b, readErr := io.ReadAll(io.LimitReader(resp.Body, maxNativeBodySize))
	if readErr != nil {
		return &HTTPError{StatusCode: resp.StatusCode, Message: readErr.Error()}
	}
	e := &HTTPError{StatusCode: resp.StatusCode, Body: append(json.RawMessage(nil), b...), Message: string(b)}
	var er ErrorResponse
	if json.Unmarshal(b, &er) == nil && er.Error.Message != "" {
		e.Type, e.Message = er.Error.Type, er.Error.Message
	}
	return e
}

// Messages performs a non-streaming native Messages call.
func (c *Client) Messages(ctx context.Context, req *MessageRequest) (*MessageResponse, error) {
	resp, err := c.request(ctx, req, false)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, parseHTTPError(resp)
	}
	var out MessageResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("anthropic: decode message: %w", err)
	}
	return &out, nil
}

// StreamEvent is one native Anthropic SSE event. Raw preserves its complete JSON payload.
type StreamEvent struct {
	Type         string           `json:"type"`
	Index        *int             `json:"index,omitempty"`
	Message      *MessageResponse `json:"message,omitempty"`
	ContentBlock *ContentBlock    `json:"content_block,omitempty"`
	Delta        json.RawMessage  `json:"delta,omitempty"`
	Usage        *MessageUsage    `json:"usage,omitempty"`
	Error        *ErrorDetail     `json:"error,omitempty"`
	Raw          json.RawMessage  `json:"-"`
}

// MessageStream reads native events without aggregating them.
type MessageStream struct {
	body io.ReadCloser
	scan *bufio.Scanner
	once sync.Once
}

// MessagesStream starts a native streaming Messages call.
func (c *Client) MessagesStream(ctx context.Context, req *MessageRequest) (*MessageStream, error) {
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
	return &MessageStream{body: resp.Body, scan: s}, nil
}

// Recv returns the next SSE event in wire order.
func (s *MessageStream) Recv() (*StreamEvent, error) {
	var data []string
	for s.scan.Scan() {
		line := s.scan.Text()
		if line == "" {
			if len(data) == 0 {
				continue
			}
			return decodeEvent(strings.Join(data, "\n"))
		}
		if after, ok := strings.CutPrefix(line, "data:"); ok {
			data = append(data, strings.TrimPrefix(after, " "))
		}
	}
	if err := s.scan.Err(); err != nil {
		_ = s.Close()
		return nil, fmt.Errorf("anthropic: read stream: %w", err)
	}
	if len(data) != 0 {
		return decodeEvent(strings.Join(data, "\n"))
	}
	_ = s.Close()
	return nil, io.EOF
}

func decodeEvent(data string) (*StreamEvent, error) {
	var e StreamEvent
	if err := json.Unmarshal([]byte(data), &e); err != nil {
		return nil, fmt.Errorf("anthropic: decode stream event: %w", err)
	}
	e.Raw = append(json.RawMessage(nil), data...)
	return &e, nil
}

// Close releases the response body and is safe to call repeatedly.
func (s *MessageStream) Close() error {
	var err error
	s.once.Do(func() { err = s.body.Close() })
	return err
}
