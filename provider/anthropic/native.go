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

// Client calls Anthropic's Messages API without canonical translation.
type Client struct {
	apiKey, baseURL, version, userProfileID string
	beta                                    []string
	httpClient                              *http.Client
}

// ClientOption configures a native Client.
type ClientOption func(*Client)

// WithBaseURL overrides the Anthropic API base URL.
func WithBaseURL(url string) ClientOption {
	return func(c *Client) { c.baseURL = strings.TrimRight(url, "/") }
}

// WithHTTPClient supplies the HTTP transport used by the client.
func WithHTTPClient(client *http.Client) ClientOption {
	if client == nil {
		panic("aimodel/anthropic: nil HTTP client")
	}
	return func(c *Client) { c.httpClient = client }
}

// WithVersion overrides the anthropic-version header.
func WithVersion(version string) ClientOption {
	return func(c *Client) {
		if version != "" {
			c.version = version
		}
	}
}

// WithBeta sets the comma-joined anthropic-beta header values.
func WithBeta(beta ...string) ClientOption {
	return func(c *Client) {
		c.beta = c.beta[:0]
		for _, value := range beta {
			if value != "" {
				c.beta = append(c.beta, value)
			}
		}
	}
}

// WithUserProfileID sets the anthropic-user-profile-id header.
func WithUserProfileID(id string) ClientOption {
	return func(c *Client) { c.userProfileID = id }
}

// NewClient constructs a native Messages client.
func NewClient(apiKey string, options ...ClientOption) *Client {
	client := &Client{
		apiKey:     apiKey,
		baseURL:    anthropicDefaultBaseURL,
		version:    anthropicAPIVersion,
		httpClient: http.DefaultClient,
	}
	for _, option := range options {
		option(client)
	}
	return client
}

// HTTPError reports a non-2xx Anthropic response and retains its bounded body.
type HTTPError struct {
	StatusCode int
	Type       string
	Message    string
	Body       json.RawMessage
	Err        error
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("anthropic: HTTP %d: %s", e.StatusCode, e.Message)
}

func (e *HTTPError) Unwrap() error { return e.Err }

func (c *Client) request(ctx context.Context, input *MessagesRequest, stream bool) (*http.Response, error) {
	if input == nil {
		return nil, fmt.Errorf("anthropic: nil messages request")
	}
	body, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("anthropic: marshal messages request: %w", err)
	}
	var request MessagesRequest
	if err := json.Unmarshal(body, &request); err != nil {
		return nil, fmt.Errorf("anthropic: copy messages request: %w", err)
	}
	request.Stream = stream
	body, err = json.Marshal(&request)
	if err != nil {
		return nil, fmt.Errorf("anthropic: marshal messages request: %w", err)
	}
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("anthropic: create messages request: %w", err)
	}
	httpRequest.Header.Set("Content-Type", "application/json")
	httpRequest.Header.Set("x-api-key", c.apiKey)
	httpRequest.Header.Set("anthropic-version", c.version)
	if beta := strings.Join(c.beta, ","); beta != "" {
		httpRequest.Header.Set("anthropic-beta", beta)
	}
	if c.userProfileID != "" {
		httpRequest.Header.Set("anthropic-user-profile-id", c.userProfileID)
	}
	response, err := c.httpClient.Do(httpRequest)
	if err != nil {
		return nil, fmt.Errorf("anthropic: send messages request: %w", err)
	}
	return response, nil
}

func parseNativeError(response *http.Response) error {
	body, err := io.ReadAll(io.LimitReader(response.Body, maxNativeBodySize))
	result := &HTTPError{
		StatusCode: response.StatusCode,
		Body:       append(json.RawMessage(nil), body...),
		Message:    string(body),
		Err:        err,
	}
	if err != nil {
		result.Message = "failed to read error response"
		return result
	}
	var wire MessagesErrorResponse
	if json.Unmarshal(body, &wire) == nil && wire.Error.Message != "" {
		result.Type = wire.Error.Type
		result.Message = wire.Error.Message
	}
	return result
}

// Messages performs a non-streaming native Messages call.
func (c *Client) Messages(ctx context.Context, request *MessagesRequest) (*MessagesResponse, error) {
	response, err := c.request(ctx, request, false)
	if err != nil {
		return nil, err
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, parseNativeError(response)
	}
	var result MessagesResponse
	if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("anthropic: decode messages response: %w", err)
	}
	return &result, nil
}

// StreamEvent is one native Anthropic SSE event. Raw preserves the payload.
type StreamEvent struct {
	Type              string
	MessageStart      *MessageStartEvent
	ContentBlockStart *ContentBlockStartEvent
	ContentBlockDelta *ContentBlockDeltaEvent
	MessageDelta      *MessageDeltaEvent
	Error             *MessagesErrorResponse
	Raw               json.RawMessage
}

// MessageStream reads native events without canonical aggregation.
type MessageStream struct {
	body io.ReadCloser
	scan *bufio.Scanner
	once sync.Once
}

// MessagesStream starts a native streaming Messages call.
func (c *Client) MessagesStream(ctx context.Context, request *MessagesRequest) (*MessageStream, error) {
	response, err := c.request(ctx, request, true)
	if err != nil {
		return nil, err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		defer func() { _ = response.Body.Close() }()
		return nil, parseNativeError(response)
	}
	scanner := bufio.NewScanner(response.Body)
	scanner.Buffer(make([]byte, 0, 64<<10), maxNativeBodySize)
	return &MessageStream{body: response.Body, scan: scanner}, nil
}

// Recv returns the next event in arrival order.
func (s *MessageStream) Recv() (*StreamEvent, error) {
	var eventType string
	var data []string
	for s.scan.Scan() {
		line := s.scan.Text()
		if line == "" {
			if len(data) == 0 {
				eventType = ""
				continue
			}
			return decodeNativeEvent(eventType, strings.Join(data, "\n"))
		}
		if strings.HasPrefix(line, ":") {
			continue
		}
		if value, ok := strings.CutPrefix(line, "event:"); ok {
			eventType = strings.TrimSpace(value)
			continue
		}
		if value, ok := strings.CutPrefix(line, "data:"); ok {
			data = append(data, strings.TrimPrefix(value, " "))
		}
	}
	if err := s.scan.Err(); err != nil {
		_ = s.Close()
		return nil, fmt.Errorf("anthropic: read stream: %w", err)
	}
	if len(data) != 0 {
		return decodeNativeEvent(eventType, strings.Join(data, "\n"))
	}
	_ = s.Close()
	return nil, io.EOF
}

func decodeNativeEvent(eventType, payload string) (*StreamEvent, error) {
	raw := []byte(payload)
	event := &StreamEvent{Type: eventType, Raw: append(json.RawMessage(nil), raw...)}
	var envelope struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil {
		return nil, fmt.Errorf("anthropic: decode %s event: %w", eventType, err)
	}
	if envelope.Type != "" {
		event.Type = envelope.Type
	}
	var target any
	switch event.Type {
	case "message_start":
		event.MessageStart = &MessageStartEvent{}
		target = event.MessageStart
	case "content_block_start":
		event.ContentBlockStart = &ContentBlockStartEvent{}
		target = event.ContentBlockStart
	case "content_block_delta":
		event.ContentBlockDelta = &ContentBlockDeltaEvent{}
		target = event.ContentBlockDelta
	case "message_delta":
		event.MessageDelta = &MessageDeltaEvent{}
		target = event.MessageDelta
	case "error":
		event.Error = &MessagesErrorResponse{}
		target = event.Error
	default:
		return event, nil
	}
	if err := json.Unmarshal(raw, target); err != nil {
		return nil, fmt.Errorf("anthropic: decode %s event: %w", event.Type, err)
	}
	return event, nil
}

// Close releases the response body and is safe to call repeatedly.
func (s *MessageStream) Close() error {
	var err error
	s.once.Do(func() { err = s.body.Close() })
	return err
}
