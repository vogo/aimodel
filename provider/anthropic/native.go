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
	"time"
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

// WithTimeout bounds the total duration of each call, including reading a
// streaming body. It copies the HTTP client configured so far and sets its
// Timeout, so the caller's own *http.Client is never mutated and a transport
// installed by an earlier WithHTTPClient is preserved. Apply it after
// WithHTTPClient; the reverse order discards the timeout.
func WithTimeout(d time.Duration) ClientOption {
	return func(c *Client) {
		client := *c.httpClient
		client.Timeout = d
		c.httpClient = &client
	}
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
	// Status is the HTTP status code. It is named Status rather than
	// StatusCode so the accessor below can carry that name: consumers match
	// any provider's transport error with
	// errors.As(err, &interface{ StatusCode() int }) without importing this
	// package.
	Status  int
	Type    string
	Message string
	Body    json.RawMessage
	Err     error
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("anthropic: HTTP %d: %s", e.Status, e.Message)
}

func (e *HTTPError) Unwrap() error { return e.Err }

// StatusCode returns the HTTP status code, satisfying the
// interface{ StatusCode() int } a consumer can declare locally.
func (e *HTTPError) StatusCode() int { return e.Status }

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
		Status:  response.StatusCode,
		Body:    append(json.RawMessage(nil), body...),
		Message: string(body),
		Err:     err,
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

// MessageStream reads the native SSE event stream. While the caller reads
// events, the stream also folds each one into the message it reconstructs, so
// Message and Usage are available without the caller tracking deltas. A stream
// has a single reader; Close may be called concurrently with Recv and is
// idempotent.
type MessageStream struct {
	body io.ReadCloser
	scan *bufio.Scanner
	once sync.Once

	mu  sync.Mutex // guards acc against concurrent Recv/Message/Usage
	acc messageAccumulator
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
			return s.decode(eventType, strings.Join(data, "\n"))
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
		return s.decode(eventType, strings.Join(data, "\n"))
	}
	_ = s.Close()
	return nil, io.EOF
}

// decode parses one SSE payload and folds it into the accumulated message.
func (s *MessageStream) decode(eventType, payload string) (*StreamEvent, error) {
	event, err := decodeNativeEvent(eventType, payload)
	if err != nil {
		return nil, err
	}

	s.mu.Lock()
	s.acc.fold(event)
	s.mu.Unlock()

	return event, nil
}

// Message returns the message assembled from the events read so far: content
// blocks in index order, with text and thinking deltas concatenated and tool
// inputs reassembled from their partial-JSON fragments. Call it after Recv
// reports io.EOF for the final result; before that it is a live snapshot, in
// which a tool block's Input may still be an incomplete JSON fragment. It
// returns nil when no event carried message content.
//
// ResponseContentBlock.Raw holds the block as it first arrived and is not
// rewritten by later deltas.
func (s *MessageStream) Message() *MessagesResponse {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.acc.result()
}

// Usage returns the token accounting of the stream, merging the baseline
// Anthropic reports on message_start with the terminal counts on
// message_delta. A later event overwrites only the fields it actually carries,
// so a terminal event reporting just output_tokens leaves the input, cache,
// geography, tier and server-tool numbers from the start event intact. It
// returns nil before the first message_start.
func (s *MessageStream) Usage() *MessagesUsage {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.acc.started {
		return nil
	}

	return &s.acc.response.Usage
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
	case StreamEventTypeMessageStart:
		event.MessageStart = &MessageStartEvent{}
		target = event.MessageStart
	case StreamEventTypeContentBlockStart:
		event.ContentBlockStart = &ContentBlockStartEvent{}
		target = event.ContentBlockStart
	case StreamEventTypeContentBlockDelta:
		event.ContentBlockDelta = &ContentBlockDeltaEvent{}
		target = event.ContentBlockDelta
	case StreamEventTypeMessageDelta:
		event.MessageDelta = &MessageDeltaEvent{}
		target = event.MessageDelta
	case StreamEventTypeError:
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
