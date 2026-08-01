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

// knownResponseEventTypes indexes the documented baseline so the decoder can
// tell a malformed known event (an error) from an event type it has never seen
// (preserved raw).
var knownResponseEventTypes = func() map[string]bool {
	known := make(map[string]bool, len(responseStreamEventTypes))
	for _, eventType := range responseStreamEventTypes {
		known[eventType] = true
	}

	return known
}()

// responsesRequest builds and sends the single POST {baseURL}/responses call.
// The caller's request is never mutated: the stream mode is forced on a copy.
func (c *Client) responsesRequest(ctx context.Context, input *ResponsesRequest, stream bool) (*http.Response, error) {
	if input == nil {
		return nil, fmt.Errorf("openai: nil responses request")
	}

	body, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal responses request: %w", err)
	}

	var request ResponsesRequest
	if err = json.Unmarshal(body, &request); err != nil {
		return nil, fmt.Errorf("openai: copy responses request: %w", err)
	}

	request.Stream = stream

	if body, err = json.Marshal(&request); err != nil {
		return nil, fmt.Errorf("openai: marshal responses request: %w", err)
	}

	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/responses", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: create responses request: %w", err)
	}

	httpRequest.Header.Set("Content-Type", "application/json")
	httpRequest.Header.Set("Authorization", "Bearer "+c.apiKey)

	response, err := c.httpClient.Do(httpRequest)
	if err != nil {
		return nil, fmt.Errorf("openai: send responses request: %w", err)
	}

	return response, nil
}

// Responses performs a non-streaming native Responses call and returns the
// complete Response, including a failed or incomplete one — a response-level
// failure is reported through Response.Status / Error / IncompleteDetails, not
// flattened into a Go error. Transport and non-2xx failures return *HTTPError.
func (c *Client) Responses(ctx context.Context, request *ResponsesRequest) (*Response, error) {
	response, err := c.responsesRequest(ctx, request, false)
	if err != nil {
		return nil, err
	}

	defer func() { _ = response.Body.Close() }()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, parseNativeError(response)
	}

	var result Response
	if err = json.NewDecoder(response.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("openai: decode responses response: %w", err)
	}

	return &result, nil
}

// ResponseStream reads native Responses SSE events in arrival order.
type ResponseStream struct {
	body io.ReadCloser
	scan *bufio.Scanner
	once sync.Once
}

// ResponsesStream starts a streaming native Responses call.
func (c *Client) ResponsesStream(ctx context.Context, request *ResponsesRequest) (*ResponseStream, error) {
	response, err := c.responsesRequest(ctx, request, true)
	if err != nil {
		return nil, err
	}

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		defer func() { _ = response.Body.Close() }()

		return nil, parseNativeError(response)
	}

	scanner := bufio.NewScanner(response.Body)
	scanner.Buffer(make([]byte, 0, 64<<10), maxNativeBodySize)

	return &ResponseStream{body: response.Body, scan: scanner}, nil
}

// Recv returns the next event. It honors SSE event boundaries (blank line),
// skips comments and heartbeats, and joins multi-line data payloads. The stream
// ends with io.EOF; an `error` event, a malformed known event, or a read
// failure returns an error. Every one of those paths closes the body.
func (s *ResponseStream) Recv() (*ResponseStreamEvent, error) {
	var (
		eventName string
		data      []string
	)

	for s.scan.Scan() {
		line := s.scan.Text()

		if line == "" {
			if len(data) == 0 {
				eventName = ""

				continue
			}

			return s.event(eventName, strings.Join(data, "\n"))
		}

		if strings.HasPrefix(line, ":") {
			continue
		}

		if value, ok := strings.CutPrefix(line, "event:"); ok {
			eventName = strings.TrimSpace(value)

			continue
		}

		if value, ok := strings.CutPrefix(line, "data:"); ok {
			data = append(data, strings.TrimPrefix(value, " "))
		}
	}

	if err := s.scan.Err(); err != nil {
		_ = s.Close()

		return nil, fmt.Errorf("openai: read responses stream: %w", err)
	}

	if len(data) != 0 {
		return s.event(eventName, strings.Join(data, "\n"))
	}

	_ = s.Close()

	return nil, io.EOF
}

// event decodes one payload and closes the stream on any terminal condition.
func (s *ResponseStream) event(eventName, payload string) (*ResponseStreamEvent, error) {
	event, err := decodeResponseEvent(eventName, payload)
	if err != nil {
		_ = s.Close()

		return nil, err
	}

	if event.Type == ResponseEventError {
		_ = s.Close()

		return nil, &HTTPError{Code: event.Code, Message: event.Message, Body: event.Raw}
	}

	return event, nil
}

// decodeResponseEvent maps one SSE payload to a typed event. The payload `type`
// discriminator wins over the SSE `event:` name; an event type outside the
// documented baseline is returned with its discriminator and raw payload rather
// than rejected.
func decodeResponseEvent(eventName, payload string) (*ResponseStreamEvent, error) {
	raw := []byte(payload)

	var envelope struct {
		Type string `json:"type"`
	}

	if err := json.Unmarshal(raw, &envelope); err != nil {
		return nil, fmt.Errorf("openai: decode responses stream event: %w", err)
	}

	eventType := envelope.Type
	if eventType == "" {
		eventType = eventName
	}

	var event ResponseStreamEvent
	if err := json.Unmarshal(raw, &event); err != nil {
		if knownResponseEventTypes[eventType] {
			return nil, fmt.Errorf("openai: decode %s event: %w", eventType, err)
		}

		event = ResponseStreamEvent{}
	}

	event.Type = eventType
	event.Raw = append(json.RawMessage(nil), raw...)

	return &event, nil
}

// Close releases the response body and is safe to call repeatedly.
func (s *ResponseStream) Close() error {
	var err error

	s.once.Do(func() { err = s.body.Close() })

	return err
}
