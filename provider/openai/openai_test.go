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
	"context"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/vogo/aimodel/ais"
)

func newProvider(t *testing.T) *provider {
	t.Helper()

	p, err := New(ais.Config{APIKey: "sk-test", BaseURL: "https://ais.example.com/v1"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	return p.(*provider)
}

func TestNewRequiresBaseURL(t *testing.T) {
	_, err := New(ais.Config{APIKey: "sk-test"})
	if !errors.Is(err, ais.ErrNoBaseURL) {
		t.Errorf("err = %v, want ErrNoBaseURL", err)
	}
}

func TestNewRejectsOptions(t *testing.T) {
	_, err := New(ais.Config{APIKey: "sk-test", BaseURL: "https://x", Options: struct{}{}})
	if err == nil {
		t.Fatal("expected error for unexpected options")
	}
}

func TestNewChatRequestBearerAndPath(t *testing.T) {
	p := newProvider(t)

	req, err := p.NewChatRequest(context.Background(), &ais.ChatRequest{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("NewChatRequest: %v", err)
	}

	if req.URL.Path != "/v1/chat/completions" {
		t.Errorf("path = %q", req.URL.Path)
	}

	if got := req.Header.Get("Authorization"); got != "Bearer sk-test" {
		t.Errorf("Authorization = %q", got)
	}

	if got := req.Header.Get("Content-Type"); got != "application/json" {
		t.Errorf("Content-Type = %q", got)
	}
}

func TestNewChatRequestStreamDefaultsIncludeUsage(t *testing.T) {
	p := newProvider(t)

	req, err := p.NewChatRequest(context.Background(), &ais.ChatRequest{Model: "gpt-4o", Stream: true})
	if err != nil {
		t.Fatalf("NewChatRequest: %v", err)
	}

	body, _ := io.ReadAll(req.Body)

	var decoded struct {
		Stream        bool `json:"stream"`
		StreamOptions *struct {
			IncludeUsage bool `json:"include_usage"`
		} `json:"stream_options"`
	}
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if !decoded.Stream {
		t.Error("stream should be true")
	}

	if decoded.StreamOptions == nil || !decoded.StreamOptions.IncludeUsage {
		t.Error("stream_options.include_usage should default to true on a stream request")
	}
}

func TestNewChatRequestNonStreamOmitsStreamOptions(t *testing.T) {
	p := newProvider(t)

	req, err := p.NewChatRequest(context.Background(), &ais.ChatRequest{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("NewChatRequest: %v", err)
	}

	body, _ := io.ReadAll(req.Body)
	var decoded map[string]json.RawMessage
	if err := json.Unmarshal(body, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, ok := decoded["stream_options"]; ok {
		t.Error("non-stream request must omit stream_options")
	}
}

func TestParseChatResponseEmptyChoices(t *testing.T) {
	p := newProvider(t)

	_, err := p.ParseChatResponse(strings.NewReader(`{"id":"x","choices":[]}`))
	if !errors.Is(err, ais.ErrEmptyResponse) {
		t.Errorf("err = %v, want ErrEmptyResponse", err)
	}
}

func TestParseErrorResponse(t *testing.T) {
	p := newProvider(t)

	err := p.ParseErrorResponse(400, []byte(`{"error":{"code":"bad","message":"nope","type":"invalid_request_error"}}`))

	var apiErr *ais.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError, got %T", err)
	}

	if apiErr.StatusCode != 400 || apiErr.Code != "bad" || apiErr.Type != "invalid_request_error" {
		t.Errorf("unexpected APIError: %+v", apiErr)
	}
}

func TestParseErrorResponseFallback(t *testing.T) {
	p := newProvider(t)

	err := p.ParseErrorResponse(500, []byte(`plain text body`))

	var apiErr *ais.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError, got %T", err)
	}

	if apiErr.Message != "plain text body" {
		t.Errorf("message = %q", apiErr.Message)
	}
}

func TestStreamDecoderChunksAndDone(t *testing.T) {
	body := strings.NewReader(
		": heartbeat\n\n" +
			`data: {"id":"1","choices":[{"index":0,"delta":{"content":"Hi"}}]}` + "\n\n" +
			`data: {"id":"1","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"prompt_tokens_details":{"cached_tokens":4}}}` + "\n\n" +
			"data: [DONE]\n\n",
	)
	decoder := newProvider(t).NewStreamDecoder(body)

	chunk, err := decoder.Next()
	if err != nil {
		t.Fatalf("first Next: %v", err)
	}
	if got := chunk.Choices[0].Delta.Content.Text(); got != "Hi" {
		t.Errorf("content = %q, want Hi", got)
	}

	chunk, err = decoder.Next()
	if err != nil {
		t.Fatalf("second Next: %v", err)
	}
	if chunk.Usage == nil || chunk.Usage.CacheReadTokens != 4 {
		t.Errorf("usage = %+v, want CacheReadTokens 4", chunk.Usage)
	}

	if _, err := decoder.Next(); !errors.Is(err, io.EOF) {
		t.Errorf("final Next error = %v, want io.EOF", err)
	}
}

func TestStreamDecoderAPIError(t *testing.T) {
	body := strings.NewReader(
		`data: {"error":{"message":"rate limited","type":"tokens","code":"rate_limit_exceeded"}}` + "\n\n",
	)
	decoder := newProvider(t).NewStreamDecoder(body)

	_, err := decoder.Next()
	var apiErr *ais.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("Next error = %T %v, want *ais.APIError", err, err)
	}
	if apiErr.Code != "rate_limit_exceeded" {
		t.Errorf("code = %q, want rate_limit_exceeded", apiErr.Code)
	}
}

func TestStreamDecoderRejectsInvalidJSON(t *testing.T) {
	decoder := newProvider(t).NewStreamDecoder(strings.NewReader("data: {invalid json}\n\n"))

	if _, err := decoder.Next(); err == nil {
		t.Fatal("Next error = nil, want JSON decoding error")
	}
}
