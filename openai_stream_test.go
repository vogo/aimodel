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
	"errors"
	"io"
	"strings"
	"testing"
)

func TestStreamRecvSingleChunk(t *testing.T) {
	body := "data: " + `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}` + "\n\n"
	body += "data: [DONE]\n\n"

	s := newStream(io.NopCloser(strings.NewReader(body)))

	chunk, err := s.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if chunk.ID != "1" {
		t.Errorf("id = %q", chunk.ID)
	}
	if len(chunk.Choices) != 1 {
		t.Fatalf("choices len = %d", len(chunk.Choices))
	}
	if chunk.Choices[0].Delta.Content.Text() != "Hi" {
		t.Errorf("delta content = %q", chunk.Choices[0].Delta.Content.Text())
	}

	_, err = s.Recv()
	if !errors.Is(err, io.EOF) {
		t.Errorf("expected io.EOF, got %v", err)
	}
}

func TestStreamRecvMultipleChunks(t *testing.T) {
	body := ""
	body += "data: " + `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}` + "\n\n"
	body += "data: " + `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}` + "\n\n"
	body += "data: " + `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n"
	body += "data: [DONE]\n\n"

	s := newStream(io.NopCloser(strings.NewReader(body)))

	var contents []string
	for {
		chunk, err := s.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
		if len(chunk.Choices) > 0 {
			contents = append(contents, chunk.Choices[0].Delta.Content.Text())
		}
	}

	if len(contents) != 3 {
		t.Fatalf("got %d chunks, want 3", len(contents))
	}
	if contents[0] != "Hello" || contents[1] != " world" {
		t.Errorf("contents = %v", contents)
	}
}

func TestStreamSkipsHeartbeats(t *testing.T) {
	body := ""
	body += "\n"            // empty line
	body += ": heartbeat\n" // SSE comment
	body += "\n"
	body += "data: " + `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}` + "\n\n"
	body += "data: [DONE]\n\n"

	s := newStream(io.NopCloser(strings.NewReader(body)))

	chunk, err := s.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if chunk.Choices[0].Delta.Content.Text() != "ok" {
		t.Errorf("content = %q", chunk.Choices[0].Delta.Content.Text())
	}
}

func TestStreamClose(t *testing.T) {
	body := "data: " + `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}` + "\n\n"

	s := newStream(io.NopCloser(strings.NewReader(body)))
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	_, err := s.Recv()
	if !errors.Is(err, ErrStreamClosed) {
		t.Errorf("got %v, want ErrStreamClosed", err)
	}
}

func TestStreamEmptyBody(t *testing.T) {
	s := newStream(io.NopCloser(strings.NewReader("")))

	_, err := s.Recv()
	if !errors.Is(err, io.EOF) {
		t.Errorf("got %v, want io.EOF", err)
	}
}

func TestStreamInvalidJSON(t *testing.T) {
	body := "data: {invalid json}\n\n"

	s := newStream(io.NopCloser(strings.NewReader(body)))

	_, err := s.Recv()
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestStreamCloseIdempotent(t *testing.T) {
	body := "data: " + `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}` + "\n\n"

	s := newStream(io.NopCloser(strings.NewReader(body)))

	// First close should succeed.
	if err := s.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}

	// Second close should be a no-op (return nil).
	if err := s.Close(); err != nil {
		t.Fatalf("second Close: %v, want nil", err)
	}

	// Recv after close should return ErrStreamClosed.
	_, err := s.Recv()
	if !errors.Is(err, ErrStreamClosed) {
		t.Errorf("got %v, want ErrStreamClosed", err)
	}
}

func TestStreamAPIError(t *testing.T) {
	body := `data: {"error":{"message":"Rate limit exceeded","type":"tokens","code":"rate_limit_exceeded"}}` + "\n\n"

	s := newStream(io.NopCloser(strings.NewReader(body)))

	_, err := s.Recv()
	if err == nil {
		t.Fatal("expected error")
	}

	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError, got %T", err)
	}
	if apiErr.Code != "rate_limit_exceeded" {
		t.Errorf("code = %q", apiErr.Code)
	}
}

func TestStreamUsageFromFinalChunk(t *testing.T) {
	body := ""
	body += "data: " + `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}` + "\n\n"
	body += "data: " + `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}` + "\n\n"
	body += "data: [DONE]\n\n"

	s := newStream(io.NopCloser(strings.NewReader(body)))

	for {
		_, err := s.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
	}

	usage := s.Usage()
	if usage == nil {
		t.Fatal("Usage() = nil, want non-nil")
	}
	if usage.PromptTokens != 10 {
		t.Errorf("prompt_tokens = %d, want 10", usage.PromptTokens)
	}
	if usage.CompletionTokens != 20 {
		t.Errorf("completion_tokens = %d, want 20", usage.CompletionTokens)
	}
	if usage.TotalTokens != 30 {
		t.Errorf("total_tokens = %d, want 30", usage.TotalTokens)
	}
}

func TestStreamUsageNilWhenNoUsageChunk(t *testing.T) {
	body := ""
	body += "data: " + `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}` + "\n\n"
	body += "data: " + `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n"
	body += "data: [DONE]\n\n"

	s := newStream(io.NopCloser(strings.NewReader(body)))

	for {
		_, err := s.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
	}

	if usage := s.Usage(); usage != nil {
		t.Errorf("Usage() = %+v, want nil", usage)
	}
}

func TestStreamCloseOnCloseCallback(t *testing.T) {
	body := ""
	body += "data: " + `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}` + "\n\n"
	body += "data: " + `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":10,"total_tokens":15}}` + "\n\n"
	body += "data: [DONE]\n\n"

	var callbackUsage *Usage
	var callbackCalled bool

	s := newStream(io.NopCloser(strings.NewReader(body)))
	s = WrapStream(s, func(u *Usage) {
		callbackCalled = true
		callbackUsage = u
	})

	for {
		_, err := s.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
	}

	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	if !callbackCalled {
		t.Fatal("onClose callback was not called")
	}
	if callbackUsage == nil {
		t.Fatal("callback received nil usage, want non-nil")
	}
	if callbackUsage.PromptTokens != 5 {
		t.Errorf("prompt_tokens = %d, want 5", callbackUsage.PromptTokens)
	}
	if callbackUsage.CompletionTokens != 10 {
		t.Errorf("completion_tokens = %d, want 10", callbackUsage.CompletionTokens)
	}
	if callbackUsage.TotalTokens != 15 {
		t.Errorf("total_tokens = %d, want 15", callbackUsage.TotalTokens)
	}
}

func TestStreamUsageCacheReadTokens(t *testing.T) {
	body := ""
	body += "data: " + `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}` + "\n\n"
	body += "data: " + `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,"prompt_tokens_details":{"cached_tokens":40}}}` + "\n\n"
	body += "data: [DONE]\n\n"

	s := newStream(io.NopCloser(strings.NewReader(body)))

	for {
		_, err := s.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
	}

	usage := s.Usage()
	if usage == nil {
		t.Fatal("Usage() = nil, want non-nil")
	}
	if usage.CacheReadTokens != 40 {
		t.Errorf("CacheReadTokens = %d, want 40", usage.CacheReadTokens)
	}
	if usage.PromptTokens != 100 {
		t.Errorf("PromptTokens = %d, want 100", usage.PromptTokens)
	}
}

func TestStreamCloseOnCloseCallbackWithoutUsage(t *testing.T) {
	body := ""
	body += "data: " + `{"id":"1","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}` + "\n\n"
	body += "data: [DONE]\n\n"

	var callbackCalled bool
	var callbackUsage *Usage

	s := newStream(io.NopCloser(strings.NewReader(body)))
	s = WrapStream(s, func(u *Usage) {
		callbackCalled = true
		callbackUsage = u
	})

	for {
		_, err := s.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
	}

	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	if !callbackCalled {
		t.Fatal("onClose callback was not called")
	}
	if callbackUsage != nil {
		t.Errorf("callback usage = %+v, want nil", callbackUsage)
	}
}

func TestWrapStreamNil(t *testing.T) {
	var callbackCalled bool
	var callbackUsage *Usage

	result := WrapStream(nil, func(u *Usage) {
		callbackCalled = true
		callbackUsage = u
	})

	if result != nil {
		t.Errorf("WrapStream(nil, cb) = %v, want nil", result)
	}
	if !callbackCalled {
		t.Fatal("onClose callback was not called immediately for nil stream")
	}
	if callbackUsage != nil {
		t.Errorf("callback usage = %+v, want nil", callbackUsage)
	}
}

func TestWrapStreamNilCallbackNil(t *testing.T) {
	// Should not panic and should return nil.
	result := WrapStream(nil, nil)
	if result != nil {
		t.Errorf("WrapStream(nil, nil) = %v, want nil", result)
	}
}
