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
