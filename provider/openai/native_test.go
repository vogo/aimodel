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
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestNativeChatCompletionsRequest(t *testing.T) {
	var got ChatCompletionRequest
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" || r.Header.Get("Authorization") != "Bearer key" {
			t.Errorf("request=%s headers=%v", r.URL.Path, r.Header)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Error(err)
		}
		if _, err := io.WriteString(w, `{"id":"c","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`); err != nil {
			t.Error(err)
		}
	}))
	defer s.Close()
	req := &ChatCompletionRequest{Model: "gpt", Stream: true}
	_, err := NewClient("key", WithBaseURL(s.URL), WithHTTPClient(s.Client())).ChatCompletions(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if got.Stream || !req.Stream {
		t.Fatalf("stream got=%v caller=%v", got.Stream, req.Stream)
	}
}

func TestNativeChatCompletionsStream(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, err := io.WriteString(w, "data: {\"id\":\"c\",\"future\":true,\ndata: \"choices\":[]}\n\ndata: [DONE]\n\n"); err != nil {
			t.Error(err)
		}
	}))
	defer s.Close()
	stream, err := NewClient("key", WithBaseURL(s.URL), WithHTTPClient(s.Client())).ChatCompletionsStream(context.Background(), &ChatCompletionRequest{Model: "gpt"})
	if err != nil {
		t.Fatal(err)
	}
	c, err := stream.Recv()
	if err != nil {
		t.Fatal(err)
	}
	if c.ID != "c" || len(c.Raw) == 0 {
		t.Fatalf("chunk=%+v", c)
	}
	if _, err = stream.Recv(); err != io.EOF {
		t.Fatalf("done=%v", err)
	}
	if err = stream.Close(); err != nil {
		t.Fatal(err)
	}
}
