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
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestNativeMessagesRequestAndImmutability(t *testing.T) {
	var got MessageRequest
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" || r.Header.Get("x-api-key") != "key" || r.Header.Get("anthropic-version") != anthropicAPIVersion {
			t.Errorf("request = %s headers=%v", r.URL.Path, r.Header)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		if _, err := io.WriteString(w, `{"id":"m","type":"message","role":"assistant","model":"claude","content":[],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":2}}`); err != nil {
			t.Error(err)
		}
	}))
	defer s.Close()
	req := &MessageRequest{Model: "claude", MaxTokens: 0, Messages: []MessageParam{{Role: "user", Content: json.RawMessage(`"hi"`)}}, Stream: true}
	_, err := NewClient("key", WithBaseURL(s.URL), WithHTTPClient(s.Client())).Messages(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if got.Stream || !req.Stream {
		t.Fatalf("stream got=%v caller=%v", got.Stream, req.Stream)
	}
}

func TestNativeMessagesStreamUnknownMultiline(t *testing.T) {
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		if _, err := io.WriteString(w, "event: future\ndata: {\"type\":\"future_event\",\ndata: \"future\":true}\n\n"); err != nil {
			t.Error(err)
		}
	}))
	defer s.Close()
	stream, err := NewClient("key", WithBaseURL(s.URL), WithHTTPClient(s.Client())).MessagesStream(context.Background(), &MessageRequest{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	e, err := stream.Recv()
	if err != nil {
		t.Fatal(err)
	}
	if e.Type != "future_event" || len(e.Raw) == 0 {
		t.Fatalf("event=%+v", e)
	}
	if _, err = stream.Recv(); err != io.EOF {
		t.Fatalf("EOF=%v", err)
	}
	if err = stream.Close(); err != nil {
		t.Fatal(err)
	}
}
