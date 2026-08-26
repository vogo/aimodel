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
	"strings"
	"testing"
)

func TestExtraBodyAddsBackendPrivateParameters(t *testing.T) {
	request := ChatCompletionRequest{
		Model:    "qwen3.7-plus",
		Messages: []ChatCompletionMessage{{Role: "user", Content: NewTextContent("hi")}},
		ExtraBody: map[string]json.RawMessage{
			"enable_thinking":      json.RawMessage(`true`),
			"chat_template_kwargs": json.RawMessage(`{"thinking":false}`),
		},
	}

	body, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}

	var decoded map[string]json.RawMessage
	if err = json.Unmarshal(body, &decoded); err != nil {
		t.Fatal(err)
	}

	if string(decoded["enable_thinking"]) != "true" {
		t.Errorf("enable_thinking = %s", decoded["enable_thinking"])
	}

	if string(decoded["chat_template_kwargs"]) != `{"thinking":false}` {
		t.Errorf("chat_template_kwargs = %s", decoded["chat_template_kwargs"])
	}

	if string(decoded["model"]) != `"qwen3.7-plus"` {
		t.Errorf("modelled fields must survive the merge, got %s", body)
	}
}

func TestExtraBodyRejectsCollisionsWithModelledFields(t *testing.T) {
	for _, key := range []string{"model", "messages", "temperature", "stream", "tools", ""} {
		request := ChatCompletionRequest{
			Model:     "gpt-5",
			ExtraBody: map[string]json.RawMessage{key: json.RawMessage(`"override"`)},
		}

		_, err := json.Marshal(request)
		if err == nil {
			t.Errorf("extra body key %q was accepted; it must be rejected at marshal time", key)
		}
	}
}

func TestExtraBodyRejectsInvalidJSON(t *testing.T) {
	request := ChatCompletionRequest{
		Model:     "gpt-5",
		ExtraBody: map[string]json.RawMessage{"enable_thinking": json.RawMessage(`{"unterminated":`)},
	}

	if _, err := json.Marshal(request); err == nil {
		t.Error("an invalid JSON value must fail the marshal, not reach the backend")
	}
}

func TestRequestRoundTripPreservesUnmodelledKeys(t *testing.T) {
	const body = `{"model":"glm-5.2","messages":[{"role":"user","content":"hi"}],"enable_thinking":true,"custom_backend_knob":{"a":[1,2]}}`

	var request ChatCompletionRequest
	if err := json.Unmarshal([]byte(body), &request); err != nil {
		t.Fatal(err)
	}

	if len(request.ExtraBody) != 2 {
		t.Fatalf("extra body = %v, want the two unmodelled keys", request.ExtraBody)
	}

	encoded, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}

	var original, roundTripped map[string]any
	if err = json.Unmarshal([]byte(body), &original); err != nil {
		t.Fatal(err)
	}

	if err = json.Unmarshal(encoded, &roundTripped); err != nil {
		t.Fatal(err)
	}

	for key, want := range original {
		got, ok := roundTripped[key]
		if !ok {
			t.Errorf("round trip dropped %q", key)

			continue
		}

		wantJSON, _ := json.Marshal(want)
		gotJSON, _ := json.Marshal(got)

		if string(wantJSON) != string(gotJSON) {
			t.Errorf("round trip changed %q: %s -> %s", key, wantJSON, gotJSON)
		}
	}
}

func TestExtraBodyReachesTheBackend(t *testing.T) {
	var received string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		received = string(body)

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"chat-1","choices":[{"index":0,"message":{"role":"assistant","content":"hi"}}]}`)
	}))
	defer server.Close()

	client := NewClient("key", WithBaseURL(server.URL), WithHTTPClient(server.Client()))
	request := &ChatCompletionRequest{
		Model:     "glm-5.2",
		Messages:  []ChatCompletionMessage{{Role: "user", Content: NewTextContent("hi")}},
		ExtraBody: map[string]json.RawMessage{"enable_thinking": json.RawMessage(`true`)},
	}

	if _, err := client.ChatCompletions(context.Background(), request); err != nil {
		t.Fatal(err)
	}

	if !strings.Contains(received, `"enable_thinking":true`) {
		t.Errorf("request body = %s", received)
	}

	if !strings.Contains(received, `"stream":false`) && strings.Contains(received, `"stream"`) {
		t.Errorf("stream flag = %s", received)
	}
}

func TestMarshalRejectsCollisionBeforeAnyRequest(t *testing.T) {
	var called bool

	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		called = true
	}))
	defer server.Close()

	client := NewClient("key", WithBaseURL(server.URL), WithHTTPClient(server.Client()))
	request := &ChatCompletionRequest{
		Model:     "gpt-5",
		ExtraBody: map[string]json.RawMessage{"model": json.RawMessage(`"sneaky"`)},
	}

	if _, err := client.ChatCompletions(context.Background(), request); err == nil {
		t.Error("a colliding extra body key must fail the call")
	}

	if called {
		t.Error("the request must fail before any network I/O")
	}
}
