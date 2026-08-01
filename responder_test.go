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

package aimodel_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/vogo/aimodel"
	"github.com/vogo/aimodel/ais"
	"github.com/vogo/aimodel/provider/anthropic"
	"github.com/vogo/aimodel/provider/openai"
)

// Compile-time check: the unified client carries the Responses capability, and
// the chat capability is untouched by it.
var (
	_ aimodel.Responder     = (*aimodel.Client)(nil)
	_ aimodel.ChatCompleter = (*aimodel.Client)(nil)
)

func TestClientResponsesDelegatesToOpenAIProvider(t *testing.T) {
	var (
		calls int
		path  string
		auth  string
		seen  openai.ResponsesRequest
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		path, auth = r.URL.Path, r.Header.Get("Authorization")

		if err := json.NewDecoder(r.Body).Decode(&seen); err != nil {
			t.Error(err)
		}

		w.Header().Set("Content-Type", "application/json")

		if seen.Stream {
			_, _ = io.WriteString(w, "event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":1,\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"hi\"}]}]}}\n\n")

			return
		}

		_, _ = io.WriteString(w, `{"id":"resp_1","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}]}`)
	}))
	defer server.Close()

	client, err := aimodel.NewClient(
		aimodel.WithAPIKey("key"),
		aimodel.WithBaseURL(server.URL+"/v1"),
		aimodel.WithDefaultModel("default-model"),
		aimodel.WithHTTPClient(server.Client()),
		aimodel.WithTimeout(15*time.Second),
	)
	if err != nil {
		t.Fatal(err)
	}

	request := &openai.ResponsesRequest{Model: "gpt-5", Input: openai.NewResponseTextInput("hi")}

	response, err := client.Responses(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}

	if response.ID != "resp_1" || response.OutputText != "hi" {
		t.Fatalf("response = %+v", response)
	}

	if path != "/v1/responses" || auth != "Bearer key" {
		t.Errorf("path=%q auth=%q", path, auth)
	}

	// The request's own model stays authoritative: the canonical default model
	// is not applied on this path.
	if seen.Model != "gpt-5" {
		t.Errorf("model = %q", seen.Model)
	}

	stream, err := client.ResponsesStream(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}

	defer func() { _ = stream.Close() }()

	event, err := stream.Recv()
	if err != nil || event.Type != openai.ResponseEventCompleted || event.Response.OutputText != "hi" {
		t.Fatalf("event=%+v err=%v", event, err)
	}

	if _, err = stream.Recv(); !errors.Is(err, io.EOF) {
		t.Fatalf("terminal Recv error = %v", err)
	}

	if calls != 2 {
		t.Fatalf("calls = %d", calls)
	}
}

func TestClientResponsesUnsupportedProviderFailsLocally(t *testing.T) {
	var calls int

	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		calls++
	}))
	defer server.Close()

	client, err := aimodel.NewClient(
		aimodel.WithAPIKey("key"),
		aimodel.WithBaseURL(server.URL),
		aimodel.WithProvider(anthropic.Name),
		aimodel.WithHTTPClient(server.Client()),
	)
	if err != nil {
		t.Fatal(err)
	}

	request := &openai.ResponsesRequest{Model: "claude-sonnet-4"}

	_, err = client.Responses(context.Background(), request)
	assertCapabilityError(t, err)

	_, err = client.ResponsesStream(context.Background(), request)
	assertCapabilityError(t, err)

	if calls != 0 {
		t.Fatalf("unsupported capability performed %d network calls", calls)
	}
}

func assertCapabilityError(t *testing.T, err error) {
	t.Helper()

	if !errors.Is(err, ais.ErrCapabilityNotSupported) {
		t.Fatalf("error = %T %v", err, err)
	}

	var capErr *ais.CapabilityError
	if !errors.As(err, &capErr) {
		t.Fatalf("error = %T %v", err, err)
	}

	if capErr.Provider != anthropic.Name || capErr.Capability != aimodel.CapabilityResponses {
		t.Fatalf("capability error = %+v", capErr)
	}
}
