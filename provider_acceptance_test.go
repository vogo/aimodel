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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vogo/aimodel"
	"github.com/vogo/aimodel/core"
)

// This file proves the extension contract: a provider defined entirely outside
// the root package can be registered and driven by NewClient without changing
// any root source. It also pins the registry's deterministic failure modes.

// fakeCalls records which provider hooks the pipeline invoked, shared with the
// test via WithProviderOptions.
type fakeCalls struct {
	newRequest    int
	parseResponse int
	parseError    int
	newDecoder    int
	sawStream     bool
	sawStreamSet  bool
	headerSeen    bool
}

type fakeProvider struct {
	baseURL string
	calls   *fakeCalls
}

const fakeHeader = "X-Fake-Provider"

func newFakeProvider(cfg core.Config) (core.ChatProvider, error) {
	calls, ok := cfg.Options.(*fakeCalls)
	if !ok {
		return nil, errors.New("fake provider requires *fakeCalls options")
	}

	return &fakeProvider{baseURL: cfg.BaseURL, calls: calls}, nil
}

func (p *fakeProvider) NewChatRequest(ctx context.Context, req *core.ChatRequest) (*http.Request, error) {
	p.calls.newRequest++
	p.calls.sawStream = req.Stream
	p.calls.sawStreamSet = true

	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	r, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/fake", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	r.Header.Set(fakeHeader, "1")

	return r, nil
}

func (p *fakeProvider) ParseChatResponse(body io.Reader) (*core.ChatResponse, error) {
	p.calls.parseResponse++

	_, _ = io.Copy(io.Discard, body)

	return &core.ChatResponse{
		ID:      "fake",
		Choices: []core.Choice{{Index: 0, Message: core.Message{Role: core.RoleAssistant, Content: core.NewTextContent("fake-ok")}}},
	}, nil
}

func (p *fakeProvider) ParseErrorResponse(statusCode int, body []byte) error {
	p.calls.parseError++

	return &core.APIError{StatusCode: statusCode, Message: "fake-error", Type: "fake"}
}

func (p *fakeProvider) NewStreamDecoder(body io.Reader) core.StreamDecoder {
	p.calls.newDecoder++

	return &fakeDecoder{}
}

type fakeDecoder struct{ done bool }

func (d *fakeDecoder) Next() (*core.StreamChunk, error) {
	if d.done {
		return nil, io.EOF
	}

	d.done = true

	return &core.StreamChunk{ID: "fake", Choices: []core.StreamChunkChoice{{
		Index: 0,
		Delta: core.Message{Content: core.NewTextContent("chunk")},
	}}}, nil
}

// The fake provider registers once for the whole test binary under a unique
// name; the duplicate-registration test re-registers this same name on purpose.
const fakeProviderName = "fake-acceptance-provider"

func init() {
	core.Register(fakeProviderName, newFakeProvider)
}

func TestFakeProvider_UnaryHooks(t *testing.T) {
	calls := &fakeCalls{}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get(fakeHeader) == "1" {
			calls.headerSeen = true
		}

		if r.URL.Path != "/fake" {
			t.Errorf("path = %q, want /fake", r.URL.Path)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	c, err := aimodel.NewClient(
		aimodel.WithAPIKey("k"),
		aimodel.WithBaseURL(srv.URL),
		aimodel.WithProvider(fakeProviderName),
		aimodel.WithProviderOptions(calls),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	resp, err := c.ChatCompletion(context.Background(), &aimodel.ChatRequest{
		Messages: []aimodel.Message{{Role: aimodel.RoleUser, Content: aimodel.NewTextContent("hi")}},
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}

	if resp.ID != "fake" || resp.Choices[0].Message.Content.Text() != "fake-ok" {
		t.Errorf("unexpected response: %+v", resp)
	}

	if calls.newRequest != 1 {
		t.Errorf("NewChatRequest called %d times, want 1", calls.newRequest)
	}

	if calls.parseResponse != 1 {
		t.Errorf("ParseChatResponse called %d times, want 1", calls.parseResponse)
	}

	if !calls.headerSeen {
		t.Error("provider header did not reach the server")
	}

	if calls.sawStream {
		t.Error("unary call should not set the stream flag")
	}
}

func TestFakeProvider_ErrorHook(t *testing.T) {
	calls := &fakeCalls{}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`boom`))
	}))
	defer srv.Close()

	c, err := aimodel.NewClient(
		aimodel.WithAPIKey("k"),
		aimodel.WithBaseURL(srv.URL),
		aimodel.WithProvider(fakeProviderName),
		aimodel.WithProviderOptions(calls),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	_, err = c.ChatCompletion(context.Background(), &aimodel.ChatRequest{
		Messages: []aimodel.Message{{Role: aimodel.RoleUser, Content: aimodel.NewTextContent("hi")}},
	})

	var apiErr *aimodel.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected *APIError, got %T: %v", err, err)
	}

	if apiErr.Message != "fake-error" || apiErr.StatusCode != http.StatusInternalServerError {
		t.Errorf("unexpected error: %+v", apiErr)
	}

	if calls.parseError != 1 {
		t.Errorf("ParseErrorResponse called %d times, want 1", calls.parseError)
	}
}

func TestFakeProvider_StreamHooks(t *testing.T) {
	calls := &fakeCalls{}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: ignored\n\n"))
	}))
	defer srv.Close()

	c, err := aimodel.NewClient(
		aimodel.WithAPIKey("k"),
		aimodel.WithBaseURL(srv.URL),
		aimodel.WithProvider(fakeProviderName),
		aimodel.WithProviderOptions(calls),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	stream, err := c.ChatCompletionStream(context.Background(), &aimodel.ChatRequest{
		Messages: []aimodel.Message{{Role: aimodel.RoleUser, Content: aimodel.NewTextContent("hi")}},
	})
	if err != nil {
		t.Fatalf("ChatCompletionStream: %v", err)
	}
	defer func() { _ = stream.Close() }()

	chunk, err := stream.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}

	if chunk.Choices[0].Delta.Content.Text() != "chunk" {
		t.Errorf("chunk = %+v", chunk)
	}

	if _, err := stream.Recv(); !errors.Is(err, io.EOF) {
		t.Errorf("second Recv = %v, want io.EOF", err)
	}

	if calls.newDecoder != 1 {
		t.Errorf("NewStreamDecoder called %d times, want 1", calls.newDecoder)
	}

	if !calls.sawStream {
		t.Error("stream call should set the stream flag on the provider request")
	}
}

func TestNewClient_UnknownProviderName(t *testing.T) {
	_, err := aimodel.NewClient(aimodel.WithAPIKey("k"), aimodel.WithProvider("no-such-provider"))
	if err == nil {
		t.Fatal("expected error for unknown provider name")
	}
}

func TestRegister_DeterministicFailures(t *testing.T) {
	assertPanic := func(name string, fn func()) {
		t.Helper()

		defer func() {
			if r := recover(); r == nil {
				t.Errorf("%s: expected panic", name)
			}
		}()

		fn()
	}

	assertPanic("duplicate name", func() {
		// fakeProviderName is already registered in init.
		core.Register(fakeProviderName, newFakeProvider)
	})

	assertPanic("empty name", func() {
		core.Register("", newFakeProvider)
	})

	assertPanic("nil factory", func() {
		core.Register("some-unique-name-for-nil", nil)
	})
}
