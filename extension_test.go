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
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/vogo/aimodel/ais"
	"github.com/vogo/aimodel/provider/anthropic"
	"github.com/vogo/aimodel/provider/openai"
)

// TestChatRequestClone_ExtensionIsolation verifies Clone duplicates the
// extension maps at every node (request, message, tool): re-pointing a
// clone's namespaces never reaches back into the caller's request.
func TestChatRequestClone_ExtensionIsolation(t *testing.T) {
	orig := &ais.ChatRequest{
		Model:    "m",
		Messages: []ais.Message{{Role: ais.RoleUser, Content: ais.NewTextContent("hi")}},
		Tools:    []ais.Tool{{Type: "function", Function: ais.FunctionDefinition{Name: "f"}}},
	}
	orig.Extensions.Set("vendor", "request-level")
	orig.Messages[0].Extensions.Set("vendor", "message-level")
	orig.Tools[0].Extensions.Set("vendor", "tool-level")

	c := orig.Clone()

	c.Extensions.Set("vendor", "mutated")
	c.Extensions.Set("other", "added")
	c.Messages[0].Extensions.Set("vendor", "mutated")
	c.Tools[0].Extensions.Set("vendor", "mutated")

	if got := orig.Extensions.Value("vendor"); got != "request-level" {
		t.Errorf("request extension = %v, want request-level untouched", got)
	}

	if got := orig.Extensions.Value("other"); got != nil {
		t.Errorf("request extension gained key: %v", got)
	}

	if got := orig.Messages[0].Extensions.Value("vendor"); got != "message-level" {
		t.Errorf("message extension = %v, want message-level untouched", got)
	}

	if got := orig.Tools[0].Extensions.Value("vendor"); got != "tool-level" {
		t.Errorf("tool extension = %v, want tool-level untouched", got)
	}
}

// mergeCounter is a test extension value implementing ais.ExtensionMerger:
// it counts merged deltas without mutating either side.
type mergeCounter struct {
	total int
}

func (m *mergeCounter) MergeExtension(delta any) any {
	d, ok := delta.(*mergeCounter)
	if !ok {
		return m
	}

	return &mergeCounter{total: m.total + d.total}
}

// TestMessageAppendDeltaExtensionMerge verifies AppendDelta merges same-name
// namespaces through ais.ExtensionMerger, replaces values that do not
// implement it, and never mutates a previously delivered delta's value.
func TestMessageAppendDeltaExtensionMerge(t *testing.T) {
	var acc ais.Message

	first := ais.Message{}
	firstVal := &mergeCounter{total: 1}
	first.Extensions.Set("vendor", firstVal)

	second := ais.Message{}
	second.Extensions.Set("vendor", &mergeCounter{total: 2})

	acc.AppendDelta(&first)
	acc.AppendDelta(&second)

	got, ok := acc.Extensions.Value("vendor").(*mergeCounter)
	if !ok || got.total != 3 {
		t.Fatalf("merged extension = %#v, want total 3", acc.Extensions.Value("vendor"))
	}

	if firstVal.total != 1 {
		t.Errorf("first delta's value mutated to %d — merge must be copy-on-write", firstVal.total)
	}

	// A value that does not implement the merge contract is replaced.
	var plain ais.Message

	d1 := ais.Message{}
	d1.Extensions.Set("vendor", "one")
	d2 := ais.Message{}
	d2.Extensions.Set("vendor", "two")

	plain.AppendDelta(&d1)
	plain.AppendDelta(&d2)

	if got := plain.Extensions.Value("vendor"); got != "two" {
		t.Errorf("non-merger value = %v, want last write to win", got)
	}
}

// TestOpenAIRequestIgnoresForeignExtensions verifies the OpenAI wire body is
// byte-for-byte identical with and without another provider's extensions
// attached — the extension channel can never leak into canonical JSON.
func TestOpenAIRequestIgnoresForeignExtensions(t *testing.T) {
	build := func(req *ais.ChatRequest) string {
		t.Helper()

		p, err := openai.New(ais.Config{APIKey: "k", BaseURL: "https://api.example.com/v1"})
		if err != nil {
			t.Fatalf("openai.New: %v", err)
		}

		httpReq, err := p.NewChatRequest(context.Background(), req)
		if err != nil {
			t.Fatalf("NewChatRequest: %v", err)
		}

		body, err := io.ReadAll(httpReq.Body)
		if err != nil {
			t.Fatalf("read body: %v", err)
		}

		return string(body)
	}

	plain := &ais.ChatRequest{
		Model:    "gpt-4o",
		Messages: []ais.Message{{Role: ais.RoleUser, Content: ais.NewTextContent("hi")}},
		Tools:    []ais.Tool{{Type: "function", Function: ais.FunctionDefinition{Name: "f"}}},
	}

	extended := plain.Clone()
	anthropic.ExtendRequest(&extended, &anthropic.RequestExtension{AutoCache: true, AutoCacheTTL: "1h", Container: "c1"})
	anthropic.ExtendMessage(&extended.Messages[0], &anthropic.MessageExtension{CacheBreakpoint: true})
	anthropic.ExtendTool(&extended.Tools[0], &anthropic.ToolExtension{CacheBreakpoint: true})

	plainBody, extendedBody := build(plain), build(&extended)
	if plainBody != extendedBody {
		t.Errorf("OpenAI body changed by foreign extensions:\nplain    %s\nextended %s", plainBody, extendedBody)
	}

	for _, leak := range []string{"cache_control", "anthropic", "AutoCache", "container", "Extensions"} {
		if strings.Contains(extendedBody, leak) {
			t.Errorf("OpenAI body leaked %q: %s", leak, extendedBody)
		}
	}
}

// fakeExtProvider demonstrates the acceptance contract of the unified
// extension channel: a third-party provider defines a brand-new proprietary
// request parameter entirely inside its own package — its extension value
// below and the read in NewChatRequest — with zero changes to the canonical
// schema (enforced by the core source-constraint tests).
type fakeExtProvider struct{}

// fakeExtName is the fake provider's extension namespace.
const fakeExtName = "fake-ext"

// fakeRequestExtension is the fake provider's proprietary parameter set.
type fakeRequestExtension struct {
	Sauce string
}

func (p *fakeExtProvider) NewChatRequest(ctx context.Context, req *ais.ChatRequest) (*http.Request, error) {
	body := map[string]any{"model": req.Model}

	if v := req.Extensions.Value(fakeExtName); v != nil {
		ext, ok := v.(*fakeRequestExtension)
		if !ok {
			return nil, &ais.ExtensionTypeError{
				Provider: fakeExtName, Node: "ChatRequest",
				Want: "*aimodel.fakeRequestExtension", Value: v,
			}
		}

		body["sauce"] = ext.Sauce
	}

	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	return http.NewRequestWithContext(ctx, http.MethodPost, "https://fake.example.com/chat", strings.NewReader(string(data)))
}

func (p *fakeExtProvider) ParseChatResponse(body io.Reader) (*ais.ChatResponse, error) {
	return nil, ais.ErrEmptyResponse
}

func (p *fakeExtProvider) ParseErrorResponse(statusCode int, body []byte) error {
	return &ais.APIError{StatusCode: statusCode}
}

func (p *fakeExtProvider) NewStreamDecoder(body io.Reader) ais.StreamDecoder { return nil }

// TestFakeProviderProprietaryParamViaExtensions proves adding a proprietary
// parameter for a new provider needs no canonical change: the fake provider
// sets and consumes its own namespace, the canonical JSON never carries it,
// and a mis-typed value fails with an identifiable error before any I/O.
func TestFakeProviderProprietaryParamViaExtensions(t *testing.T) {
	req := &ais.ChatRequest{Model: "fake-model"}
	req.Extensions.Set(fakeExtName, &fakeRequestExtension{Sauce: "extra-spicy"})

	// The canonical body stays clean.
	canonical, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal canonical: %v", err)
	}

	if strings.Contains(string(canonical), "sauce") || strings.Contains(string(canonical), "extra-spicy") {
		t.Errorf("canonical JSON leaked the proprietary param: %s", canonical)
	}

	// The fake provider consumes it from its own namespace.
	p := &fakeExtProvider{}

	httpReq, err := p.NewChatRequest(context.Background(), req)
	if err != nil {
		t.Fatalf("NewChatRequest: %v", err)
	}

	body, err := io.ReadAll(httpReq.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}

	if !strings.Contains(string(body), `"sauce":"extra-spicy"`) {
		t.Errorf("fake provider body missing proprietary param: %s", body)
	}

	// A mis-typed extension fails identifiably before any network I/O.
	bad := &ais.ChatRequest{Model: "fake-model"}
	bad.Extensions.Set(fakeExtName, 42)

	if _, err := p.NewChatRequest(context.Background(), bad); err == nil {
		t.Fatal("expected an extension type error")
	} else if !strings.Contains(err.Error(), "ChatRequest") {
		t.Errorf("error %q should name the node", err)
	}
}
