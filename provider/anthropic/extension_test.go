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
	"errors"
	"strings"
	"testing"

	"github.com/vogo/aimodel/ais"
)

// TestToAnthropicRequest_MistypedExtensionFails verifies a wrong-typed value
// in this provider's namespace fails translation — before any network I/O —
// with a *ais.ExtensionTypeError naming the canonical node, at every
// request-side node.
func TestToAnthropicRequest_MistypedExtensionFails(t *testing.T) {
	tests := []struct {
		node  string
		build func() *ChatRequest
	}{
		{
			node: "ChatRequest",
			build: func() *ChatRequest {
				req := &ChatRequest{Model: "m", Messages: []Message{{Role: RoleUser, Content: NewTextContent("hi")}}}
				req.Extensions.Set(Name, "not-a-request-extension")

				return req
			},
		},
		{
			node: "Message",
			build: func() *ChatRequest {
				req := &ChatRequest{Model: "m", Messages: []Message{{Role: RoleUser, Content: NewTextContent("hi")}}}
				req.Messages[0].Extensions.Set(Name, 42)

				return req
			},
		},
		{
			node: "Tool",
			build: func() *ChatRequest {
				req := &ChatRequest{
					Model:    "m",
					Messages: []Message{{Role: RoleUser, Content: NewTextContent("hi")}},
					Tools:    []Tool{{Type: "function", Function: FunctionDefinition{Name: "f"}}},
				}
				req.Tools[0].Extensions.Set(Name, struct{}{})

				return req
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.node, func(t *testing.T) {
			_, err := toAnthropicRequest(tt.build())
			if err == nil {
				t.Fatal("expected an extension type error")
			}

			var extErr *ais.ExtensionTypeError
			if !errors.As(err, &extErr) {
				t.Fatalf("error type = %T, want *ais.ExtensionTypeError", err)
			}

			if extErr.Provider != Name || extErr.Node != tt.node {
				t.Errorf("error = %+v, want provider %q node %q", extErr, Name, tt.node)
			}
		})
	}
}

// TestProviderNewChatRequest_MistypedExtensionFailsBeforeIO verifies the
// provider surface rejects a mis-typed extension at request-build time — the
// caller gets the error without any HTTP request being issued.
func TestProviderNewChatRequest_MistypedExtensionFailsBeforeIO(t *testing.T) {
	prov, err := New(ais.Config{APIKey: "k"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	req := &ChatRequest{Model: "m", Messages: []Message{{Role: RoleUser, Content: NewTextContent("hi")}}}
	req.Extensions.Set(Name, "garbage")

	if _, err := prov.NewChatRequest(context.Background(), req); err == nil {
		t.Fatal("expected an extension type error from NewChatRequest")
	}
}

// TestToAnthropicRequest_RepeatedTranslationNoSideEffects verifies translating
// the same extended request twice produces identical wire JSON and leaves the
// caller's extension values untouched — reuse accumulates nothing.
func TestToAnthropicRequest_RepeatedTranslationNoSideEffects(t *testing.T) {
	req := &ChatRequest{
		Model: "claude-sonnet-4",
		Messages: []Message{
			cacheMsg(Message{Role: RoleSystem, Content: NewTextContent("sys")}),
			{Role: RoleUser, Content: NewTextContent("hi")},
		},
		Tools: []Tool{
			cacheTool(Tool{Type: "function", Function: FunctionDefinition{Name: "t"}}),
		},
	}
	autoCache(req, "1h")

	marshal := func() string {
		t.Helper()

		ar, err := toAnthropicRequest(req)
		if err != nil {
			t.Fatalf("toAnthropicRequest: %v", err)
		}

		data, err := json.Marshal(ar)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}

		return string(data)
	}

	first, second := marshal(), marshal()
	if first != second {
		t.Errorf("repeated translation diverged:\nfirst  %s\nsecond %s", first, second)
	}

	if count := strings.Count(first, `"cache_control":`); count != 3 {
		t.Errorf("cache_control count = %d, want 3 (system + tool + request root): %s", count, first)
	}

	ext := RequestExtensionOf(req)
	if ext == nil || !ext.AutoCache || ext.AutoCacheTTL != "1h" {
		t.Errorf("caller request extension changed: %+v", ext)
	}
}

// TestExtendRequest_NilRemoves verifies passing nil detaches a previously
// attached extension, restoring the zero-value wire behavior.
func TestExtendRequest_NilRemoves(t *testing.T) {
	req := &ChatRequest{Model: "m", Messages: []Message{{Role: RoleUser, Content: NewTextContent("hi")}}}
	autoCache(req, "1h")
	ExtendRequest(req, nil)

	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}

	if ar.CacheControl != nil {
		t.Errorf("cache_control = %+v, want none after removal", ar.CacheControl)
	}

	if RequestExtensionOf(req) != nil {
		t.Error("request extension should be absent after ExtendRequest(req, nil)")
	}
}
