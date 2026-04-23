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
	"encoding/json"
	"strings"
	"testing"
)

// TestToAnthropicRequest_SystemCacheBreakpoint verifies that a system
// message with CacheBreakpoint=true lands as a block-array system field
// whose last block carries "cache_control":{"type":"ephemeral"}.
func TestToAnthropicRequest_SystemCacheBreakpoint(t *testing.T) {
	req := &ChatRequest{
		Model: "claude-sonnet-4",
		Messages: []Message{
			{
				Role:            RoleSystem,
				Content:         NewTextContent("You are a helpful coder."),
				CacheBreakpoint: true,
			},
			{Role: RoleUser, Content: NewTextContent("hi")},
		},
	}

	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}

	var systemBlocks []anthropicContentBlock
	if err := json.Unmarshal(ar.System, &systemBlocks); err != nil {
		t.Fatalf("system not a block array: %v (raw=%s)", err, string(ar.System))
	}
	if len(systemBlocks) == 0 {
		t.Fatal("expected at least one system block")
	}
	last := systemBlocks[len(systemBlocks)-1]
	if last.CacheControl == nil || last.CacheControl.Type != "ephemeral" {
		t.Errorf("last system block CacheControl = %+v, want {ephemeral}", last.CacheControl)
	}
}

// TestToAnthropicRequest_ToolCacheBreakpoint verifies a tool with
// CacheBreakpoint=true emits cache_control on its translated form.
func TestToAnthropicRequest_ToolCacheBreakpoint(t *testing.T) {
	req := &ChatRequest{
		Model:    "claude-sonnet-4",
		Messages: []Message{{Role: RoleUser, Content: NewTextContent("hi")}},
		Tools: []Tool{
			{Type: "function", Function: FunctionDefinition{Name: "t1", Description: "first"}},
			{Type: "function", Function: FunctionDefinition{Name: "t2", Description: "second"}, CacheBreakpoint: true},
		},
	}
	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}
	if len(ar.Tools) != 2 {
		t.Fatalf("tools = %d, want 2", len(ar.Tools))
	}
	if ar.Tools[0].CacheControl != nil {
		t.Errorf("tool 0 CacheControl should be nil, got %+v", ar.Tools[0].CacheControl)
	}
	if ar.Tools[1].CacheControl == nil || ar.Tools[1].CacheControl.Type != "ephemeral" {
		t.Errorf("tool 1 CacheControl = %+v, want {ephemeral}", ar.Tools[1].CacheControl)
	}
}

// TestToAnthropicRequest_NoCacheBreakpoint verifies that without flags
// set, no cache_control field appears anywhere in the translated request.
func TestToAnthropicRequest_NoCacheBreakpoint(t *testing.T) {
	req := &ChatRequest{
		Model: "claude-sonnet-4",
		Messages: []Message{
			{Role: RoleSystem, Content: NewTextContent("You are a helpful coder.")},
			{Role: RoleUser, Content: NewTextContent("hi")},
		},
		Tools: []Tool{
			{Type: "function", Function: FunctionDefinition{Name: "t1"}},
		},
	}
	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}

	body, err := json.Marshal(ar)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(body), "cache_control") {
		t.Errorf("unexpected cache_control in request body: %s", body)
	}
}

// TestToAnthropicRequest_BothBreakpoints verifies the happy-path case we
// actually ship: one system + one last-tool marker = exactly 2 cache_control
// occurrences in the outbound JSON.
func TestToAnthropicRequest_BothBreakpoints(t *testing.T) {
	req := &ChatRequest{
		Model: "claude-sonnet-4",
		Messages: []Message{
			{Role: RoleSystem, Content: NewTextContent("sys"), CacheBreakpoint: true},
			{Role: RoleUser, Content: NewTextContent("hi")},
		},
		Tools: []Tool{
			{Type: "function", Function: FunctionDefinition{Name: "t1"}},
			{Type: "function", Function: FunctionDefinition{Name: "t2"}, CacheBreakpoint: true},
		},
	}
	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}
	body, err := json.Marshal(ar)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	count := strings.Count(string(body), `"cache_control":`)
	if count != 2 {
		t.Errorf("cache_control count = %d, want 2 (body=%s)", count, body)
	}
}

// TestToAnthropicRequest_SystemWithPartsAndCache verifies that a system
// message using multimodal parts still gets the marker on its last text
// block.
func TestToAnthropicRequest_SystemWithPartsAndCache(t *testing.T) {
	req := &ChatRequest{
		Model: "claude-sonnet-4",
		Messages: []Message{
			{
				Role: RoleSystem,
				Content: NewPartsContent(
					ContentPart{Type: "text", Text: "first"},
					ContentPart{Type: "text", Text: "second"},
				),
				CacheBreakpoint: true,
			},
			{Role: RoleUser, Content: NewTextContent("hi")},
		},
	}
	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}
	var blocks []anthropicContentBlock
	if err := json.Unmarshal(ar.System, &blocks); err != nil {
		t.Fatalf("system not a block array: %v", err)
	}
	if len(blocks) != 2 {
		t.Fatalf("blocks = %d, want 2", len(blocks))
	}
	if blocks[0].CacheControl != nil {
		t.Errorf("first block should not carry cache_control")
	}
	if blocks[1].CacheControl == nil {
		t.Errorf("last block missing cache_control")
	}
}

// TestChatRequest_OpenAIShape_NoCacheControl verifies that marshalling the
// canonical ChatRequest (which is the on-wire body for OpenAI-compatible
// endpoints) never leaks a cache_control field, even when CacheBreakpoint
// is set on messages or tools.
func TestChatRequest_OpenAIShape_NoCacheControl(t *testing.T) {
	req := &ChatRequest{
		Model: "gpt-4",
		Messages: []Message{
			{Role: RoleSystem, Content: NewTextContent("sys"), CacheBreakpoint: true},
			{Role: RoleUser, Content: NewTextContent("hi"), CacheBreakpoint: true},
		},
		Tools: []Tool{
			{Type: "function", Function: FunctionDefinition{Name: "t1"}, CacheBreakpoint: true},
		},
	}
	body, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(body), "cache_control") {
		t.Errorf("CacheBreakpoint leaked into OpenAI request body: %s", body)
	}
	if strings.Contains(string(body), "CacheBreakpoint") {
		t.Errorf("CacheBreakpoint field name leaked into OpenAI request body: %s", body)
	}
}
