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
	"encoding/json"
	"strings"
	"testing"
)

// jsonSchemaFormat is the OpenAI-shaped response_format used by the tests.
func jsonSchemaFormat() map[string]any {
	return map[string]any{
		"type": "json_schema",
		"json_schema": map[string]any{
			"name": "person",
			"schema": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{"type": "string"},
				},
			},
		},
	}
}

// TestToAnthropicRequest_OutputConfig covers the four shapes output_config can
// take: absent, format only, effort only, and both combined. It also asserts
// the deprecated top-level "effort" is never emitted.
func TestToAnthropicRequest_OutputConfig(t *testing.T) {
	tests := []struct {
		name           string
		effort         string
		responseFormat any
		wantJSON       string // "" means output_config must be absent
	}{
		{
			name:     "neither",
			wantJSON: "",
		},
		{
			name:           "format only",
			responseFormat: jsonSchemaFormat(),
			wantJSON:       `"output_config":{"format":{"type":"json_schema","schema":{"properties":{"name":{"type":"string"}},"type":"object"}}}`,
		},
		{
			name:     "effort only",
			effort:   ReasoningEffortXHigh,
			wantJSON: `"output_config":{"effort":"xhigh"}`,
		},
		{
			name:           "effort and format",
			effort:         ReasoningEffortLow,
			responseFormat: jsonSchemaFormat(),
			wantJSON:       `"output_config":{"effort":"low","format":{"type":"json_schema","schema":{"properties":{"name":{"type":"string"}},"type":"object"}}}`,
		},
		{
			name:           "json_object format is not translated",
			responseFormat: map[string]any{"type": "json_object"},
			wantJSON:       "",
		},
		{
			name:           "flat schema shape",
			responseFormat: map[string]any{"type": "json_schema", "schema": map[string]any{"type": "object"}},
			wantJSON:       `"output_config":{"format":{"type":"json_schema","schema":{"type":"object"}}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &ChatRequest{
				Model:           ModelAnthropicClaude4Sonnet,
				Messages:        []Message{{Role: RoleUser, Content: NewTextContent("hi")}},
				ReasoningEffort: tt.effort,
				ResponseFormat:  tt.responseFormat,
			}

			ar, err := toAnthropicRequest(req)
			if err != nil {
				t.Fatalf("toAnthropicRequest: %v", err)
			}

			if ar.Effort != "" {
				t.Errorf("deprecated top-level effort = %q, want empty", ar.Effort)
			}

			data, err := json.Marshal(ar)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}

			if tt.wantJSON == "" {
				if strings.Contains(string(data), "output_config") {
					t.Errorf("output_config should be omitted, got: %s", data)
				}

				if strings.Contains(string(data), `"effort"`) {
					t.Errorf("no effort should be emitted, got: %s", data)
				}

				return
			}

			if !strings.Contains(string(data), tt.wantJSON) {
				t.Errorf("marshalled request = %s\nwant substring %s", data, tt.wantJSON)
			}
		})
	}
}

// TestToAnthropicRequest_ContainerAndInferenceGeo verifies both fields pass
// straight through and are omitted when unset.
func TestToAnthropicRequest_ContainerAndInferenceGeo(t *testing.T) {
	base := []Message{{Role: RoleUser, Content: NewTextContent("hi")}}

	withExt := &ChatRequest{
		Model:    ModelAnthropicClaude4Sonnet,
		Messages: base,
	}
	ExtendRequest(withExt, &RequestExtension{Container: "container_abc123", InferenceGeo: "eu"})

	set, err := toAnthropicRequest(withExt)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}

	if set.Container != "container_abc123" {
		t.Errorf("container = %q, want container_abc123", set.Container)
	}

	if set.InferenceGeo != "eu" {
		t.Errorf("inference_geo = %q, want eu", set.InferenceGeo)
	}

	data, err := json.Marshal(set)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	if !strings.Contains(string(data), `"container":"container_abc123"`) ||
		!strings.Contains(string(data), `"inference_geo":"eu"`) {
		t.Errorf("marshalled request missing fields: %s", data)
	}

	unset, err := toAnthropicRequest(&ChatRequest{Model: ModelAnthropicClaude4Sonnet, Messages: base})
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}

	data, err = json.Marshal(unset)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	if strings.Contains(string(data), "container") || strings.Contains(string(data), "inference_geo") {
		t.Errorf("unset fields should be omitted, got: %s", data)
	}
}

// TestToAnthropicRequest_ToolExtensions verifies each new tool field is copied
// verbatim, that false is preserved (not dropped like a zero value), and that
// a versioned built-in Type passes through while OpenAI's "function" does not.
func TestToAnthropicRequest_ToolExtensions(t *testing.T) {
	strict, deferLoading, eager := true, false, true

	req := &ChatRequest{
		Model:    ModelAnthropicClaude4Sonnet,
		Messages: []Message{{Role: RoleUser, Content: NewTextContent("hi")}},
		Tools: []Tool{
			{
				Type:     "function",
				Function: FunctionDefinition{Name: "lookup"},
				Strict:   &strict,
			},
			{
				Type:     "web_search_20260209",
				Function: FunctionDefinition{Name: "web_search"},
			},
		},
	}
	ExtendTool(&req.Tools[0], &ToolExtension{
		DeferLoading:        &deferLoading,
		AllowedCallers:      []string{"code_execution_20260120"},
		EagerInputStreaming: &eager,
		InputExamples:       []any{map[string]any{"q": "weather"}},
	})

	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}

	if len(ar.Tools) != 2 {
		t.Fatalf("tools len = %d, want 2", len(ar.Tools))
	}

	custom := ar.Tools[0]
	if custom.Type != "" {
		t.Errorf(`type = %q, want "" (OpenAI "function" maps to Anthropic's default custom tool)`, custom.Type)
	}

	if custom.Strict == nil || !*custom.Strict {
		t.Errorf("strict = %v, want true", custom.Strict)
	}

	if custom.DeferLoading == nil || *custom.DeferLoading {
		t.Errorf("defer_loading = %v, want false (explicit)", custom.DeferLoading)
	}

	if custom.EagerInputStreaming == nil || !*custom.EagerInputStreaming {
		t.Errorf("eager_input_streaming = %v, want true", custom.EagerInputStreaming)
	}

	if len(custom.AllowedCallers) != 1 || custom.AllowedCallers[0] != "code_execution_20260120" {
		t.Errorf("allowed_callers = %v", custom.AllowedCallers)
	}

	if len(custom.InputExamples) != 1 {
		t.Errorf("input_examples = %v", custom.InputExamples)
	}

	if ar.Tools[1].Type != "web_search_20260209" {
		t.Errorf("built-in type = %q, want web_search_20260209", ar.Tools[1].Type)
	}

	data, err := json.Marshal(ar)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	for _, want := range []string{
		`"strict":true`,
		`"defer_loading":false`,
		`"allowed_callers":["code_execution_20260120"]`,
		`"eager_input_streaming":true`,
		`"input_examples":[{"q":"weather"}]`,
		`"type":"web_search_20260209"`,
	} {
		if !strings.Contains(string(data), want) {
			t.Errorf("marshalled tools missing %s: %s", want, data)
		}
	}
}

// TestToAnthropicRequest_ToolExtensionsOmitted verifies a plain tool serializes
// exactly as before — no new keys appear when the extensions are unset.
func TestToAnthropicRequest_ToolExtensionsOmitted(t *testing.T) {
	req := &ChatRequest{
		Model:    ModelAnthropicClaude4Sonnet,
		Messages: []Message{{Role: RoleUser, Content: NewTextContent("hi")}},
		Tools: []Tool{
			{Type: "function", Function: FunctionDefinition{Name: "lookup"}},
		},
	}

	ar, err := toAnthropicRequest(req)
	if err != nil {
		t.Fatalf("toAnthropicRequest: %v", err)
	}

	data, err := json.Marshal(ar.Tools)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	for _, unwanted := range []string{"strict", "defer_loading", "allowed_callers", "eager_input_streaming", "input_examples", `"type"`} {
		if strings.Contains(string(data), unwanted) {
			t.Errorf("unset %s should be omitted, got: %s", unwanted, data)
		}
	}
}

// TestChatRequestClone_ToolExtensions verifies Clone() duplicates each tool's
// extension map, so re-pointing a clone's tool extension cannot reach back
// into the original request.
func TestChatRequestClone_ToolExtensions(t *testing.T) {
	orig := &ChatRequest{
		Model: ModelAnthropicClaude4Sonnet,
		Tools: []Tool{
			{Function: FunctionDefinition{Name: "lookup"}},
		},
	}
	ExtendTool(&orig.Tools[0], &ToolExtension{AllowedCallers: []string{"code_execution_20260120"}})

	c := orig.Clone()

	ExtendTool(&c.Tools[0], &ToolExtension{AllowedCallers: []string{"mutated"}})

	got := ToolExtensionOf(&orig.Tools[0])
	if got == nil || len(got.AllowedCallers) != 1 || got.AllowedCallers[0] != "code_execution_20260120" {
		t.Errorf("original tool extension mutated: %+v", got)
	}
}
