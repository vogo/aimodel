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
	"reflect"
	"strings"
	"testing"
)

// TestFromAnthropicResponse_UsageExtensions decodes a real response body and
// checks the thinking tokens, server-tool counts, geography and service tier
// all reach the canonical Usage.
func TestFromAnthropicResponse_UsageExtensions(t *testing.T) {
	body := `{
		"id":"msg_usage","type":"message","role":"assistant","model":"claude-sonnet-4",
		"content":[{"type":"text","text":"done"}],
		"stop_reason":"end_turn",
		"usage":{
			"input_tokens":100,"output_tokens":40,
			"cache_creation_input_tokens":8,"cache_read_input_tokens":12,
			"output_tokens_details":{"thinking_tokens":25},
			"server_tool_use":{"web_search_requests":3,"web_fetch_requests":2},
			"inference_geo":"us","service_tier":"priority"
		}
	}`

	var ar MessagesResponse
	if err := json.Unmarshal([]byte(body), &ar); err != nil {
		t.Fatalf("decode: %v", err)
	}

	u := fromAnthropicResponse(&ar).Usage

	if u.ReasoningTokens != 25 {
		t.Errorf("reasoning_tokens = %d, want 25", u.ReasoningTokens)
	}

	uext := UsageExtensionOf(&u)
	if uext == nil || uext.ServerToolUse == nil {
		t.Fatal("server_tool_use = nil, want counts")
	}

	if uext.ServerToolUse.WebSearchRequests != 3 || uext.ServerToolUse.WebFetchRequests != 2 {
		t.Errorf("server_tool_use = %+v, want {3 2}", uext.ServerToolUse)
	}

	if uext.InferenceGeo != "us" {
		t.Errorf("inference_geo = %q, want us", uext.InferenceGeo)
	}

	if u.ServiceTier != "priority" {
		t.Errorf("service_tier = %q, want priority", u.ServiceTier)
	}

	// The pre-existing counts must not regress.
	if u.PromptTokens != 120 || u.CompletionTokens != 40 || u.TotalTokens != 160 {
		t.Errorf("token counts = %d/%d/%d, want 120/40/160", u.PromptTokens, u.CompletionTokens, u.TotalTokens)
	}
}

// TestFromAnthropicResponse_UsageExtensionsAbsent verifies a response without
// the new usage members leaves the canonical fields at their zero values and
// omits them from JSON.
func TestFromAnthropicResponse_UsageExtensionsAbsent(t *testing.T) {
	body := `{"id":"msg_plain","model":"claude-sonnet-4","content":[{"type":"text","text":"hi"}],
		"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":2}}`

	var ar MessagesResponse
	if err := json.Unmarshal([]byte(body), &ar); err != nil {
		t.Fatalf("decode: %v", err)
	}

	u := fromAnthropicResponse(&ar).Usage

	if u.ServiceTier != "" || u.ReasoningTokens != 0 {
		t.Errorf("usage = %+v, want vendor-neutral fields unset", u)
	}

	if ext := UsageExtensionOf(&u); ext != nil && (ext.ServerToolUse != nil || ext.InferenceGeo != "") {
		t.Errorf("usage extension = %+v, want absent or empty", ext)
	}

	data, err := json.Marshal(u)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	for _, unwanted := range []string{"server_tool_use", "inference_geo", "service_tier", "reasoning_tokens"} {
		if strings.Contains(string(data), unwanted) {
			t.Errorf("%s should be omitted when zero, got: %s", unwanted, data)
		}
	}
}

// TestUsage_ExtensionNotSerialized checks the Anthropic usage accounting
// stays an in-process extension: canonical Usage JSON carries none of it.
func TestUsage_ExtensionNotSerialized(t *testing.T) {
	u := Usage{PromptTokens: 10, ServiceTier: "standard"}
	u.Extensions.Set(Name, &UsageExtension{
		CacheWriteTokens: 8,
		ServerToolUse:    &ServerToolUse{WebSearchRequests: 4},
		InferenceGeo:     "eu",
	})

	data, err := json.Marshal(u)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	for _, unwanted := range []string{"server_tool_use", "inference_geo", "cache_write", "Extensions"} {
		if strings.Contains(string(data), unwanted) {
			t.Errorf("usage JSON leaked %s: %s", unwanted, data)
		}
	}
}

// TestUsage_ReasoningTokensPrecedence verifies an explicit top-level
// reasoning_tokens wins over the nested OpenAI details, matching the existing
// cached_tokens rule that Anthropic's thinking_tokens now shares.
func TestUsage_ReasoningTokensPrecedence(t *testing.T) {
	var withTop Usage
	if err := json.Unmarshal([]byte(`{"reasoning_tokens":9,"completion_tokens_details":{"reasoning_tokens":3}}`), &withTop); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if withTop.ReasoningTokens != 9 {
		t.Errorf("reasoning_tokens = %d, want the explicit top-level 9", withTop.ReasoningTokens)
	}

	var nestedOnly Usage
	if err := json.Unmarshal([]byte(`{"completion_tokens_details":{"reasoning_tokens":3}}`), &nestedOnly); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if nestedOnly.ReasoningTokens != 3 {
		t.Errorf("reasoning_tokens = %d, want the nested 3", nestedOnly.ReasoningTokens)
	}
}

// TestUsageAdd_LeavesExtensionsAlone verifies Add sums only the canonical
// counts: the per-request tier and the provider extension stay untouched.
func TestUsageAdd_LeavesExtensionsAlone(t *testing.T) {
	total := Usage{ServiceTier: "standard"}
	total.Extensions.Set(Name, &UsageExtension{ServerToolUse: &ServerToolUse{WebSearchRequests: 1}})

	other := Usage{ReasoningTokens: 7, ServiceTier: "priority"}
	other.Extensions.Set(Name, &UsageExtension{ServerToolUse: &ServerToolUse{WebSearchRequests: 2}})

	total.Add(&other)

	if total.ReasoningTokens != 7 {
		t.Errorf("reasoning_tokens = %d, want 7", total.ReasoningTokens)
	}

	if total.ServiceTier != "standard" {
		t.Errorf("service_tier = %q, want it untouched by Add", total.ServiceTier)
	}

	ext := UsageExtensionOf(&total)
	if ext == nil || ext.ServerToolUse == nil || ext.ServerToolUse.WebSearchRequests != 1 {
		t.Errorf("usage extension = %+v, want it untouched by Add", ext)
	}
}

// TestMergeAnthropicUsage verifies the merge updates exactly the fields the
// later usage object carries and leaves every other one at the baseline.
func TestMergeAnthropicUsage(t *testing.T) {
	baseline := func() MessagesUsage {
		return MessagesUsage{
			InputTokens:              100,
			OutputTokens:             1,
			CacheCreationInputTokens: 8,
			CacheReadInputTokens:     20,
			CacheCreation:            &CacheCreation{Ephemeral5mInputTokens: 8},
			OutputTokensDetails:      &OutputTokensDetails{ThinkingTokens: 5},
			ServerToolUse:            &ServerToolUse{WebSearchRequests: 2},
			InferenceGeo:             "us",
			ServiceTier:              "priority",
		}
	}

	t.Run("output tokens only preserves the rest", func(t *testing.T) {
		got := baseline()
		mergeAnthropicUsage(&got, &MessagesUsage{OutputTokens: 42})

		want := baseline()
		want.OutputTokens = 42

		if !reflect.DeepEqual(got, want) {
			t.Errorf("merged = %+v\nwant %+v", got, want)
		}
	})

	t.Run("carried fields overwrite", func(t *testing.T) {
		got := baseline()
		next := MessagesUsage{
			InputTokens:              111,
			OutputTokens:             42,
			CacheCreationInputTokens: 9,
			CacheReadInputTokens:     21,
			CacheCreation:            &CacheCreation{Ephemeral1hInputTokens: 9},
			OutputTokensDetails:      &OutputTokensDetails{ThinkingTokens: 30},
			ServerToolUse:            &ServerToolUse{WebSearchRequests: 4, WebFetchRequests: 1},
			InferenceGeo:             "eu",
			ServiceTier:              "standard",
		}

		mergeAnthropicUsage(&got, &next)

		if !reflect.DeepEqual(got, next) {
			t.Errorf("merged = %+v\nwant %+v", got, next)
		}
	})

	t.Run("empty terminal usage changes nothing", func(t *testing.T) {
		got := baseline()
		mergeAnthropicUsage(&got, &MessagesUsage{})

		if !reflect.DeepEqual(got, baseline()) {
			t.Errorf("merged = %+v, want the untouched baseline", got)
		}
	})
}

// TestFromAnthropicResponse_Container verifies the container is surfaced on
// ChatResponse and stays nil when absent or null.
func TestFromAnthropicResponse_Container(t *testing.T) {
	tests := []struct {
		name          string
		body          string
		wantID        string
		wantExpiresAt string
	}{
		{
			name: "present",
			body: `{"id":"m","model":"claude-sonnet-4","content":[],"stop_reason":"end_turn",
				"usage":{"input_tokens":1,"output_tokens":1},
				"container":{"id":"container_abc","expires_at":"2026-07-21T10:00:00Z"}}`,
			wantID:        "container_abc",
			wantExpiresAt: "2026-07-21T10:00:00Z",
		},
		{
			name: "null",
			body: `{"id":"m","model":"claude-sonnet-4","content":[],"stop_reason":"end_turn",
				"usage":{"input_tokens":1,"output_tokens":1},"container":null}`,
		},
		{
			name: "absent",
			body: `{"id":"m","model":"claude-sonnet-4","content":[],"stop_reason":"end_turn",
				"usage":{"input_tokens":1,"output_tokens":1}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var ar MessagesResponse
			if err := json.Unmarshal([]byte(tt.body), &ar); err != nil {
				t.Fatalf("decode: %v", err)
			}

			cr := fromAnthropicResponse(&ar)

			var got *ResponseContainer
			if ext := ResponseExtensionOf(cr); ext != nil {
				got = ext.Container
			}

			if tt.wantID == "" {
				if got != nil {
					t.Errorf("container = %+v, want nil", got)
				}

				return
			}

			if got == nil {
				t.Fatal("container = nil, want a value")
			}

			if got.ID != tt.wantID || got.ExpiresAt != tt.wantExpiresAt {
				t.Errorf("container = %+v, want {%s %s}", got, tt.wantID, tt.wantExpiresAt)
			}
		})
	}
}
