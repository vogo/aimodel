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
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

// completeResponseFixture is an offline /v1/responses body covering every
// output item variant this SDK models, the three hosted tools, an item type it
// does not model, and the full usage breakdown.
const completeResponseFixture = `{
  "id": "resp_1",
  "object": "response",
  "created_at": 1786000000,
  "completed_at": 1786000009,
  "status": "completed",
  "model": "gpt-5",
  "instructions": "be brief",
  "previous_response_id": "resp_0",
  "conversation": {"id": "conv_1"},
  "parallel_tool_calls": true,
  "service_tier": "default",
  "truncation": "disabled",
  "reasoning": {"effort": "high", "summary": "auto"},
  "text": {"verbosity": "low", "format": {"type": "json_schema", "name": "answer", "schema": {"type": "object"}, "strict": true}},
  "tool_choice": "auto",
  "tools": [
    {"type": "web_search", "search_context_size": "high"},
    {"type": "mcp", "server_label": "deepwiki", "server_url": "https://example.invalid/mcp"}
  ],
  "output": [
    {
      "id": "rs_1",
      "type": "reasoning",
      "summary": [{"type": "summary_text", "text": "thinking"}],
      "content": [{"type": "reasoning_text", "text": "step one"}],
      "encrypted_content": "enc",
      "status": "completed"
    },
    {
      "id": "ws_1",
      "type": "web_search_call",
      "status": "completed",
      "action": {"type": "search", "query": "weather", "queries": ["weather"], "sources": [{"type": "url", "url": "https://example.invalid/a"}]}
    },
    {
      "id": "fs_1",
      "type": "file_search_call",
      "status": "completed",
      "queries": ["policy"],
      "results": [{"file_id": "file-1", "filename": "policy.md", "score": 0.5, "text": "chunk", "attributes": {"team": "ops"}}]
    },
    {
      "id": "ci_1",
      "type": "code_interpreter_call",
      "status": "completed",
      "container_id": "cntr_1",
      "code": "print(1)",
      "outputs": [{"type": "logs", "logs": "1"}, {"type": "image", "url": "https://example.invalid/i.png"}]
    },
    {
      "id": "fc_1",
      "type": "function_call",
      "call_id": "call_1",
      "name": "get_weather",
      "arguments": "{\"city\":\"SF\"}",
      "status": "completed"
    },
    {
      "id": "fco_1",
      "type": "function_call_output",
      "call_id": "call_1",
      "output": "18C",
      "status": "completed"
    },
    {
      "id": "mcp_1",
      "type": "mcp_call",
      "server_label": "deepwiki",
      "name": "ask",
      "arguments": "{}",
      "output": "answer"
    },
    {
      "id": "msg_1",
      "type": "message",
      "role": "assistant",
      "status": "completed",
      "content": [
        {"type": "output_text", "text": "It is ", "annotations": [{"type": "file_citation", "file_id": "file-1", "filename": "policy.md", "index": 0}]},
        {"type": "output_text", "text": "18C.", "annotations": [{"type": "url_citation", "url": "https://example.invalid/a", "title": "A", "start_index": 0, "end_index": 4}], "logprobs": [{"token": "18C", "logprob": -0.5, "bytes": [49], "top_logprobs": [{"token": "19C", "logprob": -1.5}]}]},
        {"type": "refusal", "refusal": "no"}
      ]
    }
  ],
  "usage": {
    "input_tokens": 30,
    "input_tokens_details": {"cached_tokens": 12, "cache_write_tokens": 4},
    "output_tokens": 25,
    "output_tokens_details": {"reasoning_tokens": 9},
    "total_tokens": 55
  }
}`

// jsonValue normalizes a JSON document so round-trip comparisons ignore key
// ordering differences between struct fields and decoded maps.
func jsonValue(t *testing.T, data []byte) any {
	t.Helper()

	var value any
	if err := json.Unmarshal(data, &value); err != nil {
		t.Fatalf("normalize %s: %v", data, err)
	}

	return value
}

func TestResponsesRequestRoundTripsEveryDocumentedUnion(t *testing.T) {
	request := &ResponsesRequest{
		Model: "gpt-5",
		Input: NewResponseItemsInput(
			NewResponseInputMessage(ResponseRoleDeveloper, "be brief"),
			ResponseInputItem{Type: ResponseItemTypeMessage, Message: &ResponseInputMessage{
				Type: ResponseItemTypeMessage, Role: ResponseRoleUser,
				Content: NewResponseContentParts(
					ResponseContent{Type: ResponseContentTypeInputText, Text: "hi", PromptCacheBreakpoint: &ResponsePromptCacheBreakpoint{Mode: "explicit"}},
					ResponseContent{Type: ResponseContentTypeInputImage, ImageURL: "https://example.invalid/i.png", Detail: "high"},
					ResponseContent{Type: ResponseContentTypeInputFile, FileID: "file-1", Filename: "a.pdf", FileData: "ZGF0YQ==", FileURL: "https://example.invalid/a.pdf", Detail: "low"},
					ResponseContent{Type: ResponseContentTypeInputAudio, InputAudio: &ResponseInputAudio{Data: "ZGF0YQ==", Format: "wav"}},
				),
			}},
			ResponseInputItem{Type: ResponseItemTypeFunctionCall, FunctionCall: &ResponseFunctionToolCall{
				Type: ResponseItemTypeFunctionCall, ID: "fc_1", CallID: "call_1", Name: "get_weather",
				Arguments: `{"city":"SF"}`, Namespace: "tools", Status: "completed",
				Caller: &ResponseCaller{Type: "program", CallerID: "prog_1"},
			}},
			ResponseInputItem{Type: ResponseItemTypeFunctionCallOutput, FunctionCallOutput: &ResponseFunctionCallOutput{
				Type: ResponseItemTypeFunctionCallOutput, CallID: "call_1", Status: "completed", CreatedBy: "app",
				Output: NewResponseContentParts(ResponseContent{Type: ResponseContentTypeInputText, Text: "18C"}),
			}},
			ResponseInputItem{Type: ResponseItemTypeReasoning, Reasoning: &ResponseReasoningItem{
				Type: ResponseItemTypeReasoning, ID: "rs_1", EncryptedContent: "enc",
				Summary: []ResponseReasoningSummary{{Type: ResponseContentTypeSummaryText, Text: "s"}},
				Content: []ResponseContent{{Type: ResponseContentTypeReasoningText, Text: "r"}},
			}},
			ResponseInputItem{Type: ResponseItemTypeItemReference, ItemReference: &ResponseItemReference{Type: ResponseItemTypeItemReference, ID: "msg_1"}},
			// An item type the SDK does not model must survive verbatim.
			ResponseInputItem{Type: "computer_call", Raw: json.RawMessage(`{"type":"computer_call","call_id":"c1","action":{"type":"click","x":1,"y":2}}`)},
		),
		Background:         new(true),
		ContextManagement:  []ResponseContextManagement{{Type: "compaction", CompactThreshold: new(2048)}},
		Conversation:       &ResponseConversation{ID: "conv_1"},
		Include:            []string{ResponseIncludeFileSearchResults, ResponseIncludeCodeInterpreterOutputs, ResponseIncludeReasoningEncryptedContent},
		Instructions:       "be brief",
		MaxOutputTokens:    new(512),
		MaxToolCalls:       new(4),
		Metadata:           map[string]string{"trace": "x"},
		Moderation:         &ResponseModerationConfig{Model: "omni-moderation-latest", Policy: &ResponseModerationPolicy{Input: &ResponseModerationPolicySide{Mode: "block"}, Output: &ResponseModerationPolicySide{Mode: "score"}}},
		ParallelToolCalls:  new(true),
		PreviousResponseID: "resp_0",
		Prompt:             &ResponsePrompt{ID: "pmpt_1", Version: "3", Variables: map[string]any{"city": "SF"}},
		PromptCacheKey:     "cache-key",
		PromptCacheOptions: &ResponsePromptCacheOptions{Mode: "explicit", TTL: "30m"},
		//nolint:staticcheck // superseded by prompt_cache_options.ttl upstream, still on the wire.
		PromptCacheRetention: "24h",
		Reasoning:            &ResponseReasoningConfig{Context: "all_turns", Effort: "high", Mode: "standard", Summary: "auto", GenerateSummary: "concise"},
		SafetyIdentifier:     "user-hash",
		ServiceTier:          "priority",
		Store:                new(false),
		StreamOptions:        &ResponseStreamOptions{IncludeObfuscation: new(false)},
		Temperature:          new(0.2),
		Text:                 &ResponseTextConfig{Verbosity: "low", Format: &ResponseTextFormat{Type: "json_schema", Name: "answer", Description: "d", Schema: map[string]any{"type": "object"}, Strict: new(true)}},
		ToolChoice:           ResponseToolChoiceFunction{Type: ResponseToolTypeFunction, Name: "get_weather"},
		Tools: []ResponseTool{
			{Type: ResponseToolTypeFunction, Name: "get_weather", Description: "weather", Parameters: map[string]any{"type": "object"}, Strict: new(true), OutputSchema: map[string]any{"type": "string"}, AllowedCallers: []string{"direct"}, DeferLoading: new(false)},
			{Type: ResponseToolTypeFileSearch, VectorStoreIDs: []string{"vs_1"}, MaxNumResults: new(5), Filters: map[string]any{"key": "team", "type": "eq", "value": "ops"}, RankingOptions: &ResponseFileSearchRankingOptions{Ranker: "auto", ScoreThreshold: new(0.4), HybridSearch: &ResponseFileSearchHybridSearch{EmbeddingWeight: 0.7, TextWeight: 0.3}}},
			{Type: ResponseToolTypeWebSearch, SearchContextSize: "medium", Filters: map[string]any{"allowed_domains": []any{"example.invalid"}}, UserLocation: &ResponseWebSearchUserLocation{Type: "approximate", City: "SF", Country: "US", Region: "CA", Timezone: "America/Los_Angeles"}},
			{Type: ResponseToolTypeCodeInterpreter, Container: ResponseCodeInterpreterContainerAuto{Type: "auto", FileIDs: []string{"file-1"}, MemoryLimit: "1g"}},
			// A tool type the SDK does not model must survive verbatim.
			{Type: "mcp", Raw: json.RawMessage(`{"type":"mcp","server_label":"deepwiki","server_url":"https://example.invalid/mcp"}`)},
		},
		TopLogprobs: new(3),
		TopP:        new(0.9),
		Truncation:  ResponseTruncationAuto,
		User:        "user-1",
	}

	encoded, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}

	var decoded ResponsesRequest
	if err = json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}

	reencoded, err := json.Marshal(&decoded)
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(jsonValue(t, encoded), jsonValue(t, reencoded)) {
		t.Fatalf("request lost fields on round trip:\nfirst  = %s\nsecond = %s", encoded, reencoded)
	}

	for _, field := range []string{
		"background", "context_management", "conversation", "include", "instructions", "max_output_tokens",
		"max_tool_calls", "metadata", "moderation", "parallel_tool_calls", "previous_response_id", "prompt",
		"prompt_cache_key", "prompt_cache_options", "prompt_cache_retention", "reasoning", "safety_identifier",
		"service_tier", "store", "stream_options", "temperature", "text", "tool_choice", "tools", "top_logprobs",
		"top_p", "truncation", "user", "input_audio", "prompt_cache_breakpoint", "vector_store_ids",
		"ranking_options", "hybrid_search", "user_location", "container", "item_reference", "computer_call",
		"server_label",
	} {
		if !strings.Contains(string(encoded), `"`+field+`"`) {
			t.Errorf("missing %s in %s", field, encoded)
		}
	}

	if decoded.Input.Items()[0].Message.Content.Text() != "be brief" {
		t.Errorf("scalar message content lost: %+v", decoded.Input.Items()[0].Message)
	}
}

func TestResponsesRequestScalarInputAndConversationString(t *testing.T) {
	encoded, err := json.Marshal(&ResponsesRequest{Model: "gpt-5", Input: NewResponseTextInput("hi")})
	if err != nil {
		t.Fatal(err)
	}

	if string(encoded) != `{"model":"gpt-5","input":"hi"}` {
		t.Fatalf("scalar input encoded as %s", encoded)
	}

	var decoded ResponsesRequest
	if err = json.Unmarshal([]byte(`{"input":"hi","conversation":"conv_9"}`), &decoded); err != nil {
		t.Fatal(err)
	}

	if decoded.Input.Text() != "hi" || decoded.Input.Items() != nil {
		t.Errorf("scalar input decoded as %+v", decoded.Input)
	}

	if decoded.Conversation.ID != "conv_9" {
		t.Errorf("bare conversation id decoded as %+v", decoded.Conversation)
	}
}

func TestResponseDecodesEveryOutputItemAndUsageDetail(t *testing.T) {
	var response Response
	if err := json.Unmarshal([]byte(completeResponseFixture), &response); err != nil {
		t.Fatal(err)
	}

	if response.OutputText != "It is 18C." {
		t.Errorf("OutputText = %q", response.OutputText)
	}

	wantOrder := []string{
		ResponseItemTypeReasoning, ResponseItemTypeWebSearchCall, ResponseItemTypeFileSearchCall,
		ResponseItemTypeCodeInterpreterCall, ResponseItemTypeFunctionCall, ResponseItemTypeFunctionCallOutput,
		"mcp_call", ResponseItemTypeMessage,
	}

	if len(response.Output) != len(wantOrder) {
		t.Fatalf("output length = %d", len(response.Output))
	}

	for i, want := range wantOrder {
		if response.Output[i].Type != want {
			t.Fatalf("output[%d].Type = %q, want %q", i, response.Output[i].Type, want)
		}
	}

	reasoning := response.Output[0].Reasoning
	if reasoning.Summary[0].Text != "thinking" || reasoning.Content[0].Text != "step one" || reasoning.EncryptedContent != "enc" {
		t.Errorf("reasoning item = %+v", reasoning)
	}

	search := response.Output[1].WebSearchCall
	if search.Action.Query != "weather" || search.Action.Sources[0].URL != "https://example.invalid/a" {
		t.Errorf("web search call = %+v", search.Action)
	}

	files := response.Output[2].FileSearchCall
	if files.Results[0].FileID != "file-1" || *files.Results[0].Score != 0.5 || files.Results[0].Attributes["team"] != "ops" {
		t.Errorf("file search call = %+v", files.Results)
	}

	code := response.Output[3].CodeInterpreterCall
	if code.ContainerID != "cntr_1" || code.Code != "print(1)" || code.Outputs[0].Logs != "1" || code.Outputs[1].URL == "" {
		t.Errorf("code interpreter call = %+v", code)
	}

	if call := response.Output[4].FunctionCall; call.CallID != "call_1" || call.Arguments != `{"city":"SF"}` {
		t.Errorf("function call = %+v", call)
	}

	if output := response.Output[5].FunctionCallOutput; output.CallID != "call_1" || output.Output.Text() != "18C" {
		t.Errorf("function call output = %+v", output)
	}

	// An item type the SDK does not model keeps its discriminator and payload.
	if unknown := response.Output[6]; unknown.Message != nil || !strings.Contains(string(unknown.Raw), `"server_label"`) {
		t.Errorf("unmodeled item lost its payload: %+v", unknown)
	}

	message := response.Output[7].Message
	if len(message.Content) != 3 || message.Content[2].Refusal != "no" {
		t.Fatalf("message content = %+v", message.Content)
	}

	if *message.Content[0].Annotations[0].Index != 0 || message.Content[1].Annotations[0].URL == "" {
		t.Errorf("annotations = %+v", message.Content[0].Annotations)
	}

	if message.Content[1].Logprobs[0].TopLogprobs[0].Token != "19C" {
		t.Errorf("logprobs = %+v", message.Content[1].Logprobs)
	}

	if response.Usage.InputTokensDetails.CachedTokens != 12 || response.Usage.InputTokensDetails.CacheWriteTokens != 4 ||
		response.Usage.OutputTokensDetails.ReasoningTokens != 9 || response.Usage.TotalTokens != 55 {
		t.Errorf("usage = %+v", response.Usage)
	}

	if response.Instructions.Text() != "be brief" || response.Conversation.ID != "conv_1" {
		t.Errorf("instructions/conversation = %+v %+v", response.Instructions, response.Conversation)
	}

	// The unmodeled `mcp` tool definition round-trips verbatim.
	if len(response.Tools) != 2 || !strings.Contains(string(response.Tools[1].Raw), "deepwiki") {
		t.Fatalf("tools = %+v", response.Tools)
	}

	reencoded, err := json.Marshal(&response)
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(jsonValue(t, []byte(completeResponseFixture)), jsonValue(t, reencoded)) {
		t.Fatalf("response lost fields on round trip: %s", reencoded)
	}
}

func TestResponseKeepsFailedAndIncompleteDetails(t *testing.T) {
	const fixture = `{"id":"resp_2","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},
	"error":{"code":"server_error","message":"boom"},"output":[]}`

	var response Response
	if err := json.Unmarshal([]byte(fixture), &response); err != nil {
		t.Fatal(err)
	}

	if response.Status != ResponseStatusIncomplete || response.IncompleteDetails.Reason != "max_output_tokens" ||
		response.Error.Code != "server_error" || response.OutputText != "" {
		t.Fatalf("response = %+v", response)
	}
}

func TestResponsesNonStreamingCall(t *testing.T) {
	var seen ResponsesRequest

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/responses" {
			t.Errorf("request = %s %s", r.Method, r.URL.Path)
		}

		if r.Header.Get("Authorization") != "Bearer key" || r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("headers = %v", r.Header)
		}

		if err := json.NewDecoder(r.Body).Decode(&seen); err != nil {
			t.Error(err)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, completeResponseFixture)
	}))
	defer server.Close()

	client := NewClient("key", WithBaseURL(server.URL+"/v1/"), WithHTTPClient(server.Client()))
	// Stream is deliberately true: the non-streaming call must force it off on a
	// copy without touching the caller's request.
	input := &ResponsesRequest{Model: "gpt-5", Stream: true, Input: NewResponseTextInput("hi")}

	response, err := client.Responses(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}

	if response.ID != "resp_1" || response.OutputText != "It is 18C." || response.Usage.OutputTokensDetails.ReasoningTokens != 9 {
		t.Fatalf("response = %+v", response)
	}

	if seen.Stream {
		t.Error("non-streaming call sent stream=true")
	}

	if !input.Stream || input.Input.Text() != "hi" {
		t.Errorf("client mutated the caller request: %+v", input)
	}
}

func TestResponsesRejectsNilRequest(t *testing.T) {
	client := NewClient("key")

	if _, err := client.Responses(context.Background(), nil); err == nil {
		t.Error("Responses accepted a nil request")
	}

	if _, err := client.ResponsesStream(context.Background(), nil); err == nil {
		t.Error("ResponsesStream accepted a nil request")
	}
}

func TestResponsesErrorBodies(t *testing.T) {
	for _, tc := range []struct {
		name, body  string
		status      int
		wantCode    string
		wantMessage string
	}{
		{name: "structured", status: http.StatusBadRequest, wantCode: "invalid_value", wantMessage: "bad input", body: `{"error":{"code":"invalid_value","type":"invalid_request_error","message":"bad input"}}`},
		{name: "unstructured", status: http.StatusBadGateway, wantMessage: "upstream down", body: "upstream down"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tc.status)
				_, _ = io.WriteString(w, tc.body)
			}))
			defer server.Close()

			client := NewClient("key", WithBaseURL(server.URL), WithHTTPClient(server.Client()))

			_, err := client.Responses(context.Background(), &ResponsesRequest{Model: "gpt-5"})

			var httpErr *HTTPError
			if !errors.As(err, &httpErr) || httpErr.StatusCode() != tc.status || httpErr.Code != tc.wantCode || httpErr.Message != tc.wantMessage {
				t.Fatalf("error = %T %+v", err, err)
			}

			if string(httpErr.Body) != tc.body {
				t.Errorf("error body = %s", httpErr.Body)
			}

			// A stream that never starts must fail the same way and leak nothing.
			if _, err = client.ResponsesStream(context.Background(), &ResponsesRequest{Model: "gpt-5"}); !errors.As(err, &httpErr) {
				t.Fatalf("stream setup error = %T %+v", err, err)
			}
		})
	}
}

// sseServer serves a fixed SSE body for one streaming Responses call.
func sseServer(t *testing.T, body string, seen *ResponsesRequest) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if seen != nil {
			if err := json.NewDecoder(r.Body).Decode(seen); err != nil {
				t.Error(err)
			}
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, body)
	}))
}

func TestResponsesStreamDispatchesEveryBaselineEvent(t *testing.T) {
	types := ResponseStreamEventTypes()

	var body strings.Builder

	var want []string

	for i, eventType := range types {
		if eventType == ResponseEventError {
			continue // the error event terminates the stream; covered separately
		}

		want = append(want, eventType)
		fmt.Fprintf(&body, "event: %s\ndata: {\"type\":%q,\"sequence_number\":%d}\n\n", eventType, eventType, i)
	}

	var seen ResponsesRequest

	server := sseServer(t, body.String(), &seen)
	defer server.Close()

	client := NewClient("key", WithBaseURL(server.URL), WithHTTPClient(server.Client()))
	input := &ResponsesRequest{Model: "gpt-5", Input: NewResponseTextInput("hi")}

	stream, err := client.ResponsesStream(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}

	defer func() { _ = stream.Close() }()

	if !seen.Stream {
		t.Error("streaming call did not send stream=true")
	}

	if input.Stream {
		t.Error("streaming call mutated the caller request")
	}

	for index, wantType := range want {
		event, recvErr := stream.Recv()
		if recvErr != nil {
			t.Fatalf("Recv %s: %v", wantType, recvErr)
		}

		if event.Type != wantType {
			t.Fatalf("event %d type = %q, want %q", index, event.Type, wantType)
		}

		if event.SequenceNumber != indexOf(types, wantType) {
			t.Errorf("%s sequence_number = %d", wantType, event.SequenceNumber)
		}

		if len(event.Raw) == 0 {
			t.Errorf("%s lost its raw payload", wantType)
		}
	}

	if _, err = stream.Recv(); !errors.Is(err, io.EOF) {
		t.Fatalf("terminal Recv error = %v", err)
	}
}

func indexOf(values []string, want string) int {
	for i, value := range values {
		if value == want {
			return i
		}
	}

	return -1
}

func TestResponsesStreamTypedPayloads(t *testing.T) {
	body := strings.Join([]string{
		": heartbeat",
		"",
		"event: response.created",
		`data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_1","status":"in_progress","output":[]}}`,
		"",
		"event: response.output_item.added",
		`data: {"type":"response.output_item.added","sequence_number":1,"output_index":0,`,
		`data:  "item":{"id":"ws_1","type":"web_search_call","status":"in_progress","action":{"type":"search","query":"weather"}}}`,
		"",
		"event: response.web_search_call.searching",
		`data: {"type":"response.web_search_call.searching","sequence_number":2,"item_id":"ws_1","output_index":0}`,
		"",
		"event: response.file_search_call.completed",
		`data: {"type":"response.file_search_call.completed","sequence_number":3,"item_id":"fs_1","output_index":1}`,
		"",
		"event: response.code_interpreter_call_code.delta",
		`data: {"type":"response.code_interpreter_call_code.delta","sequence_number":4,"item_id":"ci_1","output_index":2,"delta":"print("}`,
		"",
		"event: response.code_interpreter_call_code.done",
		`data: {"type":"response.code_interpreter_call_code.done","sequence_number":5,"item_id":"ci_1","output_index":2,"code":"print(1)"}`,
		"",
		"event: response.content_part.added",
		`data: {"type":"response.content_part.added","sequence_number":6,"item_id":"msg_1","output_index":3,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}`,
		"",
		"event: response.output_text.delta",
		`data: {"type":"response.output_text.delta","sequence_number":7,"item_id":"msg_1","output_index":3,"content_index":0,"delta":"18C","obfuscation":"xyz","logprobs":[{"token":"18C","logprob":-0.5}]}`,
		"",
		"event: response.output_text.annotation.added",
		`data: {"type":"response.output_text.annotation.added","sequence_number":8,"item_id":"msg_1","output_index":3,"content_index":0,"annotation_index":0,"annotation":{"type":"url_citation","url":"https://example.invalid/a"}}`,
		"",
		"event: response.reasoning_summary_part.done",
		`data: {"type":"response.reasoning_summary_part.done","sequence_number":9,"item_id":"rs_1","output_index":4,"summary_index":1,"status":"incomplete","part":{"type":"summary_text","text":"done"}}`,
		"",
		"event: response.function_call_arguments.done",
		`data: {"type":"response.function_call_arguments.done","sequence_number":10,"item_id":"fc_1","output_index":5,"name":"get_weather","arguments":"{}"}`,
		"",
		// An event type outside the documented baseline is preserved, not rejected.
		"event: response.brand_new_thing",
		`data: {"type":"response.brand_new_thing","sequence_number":11,"unknown_field":{"nested":true}}`,
		"",
		// An event whose payload carries no `type` falls back to the SSE name.
		"event: response.completed",
		`data: {"sequence_number":12,"response":{"id":"resp_1","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"18C"}]}],"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}}`,
		"",
		"",
	}, "\n")

	server := sseServer(t, body, nil)
	defer server.Close()

	stream, err := NewClient("key", WithBaseURL(server.URL), WithHTTPClient(server.Client())).
		ResponsesStream(context.Background(), &ResponsesRequest{Model: "gpt-5"})
	if err != nil {
		t.Fatal(err)
	}

	events := make(map[string]*ResponseStreamEvent)

	for {
		event, recvErr := stream.Recv()
		if errors.Is(recvErr, io.EOF) {
			break
		}

		if recvErr != nil {
			t.Fatal(recvErr)
		}

		events[event.Type] = event
	}

	created := events[ResponseEventCreated]
	if created == nil || created.Response.ID != "resp_1" || created.Response.Status != ResponseStatusInProgress {
		t.Fatalf("created = %+v", created)
	}

	// Multi-line data payloads are joined before decoding.
	added := events[ResponseEventOutputItemAdded]
	if added == nil || added.Item.WebSearchCall == nil || added.Item.WebSearchCall.Action.Query != "weather" {
		t.Fatalf("output_item.added = %+v", added)
	}

	if searching := events[ResponseEventWebSearchCallSearching]; searching == nil || searching.ItemID != "ws_1" {
		t.Fatalf("web_search_call.searching = %+v", events[ResponseEventWebSearchCallSearching])
	}

	if done := events[ResponseEventFileSearchCallCompleted]; done == nil || done.OutputIndex != 1 {
		t.Fatalf("file_search_call.completed = %+v", events[ResponseEventFileSearchCallCompleted])
	}

	if delta := events[ResponseEventCodeInterpreterCallCodeDelta]; delta == nil || delta.Delta != "print(" {
		t.Fatalf("code delta = %+v", events[ResponseEventCodeInterpreterCallCodeDelta])
	}

	if code := events[ResponseEventCodeInterpreterCallCodeDone]; code == nil || code.Code != "print(1)" {
		t.Fatalf("code done = %+v", events[ResponseEventCodeInterpreterCallCodeDone])
	}

	if part := events[ResponseEventContentPartAdded]; part == nil || part.Part.Type != ResponseContentTypeOutputText {
		t.Fatalf("content_part.added = %+v", events[ResponseEventContentPartAdded])
	}

	text := events[ResponseEventOutputTextDelta]
	if text == nil || text.Delta != "18C" || text.ContentIndex != 0 || text.Obfuscation != "xyz" || text.Logprobs[0].Token != "18C" {
		t.Fatalf("output_text.delta = %+v", text)
	}

	annotation := events[ResponseEventOutputTextAnnotationAdded]
	if annotation == nil || annotation.AnnotationIndex != 0 || !strings.Contains(string(annotation.Annotation), "url_citation") {
		t.Fatalf("annotation.added = %+v", annotation)
	}

	summary := events[ResponseEventReasoningSummaryPartDone]
	if summary == nil || summary.SummaryIndex != 1 || summary.Status != "incomplete" || summary.Part.Text != "done" {
		t.Fatalf("reasoning_summary_part.done = %+v", summary)
	}

	arguments := events[ResponseEventFunctionCallArgumentsDone]
	if arguments == nil || arguments.Name != "get_weather" || arguments.Arguments != "{}" {
		t.Fatalf("function_call_arguments.done = %+v", arguments)
	}

	unknown := events["response.brand_new_thing"]
	if unknown == nil || !strings.Contains(string(unknown.Raw), `"unknown_field"`) || unknown.SequenceNumber != 11 {
		t.Fatalf("unknown event = %+v", unknown)
	}

	completed := events[ResponseEventCompleted]
	if completed == nil || completed.Response.OutputText != "18C" || completed.Response.Usage.TotalTokens != 3 {
		t.Fatalf("completed = %+v", completed)
	}

	// Close is idempotent.
	if err = stream.Close(); err != nil {
		t.Fatal(err)
	}

	if err = stream.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestResponsesStreamErrorEvent(t *testing.T) {
	server := sseServer(t, "event: error\ndata: {\"type\":\"error\",\"sequence_number\":3,\"code\":\"rate_limit_exceeded\",\"message\":\"slow down\",\"param\":null}\n\n", nil)
	defer server.Close()

	stream, err := NewClient("key", WithBaseURL(server.URL), WithHTTPClient(server.Client())).
		ResponsesStream(context.Background(), &ResponsesRequest{Model: "gpt-5"})
	if err != nil {
		t.Fatal(err)
	}

	_, err = stream.Recv()

	var httpErr *HTTPError
	if !errors.As(err, &httpErr) || httpErr.Code != "rate_limit_exceeded" || httpErr.Message != "slow down" {
		t.Fatalf("error = %T %+v", err, err)
	}

	if !strings.Contains(string(httpErr.Body), "rate_limit_exceeded") {
		t.Errorf("error body = %s", httpErr.Body)
	}

	// The body is already closed; Close stays safe and quiet.
	if err = stream.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestResponsesStreamMalformedKnownEvent(t *testing.T) {
	server := sseServer(t, "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":\"not-an-object\"}\n\n", nil)
	defer server.Close()

	stream, err := NewClient("key", WithBaseURL(server.URL), WithHTTPClient(server.Client())).
		ResponsesStream(context.Background(), &ResponsesRequest{Model: "gpt-5"})
	if err != nil {
		t.Fatal(err)
	}

	if _, err = stream.Recv(); err == nil || !strings.Contains(err.Error(), ResponseEventCompleted) {
		t.Fatalf("error = %v", err)
	}

	if err = stream.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestResponsesStreamInvalidJSON(t *testing.T) {
	server := sseServer(t, "data: {not json}\n\n", nil)
	defer server.Close()

	stream, err := NewClient("key", WithBaseURL(server.URL), WithHTTPClient(server.Client())).
		ResponsesStream(context.Background(), &ResponsesRequest{Model: "gpt-5"})
	if err != nil {
		t.Fatal(err)
	}

	if _, err = stream.Recv(); err == nil || !strings.Contains(err.Error(), "decode responses stream event") {
		t.Fatalf("error = %v", err)
	}
}

func TestResponsesStreamScanFailure(t *testing.T) {
	// One SSE line beyond the scanner's 1 MB limit must surface as a read
	// failure rather than a truncated, valid-looking event.
	oversized := "data: {\"type\":\"response.output_text.delta\",\"delta\":\"" + strings.Repeat("a", (1<<20)+1) + "\"}\n\n"

	server := sseServer(t, oversized, nil)
	defer server.Close()

	stream, err := NewClient("key", WithBaseURL(server.URL), WithHTTPClient(server.Client())).
		ResponsesStream(context.Background(), &ResponsesRequest{Model: "gpt-5"})
	if err != nil {
		t.Fatal(err)
	}

	if _, err = stream.Recv(); err == nil || !strings.Contains(err.Error(), "read responses stream") {
		t.Fatalf("error = %v", err)
	}
}
