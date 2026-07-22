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
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestNativeRequestCoversOpenAIOnlyFieldsAndExplicitZeroes(t *testing.T) {
	request := ChatCompletionRequest{
		Model: "gpt-5", Messages: []ChatCompletionMessage{{Role: "user", Content: NewPartsContent(
			ChatCompletionContentPart{Type: "input_audio", InputAudio: &InputAudio{Data: "abc", Format: "wav"}},
			ChatCompletionContentPart{Type: "file", File: &InputFile{FileID: "file-1"}},
		)}},
		Audio: &AudioConfig{Format: "mp3", Voice: "alloy"}, FrequencyPenalty: new(0.0), PresencePenalty: new(0.0),
		Seed: new(int64(0)), User: "user-1", StreamOptions: &StreamOptions{IncludeUsage: new(false), IncludeObfuscation: new(false)},
		Verbosity: "low", Logprobs: new(false), TopLogprobs: new(0), LogitBias: map[string]int{"1": -5},
		Store: new(false), Metadata: map[string]string{"trace": "x"}, PromptCacheKey: "cache-key",
		Modalities: []string{"text", "audio"}, N: new(0), Prediction: map[string]any{"type": "content", "content": "x"},
		ServiceTier: "priority", SafetyIdentifier: "safe-user", FunctionCall: "auto",
		Functions: []ChatCompletionFunction{{Name: "legacy"}}, WebSearchOptions: &WebSearchOptions{SearchContextSize: "low"},
	}
	body, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"audio", "frequency_penalty", "presence_penalty", "seed", "user", "stream_options", "verbosity", "logprobs", "top_logprobs", "logit_bias", "store", "metadata", "prompt_cache_key", "modalities", "n", "prediction", "service_tier", "input_audio", "file", "function_call", "functions", "safety_identifier", "web_search_options"} {
		if !strings.Contains(string(body), `"`+field+`"`) {
			t.Errorf("missing %s in %s", field, body)
		}
	}
}

func TestNativeResponseDecodesExclusiveNestedFields(t *testing.T) {
	const fixture = `{"id":"chat-1","service_tier":"priority","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi","audio":{"id":"a","data":"ZGF0YQ==","expires_at":9,"transcript":"hi"}},"logprobs":{"content":[{"token":"hi","logprob":-0.1,"bytes":[104,105],"top_logprobs":[{"token":"hey","logprob":-1}]}]}}],"usage":{"prompt_tokens":4,"completion_tokens":3,"total_tokens":7,"prompt_tokens_details":{"cached_tokens":2,"audio_tokens":1},"completion_tokens_details":{"reasoning_tokens":1,"audio_tokens":1,"accepted_prediction_tokens":1,"rejected_prediction_tokens":0}}}`
	var response ChatCompletionResponse
	if err := json.Unmarshal([]byte(fixture), &response); err != nil {
		t.Fatal(err)
	}
	if response.Choices[0].Message.Audio.Transcript != "hi" || response.Choices[0].Logprobs.Content[0].TopLogprobs[0].Token != "hey" {
		t.Fatalf("response lost native fields: %+v", response)
	}
	if response.Usage.PromptTokensDetails.CachedTokens != 2 || response.Usage.CompletionTokensDetails.ReasoningTokens != 1 {
		t.Fatalf("usage details lost: %+v", response.Usage)
	}
}

func TestNativeClientNonStreamAndStream(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.URL.Path != "/v1/chat/completions" || r.Header.Get("Authorization") != "Bearer key" {
			t.Errorf("request = %s auth=%q", r.URL.Path, r.Header.Get("Authorization"))
		}
		var request ChatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
		}
		w.Header().Set("Content-Type", "application/json")
		if request.Stream {
			_, _ = io.WriteString(w, "data: {\"id\":\"chunk-1\",\"choices\":[]}\n\ndata: [DONE]\n\n")
			return
		}
		_, _ = io.WriteString(w, `{"id":"chat-1","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()
	client := NewClient("key", WithBaseURL(server.URL+"/v1/"), WithHTTPClient(server.Client()))
	input := &ChatCompletionRequest{Model: "m", Stream: true, Messages: []ChatCompletionMessage{{Role: "user", Content: NewTextContent("hi")}}}
	response, err := client.ChatCompletions(context.Background(), input)
	if err != nil || response.ID != "chat-1" {
		t.Fatalf("response=%+v err=%v", response, err)
	}
	if !input.Stream {
		t.Fatal("native client modified caller request")
	}
	stream, err := client.ChatCompletionsStream(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	chunk, err := stream.Recv()
	if err != nil || chunk.ID != "chunk-1" {
		t.Fatalf("chunk=%+v err=%v", chunk, err)
	}
	if _, err = stream.Recv(); !errors.Is(err, io.EOF) {
		t.Fatalf("Recv error=%v", err)
	}
	if err = stream.Close(); err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatalf("calls=%d", calls)
	}
}

func TestNativeClientErrors(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":{"code":"bad","type":"invalid_request_error","message":"nope"}}`)
	}))
	defer server.Close()
	_, err := NewClient("key", WithBaseURL(server.URL), WithHTTPClient(server.Client())).ChatCompletions(context.Background(), &ChatCompletionRequest{})
	var httpErr *HTTPError
	if !errors.As(err, &httpErr) || httpErr.Code != "bad" || httpErr.StatusCode != 400 {
		t.Fatalf("error=%T %+v", err, err)
	}
}

func TestNativeStreamBodyError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, "data: {\"error\":{\"code\":\"rate_limit\",\"message\":\"slow\",\"type\":\"server_error\"}}\n\n")
	}))
	defer server.Close()
	stream, err := NewClient("key", WithBaseURL(server.URL), WithHTTPClient(server.Client())).ChatCompletionsStream(context.Background(), &ChatCompletionRequest{})
	if err != nil {
		t.Fatal(err)
	}
	_, err = stream.Recv()
	var httpErr *HTTPError
	if !errors.As(err, &httpErr) || httpErr.Code != "rate_limit" {
		t.Fatalf("error=%T %+v", err, err)
	}
}
