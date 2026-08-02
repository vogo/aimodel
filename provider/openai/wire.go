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
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"sync"
)

// ChatCompletionRequest is the native OpenAI POST /chat/completions body.
type ChatCompletionRequest struct {
	Model               string                   `json:"model"`
	Messages            []ChatCompletionMessage  `json:"messages"`
	Audio               *AudioConfig             `json:"audio,omitempty"`
	FrequencyPenalty    *float64                 `json:"frequency_penalty,omitempty"`
	FunctionCall        any                      `json:"function_call,omitempty"`
	Functions           []ChatCompletionFunction `json:"functions,omitempty"`
	LogitBias           map[string]int           `json:"logit_bias,omitempty"`
	Logprobs            *bool                    `json:"logprobs,omitempty"`
	MaxCompletionTokens *int                     `json:"max_completion_tokens,omitempty"`
	MaxTokens           *int                     `json:"max_tokens,omitempty"`
	Metadata            map[string]string        `json:"metadata,omitempty"`
	Modalities          []string                 `json:"modalities,omitempty"`
	N                   *int                     `json:"n,omitempty"`
	ParallelToolCalls   *bool                    `json:"parallel_tool_calls,omitempty"`
	Prediction          any                      `json:"prediction,omitempty"`
	PresencePenalty     *float64                 `json:"presence_penalty,omitempty"`
	PromptCacheKey      string                   `json:"prompt_cache_key,omitempty"`
	ReasoningEffort     string                   `json:"reasoning_effort,omitempty"`
	ResponseFormat      any                      `json:"response_format,omitempty"`
	SafetyIdentifier    string                   `json:"safety_identifier,omitempty"`
	Seed                *int64                   `json:"seed,omitempty"`
	ServiceTier         string                   `json:"service_tier,omitempty"`
	Stop                []string                 `json:"stop,omitempty"`
	Store               *bool                    `json:"store,omitempty"`
	Stream              bool                     `json:"stream,omitempty"`
	StreamOptions       *StreamOptions           `json:"stream_options,omitempty"`
	Temperature         *float64                 `json:"temperature,omitempty"`
	ToolChoice          any                      `json:"tool_choice,omitempty"`
	Tools               []ChatCompletionTool     `json:"tools,omitempty"`
	TopK                *int                     `json:"top_k,omitempty"`
	TopLogprobs         *int                     `json:"top_logprobs,omitempty"`
	TopP                *float64                 `json:"top_p,omitempty"`
	User                string                   `json:"user,omitempty"`
	Verbosity           string                   `json:"verbosity,omitempty"`
	WebSearchOptions    *WebSearchOptions        `json:"web_search_options,omitempty"`
	Thinking            *Thinking                `json:"thinking,omitempty"`

	// ExtraBody carries top-level request parameters this package does not
	// model — the private parameters OpenAI-compatible backends add to the
	// protocol, such as `enable_thinking` or `chat_template_kwargs`. Entries
	// are merged into the top level of the request body verbatim.
	//
	// It is additive only: a key that collides with a field modelled above is
	// rejected by MarshalJSON rather than overriding or duplicating it, so a
	// backend-private parameter can never silently change a documented one.
	// Decoding a request body fills ExtraBody with every key this package does
	// not model, which is what makes a request round-trip losslessly.
	ExtraBody map[string]json.RawMessage `json:"-"`
}

// modelledRequestFields is the set of top-level JSON keys ChatCompletionRequest
// models, derived from the struct tags so it cannot drift as fields are added.
var modelledRequestFields = sync.OnceValue(func() map[string]bool {
	requestType := reflect.TypeFor[ChatCompletionRequest]()
	fields := make(map[string]bool, requestType.NumField())

	for field := range requestType.Fields() {
		name, _, _ := strings.Cut(field.Tag.Get("json"), ",")
		if name != "" && name != "-" {
			fields[name] = true
		}
	}

	return fields
})

// MarshalJSON encodes the modelled fields and merges ExtraBody into the same
// object. A key that collides with a modelled field, or a value that is not
// valid JSON, fails the call — before any network I/O.
func (r ChatCompletionRequest) MarshalJSON() ([]byte, error) {
	// wire drops the method set, so encoding it does not recurse.
	type wire ChatCompletionRequest

	modelled, err := json.Marshal((*wire)(&r))
	if err != nil {
		return nil, err
	}

	if len(r.ExtraBody) == 0 {
		return modelled, nil
	}

	body := map[string]json.RawMessage{}
	if err = json.Unmarshal(modelled, &body); err != nil {
		return nil, err
	}

	fields := modelledRequestFields()

	for key, value := range r.ExtraBody {
		if key == "" {
			return nil, fmt.Errorf("openai: extra body key is empty")
		}

		if fields[key] {
			return nil, fmt.Errorf("openai: extra body key %q collides with a modelled request field", key)
		}

		if !json.Valid(value) {
			return nil, fmt.Errorf("openai: extra body key %q holds invalid JSON", key)
		}

		body[key] = value
	}

	return json.Marshal(body)
}

// UnmarshalJSON decodes the modelled fields and collects every other top-level
// key into ExtraBody, so an encode → decode → encode round trip preserves the
// whole body.
func (r *ChatCompletionRequest) UnmarshalJSON(data []byte) error {
	type wire ChatCompletionRequest

	var decoded wire
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}

	*r = ChatCompletionRequest(decoded)

	var body map[string]json.RawMessage
	if err := json.Unmarshal(data, &body); err != nil {
		return err
	}

	fields := modelledRequestFields()

	for key, value := range body {
		if fields[key] {
			continue
		}

		if r.ExtraBody == nil {
			r.ExtraBody = make(map[string]json.RawMessage, len(body))
		}

		r.ExtraBody[key] = value
	}

	return nil
}

type StreamOptions struct {
	IncludeObfuscation *bool `json:"include_obfuscation,omitempty"`
	IncludeUsage       *bool `json:"include_usage,omitempty"`
}

type AudioConfig struct {
	Format string `json:"format"`
	Voice  any    `json:"voice"`
}

type WebSearchOptions struct {
	SearchContextSize string        `json:"search_context_size,omitempty"`
	UserLocation      *UserLocation `json:"user_location,omitempty"`
}

type UserLocation struct {
	Type        string               `json:"type"`
	Approximate *ApproximateLocation `json:"approximate,omitempty"`
}

type ApproximateLocation struct {
	City     string `json:"city,omitempty"`
	Country  string `json:"country,omitempty"`
	Region   string `json:"region,omitempty"`
	Timezone string `json:"timezone,omitempty"`
}

// Thinking is retained for OpenAI-compatible backends that support this
// shared reasoning control.
type Thinking struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens,omitempty"`
	Display      string `json:"display,omitempty"`
}

type ChatCompletionMessage struct {
	Role             string                   `json:"role"`
	Content          ChatCompletionContent    `json:"content"`
	Name             string                   `json:"name,omitempty"`
	Refusal          string                   `json:"refusal,omitempty"`
	ReasoningContent string                   `json:"reasoning_content,omitempty"`
	ToolCallID       string                   `json:"tool_call_id,omitempty"`
	ToolCalls        []ChatCompletionToolCall `json:"tool_calls,omitempty"`
	Audio            *ChatCompletionAudio     `json:"audio,omitempty"`
}

type ChatCompletionContent struct {
	text  *string
	parts []ChatCompletionContentPart
}

func NewTextContent(text string) ChatCompletionContent { return ChatCompletionContent{text: &text} }
func NewPartsContent(parts ...ChatCompletionContentPart) ChatCompletionContent {
	return ChatCompletionContent{parts: parts}
}

func (c ChatCompletionContent) Text() string {
	if c.text != nil {
		return *c.text
	}
	var result strings.Builder
	for _, part := range c.parts {
		if part.Type == "text" {
			result.WriteString(part.Text)
		}
	}
	return result.String()
}
func (c ChatCompletionContent) Parts() []ChatCompletionContentPart { return c.parts }
func (c ChatCompletionContent) MarshalJSON() ([]byte, error) {
	if c.parts != nil {
		return json.Marshal(c.parts)
	}
	if c.text == nil {
		return []byte("null"), nil
	}
	return json.Marshal(*c.text)
}

func (c *ChatCompletionContent) UnmarshalJSON(data []byte) error {
	if string(data) == "null" {
		c.text, c.parts = nil, nil
		return nil
	}
	if len(data) != 0 && data[0] == '[' {
		c.text = nil
		return json.Unmarshal(data, &c.parts)
	}
	var text string
	if err := json.Unmarshal(data, &text); err != nil {
		return err
	}
	c.text, c.parts = &text, nil
	return nil
}

type ChatCompletionContentPart struct {
	Type       string      `json:"type"`
	Text       string      `json:"text,omitempty"`
	ImageURL   *ImageURL   `json:"image_url,omitempty"`
	InputAudio *InputAudio `json:"input_audio,omitempty"`
	File       *InputFile  `json:"file,omitempty"`
}
type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}
type InputAudio struct {
	Data   string `json:"data"`
	Format string `json:"format"`
}
type InputFile struct {
	FileData string `json:"file_data,omitempty"`
	FileID   string `json:"file_id,omitempty"`
	Filename string `json:"filename,omitempty"`
}

type ChatCompletionTool struct {
	Type     string                 `json:"type"`
	Function ChatCompletionFunction `json:"function"`
}
type ChatCompletionFunction struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
	Strict      *bool  `json:"strict,omitempty"`
}
type ChatCompletionToolCall struct {
	Index    int                        `json:"index,omitempty"`
	ID       string                     `json:"id,omitempty"`
	Type     string                     `json:"type,omitempty"`
	Function ChatCompletionFunctionCall `json:"function"`
}
type ChatCompletionFunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type ChatCompletionResponse struct {
	ID                string                 `json:"id"`
	Object            string                 `json:"object"`
	Created           int64                  `json:"created"`
	Model             string                 `json:"model"`
	Choices           []ChatCompletionChoice `json:"choices"`
	Usage             *ChatCompletionUsage   `json:"usage,omitempty"`
	ServiceTier       string                 `json:"service_tier,omitempty"`
	SystemFingerprint string                 `json:"system_fingerprint,omitempty"`
	Error             *Error                 `json:"error,omitempty"`
}
type ChatCompletionChoice struct {
	Index        int                   `json:"index"`
	Message      ChatCompletionMessage `json:"message"`
	FinishReason *string               `json:"finish_reason"`
	Logprobs     *ChoiceLogprobs       `json:"logprobs,omitempty"`
}
type ChoiceLogprobs struct {
	Content []TokenLogprob `json:"content,omitempty"`
	Refusal []TokenLogprob `json:"refusal,omitempty"`
}
type TokenLogprob struct {
	Token       string       `json:"token"`
	Logprob     float64      `json:"logprob"`
	Bytes       []int        `json:"bytes,omitempty"`
	TopLogprobs []TopLogprob `json:"top_logprobs,omitempty"`
}
type TopLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
	Bytes   []int   `json:"bytes,omitempty"`
}
type ChatCompletionAudio struct {
	ID         string `json:"id"`
	Data       string `json:"data"`
	ExpiresAt  int64  `json:"expires_at"`
	Transcript string `json:"transcript"`
}

type ChatCompletionUsage struct {
	PromptTokens            int                      `json:"prompt_tokens"`
	CompletionTokens        int                      `json:"completion_tokens"`
	TotalTokens             int                      `json:"total_tokens"`
	PromptTokensDetails     *PromptTokensDetails     `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *CompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}
type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
	AudioTokens  int `json:"audio_tokens"`
}
type CompletionTokensDetails struct {
	AcceptedPredictionTokens int `json:"accepted_prediction_tokens"`
	AudioTokens              int `json:"audio_tokens"`
	ReasoningTokens          int `json:"reasoning_tokens"`
	RejectedPredictionTokens int `json:"rejected_prediction_tokens"`
}

type ChatCompletionChunk struct {
	ID                string                      `json:"id"`
	Object            string                      `json:"object"`
	Created           int64                       `json:"created"`
	Model             string                      `json:"model"`
	Choices           []ChatCompletionChunkChoice `json:"choices"`
	Usage             *ChatCompletionUsage        `json:"usage,omitempty"`
	ServiceTier       string                      `json:"service_tier,omitempty"`
	SystemFingerprint string                      `json:"system_fingerprint,omitempty"`
	Error             *Error                      `json:"error,omitempty"`
}
type ChatCompletionChunkChoice struct {
	Index        int                   `json:"index"`
	Delta        ChatCompletionMessage `json:"delta"`
	FinishReason *string               `json:"finish_reason"`
	Logprobs     *ChoiceLogprobs       `json:"logprobs,omitempty"`
}
type Error struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Param   any    `json:"param,omitempty"`
	Type    string `json:"type"`
}
