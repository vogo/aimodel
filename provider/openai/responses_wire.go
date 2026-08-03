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

import "encoding/json"

// Native wire model for the OpenAI Responses API (POST /v1/responses), verified
// against the official reference on 2026-08-01.
//
// These types are OpenAI-only, like every type in this package: a protocol is
// expressed where it is served.

// ResponsesRequest is the native OpenAI POST /v1/responses body.
type ResponsesRequest struct {
	Model                string                      `json:"model,omitempty"`
	Input                *ResponseInput              `json:"input,omitempty"`
	Background           *bool                       `json:"background,omitempty"`
	ContextManagement    []ResponseContextManagement `json:"context_management,omitempty"`
	Conversation         *ResponseConversation       `json:"conversation,omitempty"`
	Include              []string                    `json:"include,omitempty"`
	Instructions         string                      `json:"instructions,omitempty"`
	MaxOutputTokens      *int                        `json:"max_output_tokens,omitempty"`
	MaxToolCalls         *int                        `json:"max_tool_calls,omitempty"`
	Metadata             map[string]string           `json:"metadata,omitempty"`
	Moderation           *ResponseModerationConfig   `json:"moderation,omitempty"`
	ParallelToolCalls    *bool                       `json:"parallel_tool_calls,omitempty"`
	PreviousResponseID   string                      `json:"previous_response_id,omitempty"`
	Prompt               *ResponsePrompt             `json:"prompt,omitempty"`
	PromptCacheKey       string                      `json:"prompt_cache_key,omitempty"`
	PromptCacheOptions   *ResponsePromptCacheOptions `json:"prompt_cache_options,omitempty"`
	PromptCacheRetention string                      `json:"prompt_cache_retention,omitempty"`
	Reasoning            *ResponseReasoningConfig    `json:"reasoning,omitempty"`
	SafetyIdentifier     string                      `json:"safety_identifier,omitempty"`
	ServiceTier          string                      `json:"service_tier,omitempty"`
	Store                *bool                       `json:"store,omitempty"`
	Stream               bool                        `json:"stream,omitempty"`
	StreamOptions        *ResponseStreamOptions      `json:"stream_options,omitempty"`
	Temperature          *float64                    `json:"temperature,omitempty"`
	Text                 *ResponseTextConfig         `json:"text,omitempty"`
	ToolChoice           any                         `json:"tool_choice,omitempty"`
	Tools                []ResponseTool              `json:"tools,omitempty"`
	TopLogprobs          *int                        `json:"top_logprobs,omitempty"`
	TopP                 *float64                    `json:"top_p,omitempty"`
	Truncation           string                      `json:"truncation,omitempty"`
	User                 string                      `json:"user,omitempty"`
}

// Response is the native OpenAI Responses object returned by POST /v1/responses
// and carried by the lifecycle stream events.
type Response struct {
	ID                   string                      `json:"id"`
	Object               string                      `json:"object,omitempty"`
	CreatedAt            float64                     `json:"created_at,omitempty"`
	CompletedAt          *float64                    `json:"completed_at,omitempty"`
	Status               string                      `json:"status,omitempty"`
	Error                *ResponseError              `json:"error,omitempty"`
	IncompleteDetails    *ResponseIncompleteDetails  `json:"incomplete_details,omitempty"`
	Instructions         *ResponseInput              `json:"instructions,omitempty"`
	Model                string                      `json:"model,omitempty"`
	Output               []ResponseOutputItem        `json:"output,omitempty"`
	Background           *bool                       `json:"background,omitempty"`
	Conversation         *ResponseConversation       `json:"conversation,omitempty"`
	MaxOutputTokens      *int                        `json:"max_output_tokens,omitempty"`
	MaxToolCalls         *int                        `json:"max_tool_calls,omitempty"`
	Metadata             map[string]string           `json:"metadata,omitempty"`
	Moderation           *ResponseModeration         `json:"moderation,omitempty"`
	ParallelToolCalls    bool                        `json:"parallel_tool_calls,omitempty"`
	PreviousResponseID   string                      `json:"previous_response_id,omitempty"`
	Prompt               *ResponsePrompt             `json:"prompt,omitempty"`
	PromptCacheKey       string                      `json:"prompt_cache_key,omitempty"`
	PromptCacheOptions   *ResponsePromptCacheOptions `json:"prompt_cache_options,omitempty"`
	PromptCacheRetention string                      `json:"prompt_cache_retention,omitempty"`
	Reasoning            *ResponseReasoningConfig    `json:"reasoning,omitempty"`
	SafetyIdentifier     string                      `json:"safety_identifier,omitempty"`
	ServiceTier          string                      `json:"service_tier,omitempty"`
	Temperature          *float64                    `json:"temperature,omitempty"`
	Text                 *ResponseTextConfig         `json:"text,omitempty"`
	ToolChoice           any                         `json:"tool_choice,omitempty"`
	Tools                []ResponseTool              `json:"tools,omitempty"`
	TopLogprobs          *int                        `json:"top_logprobs,omitempty"`
	TopP                 *float64                    `json:"top_p,omitempty"`
	Truncation           string                      `json:"truncation,omitempty"`
	Usage                *ResponseUsage              `json:"usage,omitempty"`
	User                 string                      `json:"user,omitempty"`

	// OutputText is an SDK convenience, not a server field: it concatenates the
	// text of every output_text content part in Output order. It is derived when
	// the response is decoded, so it does not reflect later edits to Output.
	OutputText string `json:"-"`
}

// responseWire mirrors Response without its custom decoding, so UnmarshalJSON
// can decode the wire fields and then derive OutputText from them.
type responseWire Response

// UnmarshalJSON decodes the wire object and derives the non-wire OutputText.
func (r *Response) UnmarshalJSON(data []byte) error {
	var wire responseWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}

	*r = Response(wire)
	r.OutputText = r.aggregateOutputText()

	return nil
}

// aggregateOutputText concatenates every output_text part across the output
// items, preserving output order.
func (r *Response) aggregateOutputText() string {
	var text []byte

	for i := range r.Output {
		message := r.Output[i].Message
		if message == nil {
			continue
		}

		for _, part := range message.Content {
			if part.Type == ResponseContentTypeOutputText {
				text = append(text, part.Text...)
			}
		}
	}

	return string(text)
}

// ResponseError is the response-level error object.
type ResponseError struct {
	Code    string `json:"code,omitempty"`
	Message string `json:"message,omitempty"`
}

// ResponseIncompleteDetails explains why a response stopped short.
type ResponseIncompleteDetails struct {
	Reason string `json:"reason,omitempty"`
}

// ResponseConversation references the server-side conversation a response
// belongs to. The request accepts a bare conversation ID string as well, which
// decodes into ID; it is always re-encoded in object form.
type ResponseConversation struct {
	ID string `json:"id"`
}

// UnmarshalJSON accepts both the object form and the bare ID string.
func (c *ResponseConversation) UnmarshalJSON(data []byte) error {
	if len(data) != 0 && data[0] == '"' {
		return json.Unmarshal(data, &c.ID)
	}

	type alias ResponseConversation

	return json.Unmarshal(data, (*alias)(c))
}

// ResponseContextManagement is one context-management entry (e.g. compaction).
type ResponseContextManagement struct {
	Type             string `json:"type"`
	CompactThreshold *int   `json:"compact_threshold,omitempty"`
}

// ResponseModerationConfig requests moderation of the response input/output.
type ResponseModerationConfig struct {
	Model  string                    `json:"model"`
	Policy *ResponseModerationPolicy `json:"policy,omitempty"`
}

// ResponseModerationPolicy holds the per-side moderation modes.
type ResponseModerationPolicy struct {
	Input  *ResponseModerationPolicySide `json:"input,omitempty"`
	Output *ResponseModerationPolicySide `json:"output,omitempty"`
}

// ResponseModerationPolicySide is one side's moderation mode (score or block).
type ResponseModerationPolicySide struct {
	Mode string `json:"mode"`
}

// ResponseModeration carries the moderation results attached to a response.
type ResponseModeration struct {
	Input  *ResponseModerationResult `json:"input,omitempty"`
	Output *ResponseModerationResult `json:"output,omitempty"`
}

// ResponseModerationResult is either a moderation_result or an error entry.
type ResponseModerationResult struct {
	Type                      string              `json:"type"`
	Categories                map[string]bool     `json:"categories,omitempty"`
	CategoryAppliedInputTypes map[string][]string `json:"category_applied_input_types,omitempty"`
	CategoryScores            map[string]float64  `json:"category_scores,omitempty"`
	Flagged                   bool                `json:"flagged,omitempty"`
	Model                     string              `json:"model,omitempty"`
	Code                      string              `json:"code,omitempty"`
	Message                   string              `json:"message,omitempty"`
}

// ResponsePrompt references a stored prompt template and its variables.
type ResponsePrompt struct {
	ID        string         `json:"id"`
	Variables map[string]any `json:"variables,omitempty"`
	Version   string         `json:"version,omitempty"`
}

// ResponsePromptCacheOptions configures explicit prompt-cache breakpoints.
type ResponsePromptCacheOptions struct {
	Mode string `json:"mode,omitempty"`
	TTL  string `json:"ttl,omitempty"`
}

// ResponsePromptCacheBreakpoint marks the end of a reusable prompt prefix.
type ResponsePromptCacheBreakpoint struct {
	Mode string `json:"mode"`
}

// ResponseReasoningConfig configures reasoning models and is echoed back on the
// response with the effective values.
type ResponseReasoningConfig struct {
	Context         string `json:"context,omitempty"`
	Effort          string `json:"effort,omitempty"`
	GenerateSummary string `json:"generate_summary,omitempty"`
	Mode            string `json:"mode,omitempty"`
	Summary         string `json:"summary,omitempty"`
}

// ResponseStreamOptions configures streaming behavior.
type ResponseStreamOptions struct {
	IncludeObfuscation *bool `json:"include_obfuscation,omitempty"`
}

// ResponseTextConfig configures plain-text or structured output.
type ResponseTextConfig struct {
	Format    *ResponseTextFormat `json:"format,omitempty"`
	Verbosity string              `json:"verbosity,omitempty"`
}

// ResponseTextFormat is the text / json_object / json_schema output format.
type ResponseTextFormat struct {
	Type        string `json:"type"`
	Name        string `json:"name,omitempty"`
	Description string `json:"description,omitempty"`
	Schema      any    `json:"schema,omitempty"`
	Strict      *bool  `json:"strict,omitempty"`
}

// ResponseUsage is the Responses token accounting.
type ResponseUsage struct {
	InputTokens         int                          `json:"input_tokens"`
	InputTokensDetails  *ResponseInputTokensDetails  `json:"input_tokens_details,omitempty"`
	OutputTokens        int                          `json:"output_tokens"`
	OutputTokensDetails *ResponseOutputTokensDetails `json:"output_tokens_details,omitempty"`
	TotalTokens         int                          `json:"total_tokens"`
}

// ResponseInputTokensDetails breaks down the input tokens.
type ResponseInputTokensDetails struct {
	CachedTokens     int `json:"cached_tokens"`
	CacheWriteTokens int `json:"cache_write_tokens,omitempty"`
}

// ResponseOutputTokensDetails breaks down the output tokens.
type ResponseOutputTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

// ResponseInput is the polymorphic `input` value: either a plain text prompt or
// an ordered list of input items. It also decodes the response-side
// `instructions` field, which has the same string-or-items shape.
type ResponseInput struct {
	text  *string
	items []ResponseInputItem
}

// NewResponseTextInput builds the scalar string form of `input`.
func NewResponseTextInput(text string) *ResponseInput {
	return &ResponseInput{text: &text}
}

// NewResponseItemsInput builds the structured item-list form of `input`.
func NewResponseItemsInput(items ...ResponseInputItem) *ResponseInput {
	return &ResponseInput{items: items}
}

// Text returns the scalar prompt, or the concatenated text of every text part
// of every message item when the input is structured.
func (in *ResponseInput) Text() string {
	if in == nil {
		return ""
	}

	if in.text != nil {
		return *in.text
	}

	var text []byte

	for i := range in.items {
		if message := in.items[i].Message; message != nil {
			text = append(text, message.Content.Text()...)
		}
	}

	return string(text)
}

// Items returns the structured items, or nil for the scalar form.
func (in *ResponseInput) Items() []ResponseInputItem {
	if in == nil {
		return nil
	}

	return in.items
}

// MarshalJSON emits the string form or the item array, matching what was set.
func (in ResponseInput) MarshalJSON() ([]byte, error) {
	if in.items != nil {
		return json.Marshal(in.items)
	}

	if in.text == nil {
		return []byte("null"), nil
	}

	return json.Marshal(*in.text)
}

// UnmarshalJSON accepts both the string and the item-array form.
func (in *ResponseInput) UnmarshalJSON(data []byte) error {
	if string(data) == "null" {
		in.text, in.items = nil, nil

		return nil
	}

	if len(data) != 0 && data[0] == '[' {
		in.text = nil

		return json.Unmarshal(data, &in.items)
	}

	var text string
	if err := json.Unmarshal(data, &text); err != nil {
		return err
	}

	in.text, in.items = &text, nil

	return nil
}

// ResponseMessageContent is the polymorphic message `content` (and function
// call `output`) value: either a plain string or a list of content parts.
type ResponseMessageContent struct {
	text  *string
	parts []ResponseContent
}

// NewResponseTextContent builds the scalar string form.
func NewResponseTextContent(text string) ResponseMessageContent {
	return ResponseMessageContent{text: &text}
}

// NewResponseContentParts builds the content-part list form.
func NewResponseContentParts(parts ...ResponseContent) ResponseMessageContent {
	return ResponseMessageContent{parts: parts}
}

// Text returns the scalar string, or the concatenation of every part carrying
// text (input_text, output_text, reasoning_text).
func (c ResponseMessageContent) Text() string {
	if c.text != nil {
		return *c.text
	}

	var text []byte

	for _, part := range c.parts {
		switch part.Type {
		case ResponseContentTypeInputText, ResponseContentTypeOutputText, ResponseContentTypeReasoningText:
			text = append(text, part.Text...)
		}
	}

	return string(text)
}

// Parts returns the content parts, or nil for the scalar form.
func (c ResponseMessageContent) Parts() []ResponseContent { return c.parts }

// MarshalJSON emits the string form or the part array, matching what was set.
func (c ResponseMessageContent) MarshalJSON() ([]byte, error) {
	if c.parts != nil {
		return json.Marshal(c.parts)
	}

	if c.text == nil {
		return []byte("null"), nil
	}

	return json.Marshal(*c.text)
}

// UnmarshalJSON accepts both the string and the part-array form.
func (c *ResponseMessageContent) UnmarshalJSON(data []byte) error {
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

// ResponseContent is one content part. Responses reuses a small field
// vocabulary across its part types, so one struct with a `type` discriminator
// covers input_text / input_image / input_file / input_audio on the input side
// and output_text / refusal / reasoning_text on the output side.
type ResponseContent struct {
	Type                  string                         `json:"type"`
	Text                  string                         `json:"text,omitempty"`
	Refusal               string                         `json:"refusal,omitempty"`
	Annotations           []ResponseAnnotation           `json:"annotations,omitempty"`
	Logprobs              []ResponseLogprob              `json:"logprobs,omitempty"`
	Detail                string                         `json:"detail,omitempty"`
	FileID                string                         `json:"file_id,omitempty"`
	FileURL               string                         `json:"file_url,omitempty"`
	FileData              string                         `json:"file_data,omitempty"`
	Filename              string                         `json:"filename,omitempty"`
	ImageURL              string                         `json:"image_url,omitempty"`
	InputAudio            *ResponseInputAudio            `json:"input_audio,omitempty"`
	PromptCacheBreakpoint *ResponsePromptCacheBreakpoint `json:"prompt_cache_breakpoint,omitempty"`
}

// ResponseInputAudio is the payload of an input_audio content part.
type ResponseInputAudio struct {
	Data   string `json:"data"`
	Format string `json:"format"`
}

// ResponseAnnotation is one output_text annotation: a file citation, URL
// citation, container file citation, or file path.
type ResponseAnnotation struct {
	Type        string `json:"type"`
	FileID      string `json:"file_id,omitempty"`
	Filename    string `json:"filename,omitempty"`
	ContainerID string `json:"container_id,omitempty"`
	Index       *int   `json:"index,omitempty"`
	StartIndex  *int   `json:"start_index,omitempty"`
	EndIndex    *int   `json:"end_index,omitempty"`
	Title       string `json:"title,omitempty"`
	URL         string `json:"url,omitempty"`
}

// ResponseLogprob is the log probability of one output token.
type ResponseLogprob struct {
	Token       string               `json:"token"`
	Logprob     float64              `json:"logprob"`
	Bytes       []int                `json:"bytes,omitempty"`
	TopLogprobs []ResponseTopLogprob `json:"top_logprobs,omitempty"`
}

// ResponseTopLogprob is one alternative token considered at a position.
type ResponseTopLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
	Bytes   []int   `json:"bytes,omitempty"`
}

// ResponseInputMessage is a `message` item on the input side. It also carries a
// replayed assistant message, whose content parts are output_text / refusal.
type ResponseInputMessage struct {
	Type    string                 `json:"type,omitempty"`
	Role    string                 `json:"role"`
	Content ResponseMessageContent `json:"content"`
	ID      string                 `json:"id,omitempty"`
	Status  string                 `json:"status,omitempty"`
	Phase   string                 `json:"phase,omitempty"`
}

// ResponseOutputMessage is a `message` item produced by the model.
type ResponseOutputMessage struct {
	Type    string            `json:"type"`
	ID      string            `json:"id,omitempty"`
	Role    string            `json:"role,omitempty"`
	Status  string            `json:"status,omitempty"`
	Phase   string            `json:"phase,omitempty"`
	Content []ResponseContent `json:"content"`
}

// ResponseReasoningItem is a `reasoning` item: the model's chain-of-thought
// summary and, where enabled, its encrypted content for stateless replay.
type ResponseReasoningItem struct {
	Type             string                     `json:"type"`
	ID               string                     `json:"id,omitempty"`
	Summary          []ResponseReasoningSummary `json:"summary"`
	Content          []ResponseContent          `json:"content,omitempty"`
	EncryptedContent string                     `json:"encrypted_content,omitempty"`
	Status           string                     `json:"status,omitempty"`
}

// ResponseReasoningSummary is one summary_text block of a reasoning item.
type ResponseReasoningSummary struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// ResponseFunctionToolCall is a `function_call` item.
type ResponseFunctionToolCall struct {
	Type      string          `json:"type"`
	ID        string          `json:"id,omitempty"`
	CallID    string          `json:"call_id"`
	Name      string          `json:"name"`
	Arguments string          `json:"arguments"`
	Namespace string          `json:"namespace,omitempty"`
	Caller    *ResponseCaller `json:"caller,omitempty"`
	Status    string          `json:"status,omitempty"`
}

// ResponseCaller records the execution context that produced a tool call.
type ResponseCaller struct {
	Type     string `json:"type"`
	CallerID string `json:"caller_id,omitempty"`
}

// ResponseFunctionCallOutput is a `function_call_output` item — the result the
// caller feeds back for a function call. The SDK transports it; it never
// executes caller functions.
type ResponseFunctionCallOutput struct {
	Type      string                 `json:"type"`
	ID        string                 `json:"id,omitempty"`
	CallID    string                 `json:"call_id"`
	Output    ResponseMessageContent `json:"output"`
	Status    string                 `json:"status,omitempty"`
	Caller    *ResponseCaller        `json:"caller,omitempty"`
	CreatedBy string                 `json:"created_by,omitempty"`
}

// ResponseItemReference is an `item_reference` input item.
type ResponseItemReference struct {
	Type string `json:"type"`
	ID   string `json:"id"`
}

// ResponseWebSearchCall is a `web_search_call` item produced by the hosted web
// search tool.
type ResponseWebSearchCall struct {
	Type   string                   `json:"type"`
	ID     string                   `json:"id,omitempty"`
	Status string                   `json:"status,omitempty"`
	Action *ResponseWebSearchAction `json:"action,omitempty"`
}

// ResponseWebSearchAction describes what the model did with the web: a search,
// an open_page, or a find_in_page.
type ResponseWebSearchAction struct {
	Type    string                    `json:"type"`
	Query   string                    `json:"query,omitempty"`
	Queries []string                  `json:"queries,omitempty"`
	Sources []ResponseWebSearchSource `json:"sources,omitempty"`
	URL     string                    `json:"url,omitempty"`
	Pattern string                    `json:"pattern,omitempty"`
}

// ResponseWebSearchSource is one source consulted during a web search.
type ResponseWebSearchSource struct {
	Type string `json:"type"`
	URL  string `json:"url"`
}

// ResponseFileSearchCall is a `file_search_call` item produced by the hosted
// file search tool.
type ResponseFileSearchCall struct {
	Type    string                     `json:"type"`
	ID      string                     `json:"id,omitempty"`
	Status  string                     `json:"status,omitempty"`
	Queries []string                   `json:"queries,omitempty"`
	Results []ResponseFileSearchResult `json:"results,omitempty"`
}

// ResponseFileSearchResult is one retrieved chunk of a file search call. It is
// present only when `file_search_call.results` is requested via `include`.
type ResponseFileSearchResult struct {
	FileID     string         `json:"file_id,omitempty"`
	Filename   string         `json:"filename,omitempty"`
	Score      *float64       `json:"score,omitempty"`
	Text       string         `json:"text,omitempty"`
	Attributes map[string]any `json:"attributes,omitempty"`
}

// ResponseCodeInterpreterCall is a `code_interpreter_call` item produced by the
// hosted code interpreter tool.
type ResponseCodeInterpreterCall struct {
	Type        string                          `json:"type"`
	ID          string                          `json:"id,omitempty"`
	Status      string                          `json:"status,omitempty"`
	ContainerID string                          `json:"container_id,omitempty"`
	Code        string                          `json:"code,omitempty"`
	Outputs     []ResponseCodeInterpreterOutput `json:"outputs,omitempty"`
}

// ResponseCodeInterpreterOutput is one logs or image output of a code
// interpreter call. Outputs are present only when
// `code_interpreter_call.outputs` is requested via `include`.
type ResponseCodeInterpreterOutput struct {
	Type string `json:"type"`
	Logs string `json:"logs,omitempty"`
	URL  string `json:"url,omitempty"`
}

// ResponseTool is one entry of the request/response `tools` array. The three
// hosted tools this SDK covers (web_search, file_search, code_interpreter) and
// caller-defined functions decode into the typed fields; any other tool type
// keeps its payload in Raw and is re-encoded verbatim.
type ResponseTool struct {
	Type string `json:"type"`

	// function
	Name           string   `json:"name,omitempty"`
	Description    string   `json:"description,omitempty"`
	Parameters     any      `json:"parameters,omitempty"`
	Strict         *bool    `json:"strict,omitempty"`
	OutputSchema   any      `json:"output_schema,omitempty"`
	AllowedCallers []string `json:"allowed_callers,omitempty"`
	DeferLoading   *bool    `json:"defer_loading,omitempty"`

	// file_search
	VectorStoreIDs []string                          `json:"vector_store_ids,omitempty"`
	MaxNumResults  *int                              `json:"max_num_results,omitempty"`
	RankingOptions *ResponseFileSearchRankingOptions `json:"ranking_options,omitempty"`

	// file_search (comparison/compound filter) and web_search (allowed domains)
	Filters any `json:"filters,omitempty"`

	// web_search
	SearchContextSize string                         `json:"search_context_size,omitempty"`
	UserLocation      *ResponseWebSearchUserLocation `json:"user_location,omitempty"`

	// code_interpreter
	Container any `json:"container,omitempty"`

	// Raw holds the verbatim payload of a tool whose type this SDK does not
	// model. It is populated on decode for those types only, and is re-encoded
	// as-is so an unsupported tool survives a round trip.
	Raw json.RawMessage `json:"-"`
}

// ResponseFileSearchRankingOptions tunes file search ranking.
type ResponseFileSearchRankingOptions struct {
	Ranker         string                          `json:"ranker,omitempty"`
	ScoreThreshold *float64                        `json:"score_threshold,omitempty"`
	HybridSearch   *ResponseFileSearchHybridSearch `json:"hybrid_search,omitempty"`
}

// ResponseFileSearchHybridSearch weights semantic against keyword matching.
type ResponseFileSearchHybridSearch struct {
	EmbeddingWeight float64 `json:"embedding_weight"`
	TextWeight      float64 `json:"text_weight"`
}

// ResponseWebSearchUserLocation approximates the user's location for search.
type ResponseWebSearchUserLocation struct {
	Type     string `json:"type,omitempty"`
	City     string `json:"city,omitempty"`
	Country  string `json:"country,omitempty"`
	Region   string `json:"region,omitempty"`
	Timezone string `json:"timezone,omitempty"`
}

// ResponseCodeInterpreterContainerAuto is the object form of the code
// interpreter `container` parameter (the other form is a container ID string).
type ResponseCodeInterpreterContainerAuto struct {
	Type          string   `json:"type"`
	FileIDs       []string `json:"file_ids,omitempty"`
	MemoryLimit   string   `json:"memory_limit,omitempty"`
	NetworkPolicy any      `json:"network_policy,omitempty"`
}

// responseToolWire mirrors ResponseTool without its custom JSON handling.
type responseToolWire ResponseTool

// MarshalJSON re-encodes an unmodeled tool from Raw and every modeled tool from
// its typed fields.
func (t ResponseTool) MarshalJSON() ([]byte, error) {
	if len(t.Raw) != 0 {
		return append(json.RawMessage(nil), t.Raw...), nil
	}

	return json.Marshal(responseToolWire(t))
}

// UnmarshalJSON decodes the typed fields, and keeps the verbatim payload for a
// tool type this SDK does not model.
func (t *ResponseTool) UnmarshalJSON(data []byte) error {
	var wire responseToolWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}

	*t = ResponseTool(wire)

	switch t.Type {
	case ResponseToolTypeFunction, ResponseToolTypeFileSearch, ResponseToolTypeCodeInterpreter,
		ResponseToolTypeWebSearch, ResponseToolTypeWebSearch20250826:
		return nil
	}

	t.Raw = append(json.RawMessage(nil), data...)

	return nil
}

// ResponseToolChoiceTypes forces a hosted tool (`{"type": "file_search"}`).
type ResponseToolChoiceTypes struct {
	Type string `json:"type"`
}

// ResponseToolChoiceFunction forces one named function.
type ResponseToolChoiceFunction struct {
	Type string `json:"type"`
	Name string `json:"name"`
}

// ResponseToolChoiceAllowed constrains the model to a subset of tools.
type ResponseToolChoiceAllowed struct {
	Type  string           `json:"type"`
	Mode  string           `json:"mode"`
	Tools []map[string]any `json:"tools"`
}

// ResponseToolChoiceMCP forces a tool on a named MCP server.
type ResponseToolChoiceMCP struct {
	Type        string `json:"type"`
	ServerLabel string `json:"server_label"`
	Name        string `json:"name,omitempty"`
}

// ResponseToolChoiceCustom forces one named custom tool.
type ResponseToolChoiceCustom struct {
	Type string `json:"type"`
	Name string `json:"name"`
}

// ResponseInputItem is one entry of the structured `input` array. Known types
// decode into their dedicated payload; anything else keeps its verbatim payload
// in Raw so it survives a round trip.
type ResponseInputItem struct {
	Type               string
	Message            *ResponseInputMessage
	FunctionCall       *ResponseFunctionToolCall
	FunctionCallOutput *ResponseFunctionCallOutput
	Reasoning          *ResponseReasoningItem
	ItemReference      *ResponseItemReference
	Raw                json.RawMessage
}

// NewResponseInputMessage builds a `message` input item with scalar text.
func NewResponseInputMessage(role, text string) ResponseInputItem {
	return ResponseInputItem{
		Type: ResponseItemTypeMessage,
		Message: &ResponseInputMessage{
			Type:    ResponseItemTypeMessage,
			Role:    role,
			Content: NewResponseTextContent(text),
		},
	}
}

// payload returns the dedicated payload of a known item, or nil.
func (i *ResponseInputItem) payload() any {
	switch {
	case i.Message != nil:
		return i.Message
	case i.FunctionCall != nil:
		return i.FunctionCall
	case i.FunctionCallOutput != nil:
		return i.FunctionCallOutput
	case i.Reasoning != nil:
		return i.Reasoning
	case i.ItemReference != nil:
		return i.ItemReference
	}

	return nil
}

// newPayload allocates the dedicated payload for a known discriminator.
func (i *ResponseInputItem) newPayload(itemType string) any {
	switch itemType {
	case ResponseItemTypeMessage:
		i.Message = &ResponseInputMessage{}

		return i.Message
	case ResponseItemTypeFunctionCall:
		i.FunctionCall = &ResponseFunctionToolCall{}

		return i.FunctionCall
	case ResponseItemTypeFunctionCallOutput:
		i.FunctionCallOutput = &ResponseFunctionCallOutput{}

		return i.FunctionCallOutput
	case ResponseItemTypeReasoning:
		i.Reasoning = &ResponseReasoningItem{}

		return i.Reasoning
	case ResponseItemTypeItemReference:
		i.ItemReference = &ResponseItemReference{}

		return i.ItemReference
	}

	return nil
}

// MarshalJSON emits the dedicated payload, falling back to the verbatim Raw
// payload of an item type this SDK does not model.
func (i ResponseInputItem) MarshalJSON() ([]byte, error) {
	return marshalResponseItem(i.payload(), i.Type, i.Raw)
}

// UnmarshalJSON decodes the discriminator, then the dedicated payload, always
// retaining the verbatim payload in Raw.
func (i *ResponseInputItem) UnmarshalJSON(data []byte) error {
	*i = ResponseInputItem{Raw: append(json.RawMessage(nil), data...)}

	return unmarshalResponseItem(data, &i.Type, i.newPayload)
}

// ResponseOutputItem is one entry of the `output` array (and of the item
// lifecycle stream events). Known types decode into their dedicated payload;
// anything else keeps its verbatim payload in Raw.
type ResponseOutputItem struct {
	Type                string
	Message             *ResponseOutputMessage
	Reasoning           *ResponseReasoningItem
	FunctionCall        *ResponseFunctionToolCall
	FunctionCallOutput  *ResponseFunctionCallOutput
	WebSearchCall       *ResponseWebSearchCall
	FileSearchCall      *ResponseFileSearchCall
	CodeInterpreterCall *ResponseCodeInterpreterCall
	Raw                 json.RawMessage
}

// payload returns the dedicated payload of a known item, or nil.
func (i *ResponseOutputItem) payload() any {
	switch {
	case i.Message != nil:
		return i.Message
	case i.Reasoning != nil:
		return i.Reasoning
	case i.FunctionCall != nil:
		return i.FunctionCall
	case i.FunctionCallOutput != nil:
		return i.FunctionCallOutput
	case i.WebSearchCall != nil:
		return i.WebSearchCall
	case i.FileSearchCall != nil:
		return i.FileSearchCall
	case i.CodeInterpreterCall != nil:
		return i.CodeInterpreterCall
	}

	return nil
}

// newPayload allocates the dedicated payload for a known discriminator.
func (i *ResponseOutputItem) newPayload(itemType string) any {
	switch itemType {
	case ResponseItemTypeMessage:
		i.Message = &ResponseOutputMessage{}

		return i.Message
	case ResponseItemTypeReasoning:
		i.Reasoning = &ResponseReasoningItem{}

		return i.Reasoning
	case ResponseItemTypeFunctionCall:
		i.FunctionCall = &ResponseFunctionToolCall{}

		return i.FunctionCall
	case ResponseItemTypeFunctionCallOutput:
		i.FunctionCallOutput = &ResponseFunctionCallOutput{}

		return i.FunctionCallOutput
	case ResponseItemTypeWebSearchCall:
		i.WebSearchCall = &ResponseWebSearchCall{}

		return i.WebSearchCall
	case ResponseItemTypeFileSearchCall:
		i.FileSearchCall = &ResponseFileSearchCall{}

		return i.FileSearchCall
	case ResponseItemTypeCodeInterpreterCall:
		i.CodeInterpreterCall = &ResponseCodeInterpreterCall{}

		return i.CodeInterpreterCall
	}

	return nil
}

// MarshalJSON emits the dedicated payload, falling back to the verbatim Raw
// payload of an item type this SDK does not model.
func (i ResponseOutputItem) MarshalJSON() ([]byte, error) {
	return marshalResponseItem(i.payload(), i.Type, i.Raw)
}

// UnmarshalJSON decodes the discriminator, then the dedicated payload, always
// retaining the verbatim payload in Raw.
func (i *ResponseOutputItem) UnmarshalJSON(data []byte) error {
	*i = ResponseOutputItem{Raw: append(json.RawMessage(nil), data...)}

	return unmarshalResponseItem(data, &i.Type, i.newPayload)
}

// marshalResponseItem is the shared item encoder: dedicated payload first,
// verbatim raw payload second, bare discriminator last.
func marshalResponseItem(payload any, itemType string, raw json.RawMessage) ([]byte, error) {
	if payload != nil {
		return json.Marshal(payload)
	}

	if len(raw) != 0 {
		return append(json.RawMessage(nil), raw...), nil
	}

	if itemType == "" {
		return []byte("null"), nil
	}

	return json.Marshal(struct {
		Type string `json:"type"`
	}{Type: itemType})
}

// unmarshalResponseItem is the shared item decoder: read the discriminator,
// then decode into the dedicated payload the caller allocates for it.
func unmarshalResponseItem(data []byte, itemType *string, newPayload func(string) any) error {
	var envelope struct {
		Type string `json:"type"`
	}

	if err := json.Unmarshal(data, &envelope); err != nil {
		return err
	}

	*itemType = envelope.Type

	target := newPayload(envelope.Type)
	if target == nil {
		return nil
	}

	return json.Unmarshal(data, target)
}
