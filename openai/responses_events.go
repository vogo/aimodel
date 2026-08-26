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

// Responses streaming event taxonomy, verified 2026-08-01 against the official
// streaming-events reference. Unlike Chat Completions, Responses streams a
// sequence of discriminated events rather than one repeated chunk shape, and it
// has no `[DONE]` sentinel — the stream ends when the body ends.

// Response lifecycle events. Each carries the full Response snapshot.
const (
	ResponseEventCreated    = "response.created"
	ResponseEventInProgress = "response.in_progress"
	ResponseEventQueued     = "response.queued"
	ResponseEventCompleted  = "response.completed"
	ResponseEventIncomplete = "response.incomplete"
	ResponseEventFailed     = "response.failed"
)

// Output item and content-part lifecycle events.
const (
	ResponseEventOutputItemAdded  = "response.output_item.added"
	ResponseEventOutputItemDone   = "response.output_item.done"
	ResponseEventContentPartAdded = "response.content_part.added"
	ResponseEventContentPartDone  = "response.content_part.done"
)

// Text, refusal, annotation and function-argument events.
const (
	ResponseEventOutputTextDelta            = "response.output_text.delta"
	ResponseEventOutputTextDone             = "response.output_text.done"
	ResponseEventOutputTextAnnotationAdded  = "response.output_text.annotation.added"
	ResponseEventRefusalDelta               = "response.refusal.delta"
	ResponseEventRefusalDone                = "response.refusal.done"
	ResponseEventFunctionCallArgumentsDelta = "response.function_call_arguments.delta"
	ResponseEventFunctionCallArgumentsDone  = "response.function_call_arguments.done"
	ResponseEventCustomToolCallInputDelta   = "response.custom_tool_call_input.delta"
	ResponseEventCustomToolCallInputDone    = "response.custom_tool_call_input.done"
)

// Reasoning events.
const (
	ResponseEventReasoningTextDelta        = "response.reasoning_text.delta"
	ResponseEventReasoningTextDone         = "response.reasoning_text.done"
	ResponseEventReasoningSummaryPartAdded = "response.reasoning_summary_part.added"
	ResponseEventReasoningSummaryPartDone  = "response.reasoning_summary_part.done"
	ResponseEventReasoningSummaryTextDelta = "response.reasoning_summary_text.delta"
	ResponseEventReasoningSummaryTextDone  = "response.reasoning_summary_text.done"
)

// Hosted web search progress events.
const (
	ResponseEventWebSearchCallInProgress = "response.web_search_call.in_progress"
	ResponseEventWebSearchCallSearching  = "response.web_search_call.searching"
	ResponseEventWebSearchCallCompleted  = "response.web_search_call.completed"
)

// Hosted file search progress events.
const (
	ResponseEventFileSearchCallInProgress = "response.file_search_call.in_progress"
	ResponseEventFileSearchCallSearching  = "response.file_search_call.searching"
	ResponseEventFileSearchCallCompleted  = "response.file_search_call.completed"
)

// Hosted code interpreter progress events.
const (
	ResponseEventCodeInterpreterCallInProgress   = "response.code_interpreter_call.in_progress"
	ResponseEventCodeInterpreterCallInterpreting = "response.code_interpreter_call.interpreting"
	ResponseEventCodeInterpreterCallCompleted    = "response.code_interpreter_call.completed"
	ResponseEventCodeInterpreterCallCodeDelta    = "response.code_interpreter_call_code.delta"
	ResponseEventCodeInterpreterCallCodeDone     = "response.code_interpreter_call_code.done"
)

// Audio events (models with audio output).
const (
	ResponseEventAudioDelta           = "response.audio.delta"
	ResponseEventAudioDone            = "response.audio.done"
	ResponseEventAudioTranscriptDelta = "response.audio.transcript.delta"
	ResponseEventAudioTranscriptDone  = "response.audio.transcript.done"
)

// Image generation events.
const (
	ResponseEventImageGenerationCallInProgress   = "response.image_generation_call.in_progress"
	ResponseEventImageGenerationCallGenerating   = "response.image_generation_call.generating"
	ResponseEventImageGenerationCallPartialImage = "response.image_generation_call.partial_image"
	ResponseEventImageGenerationCallCompleted    = "response.image_generation_call.completed"
)

// MCP tool events.
const (
	ResponseEventMCPCallArgumentsDelta  = "response.mcp_call_arguments.delta"
	ResponseEventMCPCallArgumentsDone   = "response.mcp_call_arguments.done"
	ResponseEventMCPCallInProgress      = "response.mcp_call.in_progress"
	ResponseEventMCPCallCompleted       = "response.mcp_call.completed"
	ResponseEventMCPCallFailed          = "response.mcp_call.failed"
	ResponseEventMCPListToolsInProgress = "response.mcp_list_tools.in_progress"
	ResponseEventMCPListToolsCompleted  = "response.mcp_list_tools.completed"
	ResponseEventMCPListToolsFailed     = "response.mcp_list_tools.failed"
)

// ResponseEventError is the stream-level error event. The native stream turns
// it into an *HTTPError rather than returning it as an event.
const ResponseEventError = "error"

// responseStreamEventTypes lists every event documented for the 2026-08-01
// baseline, grouped by lifecycle stage.
var responseStreamEventTypes = []string{
	ResponseEventCreated,
	ResponseEventInProgress,
	ResponseEventQueued,
	ResponseEventCompleted,
	ResponseEventIncomplete,
	ResponseEventFailed,
	ResponseEventOutputItemAdded,
	ResponseEventOutputItemDone,
	ResponseEventContentPartAdded,
	ResponseEventContentPartDone,
	ResponseEventOutputTextDelta,
	ResponseEventOutputTextDone,
	ResponseEventOutputTextAnnotationAdded,
	ResponseEventRefusalDelta,
	ResponseEventRefusalDone,
	ResponseEventFunctionCallArgumentsDelta,
	ResponseEventFunctionCallArgumentsDone,
	ResponseEventCustomToolCallInputDelta,
	ResponseEventCustomToolCallInputDone,
	ResponseEventReasoningTextDelta,
	ResponseEventReasoningTextDone,
	ResponseEventReasoningSummaryPartAdded,
	ResponseEventReasoningSummaryPartDone,
	ResponseEventReasoningSummaryTextDelta,
	ResponseEventReasoningSummaryTextDone,
	ResponseEventWebSearchCallInProgress,
	ResponseEventWebSearchCallSearching,
	ResponseEventWebSearchCallCompleted,
	ResponseEventFileSearchCallInProgress,
	ResponseEventFileSearchCallSearching,
	ResponseEventFileSearchCallCompleted,
	ResponseEventCodeInterpreterCallInProgress,
	ResponseEventCodeInterpreterCallInterpreting,
	ResponseEventCodeInterpreterCallCompleted,
	ResponseEventCodeInterpreterCallCodeDelta,
	ResponseEventCodeInterpreterCallCodeDone,
	ResponseEventAudioDelta,
	ResponseEventAudioDone,
	ResponseEventAudioTranscriptDelta,
	ResponseEventAudioTranscriptDone,
	ResponseEventImageGenerationCallInProgress,
	ResponseEventImageGenerationCallGenerating,
	ResponseEventImageGenerationCallPartialImage,
	ResponseEventImageGenerationCallCompleted,
	ResponseEventMCPCallArgumentsDelta,
	ResponseEventMCPCallArgumentsDone,
	ResponseEventMCPCallInProgress,
	ResponseEventMCPCallCompleted,
	ResponseEventMCPCallFailed,
	ResponseEventMCPListToolsInProgress,
	ResponseEventMCPListToolsCompleted,
	ResponseEventMCPListToolsFailed,
	ResponseEventError,
}

// ResponseStreamEventTypes returns the documented baseline event types. An
// event this SDK does not list still reaches the caller with its discriminator
// and raw payload intact.
func ResponseStreamEventTypes() []string {
	types := make([]string, len(responseStreamEventTypes))
	copy(types, responseStreamEventTypes)

	return types
}

// ResponseStreamEvent is one native Responses SSE event.
//
// Responses events share a small field vocabulary: every event carries Type and
// SequenceNumber, most carry the item/content coordinates the event applies to,
// and the composite payloads decode into their dedicated types (Response, Item,
// Part). Raw always holds the verbatim event payload, so an event type this SDK
// does not model still reaches the caller intact.
type ResponseStreamEvent struct {
	Type           string `json:"type"`
	SequenceNumber int    `json:"sequence_number"`

	// Coordinates: which output item, content part, summary part or annotation
	// the event applies to.
	ItemID          string `json:"item_id,omitempty"`
	OutputIndex     int    `json:"output_index,omitempty"`
	ContentIndex    int    `json:"content_index,omitempty"`
	SummaryIndex    int    `json:"summary_index,omitempty"`
	AnnotationIndex int    `json:"annotation_index,omitempty"`

	// Composite payloads.
	Response   *Response           `json:"response,omitempty"`
	Item       *ResponseOutputItem `json:"item,omitempty"`
	Part       *ResponseContent    `json:"part,omitempty"`
	Annotation json.RawMessage     `json:"annotation,omitempty"`
	Logprobs   []ResponseLogprob   `json:"logprobs,omitempty"`

	// Incremental and completed values. Delta carries every `.delta` event's
	// increment; the matching `.done` event carries the completed value in
	// Text, Refusal, Arguments, Code or Input.
	Delta       string `json:"delta,omitempty"`
	Text        string `json:"text,omitempty"`
	Refusal     string `json:"refusal,omitempty"`
	Arguments   string `json:"arguments,omitempty"`
	Code        string `json:"code,omitempty"`
	Input       string `json:"input,omitempty"`
	Name        string `json:"name,omitempty"`
	Status      string `json:"status,omitempty"`
	Obfuscation string `json:"obfuscation,omitempty"`

	// Image generation partial results.
	PartialImageIndex int    `json:"partial_image_index,omitempty"`
	PartialImageB64   string `json:"partial_image_b64,omitempty"`

	// Error event payload. Code above doubles as the error code on an `error`
	// event, matching the wire, which reuses the same key.
	Message string `json:"message,omitempty"`
	Param   string `json:"param,omitempty"`

	// Raw is the verbatim event payload.
	Raw json.RawMessage `json:"-"`
}
