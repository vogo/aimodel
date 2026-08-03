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

// Messages API discriminator values, verified 2026-08-02 against
// https://platform.claude.com/docs/en/api/messages. Every field these name
// stays an open string, so a value this SDK does not list still decodes and is
// preserved verbatim (StreamEvent.Raw, ContentBlock.Raw, Delta.Raw).

// SSE event types on a streaming Messages response. StreamEvent.Type carries
// one of these; unlisted types reach the caller with their payload in Raw.
const (
	StreamEventTypeMessageStart      = "message_start"
	StreamEventTypeMessageDelta      = "message_delta"
	StreamEventTypeMessageStop       = "message_stop"
	StreamEventTypeContentBlockStart = "content_block_start"
	StreamEventTypeContentBlockDelta = "content_block_delta"
	StreamEventTypeContentBlockStop  = "content_block_stop"
	StreamEventTypePing              = "ping"
	StreamEventTypeError             = "error"
)

// Content block discriminators (`content[].type`). The field is an open
// string: a block type this list does not name still decodes, and its verbatim
// JSON stays on ResponseContentBlock.Raw.
const (
	ContentBlockTypeText             = "text"
	ContentBlockTypeThinking         = "thinking"
	ContentBlockTypeRedactedThinking = "redacted_thinking"
	ContentBlockTypeImage            = "image"
	ContentBlockTypeDocument         = "document"
	ContentBlockTypeToolUse          = "tool_use"
	ContentBlockTypeToolResult       = "tool_result"
	ContentBlockTypeServerToolUse    = "server_tool_use"
)

// Delta discriminators on a content_block_delta event (`delta.type`).
const (
	DeltaTypeText      = "text_delta"
	DeltaTypeThinking  = "thinking_delta"
	DeltaTypeSignature = "signature_delta"
	DeltaTypeInputJSON = "input_json_delta"
)

// stop_reason values on a response or a message_delta event.
const (
	StopReasonEndTurn                    = "end_turn"
	StopReasonStopSequence               = "stop_sequence"
	StopReasonMaxTokens                  = "max_tokens"
	StopReasonToolUse                    = "tool_use"
	StopReasonModelContextWindowExceeded = "model_context_window_exceeded"
	StopReasonRefusal                    = "refusal"
	StopReasonPauseTurn                  = "pause_turn"
)

// tool_choice discriminators. "any" requires the model to call some tool;
// "tool" pins it to the one named in ToolChoice.Name.
const (
	ToolChoiceTypeAuto = "auto"
	ToolChoiceTypeAny  = "any"
	ToolChoiceTypeTool = "tool"
	ToolChoiceTypeNone = "none"
)

// CacheControlTypeEphemeral is the only cache_control type Anthropic defines
// today.
const CacheControlTypeEphemeral = "ephemeral"
