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

// Content block discriminators (`content[].type`). Only the values the
// canonical translation understands are listed; any other block is carried
// through as an extra block on the canonical side.
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

// Message roles are not restated here: Anthropic's "user" / "assistant" are
// the same wire strings as the canonical ais.RoleUser / ais.RoleAssistant.

// Native stop_reason values on a response or a message_delta event. These are
// the wire strings; for the canonical FinishReason values the pass-through
// reasons map to, see the FinishReason constants in extension.go.
const (
	StopReasonEndTurn                    = "end_turn"
	StopReasonStopSequence               = "stop_sequence"
	StopReasonMaxTokens                  = "max_tokens"
	StopReasonToolUse                    = "tool_use"
	StopReasonModelContextWindowExceeded = "model_context_window_exceeded"
	StopReasonRefusal                    = "refusal"
	StopReasonPauseTurn                  = "pause_turn"
)

// tool_choice discriminators. Note "any" (Anthropic) is what the canonical
// "required" translates to — see convertToolChoice.
const (
	ToolChoiceTypeAuto = "auto"
	ToolChoiceTypeAny  = "any"
	ToolChoiceTypeTool = "tool"
	ToolChoiceTypeNone = "none"
)

// CacheControlTypeEphemeral is the only cache_control type Anthropic defines
// today.
const CacheControlTypeEphemeral = "ephemeral"
