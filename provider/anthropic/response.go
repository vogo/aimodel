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

	"github.com/vogo/aimodel/ais"
)

func fromAnthropicResponse(ar *anthropicResponse) *ais.ChatResponse {
	msg := ais.Message{
		Role: ais.RoleAssistant,
	}

	var textParts []string

	var thinkingParts []string

	var extraBlocks []json.RawMessage

	for _, block := range ar.Content {
		switch block.Type {
		case "thinking":
			thinkingParts = append(thinkingParts, block.Thinking)
		case "text":
			textParts = append(textParts, block.Text)

			// A text block may carry citation annotations this wrapper does
			// not promote to the canonical layer. The text is still extracted
			// above; keep the whole original block so the annotations remain
			// reachable.
			if len(block.Citations) > 0 {
				extraBlocks = append(extraBlocks, block.raw)
			}
		case "tool_use":
			msg.ToolCalls = append(msg.ToolCalls, ais.ToolCall{
				Index: len(msg.ToolCalls),
				ID:    block.ID,
				Type:  "function",
				Function: ais.FunctionCall{
					Name:      block.Name,
					Arguments: string(block.Input),
				},
			})
		default:
			// Server-tool blocks (server_tool_use, web_search_tool_result,
			// code_execution_tool_result, …) and any block type added after
			// this wrapper was written. Preserve the original JSON instead of
			// dropping it silently.
			extraBlocks = append(extraBlocks, block.raw)
		}
	}

	if len(thinkingParts) > 0 {
		msg.Thinking = strings.Join(thinkingParts, "\n")
	}

	if len(textParts) > 0 {
		msg.Content = ais.NewTextContent(strings.Join(textParts, "\n"))
	}

	if len(extraBlocks) > 0 {
		msg.Extensions.Set(Name, &MessageExtension{ExtraBlocks: extraBlocks})
	}

	choice := ais.Choice{
		Index:        0,
		Message:      msg,
		FinishReason: mapAnthropicStopReason(ar.StopReason),
	}

	if ar.StopDetails != nil {
		choice.Extensions.Set(Name, &ChoiceExtension{StopDetails: ar.StopDetails})
	}

	cr := &ais.ChatResponse{
		ID:      ar.ID,
		Object:  "chat.completion",
		Model:   ar.Model,
		Choices: []ais.Choice{choice},
		Usage:   anthropicCanonicalUsage(&ar.Usage),
	}

	if ar.Container != nil {
		cr.Extensions.Set(Name, &ResponseExtension{Container: ar.Container})
	}

	return cr
}

// anthropicCanonicalUsage builds a canonical Usage from an Anthropic usage
// object, folding cached/created tokens into PromptTokens (as before). The
// cross-provider counts stay canonical; cache-write totals, the per-TTL
// breakdown, server-tool counts and the inference geography go into the
// UsageExtension namespace (attached only when any of them is present).
func parseDataURI(uri string) (mediaType, data string, ok bool) {
	const prefix = "data:"

	if !strings.HasPrefix(uri, prefix) {
		return "", "", false
	}

	// Format: data:<mediaType>;base64,<data>
	rest := uri[len(prefix):]

	semicolon := strings.Index(rest, ";")
	if semicolon < 0 {
		return "", "", false
	}

	mediaType = rest[:semicolon]

	rest = rest[semicolon+1:]
	if !strings.HasPrefix(rest, "base64,") {
		return "", "", false
	}

	data = rest[len("base64,"):]

	return mediaType, data, true
}

func mapAnthropicStopReason(reason string) ais.FinishReason {
	switch reason {
	case "end_turn", "stop_sequence":
		return ais.FinishReasonStop
	case "max_tokens":
		return ais.FinishReasonLength
	case "tool_use":
		return ais.FinishReasonToolCalls
	case "model_context_window_exceeded":
		return FinishReasonModelContextWindowExceeded
	case "refusal":
		return FinishReasonRefusal
	case "pause_turn":
		return FinishReasonPauseTurn
	default:
		return ais.FinishReason(reason)
	}
}
