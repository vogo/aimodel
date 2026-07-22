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

import "github.com/vogo/aimodel/ais"

func toOpenAIRequest(input *ais.ChatRequest) *ChatCompletionRequest {
	request := &ChatCompletionRequest{
		// MaxTokens remains mapped for backward compatibility with older models.
		Model: input.Model, Temperature: input.Temperature, MaxTokens: input.MaxTokens, //nolint:staticcheck
		MaxCompletionTokens: input.MaxCompletionTokens, TopP: input.TopP, TopK: input.TopK, Stop: append([]string(nil), input.Stop...),
		ResponseFormat: input.ResponseFormat, Stream: input.Stream, ToolChoice: input.ToolChoice,
		ReasoningEffort: input.ReasoningEffort, ParallelToolCalls: input.ParallelToolCalls,
	}
	if input.Thinking != nil {
		// BudgetTokens remains mapped for compatible backends that still accept it.
		request.Thinking = &Thinking{Type: input.Thinking.Type, BudgetTokens: input.Thinking.BudgetTokens, Display: input.Thinking.Display} //nolint:staticcheck
	}
	if input.Stream {
		yes := true
		request.StreamOptions = &StreamOptions{IncludeUsage: &yes}
	}
	for _, message := range input.Messages {
		wire := ChatCompletionMessage{Role: string(message.Role), ReasoningContent: message.Thinking, ToolCallID: message.ToolCallID}
		if parts := message.Content.Parts(); parts != nil {
			converted := make([]ChatCompletionContentPart, 0, len(parts))
			for _, part := range parts {
				item := ChatCompletionContentPart{Type: part.Type, Text: part.Text}
				if part.ImageURL != nil {
					item.ImageURL = &ImageURL{URL: part.ImageURL.URL, Detail: part.ImageURL.Detail}
				}
				converted = append(converted, item)
			}
			wire.Content = NewPartsContent(converted...)
		} else {
			wire.Content = NewTextContent(message.Content.Text())
		}
		for _, call := range message.ToolCalls {
			wire.ToolCalls = append(wire.ToolCalls, ChatCompletionToolCall{Index: call.Index, ID: call.ID, Type: call.Type, Function: ChatCompletionFunctionCall{Name: call.Function.Name, Arguments: call.Function.Arguments}})
		}
		request.Messages = append(request.Messages, wire)
	}
	for _, tool := range input.Tools {
		request.Tools = append(request.Tools, ChatCompletionTool{Type: tool.Type, Function: ChatCompletionFunction{Name: tool.Function.Name, Description: tool.Function.Description, Parameters: tool.Function.Parameters, Strict: tool.Strict}})
	}
	return request
}

func fromOpenAIMessage(input ChatCompletionMessage) ais.Message {
	message := ais.Message{Role: ais.Role(input.Role), Thinking: input.ReasoningContent, ToolCallID: input.ToolCallID}
	if parts := input.Content.Parts(); parts != nil {
		converted := make([]ais.ContentPart, 0, len(parts))
		for _, part := range parts {
			item := ais.ContentPart{Type: part.Type, Text: part.Text}
			if part.ImageURL != nil {
				item.ImageURL = &ais.ImageURL{URL: part.ImageURL.URL, Detail: part.ImageURL.Detail}
			}
			converted = append(converted, item)
		}
		message.Content = ais.NewPartsContent(converted...)
	} else {
		message.Content = ais.NewTextContent(input.Content.Text())
	}
	for _, call := range input.ToolCalls {
		message.ToolCalls = append(message.ToolCalls, ais.ToolCall{Index: call.Index, ID: call.ID, Type: call.Type, Function: ais.FunctionCall{Name: call.Function.Name, Arguments: call.Function.Arguments}})
	}
	return message
}

func fromOpenAIUsage(input *ChatCompletionUsage, serviceTier string) *ais.Usage {
	if input == nil {
		return nil
	}
	result := &ais.Usage{PromptTokens: input.PromptTokens, CompletionTokens: input.CompletionTokens, TotalTokens: input.TotalTokens, ServiceTier: serviceTier}
	if input.PromptTokensDetails != nil {
		result.CacheReadTokens = input.PromptTokensDetails.CachedTokens
	}
	if input.CompletionTokensDetails != nil {
		result.ReasoningTokens = input.CompletionTokensDetails.ReasoningTokens
	}
	return result
}

func fromOpenAIResponse(input *ChatCompletionResponse) *ais.ChatResponse {
	result := &ais.ChatResponse{ID: input.ID, Object: input.Object, Created: input.Created, Model: input.Model}
	if usage := fromOpenAIUsage(input.Usage, input.ServiceTier); usage != nil {
		result.Usage = *usage
	}
	for _, choice := range input.Choices {
		finish := ais.FinishReason("")
		if choice.FinishReason != nil {
			finish = ais.FinishReason(*choice.FinishReason)
		}
		result.Choices = append(result.Choices, ais.Choice{Index: choice.Index, Message: fromOpenAIMessage(choice.Message), FinishReason: finish})
	}
	return result
}

func fromOpenAIChunk(input *ChatCompletionChunk) *ais.StreamChunk {
	result := &ais.StreamChunk{ID: input.ID, Object: input.Object, Created: input.Created, Model: input.Model, Usage: fromOpenAIUsage(input.Usage, input.ServiceTier)}
	for _, choice := range input.Choices {
		result.Choices = append(result.Choices, ais.StreamChunkChoice{Index: choice.Index, Delta: fromOpenAIMessage(choice.Delta), FinishReason: choice.FinishReason})
	}
	return result
}
