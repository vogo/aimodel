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

import "strings"

// objectChatCompletion is the object discriminator of a non-streaming
// completion. Chunks carry "chat.completion.chunk"; the accumulated result
// reports the unary value it reconstructs.
const objectChatCompletion = "chat.completion"

// chatAccumulator folds streaming chunks into the response the same request
// would have produced without `stream: true`. Text, reasoning content, refusals
// and tool-call argument fragments concatenate in arrival order; scalar fields
// take the last non-empty value.
type chatAccumulator struct {
	response ChatCompletionResponse
	started  bool
	choices  []*choiceAccumulator
}

// choiceAccumulator holds the in-progress concatenations of one choice. It is
// referenced by pointer so growing the slice never copies a strings.Builder.
type choiceAccumulator struct {
	text      strings.Builder
	reasoning strings.Builder
	refusal   strings.Builder
	toolArgs  []*strings.Builder
}

// fold merges one chunk into the accumulated response.
func (a *chatAccumulator) fold(chunk *ChatCompletionChunk) {
	if chunk == nil {
		return
	}

	if !a.started {
		a.started = true
		a.response.ID = chunk.ID
		a.response.Object = objectChatCompletion
		a.response.Created = chunk.Created
		a.response.Model = chunk.Model
	}

	if chunk.ServiceTier != "" {
		a.response.ServiceTier = chunk.ServiceTier
	}

	if chunk.SystemFingerprint != "" {
		a.response.SystemFingerprint = chunk.SystemFingerprint
	}

	// The usage-bearing chunk is terminal and carries the totals for the whole
	// completion, so it replaces rather than adds to what came before.
	if chunk.Usage != nil {
		a.response.Usage = chunk.Usage
	}

	for i := range chunk.Choices {
		a.foldChoice(&chunk.Choices[i])
	}
}

// foldChoice merges one chunk choice, growing the choice list as indexes
// arrive. Providers may emit choices out of order or skip indexes.
func (a *chatAccumulator) foldChoice(chunk *ChatCompletionChunkChoice) {
	if chunk.Index < 0 {
		return
	}

	for len(a.response.Choices) <= chunk.Index {
		a.response.Choices = append(a.response.Choices, ChatCompletionChoice{Index: len(a.response.Choices)})
		a.choices = append(a.choices, &choiceAccumulator{})
	}

	choice, acc, delta := &a.response.Choices[chunk.Index], a.choices[chunk.Index], &chunk.Delta

	if delta.Role != "" {
		choice.Message.Role = delta.Role
	}

	if delta.Name != "" {
		choice.Message.Name = delta.Name
	}

	if text := delta.Content.Text(); text != "" {
		acc.text.WriteString(text)
		choice.Message.Content = NewTextContent(acc.text.String())
	}

	if delta.ReasoningContent != "" {
		acc.reasoning.WriteString(delta.ReasoningContent)
		choice.Message.ReasoningContent = acc.reasoning.String()
	}

	if delta.Refusal != "" {
		acc.refusal.WriteString(delta.Refusal)
		choice.Message.Refusal = acc.refusal.String()
	}

	if delta.Audio != nil {
		choice.Message.Audio = delta.Audio
	}

	for i := range delta.ToolCalls {
		acc.foldToolCall(choice, &delta.ToolCalls[i])
	}

	if chunk.FinishReason != nil {
		choice.FinishReason = chunk.FinishReason
	}

	foldLogprobs(choice, chunk.Logprobs)
}

// foldToolCall merges one tool-call fragment. Identity fields take the last
// non-empty value; arguments concatenate, since the model streams the JSON in
// pieces that are only valid once complete.
func (acc *choiceAccumulator) foldToolCall(choice *ChatCompletionChoice, delta *ChatCompletionToolCall) {
	if delta.Index < 0 {
		return
	}

	for len(choice.Message.ToolCalls) <= delta.Index {
		choice.Message.ToolCalls = append(choice.Message.ToolCalls,
			ChatCompletionToolCall{Index: len(choice.Message.ToolCalls)})
		acc.toolArgs = append(acc.toolArgs, &strings.Builder{})
	}

	call := &choice.Message.ToolCalls[delta.Index]

	if delta.ID != "" {
		call.ID = delta.ID
	}

	if delta.Type != "" {
		call.Type = delta.Type
	}

	if delta.Function.Name != "" {
		call.Function.Name = delta.Function.Name
	}

	if delta.Function.Arguments != "" {
		args := acc.toolArgs[delta.Index]
		args.WriteString(delta.Function.Arguments)
		call.Function.Arguments = args.String()
	}
}

// foldLogprobs appends the token log probabilities of one chunk.
func foldLogprobs(choice *ChatCompletionChoice, delta *ChoiceLogprobs) {
	if delta == nil {
		return
	}

	if choice.Logprobs == nil {
		choice.Logprobs = &ChoiceLogprobs{}
	}

	choice.Logprobs.Content = append(choice.Logprobs.Content, delta.Content...)
	choice.Logprobs.Refusal = append(choice.Logprobs.Refusal, delta.Refusal...)
}

// result returns the accumulated response, or nil when no chunk arrived.
func (a *chatAccumulator) result() *ChatCompletionResponse {
	if !a.started {
		return nil
	}

	return &a.response
}
