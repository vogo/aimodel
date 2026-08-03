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
	"slices"
	"strings"
)

// messageAccumulator folds streaming events into the message the same request
// would have returned without `stream: true`. Text and thinking deltas
// concatenate onto their content block, tool inputs are reassembled from their
// partial-JSON fragments, and the terminal event supplies the stop reason and
// the final usage.
type messageAccumulator struct {
	response MessagesResponse
	started  bool

	// toolInput holds the partial-JSON fragments of each content block, indexed
	// like response.Content. Pointers keep growing the slice from copying a
	// strings.Builder.
	toolInput []*strings.Builder
}

// fold merges one event into the accumulated message.
func (a *messageAccumulator) fold(event *StreamEvent) {
	if event == nil {
		return
	}

	switch {
	case event.MessageStart != nil:
		// Copy the message and own a private Content slice. Block assembly
		// below writes into response.Content in place, and must never touch the
		// backing array of the caller's message_start event — even if a backend
		// one day pre-populates content there instead of sending an empty array.
		msg := event.MessageStart.Message
		msg.Content = slices.Clone(msg.Content)
		a.response = msg
		a.started = true
		a.toolInput = a.toolInput[:0]

		for range a.response.Content {
			a.toolInput = append(a.toolInput, &strings.Builder{})
		}
	case event.ContentBlockStart != nil:
		a.foldBlockStart(event.ContentBlockStart)
	case event.ContentBlockDelta != nil:
		a.foldBlockDelta(event.ContentBlockDelta)
	case event.MessageDelta != nil:
		a.foldMessageDelta(event.MessageDelta)
	}
}

// foldBlockStart installs a new content block at its index, growing the block
// list when a provider skips or reorders indexes.
func (a *messageAccumulator) foldBlockStart(event *ContentBlockStartEvent) {
	if event.Index < 0 {
		return
	}

	a.started = true
	a.growBlocks(event.Index)
	a.response.Content[event.Index] = event.ContentBlock
}

// foldBlockDelta appends one delta to its content block. Unrecognized delta
// types change nothing — their payload stays available on StreamEvent.Raw.
func (a *messageAccumulator) foldBlockDelta(event *ContentBlockDeltaEvent) {
	if event.Index < 0 {
		return
	}

	a.started = true
	a.growBlocks(event.Index)

	block := &a.response.Content[event.Index]

	switch event.Delta.Type {
	case DeltaTypeText:
		block.Text += event.Delta.Text
	case DeltaTypeThinking:
		block.Thinking += event.Delta.Thinking
	case DeltaTypeInputJSON:
		input := a.toolInput[event.Index]
		input.WriteString(event.Delta.PartialJSON)
		block.Input = json.RawMessage(input.String())
	}
}

// foldMessageDelta applies the terminal event: the stop classification and the
// final token counts.
func (a *messageAccumulator) foldMessageDelta(event *MessageDeltaEvent) {
	a.started = true

	if event.Delta.StopReason != "" {
		a.response.StopReason = event.Delta.StopReason
	}

	if event.Delta.StopSequence != nil {
		a.response.StopSequence = event.Delta.StopSequence
	}

	if event.Delta.StopDetails != nil {
		a.response.StopDetails = event.Delta.StopDetails
	}

	if event.Usage != nil {
		mergeAnthropicUsage(&a.response.Usage, event.Usage)
	}
}

// growBlocks extends the content-block list so index is addressable.
func (a *messageAccumulator) growBlocks(index int) {
	for len(a.response.Content) <= index {
		a.response.Content = append(a.response.Content, ResponseContentBlock{})
		a.toolInput = append(a.toolInput, &strings.Builder{})
	}
}

// result returns the accumulated message, or nil when no event carried any.
func (a *messageAccumulator) result() *MessagesResponse {
	if !a.started {
		return nil
	}

	return &a.response
}

// mergeAnthropicUsage folds a later usage object (the terminal message_delta)
// into the baseline captured at message_start. Only fields the later object
// actually carries are applied — a terminal event that reports just
// output_tokens must not blank out the input, cache, geo, tier or server-tool
// information already established.
func mergeAnthropicUsage(base, next *MessagesUsage) {
	if next.InputTokens != 0 {
		base.InputTokens = next.InputTokens
	}

	if next.OutputTokens != 0 {
		base.OutputTokens = next.OutputTokens
	}

	if next.CacheCreationInputTokens != 0 {
		base.CacheCreationInputTokens = next.CacheCreationInputTokens
	}

	if next.CacheReadInputTokens != 0 {
		base.CacheReadInputTokens = next.CacheReadInputTokens
	}

	// Copy the pointed-to values rather than adopting the pointers: the
	// accumulator owns its usage and must not share sub-structs with the
	// caller's terminal message_delta event.
	if next.CacheCreation != nil {
		cacheCreation := *next.CacheCreation
		base.CacheCreation = &cacheCreation
	}

	if next.OutputTokensDetails != nil {
		details := *next.OutputTokensDetails
		base.OutputTokensDetails = &details
	}

	if next.ServerToolUse != nil {
		serverToolUse := *next.ServerToolUse
		base.ServerToolUse = &serverToolUse
	}

	if next.InferenceGeo != "" {
		base.InferenceGeo = next.InferenceGeo
	}

	if next.ServiceTier != "" {
		base.ServiceTier = next.ServiceTier
	}
}
