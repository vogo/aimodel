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
	"io"

	"github.com/vogo/aimodel/core"
)

// Test-scoped aliases let the migrated wire tests keep referencing the
// canonical names they used before the provider moved into this subpackage,
// so the assertions stay a verbatim baseline. Production code in this package
// always qualifies canonical types with core.

type (
	ChatRequest        = core.ChatRequest
	ChatResponse       = core.ChatResponse
	Message            = core.Message
	Content            = core.Content
	ContentPart        = core.ContentPart
	ImageURL           = core.ImageURL
	Tool               = core.Tool
	FunctionDefinition = core.FunctionDefinition
	ToolCall           = core.ToolCall
	FunctionCall       = core.FunctionCall
	Usage              = core.Usage
	Choice             = core.Choice
	StreamChunk        = core.StreamChunk
	Thinking           = core.Thinking
	Error              = core.Error
	APIError           = core.APIError
	FinishReason       = core.FinishReason
)

const (
	RoleSystem    = core.RoleSystem
	RoleUser      = core.RoleUser
	RoleAssistant = core.RoleAssistant
	RoleTool      = core.RoleTool

	FinishReasonStop      = core.FinishReasonStop
	FinishReasonLength    = core.FinishReasonLength
	FinishReasonToolCalls = core.FinishReasonToolCalls

	ReasoningEffortLow    = core.ReasoningEffortLow
	ReasoningEffortMedium = core.ReasoningEffortMedium
	ReasoningEffortHigh   = core.ReasoningEffortHigh
	ReasoningEffortXHigh  = core.ReasoningEffortXHigh

	// Model constants used by the wire tests. The canonical model-name
	// constants live in the root package (protocol-agnostic); these mirror
	// the two the migrated tests reference.
	ModelAnthropicClaude4Opus   = "claude-opus-4"
	ModelAnthropicClaude4Sonnet = "claude-sonnet-4"
)

// cacheMsg returns m with the Anthropic cache-breakpoint extension attached,
// keeping the migrated wire tests close to their pre-extension baseline.
func cacheMsg(m Message) Message {
	ExtendMessage(&m, &MessageExtension{CacheBreakpoint: true})

	return m
}

// cacheTool returns t with the Anthropic cache-breakpoint extension attached.
func cacheTool(t Tool) Tool {
	ExtendTool(&t, &ToolExtension{CacheBreakpoint: true})

	return t
}

// autoCache attaches the automatic-caching request extension to req.
func autoCache(req *ChatRequest, ttl string) {
	ExtendRequest(req, &RequestExtension{AutoCache: true, AutoCacheTTL: ttl})
}

// extraBlocksOf returns the unmodelled blocks preserved on a message's
// Anthropic extension, or nil when it carries none.
func extraBlocksOf(m *Message) []json.RawMessage {
	if ext := MessageExtensionOf(m); ext != nil {
		return ext.ExtraBlocks
	}

	return nil
}

// chunkContainer returns the execution container reported on a stream chunk's
// Anthropic extension, or nil.
func chunkContainer(c *StreamChunk) *ResponseContainer {
	if ext := ChunkExtensionOf(c); ext != nil {
		return ext.Container
	}

	return nil
}

var (
	ErrEmptyResponse = core.ErrEmptyResponse
	ErrStreamClosed  = core.ErrStreamClosed
)

func NewTextContent(text string) core.Content { return core.NewTextContent(text) }

func NewPartsContent(parts ...core.ContentPart) core.Content {
	return core.NewPartsContent(parts...)
}

// newAnthropicStream adapts the provider's SSE decoder to the Recv/Close shape
// the migrated stream tests were written against. It reproduces just enough of
// the root Stream's close contract (Recv after Close yields ErrStreamClosed) to
// keep those tests intact; the full Stream lifecycle is covered in the root
// package.
func newAnthropicStream(body io.ReadCloser) *recvStream {
	p := &provider{version: anthropicAPIVersion}

	return &recvStream{decoder: p.NewStreamDecoder(body), body: body}
}

type recvStream struct {
	decoder core.StreamDecoder
	body    io.ReadCloser
	closed  bool
}

func (s *recvStream) Recv() (*core.StreamChunk, error) {
	if s.closed {
		return nil, core.ErrStreamClosed
	}

	return s.decoder.Next()
}

func (s *recvStream) Close() error {
	s.closed = true

	return s.body.Close()
}
