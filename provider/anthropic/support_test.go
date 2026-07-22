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

	"github.com/vogo/aimodel/ais"
)

// Test-scoped aliases let the migrated wire tests keep referencing the
// canonical names they used before the provider moved into this subpackage,
// so the assertions stay a verbatim baseline. Production code in this package
// always qualifies canonical types with ais.

type (
	ChatRequest        = ais.ChatRequest
	ChatResponse       = ais.ChatResponse
	Message            = ais.Message
	Content            = ais.Content
	ContentPart        = ais.ContentPart
	ImageURL           = ais.ImageURL
	Tool               = ais.Tool
	FunctionDefinition = ais.FunctionDefinition
	ToolCall           = ais.ToolCall
	FunctionCall       = ais.FunctionCall
	Usage              = ais.Usage
	Choice             = ais.Choice
	StreamChunk        = ais.StreamChunk
	Thinking           = ais.Thinking
	Error              = ais.Error
	APIError           = ais.APIError
	FinishReason       = ais.FinishReason
)

const (
	RoleSystem    = ais.RoleSystem
	RoleUser      = ais.RoleUser
	RoleAssistant = ais.RoleAssistant
	RoleTool      = ais.RoleTool

	FinishReasonStop      = ais.FinishReasonStop
	FinishReasonLength    = ais.FinishReasonLength
	FinishReasonToolCalls = ais.FinishReasonToolCalls

	ReasoningEffortLow    = ais.ReasoningEffortLow
	ReasoningEffortMedium = ais.ReasoningEffortMedium
	ReasoningEffortHigh   = ais.ReasoningEffortHigh
	ReasoningEffortXHigh  = ais.ReasoningEffortXHigh

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
	ErrEmptyResponse = ais.ErrEmptyResponse
	ErrStreamClosed  = ais.ErrStreamClosed
)

func NewTextContent(text string) ais.Content { return ais.NewTextContent(text) }

func NewPartsContent(parts ...ais.ContentPart) ais.Content {
	return ais.NewPartsContent(parts...)
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
	decoder ais.StreamDecoder
	body    io.ReadCloser
	closed  bool
}

func (s *recvStream) Recv() (*ais.StreamChunk, error) {
	if s.closed {
		return nil, ais.ErrStreamClosed
	}

	return s.decoder.Next()
}

func (s *recvStream) Close() error {
	s.closed = true

	return s.body.Close()
}
