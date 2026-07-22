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
	"fmt"
	"strings"

	"github.com/vogo/aimodel/ais"
)

// --- Translation functions ---

// toAnthropicRequest converts a ais.ChatRequest to an Anthropic API request.
// Anthropic-only parameters arrive through the unified extension channel
// (RequestExtension / MessageExtension / ToolExtension); a mis-typed
// extension value fails here — before any network I/O — with a
// *ais.ExtensionTypeError.
func toAnthropicRequest(req *ais.ChatRequest) (*MessagesRequest, error) {
	reqExt, err := extensionOf[RequestExtension](req.Extensions, "ChatRequest")
	if err != nil {
		return nil, err
	}
	if reqExt == nil {
		reqExt = &RequestExtension{}
	}

	ar := newAnthropicRequest(req, reqExt)
	if err := setAnthropicMessages(ar, req.Messages); err != nil {
		return nil, err
	}
	if err := setAnthropicTools(ar, req.Tools); err != nil {
		return nil, err
	}

	ar.ToolChoice = toAnthropicToolChoice(req)
	if req.Thinking != nil {
		ar.Thinking = &MessagesThinking{
			Type:         req.Thinking.Type,
			BudgetTokens: req.Thinking.BudgetTokens, //nolint:staticcheck // deprecated compatibility field
			Display:      req.Thinking.Display,
		}
	}
	ar.OutputConfig = toAnthropicOutputConfig(req)
	setAnthropicAutoCache(ar, reqExt)

	return ar, nil
}

func newAnthropicRequest(req *ais.ChatRequest, ext *RequestExtension) *MessagesRequest {
	ar := &MessagesRequest{
		Model:        req.Model,
		Temperature:  req.Temperature,
		TopP:         req.TopP,
		TopK:         req.TopK,
		Stream:       req.Stream,
		Container:    ext.Container,
		InferenceGeo: ext.InferenceGeo,
	}
	if len(req.Stop) > 0 {
		ar.StopSequences = req.Stop
	}

	switch {
	case req.MaxCompletionTokens != nil:
		ar.MaxTokens = *req.MaxCompletionTokens
	case req.MaxTokens != nil: //nolint:staticcheck // deprecated field read on purpose
		ar.MaxTokens = *req.MaxTokens //nolint:staticcheck // deprecated field read on purpose
	default:
		ar.MaxTokens = anthropicDefaultMaxTokens
	}

	return ar
}

func setAnthropicMessages(ar *MessagesRequest, messages []ais.Message) error {
	var systemTexts []string
	var systemBlocks []ContentBlock
	var useBlocks bool
	var anyCacheableSystem bool
	var seenNonSystem bool

	for i := 0; i < len(messages); i++ {
		m := messages[i]
		if m.Role == ais.RoleSystem && !seenNonSystem {
			bp, err := messageCacheBreakpoint(&m)
			if err != nil {
				return err
			}
			if bp {
				anyCacheableSystem = true
				useBlocks = true
			}
			if parts := m.Content.Parts(); len(parts) > 0 {
				useBlocks = true
				for _, p := range parts {
					if p.Type == "text" {
						systemBlocks = append(systemBlocks, ContentBlock{Type: "text", Text: p.Text})
					}
				}
			} else {
				text := m.Content.Text()
				systemTexts = append(systemTexts, text)
				systemBlocks = append(systemBlocks, ContentBlock{Type: "text", Text: text})
			}
			continue
		}

		if m.Role != ais.RoleSystem {
			seenNonSystem = true
		}
		if m.Role == ais.RoleTool {
			runStart := i
			for i+1 < len(messages) && messages[i+1].Role == ais.RoleTool {
				i++
			}
			am, err := toAnthropicToolResultMessage(messages[runStart : i+1])
			if err != nil {
				return err
			}
			ar.Messages = append(ar.Messages, am)
			continue
		}

		am, err := toAnthropicMessage(m)
		if err != nil {
			return err
		}
		ar.Messages = append(ar.Messages, am)
	}

	system, err := marshalAnthropicSystem(systemTexts, systemBlocks, useBlocks, anyCacheableSystem)
	if err != nil {
		return err
	}
	ar.System = system

	return nil
}

func marshalAnthropicSystem(texts []string, blocks []ContentBlock, useBlocks, cacheable bool) (json.RawMessage, error) {
	if (useBlocks || cacheable) && len(blocks) > 0 {
		if cacheable {
			blocks[len(blocks)-1].CacheControl = ephemeralCache()
		}
		data, err := json.Marshal(blocks)
		if err != nil {
			return nil, fmt.Errorf("aimodel: marshal system content: %w", err)
		}
		return data, nil
	}
	if len(texts) == 0 {
		return nil, nil
	}

	data, err := json.Marshal(strings.Join(texts, "\n"))
	if err != nil {
		return nil, fmt.Errorf("aimodel: marshal system text: %w", err)
	}
	return data, nil
}

func setAnthropicTools(ar *MessagesRequest, tools []ais.Tool) error {
	for _, t := range tools {
		tExt, err := extensionOf[ToolExtension](t.Extensions, "Tool")
		if err != nil {
			return err
		}
		if tExt == nil {
			tExt = &ToolExtension{}
		}

		at := MessagesTool{
			Name:                t.Function.Name,
			Description:         t.Function.Description,
			InputSchema:         t.Function.Parameters,
			Strict:              t.Strict,
			DeferLoading:        tExt.DeferLoading,
			AllowedCallers:      tExt.AllowedCallers,
			EagerInputStreaming: tExt.EagerInputStreaming,
			InputExamples:       tExt.InputExamples,
		}
		if t.Type != "" && t.Type != "function" {
			at.Type = t.Type
		}
		if tExt.CacheBreakpoint {
			at.CacheControl = ephemeralCache()
		}
		ar.Tools = append(ar.Tools, at)
	}
	return nil
}

func toAnthropicToolChoice(req *ais.ChatRequest) *ToolChoice {
	tc := convertToolChoice(req.ToolChoice)
	if req.ParallelToolCalls != nil && !*req.ParallelToolCalls {
		if tc == nil && len(req.Tools) > 0 {
			tc = &ToolChoice{Type: "auto"}
		}
		if tc != nil && tc.Type != "none" {
			disable := true
			tc.DisableParallelToolUse = &disable
		}
	}
	return tc
}

func setAnthropicAutoCache(ar *MessagesRequest, ext *RequestExtension) {
	if ext.AutoCache {
		ar.CacheControl = &CacheControl{Type: "ephemeral", TTL: ext.AutoCacheTTL}
	}
}

// messageCacheBreakpoint reports whether the message's Anthropic extension
// asks for a prompt-cache boundary, failing on a mis-typed extension value.
func messageCacheBreakpoint(m *ais.Message) (bool, error) {
	ext, err := extensionOf[MessageExtension](m.Extensions, "Message")
	if err != nil {
		return false, err
	}

	return ext != nil && ext.CacheBreakpoint, nil
}

// toAnthropicOutputConfig builds the output_config object from the canonical
// reasoning effort and response format. Either half may be absent; when both
// are, it returns nil so the field is omitted entirely.
func toAnthropicOutputConfig(req *ais.ChatRequest) *OutputConfig {
	format := toAnthropicOutputFormat(req.ResponseFormat)
	if req.ReasoningEffort == "" && format == nil {
		return nil
	}

	return &OutputConfig{
		Effort: req.ReasoningEffort,
		Format: format,
	}
}

// toAnthropicOutputFormat translates a canonical ResponseFormat into
// Anthropic's structured-output format. Only JSON-schema shapes are
// recognized: OpenAI's {type:"json_schema", json_schema:{schema:…}} and the
// flat {type:"json_schema", schema:…}. Anything else — including
// {type:"json_object"}, which has no Anthropic counterpart — yields nil rather
// than a fabricated config, keeping this a thin translation.
func toAnthropicOutputFormat(rf any) *OutputFormat {
	m, ok := rf.(map[string]any)
	if !ok {
		return nil
	}

	if t, _ := m["type"].(string); t != "json_schema" {
		return nil
	}

	schema := m["schema"]
	if nested, ok := m["json_schema"].(map[string]any); ok {
		schema = nested["schema"]
	}

	if schema == nil {
		return nil
	}

	return &OutputFormat{Type: "json_schema", Schema: schema}
}

// toolResultBlock builds a single tool_result content block from a canonical
// tool-result message. It is shared by the consecutive-run merge path and the
// single-message fallback so the wire shape stays identical. A missing
// ToolCallID is rejected up front.
func toolResultBlock(m ais.Message) (ContentBlock, error) {
	if m.ToolCallID == "" {
		return ContentBlock{}, fmt.Errorf("aimodel: tool result message missing tool_call_id")
	}

	block := ContentBlock{
		Type:          "tool_result",
		ToolUseID:     m.ToolCallID,
		ResultContent: m.Content.Text(),
	}

	bp, err := messageCacheBreakpoint(&m)
	if err != nil {
		return ContentBlock{}, err
	}

	if bp {
		block.CacheControl = ephemeralCache()
	}

	return block, nil
}

// toAnthropicToolResultMessage serializes a run of one or more consecutive
// canonical tool-result messages into a single Anthropic role:"user" message
// whose content array holds all the tool_result blocks in order. Anthropic
// requires the parallel results of one assistant turn to share one user
// message; merging here keeps the request valid for parallel tool use.
func toAnthropicToolResultMessage(msgs []ais.Message) (MessagesMessage, error) {
	blocks := make([]ContentBlock, 0, len(msgs))
	for _, m := range msgs {
		block, err := toolResultBlock(m)
		if err != nil {
			return MessagesMessage{}, err
		}

		blocks = append(blocks, block)
	}

	data, err := json.Marshal(blocks)
	if err != nil {
		return MessagesMessage{}, fmt.Errorf("aimodel: marshal tool result: %w", err)
	}

	return MessagesMessage{Role: "user", Content: data}, nil
}

func toAnthropicMessage(m ais.Message) (MessagesMessage, error) {
	am := MessagesMessage{
		Role: string(m.Role),
	}

	// Tool result messages become user messages with tool_result content blocks.
	if m.Role == ais.RoleTool {
		return toAnthropicToolResultMessage([]ais.Message{m})
	}

	cacheBreakpoint, err := messageCacheBreakpoint(&m)
	if err != nil {
		return MessagesMessage{}, err
	}

	// Assistant messages with thinking, tool calls, or both require content-block format.
	if m.Role == ais.RoleAssistant && (m.Thinking != "" || len(m.ToolCalls) > 0) {
		var blocks []ContentBlock

		if m.Thinking != "" {
			blocks = append(blocks, ContentBlock{
				Type:     "thinking",
				Thinking: m.Thinking,
			})
		}

		text := m.Content.Text()
		if text != "" {
			blocks = append(blocks, ContentBlock{
				Type: "text",
				Text: text,
			})
		}

		for _, tc := range m.ToolCalls {
			blocks = append(blocks, ContentBlock{
				Type:  "tool_use",
				ID:    tc.ID,
				Name:  tc.Function.Name,
				Input: json.RawMessage(tc.Function.Arguments),
			})
		}

		if cacheBreakpoint && len(blocks) > 0 {
			blocks[len(blocks)-1].CacheControl = ephemeralCache()
		}

		data, err := json.Marshal(blocks)
		if err != nil {
			return MessagesMessage{}, fmt.Errorf("aimodel: marshal assistant content: %w", err)
		}

		am.Content = data

		return am, nil
	}

	// Multimodal content with parts.
	if parts := m.Content.Parts(); len(parts) > 0 {
		var blocks []ContentBlock

		for _, p := range parts {
			switch p.Type {
			case "text":
				blocks = append(blocks, ContentBlock{
					Type: "text",
					Text: p.Text,
				})
			case "image_url":
				if p.ImageURL == nil {
					continue
				}

				block := ContentBlock{Type: "image"}

				if mediaType, b64Data, ok := parseDataURI(p.ImageURL.URL); ok {
					block.Source = &ContentSource{
						Type:      "base64",
						MediaType: mediaType,
						Data:      b64Data,
					}
				} else {
					block.Source = &ContentSource{
						Type: "url",
						URL:  p.ImageURL.URL,
					}
				}

				blocks = append(blocks, block)
			}
		}

		if cacheBreakpoint && len(blocks) > 0 {
			blocks[len(blocks)-1].CacheControl = ephemeralCache()
		}

		data, err := json.Marshal(blocks)
		if err != nil {
			return MessagesMessage{}, fmt.Errorf("aimodel: marshal multimodal content: %w", err)
		}

		am.Content = data

		return am, nil
	}

	// Plain text message. Promote to block-array form when the caller
	// flagged CacheBreakpoint so we can attach cache_control.
	if cacheBreakpoint {
		block := ContentBlock{
			Type:         "text",
			Text:         m.Content.Text(),
			CacheControl: ephemeralCache(),
		}
		data, err := json.Marshal([]ContentBlock{block})
		if err != nil {
			return MessagesMessage{}, fmt.Errorf("aimodel: marshal cached message content: %w", err)
		}
		am.Content = data
		return am, nil
	}

	data, err := json.Marshal(m.Content.Text())
	if err != nil {
		return MessagesMessage{}, fmt.Errorf("aimodel: marshal message content: %w", err)
	}

	am.Content = data

	return am, nil
}

func convertToolChoice(tc any) *ToolChoice {
	switch v := tc.(type) {
	case string:
		switch v {
		case "auto":
			return &ToolChoice{Type: "auto"}
		case "required":
			return &ToolChoice{Type: "any"}
		case "none":
			// Explicit "none" forbids any tool call; an omitted tool_choice
			// would instead let the model choose, so emit {type:"none"}.
			return &ToolChoice{Type: "none"}
		}
	case map[string]any:
		if fn, ok := v["function"].(map[string]any); ok {
			if name, ok := fn["name"].(string); ok {
				return &ToolChoice{Type: "tool", Name: name}
			}
		}
	}

	return nil
}

// fromAnthropicResponse converts an Anthropic API response to a ais.ChatResponse.
// Anthropic-only response information — unmodelled content blocks, structured
// stop details, the execution container, cache-write accounting — is written
// into this provider's extension namespaces instead of canonical fields.
