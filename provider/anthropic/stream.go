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
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/vogo/aimodel/core"
)

// NewStreamDecoder returns a decoder for the Anthropic SSE event stream.
// Streaming reference: https://platform.claude.com/docs/en/api/messages
func (p *provider) NewStreamDecoder(body io.Reader) core.StreamDecoder {
	sc := bufio.NewScanner(body)
	sc.Buffer(make([]byte, 0, 64*1024), core.MaxStreamLineSize)

	return &streamDecoder{
		sc:            sc,
		blockToTool:   make(map[int]int),
		unknownBlocks: make(map[int]bool),
	}
}

// extraBlockDelta wraps one verbatim unmodelled sub-object as a delta message
// whose Anthropic extension carries it. Message.AppendDelta accumulates these
// through MessageExtension.MergeExtension, preserving arrival order.
func extraBlockDelta(raw json.RawMessage) core.Message {
	var m core.Message

	m.Extensions.Set(Name, &MessageExtension{ExtraBlocks: []json.RawMessage{raw}})

	return m
}

type streamDecoder struct {
	sc *bufio.Scanner

	msgID string
	model string
	// startUsage captures the input/cache token counts from message_start;
	// the final output_tokens arrives later on message_delta.
	startUsage anthropicUsage

	// blockToTool maps Anthropic content block index to tool call index.
	// Anthropic uses sequential indices for all content blocks (text, thinking,
	// tool_use), but AppendDelta expects indices scoped to tool calls only.
	blockToTool map[int]int
	nextToolIdx int

	// unknownBlocks records the indices whose content_block_start carried a
	// type this wrapper does not model, so their subsequent deltas are
	// preserved verbatim instead of being interpreted or dropped.
	unknownBlocks map[int]bool
}

//nolint:gocyclo // Faithful 1:1 port of the Anthropic SSE event switch.
func (d *streamDecoder) Next() (*core.StreamChunk, error) {
	sc := d.sc

	for sc.Scan() {
		line := sc.Text()

		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}

		// Parse "event: <type>" line.
		if !strings.HasPrefix(line, "event: ") {
			continue
		}

		eventType := strings.TrimPrefix(line, "event: ")

		// Scan past empty lines and comments to find the data line.
		var data []byte

		for sc.Scan() {
			dataLine := sc.Text()
			if dataLine == "" || strings.HasPrefix(dataLine, ":") {
				continue
			}

			if after, ok := strings.CutPrefix(dataLine, "data: "); ok {
				data = []byte(after)

				break
			}

			// Non-data, non-comment field; skip.
			break
		}

		if data == nil {
			continue
		}

		switch eventType {
		case "message_start":
			var ms anthropicMessageStart
			if err := json.Unmarshal(data, &ms); err != nil {
				return nil, fmt.Errorf("aimodel: decode message_start: %w", err)
			}

			d.msgID = ms.Message.ID
			d.model = ms.Message.Model
			d.startUsage = ms.Message.Usage

			// Surface the execution container as soon as it is known.
			// Waiting for a text delta would lose it on streams that only
			// produce tool events or end immediately, and the caller needs
			// the ID to reuse the container on the next turn.
			if ms.Message.Container != nil {
				chunk := &core.StreamChunk{
					ID:    d.msgID,
					Model: d.model,
				}
				chunk.Extensions.Set(Name, &ResponseExtension{Container: ms.Message.Container})

				return chunk, nil
			}

			continue

		case "content_block_start":
			var cbs anthropicContentBlockStart
			if err := json.Unmarshal(data, &cbs); err != nil {
				return nil, fmt.Errorf("aimodel: decode content_block_start: %w", err)
			}

			switch cbs.ContentBlock.Type {
			case "tool_use":
				toolIdx := d.nextToolIdx
				d.blockToTool[cbs.Index] = toolIdx
				d.nextToolIdx++

				return &core.StreamChunk{
					ID:    d.msgID,
					Model: d.model,
					Choices: []core.StreamChunkChoice{
						{
							Index: 0,
							Delta: core.Message{
								Role: core.RoleAssistant,
								ToolCalls: []core.ToolCall{
									{
										Index: toolIdx,
										ID:    cbs.ContentBlock.ID,
										Type:  "function",
										Function: core.FunctionCall{
											Name: cbs.ContentBlock.Name,
										},
									},
								},
							},
						},
					},
				}, nil
			case "text", "thinking":
				continue
			default:
				// Unmodelled block (server_tool_use, a tool result, a
				// future type). Emit its original JSON and remember the
				// index so its deltas are preserved too.
				d.unknownBlocks[cbs.Index] = true

				return &core.StreamChunk{
					ID:    d.msgID,
					Model: d.model,
					Choices: []core.StreamChunkChoice{
						{
							Index: 0,
							Delta: extraBlockDelta(cbs.ContentBlock.raw),
						},
					},
				}, nil
			}

		case "content_block_delta":
			var cbd anthropicContentBlockDelta
			if err := json.Unmarshal(data, &cbd); err != nil {
				return nil, fmt.Errorf("aimodel: decode content_block_delta: %w", err)
			}

			chunk := &core.StreamChunk{
				ID:    d.msgID,
				Model: d.model,
			}

			// A delta belonging to an unrecognized block carries no
			// meaning this wrapper can assign, whatever its own type is —
			// keep it verbatim alongside the block start.
			if d.unknownBlocks[cbd.Index] {
				chunk.Choices = []core.StreamChunkChoice{
					{
						Index: 0,
						Delta: extraBlockDelta(cbd.Delta.raw),
					},
				}

				return chunk, nil
			}

			switch cbd.Delta.Type {
			case "text_delta":
				chunk.Choices = []core.StreamChunkChoice{
					{
						Index: 0,
						Delta: core.Message{
							Content: core.NewTextContent(cbd.Delta.Text),
						},
					},
				}
			case "thinking_delta":
				chunk.Choices = []core.StreamChunkChoice{
					{
						Index: 0,
						Delta: core.Message{
							Thinking: cbd.Delta.Thinking,
						},
					},
				}
			case "signature_delta":
				continue
			case "input_json_delta":
				toolIdx, ok := d.blockToTool[cbd.Index]
				if !ok {
					continue
				}

				chunk.Choices = []core.StreamChunkChoice{
					{
						Index: 0,
						Delta: core.Message{
							ToolCalls: []core.ToolCall{
								{
									Index: toolIdx,
									Function: core.FunctionCall{
										Arguments: cbd.Delta.PartialJSON,
									},
								},
							},
						},
					},
				}
			default:
				// A delta type added after this wrapper was written, on a
				// block it does know. Preserve it rather than drop it.
				chunk.Choices = []core.StreamChunkChoice{
					{
						Index: 0,
						Delta: extraBlockDelta(cbd.Delta.raw),
					},
				}
			}

			return chunk, nil

		case "message_delta":
			var md anthropicMessageDelta
			if err := json.Unmarshal(data, &md); err != nil {
				return nil, fmt.Errorf("aimodel: decode message_delta: %w", err)
			}

			reason := string(mapAnthropicStopReason(md.Delta.StopReason))

			terminal := core.StreamChunkChoice{
				Index:        0,
				FinishReason: &reason,
			}

			if md.Delta.StopDetails != nil {
				terminal.Extensions.Set(Name, &ChoiceExtension{StopDetails: md.Delta.StopDetails})
			}

			chunk := &core.StreamChunk{
				ID:      d.msgID,
				Model:   d.model,
				Choices: []core.StreamChunkChoice{terminal},
			}

			if md.Usage != nil {
				// message_start established the input/cache counts and the
				// geo / tier / server-tool information; the terminal event
				// typically carries only output_tokens. Merge instead of
				// replacing so the baseline survives.
				mergeAnthropicUsage(&d.startUsage, md.Usage)
				u := anthropicCanonicalUsage(&d.startUsage)
				chunk.Usage = &u
			}

			return chunk, nil

		case "message_stop":
			return nil, io.EOF

		case "error":
			var errResp anthropicErrorResponse
			if err := json.Unmarshal(data, &errResp); err != nil {
				return nil, fmt.Errorf("aimodel: decode stream error: %w", err)
			}

			return nil, &core.APIError{
				Type:    errResp.Error.Type,
				Message: errResp.Error.Message,
			}

		case "ping", "content_block_stop":
			continue
		}
	}

	if err := sc.Err(); err != nil {
		return nil, err
	}

	return nil, io.EOF
}
