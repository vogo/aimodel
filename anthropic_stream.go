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

package aimodel

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// newAnthropicStream creates a Stream that parses Anthropic SSE events.
// Streaming reference: https://platform.claude.com/docs/en/api/messages
func newAnthropicStream(body io.ReadCloser) *Stream {
	sc := bufio.NewScanner(body)
	sc.Buffer(make([]byte, 0, 64*1024), maxStreamLineSize)

	return &Stream{
		reader: body,
		recv:   anthropicRecvFunc(sc),
	}
}

func anthropicRecvFunc(sc *bufio.Scanner) func() (*StreamChunk, error) {
	var (
		msgID           string
		model           string
		inputTokens     int
		cacheReadTokens int

		// blockToTool maps Anthropic content block index to tool call index.
		// Anthropic uses sequential indices for all content blocks (text, thinking, tool_use),
		// but AppendDelta expects indices scoped to tool calls only.
		blockToTool = make(map[int]int)
		nextToolIdx int
	)

	return func() (*StreamChunk, error) {
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

				msgID = ms.Message.ID
				model = ms.Message.Model
				inputTokens = ms.Message.Usage.totalInputTokens()
				cacheReadTokens = ms.Message.Usage.CacheReadInputTokens

				continue

			case "content_block_start":
				var cbs anthropicContentBlockStart
				if err := json.Unmarshal(data, &cbs); err != nil {
					return nil, fmt.Errorf("aimodel: decode content_block_start: %w", err)
				}

				switch cbs.ContentBlock.Type {
				case "tool_use":
					toolIdx := nextToolIdx
					blockToTool[cbs.Index] = toolIdx
					nextToolIdx++

					return &StreamChunk{
						ID:    msgID,
						Model: model,
						Choices: []StreamChunkChoice{
							{
								Index: 0,
								Delta: Message{
									Role: RoleAssistant,
									ToolCalls: []ToolCall{
										{
											Index: toolIdx,
											ID:    cbs.ContentBlock.ID,
											Type:  "function",
											Function: FunctionCall{
												Name: cbs.ContentBlock.Name,
											},
										},
									},
								},
							},
						},
					}, nil
				case "thinking":
					continue
				}

				continue

			case "content_block_delta":
				var cbd anthropicContentBlockDelta
				if err := json.Unmarshal(data, &cbd); err != nil {
					return nil, fmt.Errorf("aimodel: decode content_block_delta: %w", err)
				}

				chunk := &StreamChunk{
					ID:    msgID,
					Model: model,
				}

				switch cbd.Delta.Type {
				case "text_delta":
					chunk.Choices = []StreamChunkChoice{
						{
							Index: 0,
							Delta: Message{
								Content: NewTextContent(cbd.Delta.Text),
							},
						},
					}
				case "thinking_delta":
					chunk.Choices = []StreamChunkChoice{
						{
							Index: 0,
							Delta: Message{
								Thinking: cbd.Delta.Thinking,
							},
						},
					}
				case "signature_delta":
					continue
				case "input_json_delta":
					toolIdx, ok := blockToTool[cbd.Index]
					if !ok {
						continue
					}

					chunk.Choices = []StreamChunkChoice{
						{
							Index: 0,
							Delta: Message{
								ToolCalls: []ToolCall{
									{
										Index: toolIdx,
										Function: FunctionCall{
											Arguments: cbd.Delta.PartialJSON,
										},
									},
								},
							},
						},
					}
				default:
					continue
				}

				return chunk, nil

			case "message_delta":
				var md anthropicMessageDelta
				if err := json.Unmarshal(data, &md); err != nil {
					return nil, fmt.Errorf("aimodel: decode message_delta: %w", err)
				}

				reason := string(mapAnthropicStopReason(md.Delta.StopReason))

				chunk := &StreamChunk{
					ID:    msgID,
					Model: model,
					Choices: []StreamChunkChoice{
						{
							Index:        0,
							FinishReason: &reason,
						},
					},
				}

				if md.Usage != nil {
					outputTokens := md.Usage.OutputTokens
					chunk.Usage = &Usage{
						PromptTokens:     inputTokens,
						CompletionTokens: outputTokens,
						TotalTokens:      inputTokens + outputTokens,
						CacheReadTokens:  cacheReadTokens,
					}
				}

				return chunk, nil

			case "message_stop":
				return nil, io.EOF

			case "error":
				var errResp anthropicErrorResponse
				if err := json.Unmarshal(data, &errResp); err != nil {
					return nil, fmt.Errorf("aimodel: decode stream error: %w", err)
				}

				return nil, &APIError{
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
}
