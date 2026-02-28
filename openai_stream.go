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

// streamChunkOrError combines StreamChunk and Error for single-pass unmarshal.
type streamChunkOrError struct {
	StreamChunk
	Error *Error `json:"error,omitempty"`
}

func newStream(body io.ReadCloser) *Stream {
	sc := bufio.NewScanner(body)
	sc.Buffer(make([]byte, 0, 64*1024), maxStreamLineSize)

	return &Stream{
		reader: body,
		recv:   openaiRecvFunc(sc),
	}
}

func openaiRecvFunc(sc *bufio.Scanner) func() (*StreamChunk, error) {
	return func() (*StreamChunk, error) {
		for sc.Scan() {
			line := sc.Text()

			// Skip empty lines, SSE comments, and non-data lines.
			if line == "" || strings.HasPrefix(line, ":") {
				continue
			}

			if !strings.HasPrefix(line, "data: ") {
				continue
			}

			data := strings.TrimPrefix(line, "data: ")

			if data == "[DONE]" {
				return nil, io.EOF
			}

			var parsed streamChunkOrError
			if err := json.Unmarshal([]byte(data), &parsed); err != nil {
				return nil, fmt.Errorf("aimodel: decode stream chunk: %w", err)
			}

			if parsed.Error != nil {
				return nil, &APIError{
					Code:    parsed.Error.Code,
					Message: parsed.Error.Message,
					Type:    parsed.Error.Type,
				}
			}

			chunk := parsed.StreamChunk
			return &chunk, nil
		}

		if err := sc.Err(); err != nil {
			return nil, err
		}

		return nil, io.EOF
	}
}
