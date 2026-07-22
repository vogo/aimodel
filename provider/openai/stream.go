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

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/vogo/aimodel/ais"
)

// streamChunkOrError combines StreamChunk and Error for single-pass unmarshal.
type streamChunkOrError struct {
	ais.StreamChunk
	Error *ais.Error `json:"error,omitempty"`
}

// NewStreamDecoder returns a decoder for the OpenAI SSE event stream.
func (p *provider) NewStreamDecoder(body io.Reader) ais.StreamDecoder {
	sc := bufio.NewScanner(body)
	sc.Buffer(make([]byte, 0, 64*1024), ais.MaxStreamLineSize)

	return &streamDecoder{sc: sc}
}

type streamDecoder struct {
	sc *bufio.Scanner
}

func (d *streamDecoder) Next() (*ais.StreamChunk, error) {
	sc := d.sc

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
			return nil, &ais.APIError{
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
