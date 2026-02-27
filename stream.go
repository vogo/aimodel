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
	"sync"
	"sync/atomic"
)

// maxStreamLineSize limits the maximum SSE line size to 1 MB.
const maxStreamLineSize = 1 << 20

// Stream reads streaming chat completion responses using SSE.
// Stream is safe for concurrent use between a single Recv caller and Close.
type Stream struct {
	mu     sync.Mutex
	reader io.ReadCloser
	scan   *bufio.Scanner
	closed atomic.Bool
}

func newStream(body io.ReadCloser) *Stream {
	sc := bufio.NewScanner(body)
	sc.Buffer(make([]byte, 0, 64*1024), maxStreamLineSize)

	return &Stream{
		reader: body,
		scan:   sc,
	}
}

// streamChunkOrError combines StreamChunk and Error for single-pass unmarshal.
type streamChunkOrError struct {
	StreamChunk
	Error *Error `json:"error,omitempty"`
}

// Recv reads the next chunk from the stream.
// Returns io.EOF when the stream is done.
func (s *Stream) Recv() (*StreamChunk, error) {
	if s.closed.Load() {
		return nil, ErrStreamClosed
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed.Load() {
		return nil, ErrStreamClosed
	}

	for s.scan.Scan() {
		line := s.scan.Text()

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

	if err := s.scan.Err(); err != nil {
		return nil, err
	}

	return nil, io.EOF
}

// Close closes the stream and releases resources.
func (s *Stream) Close() error {
	s.closed.Store(true)

	s.mu.Lock()
	defer s.mu.Unlock()

	return s.reader.Close()
}
