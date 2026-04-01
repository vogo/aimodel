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
	"io"
	"sync"
	"sync/atomic"
)

// maxStreamLineSize limits the maximum SSE line size to 1 MB.
const maxStreamLineSize = 1 << 20

// Stream reads streaming chat completion responses using SSE.
// Stream is safe for concurrent use between a single Recv caller and Close.
type Stream struct {
	mu      sync.Mutex
	reader  io.ReadCloser
	recv    func() (*StreamChunk, error)
	closed  atomic.Bool
	usage   *Usage // captured from the final chunk that includes usage data
	onClose func(*Usage)
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

	chunk, err := s.recv()
	if chunk != nil && chunk.Usage != nil {
		s.usage = chunk.Usage
	}

	return chunk, err
}

// Usage returns the accumulated usage from the stream, if available.
// This is typically populated from the final chunk when stream_options.include_usage is set,
// or from Anthropic's message_delta event.
func (s *Stream) Usage() *Usage {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.usage
}

// Close closes the stream and releases resources.
// Close is safe to call concurrently with Recv and is idempotent.
func (s *Stream) Close() error {
	if !s.closed.CompareAndSwap(false, true) {
		return nil
	}

	if s.onClose != nil {
		s.onClose(s.usage)
	}

	// Close the reader directly to unblock any in-progress Recv.
	// http.Response.Body.Close is safe to call concurrently.
	return s.reader.Close()
}

// WrapStream wraps an existing stream with a callback that fires on close with usage data.
// If s is nil, onClose is called immediately with nil usage and nil is returned.
func WrapStream(s *Stream, onClose func(*Usage)) *Stream {
	if s == nil {
		if onClose != nil {
			onClose(nil)
		}

		return nil
	}

	s.onClose = onClose

	return s
}
