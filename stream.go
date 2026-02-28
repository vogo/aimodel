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
	mu     sync.Mutex
	reader io.ReadCloser
	recv   func() (*StreamChunk, error)
	closed atomic.Bool
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

	return s.recv()
}

// Close closes the stream and releases resources.
func (s *Stream) Close() error {
	s.closed.Store(true)

	s.mu.Lock()
	defer s.mu.Unlock()

	return s.reader.Close()
}
