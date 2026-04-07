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

import "sync"

// InterceptStream installs onChunk and onDone callbacks on s without changing
// the consumer-visible Stream API. onChunk fires for every non-nil chunk
// returned by Recv. onDone fires exactly once: on the first non-nil error
// (including io.EOF) returned by Recv, or on Close, whichever comes first.
//
// InterceptStream is additive: callers using Recv/Close see no behavior
// change. The callbacks must be cheap and must not call Recv/Close on s.
func InterceptStream(s *Stream, onChunk func(*StreamChunk), onDone func(err error)) *Stream {
	if s == nil {
		if onDone != nil {
			onDone(nil)
		}

		return nil
	}

	var once sync.Once

	fire := func(err error) {
		once.Do(func() {
			if onDone != nil {
				onDone(err)
			}
		})
	}

	inner := s.recv
	s.recv = func() (*StreamChunk, error) {
		chunk, err := inner()
		if chunk != nil && onChunk != nil {
			onChunk(chunk)
		}

		if err != nil {
			fire(err)
		}

		return chunk, err
	}

	prevOnClose := s.onClose
	s.onClose = func(u *Usage) {
		if prevOnClose != nil {
			prevOnClose(u)
		}

		fire(nil)
	}

	return s
}
