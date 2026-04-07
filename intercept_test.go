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
	"errors"
	"io"
	"strings"
	"sync/atomic"
	"testing"
)

func newFakeStream(chunks []*StreamChunk, finalErr error) *Stream {
	idx := 0
	s := &Stream{
		reader: io.NopCloser(strings.NewReader("")),
	}
	s.recv = func() (*StreamChunk, error) {
		if idx >= len(chunks) {
			return nil, finalErr
		}
		c := chunks[idx]
		idx++
		return c, nil
	}
	return s
}

func TestInterceptStream_NormalEOF(t *testing.T) {
	chunks := []*StreamChunk{{ID: "1"}, {ID: "2"}, {ID: "3"}}
	s := newFakeStream(chunks, io.EOF)

	var got []string
	var doneCount int32
	var doneErr error

	InterceptStream(s, func(c *StreamChunk) {
		got = append(got, c.ID)
	}, func(err error) {
		atomic.AddInt32(&doneCount, 1)
		doneErr = err
	})

	for {
		_, err := s.Recv()
		if err != nil {
			break
		}
	}

	if len(got) != 3 {
		t.Errorf("expected 3 chunks, got %d", len(got))
	}
	if atomic.LoadInt32(&doneCount) != 1 {
		t.Errorf("expected onDone fired once, got %d", doneCount)
	}
	if !errors.Is(doneErr, io.EOF) {
		t.Errorf("expected EOF, got %v", doneErr)
	}
}

func TestInterceptStream_EarlyClose(t *testing.T) {
	chunks := []*StreamChunk{{ID: "1"}, {ID: "2"}, {ID: "3"}}
	s := newFakeStream(chunks, io.EOF)

	var doneCount int32
	InterceptStream(s, nil, func(err error) {
		atomic.AddInt32(&doneCount, 1)
	})

	_, _ = s.Recv()
	_ = s.Close()
	_ = s.Close() // double close idempotent

	if atomic.LoadInt32(&doneCount) != 1 {
		t.Errorf("expected onDone exactly once, got %d", doneCount)
	}
}

func TestInterceptStream_MidStreamError(t *testing.T) {
	myErr := errors.New("boom")
	chunks := []*StreamChunk{{ID: "1"}, {ID: "2"}}
	s := newFakeStream(chunks, myErr)

	var got int
	var doneErr error
	InterceptStream(s, func(c *StreamChunk) { got++ }, func(err error) { doneErr = err })

	for {
		_, err := s.Recv()
		if err != nil {
			break
		}
	}

	if got != 2 {
		t.Errorf("expected 2 chunks, got %d", got)
	}
	if doneErr == nil || doneErr.Error() != "boom" {
		t.Errorf("expected boom, got %v", doneErr)
	}
}

func TestInterceptStream_NilStream(t *testing.T) {
	var called bool
	s := InterceptStream(nil, nil, func(err error) { called = true })
	if s != nil {
		t.Errorf("expected nil")
	}
	if !called {
		t.Errorf("expected onDone called")
	}
}
