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
	"testing"

	"github.com/vogo/aimodel/ais"
)

type sequenceDecoder struct {
	chunks []*ais.StreamChunk
	index  int
	err    error
}

func (d *sequenceDecoder) Next() (*ais.StreamChunk, error) {
	if d.index >= len(d.chunks) {
		return nil, d.err
	}

	chunk := d.chunks[d.index]
	d.index++

	return chunk, nil
}

func newTestStream(chunks ...*ais.StreamChunk) *Stream {
	body := io.NopCloser(strings.NewReader(""))
	return newStream(body, &sequenceDecoder{chunks: chunks, err: io.EOF})
}

func TestStreamRecvAndUsage(t *testing.T) {
	wantUsage := &ais.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}
	stream := newTestStream(
		&ais.StreamChunk{ID: "first"},
		&ais.StreamChunk{ID: "final", Usage: wantUsage},
	)

	for _, wantID := range []string{"first", "final"} {
		chunk, err := stream.Recv()
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
		if chunk.ID != wantID {
			t.Errorf("chunk ID = %q, want %q", chunk.ID, wantID)
		}
	}

	if _, err := stream.Recv(); !errors.Is(err, io.EOF) {
		t.Errorf("Recv error = %v, want io.EOF", err)
	}
	if got := stream.Usage(); got != wantUsage {
		t.Errorf("Usage() = %+v, want %+v", got, wantUsage)
	}
}

func TestStreamUsageNilWithoutUsageChunk(t *testing.T) {
	stream := newTestStream(&ais.StreamChunk{ID: "only"})

	if _, err := stream.Recv(); err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if got := stream.Usage(); got != nil {
		t.Errorf("Usage() = %+v, want nil", got)
	}
}

func TestStreamCloseIsIdempotent(t *testing.T) {
	stream := newTestStream(&ais.StreamChunk{ID: "unread"})

	if err := stream.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if _, err := stream.Recv(); !errors.Is(err, ais.ErrStreamClosed) {
		t.Errorf("Recv error = %v, want ais.ErrStreamClosed", err)
	}
}

func TestWrapStreamCallbackReceivesUsage(t *testing.T) {
	wantUsage := &ais.Usage{TotalTokens: 9}
	stream := newTestStream(&ais.StreamChunk{Usage: wantUsage})

	var gotUsage *ais.Usage
	WrapStream(stream, func(usage *ais.Usage) { gotUsage = usage })

	if _, err := stream.Recv(); err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if err := stream.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if gotUsage != wantUsage {
		t.Errorf("callback usage = %+v, want %+v", gotUsage, wantUsage)
	}
}

func TestWrapStreamNil(t *testing.T) {
	called := false
	result := WrapStream(nil, func(usage *ais.Usage) {
		called = true
		if usage != nil {
			t.Errorf("callback usage = %+v, want nil", usage)
		}
	})

	if result != nil {
		t.Errorf("WrapStream(nil, callback) = %v, want nil", result)
	}
	if !called {
		t.Fatal("callback was not called")
	}
	if result := WrapStream(nil, nil); result != nil {
		t.Errorf("WrapStream(nil, nil) = %v, want nil", result)
	}
}
