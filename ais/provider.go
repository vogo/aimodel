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

// Package api holds the vendor-neutral foundation of aimodel: the canonical
// chat types, the error model, the provider contract, and the provider
// registry. It has no vendor dependencies; provider subpackages implement
// against it, and callers use its canonical types directly alongside the
// root aimodel client facade.
package ais

import (
	"context"
	"io"
	"net/http"
)

// MaxStreamLineSize limits the maximum SSE line size read by provider stream
// decoders to 1 MB.
const MaxStreamLineSize = 1 << 20

// ChatProvider is the vendor boundary for one chat capability call. It covers
// exactly the protocol-specific work: building the outgoing HTTP request
// (URL, serialized body, headers), normalizing a successful response,
// converting a non-2xx response body into an error, and decoding SSE events.
//
// The root pipeline owns everything else: request cloning, default model,
// sending the single HTTP request, closing response bodies on failure paths,
// and the Stream lifecycle. A provider must not retain or mutate caller
// state; the *ChatRequest it receives is a per-call working copy that it may
// rewrite (e.g. to default protocol-specific fields).
type ChatProvider interface {
	// NewChatRequest builds the complete HTTP request for the given canonical
	// request. req is the pipeline's working copy: already cloned, with the
	// stream flag and default model applied.
	NewChatRequest(ctx context.Context, req *ChatRequest) (*http.Request, error)

	// ParseChatResponse normalizes the body of a successful non-streaming
	// response into a canonical ChatResponse. The caller closes body.
	ParseChatResponse(body io.Reader) (*ChatResponse, error)

	// ParseErrorResponse converts a non-2xx response into an error (typically
	// *APIError). body is the response body already read by the caller,
	// subject to the shared read limit.
	ParseErrorResponse(statusCode int, body []byte) error

	// NewStreamDecoder returns a fresh decoder reading SSE events from body.
	// Decoders are single-use: streaming state is never shared across calls.
	NewStreamDecoder(body io.Reader) StreamDecoder
}

// StreamDecoder decodes one canonical chunk per call from a streaming
// response body. It returns io.EOF when the stream is complete. The root
// Stream owns the close state and the underlying reader; a decoder only
// translates events.
type StreamDecoder interface {
	Next() (*StreamChunk, error)
}

// Config carries the common construction-time configuration handed to a
// provider factory. Vendor-specific configuration travels in Options as a
// value defined by the provider's own package.
type Config struct {
	// APIKey is the credential for the vendor API. Never empty: the root
	// pipeline rejects key-less clients before resolving a provider.
	APIKey string

	// BaseURL is the API base URL with any trailing "/" trimmed. Empty means
	// "use the provider default"; providers without a default must fail
	// construction.
	BaseURL string

	// Options is the provider-specific configuration value (defined by the
	// provider package), or nil. A factory must reject values of a type it
	// does not recognize.
	Options any
}

// Factory constructs a ready-to-use provider instance from its
// configuration. It validates required fields and vendor options up front so
// every failure surfaces at client construction, not at call time.
type Factory func(cfg Config) (ChatProvider, error)
