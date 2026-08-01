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
	"context"
	"net/http"

	"github.com/vogo/aimodel/ais"
	"github.com/vogo/aimodel/provider/openai"
)

// CapabilityResponses names the Responses capability in a *ais.CapabilityError.
const CapabilityResponses = "responses"

// Responder is the OpenAI Responses capability contract — a second capability
// interface alongside ChatCompleter, per ADR 0004: a new interaction form gets
// its own interface instead of widening an existing one.
//
// Unlike ChatCompleter, Responder speaks provider-native types. Responses is
// currently a single-vendor interaction form, so under the "≥ 2 providers" rule
// its request, response, item and event shapes stay in provider/openai and are
// not translated through ais. This is the documented exception to
// "the unified client is canonical in, canonical out" — see
// doc/adr/0006-responses-capability-on-provider-native-types.md.
type Responder interface {
	Responses(ctx context.Context, req *openai.ResponsesRequest) (*openai.Response, error)
	ResponsesStream(ctx context.Context, req *openai.ResponsesRequest) (*openai.ResponseStream, error)
}

// Compile-time check: *Client implements Responder.
var _ Responder = (*Client)(nil)

// responsesProvider is the internal method set a provider implements to support
// the Responses capability. Only the OpenAI provider does today.
type responsesProvider interface {
	NativeResponsesClient(httpClient *http.Client) *openai.Client
}

// responsesClient resolves the provider's native Responses client, preserving
// the client's API key, base URL, HTTP client and timeout. A provider without
// the capability fails here, before any network I/O.
func (c *Client) responsesClient() (*openai.Client, error) {
	prov, ok := c.provider.(responsesProvider)
	if !ok {
		return nil, &ais.CapabilityError{Provider: c.providerName, Capability: CapabilityResponses}
	}

	return prov.NativeResponsesClient(c.httpClient), nil
}

// Responses sends a non-streaming Responses request through the resolved
// provider. It uses native wire values end to end: the client's default model
// is not applied (the request's own Model is authoritative, matching the native
// client), no canonical translation runs, and chat interception and compose
// failover do not apply.
func (c *Client) Responses(ctx context.Context, req *openai.ResponsesRequest) (*openai.Response, error) {
	client, err := c.responsesClient()
	if err != nil {
		return nil, err
	}

	return client.Responses(ctx, req)
}

// ResponsesStream sends a streaming Responses request through the resolved
// provider and returns the native event stream. The same boundaries as
// Responses apply.
func (c *Client) ResponsesStream(ctx context.Context, req *openai.ResponsesRequest) (*openai.ResponseStream, error) {
	client, err := c.responsesClient()
	if err != nil {
		return nil, err
	}

	return client.ResponsesStream(ctx, req)
}
