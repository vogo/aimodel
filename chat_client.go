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

import "context"

// ChatCompleter defines the minimal contract for AI chat completion.
// Consumers that only need basic completion should depend on this interface.
type ChatCompleter interface {
	ChatCompletion(ctx context.Context, req *ChatRequest) (*ChatResponse, error)
	ChatCompletionStream(ctx context.Context, req *ChatRequest) (*Stream, error)
}

// ChatClient extends ChatCompleter with Anthropic-native protocol support.
// Use this when protocol-specific routing is needed (e.g., compose dispatch).
type ChatClient interface {
	ChatCompleter
	AnthropicChatCompletion(ctx context.Context, req *ChatRequest) (*ChatResponse, error)
	AnthropicChatCompletionStream(ctx context.Context, req *ChatRequest) (*Stream, error)
}

// Protocol determines which API protocol to use for a model.
type Protocol string

const (
	// ProtocolOpenAI uses the OpenAI-compatible chat completions API (default).
	ProtocolOpenAI Protocol = "openai"
	// ProtocolAnthropic uses the Anthropic Messages API.
	ProtocolAnthropic Protocol = "anthropic"
)
