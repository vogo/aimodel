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

import "github.com/vogo/aimodel/core"

// The canonical chat types live in the core package so that provider
// subpackages and composes can share one definition without importing the
// root package. The root package re-exports them as type aliases — not
// copies — so callers, providers, and composes exchange identical types with
// a single source of truth for method sets and JSON behavior.

// Request/response types.
type (
	// ChatRequest represents a request to the chat completions API.
	ChatRequest = core.ChatRequest
	// ChatResponse represents a response from the chat completions API.
	ChatResponse = core.ChatResponse
	// StreamChunk represents a single chunk in a streaming response.
	StreamChunk = core.StreamChunk
	// StreamChunkChoice represents a choice within a stream chunk.
	StreamChunkChoice = core.StreamChunkChoice
	// Choice represents a single completion choice.
	Choice = core.Choice
	// Usage tracks token usage for a request.
	Usage = core.Usage
	// ServerToolUse counts server-side tool invocations reported on Usage.
	ServerToolUse = core.ServerToolUse
	// Error represents an error in the API response body.
	Error = core.Error
	// ResponseContainer is the server-side execution container returned
	// alongside a response.
	ResponseContainer = core.ResponseContainer
	// StopDetails carries a structured stop classification.
	StopDetails = core.StopDetails
	// LogProbs holds the token log-probability payload for a Choice.
	LogProbs = core.LogProbs
	// TokenLogprob is the log probability of a single output token.
	TokenLogprob = core.TokenLogprob
	// TopLogprob is one alternative token and its log probability.
	TopLogprob = core.TopLogprob
)

// Message and content types.
type (
	// Message represents a chat message.
	Message = core.Message
	// Content represents chat message content (plain text or parts).
	Content = core.Content
	// ContentPart represents a single part in a multimodal content array.
	ContentPart = core.ContentPart
	// ImageURL represents an image URL in a content part.
	ImageURL = core.ImageURL
	// InputAudio represents an audio input in a content part.
	InputAudio = core.InputAudio
	// FilePart represents a file input in a content part.
	FilePart = core.FilePart
	// MessageAudio is the generated audio attached to an assistant message.
	MessageAudio = core.MessageAudio
	// Role represents the role of a chat message participant.
	Role = core.Role
	// FinishReason represents the reason a model stopped generating.
	FinishReason = core.FinishReason
)

// Tool types.
type (
	// Tool represents a tool definition for the API.
	Tool = core.Tool
	// FunctionDefinition describes a function available to the model.
	FunctionDefinition = core.FunctionDefinition
	// ToolCall represents a function call requested by the model.
	ToolCall = core.ToolCall
	// FunctionCall contains the function name and arguments.
	FunctionCall = core.FunctionCall
)

// Configuration types.
type (
	// Thinking configures extended thinking / reasoning behavior.
	Thinking = core.Thinking
	// StreamOptions configures streaming behavior.
	StreamOptions = core.StreamOptions
	// AudioConfig configures the audio output of a request.
	AudioConfig = core.AudioConfig
)

// Role constants for chat messages.
const (
	RoleSystem    = core.RoleSystem
	RoleUser      = core.RoleUser
	RoleAssistant = core.RoleAssistant
	RoleTool      = core.RoleTool
)

// FinishReason constants. See their documentation in the core package.
const (
	FinishReasonStop                       = core.FinishReasonStop
	FinishReasonLength                     = core.FinishReasonLength
	FinishReasonToolCalls                  = core.FinishReasonToolCalls
	FinishReasonContentFilter              = core.FinishReasonContentFilter
	FinishReasonFunctionCall               = core.FinishReasonFunctionCall
	FinishReasonModelContextWindowExceeded = core.FinishReasonModelContextWindowExceeded
	FinishReasonRefusal                    = core.FinishReasonRefusal
	FinishReasonPauseTurn                  = core.FinishReasonPauseTurn
)

// ReasoningEffort constants. See their documentation in the core package.
const (
	ReasoningEffortNone    = core.ReasoningEffortNone
	ReasoningEffortMinimal = core.ReasoningEffortMinimal
	ReasoningEffortLow     = core.ReasoningEffortLow
	ReasoningEffortMedium  = core.ReasoningEffortMedium
	ReasoningEffortHigh    = core.ReasoningEffortHigh
	ReasoningEffortXHigh   = core.ReasoningEffortXHigh
)

// Verbosity constants. See their documentation in the core package.
const (
	VerbosityLow    = core.VerbosityLow
	VerbosityMedium = core.VerbosityMedium
	VerbosityHigh   = core.VerbosityHigh
)

// NewTextContent creates a Content from a plain string.
func NewTextContent(text string) Content {
	return core.NewTextContent(text)
}

// NewPartsContent creates a Content from multiple content parts.
func NewPartsContent(parts ...ContentPart) Content {
	return core.NewPartsContent(parts...)
}
