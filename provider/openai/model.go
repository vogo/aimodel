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

package openai

// Model names are protocol facts, not a vendor-neutral contract: these are the
// models reachable over the OpenAI-compatible Chat Completions protocol, which
// is what this package speaks. Backends other than OpenAI are listed here for
// the same reason — they are addressed through this protocol. Model stays a
// plain string, so any name a backend serves works without a constant.

// OpenAI model names.
const (
	ModelGPT55      = "gpt-5.5"
	ModelGPT55Pro   = "gpt-5.5-pro"
	ModelGPT56      = "gpt-5.6"
	ModelGPT56Sol   = "gpt-5.6-sol"
	ModelGPT56Terra = "gpt-5.6-terra"
	ModelGPT56Luna  = "gpt-5.6-luna"
	ModelGPT4o      = "gpt-4o"
	ModelGPT4oMini  = "gpt-4o-mini"
	ModelGPT41      = "gpt-4.1"
	ModelGPT41Mini  = "gpt-4.1-mini"
	ModelGPT41Nano  = "gpt-4.1-nano"
	ModelO1         = "o1"
	ModelO3         = "o3"
	ModelO3Mini     = "o3-mini"
	ModelO4Mini     = "o4-mini"
)

// DeepSeek model names.
const (
	ModelDeepseekV4Pro   = "deepseek-v4-pro"
	ModelDeepseekV4Flash = "deepseek-v4-flash"
)

// Google Gemini model names, as served over an OpenAI-compatible endpoint.
const (
	ModelGemini36Flash     = "gemini-3.6-flash"
	ModelGemini35FlashLite = "gemini-3.5-flash-lite"
	ModelGemini31Pro       = "gemini-3.1-pro"
	ModelGemini31FlashLite = "gemini-3.1-flash-lite"
	ModelGemini25Pro       = "gemini-2.5-pro"
	ModelGemini25Flash     = "gemini-2.5-flash"
	ModelGemini25FlashLite = "gemini-2.5-flash-lite"
)

// MiniMax model names.
const (
	ModelMinimaxM27          = "MiniMax-M2.7"
	ModelMinimaxM27Highspeed = "MiniMax-M2.7-highspeed"
	ModelMinimaxM25          = "MiniMax-M2.5"
	ModelMinimaxM25Highspeed = "MiniMax-M2.5-highspeed"
	ModelMinimaxM21          = "MiniMax-M2.1"
	ModelMinimaxM21Highspeed = "MiniMax-M2.1-highspeed"
	ModelMinimaxM2           = "MiniMax-M2"
)

// Moonshot Kimi model names.
const (
	ModelKimiK26 = "kimi-k2.6"
	ModelKimiK25 = "kimi-k2.5"
)

// ModelGLM52 is Zhipu's GLM model name.
const ModelGLM52 = "glm-5.2"

// ModelDoubaoSeed20Lite is ByteDance's Doubao model name.
const ModelDoubaoSeed20Lite = "doubao-seed-2-0-lite-260215"

// Alibaba Qwen model names.
const (
	ModelQwen37Max   = "qwen3.7-max"
	ModelQwen37Plus  = "qwen3.7-plus"
	ModelQwen36Flash = "qwen3.6-flash"
)

// ReasoningEffort values constrain how many reasoning tokens a model spends.
// GPT-5.1 and later default to ReasoningEffortNone.
// On GPT-5.4 and later, ReasoningEffortNone also disables tool calling — pick a
// higher effort when tools are active, or use the Responses API.
// ChatCompletionRequest.ReasoningEffort stays a plain string, so a value an
// OpenAI-compatible backend defines on its own passes through unchanged.
const (
	ReasoningEffortNone    = "none"
	ReasoningEffortMinimal = "minimal"
	ReasoningEffortLow     = "low"
	ReasoningEffortMedium  = "medium"
	ReasoningEffortHigh    = "high"
	ReasoningEffortXHigh   = "xhigh"
)

// finish_reason values on a choice. The field is an open string; a value this
// list does not name still decodes and reaches the caller verbatim.
const (
	FinishReasonStop          = "stop"
	FinishReasonLength        = "length"
	FinishReasonToolCalls     = "tool_calls"
	FinishReasonContentFilter = "content_filter"
	// FinishReasonFunctionCall is emitted by the deprecated functions API.
	FinishReasonFunctionCall = "function_call"
)

// Message roles.
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
	// RoleDeveloper replaces the system role on the o-series and later models.
	RoleDeveloper = "developer"
)

// Content part discriminators (`content[].type`).
const (
	ContentPartTypeText       = "text"
	ContentPartTypeImageURL   = "image_url"
	ContentPartTypeInputAudio = "input_audio"
	ContentPartTypeFile       = "file"
)

// ToolTypeFunction is the only tool kind OpenAI defines today.
const ToolTypeFunction = "function"
