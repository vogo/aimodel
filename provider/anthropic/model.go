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

package anthropic

// Model names are protocol facts, not a vendor-neutral contract, so the Claude
// names live with the protocol that serves them. MessagesRequest.Model stays a
// plain string: a model released after this list still works.

// Claude model names.
const (
	ModelClaudeFable5  = "claude-fable-5"
	ModelClaudeOpus48  = "claude-opus-4-8"
	ModelClaudeSonnet5 = "claude-sonnet-5"
	ModelClaudeHaiku45 = "claude-haiku-4-5"
)

// Effort values for OutputConfig.Effort, which sets how deeply the model
// reasons. The field is an open string; unlisted values pass through.
const (
	EffortLow    = "low"
	EffortMedium = "medium"
	EffortHigh   = "high"
	EffortXHigh  = "xhigh"
	EffortMax    = "max"
)

// Message roles are deliberately not named here. The Messages API accepts
// "user" and "assistant" only, and the migration-era translation tests still
// bind those identifiers to the canonical constants; naming them in this
// package would collide. Revisit once the canonical layer is gone.

// Thinking types for MessagesThinking.Type.
const (
	ThinkingTypeEnabled  = "enabled"
	ThinkingTypeDisabled = "disabled"
	// ThinkingTypeAdaptive lets the model size its own thinking budget.
	ThinkingTypeAdaptive = "adaptive"
)

// ThinkingDisplayOmitted suppresses thinking blocks in the response.
const ThinkingDisplayOmitted = "omitted"

// Content source types for an image or document block.
const (
	ContentSourceTypeBase64  = "base64"
	ContentSourceTypeURL     = "url"
	ContentSourceTypeText    = "text"
	ContentSourceTypeContent = "content"
)

// CacheControlTTL1h requests the 1-hour prompt cache; an empty TTL is the
// default 5-minute ephemeral cache.
const CacheControlTTL1h = "1h"
