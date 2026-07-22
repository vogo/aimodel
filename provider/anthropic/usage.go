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

import "github.com/vogo/aimodel/ais"

func anthropicCanonicalUsage(u *anthropicUsage) ais.Usage {
	cu := ais.Usage{
		PromptTokens:     u.totalInputTokens(),
		CompletionTokens: u.OutputTokens,
		TotalTokens:      u.totalInputTokens() + u.OutputTokens,
		CacheReadTokens:  u.CacheReadInputTokens,
		ServiceTier:      u.ServiceTier,
	}

	if u.OutputTokensDetails != nil {
		cu.ReasoningTokens = u.OutputTokensDetails.ThinkingTokens
	}

	ext := &UsageExtension{
		CacheWriteTokens: u.CacheCreationInputTokens,
		InferenceGeo:     u.InferenceGeo,
	}

	if u.CacheCreation != nil {
		ext.CacheWrite5mTokens = u.CacheCreation.Ephemeral5mInputTokens
		ext.CacheWrite1hTokens = u.CacheCreation.Ephemeral1hInputTokens
	}

	if u.ServerToolUse != nil {
		ext.ServerToolUse = &ServerToolUse{
			WebSearchRequests: u.ServerToolUse.WebSearchRequests,
			WebFetchRequests:  u.ServerToolUse.WebFetchRequests,
		}
	}

	if ext.CacheWriteTokens != 0 || ext.InferenceGeo != "" || ext.ServerToolUse != nil ||
		ext.CacheWrite5mTokens != 0 || ext.CacheWrite1hTokens != 0 {
		cu.Extensions.Set(Name, ext)
	}

	return cu
}

// mergeAnthropicUsage folds a later usage object (the terminal message_delta)
// into the baseline captured at message_start. Only fields the later object
// actually carries are applied — a terminal event that reports just
// output_tokens must not blank out the input, cache, geo, tier or server-tool
// information already established.
func mergeAnthropicUsage(base, next *anthropicUsage) {
	if next.InputTokens != 0 {
		base.InputTokens = next.InputTokens
	}

	if next.OutputTokens != 0 {
		base.OutputTokens = next.OutputTokens
	}

	if next.CacheCreationInputTokens != 0 {
		base.CacheCreationInputTokens = next.CacheCreationInputTokens
	}

	if next.CacheReadInputTokens != 0 {
		base.CacheReadInputTokens = next.CacheReadInputTokens
	}

	if next.CacheCreation != nil {
		base.CacheCreation = next.CacheCreation
	}

	if next.OutputTokensDetails != nil {
		base.OutputTokensDetails = next.OutputTokensDetails
	}

	if next.ServerToolUse != nil {
		base.ServerToolUse = next.ServerToolUse
	}

	if next.InferenceGeo != "" {
		base.InferenceGeo = next.InferenceGeo
	}

	if next.ServiceTier != "" {
		base.ServiceTier = next.ServiceTier
	}
}

// parseDataURI parses a data URI (e.g. "data:image/jpeg;base64,/9j...")
// and returns the media type and base64-encoded data.
