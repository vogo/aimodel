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

func anthropicCanonicalUsage(u *MessagesUsage) ais.Usage {
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

// The stream usage merge that used to live here moved to accumulate.go: it is
// native wire semantics (message_start baseline + message_delta terminal
// counts), not canonical translation, and it is now observable through
// MessageStream.Usage.
