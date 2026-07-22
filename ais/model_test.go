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

package ais

import "testing"

func TestModelConstants(t *testing.T) {
	models := map[string]string{
		"OpenaiGPT56": ModelOpenaiGPT56, "OpenaiGPT56Sol": ModelOpenaiGPT56Sol,
		"OpenaiGPT56Terra": ModelOpenaiGPT56Terra, "OpenaiGPT56Luna": ModelOpenaiGPT56Luna,
		"OpenaiGPT4o": ModelOpenaiGPT4o, "OpenaiGPT4oMini": ModelOpenaiGPT4oMini,
		"OpenaiGPT41": ModelOpenaiGPT41, "OpenaiGPT41Mini": ModelOpenaiGPT41Mini,
		"OpenaiGPT41Nano": ModelOpenaiGPT41Nano, "OpenaiO1": ModelOpenaiO1,
		"OpenaiO3": ModelOpenaiO3, "OpenaiO3Mini": ModelOpenaiO3Mini, "OpenaiO4Mini": ModelOpenaiO4Mini,
		"DeepseekV4Pro": ModelDeepseekV4Pro, "DeepseekV4Flash": ModelDeepseekV4Flash,
		"Gemini36Flash": ModelGemini36Flash, "Gemini35FlashLite": ModelGemini35FlashLite,
		"Gemini31Pro": ModelGemini31Pro, "Gemini31FlashLite": ModelGemini31FlashLite,
		"Gemini25Pro": ModelGemini25Pro, "Gemini25Flash": ModelGemini25Flash,
		"Gemini25FlashLite":     ModelGemini25FlashLite,
		"AnthropicClaudeFable5": ModelAnthropicClaudeFable5, "AnthropicClaudeOpus48": ModelAnthropicClaudeOpus48,
		"AnthropicClaudeSonnet5": ModelAnthropicClaudeSonnet5, "AnthropicClaudeHaiku45": ModelAnthropicClaudeHaiku45,
		"MinimaxM27": ModelMinimaxM27, "MinimaxM27Highspeed": ModelMinimaxM27Highspeed,
		"MinimaxM25": ModelMinimaxM25, "MinimaxM25Highspeed": ModelMinimaxM25Highspeed,
		"MinimaxM21": ModelMinimaxM21, "MinimaxM21Highspeed": ModelMinimaxM21Highspeed, "MinimaxM2": ModelMinimaxM2,
		"KimiK26": ModelKimiK26, "KimiK25": ModelKimiK25, "GLM52": ModelGLM52,
		"DoubaoSeed20Lite": ModelDoubaoSeed20Lite,
		"Qwen37Max":        ModelQwen37Max, "Qwen37Plus": ModelQwen37Plus, "Qwen36Flash": ModelQwen36Flash,
	}

	seen := make(map[string]string, len(models))
	for name, value := range models {
		if value == "" {
			t.Errorf("model constant %s must not be empty", name)
		}
		if previous, ok := seen[value]; ok {
			t.Errorf("model constants %s and %s both use %q", previous, name, value)
		}
		seen[value] = name
	}
}
