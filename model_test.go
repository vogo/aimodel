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

import "testing"

func TestModelConstantsNonEmpty(t *testing.T) {
	models := map[string]string{
		"OpenaiGPT4o":             ModelOpenaiGPT4o,
		"OpenaiGPT4oMini":         ModelOpenaiGPT4oMini,
		"OpenaiGPT41":             ModelOpenaiGPT41,
		"OpenaiGPT41Mini":         ModelOpenaiGPT41Mini,
		"OpenaiGPT41Nano":         ModelOpenaiGPT41Nano,
		"OpenaiO1":                ModelOpenaiO1,
		"OpenaiO1Mini":            ModelOpenaiO1Mini,
		"OpenaiO3":                ModelOpenaiO3,
		"OpenaiO3Mini":            ModelOpenaiO3Mini,
		"OpenaiO4Mini":            ModelOpenaiO4Mini,
		"DeepseekChat":            ModelDeepseekChat,
		"DeepseekReasoner":        ModelDeepseekReasoner,
		"Gemini25Pro":             ModelGemini25Pro,
		"Gemini25Flash":           ModelGemini25Flash,
		"Gemini20Flash":           ModelGemini20Flash,
		"Gemini20FlashExp":        ModelGemini20FlashExp,
		"AnthropicClaude4Opus":    ModelAnthropicClaude4Opus,
		"AnthropicClaude4Sonnet":  ModelAnthropicClaude4Sonnet,
		"AnthropicClaude37Sonnet": ModelAnthropicClaude37Sonnet,
		"AnthropicClaude35Haiku":  ModelAnthropicClaude35Haiku,
		"MinimaxM25":              ModelMinimaxM25,
		"MinimaxM25Highspeed":     ModelMinimaxM25Highspeed,
		"MinimaxM21":              ModelMinimaxM21,
		"MinimaxM21Highspeed":     ModelMinimaxM21Highspeed,
		"MinimaxM2":               ModelMinimaxM2,
		"KimiK2":                  ModelKimiK2,
		"KimiK25":                 ModelKimiK25,
		"Moonshot8k":              ModelMoonshot8k,
		"Moonshot32k":             ModelMoonshot32k,
		"Moonshot128k":            ModelMoonshot128k,
		"GLM4Plus":                ModelGLM4Plus,
		"GLM4Air":                 ModelGLM4Air,
		"GLM4AirX":                ModelGLM4AirX,
		"GLM4Long":                ModelGLM4Long,
		"GLM4Flash":               ModelGLM4Flash,
		"DoubaoPro32k":            ModelDoubaoPro32k,
		"DoubaoPro256k":           ModelDoubaoPro256k,
		"DoubaoLite32k":           ModelDoubaoLite32k,
		"DoubaoLite128k":          ModelDoubaoLite128k,
		"QwenMax":                 ModelQwenMax,
		"QwenPlus":                ModelQwenPlus,
		"QwenTurbo":               ModelQwenTurbo,
		"Qwen3Max":                ModelQwen3Max,
		"Qwen35Plus":              ModelQwen35Plus,
		"Qwen35Flash":             ModelQwen35Flash,
	}

	for name, val := range models {
		if val == "" {
			t.Errorf("model constant %s must not be empty", name)
		}
	}
}

func TestModelConstantsUnique(t *testing.T) {
	models := []string{
		ModelOpenaiGPT4o, ModelOpenaiGPT4oMini,
		ModelOpenaiGPT41, ModelOpenaiGPT41Mini, ModelOpenaiGPT41Nano,
		ModelOpenaiO1, ModelOpenaiO1Mini,
		ModelOpenaiO3, ModelOpenaiO3Mini, ModelOpenaiO4Mini,
		ModelDeepseekChat, ModelDeepseekReasoner,
		ModelGemini25Pro, ModelGemini25Flash, ModelGemini20Flash, ModelGemini20FlashExp,
		ModelAnthropicClaude4Opus, ModelAnthropicClaude4Sonnet,
		ModelAnthropicClaude37Sonnet, ModelAnthropicClaude35Haiku,
		ModelMinimaxM25, ModelMinimaxM25Highspeed, ModelMinimaxM21, ModelMinimaxM21Highspeed, ModelMinimaxM2,
		ModelKimiK2, ModelKimiK25, ModelMoonshot8k, ModelMoonshot32k, ModelMoonshot128k,
		ModelGLM4Plus, ModelGLM4Air, ModelGLM4AirX, ModelGLM4Long, ModelGLM4Flash,
		ModelDoubaoPro32k, ModelDoubaoPro256k, ModelDoubaoLite32k, ModelDoubaoLite128k,
		ModelQwenMax, ModelQwenPlus, ModelQwenTurbo, ModelQwen3Max, ModelQwen35Plus, ModelQwen35Flash,
	}

	seen := make(map[string]bool, len(models))
	for _, m := range models {
		if seen[m] {
			t.Errorf("duplicate model constant value: %q", m)
		}
		seen[m] = true
	}
}
