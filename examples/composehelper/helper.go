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

package composehelper

import (
	"github.com/vogo/aimodel"
)

func BuildComposeClient() ([]*aimodel.Client, error) {
	openaiClient, err := aimodel.NewClient(
		aimodel.WithAPIKey(aimodel.GetEnv("OPENAI_API_KEY")),
		aimodel.WithBaseURL(aimodel.GetEnv("OPENAI_BASE_URL")),
		aimodel.WithDefaultModel(aimodel.GetEnv("OPENAI_MODEL")),
	)
	if err != nil {
		return nil, err
	}

	anthropicClient, err := aimodel.NewClient(
		aimodel.WithAPIKey(aimodel.GetEnv("ANTHROPIC_API_KEY")),
		aimodel.WithBaseURL(aimodel.GetEnv("ANTHROPIC_BASE_URL")),
		aimodel.WithDefaultModel(aimodel.GetEnv("ANTHROPIC_MODEL")),
		aimodel.WithProtocol(aimodel.ProtocolAnthropic),
	)
	if err != nil {
		return nil, err
	}

	deepseekClient, err := aimodel.NewClient(
		aimodel.WithAPIKey(aimodel.GetEnv("DEEPSEEK_API_KEY")),
		aimodel.WithBaseURL(aimodel.GetEnv("DEEPSEEK_BASE_URL")),
		aimodel.WithDefaultModel(aimodel.GetEnv("DEEPSEEK_MODEL")),
	)
	if err != nil {
		return nil, err
	}

	return []*aimodel.Client{openaiClient, anthropicClient, deepseekClient}, nil
}
