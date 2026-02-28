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

package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/vogo/aimodel"
)

func main() {
	client, err := aimodel.NewClient(
		aimodel.WithAPIKey(aimodel.GetEnv("OPENAI_API_KEY")),
		aimodel.WithBaseURL(aimodel.GetEnv("OPENAI_BASE_URL")),
	)
	if err != nil {
		log.Fatal(err)
	}

	model := os.Getenv("OPENAI_MODEL")
	if model == "" {
		model = aimodel.ModelOpenaiGPT4o
	}

	req := &aimodel.ChatRequest{
		Model: model,
		Messages: []aimodel.Message{
			{Role: aimodel.RoleUser, Content: aimodel.NewTextContent("Say hello!")},
		},
	}

	resp, err := client.ChatCompletion(context.Background(), req)
	if err != nil {
		log.Fatal(err)
	}

	if len(resp.Choices) == 0 {
		log.Fatal("no choices in response")
	}

	fmt.Println(resp.Choices[0].Message.Content.Text())
}
