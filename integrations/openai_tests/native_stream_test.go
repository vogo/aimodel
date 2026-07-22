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

package openai_tests

import (
	"context"
	"errors"
	"io"
	"os"
	"testing"

	"github.com/vogo/aimodel/provider/openai"
)

func TestNativeChatCompletionsStream(t *testing.T) {
	apiKey, model := os.Getenv("OPENAI_API_KEY"), os.Getenv("OPENAI_MODEL")
	if apiKey == "" || model == "" {
		t.Skip("OPENAI_API_KEY and OPENAI_MODEL are required")
	}
	options := []openai.ClientOption{}
	if baseURL := os.Getenv("OPENAI_BASE_URL"); baseURL != "" {
		options = append(options, openai.WithBaseURL(baseURL))
	}
	stream, err := openai.NewClient(apiKey, options...).ChatCompletionsStream(context.Background(), &openai.ChatCompletionRequest{
		Model: model, Messages: []openai.ChatCompletionMessage{{Role: "user", Content: openai.NewTextContent("Count from one to three.")}},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = stream.Close() }()
	for {
		chunk, recvErr := stream.Recv()
		if errors.Is(recvErr, io.EOF) {
			return
		}
		if recvErr != nil {
			t.Fatal(recvErr)
		}
		t.Logf("native chunk: %+v", chunk)
	}
}
