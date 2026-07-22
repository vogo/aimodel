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

package anthropropic_tests

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"testing"

	"github.com/vogo/aimodel/provider/anthropic"
)

func TestNativeMessagesStream(t *testing.T) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	model := os.Getenv("ANTHROPIC_MODEL")
	if apiKey == "" || model == "" {
		t.Skip("ANTHROPIC_API_KEY and ANTHROPIC_MODEL are required")
	}
	options := []anthropic.ClientOption{}
	if baseURL := os.Getenv("ANTHROPIC_BASE_URL"); baseURL != "" {
		options = append(options, anthropic.WithBaseURL(baseURL))
	}
	stream, err := anthropic.NewClient(apiKey, options...).MessagesStream(context.Background(), &anthropic.MessagesRequest{
		Model:     model,
		MaxTokens: 64,
		Messages: []anthropic.MessagesMessage{{
			Role:    "user",
			Content: json.RawMessage(`"Count from one to three."`),
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = stream.Close() }()
	for {
		event, recvErr := stream.Recv()
		if errors.Is(recvErr, io.EOF) {
			return
		}
		if recvErr != nil {
			t.Fatal(recvErr)
		}
		t.Logf("%s: %s", event.Type, event.Raw)
	}
}
