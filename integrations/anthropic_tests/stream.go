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
	"errors"
	"fmt"
	"io"
	"log"

	"github.com/vogo/aimodel"
)

func testStream(client *aimodel.Client) {
	fmt.Println("=== Anthropic Stream ===")

	stream, err := client.ChatCompletionStream(context.Background(), &aimodel.ChatRequest{
		Messages: []aimodel.Message{
			{Role: aimodel.RoleUser, Content: aimodel.NewTextContent("What is AGI! Answer in 200 words.")},
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = stream.Close() }()

	for {
		chunk, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			log.Fatal(err)
		}

		if len(chunk.Choices) > 0 {
			fmt.Print(chunk.Choices[0].Delta.Content.Text())
		}
	}

	fmt.Println()
}
