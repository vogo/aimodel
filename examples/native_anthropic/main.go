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
	"encoding/json"
	"fmt"
	"os"

	"github.com/vogo/aimodel/provider/anthropic"
)

func main() {
	c := anthropic.NewClient(os.Getenv("ANTHROPIC_API_KEY"))
	resp, err := c.Messages(context.Background(), &anthropic.MessageRequest{Model: "claude-sonnet-4-20250514", MaxTokens: 256, Messages: []anthropic.MessageParam{{Role: "user", Content: json.RawMessage(`"Hello"`)}}})
	if err != nil {
		panic(err)
	}
	fmt.Println(resp.ID)
}
