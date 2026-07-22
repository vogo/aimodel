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

import (
	"io"

	"github.com/vogo/aimodel/core"
	"github.com/vogo/aimodel/provider/openai"
)

// newOpenAIStream builds a root Stream backed by the OpenAI provider's SSE
// decoder, exercising the real decoder together with the Stream lifecycle.
func newOpenAIStream(body io.ReadCloser) *Stream {
	p, err := openai.New(core.Config{APIKey: "sk-test", BaseURL: "https://example.com"})
	if err != nil {
		panic(err)
	}

	return newStream(body, p.NewStreamDecoder(body))
}
