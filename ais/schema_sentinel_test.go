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

import (
	"reflect"
	"testing"
)

// translationLayers lists the hand-written canonical↔native seams that a
// canonical field change has to be walked against.
var translationLayers = []string{
	"provider/openai/translate.go",
	"provider/anthropic/request.go",
	"provider/anthropic/response.go",
	"provider/anthropic/usage.go",
	"provider/anthropic/stream.go",
}

// canonicalNodes pins the field count of every canonical node that a provider
// translation walks field by field. Update a count here only after visiting
// the translation layers — that visit is the entire point of this test.
var canonicalNodes = []struct {
	name   string
	typ    reflect.Type
	fields int
}{
	{"ChatRequest", reflect.TypeFor[ChatRequest](), 16},
	{"Message", reflect.TypeFor[Message](), 6},
	{"ContentPart", reflect.TypeFor[ContentPart](), 3},
	{"ImageURL", reflect.TypeFor[ImageURL](), 2},
	{"Thinking", reflect.TypeFor[Thinking](), 3},
	{"Tool", reflect.TypeFor[Tool](), 4},
	{"FunctionDefinition", reflect.TypeFor[FunctionDefinition](), 3},
	{"ToolCall", reflect.TypeFor[ToolCall](), 4},
	{"FunctionCall", reflect.TypeFor[FunctionCall](), 2},
	{"ChatResponse", reflect.TypeFor[ChatResponse](), 8},
	{"Choice", reflect.TypeFor[Choice](), 4},
	{"Usage", reflect.TypeFor[Usage](), 7},
	{"StreamChunk", reflect.TypeFor[StreamChunk](), 7},
	{"StreamChunkChoice", reflect.TypeFor[StreamChunkChoice](), 4},
}

// TestCanonicalNodeFieldCountsAreStable is a sentinel, not a coverage check.
//
// Every provider maps canonical types to its own native wire model by hand
// (ADR 0005). Go gives that seam no protection: a composite literal need not
// list every field, so a canonical field with no line in a translation layer
// compiles, marshals to a valid body with the field simply absent, and gets a
// 200 back. The value is dropped in silence.
//
// This test does not assert that every field is mapped — canonical and native
// are deliberately not isomorphic, and leaving a field unmapped is often the
// right call (see doc/architecture.md §2 and each protocol document's mapping
// boundary section). It only refuses to let the shape of a canonical node
// change without someone looking at the seams. Deciding a field belongs
// nowhere is fine; never being asked is not.
func TestCanonicalNodeFieldCountsAreStable(t *testing.T) {
	for _, node := range canonicalNodes {
		if got := node.typ.NumField(); got != node.fields {
			t.Errorf("ais.%s has %d fields, the sentinel expects %d.\n"+
				"If you changed this node, walk the translation layers before updating the count:\n"+
				"  %v\n"+
				"An unmapped canonical field is dropped silently — no compile error, no runtime error, "+
				"the call still succeeds. Mapping it or recording it as an intentional boundary are both "+
				"valid outcomes; skipping the question is not.",
				node.name, got, node.fields, translationLayers)
		}
	}
}
