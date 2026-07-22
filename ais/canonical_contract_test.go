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
	"go/ast"
	"go/parser"
	"go/token"
	"reflect"
	"testing"
)

func TestCanonicalRequestFieldsHaveAnthropicConsumption(t *testing.T) {
	file, err := parser.ParseFile(token.NewFileSet(), "../provider/anthropic/request.go", nil, 0)
	if err != nil {
		t.Fatalf("parse Anthropic request translator: %v", err)
	}

	consumed := map[string]bool{}
	ast.Inspect(file, func(node ast.Node) bool {
		sel, ok := node.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		if id, ok := sel.X.(*ast.Ident); ok && id.Name == "req" {
			consumed[sel.Sel.Name] = true
		}
		return true
	})

	typ := reflect.TypeFor[ChatRequest]()
	for field := range typ.Fields() {
		name := field.Name
		if !consumed[name] {
			t.Errorf("canonical ChatRequest.%s has no actual read in the Anthropic request translation path", name)
		}
	}
}

func TestCanonicalSchemaExcludesSingleProtocolSurface(t *testing.T) {
	forbiddenFields := []string{
		"N", "FrequencyPenalty", "PresencePenalty", "Seed", "User",
		"StreamOptions", "Verbosity", "Logprobs", "TopLogprobs", "LogitBias",
		"ServiceTier", "Store", "Metadata", "PromptCacheKey", "Modalities", "Audio",
	}
	request := reflect.TypeFor[ChatRequest]()
	for _, name := range forbiddenFields {
		if _, ok := request.FieldByName(name); ok {
			t.Errorf("single-protocol field ChatRequest.%s must not be canonical", name)
		}
	}

	checks := []struct {
		typ   reflect.Type
		field string
	}{
		{reflect.TypeFor[Choice](), "LogProbs"},
		{reflect.TypeFor[Message](), "Audio"},
		{reflect.TypeFor[ContentPart](), "InputAudio"},
		{reflect.TypeFor[ContentPart](), "File"},
	}
	for _, check := range checks {
		if _, ok := check.typ.FieldByName(check.field); ok {
			t.Errorf("single-protocol field %s.%s must not be canonical", check.typ.Name(), check.field)
		}
	}

	file, err := parser.ParseFile(token.NewFileSet(), "schema.go", nil, 0)
	if err != nil {
		t.Fatalf("parse canonical schema: %v", err)
	}
	forbiddenNames := map[string]bool{
		"StreamOptions": true, "AudioConfig": true, "LogProbs": true,
		"TokenLogprob": true, "TopLogprob": true, "InputAudio": true,
		"FilePart": true, "MessageAudio": true, "VerbosityLow": true,
		"VerbosityMedium": true, "VerbosityHigh": true,
	}
	for _, decl := range file.Decls {
		gen, ok := decl.(*ast.GenDecl)
		if !ok {
			continue
		}
		for _, spec := range gen.Specs {
			switch value := spec.(type) {
			case *ast.TypeSpec:
				if forbiddenNames[value.Name.Name] {
					t.Errorf("single-protocol type %s must not be exported by ais", value.Name.Name)
				}
			case *ast.ValueSpec:
				for _, name := range value.Names {
					if forbiddenNames[name.Name] {
						t.Errorf("single-protocol constant %s must not be exported by ais", name.Name)
					}
				}
			}
		}
	}
}
