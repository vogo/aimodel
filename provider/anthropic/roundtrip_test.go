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

package anthropic

import (
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"reflect"
	"strings"
	"testing"
)

// Fidelity, as an executable check: every public wire type survives
// marshal -> unmarshal -> marshal unchanged. A field that encodes but does not
// decode — or decodes into something that re-encodes differently — is a
// silently lossy protocol, which is exactly what this package exists to avoid.
//
// Values are filled by reflection so a newly added field is covered without
// anyone remembering to extend a literal, and wireTypes is checked against the
// package's own AST so a newly added type cannot be left out.

// wireTypes is the set of exported wire types, each as a zero value that the
// filler populates.
var wireTypes = map[string]any{
	"CacheControl":           CacheControl{},
	"CacheCreation":          CacheCreation{},
	"ContentBlock":           ContentBlock{},
	"ContentBlockDelta":      ContentBlockDelta{},
	"ContentBlockDeltaEvent": ContentBlockDeltaEvent{},
	"ContentBlockStartEvent": ContentBlockStartEvent{},
	"ContentSource":          ContentSource{},
	"MessageDelta":           MessageDelta{},
	"MessageDeltaEvent":      MessageDeltaEvent{},
	"MessagesError":          MessagesError{},
	"MessagesErrorResponse":  MessagesErrorResponse{},
	"MessagesMessage":        MessagesMessage{},
	"MessagesRequest":        MessagesRequest{},
	"MessagesResponse":       MessagesResponse{},
	"MessageStartEvent":      MessageStartEvent{},
	"MessagesThinking":       MessagesThinking{},
	"MessagesTool":           MessagesTool{},
	"MessagesUsage":          MessagesUsage{},
	"OutputConfig":           OutputConfig{},
	"OutputFormat":           OutputFormat{},
	"OutputTokensDetails":    OutputTokensDetails{},
	"ResponseContainer":      ResponseContainer{},
	"ResponseContentBlock":   ResponseContentBlock{},
	"ServerToolUse":          ServerToolUse{},
	"StopDetails":            StopDetails{},
	"StreamEvent":            StreamEvent{},
	"ToolChoice":             ToolChoice{},
}

// nonWireTypes are exported types that are not part of the wire protocol:
// clients, streams and the transport error, which holds a live error value.
var nonWireTypes = map[string]bool{
	"Client":        true,
	"HTTPError":     true,
	"MessageStream": true,
}

func TestWireTypesRoundTripLosslessly(t *testing.T) {
	for name, zero := range wireTypes {
		t.Run(name, func(t *testing.T) {
			value := reflect.New(reflect.TypeOf(zero))
			fillWireValue(value.Elem(), 0)

			first, err := json.Marshal(value.Interface())
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}

			decoded := reflect.New(reflect.TypeOf(zero))
			if err = json.Unmarshal(first, decoded.Interface()); err != nil {
				t.Fatalf("unmarshal %s: %v", first, err)
			}

			second, err := json.Marshal(decoded.Interface())
			if err != nil {
				t.Fatalf("re-marshal: %v", err)
			}

			if string(first) != string(second) {
				t.Errorf("round trip is lossy\n first: %s\nsecond: %s", first, second)
			}
		})
	}
}

// TestEveryExportedTypeIsClassified fails when a new exported type is neither
// covered by the round trip nor explicitly declared not to be wire.
func TestEveryExportedTypeIsClassified(t *testing.T) {
	fset := token.NewFileSet()

	//nolint:staticcheck // ParseDir is sufficient here; this SDK stays zero-dependency.
	pkgs, err := parser.ParseDir(fset, ".", func(fi fs.FileInfo) bool {
		name := fi.Name()

		return strings.HasSuffix(name, ".go") && !strings.HasSuffix(name, "_test.go")
	}, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("parse package: %v", err)
	}

	declared := map[string]bool{}

	for _, pkg := range pkgs {
		for _, file := range pkg.Files {
			ast.Inspect(file, func(node ast.Node) bool {
				spec, ok := node.(*ast.TypeSpec)
				if !ok || !spec.Name.IsExported() {
					return true
				}

				if _, isStruct := spec.Type.(*ast.StructType); isStruct {
					declared[spec.Name.Name] = true
				}

				return true
			})
		}
	}

	for name := range declared {
		if _, covered := wireTypes[name]; !covered && !nonWireTypes[name] {
			t.Errorf("%s is exported but neither round-tripped nor listed as a non-wire type", name)
		}
	}

	for name := range wireTypes {
		if !declared[name] {
			t.Errorf("wireTypes lists %s, which no longer exists", name)
		}
	}
}

// maxFillDepth bounds recursion: some wire types nest into themselves (a
// content source holding content blocks, for example).
const maxFillDepth = 3

var rawMessageType = reflect.TypeFor[json.RawMessage]()

// fillWireValue populates a value with representative non-zero data. Types with
// custom JSON handling and unexported state are built through their
// constructors instead, since reflection cannot reach their fields.
func fillWireValue(v reflect.Value, depth int) {
	if !v.CanSet() {
		return
	}

	if sample, ok := wireSample(v.Type()); ok {
		v.Set(sample)

		return
	}

	switch v.Kind() {
	case reflect.String:
		v.SetString("x")
	case reflect.Bool:
		v.SetBool(true)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		v.SetInt(1)
	case reflect.Float32, reflect.Float64:
		v.SetFloat(1)
	case reflect.Interface:
		// An `any` field decodes back as a string, so a string round-trips
		// where a typed value would not.
		v.Set(reflect.ValueOf("x"))
	case reflect.Slice:
		if v.Type() == rawMessageType {
			v.SetBytes([]byte(`{"raw":1}`))

			return
		}

		if depth >= maxFillDepth {
			return
		}

		element := reflect.New(v.Type().Elem()).Elem()
		fillWireValue(element, depth+1)
		v.Set(reflect.Append(v, element))
	case reflect.Pointer:
		// Stop before allocating rather than after: a non-nil pointer to an
		// unfilled value is not a shape the protocol can express, so it would
		// fail the round trip for a reason the wire types are not responsible
		// for.
		if depth >= maxFillDepth {
			return
		}

		v.Set(reflect.New(v.Type().Elem()))
		fillWireValue(v.Elem(), depth+1)
	case reflect.Map:
		if depth >= maxFillDepth {
			return
		}

		key := reflect.New(v.Type().Key()).Elem()
		fillWireValue(key, depth+1)

		element := reflect.New(v.Type().Elem()).Elem()
		fillWireValue(element, depth+1)

		v.Set(reflect.MakeMap(v.Type()))
		v.SetMapIndex(key, element)
	case reflect.Struct:
		if depth >= maxFillDepth {
			return
		}

		for _, field := range v.Fields() {
			fillWireValue(field, depth+1)
		}
	}
}

// wireSample supplies values for types whose JSON form is backed by unexported
// state. This package has none today; the hook exists so adding one does not
// mean rewriting the filler.
func wireSample(_ reflect.Type) (reflect.Value, bool) {
	return reflect.Value{}, false
}
