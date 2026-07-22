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
	"io/fs"
	"strings"
	"testing"
)

// parseAisPackage parses the non-test files of this package.
func parseAisPackage(t *testing.T) map[string]*ast.File {
	t.Helper()

	fset := token.NewFileSet()

	//nolint:staticcheck // ParseDir is sufficient for a single-directory scan
	// and keeps this zero-dependency.
	pkgs, err := parser.ParseDir(fset, ".", func(fi fs.FileInfo) bool {
		// model.go is the explicit multi-vendor model-name catalog (plain
		// string constants, a writing convenience) — naming vendors there is
		// its whole purpose, so it is exempt from the vendor-neutrality scan.
		if fi.Name() == "model.go" {
			return false
		}

		return strings.HasSuffix(fi.Name(), ".go") && !strings.HasSuffix(fi.Name(), "_test.go")
	}, 0)
	if err != nil {
		t.Fatalf("parse ais package: %v", err)
	}

	pkg, ok := pkgs["ais"]
	if !ok {
		t.Fatal("ais package not found")
	}

	return pkg.Files
}

// TestCanonicalSchemaHasNoVendorIdentifiers enforces the vendor-neutrality
// invariant of the canonical layer: no declared identifier (type, field,
// constant, function, method) in the ais package names a vendor. Vendor
// semantics live in provider subpackages and reach the canonical types only
// through the unified Extensions channel.
func TestCanonicalSchemaHasNoVendorIdentifiers(t *testing.T) {
	vendors := []string{"anthropic", "openai", "claude", "gpt"}

	check := func(file string, ident *ast.Ident) {
		if ident == nil {
			return
		}

		lower := strings.ToLower(ident.Name)
		for _, v := range vendors {
			if strings.Contains(lower, v) {
				t.Errorf("%s: identifier %q names vendor %q — vendor semantics belong in the provider subpackage", file, ident.Name, v)
			}
		}
	}

	for name, file := range parseAisPackage(t) {
		ast.Inspect(file, func(n ast.Node) bool {
			switch d := n.(type) {
			case *ast.TypeSpec:
				check(name, d.Name)
			case *ast.ValueSpec:
				for _, id := range d.Names {
					check(name, id)
				}
			case *ast.FuncDecl:
				check(name, d.Name)
			case *ast.Field:
				for _, id := range d.Names {
					check(name, id)
				}
			}

			return true
		})
	}
}

// TestCanonicalSchemaJSONDashFieldsAreExtensionsOnly enforces that the only
// struct fields excluded from canonical JSON are the unified Extensions
// channel. A vendor-specific json:"-" switch smuggled into a canonical type
// fails this test.
func TestCanonicalSchemaJSONDashFieldsAreExtensionsOnly(t *testing.T) {
	for name, file := range parseAisPackage(t) {
		ast.Inspect(file, func(n ast.Node) bool {
			st, ok := n.(*ast.StructType)
			if !ok {
				return true
			}

			for _, f := range st.Fields.List {
				if f.Tag == nil || !strings.Contains(f.Tag.Value, `json:"-"`) {
					continue
				}

				ident, ok := f.Type.(*ast.Ident)
				if !ok || ident.Name != "Extensions" {
					t.Errorf("%s: field %v is tagged json:\"-\" but is not the unified Extensions channel", name, f.Names)
				}
			}

			return true
		})
	}
}
