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

package aimodel_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"strconv"
	"strings"
	"testing"

	"github.com/vogo/aimodel/provider/anthropic"
	"github.com/vogo/aimodel/provider/openai"
)

// These tests enforce ADR 0007 in CI rather than by convention, because the
// failure mode it guards against — a shared semantic layer growing back one
// helper at a time — is gradual and reads as reasonable at every step.

// statusCoder is the interface a consumer declares locally to read a status
// code off any provider's transport error. Declaring it here, rather than
// importing an error type, is the pattern itself.
type statusCoder interface{ StatusCode() int }

// Guard: both providers' HTTP errors satisfy the structural error contract.
// A compile-time assertion is the whole test — if either stops implementing
// it, this file no longer builds.
var (
	_ statusCoder = (*openai.HTTPError)(nil)
	_ statusCoder = (*anthropic.HTTPError)(nil)
)

// packageImports parses the non-test .go files in a package directory (relative
// to this module root) and returns the set of imported package paths.
func packageImports(t *testing.T, dir string) map[string]bool {
	t.Helper()

	fset := token.NewFileSet()

	//nolint:staticcheck // ParseDir with ImportsOnly is sufficient for scanning
	// import paths; build-tag precision is unnecessary here and avoids an
	// external dependency (this SDK stays zero-dependency).
	pkgs, err := parser.ParseDir(fset, dir, func(fi fs.FileInfo) bool {
		name := fi.Name()

		return strings.HasSuffix(name, ".go") && !strings.HasSuffix(name, "_test.go")
	}, parser.ImportsOnly)
	if err != nil {
		t.Fatalf("parse %s: %v", dir, err)
	}

	imports := map[string]bool{}

	for _, pkg := range pkgs {
		for _, file := range pkg.Files {
			for _, imp := range file.Imports {
				path, err := strconv.Unquote(imp.Path.Value)
				if err != nil {
					t.Fatalf("unquote import %s: %v", imp.Path.Value, err)
				}

				imports[path] = true
			}
		}
	}

	return imports
}

func hasProviderImport(imports map[string]bool, want string) bool {
	for path := range imports {
		if path == "github.com/vogo/aimodel/provider/"+want {
			return true
		}
	}

	return false
}

// TestProvidersAreIndependent verifies the two built-in provider subpackages do
// not depend on each other — a vendor API change touches only its own package.
func TestProvidersAreIndependent(t *testing.T) {
	openaiImports := packageImports(t, "provider/openai")
	if hasProviderImport(openaiImports, "anthropic") {
		t.Error("provider/openai must not import provider/anthropic")
	}

	anthropicImports := packageImports(t, "provider/anthropic")
	if hasProviderImport(anthropicImports, "openai") {
		t.Error("provider/anthropic must not import provider/openai")
	}
}

// TestProvidersDoNotDependOnRoot verifies providers depend only on the shared
// api foundation, never on the root package (which would create a cycle) or on
// composes.
func TestProvidersDoNotDependOnRoot(t *testing.T) {
	for _, dir := range []string{"provider/openai", "provider/anthropic"} {
		imports := packageImports(t, dir)

		if imports["github.com/vogo/aimodel"] {
			t.Errorf("%s must not import the root package", dir)
		}

		if imports["github.com/vogo/aimodel/composes"] {
			t.Errorf("%s must not import composes", dir)
		}
	}
}

// TestComposesDependsOnOpenAIOnly verifies composes is what ADR 0007 declares
// it to be: a tool for the OpenAI-compatible wire format, not a vendor-neutral
// package.
//
// The earlier rule — composes may import no provider at all — existed to keep
// canonical dispatch vendor-neutral. ADR 0007 gives that constraint up
// deliberately, and replaces it with this narrower one: exactly one provider,
// and no canonical layer. Dispatching Anthropic backends means an isomorphic
// loop in that package, never a shared request model here.
func TestComposesDependsOnOpenAIOnly(t *testing.T) {
	imports := packageImports(t, "composes")

	for path := range imports {
		if strings.Contains(path, "/provider/") && path != "github.com/vogo/aimodel/provider/openai" {
			t.Errorf("composes must import no provider other than openai, found %q", path)
		}
	}

	if !hasProviderImport(imports, "openai") {
		t.Error("composes dispatches over the OpenAI wire format and should import provider/openai")
	}

	if imports["github.com/vogo/aimodel"] {
		t.Error("composes must not depend on the root package")
	}

	if imports["github.com/vogo/aimodel/ais"] {
		t.Error("composes must not depend on the canonical package")
	}
}

// TestRootPackageExportsNothing verifies the root package stays empty. It has
// no unified client, no shared schema and no provider imports: a caller reaches
// a protocol by importing its own package, which is what makes the two
// protocols independent (ADR 0007).
func TestRootPackageExportsNothing(t *testing.T) {
	imports := packageImports(t, ".")

	for path := range imports {
		if strings.HasPrefix(path, "github.com/vogo/aimodel") {
			t.Errorf("root package must import nothing from this module, found %q", path)
		}
	}

	fset := token.NewFileSet()

	//nolint:staticcheck // ParseDir with ImportsOnly is sufficient here.
	pkgs, err := parser.ParseDir(fset, ".", func(fi fs.FileInfo) bool {
		name := fi.Name()

		return strings.HasSuffix(name, ".go") && !strings.HasSuffix(name, "_test.go")
	}, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("parse root package: %v", err)
	}

	for _, pkg := range pkgs {
		for path, file := range pkg.Files {
			for _, decl := range file.Decls {
				if name, ok := exportedDeclName(decl); ok {
					t.Errorf("%s declares exported %s; the root package exports nothing", path, name)
				}
			}
		}
	}
}

// exportedDeclName reports the name of an exported top-level declaration.
func exportedDeclName(decl ast.Decl) (string, bool) {
	switch d := decl.(type) {
	case *ast.FuncDecl:
		if d.Name.IsExported() {
			return d.Name.Name, true
		}
	case *ast.GenDecl:
		for _, spec := range d.Specs {
			switch s := spec.(type) {
			case *ast.TypeSpec:
				if s.Name.IsExported() {
					return s.Name.Name, true
				}
			case *ast.ValueSpec:
				for _, name := range s.Names {
					if name.IsExported() {
						return name.Name, true
					}
				}
			}
		}
	}

	return "", false
}

// TestNoSharedSemanticPackage verifies the two providers share no package from
// this module. A type both of them reach for is a canonical layer by another
// name, whatever it is called.
func TestNoSharedSemanticPackage(t *testing.T) {
	openaiImports := packageImports(t, "provider/openai")
	anthropicImports := packageImports(t, "provider/anthropic")

	for path := range openaiImports {
		if anthropicImports[path] && strings.HasPrefix(path, "github.com/vogo/aimodel") {
			t.Errorf("both providers import %q; a package they share is a shared semantic layer", path)
		}
	}
}

// protocolSemanticWords name concepts that belong to a protocol, not to a
// neutral utility. A package declared vendor-neutral that starts speaking them
// has stopped being neutral.
var protocolSemanticWords = []string{
	"message", "content", "tool", "usage", "completion", "chat", "prompt", "token", "choice",
}

// neutralPackages are the packages this module declares vendor-neutral.
// composes is deliberately absent: ADR 0007 states it is an OpenAI-wire tool,
// and its dependency guard says so explicitly.
var neutralPackages = []string{"."}

// TestNeutralPackagesDeclareNoProtocolSemantics checks declared identifiers
// over the AST, so a word inside a comment or a string literal cannot fail the
// build and a real declaration cannot hide in one.
func TestNeutralPackagesDeclareNoProtocolSemantics(t *testing.T) {
	for _, dir := range neutralPackages {
		for name, pos := range declaredIdentifiers(t, dir) {
			lower := strings.ToLower(name)

			for _, word := range protocolSemanticWords {
				if strings.Contains(lower, word) {
					t.Errorf("%s: %s declares %q, a protocol concept; it belongs in a provider package",
						dir, pos, name)
				}
			}
		}
	}
}

// declaredIdentifiers returns the names a package declares — top-level
// declarations, struct fields and interface methods — mapped to their position.
func declaredIdentifiers(t *testing.T, dir string) map[string]string {
	t.Helper()

	fset := token.NewFileSet()

	//nolint:staticcheck // ParseDir is sufficient here; this SDK stays zero-dependency.
	pkgs, err := parser.ParseDir(fset, dir, func(fi fs.FileInfo) bool {
		name := fi.Name()

		return strings.HasSuffix(name, ".go") && !strings.HasSuffix(name, "_test.go")
	}, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("parse %s: %v", dir, err)
	}

	names := map[string]string{}

	record := func(ident *ast.Ident) {
		if ident != nil && ident.Name != "_" {
			names[ident.Name] = fset.Position(ident.Pos()).String()
		}
	}

	for _, pkg := range pkgs {
		for _, file := range pkg.Files {
			ast.Inspect(file, func(node ast.Node) bool {
				switch n := node.(type) {
				case *ast.FuncDecl:
					record(n.Name)
				case *ast.TypeSpec:
					record(n.Name)
				case *ast.ValueSpec:
					for _, name := range n.Names {
						record(name)
					}
				case *ast.Field:
					for _, name := range n.Names {
						record(name)
					}
				}

				return true
			})
		}
	}

	return names
}
