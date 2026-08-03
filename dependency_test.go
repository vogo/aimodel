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

// These tests enforce provider isolation in CI rather than by convention,
// because the failure mode they guard against — a shared semantic layer growing
// back one helper at a time — is gradual and reads as reasonable at every step.

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

// TestComposesCoreImportsNoProvider verifies the routing core is what it claims
// to be: protocol-neutral machinery, not a wire-format tool.
//
// Once the core serves more than one protocol, importing any single provider
// would make one wire format privileged — and importing two would be a
// canonical layer with extra steps.
func TestComposesCoreImportsNoProvider(t *testing.T) {
	imports := packageImports(t, "composes")

	for path := range imports {
		if strings.HasPrefix(path, "github.com/vogo/aimodel") {
			t.Errorf("the composes core must import nothing from this module, found %q", path)
		}
	}
}

// TestComposeWrappersAreIsolated verifies the two protocol wrappers stay
// independent of each other: each sees exactly its own provider, neither sees
// the other's package, and neither reaches the module root. A wrapper that
// imported the other would be the first half of a cross-protocol request model.
func TestComposeWrappersAreIsolated(t *testing.T) {
	wrappers := map[string]struct{ own, forbidden string }{
		"composes/openais":    {own: "openai", forbidden: "anthropic"},
		"composes/anthropics": {own: "anthropic", forbidden: "openai"},
	}

	for dir, want := range wrappers {
		imports := packageImports(t, dir)

		if !hasProviderImport(imports, want.own) {
			t.Errorf("%s should import provider/%s, the protocol it wraps", dir, want.own)
		}

		if hasProviderImport(imports, want.forbidden) {
			t.Errorf("%s must not import provider/%s", dir, want.forbidden)
		}

		if !imports["github.com/vogo/aimodel/composes"] {
			t.Errorf("%s should build on the neutral composes core", dir)
		}

		if imports["github.com/vogo/aimodel"] {
			t.Errorf("%s must not depend on the root package", dir)
		}

		for other := range wrappers {
			if other != dir && imports["github.com/vogo/aimodel/"+other] {
				t.Errorf("%s must not import %s; the two pools share routing, never types", dir, other)
			}
		}
	}
}

// TestComposesCoreExportsNoProviderType verifies the routing core's public API
// carries no provider request or response type. The import check above already
// makes that impossible today; this guard states the invariant directly, so a
// future import of a provider fails here as a *public API* violation rather
// than looking like a mere dependency question.
func TestComposesCoreExportsNoProviderType(t *testing.T) {
	fset := token.NewFileSet()

	//nolint:staticcheck // ParseDir is sufficient here; this SDK stays zero-dependency.
	pkgs, err := parser.ParseDir(fset, "composes", func(fi fs.FileInfo) bool {
		name := fi.Name()

		return strings.HasSuffix(name, ".go") && !strings.HasSuffix(name, "_test.go")
	}, parser.SkipObjectResolution)
	if err != nil {
		t.Fatalf("parse composes: %v", err)
	}

	for _, pkg := range pkgs {
		for path, file := range pkg.Files {
			// Local names bound to a package inside this module. A qualified
			// identifier using one of these in an exported declaration is a
			// module type crossing into the neutral API.
			moduleAliases := map[string]string{}

			for _, imp := range file.Imports {
				importPath, err := strconv.Unquote(imp.Path.Value)
				if err != nil {
					t.Fatalf("unquote import %s: %v", imp.Path.Value, err)
				}

				if !strings.HasPrefix(importPath, "github.com/vogo/aimodel") {
					continue
				}

				name := importPath[strings.LastIndex(importPath, "/")+1:]
				if imp.Name != nil {
					name = imp.Name.Name
				}

				moduleAliases[name] = importPath
			}

			if len(moduleAliases) == 0 {
				continue
			}

			for _, decl := range file.Decls {
				if _, exported := exportedDeclName(decl); !exported {
					continue
				}

				ast.Inspect(decl, func(node ast.Node) bool {
					sel, ok := node.(*ast.SelectorExpr)
					if !ok {
						return true
					}

					ident, ok := sel.X.(*ast.Ident)
					if !ok {
						return true
					}

					if importPath, found := moduleAliases[ident.Name]; found {
						t.Errorf("%s: exported API references %s.%s from %q; the routing core's public API is protocol-neutral",
							path, ident.Name, sel.Sel.Name, importPath)
					}

					return true
				})
			}
		}
	}
}

// TestRootPackageExportsNothing verifies the root package stays empty. It has
// no unified client, no shared schema and no provider imports: a caller reaches
// a protocol by importing its own package, which is what makes the two
// protocols independent.
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

// TestNoSharedSemanticPackage verifies the two *providers* share no package
// from this module. A type both of them reach for is a canonical layer by
// another name, whatever it is called.
//
// The compose wrappers do share one package — the neutral routing core — and
// that is not a loophole in this check: the core is downstream of both, imports
// neither, and carries no type either provider names. The rule this test
// enforces is about what a provider depends on, which is what a canonical layer
// would have to change.
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

// neutralPackages are the packages this module declares vendor-neutral. composes
// is one of them: its OpenAI-wire surface lives in composes/openais, leaving a
// routing core that may not name a protocol concept. The wrappers are
// deliberately absent — naming their own protocol is their whole job.
var neutralPackages = []string{".", "composes"}

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
