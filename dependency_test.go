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
	"go/parser"
	"go/token"
	"io/fs"
	"strconv"
	"strings"
	"testing"
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
// core foundation, never on the root package (which would create a cycle) or on
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

// TestComposesDependsOnlyOnCapability verifies composes depends on the root
// capability surface, never on the registry, core, or any vendor provider.
func TestComposesDependsOnlyOnCapability(t *testing.T) {
	imports := packageImports(t, "composes")

	for path := range imports {
		if strings.Contains(path, "/provider/") {
			t.Errorf("composes must not import a provider subpackage, found %q", path)
		}

		if path == "github.com/vogo/aimodel/core" {
			t.Error("composes must not import core directly; it depends only on the root capability interface")
		}
	}

	if !imports["github.com/vogo/aimodel"] {
		t.Error("composes should depend on the root capability interface")
	}
}

// TestRootProviderImportsAreBuiltInsOnly verifies the root package only imports
// the two built-in provider subpackages (for default registration) and no other
// vendor package — no third-party vendor translation leaks into the root.
func TestRootProviderImportsAreBuiltInsOnly(t *testing.T) {
	imports := packageImports(t, ".")

	allowed := map[string]bool{
		"github.com/vogo/aimodel/provider/openai":    true,
		"github.com/vogo/aimodel/provider/anthropic": true,
	}

	for path := range imports {
		if strings.Contains(path, "/provider/") && !allowed[path] {
			t.Errorf("root package imports unexpected provider package %q", path)
		}
	}

	if !hasProviderImport(imports, "openai") {
		t.Error("root should import provider/openai (default provider)")
	}

	if !hasProviderImport(imports, "anthropic") {
		t.Error("root should import provider/anthropic (built-in registration)")
	}
}
