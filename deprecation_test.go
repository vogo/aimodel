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
	"os"
	"strings"
	"testing"
)

// This file guards the v0.5.1 deprecation window: every public symbol removed
// in v0.6.0 must carry a `Deprecated:` paragraph that staticcheck recognizes,
// and the migration document those comments point at must exist and cover
// every migration path. Delete this file together with the canonical layer.

const migrationDoc = "MIGRATION.md"

// deprecatedMarker is the paragraph prefix staticcheck (SA1019) looks for. It
// must start a paragraph, i.e. follow a blank comment line or open the comment.
const deprecatedMarker = "Deprecated: "

// exportedDecls returns, per exported top-level declaration in dir, whether its
// documentation carries a deprecation marker. Grouped const/var declarations
// are reported under the group's first exported name, matching how staticcheck
// propagates a GenDecl's doc comment to every spec inside it.
func exportedDecls(t *testing.T, dir string) map[string]bool {
	t.Helper()

	fset := token.NewFileSet()

	//nolint:staticcheck // ParseDir is sufficient here; this SDK stays zero-dependency.
	pkgs, err := parser.ParseDir(fset, dir, func(fi fs.FileInfo) bool {
		return strings.HasSuffix(fi.Name(), ".go") && !strings.HasSuffix(fi.Name(), "_test.go")
	}, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse %s: %v", dir, err)
	}

	found := map[string]bool{}

	for _, pkg := range pkgs {
		for _, file := range pkg.Files {
			for _, decl := range file.Decls {
				name, deprecated, ok := declDeprecation(decl)
				if !ok {
					continue
				}

				found[name] = deprecated
			}
		}
	}

	return found
}

// declDeprecation reports the exported name a declaration should be indexed
// under and whether it is deprecated. It returns false for unexported
// declarations and for methods on unexported types.
//
// A grouped const/var is deprecated when either the group's doc comment or the
// individual spec's carries the marker — the same precedence staticcheck
// applies when it attaches deprecation facts to every name in a group.
func declDeprecation(decl ast.Decl) (string, bool, bool) {
	switch d := decl.(type) {
	case *ast.FuncDecl:
		if !d.Name.IsExported() {
			return "", false, false
		}

		if d.Recv != nil {
			recv := receiverName(d.Recv)
			if !ast.IsExported(recv) {
				return "", false, false
			}

			return recv + "." + d.Name.Name, isDeprecated(d.Doc), true
		}

		return d.Name.Name, isDeprecated(d.Doc), true
	case *ast.GenDecl:
		if d.Tok == token.IMPORT {
			return "", false, false
		}

		for _, spec := range d.Specs {
			switch s := spec.(type) {
			case *ast.TypeSpec:
				if s.Name.IsExported() {
					return s.Name.Name, isDeprecated(d.Doc) || isDeprecated(s.Doc), true
				}
			case *ast.ValueSpec:
				for _, n := range s.Names {
					if n.IsExported() {
						return n.Name, isDeprecated(d.Doc) || isDeprecated(s.Doc), true
					}
				}
			}
		}
	}

	return "", false, false
}

func receiverName(recv *ast.FieldList) string {
	if len(recv.List) == 0 {
		return ""
	}

	expr := recv.List[0].Type
	if star, ok := expr.(*ast.StarExpr); ok {
		expr = star.X
	}

	if ident, ok := expr.(*ast.Ident); ok {
		return ident.Name
	}

	return ""
}

// isDeprecated reports whether doc ends with a deprecation paragraph — the
// exact shape staticcheck requires: the marker opens the final paragraph.
func isDeprecated(doc *ast.CommentGroup) bool {
	if doc == nil {
		return false
	}

	paragraphs := strings.Split(doc.Text(), "\n\n")

	return strings.HasPrefix(paragraphs[len(paragraphs)-1], deprecatedMarker)
}

// TestCanonicalPackageIsDeprecated verifies every exported declaration in the
// canonical package — and the package clause itself — is marked deprecated.
func TestCanonicalPackageIsDeprecated(t *testing.T) {
	for name, deprecated := range exportedDecls(t, "ais") {
		if !deprecated {
			t.Errorf("ais.%s is removed in v0.6.0 but carries no Deprecated: comment", name)
		}
	}

	src, err := os.ReadFile("ais/provider.go")
	if err != nil {
		t.Fatalf("read package doc: %v", err)
	}

	if !strings.Contains(string(src), "// Deprecated: this package is removed in v0.6.0") {
		t.Error("package ais must carry a package-level Deprecated: paragraph")
	}
}

// TestRootCanonicalAPIIsDeprecated verifies the root package's canonical
// client surface is marked deprecated. Everything the root package exports
// today is removed in v0.6.0.
func TestRootCanonicalAPIIsDeprecated(t *testing.T) {
	decls := exportedDecls(t, ".")

	// Spot-check the entry points a caller is most likely to hold, so a new
	// export cannot quietly join the surface without a deprecation notice.
	for _, name := range []string{
		"Client", "NewClient", "Option", "WithAPIKey", "WithBaseURL", "WithDefaultModel",
		"WithHTTPClient", "WithProvider", "WithProviderOptions", "WithTimeout",
		"ChatCompleter", "Stream", "WrapStream", "InterceptStream",
		"Responder", "CapabilityResponses", "GetEnv",
	} {
		deprecated, ok := decls[name]
		if !ok {
			t.Errorf("aimodel.%s not found; update this list if it was already removed", name)

			continue
		}

		if !deprecated {
			t.Errorf("aimodel.%s is removed in v0.6.0 but carries no Deprecated: comment", name)
		}
	}

	for name, deprecated := range decls {
		if !deprecated {
			t.Errorf("aimodel.%s is removed in v0.6.0 but carries no Deprecated: comment", name)
		}
	}
}

// TestProviderCanonicalEntryPointsAreDeprecated verifies the registry entry
// points and the canonical extension surface of both providers are marked
// deprecated, while their native wire types are not.
func TestProviderCanonicalEntryPointsAreDeprecated(t *testing.T) {
	openaiDecls := exportedDecls(t, "provider/openai")
	for _, name := range []string{"Name", "New"} {
		if !openaiDecls[name] {
			t.Errorf("openai.%s is removed in v0.6.0 but carries no Deprecated: comment", name)
		}
	}

	anthropicDecls := exportedDecls(t, "provider/anthropic")
	for _, name := range []string{
		"Name", "New", "Options",
		"RequestExtension", "MessageExtension", "ToolExtension",
		"ChoiceExtension", "ResponseExtension", "UsageExtension",
		"ExtendRequest", "ExtendMessage", "ExtendTool",
		"RequestExtensionOf", "MessageExtensionOf", "ToolExtensionOf",
		"ChoiceExtensionOf", "ChunkChoiceExtensionOf", "ResponseExtensionOf",
		"ChunkExtensionOf", "UsageExtensionOf",
		"FinishReasonModelContextWindowExceeded",
	} {
		if !anthropicDecls[name] {
			t.Errorf("anthropic.%s is removed in v0.6.0 but carries no Deprecated: comment", name)
		}
	}

	// Native wire types survive the removal and must not be marked.
	for _, name := range []string{
		"Client", "NewClient", "MessagesRequest", "MessagesResponse", "MessagesUsage",
		"StopDetails", "ResponseContainer", "ServerToolUse",
	} {
		if anthropicDecls[name] {
			t.Errorf("anthropic.%s survives v0.6.0 and must not be marked deprecated", name)
		}
	}

	for _, name := range []string{
		"Client", "NewClient", "ChatCompletionRequest", "ChatCompletionResponse",
		"ResponsesRequest", "Response",
	} {
		if openaiDecls[name] {
			t.Errorf("openai.%s survives v0.6.0 and must not be marked deprecated", name)
		}
	}
}

// TestMigrationDocCoversEveryPath verifies the document the deprecation
// comments point at exists and covers each migration path the release notes
// promise.
func TestMigrationDocCoversEveryPath(t *testing.T) {
	content, err := os.ReadFile(migrationDoc)
	if err != nil {
		t.Fatalf("read %s: %v", migrationDoc, err)
	}

	doc := string(content)

	for _, section := range []string{
		"## Construction",
		"## Chat, non-streaming",
		"## Chat, streaming",
		"## Usage accounting",
		"## Errors",
		"## Models",
		"## Vendor extensions",
		"## Responses API (OpenAI)",
		"## Compose",
		"## Removed symbols",
	} {
		if !strings.Contains(doc, section) {
			t.Errorf("%s is missing the %q section", migrationDoc, section)
		}
	}
}
