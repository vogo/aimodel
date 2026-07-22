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
	"net/http"
	"testing"

	"github.com/vogo/aimodel/core"
)

// newProvider builds a *provider from the given options, failing the test on a
// factory error.
func newProvider(t *testing.T, opts any) *provider {
	t.Helper()

	p, err := New(core.Config{APIKey: "sk-ant-test", Options: opts})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	return p.(*provider)
}

func TestSetHeaders(t *testing.T) {
	tests := []struct {
		name        string
		opts        any
		wantVersion string
		wantBeta    string // "" means header must be absent
	}{
		{
			name:        "default",
			opts:        nil,
			wantVersion: anthropicAPIVersion,
			wantBeta:    "",
		},
		{
			name:        "single beta",
			opts:        Options{Beta: []string{"context-1m-2025-08-07"}},
			wantVersion: anthropicAPIVersion,
			wantBeta:    "context-1m-2025-08-07",
		},
		{
			name:        "multiple beta",
			opts:        Options{Beta: []string{"compaction-2025-01-01", "context-editing-2025-06-01"}},
			wantVersion: anthropicAPIVersion,
			wantBeta:    "compaction-2025-01-01,context-editing-2025-06-01",
		},
		{
			name:        "empty beta values are ignored",
			opts:        Options{Beta: []string{"a", "b", "", "c"}},
			wantVersion: anthropicAPIVersion,
			wantBeta:    "a,b,c",
		},
		{
			name:        "custom version",
			opts:        Options{Version: "2099-01-01"},
			wantVersion: "2099-01-01",
			wantBeta:    "",
		},
		{
			name:        "custom version and beta",
			opts:        Options{Version: "2099-01-01", Beta: []string{"fast-mode"}},
			wantVersion: "2099-01-01",
			wantBeta:    "fast-mode",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := newProvider(t, tt.opts)

			req, err := http.NewRequest(http.MethodPost, "https://example.com/v1/messages", nil)
			if err != nil {
				t.Fatalf("NewRequest: %v", err)
			}

			p.setHeaders(req)

			if got := req.Header.Get("x-api-key"); got != "sk-ant-test" {
				t.Errorf("x-api-key = %q", got)
			}

			if got := req.Header.Get("anthropic-version"); got != tt.wantVersion {
				t.Errorf("anthropic-version = %q, want %q", got, tt.wantVersion)
			}

			_, betaSet := req.Header["Anthropic-Beta"]
			if got := req.Header.Get("anthropic-beta"); got != tt.wantBeta {
				t.Errorf("anthropic-beta = %q, want %q", got, tt.wantBeta)
			}

			if tt.wantBeta == "" && betaSet {
				t.Errorf("anthropic-beta header should be absent, got %q", req.Header.Get("anthropic-beta"))
			}
		})
	}
}

// TestSetHeadersUserProfileID verifies the profile header is sent only when
// configured with a non-empty value, and that it does not disturb the version
// / beta headers.
func TestSetHeadersUserProfileID(t *testing.T) {
	tests := []struct {
		name        string
		opts        any
		wantProfile string // "" means the header must be absent
	}{
		{
			name: "unset",
			opts: nil,
		},
		{
			name:        "set",
			opts:        Options{UserProfileID: "user_abc123"},
			wantProfile: "user_abc123",
		},
		{
			name: "empty string is absent",
			opts: Options{UserProfileID: ""},
		},
		{
			name:        "coexists with beta and version",
			opts:        Options{UserProfileID: "user_abc123", Beta: []string{"fast-mode"}, Version: "2099-01-01"},
			wantProfile: "user_abc123",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := newProvider(t, tt.opts)

			req, err := http.NewRequest(http.MethodPost, "https://example.com/v1/messages", nil)
			if err != nil {
				t.Fatalf("NewRequest: %v", err)
			}

			p.setHeaders(req)

			_, present := req.Header["Anthropic-User-Profile-Id"]

			if tt.wantProfile == "" {
				if present {
					t.Errorf("anthropic-user-profile-id should be absent, got %q", req.Header.Get("anthropic-user-profile-id"))
				}

				return
			}

			if got := req.Header.Get("anthropic-user-profile-id"); got != tt.wantProfile {
				t.Errorf("anthropic-user-profile-id = %q, want %q", got, tt.wantProfile)
			}
		})
	}
}

// TestEndpointDefault verifies the provider falls back to the public base URL
// when none is configured, and honors an explicit one.
func TestEndpointDefault(t *testing.T) {
	p := newProvider(t, nil)
	if got := p.endpoint(); got != anthropicDefaultBaseURL+"/v1/messages" {
		t.Errorf("endpoint() = %q, want %q", got, anthropicDefaultBaseURL+"/v1/messages")
	}

	withBase, err := New(core.Config{APIKey: "sk-ant-test", BaseURL: "https://custom.example.com"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	if got := withBase.(*provider).endpoint(); got != "https://custom.example.com/v1/messages" {
		t.Errorf("endpoint() = %q", got)
	}
}

// TestNewRejectsUnknownOptions verifies the factory rejects an options value of
// a type it does not recognize.
func TestNewRejectsUnknownOptions(t *testing.T) {
	_, err := New(core.Config{APIKey: "sk-ant-test", Options: "not-anthropic-options"})
	if err == nil {
		t.Fatal("expected error for unknown options type")
	}
}
