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
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/vogo/aimodel/ais"
	"github.com/vogo/aimodel/provider/anthropic"
)

// completionResponse is a minimal valid OpenAI completion body used by the
// capturing test servers below.
const completionResponse = `{"id":"x","object":"chat.completion","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`

// captured records what a test server observed about a request.
type captured struct {
	auth string
	path string
	seen bool
}

// runOpenAICapture builds a client from opts (the caller supplies the base URL,
// possibly via env pointing at the returned server), issues one chat call, and
// reports what the server saw. The server URL is passed to withURL so the
// caller can wire it into an option or an env var before construction.
func runOpenAICapture(t *testing.T, build func(url string) (*Client, error)) captured {
	t.Helper()

	var got captured

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got.seen = true
		got.auth = r.Header.Get("Authorization")
		got.path = r.URL.Path

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(completionResponse))
	}))
	defer srv.Close()

	c, err := build(srv.URL)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	_, err = c.ChatCompletion(context.Background(), &ais.ChatRequest{
		Model:    "gpt-4o",
		Messages: []ais.Message{{Role: ais.RoleUser, Content: ais.NewTextContent("hi")}},
	})
	if err != nil {
		t.Fatalf("ChatCompletion: %v", err)
	}

	return got
}

func TestNewClientWithAPIKey(t *testing.T) {
	got := runOpenAICapture(t, func(url string) (*Client, error) {
		return NewClient(WithAPIKey("sk-test-key"), WithBaseURL(url))
	})

	if got.auth != "Bearer sk-test-key" {
		t.Errorf("Authorization = %q, want Bearer sk-test-key", got.auth)
	}
}

func TestNewClientBaseURLReached(t *testing.T) {
	got := runOpenAICapture(t, func(url string) (*Client, error) {
		return NewClient(WithAPIKey("sk-test"), WithBaseURL(url))
	})

	if !got.seen {
		t.Fatal("request did not reach the configured base URL")
	}

	if got.path != "/chat/completions" {
		t.Errorf("path = %q, want /chat/completions", got.path)
	}
}

func TestNewClientTrailingSlashTrimmed(t *testing.T) {
	got := runOpenAICapture(t, func(url string) (*Client, error) {
		// A trailing slash must be trimmed so the path is not "//chat/completions".
		return NewClient(WithAPIKey("sk-test"), WithBaseURL(url+"/"))
	})

	if got.path != "/chat/completions" {
		t.Errorf("path = %q, want /chat/completions (trailing slash not trimmed)", got.path)
	}
}

func TestNewClientCustomTimeout(t *testing.T) {
	c, err := NewClient(
		WithAPIKey("sk-test"),
		WithBaseURL("https://ais.example.com/v1"),
		WithTimeout(5*time.Minute),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	if c.httpClient.Timeout != 5*time.Minute {
		t.Errorf("timeout = %v", c.httpClient.Timeout)
	}
}

func TestNewClientTimeoutWithCustomHTTPClient(t *testing.T) {
	custom := &http.Client{Timeout: 10 * time.Second}

	// Timeout should apply regardless of option ordering.
	c, err := NewClient(
		WithAPIKey("sk-test"),
		WithBaseURL("https://ais.example.com/v1"),
		WithTimeout(5*time.Minute),
		WithHTTPClient(custom),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	if c.httpClient != custom {
		t.Error("httpClient should be the custom client")
	}

	if c.httpClient.Timeout != 5*time.Minute {
		t.Errorf("timeout = %v, want 5m (timeout should apply to custom client)", c.httpClient.Timeout)
	}
}

func TestNewClientCustomHTTPClient(t *testing.T) {
	custom := &http.Client{Timeout: 10 * time.Second}

	c, err := NewClient(
		WithAPIKey("sk-test"),
		WithBaseURL("https://ais.example.com/v1"),
		WithHTTPClient(custom),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}

	if c.httpClient != custom {
		t.Error("httpClient should be the custom client")
	}
}

func TestNewClientNilHTTPClientPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for nil *http.Client")
		}
	}()

	WithHTTPClient(nil)
}

func TestNewClientEnvFallback(t *testing.T) {
	got := runOpenAICapture(t, func(url string) (*Client, error) {
		t.Setenv("AI_API_KEY", "sk-env-key")
		t.Setenv("AI_BASE_URL", url)

		return NewClient()
	})

	if got.auth != "Bearer sk-env-key" {
		t.Errorf("Authorization = %q, want Bearer sk-env-key", got.auth)
	}

	if !got.seen {
		t.Error("request did not reach the env base URL")
	}
}

func TestNewClientOpenAIEnvFallback(t *testing.T) {
	got := runOpenAICapture(t, func(url string) (*Client, error) {
		t.Setenv("OPENAI_API_KEY", "sk-openai-key")
		t.Setenv("OPENAI_BASE_URL", url)

		return NewClient()
	})

	if got.auth != "Bearer sk-openai-key" {
		t.Errorf("Authorization = %q, want Bearer sk-openai-key", got.auth)
	}
}

func TestNewClientAIEnvOverridesOpenAIEnv(t *testing.T) {
	got := runOpenAICapture(t, func(url string) (*Client, error) {
		t.Setenv("AI_API_KEY", "sk-ai")
		t.Setenv("AI_BASE_URL", url)
		t.Setenv("OPENAI_API_KEY", "sk-openai")
		t.Setenv("OPENAI_BASE_URL", "https://openai.ais.com/v1")

		return NewClient()
	})

	if got.auth != "Bearer sk-ai" {
		t.Errorf("Authorization = %q, want Bearer sk-ai (AI_ should take precedence)", got.auth)
	}

	if !got.seen {
		t.Error("request should reach the AI_BASE_URL server, not the OPENAI_BASE_URL one")
	}
}

func TestNewClientOptionOverridesEnv(t *testing.T) {
	got := runOpenAICapture(t, func(url string) (*Client, error) {
		t.Setenv("AI_API_KEY", "sk-env")
		t.Setenv("AI_BASE_URL", "https://env.ais.com/v1")

		return NewClient(WithAPIKey("sk-option"), WithBaseURL(url))
	})

	if got.auth != "Bearer sk-option" {
		t.Errorf("Authorization = %q, want Bearer sk-option", got.auth)
	}

	if !got.seen {
		t.Error("request should reach the option base URL, not the env one")
	}
}

func TestNewClientNoAPIKeyError(t *testing.T) {
	t.Setenv("AI_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("ANTHROPIC_API_KEY", "")

	_, err := NewClient(WithBaseURL("https://ais.example.com/v1"))
	if !errors.Is(err, ais.ErrNoAPIKey) {
		t.Errorf("err = %v, want ais.ErrNoAPIKey", err)
	}
}

func TestNewClientNoBaseURLAllowedForAnthropic(t *testing.T) {
	t.Setenv("AI_BASE_URL", "")
	t.Setenv("OPENAI_BASE_URL", "")
	t.Setenv("ANTHROPIC_BASE_URL", "")

	// The Anthropic provider has a default base URL, so construction succeeds
	// without one.
	_, err := NewClient(WithAPIKey("sk-test"), WithProvider(anthropic.Name))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
}

func TestNewClientNoBaseURLErrorForOpenAI(t *testing.T) {
	t.Setenv("AI_BASE_URL", "")
	t.Setenv("OPENAI_BASE_URL", "")
	t.Setenv("ANTHROPIC_BASE_URL", "")

	// The OpenAI provider requires a base URL at construction time.
	_, err := NewClient(WithAPIKey("sk-test"))
	if !errors.Is(err, ais.ErrNoBaseURL) {
		t.Errorf("err = %v, want ais.ErrNoBaseURL", err)
	}
}

func TestNewClientUnknownProvider(t *testing.T) {
	_, err := NewClient(WithAPIKey("sk-test"), WithProvider("does-not-exist"))
	if err == nil {
		t.Fatal("expected error for unknown provider")
	}
}
