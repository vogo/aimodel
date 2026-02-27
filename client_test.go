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
	"errors"
	"net/http"
	"testing"
	"time"
)

func TestNewClientWithAPIKey(t *testing.T) {
	c, err := NewClient(WithAPIKey("sk-test-key"), WithBaseURL("https://api.example.com/v1"))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	if c.apiKey != "sk-test-key" {
		t.Errorf("apiKey = %q", c.apiKey)
	}
}

func TestNewClientCustomBaseURL(t *testing.T) {
	c, err := NewClient(
		WithAPIKey("sk-test"),
		WithBaseURL("https://custom.api.com/v2"),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	if c.baseURL != "https://custom.api.com/v2" {
		t.Errorf("baseURL = %q", c.baseURL)
	}
}

func TestNewClientTrailingSlashTrimmed(t *testing.T) {
	c, err := NewClient(
		WithAPIKey("sk-test"),
		WithBaseURL("https://custom.api.com/v2/"),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	if c.baseURL != "https://custom.api.com/v2" {
		t.Errorf("baseURL = %q", c.baseURL)
	}
}

func TestNewClientCustomTimeout(t *testing.T) {
	c, err := NewClient(
		WithAPIKey("sk-test"),
		WithBaseURL("https://api.example.com/v1"),
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
		WithBaseURL("https://api.example.com/v1"),
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
		WithBaseURL("https://api.example.com/v1"),
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
	t.Setenv("AI_API_KEY", "sk-env-key")
	t.Setenv("AI_BASE_URL", "https://env.api.com/v1")

	c, err := NewClient()
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	if c.apiKey != "sk-env-key" {
		t.Errorf("apiKey = %q, want sk-env-key", c.apiKey)
	}
	if c.baseURL != "https://env.api.com/v1" {
		t.Errorf("baseURL = %q", c.baseURL)
	}
}

func TestNewClientOpenAIEnvFallback(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-openai-key")
	t.Setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

	c, err := NewClient()
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	if c.apiKey != "sk-openai-key" {
		t.Errorf("apiKey = %q, want sk-openai-key", c.apiKey)
	}
	if c.baseURL != "https://api.openai.com/v1" {
		t.Errorf("baseURL = %q", c.baseURL)
	}
}

func TestNewClientAIEnvOverridesOpenAIEnv(t *testing.T) {
	t.Setenv("AI_API_KEY", "sk-ai")
	t.Setenv("AI_BASE_URL", "https://ai.api.com/v1")
	t.Setenv("OPENAI_API_KEY", "sk-openai")
	t.Setenv("OPENAI_BASE_URL", "https://openai.api.com/v1")

	c, err := NewClient()
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	if c.apiKey != "sk-ai" {
		t.Errorf("apiKey = %q, want sk-ai (AI_ should take precedence)", c.apiKey)
	}
	if c.baseURL != "https://ai.api.com/v1" {
		t.Errorf("baseURL = %q, want https://ai.api.com/v1", c.baseURL)
	}
}

func TestNewClientOptionOverridesEnv(t *testing.T) {
	t.Setenv("AI_API_KEY", "sk-env")
	t.Setenv("AI_BASE_URL", "https://env.api.com/v1")

	c, err := NewClient(
		WithAPIKey("sk-option"),
		WithBaseURL("https://option.api.com/v1"),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	if c.apiKey != "sk-option" {
		t.Errorf("apiKey = %q", c.apiKey)
	}
	if c.baseURL != "https://option.api.com/v1" {
		t.Errorf("baseURL = %q", c.baseURL)
	}
}

func TestNewClientNoAPIKeyError(t *testing.T) {
	t.Setenv("AI_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")

	_, err := NewClient(WithBaseURL("https://api.example.com/v1"))
	if !errors.Is(err, ErrNoAPIKey) {
		t.Errorf("err = %v, want ErrNoAPIKey", err)
	}
}

func TestNewClientNoBaseURLError(t *testing.T) {
	t.Setenv("AI_BASE_URL", "")
	t.Setenv("OPENAI_BASE_URL", "")

	_, err := NewClient(WithAPIKey("sk-test"))
	if !errors.Is(err, ErrNoBaseURL) {
		t.Errorf("err = %v, want ErrNoBaseURL", err)
	}
}
