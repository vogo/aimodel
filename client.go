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
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/vogo/aimodel/core"
	"github.com/vogo/aimodel/provider/openai"

	// Register the built-in Anthropic provider so WithProvider(anthropic.Name)
	// resolves without the caller importing the subpackage explicitly. Any
	// further protocol is added by importing its own provider subpackage.
	_ "github.com/vogo/aimodel/provider/anthropic"
)

const defaultTimeout = 60 * time.Second

// Client is an AI API client. It resolves a registered provider by name at
// construction time and delegates every chat call to that provider through a
// single shared execution pipeline. The default provider is the
// OpenAI-compatible one; select another with WithProvider (e.g.
// WithProvider(anthropic.Name)).
type Client struct {
	model      string
	httpClient *http.Client
	provider   core.ChatProvider
}

// clientConfig holds the construction-time configuration mutated by Options.
// The generic fields (apiKey, baseURL, model, timeout, httpClient) belong to
// every client; providerName selects the implementation and providerOptions
// carries the provider-specific configuration to its factory.
type clientConfig struct {
	apiKey          string
	baseURL         string
	model           string
	providerName    string
	providerOptions any
	timeout         time.Duration
	httpClient      *http.Client
}

// Option configures a Client.
type Option func(*clientConfig)

// WithAPIKey sets the API key.
func WithAPIKey(key string) Option {
	return func(c *clientConfig) {
		c.apiKey = key
	}
}

// WithDefaultModel sets the default model name.
// If a ChatRequest has an empty Model field, this default is used.
func WithDefaultModel(model string) Option {
	return func(c *clientConfig) {
		c.model = model
	}
}

// WithBaseURL sets the API base URL.
func WithBaseURL(url string) Option {
	return func(c *clientConfig) {
		c.baseURL = strings.TrimRight(url, "/")
	}
}

// WithHTTPClient sets a custom HTTP client.
// Panics if hc is nil — passing nil indicates a programming error.
func WithHTTPClient(hc *http.Client) Option {
	if hc == nil {
		panic("aimodel: WithHTTPClient called with nil *http.Client")
	}

	return func(c *clientConfig) {
		c.httpClient = hc
	}
}

// WithProvider selects the provider implementation by its registered name.
// The default (unset) is the OpenAI-compatible provider (openai.Name). Import
// a provider subpackage (e.g. provider/anthropic) to register its name, then
// pass it here.
func WithProvider(name string) Option {
	return func(c *clientConfig) {
		if name != "" {
			c.providerName = name
		}
	}
}

// WithProviderOptions attaches provider-specific configuration, forwarded
// verbatim to the selected provider's factory. The concrete type is defined by
// the provider package (e.g. anthropic.Options); passing a type the provider
// does not recognize fails construction.
func WithProviderOptions(opts any) Option {
	return func(c *clientConfig) {
		c.providerOptions = opts
	}
}

// WithTimeout sets the HTTP client timeout.
// The timeout is applied after all options, so it works regardless of option ordering.
func WithTimeout(d time.Duration) Option {
	return func(c *clientConfig) {
		c.timeout = d
	}
}

// NewClient creates a new Client with the given options.
//
// Generic configuration falls back to the environment when not set explicitly:
// the API key to AI_API_KEY then OPENAI_API_KEY then ANTHROPIC_API_KEY, the
// base URL to AI_BASE_URL then OPENAI_BASE_URL then ANTHROPIC_BASE_URL, and the
// default model to AI_MODEL. The selected provider's factory validates its own
// required fields (e.g. a base URL for OpenAI, none for Anthropic) and vendor
// options, so those failures surface here at construction time.
func NewClient(opts ...Option) (*Client, error) {
	cfg := &clientConfig{
		timeout:      defaultTimeout,
		providerName: openai.Name,
	}

	if model := GetEnv("AI_MODEL"); model != "" {
		cfg.model = model
	}

	if key := GetEnv("AI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"); key != "" {
		cfg.apiKey = key
	}

	if base := GetEnv("AI_BASE_URL", "OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"); base != "" {
		cfg.baseURL = strings.TrimRight(base, "/")
	}

	// Apply explicit options (override env).
	for _, opt := range opts {
		opt(cfg)
	}

	if cfg.apiKey == "" {
		return nil, ErrNoAPIKey
	}

	factory, ok := core.Lookup(cfg.providerName)
	if !ok {
		return nil, fmt.Errorf("aimodel: unknown provider %q", cfg.providerName)
	}

	prov, err := factory(core.Config{
		APIKey:  cfg.apiKey,
		BaseURL: cfg.baseURL,
		Options: cfg.providerOptions,
	})
	if err != nil {
		return nil, err
	}

	// Apply timeout to the HTTP client at the end, ensuring correct ordering.
	httpClient := cfg.httpClient
	if httpClient == nil {
		httpClient = &http.Client{}
	}

	httpClient.Timeout = cfg.timeout

	return &Client{
		model:      cfg.model,
		httpClient: httpClient,
		provider:   prov,
	}, nil
}
