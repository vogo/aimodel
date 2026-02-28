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
	"net/http"
	"strings"
	"time"
)

const defaultTimeout = 60 * time.Second

// Client is an AI API client compatible with OpenAI-style endpoints.
type Client struct {
	apiKey     string
	baseURL    string
	timeout    time.Duration
	httpClient *http.Client
}

// Option configures a Client.
type Option func(*Client)

// WithAPIKey sets the API key.
func WithAPIKey(key string) Option {
	return func(c *Client) {
		c.apiKey = key
	}
}

// WithBaseURL sets the API base URL.
func WithBaseURL(url string) Option {
	return func(c *Client) {
		c.baseURL = strings.TrimRight(url, "/")
	}
}

// WithHTTPClient sets a custom HTTP client.
// Panics if hc is nil — passing nil indicates a programming error.
func WithHTTPClient(hc *http.Client) Option {
	if hc == nil {
		panic("aimodel: WithHTTPClient called with nil *http.Client")
	}

	return func(c *Client) {
		c.httpClient = hc
	}
}

// WithTimeout sets the HTTP client timeout.
// The timeout is applied after all options, so it works regardless of option ordering.
func WithTimeout(d time.Duration) Option {
	return func(c *Client) {
		c.timeout = d
	}
}

// NewClient creates a new Client with the given options.
// If no API key is provided, it falls back to OPENAI_API_KEY then ANTHROPIC_API_KEY.
// If no base URL is provided, it falls back to OPENAI_BASE_URL then ANTHROPIC_BASE_URL.
// For Anthropic models (claude-*), baseURL defaults to https://api.anthropic.com at request time.
func NewClient(opts ...Option) (*Client, error) {
	c := &Client{
		timeout: defaultTimeout,
	}

	if key := GetEnv("AI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"); key != "" {
		c.apiKey = key
	}

	if base := GetEnv("AI_BASE_URL", "OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"); base != "" {
		c.baseURL = strings.TrimRight(base, "/")
	}

	// Apply explicit options (override env).
	for _, opt := range opts {
		opt(c)
	}

	if c.apiKey == "" {
		return nil, ErrNoAPIKey
	}

	// Apply timeout to the HTTP client at the end, ensuring correct ordering.
	if c.httpClient == nil {
		c.httpClient = &http.Client{}
	}

	c.httpClient.Timeout = c.timeout

	return c, nil
}
