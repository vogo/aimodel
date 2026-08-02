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
	"errors"
	"fmt"
	"strings"
)

// Sentinel errors for common failure conditions.
//
// Deprecated: the canonical layer is removed in v0.6.0. Use the native
// client and wire types of provider/openai or provider/anthropic instead;
// see MIGRATION.md for the symbol-by-symbol migration table.
var (
	ErrNoAPIKey       = errors.New("aimodel: API key is required")
	ErrNoBaseURL      = errors.New("aimodel: base URL is required")
	ErrStreamClosed   = errors.New("aimodel: stream is closed")
	ErrEmptyResponse  = errors.New("aimodel: empty response from API")
	ErrNoActiveModels = errors.New("aimodel: no active models available")
)

// ErrCapabilityNotSupported reports that the client's resolved provider does
// not implement the requested capability. Match it with errors.Is; the wrapping
// CapabilityError names the provider and the capability.
//
// Deprecated: the canonical layer is removed in v0.6.0. Use the native
// client and wire types of provider/openai or provider/anthropic instead;
// see MIGRATION.md for the symbol-by-symbol migration table.
var ErrCapabilityNotSupported = errors.New("aimodel: capability not supported by provider")

// CapabilityError says which provider was selected and which capability it does
// not implement. It is returned before any network I/O.
//
// Deprecated: the canonical layer is removed in v0.6.0. Use the native
// client and wire types of provider/openai or provider/anthropic instead;
// see MIGRATION.md for the symbol-by-symbol migration table.
type CapabilityError struct {
	Provider   string
	Capability string
}

// Deprecated: the canonical layer is removed in v0.6.0. Use the native
// client and wire types of provider/openai or provider/anthropic instead;
// see MIGRATION.md for the symbol-by-symbol migration table.
func (e *CapabilityError) Error() string {
	return fmt.Sprintf("aimodel: provider %q does not support the %s capability", e.Provider, e.Capability)
}

// Deprecated: the canonical layer is removed in v0.6.0. Use the native
// client and wire types of provider/openai or provider/anthropic instead;
// see MIGRATION.md for the symbol-by-symbol migration table.
func (e *CapabilityError) Unwrap() error {
	return ErrCapabilityNotSupported
}

// APIError represents an error returned by an AI API.
//
// Deprecated: the canonical layer is removed in v0.6.0. Use the native
// client and wire types of provider/openai or provider/anthropic instead;
// see MIGRATION.md for the symbol-by-symbol migration table.
type APIError struct {
	StatusCode int
	Code       string
	Message    string
	Type       string
	Err        error
}

// Deprecated: the canonical layer is removed in v0.6.0. Use the native
// client and wire types of provider/openai or provider/anthropic instead;
// see MIGRATION.md for the symbol-by-symbol migration table.
func (e *APIError) Error() string {
	return fmt.Sprintf("aimodel: API error (status %d): %s - %s", e.StatusCode, e.Code, e.Message)
}

// Deprecated: the canonical layer is removed in v0.6.0. Use the native
// client and wire types of provider/openai or provider/anthropic instead;
// see MIGRATION.md for the symbol-by-symbol migration table.
func (e *APIError) Unwrap() error {
	return e.Err
}

// ModelError associates an error with a specific model name.
//
// Deprecated: the canonical layer is removed in v0.6.0. Use the native
// client and wire types of provider/openai or provider/anthropic instead;
// see MIGRATION.md for the symbol-by-symbol migration table.
type ModelError struct {
	Model string
	Err   error
}

// Deprecated: the canonical layer is removed in v0.6.0. Use the native
// client and wire types of provider/openai or provider/anthropic instead;
// see MIGRATION.md for the symbol-by-symbol migration table.
func (e *ModelError) Error() string {
	return fmt.Sprintf("aimodel: model %s: %v", e.Model, e.Err)
}

// Deprecated: the canonical layer is removed in v0.6.0. Use the native
// client and wire types of provider/openai or provider/anthropic instead;
// see MIGRATION.md for the symbol-by-symbol migration table.
func (e *ModelError) Unwrap() error {
	return e.Err
}

// MultiError collects errors from multiple model attempts.
//
// Deprecated: the canonical layer is removed in v0.6.0. Use the native
// client and wire types of provider/openai or provider/anthropic instead;
// see MIGRATION.md for the symbol-by-symbol migration table.
type MultiError struct {
	Errors []ModelError
}

// Deprecated: the canonical layer is removed in v0.6.0. Use the native
// client and wire types of provider/openai or provider/anthropic instead;
// see MIGRATION.md for the symbol-by-symbol migration table.
func (e *MultiError) Error() string {
	if len(e.Errors) == 0 {
		return ErrNoActiveModels.Error()
	}

	var b strings.Builder

	b.WriteString("aimodel: all models failed: ")

	for i, me := range e.Errors {
		if i > 0 {
			b.WriteString("; ")
		}

		fmt.Fprintf(&b, "%s: %v", me.Model, me.Err)
	}

	return b.String()
}

// Unwrap returns all wrapped errors for Go 1.20+ multi-error unwrapping.
// This allows errors.Is and errors.As to match any model's error.
//
// Deprecated: the canonical layer is removed in v0.6.0. Use the native
// client and wire types of provider/openai or provider/anthropic instead;
// see MIGRATION.md for the symbol-by-symbol migration table.
func (e *MultiError) Unwrap() []error {
	if len(e.Errors) == 0 {
		return []error{ErrNoActiveModels}
	}

	errs := make([]error, len(e.Errors))
	for i := range e.Errors {
		errs[i] = &e.Errors[i]
	}

	return errs
}
