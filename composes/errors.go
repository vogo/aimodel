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

package composes

import (
	"errors"
	"fmt"
	"strings"
)

// ErrNoActiveModels reports that every backend is currently marked unhealthy
// and none is due for a recovery probe.
var ErrNoActiveModels = errors.New("aimodel/composes: no active models available")

// ModelError associates a backend failure with the model that produced it.
// The underlying error is whatever that backend's client returned — this
// package does not classify it, so a caller inspecting a status code declares
// its own interface{ StatusCode() int } and reaches it with errors.As.
type ModelError struct {
	Model string
	Err   error
}

func (e *ModelError) Error() string {
	return fmt.Sprintf("aimodel/composes: model %s: %v", e.Model, e.Err)
}

func (e *ModelError) Unwrap() error { return e.Err }

// MultiError collects the failures of every backend tried for one request.
type MultiError struct {
	Errors []ModelError
}

func (e *MultiError) Error() string {
	if len(e.Errors) == 0 {
		return ErrNoActiveModels.Error()
	}

	var b strings.Builder

	b.WriteString("aimodel/composes: all models failed: ")

	for i := range e.Errors {
		if i > 0 {
			b.WriteString("; ")
		}

		fmt.Fprintf(&b, "%s: %v", e.Errors[i].Model, e.Errors[i].Err)
	}

	return b.String()
}

// Unwrap returns every collected error so errors.Is and errors.As match any
// backend's failure, including through to the provider's own error type.
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
