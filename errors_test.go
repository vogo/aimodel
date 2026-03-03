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
	"strings"
	"testing"
)

func TestModelError_Error(t *testing.T) {
	inner := errors.New("connection refused")
	me := &ModelError{Model: "gpt-4", Err: inner}

	got := me.Error()
	if !strings.Contains(got, "gpt-4") {
		t.Fatalf("expected model name in error, got %s", got)
	}

	if !strings.Contains(got, "connection refused") {
		t.Fatalf("expected inner error in message, got %s", got)
	}
}

func TestModelError_Unwrap(t *testing.T) {
	inner := errors.New("timeout")
	me := &ModelError{Model: "m0", Err: inner}

	if !errors.Is(me, inner) {
		t.Fatal("Unwrap should allow errors.Is to match inner error")
	}
}

func TestMultiError_Error_Empty(t *testing.T) {
	me := &MultiError{}
	got := me.Error()

	if got != ErrNoActiveModels.Error() {
		t.Fatalf("empty MultiError.Error() = %q, want %q", got, ErrNoActiveModels.Error())
	}
}

func TestMultiError_Error_WithErrors(t *testing.T) {
	me := &MultiError{
		Errors: []ModelError{
			{Model: "m0", Err: errors.New("fail-0")},
			{Model: "m1", Err: errors.New("fail-1")},
		},
	}

	got := me.Error()
	if !strings.Contains(got, "all models failed") {
		t.Fatalf("expected 'all models failed' prefix, got %s", got)
	}

	if !strings.Contains(got, "m0") || !strings.Contains(got, "m1") {
		t.Fatalf("expected both model names, got %s", got)
	}

	if !strings.Contains(got, "fail-0") || !strings.Contains(got, "fail-1") {
		t.Fatalf("expected both error messages, got %s", got)
	}
}

func TestMultiError_Unwrap_Empty(t *testing.T) {
	me := &MultiError{}
	errs := me.Unwrap()

	if len(errs) != 1 {
		t.Fatalf("expected 1 error, got %d", len(errs))
	}

	if !errors.Is(errs[0], ErrNoActiveModels) {
		t.Fatal("empty MultiError should unwrap to ErrNoActiveModels")
	}
}

func TestMultiError_Unwrap_MatchesInnerErrors(t *testing.T) {
	apiErr := &APIError{StatusCode: 500, Code: "server_error", Message: "down"}
	me := &MultiError{
		Errors: []ModelError{
			{Model: "m0", Err: apiErr},
			{Model: "m1", Err: errors.New("timeout")},
		},
	}

	// errors.As should find APIError through the chain.
	var found *APIError
	if !errors.As(me, &found) {
		t.Fatal("errors.As should find APIError in MultiError chain")
	}

	if found.StatusCode != 500 {
		t.Fatalf("StatusCode = %d, want 500", found.StatusCode)
	}
}

func TestMultiError_ErrorsIs_ErrNoActiveModels(t *testing.T) {
	me := &MultiError{}
	if !errors.Is(me, ErrNoActiveModels) {
		t.Fatal("empty MultiError should match ErrNoActiveModels via errors.Is")
	}
}

func TestAPIError_Error(t *testing.T) {
	e := &APIError{StatusCode: 429, Code: "rate_limit", Message: "too many requests"}
	got := e.Error()

	if !strings.Contains(got, "429") {
		t.Fatalf("expected status code in error, got %s", got)
	}

	if !strings.Contains(got, "rate_limit") {
		t.Fatalf("expected error code in message, got %s", got)
	}
}

func TestAPIError_Unwrap(t *testing.T) {
	inner := errors.New("network error")
	e := &APIError{StatusCode: 500, Err: inner}

	if !errors.Is(e, inner) {
		t.Fatal("APIError.Unwrap should allow errors.Is to match inner error")
	}
}
