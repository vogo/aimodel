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
	"fmt"
)

// Sentinel errors for common failure conditions.
var (
	ErrNoAPIKey      = errors.New("aimodel: API key is required")
	ErrNoBaseURL     = errors.New("aimodel: base URL is required")
	ErrStreamClosed  = errors.New("aimodel: stream is closed")
	ErrEmptyResponse = errors.New("aimodel: empty response from API")
)

// APIError represents an error returned by the OpenAI API.
type APIError struct {
	StatusCode int
	Code       string
	Message    string
	Type       string
	Err        error
}

func (e *APIError) Error() string {
	return fmt.Sprintf("aimodel: API error (status %d): %s - %s", e.StatusCode, e.Code, e.Message)
}

func (e *APIError) Unwrap() error {
	return e.Err
}
