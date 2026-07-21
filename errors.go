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

import "github.com/vogo/aimodel/core"

// The error model lives in the core package (providers convert vendor wire
// errors into it); the root package re-exports it. Sentinel variables are
// the same instances, so errors.Is matches across packages.

// Sentinel errors for common failure conditions.
var (
	ErrNoAPIKey       = core.ErrNoAPIKey
	ErrNoBaseURL      = core.ErrNoBaseURL
	ErrStreamClosed   = core.ErrStreamClosed
	ErrEmptyResponse  = core.ErrEmptyResponse
	ErrNoActiveModels = core.ErrNoActiveModels
)

type (
	// APIError represents an error returned by an AI API.
	APIError = core.APIError
	// ModelError associates an error with a specific model name.
	ModelError = core.ModelError
	// MultiError collects errors from multiple model attempts.
	MultiError = core.MultiError
)
