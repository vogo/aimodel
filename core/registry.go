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

package core

import (
	"fmt"
	"sync"
)

var (
	registryMu sync.RWMutex
	registry   = map[string]Factory{}
)

// Register makes a provider factory resolvable by name. It is meant to run
// during package initialization (built-in and third-party provider packages
// register themselves in init) or application startup.
//
// Registration is monotonic and deterministic: an empty name, a nil factory,
// or a duplicate name is a programming error and panics immediately —
// silent overwrites would make dispatch depend on import order.
func Register(name string, factory Factory) {
	if name == "" {
		panic("aimodel: Register called with empty provider name")
	}

	if factory == nil {
		panic(fmt.Sprintf("aimodel: Register called with nil factory for provider %q", name))
	}

	registryMu.Lock()
	defer registryMu.Unlock()

	if _, dup := registry[name]; dup {
		panic(fmt.Sprintf("aimodel: provider %q already registered", name))
	}

	registry[name] = factory
}

// Lookup resolves a registered provider factory by name. It is safe for
// concurrent use.
func Lookup(name string) (Factory, bool) {
	registryMu.RLock()
	defer registryMu.RUnlock()

	f, ok := registry[name]

	return f, ok
}
