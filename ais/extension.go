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
	"fmt"
	"maps"
)

// Extensions is the unified provider extension channel of the canonical
// schema. It carries provider-scoped values on a canonical node, keyed by the
// provider's registered name (e.g. "anthropic"). Each provider package defines
// the concrete type it stores under its own key and only ever reads its own
// namespace; a provider must ignore every other key.
//
// The canonical layer owns only the container lifecycle: cloning (Clone) and
// streaming accumulation (via ExtensionMerger). It never interprets,
// validates, or serializes the values — every Extensions field is tagged
// json:"-", so canonical JSON encoding is byte-for-byte unaffected by any
// extension. Extension values are an in-process translation contract, not a
// cross-process JSON contract; callers needing the full vendor payload should
// use the provider's native surface.
//
// Values are treated as read-only configuration once attached: Clone copies
// the map but shares the values, so a provider must not mutate a stored value
// in place.
type Extensions map[string]any

// Value returns the extension value stored under the provider name, or nil
// when the namespace is empty.
func (e Extensions) Value(provider string) any {
	return e[provider]
}

// Set stores value under the provider name, allocating the map on first use.
func (e *Extensions) Set(provider string, value any) {
	if *e == nil {
		*e = Extensions{}
	}

	(*e)[provider] = value
}

// Clone returns a copy of the extension map. Values are shared, not copied —
// they are read-only configuration by contract.
func (e Extensions) Clone() Extensions {
	if len(e) == 0 {
		return nil
	}

	return maps.Clone(e)
}

// mergeDelta folds a delta extension map into this one, namespace by
// namespace. When the existing value implements ExtensionMerger it decides
// the merge; otherwise the delta value replaces it (last write wins).
func (e *Extensions) mergeDelta(delta Extensions) {
	for provider, dv := range delta {
		if cur, ok := (*e)[provider]; ok {
			if merger, ok := cur.(ExtensionMerger); ok {
				e.Set(provider, merger.MergeExtension(dv))

				continue
			}
		}

		e.Set(provider, dv)
	}
}

// ExtensionMerger is implemented by extension values that accumulate across
// streaming deltas. Message.AppendDelta consults it for the already-stored
// value of a namespace when a delta carries the same namespace; the returned
// value replaces the stored one. Implementations must not mutate the receiver
// or the delta — both may still be referenced by previously delivered chunks —
// and must return a freshly merged value instead.
//
// The canonical layer knows nothing about the value's vendor semantics; a
// stored value that does not implement this interface is simply replaced by
// the delta.
type ExtensionMerger interface {
	MergeExtension(delta any) any
}

// ExtensionTypeError reports that an Extensions namespace holds a value of a
// type the owning provider does not recognize. Providers return it from
// request translation — before any network I/O — so a mis-typed extension
// fails the call instead of being silently dropped.
type ExtensionTypeError struct {
	// Provider is the extension namespace (registered provider name).
	Provider string
	// Node names the canonical node carrying the bad value, e.g. "ChatRequest".
	Node string
	// Want is the expected Go type, e.g. "*anthropic.RequestExtension".
	Want string
	// Value is the offending value as stored.
	Value any
}

// Error implements the error interface.
func (e *ExtensionTypeError) Error() string {
	return fmt.Sprintf("aimodel: %s extension on %s has type %T, want %s",
		e.Provider, e.Node, e.Value, e.Want)
}
