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

// Package composes is the protocol-neutral routing core shared by every
// multi-backend wrapper in this module: the pool's active endpoint, selection
// strategies, in-call retries, endpoint health, alias identity, observation and
// failure attribution.
//
// It has no client, no request type and no provider dependency. A wrapper
// package binds the protocol — [github.com/vogo/aimodel/composes/openais] for
// the OpenAI wire (Chat Completions and Responses),
// [github.com/vogo/aimodel/composes/anthropics] for Anthropic Messages — and
// calls [Dispatch] with a closure that performs one attempt against one
// endpoint index:
//
//	router, err := composes.NewRouter(composes.StrategyFailover, endpoints)
//	value, err := composes.Dispatch(ctx, router, call,
//	    func(ctx context.Context, endpoint int) (*T, error) { /* one attempt */ })
//
// Typical use does not touch this package directly beyond its strategies,
// options and errors:
//
//	cc, err := openais.NewFromEndpoints(composes.StrategyWeight, specs,
//	    composes.WithRetryPolicy(time.Second, 3),
//	    composes.WithRecoverTime(5*time.Minute))
//
// # One active endpoint
//
// A pool serves its calls from a single active endpoint. The strategy chooses
// that endpoint when the pool has none or the current one is judged dead; it
// does not run per call, so successive successful calls stay on one backend even
// under [StrategyRandom] or [StrategyWeight]. A failing endpoint is retried in
// place with exponential waits, then marked dead and replaced — and a dead
// endpoint returns to candidacy after the recover time without displacing
// whoever took its place.
//
// A router also serves one dispatch at a time; concurrent callers queue for it,
// and a queued caller is released by its own context. Throughput per pool is one
// request, so parallelism means one pool per worker. See
// doc/adr/0009-stateful-active-endpoint-with-in-call-retry.md.
//
// # The boundary
//
// What is shared here is routing *mechanism*. What is deliberately not shared
// is protocol *semantics*: this package must never gain a request or response
// model, a field mapping, or an identifier that names a protocol concept
// (message, content, tool, usage, …). Capability declarations reach it as
// opaque strings whose meaning lives in the wrapper that produced them.
//
// The consequence is that pools do not mix: an OpenAI pool and an Anthropic
// pool are separate routers with separate health, and there is no
// cross-protocol failover. Sharing the state machine is what this design
// permits; sharing a request is what it forbids. See
// doc/adr/0008-shared-routing-core-across-protocol-wrappers.md and
// doc/design/compose.md.
package composes
