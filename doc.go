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

// Package aimodel is the module root and exports nothing. A caller picks a
// protocol by importing its client package:
//
//   - [github.com/vogo/aimodel/provider/openai] — Chat Completions and
//     Responses, for OpenAI and every OpenAI-compatible backend.
//   - [github.com/vogo/aimodel/provider/anthropic] — the Messages API.
//   - [github.com/vogo/aimodel/composes] — the protocol-neutral routing core
//     (strategies, health tracking, failover), with
//     [github.com/vogo/aimodel/composes/openais] and
//     [github.com/vogo/aimodel/composes/anthropics] binding it to their wire
//     types.
//
// Up to v0.5.x this package held a unified client over a vendor-neutral
// request/response model. v0.7.0 removed it: the two protocols are expressed
// completely and independently instead of through a shared schema that could
// only carry what both had in common. See MIGRATION.md for the symbol-by-symbol
// migration table and doc/adr/0007-provider-native-as-the-only-public-interface.md
// for the reasoning.
//
// The package clause survives so the module's architectural guard tests have a
// home; it declares no API of its own.
package aimodel
