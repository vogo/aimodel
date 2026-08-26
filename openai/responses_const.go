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

package openai

// Responses API discriminator values, verified 2026-08-01. Every field these
// name stays an open string, so a value this SDK does not list still passes
// through untouched.

// Message roles accepted by a Responses `message` item.
const (
	ResponseRoleUser      = "user"
	ResponseRoleAssistant = "assistant"
	ResponseRoleSystem    = "system"
	ResponseRoleDeveloper = "developer"
)

// Item discriminators (`output[].type` and structured `input[].type`). Only the
// values listed here decode into a dedicated payload; every other item type is
// preserved verbatim on ResponseInputItem.Raw / ResponseOutputItem.Raw.
const (
	ResponseItemTypeMessage             = "message"
	ResponseItemTypeReasoning           = "reasoning"
	ResponseItemTypeFunctionCall        = "function_call"
	ResponseItemTypeFunctionCallOutput  = "function_call_output"
	ResponseItemTypeWebSearchCall       = "web_search_call"
	ResponseItemTypeFileSearchCall      = "file_search_call"
	ResponseItemTypeCodeInterpreterCall = "code_interpreter_call"
	ResponseItemTypeItemReference       = "item_reference"
)

// Content-part discriminators.
const (
	ResponseContentTypeInputText     = "input_text"
	ResponseContentTypeInputImage    = "input_image"
	ResponseContentTypeInputFile     = "input_file"
	ResponseContentTypeInputAudio    = "input_audio"
	ResponseContentTypeOutputText    = "output_text"
	ResponseContentTypeRefusal       = "refusal"
	ResponseContentTypeReasoningText = "reasoning_text"
	ResponseContentTypeSummaryText   = "summary_text"
)

// Annotation discriminators on an output_text part.
const (
	ResponseAnnotationTypeFileCitation          = "file_citation"
	ResponseAnnotationTypeURLCitation           = "url_citation"
	ResponseAnnotationTypeContainerFileCitation = "container_file_citation"
	ResponseAnnotationTypeFilePath              = "file_path"
)

// Tool discriminators this SDK models. web_search_2025_08_26 is the dated alias
// of the same hosted web search contract.
const (
	ResponseToolTypeFunction          = "function"
	ResponseToolTypeFileSearch        = "file_search"
	ResponseToolTypeWebSearch         = "web_search"
	ResponseToolTypeWebSearch20250826 = "web_search_2025_08_26"
	ResponseToolTypeCodeInterpreter   = "code_interpreter"
)

// Web search action discriminators.
const (
	ResponseWebSearchActionSearch     = "search"
	ResponseWebSearchActionOpenPage   = "open_page"
	ResponseWebSearchActionFindInPage = "find_in_page"
)

// Code interpreter output discriminators.
const (
	ResponseCodeInterpreterOutputLogs  = "logs"
	ResponseCodeInterpreterOutputImage = "image"
)

// Response lifecycle statuses (`response.status`).
const (
	ResponseStatusQueued     = "queued"
	ResponseStatusInProgress = "in_progress"
	ResponseStatusCompleted  = "completed"
	ResponseStatusIncomplete = "incomplete"
	ResponseStatusFailed     = "failed"
	ResponseStatusCancelled  = "cancelled"
)

// `include` values that ask the server for extra output data. Hosted-tool
// results are omitted unless the matching value is requested.
const (
	ResponseIncludeFileSearchResults         = "file_search_call.results"
	ResponseIncludeWebSearchResults          = "web_search_call.results"
	ResponseIncludeWebSearchActionSources    = "web_search_call.action.sources"
	ResponseIncludeCodeInterpreterOutputs    = "code_interpreter_call.outputs"
	ResponseIncludeInputImageURL             = "message.input_image.image_url"
	ResponseIncludeComputerCallOutputImage   = "computer_call_output.output.image_url"
	ResponseIncludeReasoningEncryptedContent = "reasoning.encrypted_content"
	ResponseIncludeOutputTextLogprobs        = "message.output_text.logprobs"
)

// Truncation strategies.
const (
	ResponseTruncationAuto     = "auto"
	ResponseTruncationDisabled = "disabled"
)
