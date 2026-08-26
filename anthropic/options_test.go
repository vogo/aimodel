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

package anthropic

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestNativeWithTimeoutBoundsTheCall(t *testing.T) {
	// The handler blocks until the test releases it. A client-side timeout does
	// not cancel the server's request context on HTTP/1.1, so releasing it
	// explicitly is what lets httptest.Server.Close return.
	release := make(chan struct{})

	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		<-release
	}))
	defer server.Close()
	defer close(release)

	client := NewClient("key", WithBaseURL(server.URL), WithHTTPClient(server.Client()), WithTimeout(50*time.Millisecond))

	start := time.Now()

	_, err := client.Messages(context.Background(), &MessagesRequest{Model: "claude-sonnet-5", MaxTokens: 16})
	if err == nil {
		t.Fatal("a hanging backend must fail the call")
	}

	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Errorf("call took %v; the timeout did not apply", elapsed)
	}
}

func TestNativeWithTimeoutDoesNotMutateTheCallersClient(t *testing.T) {
	caller := &http.Client{}

	client := NewClient("key", WithHTTPClient(caller), WithTimeout(time.Second))

	if caller.Timeout != 0 {
		t.Errorf("caller's client was mutated: timeout = %v", caller.Timeout)
	}

	if client.httpClient.Timeout != time.Second {
		t.Errorf("client timeout = %v, want 1s", client.httpClient.Timeout)
	}

	if client.httpClient == caller {
		t.Error("WithTimeout must configure a copy, not the caller's client")
	}
}

func TestNativeWithTimeoutPreservesAnEarlierTransport(t *testing.T) {
	transport := &http.Transport{}

	client := NewClient("key", WithHTTPClient(&http.Client{Transport: transport}), WithTimeout(time.Second))

	if client.httpClient.Transport != transport {
		t.Error("WithTimeout must preserve a transport installed by WithHTTPClient")
	}
}
