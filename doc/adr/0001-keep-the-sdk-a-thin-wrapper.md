# ADR 0001: Keep the SDK a thin API wrapper

- Status: Accepted
- Date: 2026-07-21

## Context

Cross-provider SDKs can accumulate retries, rate limiting, validation, caching, persistence, and telemetry. Those mechanisms need application-specific policies and can create hidden latency, cost, and side effects when embedded in a low-level client.

## Decision

The SDK is limited to request translation, HTTP connection management, and response normalization. Policy mechanisms such as retry, rate limiting, validation, caching, persistence, logging, and metrics remain the caller's responsibility. One SDK call produces one HTTP request, except for the explicitly multi-model compose path.

## Consequences

- Call behavior and cost stay predictable.
- Requests remain reusable and do not carry side-effecting state.
- Callers must add their own resilience and observability policies.
- Provider-specific values can pass through without SDK validation, preserving compatibility with OpenAI-compatible extensions.

## References

- [Architecture §1](../architecture.md#1-design-scope)
