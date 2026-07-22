# ADR 0002: Use OpenAI Chat Completions as the canonical model

- Status: Accepted
- Date: 2026-07-21

## Context

The SDK needs one public request and response model across OpenAI-compatible and Anthropic APIs. A vendor-neutral model designed from scratch would add another translation path and could hide useful fields. Most supported backends already accept the OpenAI Chat Completions shape.

## Decision

Use the OpenAI Chat Completions shape as the canonical representation in package `ais`. OpenAI-compatible providers serialize it directly. Providers with different wire protocols, currently Anthropic, translate at their package boundary.

Provider-only controls that must not appear on the OpenAI wire use struct-local fields marked `json:"-"`. Common semantics are normalized in canonical fields; unsupported fields are ignored by providers that cannot represent them.

## Consequences

- The common OpenAI-compatible path has no model translation layer.
- Callers use the same types when switching providers.
- Anthropic bears a bidirectional translation cost and some semantics cannot map perfectly.
- The canonical API naturally follows OpenAI naming and structure rather than being vendor-neutral in appearance.

## References

- [Architecture §2](../architecture.md#2-canonical-representation-openai-shaped)
- [Canonical data model](../design/data-model.md)
