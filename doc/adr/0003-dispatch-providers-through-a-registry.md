# ADR 0003: Dispatch providers through a registry

- Status: Accepted
- Date: 2026-07-22

## Context

Selecting providers with root-package conditionals couples the client to every protocol and requires central changes whenever a provider is added. Inferring a protocol from a model name is ambiguous for proxies and OpenAI-compatible services.

## Decision

Providers register a stable name and factory with the concurrency-safe `ais` registry. `NewClient` resolves the explicitly configured provider once during construction and delegates request building, response parsing, error parsing, and stream decoding to it. The default provider is `openai`; the registry never guesses from the model name.

Registration is monotonic. Empty names, nil factories, and duplicate names panic so dispatch cannot silently depend on import order. Unknown names and invalid provider options fail client construction.

## Consequences

- New protocols live in provider subpackages and do not require root dispatch changes.
- Provider configuration errors surface before a request is sent.
- Applications must import a provider package so its registration runs.
- Stable provider names become part of the public compatibility surface.

## References

- [Architecture §3.4](../architecture.md#34-registry-dispatch-and-the-provider-contract)
- [`ais/registry.go`](../../ais/registry.go)
