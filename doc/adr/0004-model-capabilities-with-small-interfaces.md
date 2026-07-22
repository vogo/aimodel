# ADR 0004: Model capabilities with small interfaces

- Status: Accepted
- Date: 2026-07-22

## Context

AI APIs expose different interaction forms and not every provider supports all of them. A single broad client or provider interface would force unrelated implementations to add placeholder methods and would make each new interaction form a breaking change.

## Decision

Represent interaction forms with small capability interfaces. Chat completion is exposed through `ChatCompleter`; provider-specific chat behavior implements `ais.ChatProvider`. A new interaction form gets a new capability interface and matching client method instead of widening an existing interface.

The compose layer depends on the smallest root capability it needs rather than on concrete providers.

## Consequences

- Consumers can accept narrow interfaces and test them with small fakes.
- Providers implement only capabilities they support.
- Existing capability contracts remain stable as new interaction forms are added.
- Capability discovery and composition require explicit interfaces rather than one all-purpose client contract.

## References

- [Architecture §3.4](../architecture.md#34-registry-dispatch-and-the-provider-contract)
- [`ais/provider.go`](../../ais/provider.go)
