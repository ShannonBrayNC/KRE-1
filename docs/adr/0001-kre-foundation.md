# ADR-0001: KRE-1 Foundation

- Status: Accepted
- Date: 2026-07-18
- Sprint: KRE-001

## Context

Lantern services currently require a common way to ingest, normalize, classify, trace, and retrieve enterprise knowledge. Direct access from each product to GitHub, SharePoint, PDFs, and local documents would duplicate connector logic, weaken provenance, and create inconsistent authorization and citation behavior.

## Decision

KRE-1 will be the Lantern Platform's governed knowledge service.

The initial implementation will use:

- Python 3.11+
- FastAPI for service contracts
- Pydantic v2 for canonical models and schema generation
- Asynchronous connector interfaces
- Provenance metadata on every document
- Provider-neutral storage, search, and embedding abstractions
- GitHub Actions for linting and automated tests

The first vertical slice will establish the domain models, connector contract, service health endpoint, CI, and golden acceptance test. Later KRE-001 pull requests will add Markdown and GitHub connectors, normalization, chunking, persistence, embeddings, and hybrid search.

## Consequences

### Positive

- Lantern products consume one stable knowledge contract.
- Provenance and security metadata are mandatory rather than optional add-ons.
- Connectors and model providers can evolve independently.
- KRE-1 can later integrate with SignalForge's provenance spine and orchestration layer.

### Tradeoffs

- A separate service adds deployment and operational overhead.
- Security trimming requires source ACL preservation and identity integration.
- Search and embedding infrastructure are intentionally deferred until the canonical contracts are stable.

## Non-goals for this decision

- Selecting the production vector database
- Implementing a full ontology or knowledge graph
- Multi-agent orchestration
- Product-specific user interfaces
