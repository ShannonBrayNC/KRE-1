from __future__ import annotations

from uuid import uuid4

import pytest

from kre.schemas import SearchHit, SearchMode, SearchResponse
from kre.security import SearchAuthorizationContext, SecurityTrimmer


def hit(classification: str | None, labels: tuple[str, ...] = ()) -> SearchHit:
    return SearchHit(
        document_id=uuid4(),
        chunk_id=uuid4(),
        document_title="Knowledge",
        sequence=0,
        text="Governed content",
        score=1.0,
        classification=classification,
        security_labels=labels,
    )


def test_security_trimmer_enforces_clearance_and_labels() -> None:
    public = hit("public")
    confidential = hit("confidential", ("engineering",))
    response = SearchResponse(
        query="knowledge",
        mode=SearchMode.HYBRID,
        count=2,
        results=(public, confidential),
    )

    trimmed = SecurityTrimmer().trim(
        response,
        SearchAuthorizationContext(
            clearance="confidential",
            security_labels=frozenset({"engineering"}),
        ),
    )

    assert trimmed.count == 2
    assert trimmed.results == (public, confidential)

    without_label = SecurityTrimmer().trim(
        response,
        SearchAuthorizationContext(clearance="confidential"),
    )
    assert without_label.results == (public,)


def test_security_trimmer_fails_closed_for_missing_or_unknown_classification() -> None:
    response = SearchResponse(
        query="knowledge",
        mode=SearchMode.HYBRID,
        count=2,
        results=(hit(None), hit("unknown")),
    )

    trimmed = SecurityTrimmer().trim(
        response,
        SearchAuthorizationContext(clearance="restricted"),
    )

    assert trimmed.count == 0


def test_authorization_context_rejects_unknown_clearance() -> None:
    with pytest.raises(ValueError, match="known classification"):
        SearchAuthorizationContext(clearance="top-secret")
