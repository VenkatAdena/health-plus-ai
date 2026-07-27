"""Batch orchestration tests use fakes, so no model or database is loaded."""

from pathlib import Path

from healthplus.core.exceptions import DocumentLoadError
from healthplus.application import IngestionReport, KnowledgeBaseService


def test_batch_continues_after_a_failed_document() -> None:
    service = object.__new__(KnowledgeBaseService)

    def ingest(path: Path) -> IngestionReport:
        if path.name == "broken.pdf":
            raise DocumentLoadError("cannot parse")
        return IngestionReport(
            source=path.name,
            category="pricing",
            doc_id="abc",
            pages=1,
            chunks=1,
            duration_seconds=0.01,
        )

    service.ingest_pdf = ingest  # type: ignore[method-assign]
    report = service.ingest_many([Path("Pricing.pdf"), Path("broken.pdf")])

    assert [item.source for item in report.succeeded] == ["Pricing.pdf"]
    assert report.failed[0].source == "broken.pdf"
    assert report.failed[0].error == "cannot parse"
    assert not report.is_successful
