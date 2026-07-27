"""Preview every document-processing stage without writing to ChromaDB.

Usage:
    python scripts/preview_pipeline.py data/knowledge_base/pdfs/Pricing.pdf
    python scripts/preview_pipeline.py data/knowledge_base/pdfs/Doctors.pdf --page 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import fitz
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from healthplus.config import get_settings
from healthplus.knowledge_base.models import (
    Document,
    DocumentCategory,
    DocumentMetadata,
    PageContent,
)
from healthplus.document_pipeline import DocumentChunker, TextCleaner
from healthplus.vector_database import EmbeddingService


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}\n… [{len(text) - limit} more characters]"


def build_preview(
    path: Path, page_number: int, include_embedding: bool = True
) -> dict[str, object]:
    """Build a serializable preview using the production cleaning/chunking logic."""
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    file_bytes = path.read_bytes()
    doc_id = hashlib.sha256(file_bytes).hexdigest()[:16]
    category = DocumentCategory.from_filename(path.name)
    cleaner = TextCleaner()

    with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
        if page_number > len(pdf):
            raise ValueError(f"Page {page_number} does not exist; PDF has {len(pdf)} pages")
        raw_pages = [page.get_text() for page in pdf]
        title = (pdf.metadata or {}).get("title") or None

    cleaned_pages = [cleaner.clean(text) for text in raw_pages]
    document = Document(
        doc_id=doc_id,
        metadata=DocumentMetadata(
            source=path.name,
            category=category,
            title=title,
            page_count=len(cleaned_pages),
        ),
        pages=[
            PageContent(page_number=index + 1, text=text)
            for index, text in enumerate(cleaned_pages)
        ],
    )

    settings = get_settings()
    chunks = DocumentChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    ).chunk(document)
    page_chunks = [chunk for chunk in chunks if chunk.page_number == page_number]

    vectors: list[list[float]] = []
    if include_embedding and page_chunks:
        vectors = EmbeddingService(settings.embedding_model).embed_texts(
            [chunk.text for chunk in page_chunks]
        )

    chunk_records = []
    for index, chunk in enumerate(page_chunks):
        vector = vectors[index] if vectors else []
        chunk_records.append(
            {
                "id": chunk.chunk_id,
                "document": chunk.text,
                "metadata": {
                    "doc_id": chunk.doc_id,
                    "source": chunk.source,
                    "category": chunk.category.value,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                },
                "embedding": {
                    "model": settings.embedding_model,
                    "dimensions": len(vector),
                    "first_8_values": [round(value, 6) for value in vector[:8]],
                    "normalized": True if vector else None,
                },
            }
        )

    return {
        "pipeline": [
            "PDF input",
            "page extraction",
            "text cleaning",
            "overlapping chunking",
            "embedding generation",
            "ChromaDB-ready record",
        ],
        "input": {
            "source": path.name,
            "category": category.value,
            "doc_id": doc_id,
            "total_pages": len(raw_pages),
            "selected_page": page_number,
            "file_size_bytes": len(file_bytes),
        },
        "raw_text": raw_pages[page_number - 1],
        "cleaned_text": cleaned_pages[page_number - 1],
        "cleaning_changes": {
            "raw_characters": len(raw_pages[page_number - 1]),
            "cleaned_characters": len(cleaned_pages[page_number - 1]),
            "characters_removed": len(raw_pages[page_number - 1])
            - len(cleaned_pages[page_number - 1]),
        },
        "chunking": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "page_chunks": len(page_chunks),
            "document_chunks": len(chunks),
        },
        "records": chunk_records,
        "database_write_performed": False,
    }


def render(preview: dict[str, object], text_limit: int, max_chunks: int) -> None:
    console = Console()
    source = preview["input"]
    assert isinstance(source, dict)
    records = preview["records"]
    assert isinstance(records, list)

    console.rule("[bold cyan]DOCUMENT PROCESSING PIPELINE")
    console.print(
        "[bold]PDF INPUT[/bold]  →  [bold]EXTRACT[/bold]  →  [bold]CLEAN[/bold]  →  "
        "[bold]CHUNK[/bold]  →  [bold]EMBED[/bold]  →  [bold]CHROMADB RECORD[/bold]\n"
    )
    input_table = Table(title="1. PDF Input")
    input_table.add_column("Source")
    input_table.add_column("Category")
    input_table.add_column("Document ID")
    input_table.add_column("Pages", justify="right")
    input_table.add_row(
        str(source["source"]),
        str(source["category"]),
        str(source["doc_id"]),
        str(source["total_pages"]),
    )
    console.print(input_table)
    console.print(Panel(_clip(str(preview["raw_text"]), text_limit), title="2. Raw Extracted Text"))
    console.print(Panel(_clip(str(preview["cleaned_text"]), text_limit), title="3. Cleaned Text", border_style="green"))

    chunking = preview["chunking"]
    assert isinstance(chunking, dict)
    console.print(
        f"[bold]4. Chunking[/bold] — size={chunking['chunk_size']}, "
        f"overlap={chunking['chunk_overlap']}, selected-page chunks={chunking['page_chunks']}"
    )
    for record in records[:max_chunks]:
        assert isinstance(record, dict)
        metadata = record["metadata"]
        embedding = record["embedding"]
        assert isinstance(metadata, dict) and isinstance(embedding, dict)
        body = (
            f"[bold]Text[/bold]\n{_clip(str(record['document']), text_limit)}\n\n"
            f"[bold]Metadata[/bold]\n{json.dumps(metadata, indent=2)}\n\n"
            f"[bold]5. Embedding[/bold]\n"
            f"dimensions={embedding['dimensions']}\n"
            f"first 8 values={embedding['first_8_values']}"
        )
        console.print(Panel(body, title=f"Chunk {record['id']}", border_style="magenta"))

    console.print(
        Panel(
            "Each panel above is the exact shape sent to VectorStore.upsert():\n"
            "[bold]id + cleaned document + metadata + 384-dimensional embedding[/bold]\n\n"
            "Preview mode is read-only: no database write was performed.",
            title="6. ChromaDB-ready Output",
            border_style="cyan",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Show every PDF pipeline transformation")
    parser.add_argument("path", type=Path, help="PDF to preview")
    parser.add_argument("--page", type=int, default=1, help="1-based page to display")
    parser.add_argument("--max-chunks", type=int, default=3)
    parser.add_argument("--text-limit", type=int, default=1200)
    parser.add_argument("--no-embedding", action="store_true", help="Skip model loading")
    parser.add_argument("--output", type=Path, help="Write the complete preview as JSON")
    args = parser.parse_args()

    try:
        preview = build_preview(args.path, args.page, not args.no_embedding)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        Console(stderr=True).print(f"[red]Error:[/red] {exc}")
        return 1

    render(preview, args.text_limit, args.max_chunks)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(preview, indent=2), encoding="utf-8")
        Console().print(f"\n[green]Complete JSON saved to:[/green] {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
