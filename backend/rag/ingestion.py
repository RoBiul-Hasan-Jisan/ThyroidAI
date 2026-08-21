"""
RAG ingestion pipeline.

    PDF -> extract text -> clean -> detect sections -> chunk -> metadata
        -> embed (local sentence-transformers) -> ChromaDB
        -> BM25 index -> persisted to disk

Run as:

    python -m rag.ingestion            # incremental (adds any new PDFs)
    python -m rag.ingestion --rebuild  # wipes and rebuilds the whole index

Source documents live under backend/rag/documents/{guidelines,educational,research}/
as PDFs. This script does NOT fabricate or download document content -- it
only ingests PDFs you place there yourself. See
backend/rag/documents/README.md for where to obtain legitimate, publicly
accessible thyroid-cancer reference material.
"""
import argparse
import json
import os
import re
import sys
from typing import List, Dict, Any

from rag.config import (
    DOCUMENTS_DIR,
    VECTORSTORE_DIR,
    BM25_INDEX_PATH,
    CHUNK_METADATA_PATH,
    CHUNK_TOKEN_SIZE,
    CHUNK_TOKEN_OVERLAP,
    RAG_EMBEDDING_MODEL,
)
from rag.chunking import chunk_document
from rag.vector_store import VectorStore
from rag.bm25_retriever import BM25Retriever

TOPIC_SUBDIRS = {
    "guidelines": "clinical_guideline",
    "educational": "patient_education",
    "research": "research_paper",
}


def _clean_text(text: str) -> str:
    """Light cleanup: normalize whitespace, drop obvious page-number-only lines."""
    text = text.replace("\x0c", "\n")  # form feed -> newline
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"\d{1,4}", stripped):  # bare page number
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text per page using pypdf. Never discards page numbers."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        pages.append({"page": i, "text": _clean_text(raw)})
    return pages


def find_source_pdfs() -> List[Dict[str, str]]:
    """Walk backend/rag/documents/{guidelines,educational,research}/ for PDFs."""
    found = []
    for subdir, topic in TOPIC_SUBDIRS.items():
        folder = os.path.join(DOCUMENTS_DIR, subdir)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(".pdf"):
                found.append({
                    "path": os.path.join(folder, fname),
                    "document": os.path.splitext(fname)[0],
                    "topic": topic,
                })
    return found


def run_ingestion(rebuild: bool = False) -> Dict[str, Any]:
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    source_pdfs = find_source_pdfs()
    if not source_pdfs:
        print(
            "No PDFs found under backend/rag/documents/{guidelines,educational,research}/.\n"
            "Add source PDFs there (see documents/README.md), then re-run ingestion."
        )
        return {"documents": 0, "chunks": 0}

    all_chunks: List[Dict[str, Any]] = []
    for src in source_pdfs:
        print(f"Extracting: {src['path']}")
        try:
            pages = extract_pdf_pages(src["path"])
        except Exception as e:
            print(f"  Skipped ({e})")
            continue

        # source label = filename metadata is what gets cited to the user;
        # keep it human-readable.
        source_label = src["document"].replace("_", " ").replace("-", " ")

        chunks = chunk_document(
            pages,
            document_name=src["document"],
            source=source_label,
            topic=src["topic"],
            chunk_size=CHUNK_TOKEN_SIZE,
            overlap=CHUNK_TOKEN_OVERLAP,
        )
        print(f"  {len(pages)} pages -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No text could be extracted from the available PDFs.")
        return {"documents": len(source_pdfs), "chunks": 0}

    # Lazy import: embeddings pulls in sentence-transformers/torch, which is
    # heavy -- only pay for it when actually ingesting.
    from rag.embeddings import get_embedder

    print(f"\nEmbedding {len(all_chunks)} chunks with {RAG_EMBEDDING_MODEL} ...")
    embedder = get_embedder()
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.embed_documents(texts)

    print("Writing to ChromaDB ...")
    store = VectorStore()
    store.create_collection(reset=rebuild)
    store.add_documents(all_chunks, embeddings)

    print("Building BM25 index ...")
    bm25 = BM25Retriever()
    bm25.build(all_chunks)
    bm25.save(BM25_INDEX_PATH)

    with open(CHUNK_METADATA_PATH, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    stats = store.get_stats()
    print("\nRAG ingestion complete\n")
    print(f"Documents: {len(source_pdfs)}")
    print(f"Chunks: {stats['num_chunks']}\n")
    print(f"Embedding:\n{RAG_EMBEDDING_MODEL}\n")
    print("Vector DB:\nChromaDB\n")
    print("Retrieval:\nBM25 + Vector")

    return {"documents": len(source_pdfs), "chunks": stats["num_chunks"]}


def main():
    parser = argparse.ArgumentParser(description="Ingest thyroid-cancer reference PDFs into the RAG index.")
    parser.add_argument("--rebuild", action="store_true", help="Wipe and rebuild the vector store + BM25 index from scratch.")
    args = parser.parse_args()

    try:
        run_ingestion(rebuild=args.rebuild)
    except Exception as e:
        print(f"Ingestion failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
