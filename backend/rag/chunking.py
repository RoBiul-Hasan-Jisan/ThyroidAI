"""
Section-aware chunking for the RAG ingestion pipeline.

We avoid pulling in a tokenizer dependency purely for chunk sizing: token
counts are approximated as `len(text.split())` (whitespace tokens), which is
close enough to real subword-token counts for chunk-size bookkeeping and
keeps the pipeline dependency-light (per the "no unnecessary frameworks"
requirement). Target sizes are configurable via rag/config.py.
"""
import re
from typing import List, Dict, Any

from rag.config import CHUNK_TOKEN_SIZE, CHUNK_TOKEN_OVERLAP

# Headings like "TNM STAGING", "Response to Therapy:", "3. Follow-up" etc.
_SECTION_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\s+)?"          # optional numbering: "3.2 "
    r"([A-Z][A-Za-z0-9 ,/\-()]{2,80})" # heading text
    r"\s*:?\s*$"
)


def _count_tokens(text: str) -> int:
    return len(text.split())


def detect_sections(page_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Given a list of {"page": int, "text": str} entries (one per PDF page, in
    order), split the concatenated document into sections by looking for
    short, capitalized, standalone lines that look like headings.

    Returns a list of {"section": str, "page": int, "text": str} blocks,
    where `page` is the page the block started on. Falls back to a single
    "Document" section if no headings are detected.
    """
    blocks: List[Dict[str, Any]] = []
    current_section = "Document"
    current_lines: List[str] = []
    current_page = page_texts[0]["page"] if page_texts else 1

    def flush():
        text = "\n".join(current_lines).strip()
        if text:
            blocks.append({"section": current_section, "page": current_page, "text": text})

    for page_entry in page_texts:
        page_num = page_entry["page"]
        for raw_line in page_entry["text"].splitlines():
            line = raw_line.strip()
            if not line:
                current_lines.append("")
                continue

            is_heading = (
                len(line) <= 80
                and not line.endswith(".")
                and _SECTION_HEADING_RE.match(line) is not None
                and _count_tokens(line) <= 10
            )
            if is_heading:
                flush()
                current_section = line.rstrip(":").strip()
                current_lines = []
                current_page = page_num
            else:
                if not current_lines:
                    current_page = page_num
                current_lines.append(raw_line)

    flush()
    return blocks


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_TOKEN_SIZE,
    overlap: int = CHUNK_TOKEN_OVERLAP,
) -> List[str]:
    """
    Split `text` into overlapping chunks of approximately `chunk_size`
    whitespace-tokens, with `overlap` tokens shared between consecutive
    chunks. Splits on paragraph/sentence boundaries where possible so we
    don't cut a sentence in half.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    # Flatten into sentence-ish units so we can pack them into ~chunk_size chunks.
    units: List[str] = []
    for para in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", para)
        units.extend(s.strip() for s in sentences if s.strip())

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    i = 0
    while i < len(units):
        unit = units[i]
        unit_tokens = _count_tokens(unit)

        if current_tokens + unit_tokens > chunk_size and current:
            chunk_str = " ".join(current)
            chunks.append(chunk_str)

            # Build overlap: keep trailing units worth ~overlap tokens.
            overlap_units: List[str] = []
            overlap_tokens = 0
            for prev in reversed(current):
                t = _count_tokens(prev)
                if overlap_tokens + t > overlap:
                    break
                overlap_units.insert(0, prev)
                overlap_tokens += t
            current = overlap_units
            current_tokens = overlap_tokens
            continue  # re-process the same unit against the reset window

        current.append(unit)
        current_tokens += unit_tokens
        i += 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_document(
    page_texts: List[Dict[str, Any]],
    document_name: str,
    source: str,
    topic: str,
    chunk_size: int = CHUNK_TOKEN_SIZE,
    overlap: int = CHUNK_TOKEN_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Full pipeline: page texts -> section-aware blocks -> token-bounded
    overlapping chunks -> chunk dicts with full metadata (page/source/section
    are never discarded).
    """
    sections = detect_sections(page_texts)
    chunk_records: List[Dict[str, Any]] = []
    chunk_idx = 0

    for block in sections:
        pieces = chunk_text(block["text"], chunk_size=chunk_size, overlap=overlap)
        for piece in pieces:
            chunk_idx += 1
            chunk_records.append({
                "chunk_id": f"{document_name}::{chunk_idx:04d}",
                "text": piece,
                "source": source,
                "document": document_name,
                "section": block["section"],
                "page": block["page"],
                "topic": topic,
            })

    return chunk_records
