"""
chunker.py — Step 1 of the RAG pipeline

Reads .md files from a folder, splits them into retrievable chunks
using markdown header boundaries. Each chunk carries metadata so the
retriever can filter and the user can trace answers back to source notes.

No frameworks used — pure Python + regex.
"""
import logging
import re
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
MAX_CHUNK_CHARS = 1000  # sections larger than this get sub-split
OVERLAP_CHARS = 100  # overlap between sub-chunks to avoid losing context at boundaries


# ---------------------------------------------------------------------------
# Data shape — one chunk
# ---------------------------------------------------------------------------

def make_chunk(
        content: str,
        source_file: str,
        folder: str,
        header_path: str,
        chunk_index: int,
) -> dict:
    """
    Every chunk is a plain dict — simple to inspect, easy to pass around.
    chunk_id is deterministic (no randomness) so re-ingesting the same notes
    produces the same IDs, enabling clean upserts into Qdrant later.
    """
    # Build a stable ID from source + header + index rather than random UUID
    raw_id = f"{source_file}__{header_path}__{chunk_index}"
    chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))

    return {
        "chunk_id": chunk_id,
        "content": content.strip(),
        "source_file": source_file,  # e.g. "Equity Funds.md"
        "folder": folder,  # e.g. "Mutual Fund Notes"
        "header_path": header_path,  # e.g. "Equity Funds > Types > Large Cap"
        "char_count": len(content.strip()),
    }


# ---------------------------------------------------------------------------
# Table detection
# ---------------------------------------------------------------------------

def contains_table(text: str) -> bool:
    """
    Markdown tables have lines with | characters and a separator row (|---|).
    If a section contains a table, we never sub-split it — a table row
    without its header row is meaningless to an LLM.
    """
    lines = text.splitlines()
    has_pipe_row = any("|" in line for line in lines)
    has_separator_row = any(re.match(r"^\s*\|?[\s\-|:]+\|", line) for line in lines)
    return has_pipe_row and has_separator_row


# ---------------------------------------------------------------------------
# Sub-splitting — only for large sections without tables
# ---------------------------------------------------------------------------

def sub_split(text: str, source_file: str, folder: str, header_path: str, start_index: int) -> list[dict]:
    """
    When a header section exceeds MAX_CHUNK_CHARS, split it further.

    Split preference order:
      1. On body --- separators (Obsidian visual dividers)
      2. On double newlines (paragraph breaks)
      3. Hard cut at MAX_CHUNK_CHARS if nothing else works

    Overlap of OVERLAP_CHARS is appended from the previous chunk's tail
    to avoid losing context at split boundaries.
    """
    chunks = []
    index = start_index

    # Try splitting on --- separators first (body-level, not frontmatter)
    # A valid body --- is a line that is only dashes (3+), not part of a table
    parts = re.split(r"\n---+\n", text)

    # If --- didn't help (only one part), fall back to paragraph breaks
    if len(parts) == 1:
        parts = re.split(r"\n{2,}", text)

    # If still one big blob, hard-cut it
    if len(parts) == 1:
        raw = text
        while len(raw) > MAX_CHUNK_CHARS:
            chunks.append(make_chunk(raw[:MAX_CHUNK_CHARS], source_file, folder, header_path, index))
            raw = raw[MAX_CHUNK_CHARS - OVERLAP_CHARS:]  # back up by overlap
            index += 1
        if raw.strip():
            chunks.append(make_chunk(raw, source_file, folder, header_path, index))
        return chunks

    # Process the split parts, carrying overlap from previous part
    previous_tail = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        content = (previous_tail + "\n\n" + part).strip() if previous_tail else part
        chunks.append(make_chunk(content, source_file, folder, header_path, index))
        previous_tail = part[-OVERLAP_CHARS:] if len(part) > OVERLAP_CHARS else part
        index += 1

    return chunks


# ---------------------------------------------------------------------------
# Core: split one file into chunks
# ---------------------------------------------------------------------------

def chunk_file(filepath: Path, base_folder: Path) -> list[dict]:
    """
    Split a single .md file into chunks using header boundaries.

    Steps:
      1. Read raw text
      2. Split on any markdown header (# through ####)
      3. For each section:
           - If it has a table → emit as single chunk (never split tables)
           - If it's small enough → emit as single chunk
           - If it's too large → sub_split it
    """
    logging.debug("Chunking the notes content")
    raw_text = filepath.read_text(encoding="utf-8", errors="ignore")

    # Relative folder path for metadata (e.g. "Mutual Fund Notes")
    relative_folder = str(filepath.parent.relative_to(base_folder))
    if relative_folder == ".":
        relative_folder = base_folder.name  # root folder, use its name

    source_file = filepath.name
    note_title = filepath.stem
    chunks = []
    chunk_index = 0

    # Split on markdown headers — capture the header line itself
    # Pattern: a line starting with one or more # characters
    header_pattern = re.compile(r"^(#{1,4}\s+.+)$", re.MULTILINE)
    parts = header_pattern.split(raw_text)

    # parts layout after split with a capturing group:
    # [text_before_first_header, header1, section1_body, header2, section2_body, ...]
    # We process them in pairs: (header, body)

    # Handle text before the first header (preamble — no header)
    preamble = parts[0].strip()
    if preamble:
        chunks.append(make_chunk(
            content=f"{note_title}\n\n{preamble}",
            source_file=source_file,
            folder=relative_folder,
            header_path=filepath.stem,  # just the note title as breadcrumb
            chunk_index=chunk_index,
        ))
        chunk_index += 1

    # Walk header+body pairs
    # parts[1], parts[2] → header, body; parts[3], parts[4] → header, body; ...
    i = 1
    header_stack = []  # tracks nested headers for breadcrumb

    while i < len(parts) - 1:
        header_line = parts[i].strip()  # e.g. "## Types of Equity Funds"
        body = parts[i + 1].strip()  # text under this header
        i += 2

        # Determine header depth to maintain breadcrumb
        header_tags = re.match(r"^(#+)", header_line)
        depth = len(header_tags.group(1) if header_tags is not None else [])
        title = re.sub(r"^#+\s+", "", header_line)  # strip the # symbols

        # Trim the stack to current depth and push current header
        header_stack = header_stack[:depth - 1]
        header_stack.append(title)
        header_path = " > ".join(header_stack)

        # Full content = header line + body (so the chunk is self-contained)
        section_content = f"# {note_title}\n{header_line}\n\n{body}".strip()

        if not section_content:
            continue

        # Decision: table → keep whole; large → sub-split; normal → single chunk
        if contains_table(section_content):
            chunks.append(make_chunk(section_content, source_file, relative_folder, header_path, chunk_index))
            chunk_index += 1

        elif len(section_content) <= MAX_CHUNK_CHARS:
            chunks.append(make_chunk(section_content, source_file, relative_folder, header_path, chunk_index))
            chunk_index += 1

        else:
            sub_chunks = sub_split(section_content, source_file, relative_folder, header_path, chunk_index)
            chunks += sub_chunks
            chunk_index += len(sub_chunks)

    return chunks


# ---------------------------------------------------------------------------
# Public entry point — chunk an entire folder
# ---------------------------------------------------------------------------

def chunk_folder(notes_folder: str) -> list[dict]:
    """
    Walk a folder recursively, chunk every .md file found.
    Returns a flat list of all chunks across all files.
    """
    if not Path(notes_folder).is_absolute():
        base = (PROJECT_ROOT / notes_folder).resolve()
    else:
        base = Path(notes_folder)

    all_chunks = []

    md_files = sorted(base.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files in '{notes_folder}'")

    for filepath in md_files:
        file_chunks = chunk_file(filepath, base)
        all_chunks += file_chunks
        print(f"  {filepath.name:<40} → {len(file_chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks


# ---------------------------------------------------------------------------
# Quick self-test — run this file directly to see chunks from a folder
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    folder = sys.argv[1] if len(sys.argv) > 1 else "data/notes"
    chunks = chunk_folder(folder)

    # Print first 3 chunks as a sanity check
    print("\n--- Sample chunks (first 3) ---\n")
    for chunk in chunks[:3]:
        print(json.dumps(chunk, indent=2))
        print()
