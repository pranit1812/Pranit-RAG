# Construction RAG – Progress Summary

This document summarizes the key changes made so far, stabilizations, and current behavior across the system.

## Highlights
- Token-aware chunking enforces target/max at 500/900 tokens; no chunks exceed 900.
- Extraction router hardened with detailed logging, timeouts, and memory checks; memory-aware extractor gating added.
- Whoosh BM25 indexing fixed by coercing all fields to strings (avoids utf-8 encode errors).
- Project context generation added (chunk sampling → summarization) with safe metadata normalization.
- Streamlit UI stabilized: project creation and selection flow fixed; upload/display/process paths clarified; duplicate chat renderer removed.
- Vision toggle unified in Streamlit; RAG services wired to the consolidated registry.

## Key Code Changes

### Configuration (`src/config.py`)
- Increased `extract.timeout_seconds` to 300 and `extract.max_memory_mb` to 8192.
- Validation for minimum timeouts/memory.
- Default `extract.pipeline_priority` set to prefer `native_pdf` first.

### Extraction Router (`src/extractors/extraction_router.py`)
- Comprehensive per-attempt logging (elapsed, pages, quality).
- Timeouts and memory monitoring around extraction calls.
- Memory-aware extractor selection:
  - ≥150MB: only `native_pdf`/office extractors
  - ≥60MB: force `native_pdf` first
- Text-only fast fallback via PyMuPDF.

### BM25 Search (`src/services/bm25_search.py`)
- Coerce all indexed fields (e.g., `page_start`, `page_end`, `chunk_id`, `content_type`, etc.) to `str` to prevent `utf_8_encode` errors.

### Document Processor (`src/services/document_processor.py`)
- Added `generate_project_context_from_chunks`:
  - Normalizes metadata types; samples diverse chunks; passes synthetic hits into QA.
- Ensured hit structure contains `chunk` where required by QA.

### Chunking
- `src/chunking/chunker.py`: Guaranteed `content_type` becomes a plain string.
- `src/chunking/list_processor.py`: Fixed regex `bad character range` by escaping `-` and `*`.

### Streamlit App (`app.py`)
- Removed incorrect `utils.error_handling.handle_error` import; use local `handle_error`.
- Fixed project selector default and create-project form flow; removed duplicate `render_chat_interface` and callsite mismatch.
- Upload flow: displays selected files, persists to `raw/`, shows processing status; separate “Process Existing Documents” for already uploaded files.
- Vision toggle unified via `st.session_state.vision_enabled`.
- Improved chat UI and sources display.

### RAG Integration (`src/services/rag_integration.py`)
- Uses consolidated `services.registry.create_rag_services`.

### .gitignore
- Added Python caches, venvs, logs, storage (kept `.gitkeep`), bm25_index, images/, raw/, IDE folders, etc.

## Current System Behavior
- For large PDFs, `native_pdf` runs first and typically succeeds quickly; Docling may time out on very large/complex PDFs (guarded by size-based gating), Unstructured requires Poppler (see pending fixes).
- Chunk stats validated on sample project: avg ~624 tokens, max 900, no overflow.
- Streamlit app starts reliably; project management and processing flow functional.

## Known External Requirements
- Poppler for Unstructured.io PDF support (Windows install instructions in pending_fixes.md).


