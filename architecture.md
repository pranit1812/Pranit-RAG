# Construction RAG – Detailed Architecture

This document explains the end-to-end architecture, key components, and runtime flow of the Construction RAG system.

## High-Level Overview
- Ingestion: Multi-extractor pipeline (NativePDF, Docling, Unstructured, Office) with routing, timeouts, and memory guardrails.
- Chunking: Layout-aware processing preserving tables/lists; token-aware chunk sizes (500 target / 900 max).
- Indexing: Dense embeddings + vector store (Chroma); sparse BM25 (Whoosh) for hybrid retrieval.
- Retrieval: Hybrid dense+sparse with optional reranker and sliding window.
- QA Assembly: Context builder → LLM answer generation → citation assembly.
- Vision: Optional page-image attachments and vision analysis for answers (OpenAI Vision).
- UI: Streamlit app for project management, upload/processing, search/QA, and project context.

## Modules and Responsibilities

### 1) Configuration (`src/config.py`)
- Central config with validation (timeouts, memory, extractor order, chunking caps, retrieval params, vision toggles).

### 2) Extraction (`src/extractors/`)
- `extraction_router.py`:
  - Initializes available extractors.
  - Uses `config.extract.pipeline_priority` and size-aware gating.
  - Per-attempt timeout and memory monitoring.
  - Quality scoring; accepts first result above threshold; fallback to best attempt or fast text-only.
- Extractors:
  - `native_pdf`: Fast, table-capable baseline; preferred for PDFs.
  - `docling`: High-fidelity but heavier; gated for smaller PDFs only.
  - `unstructured_hi_res`: Requires Poppler; used when available.
  - `docx`, `xlsx` for Office files.

### 3) Chunking (`src/chunking/`)
- Layout-aware segmentation of blocks (paragraphs, tables, lists, drawings).
- Tokenization policy: 500 target tokens, 900 hard cap.
- Normalizes `content_type` to plain strings; fixes list continuation regex.

### 4) Indexing & Search
- Dense: Embedding model defined in config; vectors persisted to Chroma per project.
- Sparse: Whoosh BM25 index; all fields coerced to strings to avoid encoding errors.
- Hybrid: Combine dense and sparse results with optional rank fusion; reranker optional.

### 5) RAG Services (`src/services/`)
- `document_processor.py`:
  - `process_uploaded_documents`: Orchestrates extraction → chunking → indexing; progress callbacks.
  - `query_project_documents`: Builds hybrid retrieval, assembles context, calls QA, returns answer+citations.
  - `generate_project_context_from_chunks`: Samples chunks → summarization via QA path; metadata normalization.
- `rag_integration.py`: Unified registry entrypoint wiring all RAG services.

### 6) Streamlit App (`app.py`)
- Session state, project management, settings, filters, chat UI.
- Project creation/selection, uploads, reprocess, processing progress, results display.
- Error handling surfaces with stack traces for debugging.

### 7) Monitoring & Logging
- Structured logs for attempts, timeouts, memory, quality, and performance.

## Data Layout (`storage/`)
- `storage/{project_id}/raw/` – uploaded originals
- `storage/{project_id}/pages/` – cached page images
- `storage/{project_id}/chroma/` – vector store
- `storage/{project_id}/chunks.jsonl` – chunk records
- `storage/projects.json` – project registry

## Request Flow (Query)
1. User enters query in Streamlit.
2. `query_project_documents` loads project stores and search params.
3. Hybrid retrieval returns candidate chunks (+ sliding window if enabled).
4. QA assembly builds prompt/context; LLM returns answer.
5. Citations are attached; optional vision augmentation adds page images/analysis.

## Performance & Safety
- Timeouts per extractor; memory guardrails with warnings.
- Size-aware extractor gating to prefer stable/fast extractors first.
- Token caps for chunking.

## Extensibility
- Add extractors by registering in router and adding to config priority.
- Swap embedding providers via `config.embeddings.provider`.
- Add rerankers and rank fusion strategies in retrieval.
- Vision provider can be swapped via config.

## External Dependencies
- Poppler (Windows) for Unstructured PDF path.
- OpenAI APIs for embeddings/LLM/Vision when enabled.
