# Pending Fixes and Action Items

This document tracks remaining issues, their impact, and proposed resolutions.

## High Priority
- Poppler installation on Windows for Unstructured.io
  - Impact: Unstructured PDF extractor fails with “Unable to get page count.”
  - Resolution:
    - Option A (Scoop):
      - Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
      - iwr -useb get.scoop.sh | iex
      - scoop install poppler
      - $env:PATH += ";$((scoop prefix poppler))\bin"
    - Option B (ZIP):
      - Download ZIP from GitHub (oschwartz poppler-windows release)
      - Expand to C:\poppler and add C:\poppler\<version>\Library\bin to PATH

- Enforce extractor order strictly at runtime
  - Impact: Docling attempted first on large PDFs can cause timeouts and memory spikes.
  - Resolution: Ensure `config.extract.pipeline_priority` begins with `native_pdf` and is respected by the router; keep size-based gating (≥60MB → native first; ≥150MB → skip heavy extractors).

## Medium Priority
- Reduce Docling timeout for small PDFs only
  - Impact: Stalls when Docling is tried on borderline cases.
  - Resolution: Conditional timeout (e.g., 120s when file < 20MB), keep 300s global for exceptional cases.

- Performance tuning for memory spikes
  - Impact: Occasional memory usage > 8GB on very large PDFs.
  - Resolution: Lower concurrency within extractors if supported; optionally chunk-per-page parsing in Docling; add backoff when crossing thresholds.

- Vision integration smoke tests
  - Impact: Vision toggle wired; needs end-to-end verification against sample pages.
  - Resolution: Add minimal test to attach 1–2 page images and verify API path.

## Low Priority
- Streamlit UX polish
  - Impact: Inconsistent visibility of process buttons in some states.
  - Resolution: Further simplify conditions around `render_existing_files_processing()` and upload flow to ensure buttons are always discoverable.

- Repository housekeeping
  - Impact: Minor unused imports and comments.
  - Resolution: Run a pass with linters/formatters and prune unused modules.

## Notes
- Chunking policy validated: avg ~624 tokens; max 900; zero overflow.
- Native PDF extractor is stable and should remain first in priority.
