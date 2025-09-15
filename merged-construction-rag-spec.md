# Local RAG for Construction ITB Packages — Consolidated Spec (Streamlit + OpenAI)

**What we’re building (short intro):**  
A local-first Retrieval‑Augmented Generation (RAG) app that lets a subcontractor load mixed construction bid packages (drawings, specs, ITB) and ask natural‑language questions. The system prioritizes **extraction fidelity** (complex tables + drawings), **layout‑aware chunking**, **hybrid retrieval** (semantic + keyword + re‑rank), and **traceable citations**. UI stays simple: projects, upload, minimal settings, chat, and filters.

---

## 1) Scope & Goals

**Goals**
- High‑accuracy Q&A over large, mixed ITB packages (native + scanned; 40–5000+ pages).
- Robust parsing for **specs, drawings, and complex tables** (HTML/CSV), with sheet/section metadata.
- Hybrid retrieval (dense + BM25 + optional cross‑encoder re‑rank) with user filters.
- Deterministic, layout‑aware chunking; transparent **citations with page/sheet**.

**Non‑Goals**
- No multi‑tenant auth, RBAC, or production SRE.
- No model picker; **LLM = OpenAI** only (name via config).
- No Microsoft GraphRAG; indexing must remain fast enough for local use.

---

## 2) Configuration (YAML)

`config.yaml`
```yaml
app:
  data_dir: ./storage
  project_cache_size: 3
  max_upload_mb: 100000   # also mirrored in .streamlit/config.toml

llm:
  chat_model: gpt-4o
  embed_model: text-embedding-3-large
  vision_assist: false

embeddings:
  provider: openai              # openai | local
  local_model: all-MiniLM-L12-v2
  batch_size: 64

extract:
  pipeline_priority:            # escalate until page is adequately parsed
    - docling
    - unstructured_hi_res
    - native_pdf
    - ocr_ppstructure
  languages: ["en"]
  ocr:
    engine: paddleocr           # paddleocr | doctr
    ppstructure_model: TableMaster   # alt: SLANet_en
    min_conf: 0.5
  unstructured:
    strategy: hi_res
    include_page_breaks: true
  docling:
    to_html: true
    keep_vector_graphics: true

chunk:
  target_tokens: 500
  max_tokens: 900
  preserve:
    tables: true
    lists: true
  drawing:
    cluster_text: true
    max_regions: 8

index:
  vectordb: chroma
  persist_dir: ./storage/{project}/chroma

retrieve:
  top_k: 5
  hybrid: true                 # dense + BM25
  reranker: none               # none | cross_encoder
  filters:
    content_types: []          # SpecSection | Drawing | ITB | Table | List
    divisions: []              # ["03","09","26",...]
    disciplines: []            # ["A","S","M","E","P","FP","EL"]
```

`.streamlit/config.toml`
```toml
[server]
maxUploadSize = 100000
```

`.env`
```
OPENAI_API_KEY=sk-...
```

---

## 3) Architecture Overview

**Local pipeline**: Upload → Parse/OCR (Docling/Unstructured/PyMuPDF → PaddleOCR+PP-Structure) → Normalize → Layout‑aware Chunking (headings/sections, tables, lists, drawings) → Embeddings → Chroma (with metadata) → Hybrid Retrieval (dense + BM25 + optional re‑rank) → LLM Answering (citations) → Streamlit UI (chat, filters, source viewer).  
All extraction and retrieval remain local; only embeddings/LLM calls may hit OpenAI (unless `embeddings.provider=local`).

---

## 4) Data Layout

```
storage/<project>/
  raw/             # uploaded files
  pages/           # cached page images (png) + OCR JSON
  chunks.jsonl     # serialized chunks (debug/export)
  chroma/          # Chroma DB persistence
```

---

## 5) Extraction Subsystem

### 5.1 Common Output (per page)
All providers return a normalized `PageParse`:
```python
PageParse = {
  "page_no": int,
  "width": int, "height": int,
  "blocks": [  # ordered reading flow
    {
      "type": "paragraph|heading|table|list|figure|caption|artifact|titleblock|drawing",
      "text": "…",                       # UTF-8 plain text
      "html": "<table>…</table>",        # for tables (and only when available)
      "bbox": [x0,y0,x1,y1],             # page coordinates
      "spans": [{"text":"…","bbox":[…],"rot":0,"conf":0.94}, ...],
      "meta": {"cols":4,"rows":20,"rot":0,"sheet_num":"A-101","sheet_title":"..."}
    }
  ],
  "artifacts_removed": ["header","footer"]  # if detected/stripped
}
```

### 5.2 Provider Priority (escalation)
1) **Docling** → advanced layout parsing to **HTML blocks** (multi‑column, tables, lists) for PDFs/DOCX/XLSX.  
2) **Unstructured (hi_res)** → ML partitioning into semantic elements; integrates OCR for image pages.  
3) **Native PDF** → PyMuPDF (glyphs/text) + pdfplumber (table hints, line detection) for digital PDFs.  
4) **OCR (PP‑Structure)** → PaddleOCR + **Table Structure Recognition** (TableMaster/SLANet) to emit **HTML tables** + tokens (with bbox/conf) for scanned pages, drawings, images.

### 5.3 Table Policy (strict)
- Prefer **HTML** for tables; also emit **CSV** alongside for search/debug.
- Digital PDFs: try pdfplumber tables; fallback to Camelot (lattice/stream); stabilize with glyph positions if lines missing.
- Scanned: PaddleOCR **TSR** → HTML (cell grid); if grid weak, reconstruct columns/rows via bbox clustering along X/Y axes.
- When splitting long tables: **repeat header row** and **carry caption/intro** into each split.
- Detect **merged cells**, ensure semantic row grouping, and **unit capture** (e.g., psi, mm).

### 5.4 Drawings (Plan Sheets)
- Treat each sheet page as `type="drawing"`; detect **title block**; extract:
  - `sheet_number`, `sheet_title`, `discipline` (A/S/M/E/P/FP/EL), optional `scale`, `revision`.
- OCR tokens → **cluster by proximity** (DBSCAN) to produce regional sub‑blocks (details A/B/C, notes).
- Normalize rotations; retain **bbox per span** for potential spatial re‑rank and UI highlighting.

### 5.5 Word & Excel
- DOCX: python‑docx (headings, paragraphs, tables).  
- XLSX/CSV: pandas/openpyxl → per‑sheet to CSV + markdown table text; large sheets may be chunked by row groups.

---

## 6) Classification & Metadata

**Per chunk** (see §7) attach:
```json
{
  "project_id": "...",
  "doc_id": "...", "doc_name": "A-101_FloorPlan.pdf", "file_type": "pdf|docx|xlsx|image",
  "page_start": 12, "page_end": 12,
  "content_type": "SpecSection|Drawing|ITB|Table|List",
  "division_code": "03", "division_title": "Concrete",
  "section_code": "09 91 23", "section_title": "Interior Painting",
  "discipline": "A|S|M|E|P|FP|EL",
  "sheet_number": "A-101", "sheet_title": "First Floor Plan",
  "bbox_regions": [[x0,y0,x1,y1], ...],
  "token_count": 542,
  "low_conf": false,
  "text_hash": "sha256:…"
}
```

**Heuristics**
- `discipline`: sheet prefix (A/S/M/E/P/FP/EL).
- `division_*`/`section_*`: regex on spec headers (no hard count on divisions).
- `content_type`: provider block types + filename cues (ITB/Instructions/Bid Form).

---

## 7) Layout‑Aware Chunking

**Rules**
- Headings start chunks; heading text is **retained** in all child chunks for context.
- Keep **paragraphs** intact; if overflow, split on sentence boundary; ≤1 sentence overlap.
- **Tables** are standalone chunks; long tables split by row windows (repeat header & caption).
- **Lists** split in groups; include list title/intro in each split.
- **Drawings** default = **1 chunk/page** + optional regional sub‑chunks (A/B/C) when clusters detected.
- Target `~500 tokens`, max `~900`; never split mid‑sentence.

---

## 8) Embeddings & Vector Store

- Embeddings: **OpenAI** (default) or local SentenceTransformers (toggle via `embeddings.provider`).
- Vector DB: **Chroma** (collection per project). Store `{id,text,metadata,embedding}`.
- Persist to disk; cache embeddings by `text_hash` to avoid re‑embedding unchanged chunks.

---

## 9) Retrieval

- **Dense semantic** top‑k (config `retrieve.top_k`; default 5).
- **BM25 keyword** channel (Whoosh or simple TF‑IDF).  
- **Rank fusion** (RRF) of dense + BM25; optional **cross‑encoder** re‑ranks top 50→top k (config `retrieve.reranker`).
- **Metadata filters**: content_type, division_code, discipline, sheet_number, doc_name.
- De‑duplicate by section; diversify sources across documents/sections if near‑duplicate chunks bubble up.

---

## 10) QA Assembly (Citations, not prompts)

- Build **context packet** from top‑k chunk texts (+ minimal meta).
- Enforce **token budget**; trim lowest‑score chunks first.
- Return **Markdown** answer with **sources** list mapping `S# → doc_name + page/sheet`.
- Provide expandable **retrieved snippets** (raw chunk text/HTML) for verification.

*(No prompt examples in spec; only behavior.)*

---

## 11) Vision‑Assist (Optional)

- If enabled and top hit is `Drawing|Table|low_conf=true`:
  - Render PDF page → PNG (2× scale) or take original image.
  - Send image + minimal text context to OpenAI Vision model in same call.
  - Use only if it improves explicit numeric/label extraction; otherwise keep text‑only result.
- Modular toggle (`llm.vision_assist`).

---

## 12) Streamlit UI

**Sidebar**
- Project picker/creator; multi‑file uploader; **Process** button.
- Settings: toggles mirror `config.yaml` (chunking preserve tables/lists, hybrid, reranker on/off, k).
- Filters: multiselect for content types, divisions (code+name), disciplines.

**Main**
- Processing progress; stats (docs/pages/chunks).
- **Chat** (question box + conversation history).
- Answers with citations; “Show retrieved snippets”; **Open source page** (render image of cited page).

**Do NOT include:**
- Model picker (fixed to OpenAI).  
- “Fixed‑size chunking” option.

---

## 13) Non‑Functional Requirements

- **Local‑first**: parsing/index/search local; only embeddings/LLM external (unless local embeddings).
- **Scale**: 5k+ pages / 5–20k chunks per project.
- **Throughput targets**:  
  - Native text pages: **≥ 20 pages/min** (Docling/Unstructured/native).  
  - Scanned pages (OCR+TSR): **≥ 5 pages/min** baseline.
- **Latency**: Retrieval < **400 ms**; LLM dominates overall response.
- **Determinism**: stable chunk IDs & metadata across runs.
- **Privacy**: No doc text leaves machine except in LLM prompts/embeddings when OpenAI used.

---

## 14) Dependencies

```
streamlit
pymupdf
pdfplumber
opencv-python
paddlepaddle
paddleocr
camelot-py[cv]         # optional helper for digital tables
tabula-py              # optional helper if Java available
chromadb
openai
pandas
openpyxl
tiktoken
whoosh                 # BM25
unstructured
docling                # advanced layout parser
sentence-transformers  # optional local embeddings
```

---

## 15) Repo Layout

```
rag-construction/
  app.py
  .streamlit/config.toml
  config.yaml
  requirements.txt
  .env.example
  storage/
    <project>/raw/
    <project>/pages/
    <project>/chunks.jsonl
    <project>/chroma/
  src/
    config.py
    projects.py
    ingest/
      loader.py
      parse_pdf.py
      parse_docx.py
      parse_xlsx.py
      ocr_ppstructure.py
      unstructured_adapter.py
      docling_adapter.py
    chunking/
      chunker.py
      tables.py
      classify.py
    index/
      embed.py
      store.py
    retrieve/
      search.py        # dense + BM25 + fusion + rerank
    qa/
      assemble.py
      vision.py
    ui/
      widgets.py
      preview.py
    utils/
      hashing.py io.py pdf.py bbox.py log.py
  tests/
    unit/ e2e/ fixtures/
```

---

## 16) Module Interfaces

**`extract/base.py`**
```python
class BaseExtractor:
    def supports(self, path: str) -> bool: ...
    def parse_page(self, path: str, page_no: int) -> PageParse: ...
```

**`extract/docling_adapter.py`**
- Run Docling; map DOM → PageParse (paragraph/heading/table with HTML).

**`extract/unstructured_adapter.py`**
- `partition_pdf(strategy="hi_res")` → map Elements → blocks; includes OCR for image pages.

**`ingest/ocr_ppstructure.py`**
- PaddleOCR detector/recognizer + PP‑Structure TSR → table HTML; spans with bbox/conf.

**`chunking/chunker.py`**
```python
def chunk(page: PageParse, policy: ChunkPolicy) -> list[Chunk]
```

**`index/embed.py`**
```python
def embed_texts(texts: list[str]) -> list[list[float]]
```

**`index/store.py`**
```python
def upsert(chunks: list[Chunk]) -> None
def query(qvec, k:int, where:dict) -> list[Hit]
```

**`retrieve/search.py`**
```python
def dense_search(query: str, k:int, where:dict) -> list[Hit]
def keyword_search(query: str, where:dict) -> list[Hit]
def fuse(hits_a, hits_b, k:int) -> list[Hit]
def rerank_cross_encoder(hits: list[Hit], k:int) -> list[Hit]  # optional
```

**`qa/assemble.py`**
```python
def build_context(hits: list[Hit], max_tokens:int) -> ContextPacket
```

**`qa/vision.py`**
```python
def get_page_image(meta: dict) -> bytes
```

---

## 17) Acceptance Tests

1) **Spec table (digital)**: multi‑row header + merged cells → HTML+CSV emitted; querying value returns correct cell + citation.  
2) **Spec table (scanned)**: PaddleOCR+TSR reconstructs grid; queries match expected cells.  
3) **Drawing sheet**: parse title block; query “beam B3 size” resolves from labels with correct sheet citation; improves with vision ON.  
4) **ITB narrative**: multi‑column text parsed without headers/footers noise; answers cite the right page.  
5) **Hybrid beats dense‑only** on exact section number/code queries.  
6) **Filters** (division/discipline/content_type) scope results; retrieved snippets’ metadata reflects filters.  
7) **Throughput** meets targets (see §13).

---

## 18) Performance & Caching

- Cache per‑page outputs (`pages/*.json`), page images (PNG), and per‑chunk embeddings by `text_hash`.  
- Parallelize by file then page; batch embeddings (64).  
- Auto‑retry OCR/TSR with alternate model (`SLANet_en`) when confidence low; flag pages as `low_conf=true` for vision assist/UI review.

---

## 19) Install & Run

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY
streamlit run app.py
```
