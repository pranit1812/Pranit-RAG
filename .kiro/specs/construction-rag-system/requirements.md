# Requirements Document

## Introduction

This feature implements a local-first Retrieval-Augmented Generation (RAG) application specifically designed for construction subcontractors to process and query mixed construction bid packages. The system will handle drawings, specifications, and ITB documents with high extraction fidelity, layout-aware chunking, hybrid retrieval capabilities, and traceable citations through a simple Streamlit interface.

## Requirements

### Requirement 1

**User Story:** As a construction subcontractor, I want to upload mixed construction bid packages (PDFs, DOCX, XLSX, images) so that I can query them with natural language questions.

#### Acceptance Criteria

1. WHEN a user uploads files through the Streamlit interface THEN the system SHALL accept PDF, DOCX, XLSX, and image files up to 100GB total
2. WHEN files are uploaded THEN the system SHALL store them in the project's raw directory with original filenames preserved
3. WHEN the upload process begins THEN the system SHALL display progress indicators for file processing
4. IF a file format is unsupported THEN the system SHALL display a clear error message and continue processing other files

### Requirement 2

**User Story:** As a user, I want the system to accurately extract content from complex construction documents including tables, drawings, and specifications so that I can get reliable answers to my queries.

#### Acceptance Criteria

1. WHEN processing digital PDFs THEN the system SHALL use Docling as the primary extraction method with fallback to Unstructured hi-res and native PDF parsing
2. WHEN processing scanned documents THEN the system SHALL use OCR with PaddleOCR and PP-Structure for table structure recognition
3. WHEN extracting tables THEN the system SHALL preserve table structure in HTML format and generate CSV for search capabilities
4. WHEN processing drawing sheets THEN the system SHALL extract title block information including sheet number, title, discipline, and revision
5. WHEN extraction confidence is low THEN the system SHALL flag the content as low-confidence for potential review

### Requirement 3

**User Story:** As a user, I want the system to chunk documents intelligently while preserving layout and context so that my queries return relevant and complete information.

#### Acceptance Criteria

1. WHEN chunking documents THEN the system SHALL target 500 tokens per chunk with a maximum of 900 tokens with user-configurable chunking parameters
2. WHEN encountering headings THEN the system SHALL start new chunks and retain heading context in child chunks
3. WHEN processing tables THEN the system SHALL keep tables as standalone chunks and repeat headers when splitting long tables
4. WHEN processing lists THEN the system SHALL preserve list structure and include list titles in each chunk
5. WHEN processing drawings THEN the system SHALL create one chunk per page with optional regional sub-chunks for clustered text
6. WHEN retrieving chunks THEN the system SHALL optionally include adjacent chunks using a sliding window approach for additional context

### Requirement 4

**User Story:** As a user, I want to search through my documents using natural language queries with hybrid retrieval capabilities so that I can find relevant information quickly and accurately.

#### Acceptance Criteria

1. WHEN performing a search THEN the system SHALL use both dense semantic search and BM25 keyword search
2. WHEN combining search results THEN the system SHALL use rank fusion to merge dense and BM25 results
3. WHEN reranker is enabled THEN the system SHALL optionally apply cross-encoder reranking to improve result quality
4. WHEN returning results THEN the system SHALL provide configurable top-k results (default 5)
5. WHEN multiple similar chunks exist THEN the system SHALL deduplicate and diversify sources across documents

### Requirement 5

**User Story:** As a user, I want to filter my search results by content type, division, and discipline so that I can focus on specific aspects of my construction documents.

#### Acceptance Criteria

1. WHEN applying filters THEN the system SHALL support filtering by content types (SpecSection, Drawing, ITB, Table, List)
2. WHEN applying filters THEN the system SHALL support filtering by division codes and titles
3. WHEN applying filters THEN the system SHALL support filtering by discipline codes (A, S, M, E, P, FP, EL)
4. WHEN filters are active THEN the system SHALL only return results matching the selected criteria
5. WHEN no filters are selected THEN the system SHALL search across all content types and categories

### Requirement 6

**User Story:** As a user, I want to receive answers with clear citations and source references so that I can verify the information and trace it back to the original documents.

#### Acceptance Criteria

1. WHEN generating answers THEN the system SHALL provide markdown-formatted responses with numbered source citations
2. WHEN citing sources THEN the system SHALL include document name, page number, and sheet number where applicable
3. WHEN displaying results THEN the system SHALL provide expandable retrieved snippets showing the raw chunk text
4. WHEN a user clicks on a source THEN the system SHALL display the original page image for verification
5. WHEN multiple sources support an answer THEN the system SHALL list all relevant sources with clear numbering

### Requirement 7

**User Story:** As a user, I want to manage multiple projects independently so that I can organize different construction jobs separately.

#### Acceptance Criteria

1. WHEN creating a new project THEN the system SHALL create an isolated storage directory with separate raw, pages, and index folders
2. WHEN switching between projects THEN the system SHALL maintain separate document collections and search indices
3. WHEN listing projects THEN the system SHALL display project names with document and chunk counts
4. WHEN a project is selected THEN the system SHALL load only that project's documents and maintain separate chat history
5. WHEN the system starts THEN the system SHALL cache up to 3 projects for quick switching

### Requirement 8

**User Story:** As a user, I want to run this system locally as a proof-of-concept application so that I can test RAG capabilities on my construction documents without complex deployment requirements.

#### Acceptance Criteria

1. WHEN setting up the system THEN the system SHALL run locally with simple Python virtual environment setup
2. WHEN processing documents THEN the system SHALL perform all extraction, chunking, and indexing operations on the local machine
3. WHEN using OpenAI services THEN the system SHALL only send embeddings and LLM queries via API calls
4. WHEN local embeddings are enabled THEN the system SHALL use SentenceTransformers for offline operation
5. WHEN storing data THEN the system SHALL persist all project data in local file system directories without requiring databases or containers

### Requirement 9

**User Story:** As a user, I want the system to process documents efficiently and provide responsive search capabilities so that I can work productively with my construction documents.

#### Acceptance Criteria

1. WHEN processing multiple files THEN the system SHALL parallelize extraction by file and page to optimize performance
2. WHEN embedding text THEN the system SHALL batch process chunks to improve efficiency
3. WHEN performing operations THEN the system SHALL display progress indicators to keep users informed
4. WHEN processing large document sets THEN the system SHALL provide cancellation options for long-running operations
5. WHEN system resources are limited THEN the system SHALL gracefully handle memory constraints and provide appropriate feedback

### Requirement 10

**User Story:** As a user, I want optional vision assistance during query time so that I can get enhanced answers that incorporate visual analysis of relevant document pages.

#### Acceptance Criteria

1. WHEN vision assist is enabled THEN the system SHALL send page images of the top 1-5 retrieved chunks to OpenAI Vision along with the query
2. WHEN using vision assist THEN the system SHALL allow users to configure how many chunk images to include (1-5 chunks)
3. WHEN vision assist is enabled THEN the system SHALL always incorporate visual analysis into the final answer regardless of chunk type or confidence
4. WHEN using vision assist THEN the system SHALL render PDF pages at high resolution for better image quality
5. WHEN vision assist is disabled THEN the system SHALL rely only on text-based retrieval and answering
6. WHEN vision assist fails THEN the system SHALL gracefully fall back to text-only results with appropriate user notification

### Requirement 11

**User Story:** As a user, I want the system to be configurable and handle errors gracefully so that I can customize behavior and recover from processing failures.

#### Acceptance Criteria

1. WHEN the system starts THEN the system SHALL load configuration from YAML files for all major system parameters
2. WHEN embeddings are generated THEN the system SHALL cache them by text hash to avoid reprocessing unchanged content
3. WHEN errors occur during processing THEN the system SHALL log detailed error information and continue processing other files
4. WHEN system settings change THEN the system SHALL allow users to modify key parameters through the UI that mirror the configuration file

### Requirement 12

**User Story:** As a user, I want the system to understand my project context so that queries are more relevant and accurate for my specific construction project.

#### Acceptance Criteria

1. WHEN documents are first processed THEN the system SHALL automatically generate initial project context including project type, key systems, and disciplines
2. WHEN project context is available THEN the system SHALL use it to enhance user queries with relevant construction terminology and domain knowledge
3. WHEN displaying project information THEN the system SHALL allow users to view and edit the project context through the UI
4. WHEN performing searches THEN the system SHALL use project context to improve semantic search relevance and disambiguate technical terms
5. WHEN project context is updated THEN the system SHALL apply the changes to subsequent queries without requiring re-indexing