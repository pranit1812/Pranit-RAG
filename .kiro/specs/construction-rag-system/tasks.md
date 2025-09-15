# Implementation Plan

- [x] 1. Set up project structure and core configuration system





  - Create directory structure following the design layout (src/, storage/, tests/)
  - Implement configuration loading from YAML files with validation
  - Set up environment variable handling for API keys
  - Create base data models and type definitions for PageParse, Chunk, and metadata schemas
  - _Requirements: 11.1, 8.5_

- [x] 2. Implement core utility functions and base classes





  - [x] 2.1 Create utility modules for common operations


    - Write hashing utilities for text_hash generation using SHA-256
    - Implement bbox manipulation functions for coordinate handling
    - Create PDF utilities for page rendering and image extraction
    - Write I/O utilities for file handling and path management
    - _Requirements: 11.2_

  - [x] 2.2 Implement base extractor interface and common structures


    - Define BaseExtractor abstract class with supports() and parse_page() methods
    - Create PageParse, Block, Span, and Chunk data structures with proper typing
    - Implement ChunkMetadata class with all required fields including MasterFormat divisions
    - Write validation functions for data structure integrity
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 3. Build document extraction pipeline with multi-provider support





  - [x] 3.1 Implement native PDF extraction


    - Create NativePDFExtractor using PyMuPDF for text extraction
    - Add pdfplumber integration for table detection and structure analysis
    - Implement Camelot fallback for complex table extraction
    - Write bbox coordinate normalization and text span creation
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 3.2 Implement OCR extraction with PaddleOCR


    - Create OCRExtractor with PaddleOCR text detection and recognition
    - Integrate PP-Structure for table structure recognition (TableMaster/SLANet)
    - Implement DBSCAN clustering for drawing text region grouping
    - Add confidence scoring and low-confidence flagging
    - _Requirements: 2.2, 2.4, 2.5_

  - [x] 3.3 Implement Docling adapter


    - Create DoclingExtractor that converts Docling DOM to PageParse format
    - Map Docling elements (paragraphs, headings, tables) to Block structures
    - Preserve HTML table structure and convert to required format
    - Handle multi-column layouts and complex document structures
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 3.4 Implement Unstructured adapter


    - Create UnstructuredExtractor using hi-res strategy
    - Map Unstructured Elements to Block format with proper type classification
    - Integrate OCR capabilities for image pages
    - Handle page break detection and document structure preservation
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 3.6 Implement DOCX and XLSX extraction


    - Create DOCXExtractor using python-docx for Word document processing
    - Implement XLSXExtractor using pandas/openpyxl for Excel spreadsheet processing
    - Add per-sheet processing for Excel files with proper metadata
    - Write table extraction from both DOCX and XLSX formats
    - _Requirements: 1.1, 2.3_

  - [x] 3.5 Create extraction router with provider escalation


    - Implement extraction pipeline that tries providers in priority order
    - Add fallback logic when extraction fails or confidence is low
    - Create extraction result validation and quality assessment
    - Write comprehensive error handling and logging for extraction failures
    - _Requirements: 2.1, 11.3_

- [x] 4. Implement classification and metadata extraction system





  - [x] 4.1 Create content type classification


    - Write heuristics to classify blocks as SpecSection, Drawing, ITB, Table, List
    - Implement filename-based classification for ITB and instruction documents
    - Add provider block type mapping to content types
    - Create confidence scoring for classification decisions
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 4.2 Implement MasterFormat division and section detection


    - Create regex patterns for detecting division codes (00-48) in spec headers
    - Implement section code extraction (e.g., "09 91 23") with title parsing
    - Add division title lookup using the complete MasterFormat division list
    - Write validation for detected codes and titles
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 4.3 Create drawing metadata extraction


    - Implement title block detection and parsing for sheet information
    - Extract sheet numbers, titles, disciplines (A/S/M/E/P/FP/EL), scale, and revision
    - Add discipline classification based on sheet number prefixes
    - Create bbox region tracking for drawing elements
    - _Requirements: 2.4, 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 5. Build layout-aware chunking system





  - [x] 5.1 Implement core chunking logic


    - Create chunker that respects document structure and layout
    - Implement heading-based chunk initiation with context propagation
    - Add paragraph integrity preservation with sentence-boundary splitting
    - Write token counting using tiktoken for OpenAI compatibility and chunk size management (500-900 tokens)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 5.2 Implement specialized chunking for tables and lists


    - Create table chunking that preserves structure as standalone chunks
    - Implement header repetition for long table splits
    - Add list chunking with title/intro context preservation
    - Write HTML structure preservation for complex tables
    - Generate CSV output alongside HTML for table search capabilities
    - _Requirements: 2.3, 3.3, 3.4_

  - [x] 5.3 Implement drawing-specific chunking


    - Create page-level chunking for drawing sheets
    - Add optional regional sub-chunking based on text clustering
    - Implement bbox region tracking and spatial organization
    - Write drawing text cluster analysis using DBSCAN
    - _Requirements: 3.5, 3.6_

  - [x] 5.4 Add sliding window context support



    - Implement configurable sliding window for adjacent chunk retrieval
    - Create context expansion logic for better chunk relationships
    - Add user-configurable window size parameters
    - Write tests for context window accuracy and relevance
    - _Requirements: 3.6_

- [x] 6. Create project context system





  - [x] 6.1 Implement project context generation


    - Create automatic project context extraction from document analysis
    - Implement project type detection (commercial, residential, industrial, etc.)
    - Add key systems identification (HVAC, electrical, plumbing, structural)
    - Write discipline involvement detection from document content
    - _Requirements: 12.1, 12.2_



  - [x] 6.2 Build project context management

    - Create ProjectContext data structure with all required fields
    - Implement project context persistence to project_context.md files
    - Add project context editing and validation functionality
    - Write project context loading and caching system


    - _Requirements: 12.3, 12.5_

  - [x] 6.3 Implement query enhancement with project context


    - Create query expansion using project-specific terminology
    - Implement technical term disambiguation based on project type
    - Add construction domain knowledge integration
    - Write query enhancement functions that preserve user intent
    - _Requirements: 12.2, 12.4_

- [x] 7. Build embedding and vector store system





  - [x] 7.1 Implement embedding service with provider support


    - Create EmbeddingService interface with OpenAI and local implementations
    - Implement OpenAIEmbedding using text-embedding-3-large model
    - Write batch processing for efficient embedding generation
    - _Requirements: 8.2, 8.4, 9.2_



  - [x] 7.2 Create vector store with Chroma integration

    - Implement VectorStore interface with Chroma backend
    - Create project-specific collections with proper isolation
    - Add metadata filtering capabilities for all chunk metadata fields
    - Write persistence and caching logic for embeddings by text_hash

    - _Requirements: 8.1, 8.5, 11.2_

  - [x] 7.3 Implement chunk indexing pipeline

    - Create chunk processing pipeline from extraction to vector storage
    - Add embedding caching to avoid reprocessing unchanged content
    - Implement batch processing for large document sets
    - Write progress tracking and cancellation support for long operations
    - _Requirements: 9.1, 9.4, 11.2_

- [x] 8. Build hybrid retrieval system





  - [x] 8.1 Implement dense semantic search






    - Create dense vector search using Chroma with metadata filtering
    - Implement query embedding with project context enhancement
    - Add configurable top-k retrieval with relevance scoring
    - Write result deduplication and source diversification
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 12.4_

  - [x] 8.2 Create BM25 keyword search





    - Implement BM25Index using Whoosh for keyword-based retrieval
    - Create text preprocessing and tokenization for construction documents
    - Add metadata filtering support for keyword search
    - Write keyword search result scoring and ranking
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [x] 8.3 Implement rank fusion and reranking


    - Create Reciprocal Rank Fusion (RRF) for combining dense and BM25 results
    - Implement optional cross-encoder reranking for result refinement
    - Add result deduplication by section to diversify sources
    - Write configurable fusion parameters and reranking options
    - _Requirements: 4.2, 4.4, 4.5_

  - [x] 8.4 Create metadata filtering system


    - Implement filtering by content types (SpecSection, Drawing, ITB, Table, List)
    - Add division code filtering with MasterFormat division support
    - Create discipline filtering (A/S/M/E/P/FP/EL) for drawings
    - Write filter combination logic and validation
    - _Requirements: 5.1, 5.2, 5.3, 5.4_


- [x] 9. Implement QA assembly and citation system




  - [x] 9.1 Create context building system


    - Implement ContextPacket creation from retrieved chunks
    - Add token budget management with intelligent chunk trimming
    - Create source information extraction and organization
    - Write context optimization for LLM consumption
    - _Requirements: 6.1, 6.2, 6.3_

  - [x] 9.2 Build citation generation system


    - Create numbered source mapping (S1, S2, etc.) for citations
    - Implement document name and page/sheet reference generation
    - Add expandable snippet display for source verification
    - Write citation accuracy validation and source traceability
    - _Requirements: 6.2, 6.3, 6.4_

  - [x] 9.3 Implement LLM integration for answer generation


    - Create OpenAI API integration for chat completion
    - Implement context injection with project context and retrieved chunks
    - Add response parsing and citation extraction
    - Write error handling for API failures and rate limiting
    - _Requirements: 6.1, 6.5_

- [x] 10. Build vision assistance system





  - [x] 10.1 Implement page image rendering


    - Create PDF page to high-resolution PNG conversion (2x scale)
    - Implement image extraction for non-PDF documents
    - Add image quality optimization for vision model consumption
    - Write image caching and storage management
    - _Requirements: 10.4_

  - [x] 10.2 Create vision service integration


    - Implement OpenAI Vision API integration for image analysis
    - Create configurable image selection (1-5 top chunks)
    - Add vision context building with query and text context
    - Write vision response integration with text-based answers
    - _Requirements: 10.1, 10.2, 10.3_

  - [x] 10.3 Implement vision assist workflow


    - Create vision assist toggle and configuration in UI
    - Add automatic image inclusion for top retrieved chunks when enabled
    - Implement graceful fallback to text-only results on vision failures
    - Write vision assist result integration that always incorporates visual analysis
    - _Requirements: 10.3, 10.5, 10.6_

- [x] 11. Create project management system





  - [x] 11.1 Implement project creation and management


    - Create project directory structure initialization
    - Implement project metadata storage and retrieval
    - Add project listing with document and chunk statistics
    - Write project deletion and cleanup functionality
    - _Requirements: 7.1, 7.2, 7.3_

  - [x] 11.2 Build file upload and processing system

    - Create multi-file upload handling with progress tracking
    - Implement file validation (type, size) and error reporting
    - Add processing pipeline orchestration from upload to indexing
    - Write cancellation support for long-running processing operations
    - _Requirements: 1.1, 1.2, 1.3, 9.4_

  - [x] 11.3 Implement project switching and caching

    - Create project cache system for quick switching between projects
    - Add project state persistence and restoration
    - Implement configurable cache size (default 3 projects)
    - Write project loading optimization and memory management
    - _Requirements: 7.4, 7.5_


- [ ] 12. Build Streamlit user interface




  - [x] 12.1 Create main application structure


    - Implement Streamlit app.py with proper page configuration
    - Create sidebar layout with project management and settings
    - Add main content area with chat interface and results display
    - Write session state management for project and conversation data
    - _Requirements: 1.3, 7.4, 11.4_

  - [x] 12.2 Implement project management UI


    - Create project picker dropdown with project statistics
    - Add new project creation form with name validation
    - Implement multi-file uploader with drag-and-drop support
    - Write processing progress display with real-time updates
    - _Requirements: 1.1, 1.2, 7.1, 7.2, 7.3_

  - [x] 12.3 Build settings and configuration UI


    - Create settings panel that mirrors config.yaml parameters
    - Implement toggles for chunking options (preserve tables/lists)
    - Add hybrid search and reranker configuration controls
    - Write vision assist settings with configurable image count
    - _Requirements: 11.4, 10.2_

  - [x] 12.4 Create filtering and search interface


    - Implement content type multiselect (SpecSection, Drawing, ITB, Table, List)
    - Add division code filtering with MasterFormat division names
    - Create discipline filtering for drawings (A/S/M/E/P/FP/EL)
    - Write filter state persistence and reset functionality
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [x] 12.5 Build chat interface and results display


    - Create chat input with query submission and history display
    - Implement answer display with markdown formatting and citations
    - Add expandable retrieved snippets with source information
    - Write source page viewer with image rendering for verification
    - _Requirements: 6.1, 6.3, 6.4, 6.5_

  - [x] 12.6 Implement project context management UI


    - Create project context display and editing interface
    - Add project summary and key systems visualization
    - Implement project context editing with validation
    - Write project context auto-generation trigger and status display
    - _Requirements: 12.1, 12.3, 12.5_
- [ ] 13. Add error handling and logging system







- [ ] 13. Add error handling and logging system

  - [x] 13.1 Implement comprehensive error handling


    - Create error handling for all extraction providers with graceful degradation
    - Add API error handling for OpenAI services with retry logic
    - Implement file processing error recovery and continuation
    - Write user-friendly error messages and troubleshooting guidance
    - Implement memory management and cleanup for large file processing
    - _Requirements: 11.3, 10.6, 9.5_

  - [x] 13.2 Create logging and monitoring system


    - Implement structured logging for all major operations
    - Add performance monitoring for extraction and retrieval operations
    - Create debug logging for troubleshooting extraction and chunking issues
    - Write log rotation and storage management for long-running operations
    - _Requirements: 11.3, 9.3_

- [ ] 14. Write comprehensive tests





  - [x] 14.1 Create unit tests for core components


    - Write tests for extraction providers with sample documents ("sample docs" is the folder name)
    - Create tests for chunking logic with various document structures
    - Add tests for classification and metadata extraction accuracy
    - Write tests for embedding and vector store operations
    - _Requirements: All requirements validation_

  - [x] 14.2 Implement integration tests


    - Create end-to-end document processing pipeline tests
    - Add hybrid retrieval system tests with real document collections
    - Write vision service integration tests with sample images
    - Create project management and UI workflow tests
    - _Requirements: All requirements validation_

- [x] 15. Create deployment and documentation




  - [x] 15.1 Set up deployment configuration


    - Create requirements.txt with all necessary dependencies check if it exists
    - Write .env.example with required environment variables check if it exists
    - Create .streamlit/config.toml with upload size configuration check if it exists
    - Add config.yaml with default system configuration check if it exsits
    - Clean up all unneccessary files in the directory
    - _Requirements: 8.1, 8.5_

  - [x] 15.2 Write user documentation


    - Create installation and setup instructions
    - Write user guide for document upload and processing
    - Add troubleshooting guide for common issues
    - Create configuration reference documentation
    - _Requirements: 8.1, 8.5_