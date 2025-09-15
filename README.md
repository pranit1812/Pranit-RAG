# Construction RAG System

A local-first Retrieval-Augmented Generation (RAG) application specifically designed for construction subcontractors to process and query mixed construction bid packages including drawings, specifications, and ITB documents.

## Features

- **Multi-format Document Support**: Process PDF, DOCX, XLSX, and image files
- **Advanced Extraction**: Multiple extraction providers with intelligent fallback
- **Layout-Aware Chunking**: Preserves document structure and context
- **Hybrid Search**: Combines semantic search with keyword matching
- **MasterFormat Integration**: Automatic division and section code detection
- **Vision Assistance**: Optional image analysis for enhanced answers
- **Project Management**: Isolated workspaces for different construction jobs
- **Traceable Citations**: Clear source references with page/sheet numbers

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (for embeddings and LLM services)
- At least 4GB RAM recommended for document processing
- 10GB+ free disk space for document storage

### Step 1: Clone and Install

```bash
# Clone the repository
git clone <repository-url>
cd construction-rag-system

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
venv\Scripts\activate.bat
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file and add your API keys
# Required:
OPENAI_API_KEY=your_openai_api_key_here

# Optional (for future features):
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Step 3: Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Quick Start Guide

### 1. Create Your First Project

1. Open the application in your browser
2. In the sidebar, click "Create New Project"
3. Enter a project name (e.g., "Office Building Renovation")
4. Click "Create Project"

### 2. Upload Documents

1. Use the file uploader in the sidebar
2. Drag and drop or browse for your construction documents
3. Supported formats: PDF, DOCX, XLSX, PNG, JPG
4. Maximum total upload size: 100GB
5. Wait for processing to complete (progress bar will show status)

### 3. Configure Settings (Optional)

In the sidebar Settings panel, you can adjust:
- **Chunking Options**: Preserve table/list structure
- **Search Settings**: Enable hybrid search and reranking
- **Vision Assist**: Enable image analysis (uses additional API calls)

### 4. Apply Filters (Optional)

Use the Search Filters to focus on specific content:
- **Content Types**: SpecSection, Drawing, ITB, Table, List
- **Divisions**: MasterFormat divisions (00-48)
- **Disciplines**: A/S/M/E/P/FP/EL for drawings

### 5. Ask Questions

1. Type your question in the chat input
2. Examples:
   - "What are the HVAC requirements for the mechanical room?"
   - "Show me the electrical specifications for lighting"
   - "What materials are specified for the exterior walls?"
3. Review the answer with numbered citations
4. Click on source references to view original document pages

## User Guide

### Project Management

**Creating Projects**
- Each project creates an isolated workspace
- Projects store documents, embeddings, and chat history separately
- Use descriptive names like "Hospital Wing Addition" or "Retail Store Buildout"

**Switching Projects**
- Use the project dropdown in the sidebar
- The system caches up to 3 projects for quick switching
- All data remains isolated between projects

**Project Context**
- The system automatically generates project context from your documents
- Edit project context to improve search relevance
- Include project type, location, key systems, and disciplines

### Document Processing

**Supported File Types**
- **PDF**: Digital and scanned construction documents
- **DOCX**: Word documents with specifications
- **XLSX**: Excel spreadsheets with schedules and data
- **Images**: PNG, JPG drawings and photos

**Processing Pipeline**
1. **Extraction**: Text, tables, and metadata extraction
2. **Classification**: Content type and division detection
3. **Chunking**: Layout-aware text segmentation
4. **Embedding**: Vector representation generation
5. **Indexing**: Storage in searchable database

**Quality Indicators**
- Low confidence content is flagged for review
- Processing errors are logged and reported
- Progress indicators show real-time status

### Search and Retrieval

**Query Types**
- **Natural Language**: "What are the fire safety requirements?"
- **Technical Terms**: "HVAC ductwork specifications"
- **Specific Sections**: "Division 26 electrical requirements"
- **Drawing References**: "Show me the floor plan details"

**Search Modes**
- **Hybrid Search** (recommended): Combines semantic and keyword search
- **Semantic Only**: Uses AI embeddings for conceptual matching
- **Keyword Only**: Traditional text matching

**Filtering Options**
- **Content Type**: Focus on specifications, drawings, or tables
- **Division Codes**: Filter by MasterFormat divisions
- **Disciplines**: Filter drawings by architectural, structural, etc.

**Results Display**
- Answers include numbered source citations (S1, S2, etc.)
- Click citations to view original document pages
- Expandable snippets show raw retrieved text
- Source information includes document name and page numbers

### Vision Assistance

**When to Use**
- Questions about drawings, diagrams, or visual content
- Complex tables that benefit from visual analysis
- Situations where text extraction may be incomplete

**Configuration**
- Enable in Settings panel
- Choose number of images to analyze (1-5)
- Higher image counts provide more context but use more API calls

**How It Works**
1. System retrieves relevant text chunks as normal
2. Renders corresponding document pages as high-resolution images
3. Sends images + query + text context to OpenAI Vision
4. Incorporates visual analysis into the final answer

## Configuration Reference

The `config.yaml` file controls system behavior. Key sections:

### Application Settings
```yaml
app:
  data_dir: ./storage          # Where projects are stored
  project_cache_size: 3        # Number of projects to keep in memory
  max_upload_mb: 100000       # Maximum total upload size
```

### Language Model Settings
```yaml
llm:
  chat_model: gpt-4o          # OpenAI model for answers
  embed_model: text-embedding-3-large  # Embedding model
  vision_assist: false        # Enable vision by default
```

### Extraction Pipeline
```yaml
extract:
  pipeline_priority:          # Extraction provider order
    - docling                 # Advanced layout parsing
    - unstructured_hi_res     # ML-based extraction
    - native_pdf             # Direct PDF parsing
    - ocr_ppstructure        # OCR fallback
```

### Chunking Behavior
```yaml
chunk:
  target_tokens: 500          # Target chunk size
  max_tokens: 900            # Maximum chunk size
  preserve:
    tables: true             # Keep tables intact
    lists: true              # Keep lists intact
```

### Retrieval Settings
```yaml
retrieve:
  top_k: 5                   # Number of results to return
  hybrid: true               # Use both semantic and keyword search
  reranker: none             # Optional reranking (none | cross_encoder)
```

## Troubleshooting

### Common Issues

**"No OpenAI API key found"**
- Ensure `.env` file exists with `OPENAI_API_KEY=your_key_here`
- Restart the application after adding the key
- Check that the key is valid and has sufficient credits

**"Document processing failed"**
- Check file format is supported (PDF, DOCX, XLSX, images)
- Ensure files are not corrupted or password-protected
- Try processing files individually to isolate issues
- Check logs in the `logs/` directory for detailed error information

**"Out of memory during processing"**
- Reduce batch size in configuration
- Process fewer files at once
- Increase system RAM or use smaller documents
- Enable memory monitoring in configuration

**"Search returns no results"**
- Check that documents have been processed successfully
- Try broader search terms
- Remove or adjust search filters
- Verify project contains indexed documents

**"Vision assist not working"**
- Ensure vision assist is enabled in settings
- Check OpenAI API key has vision model access
- Verify document pages can be rendered as images
- Try with fewer images if hitting rate limits

### Performance Optimization

**For Large Document Sets**
- Process documents in smaller batches
- Enable local embeddings to reduce API calls
- Increase system RAM for better performance
- Use SSD storage for faster file access

**For Better Search Results**
- Maintain detailed project context
- Use specific technical terminology
- Apply appropriate filters
- Enable hybrid search mode

**For API Cost Management**
- Use local embeddings when possible
- Limit vision assist to essential queries
- Monitor usage in OpenAI dashboard
- Consider batch processing during off-peak hours

### Getting Help

**Log Files**
- Application logs: `logs/app.log`
- Error logs: `logs/error.log`
- Debug information available with debug mode enabled

**System Information**
- Check Python version: `python --version`
- Verify dependencies: `pip list`
- Test configuration: Check config.yaml syntax

**Support Resources**
- Review error messages in the UI
- Check the troubleshooting section above
- Examine log files for detailed error information
- Verify all prerequisites are met

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # Data models and type definitions
│   ├── services/          # Service layer components
│   ├── extractors/        # Document extraction providers
│   ├── chunking/          # Layout-aware chunking system
│   ├── retrieval/         # Hybrid retrieval system
│   └── utils/             # Utility functions
├── storage/               # Project data storage
├── tests/                 # Test suite
├── ui_components/         # Streamlit UI components
├── config.yaml           # System configuration
├── .env.example          # Environment variables template
├── requirements.txt      # Python dependencies
└── app.py                # Main Streamlit application
```

## Advanced Usage

### Custom Configuration

Create custom configuration files for different use cases:
```bash
# Copy default configuration
cp config.yaml config-custom.yaml

# Modify settings as needed
# Use with: CONFIG_FILE=config-custom.yaml streamlit run app.py
```

### Batch Processing

For processing many documents programmatically:
```python
from src.services.project_manager import ProjectManager
from src.services.file_processor import FileProcessor

# Create project and process files
pm = ProjectManager()
project_id = pm.create_project("Batch Project")
processor = FileProcessor(project_id)
processor.process_files(["doc1.pdf", "doc2.docx"])
```

### API Integration

The system can be extended with REST API endpoints for integration with other tools and workflows.

## Requirements

- Python 3.8+
- OpenAI API key (for embeddings and LLM services)
- See `requirements.txt` for complete dependency list