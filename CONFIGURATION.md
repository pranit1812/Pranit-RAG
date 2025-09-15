# Configuration Reference

This document provides comprehensive reference for all configuration options in the Construction RAG System.

## Configuration Files

The system uses multiple configuration files:

- `config.yaml` - Main system configuration
- `.env` - Environment variables (API keys)
- `.streamlit/config.toml` - Streamlit web interface settings

## Main Configuration (config.yaml)

### Application Settings

```yaml
app:
  data_dir: ./storage           # Directory for project data storage
  project_cache_size: 3         # Number of projects to keep in memory
  max_upload_mb: 100000        # Maximum total upload size in MB
```

**Options:**
- `data_dir`: Path to storage directory (relative or absolute)
- `project_cache_size`: 1-10 projects (higher uses more memory)
- `max_upload_mb`: Upload limit in megabytes (1000 = 1GB)

### Language Model Settings

```yaml
llm:
  chat_model: gpt-4o           # OpenAI model for generating answers
  embed_model: text-embedding-3-large  # Model for text embeddings
  vision_assist: false         # Enable vision analysis by default
```

**Chat Models:**
- `gpt-4o` - Latest GPT-4 model (recommended)
- `gpt-4-turbo` - Fast GPT-4 variant
- `gpt-3.5-turbo` - Cheaper option with lower quality

**Embedding Models:**
- `text-embedding-3-large` - Best quality (recommended)
- `text-embedding-3-small` - Faster, smaller embeddings
- `text-embedding-ada-002` - Legacy model

### Vision Settings

```yaml
vision:
  enabled: false              # Enable vision assist globally
  max_images: 3               # Number of images to analyze (1-5)
  resolution_scale: 2.0       # Image resolution multiplier
```

**Options:**
- `enabled`: Default vision assist state for new queries
- `max_images`: More images = better context but higher API costs
- `resolution_scale`: Higher resolution = better quality but larger files

### Embedding Configuration

```yaml
embeddings:
  provider: openai            # Embedding provider (openai | local)
  local_model: all-MiniLM-L12-v2  # Local model name
  batch_size: 64              # Batch size for embedding generation
```

**Providers:**
- `openai`: Use OpenAI API (requires API key, best quality)
- `local`: Use SentenceTransformers (offline, free, lower quality)

**Local Models:**
- `all-MiniLM-L12-v2`: Good balance of speed and quality
- `all-MiniLM-L6-v2`: Faster, smaller model
- `all-mpnet-base-v2`: Higher quality, slower

### Extraction Pipeline

```yaml
extract:
  pipeline_priority:          # Order of extraction providers to try
    - docling                 # Advanced layout parsing
    - unstructured_hi_res     # ML-based extraction with OCR
    - native_pdf             # Direct PDF text extraction
    - ocr_ppstructure        # OCR with table structure recognition
  languages:                  # OCR languages
    - en                      # English
  ocr:
    engine: paddleocr         # OCR engine
    ppstructure_model: TableMaster  # Table structure model
    min_conf: 0.5            # Minimum confidence threshold
```

**Extraction Providers:**
- `docling`: Best for complex layouts, tables, multi-column documents
- `unstructured_hi_res`: Good for mixed content with OCR fallback
- `native_pdf`: Fastest for digital PDFs with embedded text
- `ocr_ppstructure`: Best for scanned documents and complex tables

**OCR Languages:**
- `en`: English
- `ch`: Chinese
- `fr`: French
- `de`: German
- `ja`: Japanese
- `ko`: Korean

**PP-Structure Models:**
- `TableMaster`: Best table structure recognition
- `SLANet`: Alternative table model
- `LGPMA`: Layout analysis model

### Chunking Configuration

```yaml
chunk:
  target_tokens: 500          # Target chunk size in tokens
  max_tokens: 900            # Maximum chunk size in tokens
  preserve:
    tables: true             # Keep tables as standalone chunks
    lists: true              # Keep lists as standalone chunks
  drawing:
    cluster_text: true       # Group nearby text in drawings
    max_regions: 8           # Maximum text regions per drawing
```

**Token Settings:**
- `target_tokens`: Ideal chunk size (300-800 recommended)
- `max_tokens`: Hard limit to prevent oversized chunks

**Preservation Options:**
- `tables: true`: Tables become separate chunks with full structure
- `lists: true`: Lists grouped with their titles/context

**Drawing Options:**
- `cluster_text: true`: Use DBSCAN clustering for drawing text
- `max_regions`: Limit text regions to prevent over-segmentation

### Retrieval Configuration

```yaml
retrieve:
  top_k: 5                   # Number of results to return
  hybrid: true               # Use both semantic and keyword search
  reranker: none             # Reranking method (none | cross_encoder)
  sliding_window: false      # Include adjacent chunks
  window_size: 1             # Number of adjacent chunks to include
```

**Search Options:**
- `top_k`: 3-10 results (more = better context, slower responses)
- `hybrid: true`: Combines semantic + keyword search (recommended)
- `hybrid: false`: Semantic search only

**Reranking:**
- `none`: No reranking (fastest)
- `cross_encoder`: Use cross-encoder model for better relevance

**Context Window:**
- `sliding_window: true`: Include chunks before/after retrieved chunks
- `window_size`: Number of adjacent chunks (1-3 recommended)

### Logging Configuration

```yaml
logging:
  log_level: INFO             # Logging level
  log_dir: logs              # Log directory
  max_file_size_mb: 10       # Maximum log file size
  backup_count: 5            # Number of backup log files
  console_logging: true      # Enable console output
```

**Log Levels:**
- `DEBUG`: Detailed debugging information
- `INFO`: General information (recommended)
- `WARNING`: Warning messages only
- `ERROR`: Error messages only

### Error Handling

```yaml
error_handling:
  max_retries: 3             # Maximum retry attempts
  backoff_factor: 1.0        # Exponential backoff multiplier
  continue_on_extraction_error: true  # Continue processing other files
  max_memory_mb: 4096        # Memory limit for processing
  api_timeout_seconds: 30    # API request timeout
```

**Retry Settings:**
- `max_retries`: 0-5 retries for failed operations
- `backoff_factor`: Delay multiplier between retries

**Resource Limits:**
- `max_memory_mb`: Memory limit to prevent system overload
- `api_timeout_seconds`: Timeout for API calls

### Monitoring Configuration

```yaml
monitoring:
  enabled: true              # Enable performance monitoring
  max_history: 1000          # Maximum number of operations to track
  system_monitoring_interval: 30.0  # System stats update interval
  performance_thresholds:
    max_operation_duration: 300.0   # Maximum operation time (seconds)
    max_memory_usage_mb: 2048.0     # Memory usage threshold
  export_metrics_on_shutdown: true  # Save metrics when shutting down
```

**Monitoring Options:**
- `enabled`: Track performance metrics
- `max_history`: Number of operations to keep in memory
- `system_monitoring_interval`: How often to check system resources

### Debug Configuration

```yaml
debug:
  enabled: false             # Enable debug mode
  output_dir: debug_logs     # Debug output directory
  log_extraction_details: false    # Log detailed extraction info
  log_chunking_details: false      # Log chunking process details
  log_retrieval_details: false     # Log search and retrieval details
```

**Debug Options:**
- `enabled`: Enable debug mode (generates large log files)
- Individual logging flags for specific components

## Environment Variables (.env)

```bash
# Required: OpenAI API Key
OPENAI_API_KEY=sk-proj-your-key-here

# Optional: Anthropic API Key (for future features)
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Optional: Custom configuration file
CONFIG_FILE=config-custom.yaml

# Optional: Override data directory
DATA_DIR=/path/to/custom/storage
```

**Required Variables:**
- `OPENAI_API_KEY`: Your OpenAI API key for embeddings and LLM

**Optional Variables:**
- `ANTHROPIC_API_KEY`: For future Claude integration
- `CONFIG_FILE`: Use custom configuration file
- `DATA_DIR`: Override storage location

## Streamlit Configuration (.streamlit/config.toml)

```toml
[server]
maxUploadSize = 100000      # Maximum upload size in MB
port = 8501                 # Port number
address = "localhost"       # Server address
headless = false           # Run in headless mode

[theme]
primaryColor = "#1f77b4"    # Primary color
backgroundColor = "#ffffff"  # Background color
secondaryBackgroundColor = "#f0f2f6"  # Secondary background
textColor = "#262730"       # Text color

[browser]
gatherUsageStats = false    # Disable usage statistics
serverAddress = "localhost" # Server address for browser
serverPort = 8501          # Server port for browser
```

**Server Settings:**
- `maxUploadSize`: Must match or exceed config.yaml max_upload_mb
- `port`: Web interface port (8501 default)
- `headless`: Run without opening browser

**Theme Settings:**
- Customize colors to match your organization's branding
- Use hex color codes

## Configuration Profiles

### Development Profile

```yaml
# config-dev.yaml
app:
  data_dir: ./dev-storage
  project_cache_size: 1

llm:
  chat_model: gpt-3.5-turbo  # Cheaper for development

embeddings:
  provider: local            # No API costs
  batch_size: 16            # Smaller batches

logging:
  log_level: DEBUG          # Detailed logging

debug:
  enabled: true             # Enable debug mode
```

### Production Profile

```yaml
# config-prod.yaml
app:
  data_dir: /data/construction-rag
  project_cache_size: 5

llm:
  chat_model: gpt-4o        # Best quality

embeddings:
  provider: openai          # Best embeddings
  batch_size: 64           # Efficient batching

logging:
  log_level: INFO          # Standard logging

monitoring:
  enabled: true            # Track performance
```

### Low-Resource Profile

```yaml
# config-minimal.yaml
app:
  project_cache_size: 1     # Minimal memory usage

chunk:
  target_tokens: 300        # Smaller chunks
  max_tokens: 500

embeddings:
  provider: local           # No API costs
  local_model: all-MiniLM-L6-v2  # Smaller model
  batch_size: 16           # Small batches

extract:
  pipeline_priority:
    - native_pdf           # Fastest extraction
```

## Configuration Validation

The system validates configuration on startup. Common validation errors:

**Invalid Values:**
```yaml
# Error: Invalid model name
llm:
  chat_model: gpt-5  # Model doesn't exist

# Error: Invalid token count
chunk:
  target_tokens: 2000  # Too large

# Error: Invalid provider
embeddings:
  provider: invalid_provider
```

**Missing Required Fields:**
```yaml
# Error: Missing required sections
app:
  # data_dir is required
  project_cache_size: 3
```

## Environment-Specific Configuration

### Using Custom Configuration Files

```bash
# Set custom config file
export CONFIG_FILE=config-production.yaml
streamlit run app.py

# Or use command line
CONFIG_FILE=config-dev.yaml streamlit run app.py
```

### Override Specific Settings

```bash
# Override data directory
export DATA_DIR=/custom/storage/path
streamlit run app.py
```

### Docker Configuration

```yaml
# docker-compose.yml
version: '3.8'
services:
  construction-rag:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - CONFIG_FILE=config-docker.yaml
    volumes:
      - ./storage:/app/storage
      - ./config-docker.yaml:/app/config-docker.yaml
```

## Performance Tuning

### For Large Document Sets

```yaml
chunk:
  target_tokens: 400        # Slightly smaller chunks
  max_tokens: 700

embeddings:
  batch_size: 32           # Reduce memory usage

app:
  project_cache_size: 2    # Reduce memory usage
```

### For Fast Processing

```yaml
extract:
  pipeline_priority:
    - native_pdf           # Fastest for digital PDFs
    - docling

embeddings:
  provider: local          # No API latency
  local_model: all-MiniLM-L6-v2  # Fastest model
```

### For High Quality Results

```yaml
llm:
  chat_model: gpt-4o       # Best model

embeddings:
  provider: openai         # Best embeddings
  embed_model: text-embedding-3-large

retrieve:
  hybrid: true             # Best search quality
  reranker: cross_encoder  # Better relevance
```

## Troubleshooting Configuration

### Common Issues

**Configuration not loading:**
```bash
# Test configuration syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Test configuration loading
python -c "from src.config import load_config; print(load_config())"
```

**Environment variables not found:**
```bash
# Check .env file
cat .env

# Test environment loading
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('OPENAI_API_KEY'))"
```

**Invalid configuration values:**
- Check log files for validation errors
- Compare with examples in this document
- Use default configuration as baseline

### Configuration Reset

To reset to default configuration:

```bash
# Backup current config
cp config.yaml config.yaml.backup

# Generate new default config
python -c "
from src.config import generate_default_config
generate_default_config()
print('Default configuration generated')
"
```

This configuration reference covers all available options. Start with the default configuration and modify settings based on your specific needs and system resources.