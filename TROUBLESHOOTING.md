# Troubleshooting Guide

This guide helps resolve common issues with the Construction RAG System.

## Quick Diagnostics

### System Health Check

Run these commands to verify your installation:

```bash
# Check Python version
python --version

# Check if virtual environment is active
which python  # Should show path to venv/bin/python

# Test basic imports
python -c "import streamlit, yaml, openai; print('Core dependencies OK')"

# Check configuration
python -c "from src.config import load_config; print('Configuration loaded successfully')"
```

### Log File Locations

Check these files for detailed error information:
- `logs/app.log` - Application logs
- `logs/error.log` - Error logs  
- `logs/debug.log` - Debug information (if enabled)

## Installation Issues

### Python and Dependencies

**Issue: "Python not found" or "Command not found"**
```bash
# Windows
py --version
python3 --version

# macOS/Linux
python3 --version
which python3
```

**Solution:**
- Install Python 3.8+ from python.org
- Add Python to system PATH
- Use `python3` instead of `python` on some systems

**Issue: "pip install fails with compilation errors"**

*Windows:*
```bash
# Install Microsoft C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Or use pre-compiled wheels
pip install --only-binary=all -r requirements.txt
```

*macOS:*
```bash
# Install Xcode command line tools
xcode-select --install

# Update pip and setuptools
pip install --upgrade pip setuptools wheel
```

*Linux:*
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev build-essential

# CentOS/RHEL
sudo yum install python3-devel gcc gcc-c++
```

**Issue: "Memory error during installation"**
```bash
# Install packages individually
pip install streamlit
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir -r requirements.txt

# Or increase virtual memory/swap space
```

### Virtual Environment Issues

**Issue: "Virtual environment not activating"**

*Windows:*
```bash
# Try different activation methods
venv\Scripts\activate.bat
venv\Scripts\Activate.ps1

# If PowerShell execution policy blocks:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

*macOS/Linux:*
```bash
# Ensure correct path
source venv/bin/activate

# Check if venv directory exists
ls -la venv/
```

**Issue: "Wrong Python version in virtual environment"**
```bash
# Create venv with specific Python version
python3.9 -m venv venv
# or
virtualenv -p python3.9 venv
```

## Configuration Issues

### Environment Variables

**Issue: "No OpenAI API key found"**

1. **Check .env file exists:**
   ```bash
   ls -la .env
   cat .env  # Should show OPENAI_API_KEY=sk-...
   ```

2. **Verify key format:**
   ```
   # Correct format
   OPENAI_API_KEY=sk-proj-abcd1234...
   
   # Incorrect (no quotes, no spaces)
   OPENAI_API_KEY = "sk-proj-abcd1234..."
   ```

3. **Test API key:**
   ```bash
   python -c "
   import openai
   from dotenv import load_dotenv
   load_dotenv()
   client = openai.OpenAI()
   print('API key valid')
   "
   ```

**Issue: "Configuration file not found or invalid"**

1. **Check config.yaml exists:**
   ```bash
   ls -la config.yaml
   ```

2. **Validate YAML syntax:**
   ```bash
   python -c "import yaml; yaml.safe_load(open('config.yaml'))"
   ```

3. **Reset to defaults:**
   ```bash
   # Backup current config
   cp config.yaml config.yaml.backup
   
   # Regenerate default config
   python -c "from src.config import generate_default_config; generate_default_config()"
   ```

### Streamlit Configuration

**Issue: "Upload size limit exceeded"**

Edit `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 200000  # Increase limit (in MB)
```

**Issue: "Port already in use"**
```bash
# Use different port
streamlit run app.py --server.port 8502

# Or kill process using port 8501
# Windows:
netstat -ano | findstr :8501
taskkill /PID <process_id> /F

# macOS/Linux:
lsof -ti:8501 | xargs kill -9
```

## Application Runtime Issues

### Startup Problems

**Issue: "Streamlit app won't start"**

1. **Check for syntax errors:**
   ```bash
   python -m py_compile app.py
   ```

2. **Run with verbose output:**
   ```bash
   streamlit run app.py --logger.level debug
   ```

3. **Check imports:**
   ```bash
   python -c "
   import sys
   sys.path.append('.')
   import app
   print('App imports successfully')
   "
   ```

**Issue: "Module not found errors"**

1. **Verify Python path:**
   ```bash
   python -c "import sys; print('\n'.join(sys.path))"
   ```

2. **Install missing packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Check virtual environment:**
   ```bash
   which python  # Should point to venv
   pip list | grep streamlit
   ```

### Document Processing Issues

**Issue: "Document upload fails"**

1. **Check file size and format:**
   - Maximum total upload: 100GB (configurable)
   - Supported formats: PDF, DOCX, XLSX, PNG, JPG
   - Files must not be password-protected

2. **Check storage space:**
   ```bash
   df -h .  # Check available disk space
   ```

3. **Verify file permissions:**
   ```bash
   ls -la storage/
   # Should be writable by current user
   ```

**Issue: "Document processing hangs or fails"**

1. **Check memory usage:**
   ```bash
   # Monitor during processing
   top -p $(pgrep -f streamlit)
   ```

2. **Process smaller batches:**
   - Upload fewer files at once
   - Split large PDFs into smaller files

3. **Check extraction logs:**
   ```bash
   tail -f logs/app.log
   # Look for extraction errors
   ```

4. **Test with simple document:**
   - Try with a basic text PDF first
   - Verify system works before processing complex documents

**Issue: "OCR processing fails"**

1. **Check PaddleOCR installation:**
   ```bash
   python -c "import paddleocr; print('PaddleOCR installed')"
   ```

2. **Download models manually:**
   ```bash
   python -c "
   import paddleocr
   ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en')
   print('Models downloaded')
   "
   ```

3. **Check image quality:**
   - Ensure images are clear and readable
   - Try with higher resolution images

### Search and Retrieval Issues

**Issue: "No search results returned"**

1. **Verify documents are processed:**
   - Check project has processed documents
   - Look for chunks in project directory

2. **Check search filters:**
   - Remove all filters and try again
   - Verify filter settings match document content

3. **Try different query types:**
   ```
   # Broad queries
   "construction requirements"
   "building specifications"
   
   # Specific queries
   "HVAC ductwork"
   "electrical conduit"
   ```

4. **Check embedding service:**
   ```bash
   python -c "
   from src.services.embedding import OpenAIEmbedding
   e = OpenAIEmbedding()
   result = e.embed_texts(['test'])
   print(f'Embedding dimension: {len(result[0])}')
   "
   ```

**Issue: "Search results are irrelevant"**

1. **Update project context:**
   - Edit project context to be more specific
   - Include relevant technical terms

2. **Adjust search settings:**
   - Enable hybrid search
   - Try different reranking options

3. **Use more specific queries:**
   - Include technical terms
   - Reference specific sections or divisions

### API and Network Issues

**Issue: "OpenAI API errors"**

1. **Check API key validity:**
   ```bash
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $OPENAI_API_KEY"
   ```

2. **Verify account credits:**
   - Check OpenAI dashboard for usage limits
   - Ensure sufficient credits available

3. **Handle rate limits:**
   - Reduce batch sizes in configuration
   - Add delays between API calls

**Issue: "Network connection errors"**

1. **Check internet connectivity:**
   ```bash
   ping api.openai.com
   ```

2. **Configure proxy if needed:**
   ```bash
   export HTTP_PROXY=http://proxy.company.com:8080
   export HTTPS_PROXY=http://proxy.company.com:8080
   ```

3. **Use local embeddings:**
   ```yaml
   # In config.yaml
   embeddings:
     provider: local
   ```

## Performance Issues

### Memory Problems

**Issue: "Out of memory errors"**

1. **Monitor memory usage:**
   ```bash
   # Linux/macOS
   free -h
   top -p $(pgrep -f streamlit)
   
   # Windows
   tasklist /fi "imagename eq python.exe"
   ```

2. **Reduce memory usage:**
   ```yaml
   # In config.yaml
   chunk:
     target_tokens: 300
     max_tokens: 500
   
   embeddings:
     batch_size: 16
   
   app:
     project_cache_size: 1
   ```

3. **Process documents in smaller batches:**
   - Upload fewer files at once
   - Split large documents

**Issue: "Slow processing performance"**

1. **Enable performance monitoring:**
   ```yaml
   # In config.yaml
   monitoring:
     enabled: true
   ```

2. **Optimize extraction pipeline:**
   ```yaml
   extract:
     pipeline_priority:
       - native_pdf  # Fastest for digital PDFs
       - docling
   ```

3. **Use local embeddings:**
   ```yaml
   embeddings:
     provider: local
     local_model: all-MiniLM-L6-v2  # Smaller, faster model
   ```

### Storage Issues

**Issue: "Disk space errors"**

1. **Check available space:**
   ```bash
   df -h storage/
   ```

2. **Clean up old projects:**
   - Delete unused projects through UI
   - Clear cache directories

3. **Move storage location:**
   ```yaml
   # In config.yaml
   app:
     data_dir: /path/to/larger/drive/storage
   ```

## UI and Browser Issues

### Streamlit Interface Problems

**Issue: "UI not loading or displaying incorrectly"**

1. **Clear browser cache:**
   - Hard refresh: Ctrl+F5 (Windows) or Cmd+Shift+R (Mac)
   - Clear browser cache and cookies

2. **Try different browser:**
   - Test with Chrome, Firefox, Safari, or Edge
   - Disable browser extensions

3. **Check browser console:**
   - Open developer tools (F12)
   - Look for JavaScript errors in console

**Issue: "File upload not working"**

1. **Check file size limits:**
   - Individual files: Check browser limits
   - Total upload: Check Streamlit configuration

2. **Verify file formats:**
   - Only PDF, DOCX, XLSX, PNG, JPG supported
   - Files must not be corrupted

3. **Try different upload method:**
   - Use drag-and-drop vs. browse button
   - Upload files individually

### Session and State Issues

**Issue: "Session state lost or corrupted"**

1. **Clear Streamlit cache:**
   ```bash
   # In browser, go to:
   # http://localhost:8501/?clear_cache=true
   ```

2. **Restart application:**
   ```bash
   # Stop with Ctrl+C, then restart
   streamlit run app.py
   ```

3. **Reset session state:**
   - Refresh browser page
   - Clear browser local storage

## Data and Project Issues

### Project Management Problems

**Issue: "Cannot create or switch projects"**

1. **Check storage permissions:**
   ```bash
   ls -la storage/
   mkdir storage/test_project  # Should succeed
   ```

2. **Verify project structure:**
   ```bash
   ls -la storage/project_name/
   # Should contain: raw/, pages/, chunks.jsonl, etc.
   ```

3. **Check project cache:**
   ```yaml
   # In config.yaml
   app:
     project_cache_size: 3  # Increase if needed
   ```

**Issue: "Project data corrupted or missing"**

1. **Check project files:**
   ```bash
   ls -la storage/project_name/
   cat storage/project_name/chunks.jsonl | head -5
   ```

2. **Rebuild project index:**
   - Delete vector database: `rm -rf storage/project_name/chroma/`
   - Reprocess documents through UI

3. **Backup and restore:**
   ```bash
   # Backup project
   cp -r storage/project_name storage/project_name.backup
   
   # Restore if needed
   cp -r storage/project_name.backup storage/project_name
   ```

## Advanced Troubleshooting

### Debug Mode

Enable detailed logging:

```yaml
# In config.yaml
debug:
  enabled: true
  log_extraction_details: true
  log_chunking_details: true
  log_retrieval_details: true

logging:
  log_level: DEBUG
```

### Component Testing

Test individual components:

```bash
# Test configuration
python -c "from src.config import load_config; print(load_config())"

# Test extraction
python -c "
from src.extractors.native_pdf import NativePDFExtractor
extractor = NativePDFExtractor()
print('PDF extractor loaded')
"

# Test embeddings
python -c "
from src.services.embedding import OpenAIEmbedding
e = OpenAIEmbedding()
result = e.embed_texts(['test'])
print(f'Embedding successful: {len(result[0])} dimensions')
"

# Test vector store
python -c "
from src.services.vector_store import ChromaVectorStore
vs = ChromaVectorStore('test_project')
print('Vector store initialized')
"
```

### System Information

Collect system information for support:

```bash
# Python environment
python --version
pip list > installed_packages.txt

# System information
uname -a  # Linux/macOS
systeminfo  # Windows

# Disk space
df -h

# Memory
free -h  # Linux
vm_stat  # macOS
```

## Getting Additional Help

### Log Analysis

When reporting issues, include:
1. Error messages from logs
2. Steps to reproduce the problem
3. System information
4. Configuration files (remove API keys)

### Common Log Patterns

Look for these patterns in logs:

```
ERROR: OpenAI API key not found
ERROR: Failed to extract from document
WARNING: Low confidence extraction
INFO: Processing completed successfully
DEBUG: Chunk created with 450 tokens
```

### Support Checklist

Before seeking help:
- [ ] Check this troubleshooting guide
- [ ] Review error messages and logs
- [ ] Test with minimal configuration
- [ ] Verify API keys and network connectivity
- [ ] Try with simple test documents
- [ ] Check system requirements are met

### Performance Monitoring

Monitor system performance:

```bash
# CPU and memory usage
htop  # Linux
top   # macOS
taskmgr  # Windows

# Disk I/O
iotop  # Linux
iostat  # macOS

# Network usage
nethogs  # Linux
nettop   # macOS
```

This troubleshooting guide covers the most common issues. For persistent problems, check the application logs and verify your configuration matches the examples provided.