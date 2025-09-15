# Installation Guide

This guide provides detailed installation instructions for the Construction RAG System.

## System Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: Minimum 4GB, 8GB+ recommended for large documents
- **Storage**: 10GB+ free space for document storage and processing
- **Network**: Internet connection for OpenAI API calls

### Software Requirements
- **Python**: Version 3.8 or higher
- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Browser**: Modern web browser (Chrome, Firefox, Safari, Edge)

## Installation Methods

### Method 1: Standard Installation (Recommended)

1. **Download and Extract**
   ```bash
   # If using git
   git clone <repository-url>
   cd construction-rag-system
   
   # Or download and extract ZIP file
   ```

2. **Create Virtual Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows (PowerShell):
   venv\Scripts\Activate.ps1
   
   # On Windows (Command Prompt):
   venv\Scripts\activate.bat
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   # Upgrade pip first
   python -m pip install --upgrade pip
   
   # Install requirements
   pip install -r requirements.txt
   ```

4. **Configure Environment**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env file with your API keys
   # Windows users can use: copy .env.example .env
   ```

5. **Test Installation**
   ```bash
   # Run basic configuration test
   python -c "import streamlit; print('Streamlit installed successfully')"
   python -c "import src.config; print('Configuration loaded successfully')"
   ```

### Method 2: Development Installation

For developers who want to modify the system:

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd construction-rag-system
   ```

2. **Install in Development Mode**
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   venv\Scripts\Activate.ps1
   # On Windows (Command Prompt):
   venv\Scripts\activate.bat
   # On macOS/Linux:
   source venv/bin/activate
   
   # Install with development dependencies
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Install Pre-commit Hooks** (Optional)
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Configuration Setup

### 1. Environment Variables

Edit the `.env` file with your API credentials:

```bash
# Required: OpenAI API Key
OPENAI_API_KEY=sk-your-openai-api-key-here

# Optional: Anthropic API Key (for future features)
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
```

**Getting an OpenAI API Key:**
1. Visit https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key and paste it in your `.env` file
5. Ensure you have sufficient credits in your OpenAI account

### 2. System Configuration

The `config.yaml` file contains system settings. Default values work for most users, but you can customize:

```yaml
# Example customizations
app:
  data_dir: ./my-projects     # Change storage location
  max_upload_mb: 50000       # Reduce upload limit

llm:
  chat_model: gpt-3.5-turbo  # Use less expensive model

embeddings:
  provider: local            # Use local embeddings (no API calls)
```

### 3. Streamlit Configuration

The `.streamlit/config.toml` file configures the web interface:

```toml
[server]
maxUploadSize = 100000      # 100GB upload limit
port = 8501                 # Default port

[theme]
primaryColor = "#1f77b4"    # Customize colors
```

## Verification

### 1. Basic Functionality Test

```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Start the application
streamlit run app.py
```

The application should open in your browser at `http://localhost:8501`

### 2. Component Tests

Test individual components:

```bash
# Test configuration loading
python -c "from src.config import load_config; print('Config loaded:', load_config())"

# Test OpenAI connection (requires API key)
python -c "from src.services.embedding import OpenAIEmbedding; e = OpenAIEmbedding(); print('OpenAI connection successful')"

# Run test suite
python -m pytest tests/ -v
```

### 3. Document Processing Test

1. Create a test project in the UI
2. Upload a small PDF document
3. Verify processing completes without errors
4. Ask a test question and verify response

## Troubleshooting Installation

### Common Issues

**Python Version Issues**
```bash
# Check Python version
python --version

# If using Python 3.8+, ensure pip is updated
python -m pip install --upgrade pip
```

**Dependency Installation Failures**

For Windows users with compilation errors:
```bash
# Install Microsoft C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Or use conda for problematic packages
conda install -c conda-forge paddlepaddle
```

For macOS users:
```bash
# Install Xcode command line tools
xcode-select --install

# Use Homebrew for system dependencies
brew install python@3.9
```

For Linux users:
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-dev python3-pip build-essential

# For CentOS/RHEL
sudo yum install python3-devel gcc gcc-c++
```

**Memory Issues During Installation**
```bash
# Install packages one at a time
pip install streamlit
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt --no-cache-dir
```

**Permission Issues**
```bash
# Use user installation
pip install --user -r requirements.txt

# Or fix permissions (Linux/macOS)
sudo chown -R $USER:$USER venv/
```

### Platform-Specific Notes

**Windows**
- Use Command Prompt or PowerShell
- Ensure Python is added to PATH
- Consider using Anaconda for easier dependency management

**macOS**
- Use Terminal
- Install Homebrew for system dependencies
- May need to install Xcode command line tools

**Linux**
- Use terminal
- Install system development packages
- Consider using pyenv for Python version management

### Performance Optimization

**For Low-Memory Systems**
```yaml
# In config.yaml
chunk:
  target_tokens: 300        # Smaller chunks
  max_tokens: 600

embeddings:
  batch_size: 32           # Smaller batches
```

**For Slow Networks**
```yaml
# Use local embeddings to reduce API calls
embeddings:
  provider: local
  local_model: all-MiniLM-L6-v2  # Smaller model
```

## Next Steps

After successful installation:

1. **Read the User Guide** in README.md
2. **Create your first project** and upload test documents
3. **Explore the configuration options** to customize behavior
4. **Review the troubleshooting section** for common issues

## Getting Help

If you encounter issues:

1. **Check the logs** in the `logs/` directory
2. **Review error messages** in the Streamlit interface
3. **Verify your configuration** matches the examples
4. **Test with smaller documents** first
5. **Check your OpenAI API key** and account credits

## Uninstallation

To remove the system:

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf venv/  # or rmdir /s venv on Windows

# Remove project files
rm -rf construction-rag-system/
```

Your project data in the `storage/` directory will be preserved unless explicitly deleted.