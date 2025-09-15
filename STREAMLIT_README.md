# Construction RAG System - Streamlit UI

A local-first Retrieval-Augmented Generation application for construction subcontractors to process and query mixed construction bid packages.

## Features Implemented

### ✅ Main Application Structure (Task 12.1)
- Streamlit app with proper page configuration
- Sidebar layout with project management and settings
- Main content area with chat interface and results display
- Session state management for project and conversation data
- Error handling and graceful degradation

### ✅ Project Management UI (Task 12.2)
- Project picker dropdown with project statistics
- New project creation form with name validation
- Multi-file uploader with drag-and-drop support
- Processing progress display with real-time updates
- Project deletion with confirmation

### ✅ Settings and Configuration UI (Task 12.3)
- Settings panel that mirrors config.yaml parameters
- Toggles for chunking options (preserve tables/lists)
- Hybrid search and reranker configuration controls
- Vision assist settings with configurable image count
- Export/import configuration functionality

### ✅ Filtering and Search Interface (Task 12.4)
- Content type multiselect (SpecSection, Drawing, ITB, Table, List)
- Division code filtering with MasterFormat division names
- Discipline filtering for drawings (A/S/M/E/P/FP/EL)
- Filter state persistence and reset functionality
- Filter summary and active filter display

### ✅ Chat Interface and Results Display (Task 12.5)
- Chat input with query submission and history display
- Answer display with markdown formatting and citations
- Expandable retrieved snippets with source information
- Source page viewer with image rendering for verification
- Mock search results and LLM responses for demonstration

### ✅ Project Context Management UI (Task 12.6)
- Project context display and editing interface
- Project summary and key systems visualization
- Project context editing with validation
- Auto-generation trigger and status display
- Markdown-based context file management

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install streamlit pyyaml
   ```

2. **Run the Application**
   ```bash
   streamlit run app.py
   ```

3. **Access the Interface**
   - Open your browser to `http://localhost:8501`
   - Create a new project or select an existing one
   - Upload construction documents (PDF, DOCX, XLSX, images)
   - Configure settings and filters as needed
   - Start asking questions about your documents

## File Structure

```
├── app.py                          # Main Streamlit application
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
├── ui_components/
│   └── project_manager_simple.py  # Simplified project manager for UI
├── src/                            # Core system components
└── storage/                        # Project data storage
```

## Configuration

The application uses the existing `config.yaml` file for system configuration. Key settings can be modified through the UI Settings panel.

## Current Limitations

This is a UI-only implementation with mock data for demonstration purposes. The following features are simulated:

- **Document Processing**: Files are uploaded but not actually processed through the extraction pipeline
- **Search Results**: Mock search results are generated for demonstration
- **LLM Responses**: Mock responses with citations are displayed
- **Vision Analysis**: Placeholder for vision assist functionality
- **Source Viewing**: Mock document page viewer

## Integration with Full System

To integrate with the complete Construction RAG System:

1. **Document Processing**: Connect file uploads to the extraction pipeline
2. **Search Backend**: Integrate with hybrid retrieval system (dense + BM25)
3. **LLM Integration**: Connect to OpenAI API for actual response generation
4. **Vision Service**: Implement actual vision analysis with document images
5. **Source Viewing**: Add PDF rendering and page image display

## UI Components

### Sidebar
- **Project Management**: Create, select, and manage projects
- **Settings**: Configure system parameters
- **Search Filters**: Apply content type, division, and discipline filters

### Main Content
- **Project Context**: View and edit project information
- **Chat Interface**: Ask questions and view responses
- **Search Results**: Browse retrieved document chunks
- **Source Viewer**: View original document pages

## Styling

The application includes custom CSS for:
- Professional construction industry appearance
- Clear visual hierarchy
- Responsive layout
- Error and success message styling
- Chat message formatting
- Source citation display

## Error Handling

- Graceful error handling with user-friendly messages
- Detailed error information in expandable sections
- Fallback behavior for missing components
- Session state recovery

## Future Enhancements

- Real-time document processing progress
- Advanced search query suggestions
- Collaborative project sharing
- Export functionality for search results
- Integration with external construction databases
- Mobile-responsive design improvements