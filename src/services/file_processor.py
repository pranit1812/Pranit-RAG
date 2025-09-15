"""
File upload and processing system for the Construction RAG System.
Handles multi-file uploads, validation, and processing pipeline orchestration.
"""
import os
import shutil
import threading
from typing import List, Dict, Optional, Callable, Any, Set
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from models.types import Chunk, ChunkMetadata
from config import get_config
from utils.io_utils import write_text_file
from ui_components.project_manager_simple import SimpleProjectManager


class ProcessingStatus(Enum):
    """Processing status enumeration."""
    PENDING = "pending"
    VALIDATING = "validating"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class FileUpload:
    """Information about an uploaded file."""
    filename: str
    file_path: Path
    file_size: int
    file_type: str
    upload_time: datetime
    status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0


@dataclass
class ProcessingProgress:
    """Overall processing progress information."""
    total_files: int
    completed_files: int
    failed_files: int
    current_file: Optional[str]
    current_status: ProcessingStatus
    overall_progress: float  # 0.0 to 1.0
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    is_cancelled: bool = False


class FileValidator:
    """Validates uploaded files."""
    
    # Supported file types and extensions
    SUPPORTED_EXTENSIONS = {
        '.pdf': 'pdf',
        '.docx': 'docx',
        '.xlsx': 'xlsx',
        '.png': 'image',
        '.jpg': 'image',
        '.jpeg': 'image',
        '.tiff': 'image',
        '.tif': 'image'
    }
    
    def __init__(self, max_file_size_mb: int = 100000):
        """
        Initialize file validator.
        
        Args:
            max_file_size_mb: Maximum file size in MB
        """
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
    
    def validate_file(self, file_path: Path) -> tuple[bool, Optional[str]]:
        """
        Validate a single file.
        
        Args:
            file_path: Path to file to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if file exists
        if not file_path.exists():
            return False, "File does not exist"
        
        # Check if it's a file (not directory)
        if not file_path.is_file():
            return False, "Path is not a file"
        
        # Check file extension
        extension = file_path.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            supported = ', '.join(self.SUPPORTED_EXTENSIONS.keys())
            return False, f"Unsupported file type. Supported: {supported}"
        
        # Check file size
        try:
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size_bytes:
                max_mb = self.max_file_size_bytes / (1024 * 1024)
                return False, f"File too large. Maximum size: {max_mb:.0f}MB"
        except Exception as e:
            return False, f"Cannot read file size: {e}"
        
        # Check if file is readable
        try:
            with open(file_path, 'rb') as f:
                f.read(1024)  # Try to read first 1KB
        except Exception as e:
            return False, f"File is not readable: {e}"
        
        return True, None
    
    def get_file_type(self, file_path: Path) -> str:
        """
        Get file type from extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            File type string
        """
        extension = file_path.suffix.lower()
        return self.SUPPORTED_EXTENSIONS.get(extension, "unknown")


class ProcessingPipeline:
    """Orchestrates the document processing pipeline."""
    
    def __init__(self, project_id: str):
        """
        Initialize processing pipeline.
        
        Args:
            project_id: Project ID for processing
        """
        self.project_id = project_id
        self.config = get_config()
        self.project_manager = SimpleProjectManager()
        
        # Processing components (will be imported when needed to avoid circular imports)
        self._extraction_router = None
        self._chunker = None
        self._indexing_service = None
    
    def process_file(self, file_upload: FileUpload, 
                    progress_callback: Optional[Callable[[float, str], None]] = None) -> bool:
        """
        Process a single file through the extraction → chunking → indexing pipeline.
        
        Args:
            file_upload: FileUpload object to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if processing successful, False otherwise
        """
        try:
            # Update status
            file_upload.status = ProcessingStatus.EXTRACTING
            if progress_callback:
                progress_callback(0.1, f"Extracting {file_upload.filename}")
            
            # Extract document
            chunks = self._extract_document(file_upload)
            if not chunks:
                file_upload.error_message = "No content extracted from document"
                file_upload.status = ProcessingStatus.FAILED
                return False
            
            # Update progress
            file_upload.progress = 0.5
            if progress_callback:
                progress_callback(0.5, f"Processing {len(chunks)} chunks")
            
            # Index chunks
            success = self._index_chunks(chunks)
            if not success:
                file_upload.error_message = "Failed to index document chunks"
                file_upload.status = ProcessingStatus.FAILED
                return False
            
            # Update progress
            file_upload.progress = 1.0
            file_upload.status = ProcessingStatus.COMPLETED
            if progress_callback:
                progress_callback(1.0, f"Completed {file_upload.filename}")
            
            return True
            
        except Exception as e:
            file_upload.error_message = str(e)
            file_upload.status = ProcessingStatus.FAILED
            return False
    
    def _extract_document(self, file_upload: FileUpload) -> List[Chunk]:
        """
        Extract document content and create chunks.
        
        Args:
            file_upload: FileUpload object
            
        Returns:
            List of extracted chunks
        """
        # Lazy import to avoid circular dependencies
        from config import get_config
        config = get_config()
        
        if self._extraction_router is None:
            from extractors.extraction_router import ExtractionRouter
            self._extraction_router = ExtractionRouter(config)
        
        if self._chunker is None:
            from chunking.chunker import DocumentChunker
            from models.types import ChunkPolicy
            
            # Create chunk policy from config
            policy = ChunkPolicy(
                target_tokens=config.chunk.target_tokens,
                max_tokens=config.chunk.max_tokens,
                preserve_tables=config.chunk.preserve.tables,
                preserve_lists=config.chunk.preserve.lists,
                drawing_cluster_text=config.chunk.drawing.cluster_text,
                drawing_max_regions=config.chunk.drawing.max_regions
            )
            
            self._chunker = DocumentChunker(policy, config.llm.chat_model)
        
        # Extract pages
        extraction_result = self._extraction_router.extract_document(str(file_upload.file_path))
        if not extraction_result or not extraction_result.success or not extraction_result.pages:
            return []
        
        pages = extraction_result.pages
        
        # Create chunks from pages using cross-page token-aware chunking
        all_chunks = []
        doc_metadata = {
            "project_id": self.project_id,
            "doc_id": f"doc_{abs(hash(file_upload.filename)) % 10000}",
            "doc_name": file_upload.filename,
            "file_type": file_upload.file_type,
            "file_path": str(file_upload.file_path)
        }
        
        # Prefer document-level chunking to respect token limits across pages
        try:
            all_chunks = self._chunker.chunk_document(pages, doc_metadata)
        except AttributeError:
            # Fallback to per-page chunking if method unavailable
            for page in pages:
                chunks = self._chunker.chunk_page(page, doc_metadata)
                all_chunks.extend(chunks)

        return all_chunks
    
    def _index_chunks(self, chunks: List[Chunk]) -> bool:
        """
        Index chunks in vector store and BM25 index.
        
        Args:
            chunks: List of chunks to index
            
        Returns:
            True if indexing successful
        """
        # Lazy import to avoid circular dependencies
        if self._indexing_service is None:
            from services.indexing import IndexingPipeline
            self._indexing_service = IndexingPipeline(
                project_id=self.project_id,
                data_dir=str(Path("./storage") / self.project_id),
                embedding_service=None,  # Will be created internally
                vector_store=None        # Will be created internally
            )
        
        try:
            # Index chunks
            self._indexing_service.index_chunks(chunks)
            
            # Save chunks to JSONL file for debugging/export
            self._save_chunks_to_file(chunks)
            
            return True
        except Exception:
            return False
    
    def _save_chunks_to_file(self, chunks: List[Chunk]) -> None:
        """
        Save chunks to JSONL file for debugging and export.
        
        Args:
            chunks: List of chunks to save
        """
        project_path = self.project_manager.get_project_path(self.project_id)
        if not project_path:
            return
        
        chunks_file = project_path / "chunks.jsonl"
        
        try:
            import json
            
            # Read existing chunks
            existing_chunks = []
            if chunks_file.exists():
                content = chunks_file.read_text(encoding='utf-8')
                for line in content.strip().split('\n'):
                    if line.strip():
                        existing_chunks.append(json.loads(line))
            
            # Add new chunks
            all_chunks = existing_chunks + chunks
            
            # Write all chunks back
            lines = []
            for chunk in all_chunks:
                lines.append(json.dumps(chunk, ensure_ascii=False))
            
            write_text_file(str(chunks_file), '\n'.join(lines))
            
        except Exception:
            # Don't fail processing if chunk saving fails
            pass


class FileProcessor:
    """Main file processing service with progress tracking and cancellation."""
    
    def __init__(self, project_id: str):
        """
        Initialize file processor.
        
        Args:
            project_id: Project ID for processing
        """
        self.project_id = project_id
        self.config = get_config()
        self.project_manager = SimpleProjectManager()
        self.validator = FileValidator(self.config.app.max_upload_mb)
        self.pipeline = ProcessingPipeline(project_id)
        
        # Processing state
        self._processing_thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()
        self._progress: Optional[ProcessingProgress] = None
        self._progress_callbacks: List[Callable[[ProcessingProgress], None]] = []
    
    def upload_files(self, file_paths: List[str]) -> List[FileUpload]:
        """
        Upload files to project and validate them.
        
        Args:
            file_paths: List of file paths to upload
            
        Returns:
            List of FileUpload objects with validation results
        """
        uploads = []
        project_path = self.project_manager.get_project_path(self.project_id)
        
        if not project_path:
            raise ValueError(f"Project {self.project_id} does not exist")
        
        raw_dir = project_path / "raw"
        raw_dir.mkdir(exist_ok=True)
        
        for file_path_str in file_paths:
            file_path = Path(file_path_str)
            
            # Validate file
            is_valid, error_msg = self.validator.validate_file(file_path)
            
            # Create upload object
            upload = FileUpload(
                filename=file_path.name,
                file_path=file_path,
                file_size=file_path.stat().st_size if file_path.exists() else 0,
                file_type=self.validator.get_file_type(file_path),
                upload_time=datetime.now(),
                status=ProcessingStatus.PENDING if is_valid else ProcessingStatus.FAILED,
                error_message=error_msg
            )
            
            # Copy file to project if valid
            if is_valid:
                try:
                    dest_path = raw_dir / file_path.name
                    
                    # Handle duplicate filenames
                    counter = 1
                    base_name = file_path.stem
                    extension = file_path.suffix
                    while dest_path.exists():
                        new_name = f"{base_name}_{counter}{extension}"
                        dest_path = raw_dir / new_name
                        counter += 1
                    
                    shutil.copy2(file_path, dest_path)
                    upload.file_path = dest_path
                    upload.filename = dest_path.name
                    
                except Exception as e:
                    upload.status = ProcessingStatus.FAILED
                    upload.error_message = f"Failed to copy file: {e}"
            
            uploads.append(upload)
        
        return uploads
    
    def start_processing(self, uploads: List[FileUpload]) -> None:
        """
        Start processing uploaded files in background thread.
        
        Args:
            uploads: List of FileUpload objects to process
        """
        if self._processing_thread and self._processing_thread.is_alive():
            raise RuntimeError("Processing already in progress")
        
        # Filter to only valid uploads
        valid_uploads = [u for u in uploads if u.status == ProcessingStatus.PENDING]
        
        if not valid_uploads:
            return
        
        # Initialize progress tracking
        self._progress = ProcessingProgress(
            total_files=len(valid_uploads),
            completed_files=0,
            failed_files=0,
            current_file=None,
            current_status=ProcessingStatus.PENDING,
            overall_progress=0.0,
            start_time=datetime.now(),
            is_cancelled=False
        )
        
        # Reset cancel event
        self._cancel_event.clear()
        
        # Start processing thread
        self._processing_thread = threading.Thread(
            target=self._process_files_worker,
            args=(valid_uploads,),
            daemon=True
        )
        self._processing_thread.start()
    
    def cancel_processing(self) -> None:
        """Cancel ongoing processing."""
        self._cancel_event.set()
        if self._progress:
            self._progress.is_cancelled = True
            self._notify_progress_callbacks()
    
    def is_processing(self) -> bool:
        """Check if processing is currently running."""
        return (self._processing_thread is not None and 
                self._processing_thread.is_alive() and 
                not self._cancel_event.is_set())
    
    def get_progress(self) -> Optional[ProcessingProgress]:
        """Get current processing progress."""
        return self._progress
    
    def add_progress_callback(self, callback: Callable[[ProcessingProgress], None]) -> None:
        """
        Add callback for progress updates.
        
        Args:
            callback: Function to call with ProcessingProgress updates
        """
        self._progress_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable[[ProcessingProgress], None]) -> None:
        """
        Remove progress callback.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self._progress_callbacks:
            self._progress_callbacks.remove(callback)
    
    def _process_files_worker(self, uploads: List[FileUpload]) -> None:
        """
        Worker thread function for processing files.
        
        Args:
            uploads: List of FileUpload objects to process
        """
        try:
            for i, upload in enumerate(uploads):
                # Check for cancellation
                if self._cancel_event.is_set():
                    upload.status = ProcessingStatus.CANCELLED
                    break
                
                # Update progress
                if self._progress:
                    self._progress.current_file = upload.filename
                    self._progress.current_status = ProcessingStatus.EXTRACTING
                    self._progress.overall_progress = i / len(uploads)
                    self._notify_progress_callbacks()
                
                # Process file
                def progress_callback(file_progress: float, status: str):
                    if self._progress:
                        self._progress.current_status = ProcessingStatus(status.split()[0].lower())
                        file_contribution = 1.0 / len(uploads)
                        self._progress.overall_progress = (i + file_progress) / len(uploads)
                        self._notify_progress_callbacks()
                
                success = self.pipeline.process_file(upload, progress_callback)
                
                # Update counters
                if self._progress:
                    if success:
                        self._progress.completed_files += 1
                    else:
                        self._progress.failed_files += 1
            
            # Final progress update
            if self._progress and not self._cancel_event.is_set():
                self._progress.overall_progress = 1.0
                self._progress.current_status = ProcessingStatus.COMPLETED
                self._progress.current_file = None
                self._notify_progress_callbacks()
            
            # Update project statistics
            self._update_project_stats()
            
        except Exception as e:
            # Handle unexpected errors
            if self._progress:
                self._progress.current_status = ProcessingStatus.FAILED
                self._notify_progress_callbacks()
    
    def _notify_progress_callbacks(self) -> None:
        """Notify all registered progress callbacks."""
        if self._progress:
            for callback in self._progress_callbacks:
                try:
                    callback(self._progress)
                except Exception:
                    # Don't let callback errors break processing
                    pass
    
    def _update_project_stats(self) -> None:
        """Update project statistics after processing."""
        try:
            project_path = self.project_manager.get_project_path(self.project_id)
            if not project_path:
                return
            
            # Count documents
            raw_dir = project_path / "raw"
            doc_count = len([f for f in raw_dir.iterdir() if f.is_file()]) if raw_dir.exists() else 0
            
            # Count chunks
            chunks_file = project_path / "chunks.jsonl"
            chunk_count = 0
            if chunks_file.exists():
                content = chunks_file.read_text(encoding='utf-8')
                chunk_count = len([line for line in content.split('\n') if line.strip()])
            
            # Update project manager
            self.project_manager.update_project_statistics(self.project_id, doc_count, chunk_count)
            
        except Exception:
            # Don't fail if stats update fails
            pass