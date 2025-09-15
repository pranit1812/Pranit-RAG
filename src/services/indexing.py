"""
Chunk indexing pipeline for the Construction RAG System.
"""
import os
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

from models.types import Chunk, ChunkMetadata, generate_text_hash
from extractors.extraction_router import ExtractionRouter
from chunking.chunker import DocumentChunker
from services.classification import ClassificationService
from services.embedding import EmbeddingService, create_embedding_service
from services.vector_store import VectorStore, create_vector_store
from config import get_config


logger = logging.getLogger(__name__)


@dataclass
class IndexingProgress:
    """Progress information for indexing operations."""
    total_files: int = 0
    processed_files: int = 0
    total_pages: int = 0
    processed_pages: int = 0
    total_chunks: int = 0
    processed_chunks: int = 0
    current_file: str = ""
    current_operation: str = ""
    start_time: float = 0.0
    is_cancelled: bool = False
    error_message: str = ""


@dataclass
class IndexingResult:
    """Result of indexing operation."""
    success: bool
    total_chunks: int
    new_chunks: int
    cached_chunks: int
    processing_time: float
    error_message: str = ""


class ChunkCache:
    """Cache for tracking processed chunks by text hash."""
    
    def __init__(self, cache_file: Path):
        """
        Initialize chunk cache.
        
        Args:
            cache_file: Path to cache file
        """
        self.cache_file = cache_file
        self._cache: Set[str] = set()
        self._lock = threading.Lock()
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    self._cache = set(cache_data.get('text_hashes', []))
                logger.info(f"Loaded {len(self._cache)} cached chunk hashes")
            except Exception as e:
                logger.warning(f"Failed to load chunk cache: {e}")
                self._cache = set()
    
    def _save_cache(self) -> None:
        """Save cache to file."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump({'text_hashes': list(self._cache)}, f)
        except Exception as e:
            logger.warning(f"Failed to save chunk cache: {e}")
    
    def contains(self, text_hash: str) -> bool:
        """Check if text hash is in cache."""
        with self._lock:
            return text_hash in self._cache
    
    def add(self, text_hash: str) -> None:
        """Add text hash to cache."""
        with self._lock:
            self._cache.add(text_hash)
    
    def add_batch(self, text_hashes: List[str]) -> None:
        """Add multiple text hashes to cache."""
        with self._lock:
            self._cache.update(text_hashes)
    
    def remove(self, text_hash: str) -> None:
        """Remove text hash from cache."""
        with self._lock:
            self._cache.discard(text_hash)
    
    def clear(self) -> None:
        """Clear all cached hashes."""
        with self._lock:
            self._cache.clear()
            self._save_cache()
    
    def save(self) -> None:
        """Save cache to disk."""
        self._save_cache()


class IndexingPipeline:
    """Pipeline for processing documents from extraction to vector storage."""
    
    def __init__(
        self,
        project_id: str,
        data_dir: str,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
        progress_callback: Optional[Callable[[IndexingProgress], None]] = None
    ):
        """
        Initialize indexing pipeline.
        
        Args:
            project_id: Project identifier
            data_dir: Data directory for the project
            embedding_service: Embedding service instance
            vector_store: Vector store instance
            progress_callback: Optional callback for progress updates
        """
        self.project_id = project_id
        self.data_dir = Path(data_dir)
        self.progress_callback = progress_callback
        
        # Initialize services
        self.embedding_service = embedding_service or create_embedding_service()
        self.vector_store = vector_store or create_vector_store()
        
        # Initialize processing components
        config = get_config()
        self.extraction_router = ExtractionRouter(config)
        
        # Create chunk policy for chunker
        from models.types import ChunkPolicy
        policy = ChunkPolicy(
            target_tokens=config.chunk.target_tokens,
            max_tokens=config.chunk.max_tokens,
            preserve_tables=config.chunk.preserve.tables,
            preserve_lists=config.chunk.preserve.lists,
            drawing_cluster_text=config.chunk.drawing.cluster_text,
            drawing_max_regions=config.chunk.drawing.max_regions
        )
        self.chunker = DocumentChunker(policy, config.llm.chat_model)
        self.classifier = ClassificationService()
        
        # Initialize cache
        cache_file = self.data_dir / "chunk_cache.json"
        self.chunk_cache = ChunkCache(cache_file)
        
        # Progress tracking
        self.progress = IndexingProgress()
        self._cancel_event = threading.Event()
        
        logger.info(f"Initialized indexing pipeline for project: {project_id}")
    
    def _update_progress(self, **kwargs) -> None:
        """Update progress and call callback if provided."""
        for key, value in kwargs.items():
            if hasattr(self.progress, key):
                setattr(self.progress, key, value)
        
        if self.progress_callback:
            self.progress_callback(self.progress)
    
    def cancel(self) -> None:
        """Cancel the indexing operation."""
        self._cancel_event.set()
        self._update_progress(is_cancelled=True, current_operation="Cancelling...")
        logger.info("Indexing operation cancelled")
    
    def is_cancelled(self) -> bool:
        """Check if operation is cancelled."""
        return self._cancel_event.is_set()
    
    def process_files(
        self,
        file_paths: List[str],
        batch_size: int = 32,
        max_workers: int = 4
    ) -> IndexingResult:
        """
        Process multiple files through the complete indexing pipeline.
        
        Args:
            file_paths: List of file paths to process
            batch_size: Batch size for embedding generation
            max_workers: Maximum number of worker threads
            
        Returns:
            IndexingResult with processing statistics
        """
        start_time = time.time()
        self.progress = IndexingProgress(
            total_files=len(file_paths),
            start_time=start_time
        )
        
        try:
            self._update_progress(current_operation="Starting indexing pipeline...")
            
            # Process files to extract and chunk
            all_chunks = []
            
            for i, file_path in enumerate(file_paths):
                if self.is_cancelled():
                    break
                
                self._update_progress(
                    processed_files=i,
                    current_file=os.path.basename(file_path),
                    current_operation="Extracting and chunking..."
                )
                
                try:
                    file_chunks = self._process_single_file(file_path)
                    all_chunks.extend(file_chunks)
                    
                    self._update_progress(
                        processed_files=i + 1,
                        total_chunks=len(all_chunks)
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue
            
            if self.is_cancelled():
                return IndexingResult(
                    success=False,
                    total_chunks=0,
                    new_chunks=0,
                    cached_chunks=0,
                    processing_time=time.time() - start_time,
                    error_message="Operation cancelled"
                )
            
            # Filter out cached chunks
            self._update_progress(current_operation="Checking cache...")
            new_chunks, cached_count = self._filter_cached_chunks(all_chunks)
            
            if not new_chunks:
                logger.info("No new chunks to process - all content was cached")
                return IndexingResult(
                    success=True,
                    total_chunks=len(all_chunks),
                    new_chunks=0,
                    cached_chunks=cached_count,
                    processing_time=time.time() - start_time
                )
            
            # Generate embeddings for new chunks
            self._update_progress(
                current_operation="Generating embeddings...",
                total_chunks=len(new_chunks)
            )
            
            embeddings = self._generate_embeddings_batch(new_chunks, batch_size)
            
            if self.is_cancelled():
                return IndexingResult(
                    success=False,
                    total_chunks=len(all_chunks),
                    new_chunks=0,
                    cached_chunks=cached_count,
                    processing_time=time.time() - start_time,
                    error_message="Operation cancelled"
                )
            
            # Store in vector database
            self._update_progress(current_operation="Storing in vector database...")
            self.vector_store.upsert_chunks_with_embeddings(new_chunks, embeddings)
            
            # Update cache
            new_hashes = [chunk["text_hash"] for chunk in new_chunks]
            self.chunk_cache.add_batch(new_hashes)
            self.chunk_cache.save()
            
            processing_time = time.time() - start_time
            self._update_progress(
                current_operation="Completed",
                processed_chunks=len(new_chunks)
            )
            
            logger.info(
                f"Indexing completed: {len(new_chunks)} new chunks, "
                f"{cached_count} cached chunks in {processing_time:.2f}s"
            )
            
            return IndexingResult(
                success=True,
                total_chunks=len(all_chunks),
                new_chunks=len(new_chunks),
                cached_chunks=cached_count,
                processing_time=processing_time
            )
            
        except Exception as e:
            error_msg = f"Indexing pipeline failed: {e}"
            logger.error(error_msg)
            self._update_progress(error_message=error_msg)
            
            return IndexingResult(
                success=False,
                total_chunks=0,
                new_chunks=0,
                cached_chunks=0,
                processing_time=time.time() - start_time,
                error_message=error_msg
            )
    
    def _process_single_file(self, file_path: str) -> List[Chunk]:
        """
        Process a single file through extraction and chunking.
        
        Args:
            file_path: Path to file to process
            
        Returns:
            List of chunks from the file
        """
        file_chunks = []
        
        try:
            # Extract pages from file
            extraction_result = self.extraction_router.extract_document(file_path)
            if extraction_result.success and extraction_result.pages:
                pages = extraction_result.pages
            else:
                logger.warning(f"No pages extracted from {file_path}")
                pages = []
            self._update_progress(total_pages=self.progress.total_pages + len(pages))
            
            # Process each page
            for page in pages:
                if self.is_cancelled():
                    break
                
                # Validate and convert page structure
                if isinstance(page, dict):
                    # Convert dict to PageParse if needed
                    if 'blocks' not in page:
                        logger.warning(f"Page dict missing 'blocks' in {file_path}, skipping page")
                        continue
                    # Page is already a dict, use as-is
                else:
                    # Convert PageParse object to dict if needed
                    try:
                        if hasattr(page, 'blocks'):
                            page = {
                                'page_no': getattr(page, 'page_no', 1),
                                'width': getattr(page, 'width', 800),
                                'height': getattr(page, 'height', 600),
                                'blocks': page.blocks,
                                'artifacts_removed': getattr(page, 'artifacts_removed', [])
                            }
                        else:
                            logger.warning(f"Invalid page object in {file_path}, skipping page")
                            continue
                    except Exception as e:
                        logger.error(f"Failed to convert page object in {file_path}: {e}")
                        continue
                
                # Create document metadata
                doc_metadata = {
                    "project_id": self.project_id,
                    "doc_id": f"doc_{hash(file_path) % 10000}",
                    "doc_name": Path(file_path).name,
                    "file_type": Path(file_path).suffix[1:] if Path(file_path).suffix else "unknown"
                }
                
                try:
                    # Chunk the page
                    page_chunks = self.chunker.chunk_page(page, doc_metadata)
                except Exception as e:
                    logger.error(f"Chunking failed for page in {file_path}: {e}")
                    continue
                
                # Classify and add metadata to chunks
                for chunk in page_chunks:
                    # Add project-specific metadata
                    chunk["metadata"]["project_id"] = self.project_id
                    chunk["metadata"]["doc_name"] = os.path.basename(file_path)
                    
                    # Set basic content type (classification can be done later)
                    chunk["metadata"]["content_type"] = "Document"
                    
                    file_chunks.append(chunk)
                
                self._update_progress(processed_pages=self.progress.processed_pages + 1)
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise
        
        return file_chunks
    
    def _filter_cached_chunks(self, chunks: List[Chunk]) -> tuple[List[Chunk], int]:
        """
        Filter out chunks that are already cached.
        
        Args:
            chunks: List of chunks to filter
            
        Returns:
            Tuple of (new_chunks, cached_count)
        """
        new_chunks = []
        cached_count = 0
        
        for chunk in chunks:
            if self.chunk_cache.contains(chunk["text_hash"]):
                cached_count += 1
            else:
                new_chunks.append(chunk)
        
        logger.info(f"Filtered chunks: {len(new_chunks)} new, {cached_count} cached")
        return new_chunks, cached_count
    
    def _generate_embeddings_batch(self, chunks: List[Chunk], batch_size: int) -> List[List[float]]:
        """
        Generate embeddings for chunks in batches.
        
        Args:
            chunks: List of chunks to embed
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        texts = [chunk["text"] for chunk in chunks]
        
        for i in range(0, len(texts), batch_size):
            if self.is_cancelled():
                break
            
            batch_texts = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            self._update_progress(
                current_operation=f"Generating embeddings (batch {batch_num}/{total_batches})...",
                processed_chunks=i
            )
            
            try:
                result = self.embedding_service.embed_texts(batch_texts)
                all_embeddings.extend(result.embeddings)
                
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {batch_num}: {e}")
                raise
        
        return all_embeddings
    
    def reindex_project(self) -> IndexingResult:
        """
        Reindex all files in the project by clearing cache and reprocessing.
        
        Returns:
            IndexingResult with processing statistics
        """
        logger.info(f"Starting full reindex for project: {self.project_id}")
        
        # Clear existing data
        self.chunk_cache.clear()
        self.vector_store.delete_project(self.project_id)
        
        # Find all files in project raw directory
        raw_dir = self.data_dir / "raw"
        if not raw_dir.exists():
            return IndexingResult(
                success=False,
                total_chunks=0,
                new_chunks=0,
                cached_chunks=0,
                processing_time=0.0,
                error_message="No raw files directory found"
            )
        
        # Get all supported file types
        file_paths = []
        supported_extensions = {'.pdf', '.docx', '.xlsx', '.png', '.jpg', '.jpeg'}
        
        for file_path in raw_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                file_paths.append(str(file_path))
        
        if not file_paths:
            return IndexingResult(
                success=False,
                total_chunks=0,
                new_chunks=0,
                cached_chunks=0,
                processing_time=0.0,
                error_message="No supported files found in raw directory"
            )
        
        return self.process_files(file_paths)
    
    def get_project_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed project.
        
        Returns:
            Dictionary with project statistics
        """
        try:
            chunk_count = self.vector_store.get_chunk_count(self.project_id)
            cache_size = len(self.chunk_cache._cache)
            
            return {
                "project_id": self.project_id,
                "total_chunks": chunk_count,
                "cached_hashes": cache_size,
                "embedding_model": self.embedding_service.get_model_name(),
                "embedding_dimensions": self.embedding_service.get_dimensions()
            }
            
        except Exception as e:
            logger.error(f"Error getting project stats: {e}")
            return {
                "project_id": self.project_id,
                "error": str(e)
            }


class IndexingManager:
    """Manager for handling multiple indexing pipelines."""
    
    def __init__(self, data_dir: str):
        """
        Initialize indexing manager.
        
        Args:
            data_dir: Base data directory
        """
        self.data_dir = Path(data_dir)
        self._pipelines: Dict[str, IndexingPipeline] = {}
        self._lock = threading.Lock()
    
    def get_pipeline(
        self,
        project_id: str,
        progress_callback: Optional[Callable[[IndexingProgress], None]] = None
    ) -> IndexingPipeline:
        """
        Get or create an indexing pipeline for a project.
        
        Args:
            project_id: Project identifier
            progress_callback: Optional progress callback
            
        Returns:
            IndexingPipeline instance
        """
        with self._lock:
            if project_id not in self._pipelines:
                project_dir = self.data_dir / project_id
                self._pipelines[project_id] = IndexingPipeline(
                    project_id=project_id,
                    data_dir=str(project_dir),
                    progress_callback=progress_callback
                )
            
            return self._pipelines[project_id]
    
    def remove_pipeline(self, project_id: str) -> None:
        """
        Remove a pipeline from the manager.
        
        Args:
            project_id: Project identifier
        """
        with self._lock:
            if project_id in self._pipelines:
                del self._pipelines[project_id]
    
    def get_all_project_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all projects.
        
        Returns:
            Dictionary mapping project IDs to their statistics
        """
        stats = {}
        
        # Get stats for active pipelines
        with self._lock:
            for project_id, pipeline in self._pipelines.items():
                stats[project_id] = pipeline.get_project_stats()
        
        # Also check for projects that might exist on disk but not in memory
        if self.data_dir.exists():
            for project_dir in self.data_dir.iterdir():
                if project_dir.is_dir() and project_dir.name not in stats:
                    # Create temporary pipeline to get stats
                    temp_pipeline = IndexingPipeline(
                        project_id=project_dir.name,
                        data_dir=str(project_dir)
                    )
                    stats[project_dir.name] = temp_pipeline.get_project_stats()
        
        return stats