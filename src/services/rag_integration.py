"""
RAG Service Integration for Construction RAG System.
Provides a unified interface to all RAG services for the Streamlit application.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from models.types import Chunk, Hit, ProjectContext
from services.registry import create_rag_services

logger = logging.getLogger(__name__)


class RAGServiceIntegration:
    """Complete RAG service integration for the Construction RAG System."""
    
    def __init__(self, project_id: str, project_path: Path):
        """
        Initialize RAG services for a project.
        
        Args:
            project_id: Project identifier
            project_path: Path to project directory
        """
        self.project_id = project_id
        self.project_path = project_path
        self.services_initialized = False
        
        try:
            self._initialize_services()
        except Exception as e:
            logger.error(f"Failed to initialize RAG services: {e}")
            self.services_initialized = False
    
    def _initialize_services(self):
        """Initialize all RAG services using the service registry."""
        try:
            logger.info(f"Initializing RAG services for project: {self.project_id}")
            
            # Create project-specific services using clean registry
            logger.info("Creating project-specific services...")
            services = create_rag_services(self.project_id, self.project_path)
            logger.info(f"Created {len(services)} services")
            
            # Store services as instance attributes
            self.config = services["config"]
            self.extraction_router = services["extraction_router"]
            self.chunker = services["chunker"]
            self.embedding_service = services["embedding_service"]
            self.vector_store = services["vector_store"]
            self.bm25_index = services["bm25_index"]
            self.retriever = services["retrieval_service"]
            self.qa_assembler = services["qa_assembly_service"]
            self.context_manager = services["project_context_manager"]
            self.vision_service = services.get("vision_service")
            self.indexing_pipeline = services["indexing_pipeline"]
            
            self.services_initialized = True
            logger.info("RAG services initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG services: {e}")
            self.services_initialized = False
            raise
    
    def process_document(self, file_path: Path, progress_callback=None) -> Dict[str, Any]:
        """
        Process a single document through the complete RAG pipeline.
        
        Args:
            file_path: Path to the document file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Processing result dictionary
        """
        if not self.services_initialized:
            return {
                "success": False,
                "error": "RAG services not initialized",
                "chunks_created": 0,
                "pages_processed": 0
            }
        
        try:
            if progress_callback:
                progress_callback(0.1, "Starting extraction")
            
            # Use the indexing pipeline to process the document
            result = self.indexing_pipeline.process_file(
                file_path=str(file_path),
                progress_callback=progress_callback
            )
            
            return {
                "success": result.success,
                "error": result.error_message if not result.success else None,
                "chunks_created": result.chunks_created,
                "pages_processed": result.pages_processed
            }
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return {
                "success": False,
                "error": f"Processing failed: {str(e)}",
                "chunks_created": 0,
                "pages_processed": 0
            }
    
    def query_documents(self, query: str, top_k: int = 5, use_vision: bool = False) -> Dict[str, Any]:
        """
        Query the indexed documents using hybrid retrieval.
        
        Args:
            query: User query
            top_k: Number of results to return
            use_vision: Whether to use vision assistance
            
        Returns:
            Query result with answer and sources
        """
        if not self.services_initialized:
            return {
                "success": False,
                "error": "RAG services not initialized",
                "answer": "",
                "sources": []
            }
        
        try:
            # Get project context
            project_context = self.context_manager.get_project_context()
            
            # Perform hybrid retrieval
            search_result = self.retriever.search(
                query=query,
                project_context=project_context,
                top_k=top_k,
                filters={}
            )
            
            if not search_result.hits:
                return {
                    "success": True,
                    "answer": "I couldn't find any relevant information for your query in the uploaded documents.",
                    "sources": [],
                    "retrieved_chunks": []
                }
            
            # Generate answer using QA assembler
            if use_vision and self.vision_service:
                # Use vision-enhanced answering
                response = self.qa_assembler.generate_answer_with_vision(
                    query=query,
                    hits=search_result.hits,
                    project_context=project_context,
                    vision_service=self.vision_service
                )
            else:
                # Standard text-only answering
                response = self.qa_assembler.generate_answer(
                    query=query,
                    hits=search_result.hits,
                    project_context=project_context
                )
            
            # Extract sources from response
            sources = []
            if "sources" in response:
                for i, source_info in enumerate(response["sources"], 1):
                    sources.append({
                        "id": f"S{i}",
                        "doc_name": source_info.get("doc_name", "Unknown"),
                        "page_number": source_info.get("page_number", "?"),
                        "sheet_number": source_info.get("sheet_number"),
                        "content_type": source_info.get("content_type", "Unknown"),
                        "text": source_info.get("text", "")
                    })
            
            return {
                "success": True,
                "answer": response.get("answer", ""),
                "sources": sources,
                "retrieved_chunks": search_result.hits
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "success": False,
                "error": f"Query failed: {str(e)}",
                "answer": "",
                "sources": []
            }
    
    def get_project_statistics(self) -> Dict[str, int]:
        """Get project statistics."""
        try:
            # Count documents
            raw_dir = self.project_path / "raw"
            doc_count = len(list(raw_dir.glob("*"))) if raw_dir.exists() else 0
            
            # Count chunks from vector store
            chunk_count = 0
            if self.services_initialized and self.vector_store:
                try:
                    # Try to get chunk count from vector store
                    collection_info = self.vector_store.get_collection_info()
                    chunk_count = collection_info.get("count", 0)
                except:
                    # Fallback to counting from chunks.jsonl
                    chunks_file = self.project_path / "chunks.jsonl"
                    if chunks_file.exists():
                        with open(chunks_file, 'r', encoding='utf-8') as f:
                            chunk_count = sum(1 for line in f if line.strip())
            
            return {
                "doc_count": doc_count,
                "chunk_count": chunk_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get project statistics: {e}")
            return {"doc_count": 0, "chunk_count": 0}


def create_rag_integration(project_id: str, project_path: Path) -> Optional[RAGServiceIntegration]:
    """
    Create RAG service integration for a project.
    
    Args:
        project_id: Project identifier
        project_path: Path to project directory
        
    Returns:
        RAG service integration instance or None if failed
    """
    try:
        return RAGServiceIntegration(project_id, project_path)
    except Exception as e:
        logger.error(f"Failed to create RAG integration: {e}")
        return None
