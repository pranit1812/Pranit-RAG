"""
Services module for the Construction RAG System.
"""
from services.registry import (
    get_registry,
    register_factory,
    register_singleton,
    get_service,
    initialize_services,
    create_rag_services
)

# Import service classes for type hints (lazy loading to avoid circular imports)
def get_embedding_service():
    """Get embedding service classes."""
    from services.embedding import EmbeddingService, EmbeddingResult, OpenAIEmbedding, LocalEmbedding, create_embedding_service
    return EmbeddingService, EmbeddingResult, OpenAIEmbedding, LocalEmbedding, create_embedding_service

def get_vector_store():
    """Get vector store classes."""
    from services.vector_store import VectorStore, ChromaVectorStore, create_vector_store
    return VectorStore, ChromaVectorStore, create_vector_store

def get_project_context():
    """Get project context classes."""
    from services.project_context import ProjectContextGenerator, ProjectContextManager, ProjectContextCache, QueryEnhancer
    return ProjectContextGenerator, ProjectContextManager, ProjectContextCache, QueryEnhancer

def get_indexing():
    """Get indexing classes."""
    from services.indexing import IndexingPipeline, IndexingProgress, IndexingResult, ChunkCache
    return IndexingPipeline, IndexingProgress, IndexingResult, ChunkCache

def get_qa_assembly():
    """Get QA assembly classes."""
    from services.qa_assembly import QAAssemblyService, ContextBuilder, CitationGenerator, LLMService
    return QAAssemblyService, ContextBuilder, CitationGenerator, LLMService

def get_retrieval():
    """Get retrieval classes."""
    from services.retrieval import HybridRetriever, DenseSemanticSearch, SearchResult
    return HybridRetriever, DenseSemanticSearch, SearchResult

def get_vision():
    """Get vision service classes."""
    from services.vision import VisionService, ImageRenderer, VisionServiceError
    return VisionService, ImageRenderer, VisionServiceError

def get_classification():
    """Get classification service classes."""
    from services.classification import ClassificationService
    return ClassificationService,

def get_bm25():
    """Get BM25 search classes."""
    from services.bm25_search import BM25Index, BM25SearchResult, create_bm25_index
    return BM25Index, BM25SearchResult, create_bm25_index

def get_filtering():
    """Get filtering classes."""
    from services.filtering import MetadataFilter, FilterCriteria, create_metadata_filter
    return MetadataFilter, FilterCriteria, create_metadata_filter

def get_reranking():
    """Get reranking classes."""
    from services.reranking import create_reranker_from_config, RerankingResult
    return create_reranker_from_config, RerankingResult

def get_rag_integration():
    """Get RAG integration classes."""
    from services.rag_integration import RAGServiceIntegration, create_rag_integration
    return RAGServiceIntegration, create_rag_integration

def get_document_processor():
    """Get document processor functions."""
    from services.document_processor import process_uploaded_documents, query_project_documents
    return process_uploaded_documents, query_project_documents

# Initialize services on import
try:
    initialize_services()
except Exception as e:
    import logging
    logging.getLogger(__name__).warning(f"Failed to initialize services: {e}")

__all__ = [
    # Registry functions
    "get_registry",
    "register_factory", 
    "register_singleton",
    "get_service",
    "initialize_services",
    "create_rag_services",
    
    # Service getter functions
    "get_embedding_service",
    "get_vector_store",
    "get_project_context",
    "get_indexing",
    "get_qa_assembly",
    "get_retrieval",
    "get_vision",
    "get_classification",
    "get_bm25",
    "get_filtering",
    "get_reranking",
    "get_rag_integration",
    "get_document_processor"
]