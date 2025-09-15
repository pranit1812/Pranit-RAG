"""
Service Registry for Construction RAG System.
Provides dependency injection and service management without circular imports.
"""
import logging
from typing import Dict, Any, Optional, Type, TypeVar, Callable
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceRegistry:
    """Registry for managing service instances and dependencies."""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._singletons: Dict[str, Any] = {}
        self._lock = Lock()
        self._initialized = False
    
    def register_factory(self, service_name: str, factory: Callable) -> None:
        """Register a factory function for a service."""
        with self._lock:
            self._factories[service_name] = factory
            logger.debug(f"Registered factory for service: {service_name}")
    
    def register_singleton(self, service_name: str, instance: Any) -> None:
        """Register a singleton service instance."""
        with self._lock:
            self._singletons[service_name] = instance
            logger.debug(f"Registered singleton service: {service_name}")
    
    def get_service(self, service_name: str, **kwargs) -> Any:
        """Get a service instance, creating it if necessary."""
        # Check singletons first (no lock needed for read-only check)
        if service_name in self._singletons:
            return self._singletons[service_name]
        
        # Check cached instances
        if service_name in self._services:
            return self._services[service_name]
        
        # Create new instance - avoid deadlock by not holding lock during factory call
        if service_name in self._factories:
            try:
                factory = self._factories[service_name]
                # Call factory WITHOUT holding the lock to avoid deadlock
                instance = factory(**kwargs)
                
                # Now lock only to cache the result
                with self._lock:
                    self._services[service_name] = instance
                
                logger.debug(f"Created service instance: {service_name}")
                return instance
            except Exception as e:
                logger.error(f"Failed to create service {service_name}: {e}")
                raise
        
        raise ValueError(f"Service not registered: {service_name}")
    
    def clear_cache(self) -> None:
        """Clear cached service instances (not singletons)."""
        with self._lock:
            self._services.clear()
            logger.debug("Cleared service cache")
    
    def reset(self) -> None:
        """Reset the entire registry."""
        with self._lock:
            self._services.clear()
            self._factories.clear()
            self._singletons.clear()
            self._initialized = False
            logger.debug("Reset service registry")
    
    def is_initialized(self) -> bool:
        """Check if the registry has been initialized."""
        return self._initialized
    
    def mark_initialized(self) -> None:
        """Mark the registry as initialized."""
        self._initialized = True


# Global service registry instance
_registry = ServiceRegistry()


def get_registry() -> ServiceRegistry:
    """Get the global service registry."""
    return _registry


def register_factory(service_name: str, factory: Callable) -> None:
    """Register a factory function for a service."""
    _registry.register_factory(service_name, factory)


def register_singleton(service_name: str, instance: Any) -> None:
    """Register a singleton service instance."""
    _registry.register_singleton(service_name, instance)


def get_service(service_name: str, **kwargs) -> Any:
    """Get a service instance."""
    return _registry.get_service(service_name, **kwargs)


def initialize_services() -> None:
    """Initialize all services with their factories."""
    if _registry.is_initialized():
        return
    
    try:
        logger.info("Initializing service registry...")
        
        # Register config as singleton (no dependencies)
        from config import get_config
        config_instance = get_config()
        register_singleton("config", config_instance)
        
        # Register embedding service factory
        def embedding_factory(**kwargs):
            from services.embedding import create_embedding_service
            return create_embedding_service(**kwargs)
        
        register_factory("embedding_service", embedding_factory)
        
        # Register vector store factory
        def vector_store_factory(**kwargs):
            from services.vector_store import create_vector_store
            return create_vector_store(**kwargs)
        
        register_factory("vector_store", vector_store_factory)
        
        # Register BM25 index factory
        def bm25_factory(**kwargs):
            from services.bm25_search import create_bm25_index
            return create_bm25_index(**kwargs)
        
        register_factory("bm25_index", bm25_factory)
        
        # Register extraction router factory
        def extraction_router_factory(**kwargs):
            from extractors.extraction_router import ExtractionRouter
            config = get_service("config")
            return ExtractionRouter(config)
        
        register_factory("extraction_router", extraction_router_factory)
        
        # Register chunker factory
        def chunker_factory(**kwargs):
            from chunking.chunker import DocumentChunker
            from models.types import ChunkPolicy
            config = get_service("config")
            
            # Create chunk policy from config
            policy = ChunkPolicy(
                target_tokens=config.chunk.target_tokens,
                max_tokens=config.chunk.max_tokens,
                preserve_tables=config.chunk.preserve.tables,
                preserve_lists=config.chunk.preserve.lists,
                drawing_cluster_text=config.chunk.drawing.cluster_text,
                drawing_max_regions=config.chunk.drawing.max_regions
            )
            
            return DocumentChunker(policy, config.llm.chat_model)
        
        register_factory("chunker", chunker_factory)
        
        # Register classification service factory
        def classification_factory(**kwargs):
            from services.classification import ClassificationService
            return ClassificationService(**kwargs)
        
        register_factory("classification_service", classification_factory)
        
        # Register project context manager factory
        def project_context_factory(**kwargs):
            from services.project_context import ProjectContextManager
            return ProjectContextManager(**kwargs)
        
        register_factory("project_context_manager", project_context_factory)
        
        # Register retrieval service factory
        def retrieval_factory(**kwargs):
            from services.retrieval import HybridRetriever
            embedding_service = get_service("embedding_service")
            
            # Extract project_id from kwargs for vector store and retrieval
            project_id = kwargs.get("project_id")
            persist_dir = kwargs.get("persist_dir")
            
            # Create vector store with project-specific parameters
            vector_store = get_service("vector_store", 
                                     project_id=project_id,
                                     persist_dir=persist_dir)
            
            return HybridRetriever(
                embedding_service=embedding_service,
                vector_store=vector_store,
                project_id=project_id
            )
        
        register_factory("retrieval_service", retrieval_factory)
        
        # Register QA assembly service factory
        def qa_assembly_factory(**kwargs):
            from services.qa_assembly import QAAssemblyService
            config = get_service("config")
            return QAAssemblyService(
                model_name=config.llm.chat_model,
                **kwargs
            )
        
        register_factory("qa_assembly_service", qa_assembly_factory)
        
        # Register vision service factory
        def vision_factory(**kwargs):
            from services.vision import VisionService
            config = get_service("config")
            return VisionService(
                config=config.vision,
                **kwargs
            )
        
        register_factory("vision_service", vision_factory)
        
        # Register indexing pipeline factory
        def indexing_factory(**kwargs):
            from services.indexing import IndexingPipeline
            
            # Extract required parameters
            project_id = kwargs.get("project_id")
            data_dir = kwargs.get("data_dir")
            persist_dir = kwargs.get("persist_dir")
            progress_callback = kwargs.get("progress_callback")
            
            # Get services
            embedding_service = get_service("embedding_service")
            vector_store = get_service("vector_store", 
                                     project_id=project_id,
                                     persist_dir=persist_dir)
            
            return IndexingPipeline(
                project_id=project_id,
                data_dir=data_dir,
                embedding_service=embedding_service,
                vector_store=vector_store,
                progress_callback=progress_callback
            )
        
        register_factory("indexing_pipeline", indexing_factory)
        
        _registry.mark_initialized()
        logger.info("Service registry initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize service registry: {e}")
        raise


def create_rag_services(project_id: str, project_path: Path) -> Dict[str, Any]:
    """
    Create a complete set of RAG services for a project.
    
    Args:
        project_id: Project identifier
        project_path: Path to project directory
        
    Returns:
        Dictionary of initialized services
    """
    try:
        # Initialize the registry if not already done
        initialize_services()
        
        # Create project-specific services
        services = {}
        
        # Get base services
        logger.info("Creating config service...")
        services["config"] = get_service("config")
        
        logger.info("Creating embedding service...")
        services["embedding_service"] = get_service("embedding_service")
        
        logger.info("Creating extraction router...")
        services["extraction_router"] = get_service("extraction_router")
        
        logger.info("Creating chunker...")
        services["chunker"] = get_service("chunker")
        
        logger.info("Creating classification service...")
        services["classification_service"] = get_service("classification_service")
        
        # Create project-specific services
        logger.info("Creating vector store...")
        services["vector_store"] = get_service("vector_store", 
                                              project_id=project_id,
                                              persist_dir=str(project_path / "chroma"))
        
        logger.info("Creating BM25 index...")
        services["bm25_index"] = get_service("bm25_index",
                                           index_dir=str(project_path / "bm25_index"))
        
        logger.info("Creating project context manager...")
        services["project_context_manager"] = get_service("project_context_manager",
                                                         project_path=project_path)
        
        logger.info("Creating retrieval service...")
        services["retrieval_service"] = get_service("retrieval_service",
                                                   project_id=project_id,
                                                   persist_dir=str(project_path / "chroma"),
                                                   index_dir=str(project_path / "bm25_index"))
        
        logger.info("Creating QA assembly service...")
        services["qa_assembly_service"] = get_service("qa_assembly_service")
        
        # Vision service (optional)
        if services["config"].vision.enabled:
            services["vision_service"] = get_service("vision_service",
                                                   project_path=project_path)
        
        services["indexing_pipeline"] = get_service("indexing_pipeline",
                                                   project_id=project_id,
                                                   data_dir=str(project_path),
                                                   persist_dir=str(project_path / "chroma"),
                                                   index_dir=str(project_path / "bm25_index"))
        
        logger.info(f"Created RAG services for project: {project_id}")
        return services
        
    except Exception as e:
        logger.error(f"Failed to create RAG services: {e}")
        raise
