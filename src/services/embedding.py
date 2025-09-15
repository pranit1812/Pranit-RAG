"""
Embedding service with provider support for the Construction RAG System.
"""
import os
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import logging
from dataclasses import dataclass

# Third-party imports
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from config import get_config, get_env_var
    from utils.error_handling import (
        safe_api_call,
        retry_with_backoff,
        memory_monitor,
        performance_monitor,
        log_error_with_context,
        EmbeddingError,
        APIError
    )
except ImportError:
    from config import get_config, get_env_var
    from utils.error_handling import (
        safe_api_call,
        retry_with_backoff,
        memory_monitor,
        performance_monitor,
        log_error_with_context,
        EmbeddingError,
        APIError
    )


logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result from embedding operation."""
    embeddings: List[List[float]]
    model: str
    dimensions: int
    total_tokens: Optional[int] = None


class EmbeddingService(ABC):
    """Abstract base class for embedding services."""
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        pass
    
    @abstractmethod
    def get_dimensions(self) -> int:
        """Get the dimensionality of embeddings."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name being used."""
        pass


class OpenAIEmbedding(EmbeddingService):
    """OpenAI embedding service using text-embedding-3-large."""
    
    def __init__(self, model: str = "text-embedding-3-large", batch_size: int = 64):
        """
        Initialize OpenAI embedding service.
        
        Args:
            model: OpenAI embedding model name
            batch_size: Batch size for processing multiple texts
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available. Install with: pip install openai")
        
        api_key = get_env_var("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self._dimensions = self._get_model_dimensions()
        
        logger.info(f"Initialized OpenAI embedding service with model: {model}")
    
    def _get_model_dimensions(self) -> int:
        """Get dimensions for the embedding model."""
        # Known dimensions for OpenAI models
        model_dimensions = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536,
        }
        return model_dimensions.get(self.model, 1536)  # Default to 1536
    
    @safe_api_call(max_retries=3, backoff_factor=1.0, timeout=60)
    def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts using OpenAI API with comprehensive error handling.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        if not texts:
            return EmbeddingResult(
                embeddings=[],
                model=self.model,
                dimensions=self._dimensions,
                total_tokens=0
            )
        
        with performance_monitor(f"embed_texts_{len(texts)}_texts"):
            with memory_monitor(f"embed_texts_{len(texts)}_texts", max_memory_mb=1024):
                return self._embed_texts_internal(texts)
    
    def _embed_texts_internal(self, texts: List[str]) -> EmbeddingResult:
        """Internal embedding method with error handling."""
        all_embeddings = []
        total_tokens = 0
        failed_batches = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            
            try:
                logger.debug(f"Processing embedding batch {batch_num}/{(len(texts) + self.batch_size - 1) // self.batch_size}")
                
                # Validate batch content
                valid_batch = []
                for text in batch:
                    if text and text.strip():
                        # Truncate very long texts to avoid API limits
                        if len(text) > 8000:  # Conservative limit
                            text = text[:8000] + "..."
                            logger.warning(f"Truncated long text to 8000 characters")
                        valid_batch.append(text)
                    else:
                        logger.warning("Skipping empty text in batch")
                        valid_batch.append("empty")  # Placeholder for empty text
                
                if not valid_batch:
                    logger.warning(f"Batch {batch_num} contains no valid texts")
                    continue
                
                response = self.client.embeddings.create(
                    input=valid_batch,
                    model=self.model
                )
                
                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                
                if len(batch_embeddings) != len(valid_batch):
                    logger.warning(f"Embedding count mismatch: expected {len(valid_batch)}, got {len(batch_embeddings)}")
                
                all_embeddings.extend(batch_embeddings)
                
                # Track token usage
                if hasattr(response, 'usage') and response.usage:
                    total_tokens += response.usage.total_tokens
                
                # Rate limiting - small delay between batches
                if i + self.batch_size < len(texts):
                    time.sleep(0.1)
                    
            except openai.RateLimitError as e:
                logger.warning(f"Rate limit hit for batch {batch_num}, retrying with longer delay: {e}")
                time.sleep(5)  # Longer delay for rate limits
                failed_batches.append((i, batch))
                
            except openai.APIError as e:
                error_msg = f"OpenAI API error for batch {batch_num}: {e}"
                logger.error(error_msg)
                log_error_with_context(e, {"batch_num": batch_num, "batch_size": len(batch)}, "embedding_api_error")
                raise EmbeddingError(error_msg) from e
                
            except openai.AuthenticationError as e:
                error_msg = f"OpenAI authentication error: {e}"
                logger.error(error_msg)
                raise EmbeddingError(error_msg) from e
                
            except Exception as e:
                error_msg = f"Unexpected error generating embeddings for batch {batch_num}: {e}"
                logger.error(error_msg)
                log_error_with_context(e, {"batch_num": batch_num, "batch_size": len(batch)}, "embedding_error")
                raise EmbeddingError(error_msg) from e
        
        # Retry failed batches
        for i, batch in failed_batches:
            batch_num = i // self.batch_size + 1
            try:
                logger.info(f"Retrying failed batch {batch_num}")
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                if hasattr(response, 'usage') and response.usage:
                    total_tokens += response.usage.total_tokens
                    
            except Exception as e:
                error_msg = f"Failed to retry batch {batch_num}: {e}"
                logger.error(error_msg)
                raise EmbeddingError(error_msg) from e
        
        if len(all_embeddings) != len(texts):
            logger.warning(f"Final embedding count mismatch: expected {len(texts)}, got {len(all_embeddings)}")
        
        logger.info(f"Generated {len(all_embeddings)} embeddings using {total_tokens} tokens")
        
        return EmbeddingResult(
            embeddings=all_embeddings,
            model=self.model,
            dimensions=self._dimensions,
            total_tokens=total_tokens
        )
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        result = self.embed_texts([query])
        return result.embeddings[0] if result.embeddings else []
    
    def get_dimensions(self) -> int:
        """Get the dimensionality of embeddings."""
        return self._dimensions
    
    def get_model_name(self) -> str:
        """Get the model name being used."""
        return self.model


class LocalEmbedding(EmbeddingService):
    """Local embedding service using SentenceTransformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L12-v2", batch_size: int = 64):
        """
        Initialize local embedding service.
        
        Args:
            model_name: SentenceTransformers model name
            batch_size: Batch size for processing multiple texts
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "SentenceTransformers package not available. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Load model
        try:
            self.model = SentenceTransformer(model_name)
            self._dimensions = self.model.get_sentence_embedding_dimension()
            
            # Use GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info(f"Using GPU for local embeddings")
            else:
                logger.info(f"Using CPU for local embeddings")
                
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model {model_name}: {e}")
            raise
        
        logger.info(f"Initialized local embedding service with model: {model_name}")
    
    def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts using local model.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        if not texts:
            return EmbeddingResult(
                embeddings=[],
                model=self.model_name,
                dimensions=self._dimensions
            )
        
        try:
            # Generate embeddings in batches
            all_embeddings = []
            
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                all_embeddings.extend(batch_embeddings.tolist())
            
            logger.info(f"Generated {len(all_embeddings)} local embeddings")
            
            return EmbeddingResult(
                embeddings=all_embeddings,
                model=self.model_name,
                dimensions=self._dimensions
            )
            
        except Exception as e:
            logger.error(f"Error generating local embeddings: {e}")
            raise
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        result = self.embed_texts([query])
        return result.embeddings[0] if result.embeddings else []
    
    def get_dimensions(self) -> int:
        """Get the dimensionality of embeddings."""
        return self._dimensions
    
    def get_model_name(self) -> str:
        """Get the model name being used."""
        return self.model_name


class EmbeddingServiceFactory:
    """Factory for creating embedding services based on configuration."""
    
    @staticmethod
    def create_service(provider: Optional[str] = None, **kwargs) -> EmbeddingService:
        """
        Create an embedding service based on configuration.
        
        Args:
            provider: Override provider from config ("openai" or "local")
            **kwargs: Additional arguments for the service
            
        Returns:
            EmbeddingService instance
        """
        config = get_config()
        
        # Use provided provider or fall back to config
        provider = provider or config.embeddings.provider
        
        if provider == "openai":
            model = kwargs.get("model", config.llm.embed_model)
            batch_size = kwargs.get("batch_size", config.embeddings.batch_size)
            return OpenAIEmbedding(model=model, batch_size=batch_size)
        
        elif provider == "local":
            model_name = kwargs.get("model_name", config.embeddings.local_model)
            batch_size = kwargs.get("batch_size", config.embeddings.batch_size)
            return LocalEmbedding(model_name=model_name, batch_size=batch_size)
        
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")


def create_embedding_service(**kwargs) -> EmbeddingService:
    """
    Convenience function to create an embedding service.
    
    Args:
        **kwargs: Arguments passed to EmbeddingServiceFactory.create_service
        
    Returns:
        EmbeddingService instance
    """
    return EmbeddingServiceFactory.create_service(**kwargs)