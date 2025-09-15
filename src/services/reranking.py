"""
Cross-encoder reranking service for the Construction RAG System.
"""
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import time

from models.types import Hit
from config import get_config


logger = logging.getLogger(__name__)


@dataclass
class RerankingResult:
    """Result from reranking operation."""
    hits: List[Hit]
    rerank_time: float
    model_used: str


class CrossEncoderReranker:
    """
    Cross-encoder reranker for improving search result relevance.
    
    Uses a cross-encoder model to score query-document pairs directly,
    providing more accurate relevance scores than bi-encoder approaches.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Name of the cross-encoder model to use
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
        
        logger.info(f"Initialized cross-encoder reranker with model: {model_name}")
    
    def _load_model(self):
        """Load the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Loaded cross-encoder model: {self.model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not available. Cross-encoder reranking disabled. "
                "Install with: pip install sentence-transformers"
            )
            self.model = None
        except Exception as e:
            logger.error(f"Error loading cross-encoder model: {e}")
            self.model = None
    
    def rerank(
        self,
        query: str,
        hits: List[Hit],
        top_k: Optional[int] = None
    ) -> RerankingResult:
        """
        Rerank search hits using cross-encoder model.
        
        Args:
            query: Original search query
            hits: List of hits to rerank
            top_k: Number of top results to return (None = return all)
            
        Returns:
            RerankingResult with reranked hits
        """
        start_time = time.time()
        
        if not self.model:
            logger.warning("Cross-encoder model not available, returning original hits")
            return RerankingResult(
                hits=hits[:top_k] if top_k else hits,
                rerank_time=0.0,
                model_used="none"
            )
        
        if not hits:
            return RerankingResult(
                hits=[],
                rerank_time=0.0,
                model_used=self.model_name
            )
        
        try:
            # Prepare query-document pairs for cross-encoder
            pairs = []
            for hit in hits:
                # Use chunk text for reranking
                document_text = hit["chunk"]["text"]
                
                # Truncate very long documents to avoid model limits
                if len(document_text) > 2000:
                    document_text = document_text[:2000] + "..."
                
                pairs.append([query, document_text])
            
            # Get cross-encoder scores
            scores = self.model.predict(pairs)
            
            # Update hits with new scores
            reranked_hits = []
            for hit, score in zip(hits, scores):
                reranked_hit = hit.copy()
                reranked_hit["score"] = float(score)
                reranked_hits.append(reranked_hit)
            
            # Sort by new scores (higher is better)
            reranked_hits.sort(key=lambda h: h["score"], reverse=True)
            
            # Apply top_k limit if specified
            if top_k:
                reranked_hits = reranked_hits[:top_k]
            
            rerank_time = time.time() - start_time
            
            logger.info(
                f"Reranked {len(hits)} hits to {len(reranked_hits)} results "
                f"in {rerank_time:.3f}s using {self.model_name}"
            )
            
            return RerankingResult(
                hits=reranked_hits,
                rerank_time=rerank_time,
                model_used=self.model_name
            )
            
        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {e}")
            # Return original hits on error
            return RerankingResult(
                hits=hits[:top_k] if top_k else hits,
                rerank_time=0.0,
                model_used="error_fallback"
            )


class NoOpReranker:
    """
    No-operation reranker that returns hits unchanged.
    Used when reranking is disabled.
    """
    
    def rerank(
        self,
        query: str,
        hits: List[Hit],
        top_k: Optional[int] = None
    ) -> RerankingResult:
        """
        Return hits unchanged.
        
        Args:
            query: Original search query (unused)
            hits: List of hits to return
            top_k: Number of top results to return
            
        Returns:
            RerankingResult with original hits
        """
        result_hits = hits[:top_k] if top_k else hits
        
        return RerankingResult(
            hits=result_hits,
            rerank_time=0.0,
            model_used="none"
        )


def create_reranker(reranker_type: str = "none") -> object:
    """
    Create a reranker based on configuration.
    
    Args:
        reranker_type: Type of reranker ("none", "cross_encoder")
        
    Returns:
        Reranker instance
    """
    if reranker_type == "cross_encoder":
        return CrossEncoderReranker()
    else:
        return NoOpReranker()


def create_reranker_from_config() -> object:
    """
    Create a reranker based on current configuration.
    
    Returns:
        Reranker instance
    """
    config = get_config()
    reranker_type = config.retrieve.reranker
    return create_reranker(reranker_type)