"""
Dense semantic search and hybrid retrieval system for the Construction RAG System.
"""
import logging
from typing import List, Dict, Any, Optional, Set, Union
from dataclasses import dataclass
from collections import defaultdict

from models.types import Hit, Chunk, ProjectContext
from services.embedding import EmbeddingService, create_embedding_service
from services.vector_store import VectorStore, create_vector_store
from services.bm25_search import BM25Index, create_bm25_index, BM25SearchResult
from services.reranking import create_reranker_from_config, RerankingResult
from services.filtering import MetadataFilter, FilterCriteria, create_metadata_filter
from config import get_config

# Import QueryEnhancer directly to avoid circular import
try:
    from services.project_context import QueryEnhancer
except ImportError:
    # Fallback if project_context has import issues
    class QueryEnhancer:
        def enhance_query(self, query: str, project_context=None) -> str:
            return query


logger = logging.getLogger(__name__)


# Use FilterCriteria from filtering module instead of SearchFilters
# Keeping this for backward compatibility
SearchFilters = FilterCriteria


@dataclass
class SearchResult:
    """Result from search operation."""
    hits: List[Hit]
    total_found: int
    query_enhanced: str
    search_time: float


class DenseSemanticSearch:
    """Dense vector search using embeddings and vector store."""
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None
    ):
        """
        Initialize dense semantic search.
        
        Args:
            embedding_service: Embedding service for query encoding
            vector_store: Vector store for similarity search
        """
        self.embedding_service = embedding_service or create_embedding_service()
        self.vector_store = vector_store or create_vector_store()
        self.query_enhancer = QueryEnhancer()
        self.metadata_filter = create_metadata_filter()
        
        logger.info("Initialized dense semantic search")
    
    def search(
        self,
        query: str,
        project_id: str,
        project_context: Optional[ProjectContext] = None,
        k: int = 5,
        filters: Optional[FilterCriteria] = None
    ) -> SearchResult:
        """
        Perform dense semantic search with project context enhancement.
        
        Args:
            query: User query string
            project_id: Project identifier
            project_context: Optional project context for query enhancement
            k: Number of results to return
            filters: Optional search filters
            
        Returns:
            SearchResult with hits and metadata
        """
        import time
        start_time = time.time()
        
        try:
            # Enhance query with project context
            enhanced_query = query
            if project_context:
                enhanced_query = self.query_enhancer.enhance_query(query, project_context)
                logger.debug(f"Enhanced query: {enhanced_query}")
            
            # Generate query embedding
            query_vector = self.embedding_service.embed_query(enhanced_query)
            
            # Build metadata filters for vector store
            where_clause = self._build_where_clause(project_id, filters)
            
            # Perform vector search
            hits = self.vector_store.query(
                vector=query_vector,
                k=k * 2,  # Get more results for deduplication
                where=where_clause
            )
            
            # Apply post-processing
            processed_hits = self._post_process_hits(hits, k)
            
            search_time = time.time() - start_time
            
            logger.info(
                f"Dense search completed: {len(processed_hits)} hits in {search_time:.3f}s"
            )
            
            return SearchResult(
                hits=processed_hits,
                total_found=len(hits),
                query_enhanced=enhanced_query,
                search_time=search_time
            )
            
        except Exception as e:
            logger.error(f"Error in dense semantic search: {e}")
            raise
    
    def _build_where_clause(
        self,
        project_id: str,
        filters: Optional[FilterCriteria]
    ) -> Dict[str, Any]:
        """
        Build where clause for vector store query.
        
        Args:
            project_id: Project identifier
            filters: Optional search filters
            
        Returns:
            Where clause dictionary
        """
        if not filters:
            filters = FilterCriteria(project_id=project_id)
        else:
            # Ensure project_id is set
            filters.project_id = project_id
        
        # Use metadata filter to build vector store filter
        return self.metadata_filter.build_vector_store_filter(filters)
    
    def _post_process_hits(self, hits: List[Hit], k: int) -> List[Hit]:
        """
        Post-process search hits for deduplication and source diversification.
        
        Args:
            hits: Raw hits from vector search
            k: Target number of results
            
        Returns:
            Processed and filtered hits
        """
        if not hits:
            return []
        
        # Deduplicate by text hash (exact content matches)
        seen_hashes = set()
        deduplicated = []
        
        for hit in hits:
            text_hash = hit["chunk"]["text_hash"]
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                deduplicated.append(hit)
        
        # Source diversification - prefer results from different documents
        diversified = self._diversify_sources(deduplicated, k)
        
        return diversified[:k]
    
    def _diversify_sources(self, hits: List[Hit], k: int) -> List[Hit]:
        """
        Diversify results across different source documents.
        
        Args:
            hits: Deduplicated hits
            k: Target number of results
            
        Returns:
            Source-diversified hits
        """
        if len(hits) <= k:
            return hits
        
        # Group hits by document
        hits_by_doc = defaultdict(list)
        for hit in hits:
            doc_name = hit["chunk"]["metadata"]["doc_name"]
            hits_by_doc[doc_name].append(hit)
        
        # Select results with round-robin from different documents
        diversified = []
        doc_names = list(hits_by_doc.keys())
        doc_index = 0
        
        while len(diversified) < k and any(hits_by_doc.values()):
            doc_name = doc_names[doc_index]
            
            if hits_by_doc[doc_name]:
                # Take the best remaining hit from this document
                hit = hits_by_doc[doc_name].pop(0)
                diversified.append(hit)
            
            doc_index = (doc_index + 1) % len(doc_names)
            
            # Break if we've exhausted all documents
            if not any(hits_by_doc.values()):
                break
        
        # Fill remaining slots with best remaining hits
        remaining_hits = []
        for doc_hits in hits_by_doc.values():
            remaining_hits.extend(doc_hits)
        
        # Sort remaining by score and add to fill quota
        remaining_hits.sort(key=lambda h: h["score"])
        diversified.extend(remaining_hits[:k - len(diversified)])
        
        return diversified


class RelevanceScorer:
    """Scores and ranks search results based on multiple factors."""
    
    def __init__(self):
        """Initialize relevance scorer."""
        self.content_type_weights = {
            "SpecSection": 1.0,
            "Table": 0.9,
            "Drawing": 0.8,
            "List": 0.7,
            "ITB": 0.6
        }
        
        self.confidence_weight = 0.1  # Boost for high-confidence content
    
    def score_hits(self, hits: List[Hit], query: str) -> List[Hit]:
        """
        Score and re-rank hits based on relevance factors.
        
        Args:
            hits: List of hits to score
            query: Original query for context
            
        Returns:
            Re-ranked hits with updated scores
        """
        scored_hits = []
        
        for hit in hits:
            # Start with vector similarity score (lower is better for distance)
            base_score = 1.0 - hit["score"]  # Convert distance to similarity
            
            # Apply content type weighting
            content_type = hit["chunk"]["metadata"]["content_type"]
            content_weight = self.content_type_weights.get(content_type, 0.5)
            
            # Apply confidence weighting
            confidence_boost = 0.0
            if not hit["chunk"]["metadata"]["low_conf"]:
                confidence_boost = self.confidence_weight
            
            # Calculate final relevance score
            relevance_score = base_score * content_weight + confidence_boost
            
            # Update hit with new score
            scored_hit = hit.copy()
            scored_hit["score"] = relevance_score
            scored_hits.append(scored_hit)
        
        # Sort by relevance score (higher is better)
        scored_hits.sort(key=lambda h: h["score"], reverse=True)
        
        return scored_hits


class RankFusion:
    """Reciprocal Rank Fusion (RRF) for combining search results."""
    
    def __init__(self, k: int = 60):
        """
        Initialize rank fusion.
        
        Args:
            k: RRF parameter (typically 60)
        """
        self.k = k
    
    def fuse_results(
        self,
        dense_hits: List[Hit],
        bm25_hits: List[Hit],
        alpha: float = 0.5
    ) -> List[Hit]:
        """
        Combine dense and BM25 results using Reciprocal Rank Fusion.
        
        Args:
            dense_hits: Results from dense semantic search
            bm25_hits: Results from BM25 keyword search
            alpha: Weight for dense vs BM25 results (0.5 = equal weight)
            
        Returns:
            Fused and ranked results
        """
        # Create score maps for each result set
        dense_scores = {}
        bm25_scores = {}
        
        # Calculate RRF scores for dense results
        for rank, hit in enumerate(dense_hits, 1):
            rrf_score = 1.0 / (self.k + rank)
            dense_scores[hit["id"]] = rrf_score
        
        # Calculate RRF scores for BM25 results
        for rank, hit in enumerate(bm25_hits, 1):
            rrf_score = 1.0 / (self.k + rank)
            bm25_scores[hit["id"]] = rrf_score
        
        # Combine scores and collect all unique hits
        all_hits = {}
        
        # Add dense hits
        for hit in dense_hits:
            hit_id = hit["id"]
            dense_score = dense_scores.get(hit_id, 0.0)
            bm25_score = bm25_scores.get(hit_id, 0.0)
            
            combined_score = alpha * dense_score + (1 - alpha) * bm25_score
            
            fused_hit = hit.copy()
            fused_hit["score"] = combined_score
            all_hits[hit_id] = fused_hit
        
        # Add BM25-only hits
        for hit in bm25_hits:
            hit_id = hit["id"]
            if hit_id not in all_hits:
                dense_score = dense_scores.get(hit_id, 0.0)
                bm25_score = bm25_scores.get(hit_id, 0.0)
                
                combined_score = alpha * dense_score + (1 - alpha) * bm25_score
                
                fused_hit = hit.copy()
                fused_hit["score"] = combined_score
                all_hits[hit_id] = fused_hit
        
        # Sort by combined score (higher is better)
        fused_results = list(all_hits.values())
        fused_results.sort(key=lambda h: h["score"], reverse=True)
        
        return fused_results


class HybridRetriever:
    """Hybrid retrieval system combining dense and sparse search methods."""
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
        project_id: Optional[str] = None
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            embedding_service: Embedding service for dense search
            vector_store: Vector store for dense search
            project_id: Project ID for BM25 index initialization
        """
        self.dense_search = DenseSemanticSearch(embedding_service, vector_store)
        self.relevance_scorer = RelevanceScorer()
        self.rank_fusion = RankFusion()
        self.reranker = create_reranker_from_config()
        self.metadata_filter = create_metadata_filter()
        
        # Initialize BM25 index if project_id provided
        self.bm25_index = None
        if project_id:
            self.bm25_index = create_bm25_index(project_id)
        
        # Load configuration
        config = get_config()
        self.default_k = config.retrieve.top_k
        self.use_hybrid = config.retrieve.hybrid
        self.use_reranker = config.retrieve.reranker != "none"
        
        logger.info("Initialized hybrid retriever")
    
    def retrieve(
        self,
        query: str,
        project_id: str,
        project_context: Optional[ProjectContext] = None,
        k: Optional[int] = None,
        filters: Optional[FilterCriteria] = None,
        use_hybrid: Optional[bool] = None
    ) -> SearchResult:
        """
        Retrieve relevant chunks using hybrid search approach.
        
        Args:
            query: User query string
            project_id: Project identifier
            project_context: Optional project context for enhancement
            k: Number of results to return (defaults to config value)
            filters: Optional search filters
            use_hybrid: Whether to use hybrid search (defaults to config value)
            
        Returns:
            SearchResult with retrieved chunks
        """
        k = k or self.default_k
        use_hybrid = use_hybrid if use_hybrid is not None else self.use_hybrid
        
        try:
            if use_hybrid and self.bm25_index:
                # Perform hybrid search with both dense and BM25
                return self._hybrid_search(
                    query=query,
                    project_id=project_id,
                    project_context=project_context,
                    k=k,
                    filters=filters
                )
            else:
                # Fall back to dense search only
                result = self.dense_search.search(
                    query=query,
                    project_id=project_id,
                    project_context=project_context,
                    k=k,
                    filters=filters
                )
                
                # Apply reranking if enabled
                if self.use_reranker and result.hits:
                    rerank_result = self.reranker.rerank(query, result.hits, k)
                    result.hits = rerank_result.hits
                    logger.debug(f"Reranking completed in {rerank_result.rerank_time:.3f}s using {rerank_result.model_used}")
                elif result.hits:
                    # Apply relevance scoring if not reranking
                    scored_hits = self.relevance_scorer.score_hits(result.hits, query)
                    result.hits = scored_hits
                
                logger.info(f"Dense-only retrieval completed: {len(result.hits)} results")
                
                return result
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            raise
    
    def _hybrid_search(
        self,
        query: str,
        project_id: str,
        project_context: Optional[ProjectContext] = None,
        k: int = 5,
        filters: Optional[FilterCriteria] = None
    ) -> SearchResult:
        """
        Perform hybrid search combining dense and BM25 results.
        
        Args:
            query: User query string
            project_id: Project identifier
            project_context: Optional project context for enhancement
            k: Number of results to return
            filters: Optional search filters
            
        Returns:
            SearchResult with fused results
        """
        import time
        start_time = time.time()
        
        # Get more results from each method for better fusion
        search_k = k * 2
        
        # Perform dense semantic search
        dense_result = self.dense_search.search(
            query=query,
            project_id=project_id,
            project_context=project_context,
            k=search_k,
            filters=filters
        )
        
        # Perform BM25 keyword search
        bm25_filters = self._convert_filters_for_bm25(filters)
        bm25_result = self.bm25_index.search(
            query=query,
            project_id=project_id,
            k=search_k,
            filters=bm25_filters
        )
        
        # Enrich BM25 results with full chunk data from vector store
        enriched_bm25_hits = self._enrich_bm25_hits(bm25_result.hits)
        
        # Fuse results using RRF
        fused_hits = self.rank_fusion.fuse_results(
            dense_hits=dense_result.hits,
            bm25_hits=enriched_bm25_hits,
            alpha=0.6  # Slightly favor dense search
        )
        
        # Apply post-processing and limit results
        final_hits = self._post_process_hybrid_hits(fused_hits, k * 2)  # Get more for reranking
        
        # Apply reranking if enabled
        if self.use_reranker and final_hits:
            rerank_result = self.reranker.rerank(query, final_hits, k)
            final_hits = rerank_result.hits
            logger.debug(f"Reranking completed in {rerank_result.rerank_time:.3f}s using {rerank_result.model_used}")
        else:
            # Apply relevance scoring if not reranking
            if final_hits:
                scored_hits = self.relevance_scorer.score_hits(final_hits, query)
                final_hits = scored_hits[:k]
        
        search_time = time.time() - start_time
        
        logger.info(
            f"Hybrid search completed: {len(final_hits)} results "
            f"(dense: {len(dense_result.hits)}, bm25: {len(bm25_result.hits)}) "
            f"in {search_time:.3f}s"
        )
        
        return SearchResult(
            hits=final_hits,
            total_found=len(fused_hits),
            query_enhanced=dense_result.query_enhanced,
            search_time=search_time
        )
    
    def _convert_filters_for_bm25(self, filters: Optional[FilterCriteria]) -> Optional[Dict[str, Any]]:
        """
        Convert FilterCriteria to format expected by BM25 search.
        
        Args:
            filters: FilterCriteria object
            
        Returns:
            Dictionary format for BM25 search
        """
        if not filters:
            return None
        
        # Use metadata filter to build BM25 filter
        return self.metadata_filter.build_bm25_filter(filters)
    
    def _enrich_bm25_hits(self, bm25_hits: List[Hit]) -> List[Hit]:
        """
        Enrich BM25 hits with full chunk data from vector store.
        
        Args:
            bm25_hits: Hits from BM25 search with minimal chunk data
            
        Returns:
            Enriched hits with full chunk data
        """
        if not bm25_hits:
            return []
        
        enriched_hits = []
        
        for hit in bm25_hits:
            try:
                # Get full chunk data from vector store by ID
                full_chunks = self.dense_search.vector_store.get_by_ids([hit["id"]])
                
                if full_chunks:
                    # Update hit with full chunk data
                    enriched_hit = hit.copy()
                    enriched_hit["chunk"] = full_chunks[0]
                    enriched_hits.append(enriched_hit)
                else:
                    # Keep original hit if chunk not found in vector store
                    logger.warning(f"Chunk {hit['id']} not found in vector store")
                    enriched_hits.append(hit)
                    
            except Exception as e:
                logger.warning(f"Error enriching BM25 hit {hit['id']}: {e}")
                # Keep original hit on error
                enriched_hits.append(hit)
        
        return enriched_hits
    
    def _post_process_hybrid_hits(self, hits: List[Hit], k: int) -> List[Hit]:
        """
        Post-process hybrid search hits for deduplication and final ranking.
        
        Args:
            hits: Fused hits from hybrid search
            k: Target number of results
            
        Returns:
            Post-processed hits
        """
        if not hits:
            return []
        
        # Deduplicate by text hash (exact content matches)
        seen_hashes = set()
        deduplicated = []
        
        for hit in hits:
            # Use chunk ID as fallback if text_hash not available
            dedup_key = hit["chunk"].get("text_hash", hit["id"])
            
            if dedup_key not in seen_hashes:
                seen_hashes.add(dedup_key)
                deduplicated.append(hit)
        
        # Apply source diversification
        diversified = self.dense_search._diversify_sources(deduplicated, k)
        
        return diversified[:k]
    
    def index_chunks_for_bm25(self, chunks: List[Chunk], project_id: str) -> None:
        """
        Index chunks in BM25 search index.
        
        Args:
            chunks: List of chunks to index
            project_id: Project identifier
        """
        if not self.bm25_index:
            self.bm25_index = create_bm25_index(project_id)
        
        self.bm25_index.index_chunks(chunks)
        logger.info(f"Indexed {len(chunks)} chunks in BM25 index for project {project_id}")
    
    def delete_project_from_bm25(self, project_id: str) -> None:
        """
        Delete project data from BM25 index.
        
        Args:
            project_id: Project identifier
        """
        if self.bm25_index:
            self.bm25_index.delete_project(project_id)
            logger.info(f"Deleted project {project_id} from BM25 index")
    
    def retrieve_with_context_window(
        self,
        query: str,
        project_id: str,
        project_context: Optional[ProjectContext] = None,
        k: Optional[int] = None,
        filters: Optional[FilterCriteria] = None,
        window_size: int = 1
    ) -> SearchResult:
        """
        Retrieve chunks with sliding window context expansion.
        
        Args:
            query: User query string
            project_id: Project identifier
            project_context: Optional project context for enhancement
            k: Number of results to return
            filters: Optional search filters
            window_size: Number of adjacent chunks to include
            
        Returns:
            SearchResult with expanded context chunks
        """
        # Get initial results
        result = self.retrieve(
            query=query,
            project_id=project_id,
            project_context=project_context,
            k=k,
            filters=filters
        )
        
        if not result.hits or window_size <= 0:
            return result
        
        # TODO: Implement sliding window context expansion
        # This would require additional vector store methods to get adjacent chunks
        # For now, return the original results
        logger.info("Context window expansion not yet implemented")
        
        return result
    
    def validate_filters(self, filters: FilterCriteria) -> List[str]:
        """
        Validate filter criteria.
        
        Args:
            filters: Filter criteria to validate
            
        Returns:
            List of validation error messages
        """
        return self.metadata_filter.validate_criteria(filters)
    
    def suggest_filters(self, query: str) -> FilterCriteria:
        """
        Suggest filter criteria based on query content.
        
        Args:
            query: User query string
            
        Returns:
            Suggested filter criteria
        """
        return self.metadata_filter.suggest_filters(query)
    
    def get_available_filter_values(self, project_id: str, field: str) -> List[str]:
        """
        Get available values for a specific filter field.
        
        Args:
            project_id: Project identifier
            field: Filter field name
            
        Returns:
            List of available values
        """
        return self.metadata_filter.get_available_values(project_id, field)
    
    def get_division_info(self, division_code: Optional[str] = None) -> Union[str, Dict[str, str]]:
        """
        Get division information.
        
        Args:
            division_code: Specific division code (returns title) or None (returns all)
            
        Returns:
            Division title for specific code, or all divisions dict
        """
        if division_code:
            return self.metadata_filter.get_division_title(division_code)
        else:
            return self.metadata_filter.get_all_divisions()


def create_hybrid_retriever(project_id: Optional[str] = None, **kwargs) -> HybridRetriever:
    """
    Convenience function to create a hybrid retriever.
    
    Args:
        project_id: Project ID for BM25 index initialization
        **kwargs: Arguments passed to HybridRetriever constructor
        
    Returns:
        HybridRetriever instance
    """
    return HybridRetriever(project_id=project_id, **kwargs)