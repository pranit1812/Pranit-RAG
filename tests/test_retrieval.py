"""
Tests for the retrieval system.
"""
import pytest
from unittest.mock import Mock, patch
from typing import List

from src.models.types import Hit, Chunk, ChunkMetadata, ProjectContext
from src.services.retrieval import (
    DenseSemanticSearch, HybridRetriever, SearchFilters, SearchResult,
    RelevanceScorer, RankFusion, create_hybrid_retriever
)


@pytest.fixture
def sample_chunk_metadata():
    """Sample chunk metadata for testing."""
    return ChunkMetadata(
        project_id="test_project",
        doc_id="doc_1",
        doc_name="test_spec.pdf",
        file_type="pdf",
        page_start=1,
        page_end=1,
        content_type="SpecSection",
        division_code="23",
        division_title="HVAC",
        section_code="23 05 00",
        section_title="Common Work Results for HVAC",
        discipline="M",
        sheet_number=None,
        sheet_title=None,
        bbox_regions=[[0, 0, 100, 100]],
        low_conf=False
    )


@pytest.fixture
def sample_chunk(sample_chunk_metadata):
    """Sample chunk for testing."""
    return Chunk(
        id="chunk_1",
        text="HVAC system requirements for mechanical ventilation",
        html=None,
        metadata=sample_chunk_metadata,
        token_count=10,
        text_hash="abc123"
    )


@pytest.fixture
def sample_hit(sample_chunk):
    """Sample hit for testing."""
    return Hit(
        id="chunk_1",
        score=0.85,
        chunk=sample_chunk
    )


@pytest.fixture
def sample_project_context():
    """Sample project context for testing."""
    return ProjectContext(
        project_name="Test Office Building",
        description="A commercial office building project",
        project_type="Commercial Office Building",
        location="New York, NY",
        key_systems=["HVAC", "Electrical", "Plumbing"],
        disciplines_involved=["Mechanical", "Electrical", "Plumbing"],
        summary="Commercial office building with HVAC, electrical, and plumbing systems"
    )


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service."""
    mock = Mock()
    mock.embed_query.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    return mock


@pytest.fixture
def mock_vector_store():
    """Mock vector store."""
    mock = Mock()
    return mock


class TestSearchFilters:
    """Test SearchFilters functionality."""
    
    def test_search_filters_creation(self):
        """Test creating search filters."""
        filters = SearchFilters(
            content_types=["SpecSection", "Table"],
            division_codes=["23", "26"],
            disciplines=["M", "E"]
        )
        
        assert filters.content_types == ["SpecSection", "Table"]
        assert filters.division_codes == ["23", "26"]
        assert filters.disciplines == ["M", "E"]
        assert filters.low_conf_only is None
        assert filters.doc_names is None
    
    def test_search_filters_defaults(self):
        """Test search filters with default values."""
        filters = SearchFilters()
        
        assert filters.content_types is None
        assert filters.division_codes is None
        assert filters.disciplines is None
        assert filters.low_conf_only is None
        assert filters.doc_names is None


class TestDenseSemanticSearch:
    """Test DenseSemanticSearch functionality."""
    
    def test_initialization(self, mock_embedding_service, mock_vector_store):
        """Test dense search initialization."""
        search = DenseSemanticSearch(
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store
        )
        
        assert search.embedding_service == mock_embedding_service
        assert search.vector_store == mock_vector_store
        assert search.query_enhancer is not None
    
    def test_build_where_clause_basic(self, mock_embedding_service, mock_vector_store):
        """Test building basic where clause."""
        search = DenseSemanticSearch(
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store
        )
        
        where = search._build_where_clause("test_project", None)
        
        assert where == {"project_id": "test_project"}
    
    def test_build_where_clause_with_filters(self, mock_embedding_service, mock_vector_store):
        """Test building where clause with filters."""
        search = DenseSemanticSearch(
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store
        )
        
        filters = SearchFilters(
            content_types=["SpecSection"],
            division_codes=["23"],
            disciplines=["M"],
            low_conf_only=False
        )
        
        where = search._build_where_clause("test_project", filters)
        
        expected = {
            "project_id": "test_project",
            "content_type": "SpecSection",  # Single value, no $in
            "division_code": "23",  # Single value, no $in
            "discipline": "M",  # Single value, no $in
            "low_conf": "false"
        }
        
        assert where == expected
    
    def test_post_process_hits_deduplication(self, mock_embedding_service, mock_vector_store, sample_chunk):
        """Test hit deduplication."""
        search = DenseSemanticSearch(
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store
        )
        
        # Create duplicate hits with same text hash
        hit1 = Hit(id="chunk_1", score=0.8, chunk=sample_chunk)
        hit2 = Hit(id="chunk_2", score=0.9, chunk=sample_chunk)  # Same chunk
        
        hits = [hit1, hit2]
        processed = search._post_process_hits(hits, 5)
        
        # Should deduplicate to single hit
        assert len(processed) == 1
        assert processed[0]["id"] == "chunk_1"  # First one kept
    
    def test_diversify_sources(self, mock_embedding_service, mock_vector_store, sample_chunk_metadata):
        """Test source diversification."""
        search = DenseSemanticSearch(
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store
        )
        
        # Create hits from different documents
        chunk1 = Chunk(
            id="chunk_1", text="text1", html=None, metadata={**sample_chunk_metadata, "doc_name": "doc1.pdf"},
            token_count=10, text_hash="hash1"
        )
        chunk2 = Chunk(
            id="chunk_2", text="text2", html=None, metadata={**sample_chunk_metadata, "doc_name": "doc1.pdf"},
            token_count=10, text_hash="hash2"
        )
        chunk3 = Chunk(
            id="chunk_3", text="text3", html=None, metadata={**sample_chunk_metadata, "doc_name": "doc2.pdf"},
            token_count=10, text_hash="hash3"
        )
        
        hits = [
            Hit(id="chunk_1", score=0.9, chunk=chunk1),
            Hit(id="chunk_2", score=0.8, chunk=chunk2),
            Hit(id="chunk_3", score=0.7, chunk=chunk3)
        ]
        
        diversified = search._diversify_sources(hits, 2)
        
        # Should prefer different documents
        assert len(diversified) == 2
        doc_names = {hit["chunk"]["metadata"]["doc_name"] for hit in diversified}
        assert len(doc_names) == 2  # Two different documents
    
    @patch('src.services.retrieval.QueryEnhancer')
    def test_search_with_context(self, mock_query_enhancer, mock_embedding_service, 
                                mock_vector_store, sample_project_context, sample_hit):
        """Test search with project context enhancement."""
        # Setup mocks
        mock_enhancer_instance = Mock()
        mock_enhancer_instance.enhance_query.return_value = "enhanced query"
        mock_query_enhancer.return_value = mock_enhancer_instance
        
        mock_vector_store.query.return_value = [sample_hit]
        
        search = DenseSemanticSearch(
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store
        )
        
        result = search.search(
            query="test query",
            project_id="test_project",
            project_context=sample_project_context,
            k=5
        )
        
        # Verify query enhancement was called
        mock_enhancer_instance.enhance_query.assert_called_once_with("test query", sample_project_context)
        
        # Verify embedding service was called with enhanced query
        mock_embedding_service.embed_query.assert_called_once_with("enhanced query")
        
        # Verify result
        assert isinstance(result, SearchResult)
        assert len(result.hits) == 1
        assert result.query_enhanced == "enhanced query"
        assert result.total_found == 1


class TestRelevanceScorer:
    """Test RelevanceScorer functionality."""
    
    def test_initialization(self):
        """Test relevance scorer initialization."""
        scorer = RelevanceScorer()
        
        assert scorer.content_type_weights["SpecSection"] == 1.0
        assert scorer.content_type_weights["Table"] == 0.9
        assert scorer.confidence_weight == 0.1
    
    def test_score_hits(self, sample_chunk_metadata):
        """Test hit scoring."""
        scorer = RelevanceScorer()
        
        # Create hits with different content types and confidence levels
        chunk1 = Chunk(
            id="chunk_1", text="text1", html=None,
            metadata={**sample_chunk_metadata, "content_type": "SpecSection", "low_conf": False},
            token_count=10, text_hash="hash1"
        )
        chunk2 = Chunk(
            id="chunk_2", text="text2", html=None,
            metadata={**sample_chunk_metadata, "content_type": "Table", "low_conf": True},
            token_count=10, text_hash="hash2"
        )
        
        hits = [
            Hit(id="chunk_1", score=0.2, chunk=chunk1),  # Lower distance = higher similarity
            Hit(id="chunk_2", score=0.3, chunk=chunk2)
        ]
        
        scored_hits = scorer.score_hits(hits, "test query")
        
        # Should be sorted by relevance score (higher is better)
        assert len(scored_hits) == 2
        assert scored_hits[0]["id"] == "chunk_1"  # SpecSection + high conf should score higher
        assert scored_hits[0]["score"] > scored_hits[1]["score"]


class TestRankFusion:
    """Test RankFusion functionality."""
    
    def test_initialization(self):
        """Test rank fusion initialization."""
        fusion = RankFusion(k=60)
        assert fusion.k == 60
    
    def test_fuse_results(self, sample_chunk_metadata):
        """Test result fusion with RRF."""
        fusion = RankFusion(k=60)
        
        # Create sample chunks
        chunk1 = Chunk(
            id="chunk_1", text="text1", html=None, metadata=sample_chunk_metadata,
            token_count=10, text_hash="hash1"
        )
        chunk2 = Chunk(
            id="chunk_2", text="text2", html=None, metadata=sample_chunk_metadata,
            token_count=10, text_hash="hash2"
        )
        chunk3 = Chunk(
            id="chunk_3", text="text3", html=None, metadata=sample_chunk_metadata,
            token_count=10, text_hash="hash3"
        )
        
        # Dense results (chunk_1 ranked first)
        dense_hits = [
            Hit(id="chunk_1", score=0.9, chunk=chunk1),
            Hit(id="chunk_2", score=0.7, chunk=chunk2)
        ]
        
        # BM25 results (chunk_3 only appears here, chunk_2 ranked first)
        bm25_hits = [
            Hit(id="chunk_2", score=2.5, chunk=chunk2),
            Hit(id="chunk_3", score=1.8, chunk=chunk3)
        ]
        
        fused_results = fusion.fuse_results(dense_hits, bm25_hits, alpha=0.5)
        
        # Should combine all unique chunks
        assert len(fused_results) == 3
        
        # Results should be sorted by combined RRF score (higher is better)
        for i in range(len(fused_results) - 1):
            assert fused_results[i]["score"] >= fused_results[i + 1]["score"]
        
        # All chunks should be present
        chunk_ids = {hit["id"] for hit in fused_results}
        assert chunk_ids == {"chunk_1", "chunk_2", "chunk_3"}
    
    def test_fuse_results_with_overlap(self, sample_chunk_metadata):
        """Test fusion with overlapping results."""
        fusion = RankFusion(k=60)
        
        chunk1 = Chunk(
            id="chunk_1", text="text1", html=None, metadata=sample_chunk_metadata,
            token_count=10, text_hash="hash1"
        )
        
        # Same chunk appears in both result sets
        dense_hits = [Hit(id="chunk_1", score=0.9, chunk=chunk1)]
        bm25_hits = [Hit(id="chunk_1", score=2.5, chunk=chunk1)]
        
        fused_results = fusion.fuse_results(dense_hits, bm25_hits, alpha=0.5)
        
        # Should have only one result (deduplicated)
        assert len(fused_results) == 1
        assert fused_results[0]["id"] == "chunk_1"
        
        # Score should be combination of both rankings
        assert fused_results[0]["score"] > 0


class TestHybridRetriever:
    """Test HybridRetriever functionality."""
    
    def test_initialization(self, mock_embedding_service, mock_vector_store):
        """Test hybrid retriever initialization."""
        retriever = HybridRetriever(
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store
        )
        
        assert retriever.dense_search is not None
        assert retriever.relevance_scorer is not None
        assert retriever.rank_fusion is not None
        assert retriever.bm25_index is None  # No project_id provided
    
    @patch('src.services.retrieval.create_bm25_index')
    def test_initialization_with_project_id(self, mock_create_bm25_index, 
                                          mock_embedding_service, mock_vector_store):
        """Test hybrid retriever initialization with project ID."""
        mock_bm25_index = Mock()
        mock_create_bm25_index.return_value = mock_bm25_index
        
        retriever = HybridRetriever(
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
            project_id="test_project"
        )
        
        assert retriever.bm25_index == mock_bm25_index
        mock_create_bm25_index.assert_called_once_with("test_project")
    
    @patch('src.services.retrieval.get_config')
    def test_retrieve_dense_only(self, mock_get_config, mock_embedding_service, mock_vector_store, 
                                sample_project_context, sample_hit):
        """Test retrieval with dense search only (no BM25)."""
        # Mock config
        mock_config = Mock()
        mock_config.retrieve.top_k = 5
        mock_config.retrieve.hybrid = False
        mock_config.retrieve.reranker = "none"
        mock_get_config.return_value = mock_config
        
        # Mock dense search
        mock_vector_store.query.return_value = [sample_hit]
        
        retriever = HybridRetriever(
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store
        )
        
        result = retriever.retrieve(
            query="test query",
            project_id="test_project",
            project_context=sample_project_context
        )
        
        assert isinstance(result, SearchResult)
        assert len(result.hits) == 1
        assert result.hits[0]["id"] == "chunk_1"
    
    @patch('src.services.retrieval.get_config')
    @patch('src.services.retrieval.create_bm25_index')
    def test_retrieve_hybrid(self, mock_create_bm25_index, mock_get_config, 
                           mock_embedding_service, mock_vector_store, 
                           sample_project_context, sample_hit, sample_chunk_metadata):
        """Test hybrid retrieval with both dense and BM25."""
        # Mock config
        mock_config = Mock()
        mock_config.retrieve.top_k = 5
        mock_config.retrieve.hybrid = True
        mock_config.retrieve.reranker = "none"
        mock_get_config.return_value = mock_config
        
        # Mock BM25 index
        mock_bm25_index = Mock()
        mock_bm25_result = Mock()
        mock_bm25_result.hits = [sample_hit]
        mock_bm25_index.search.return_value = mock_bm25_result
        mock_create_bm25_index.return_value = mock_bm25_index
        
        # Mock dense search
        mock_vector_store.query.return_value = [sample_hit]
        mock_vector_store.get_by_ids.return_value = [sample_hit["chunk"]]
        
        retriever = HybridRetriever(
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store,
            project_id="test_project"
        )
        
        result = retriever.retrieve(
            query="test query",
            project_id="test_project",
            project_context=sample_project_context,
            use_hybrid=True
        )
        
        assert isinstance(result, SearchResult)
        # Should call both dense and BM25 search
        mock_vector_store.query.assert_called_once()
        mock_bm25_index.search.assert_called_once()
    
    def test_convert_filters_for_bm25(self, mock_embedding_service, mock_vector_store):
        """Test filter conversion for BM25 search."""
        retriever = HybridRetriever(
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store
        )
        
        filters = SearchFilters(
            content_types=["SpecSection"],
            division_codes=["23"],
            disciplines=["M"],
            low_conf_only=False
        )
        
        bm25_filters = retriever._convert_filters_for_bm25(filters)
        
        expected = {
            "content_types": ["SpecSection"],
            "division_codes": ["23"],
            "disciplines": ["M"],
            "low_conf_only": False
        }
        
        assert bm25_filters == expected
    
    def test_convert_filters_for_bm25_none(self, mock_embedding_service, mock_vector_store):
        """Test filter conversion when no filters provided."""
        retriever = HybridRetriever(
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store
        )
        
        bm25_filters = retriever._convert_filters_for_bm25(None)
        assert bm25_filters is None
    
    def test_enrich_bm25_hits(self, mock_embedding_service, mock_vector_store, sample_hit):
        """Test enriching BM25 hits with full chunk data."""
        mock_vector_store.get_by_ids.return_value = [sample_hit["chunk"]]
        
        retriever = HybridRetriever(
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store
        )
        
        # Create minimal BM25 hit
        bm25_hit = Hit(
            id="chunk_1",
            score=2.5,
            chunk=Chunk(
                id="chunk_1",
                text="",  # Empty in BM25 result
                html=None,
                metadata=sample_hit["chunk"]["metadata"],
                token_count=0,
                text_hash=""
            )
        )
        
        enriched_hits = retriever._enrich_bm25_hits([bm25_hit])
        
        assert len(enriched_hits) == 1
        assert enriched_hits[0]["chunk"]["text"] == sample_hit["chunk"]["text"]
        mock_vector_store.get_by_ids.assert_called_once_with(["chunk_1"])
    
    @patch('src.services.retrieval.create_bm25_index')
    def test_index_chunks_for_bm25(self, mock_create_bm25_index, 
                                  mock_embedding_service, mock_vector_store, sample_chunk):
        """Test indexing chunks for BM25."""
        mock_bm25_index = Mock()
        mock_create_bm25_index.return_value = mock_bm25_index
        
        retriever = HybridRetriever(
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store
        )
        
        retriever.index_chunks_for_bm25([sample_chunk], "test_project")
        
        # Should create BM25 index and index chunks
        mock_create_bm25_index.assert_called_once_with("test_project")
        mock_bm25_index.index_chunks.assert_called_once_with([sample_chunk])
    
    def test_retrieve_with_context_window(self, mock_embedding_service, mock_vector_store, 
                                        sample_project_context, sample_hit):
        """Test retrieval with context window (placeholder implementation)."""
        mock_vector_store.query.return_value = [sample_hit]
        
        retriever = HybridRetriever(
            embedding_service=mock_embedding_service,
            vector_store=mock_vector_store
        )
        
        result = retriever.retrieve_with_context_window(
            query="test query",
            project_id="test_project",
            project_context=sample_project_context,
            window_size=1
        )
        
        # For now, should return same as regular retrieve
        assert isinstance(result, SearchResult)
        assert len(result.hits) == 1


class TestFactoryFunction:
    """Test factory function."""
    
    @patch('src.services.retrieval.create_embedding_service')
    @patch('src.services.retrieval.create_vector_store')
    def test_create_hybrid_retriever(self, mock_create_vector_store, mock_create_embedding_service):
        """Test hybrid retriever factory function."""
        mock_embedding_service = Mock()
        mock_vector_store = Mock()
        mock_create_embedding_service.return_value = mock_embedding_service
        mock_create_vector_store.return_value = mock_vector_store
        
        retriever = create_hybrid_retriever()
        
        assert isinstance(retriever, HybridRetriever)
        mock_create_embedding_service.assert_called_once()
        mock_create_vector_store.assert_called_once()


class TestIntegration:
    """Integration tests for retrieval system."""
    
    @patch('src.services.retrieval.create_embedding_service')
    @patch('src.services.retrieval.create_vector_store')
    @patch('src.services.retrieval.get_config')
    def test_end_to_end_retrieval(self, mock_get_config, mock_create_vector_store, 
                                 mock_create_embedding_service, sample_project_context, 
                                 sample_hit):
        """Test end-to-end retrieval flow."""
        # Setup mocks
        mock_config = Mock()
        mock_config.retrieve.top_k = 5
        mock_config.retrieve.reranker = "none"
        mock_get_config.return_value = mock_config
        
        mock_embedding_service = Mock()
        mock_embedding_service.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_create_embedding_service.return_value = mock_embedding_service
        
        mock_vector_store = Mock()
        mock_vector_store.query.return_value = [sample_hit]
        mock_create_vector_store.return_value = mock_vector_store
        
        # Create retriever and perform search
        retriever = create_hybrid_retriever()
        
        result = retriever.retrieve(
            query="HVAC system requirements",
            project_id="test_project",
            project_context=sample_project_context,
            k=5,
            filters=SearchFilters(content_types=["SpecSection"])
        )
        
        # Verify the flow
        assert isinstance(result, SearchResult)
        assert len(result.hits) == 1
        assert result.hits[0]["chunk"]["text"] == "HVAC system requirements for mechanical ventilation"
        
        # Verify embedding service was called
        mock_embedding_service.embed_query.assert_called_once()
        
        # Verify vector store was called with correct parameters
        mock_vector_store.query.assert_called_once()
        call_args = mock_vector_store.query.call_args
        assert call_args[1]["k"] == 10  # k*2 for deduplication
        assert call_args[1]["where"]["project_id"] == "test_project"
        assert call_args[1]["where"]["content_type"] == "SpecSection"  # Single value, no $in