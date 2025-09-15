"""
Tests for embedding service and vector store integration.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from src.services.embedding import EmbeddingService, EmbeddingResult, EmbeddingServiceFactory
from src.services.vector_store import VectorStore, ChromaVectorStore, VectorStoreFactory
from src.services.indexing import IndexingPipeline, ChunkCache
from src.models.types import Chunk, ChunkMetadata, Hit


class MockEmbeddingService(EmbeddingService):
    """Mock embedding service for testing."""
    
    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions
        self.model_name = "mock-model"
    
    def embed_texts(self, texts):
        # Return mock embeddings (just repeated values for simplicity)
        embeddings = []
        for i, text in enumerate(texts):
            # Create a simple embedding based on text hash
            embedding = [float(hash(text) % 1000) / 1000.0] * self.dimensions
            embeddings.append(embedding)
        
        return EmbeddingResult(
            embeddings=embeddings,
            model=self.model_name,
            dimensions=self.dimensions,
            total_tokens=len(texts) * 10
        )
    
    def embed_query(self, query):
        result = self.embed_texts([query])
        return result.embeddings[0]
    
    def get_dimensions(self):
        return self.dimensions
    
    def get_model_name(self):
        return self.model_name


def create_test_chunk(chunk_id: str, text: str, project_id: str = "test_project") -> Chunk:
    """Create a test chunk."""
    return Chunk(
        id=chunk_id,
        text=text,
        html=None,
        metadata=ChunkMetadata(
            project_id=project_id,
            doc_id="test_doc",
            doc_name="test.pdf",
            file_type="pdf",
            page_start=1,
            page_end=1,
            content_type="SpecSection",
            division_code="09",
            division_title="Finishes",
            section_code=None,
            section_title=None,
            discipline=None,
            sheet_number=None,
            sheet_title=None,
            bbox_regions=[[0, 0, 100, 100]],
            low_conf=False
        ),
        token_count=len(text.split()),
        text_hash=f"hash_{chunk_id}"
    )


class TestEmbeddingService:
    """Test embedding service functionality."""
    
    def test_mock_embedding_service(self):
        """Test mock embedding service."""
        service = MockEmbeddingService()
        
        texts = ["This is a test", "Another test text"]
        result = service.embed_texts(texts)
        
        assert len(result.embeddings) == 2
        assert len(result.embeddings[0]) == 384
        assert result.model == "mock-model"
        assert result.dimensions == 384
    
    def test_embed_query(self):
        """Test single query embedding."""
        service = MockEmbeddingService()
        
        query = "Test query"
        embedding = service.embed_query(query)
        
        assert len(embedding) == 384
        assert isinstance(embedding[0], float)
    
    @patch('src.services.embedding.OPENAI_AVAILABLE', False)
    def test_factory_without_openai(self):
        """Test factory when OpenAI is not available."""
        with pytest.raises(ImportError):
            EmbeddingServiceFactory.create_service(provider="openai")
    
    @patch('src.services.embedding.SENTENCE_TRANSFORMERS_AVAILABLE', False)
    def test_factory_without_sentence_transformers(self):
        """Test factory when SentenceTransformers is not available."""
        with pytest.raises(ImportError):
            EmbeddingServiceFactory.create_service(provider="local")


class TestVectorStore:
    """Test vector store functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.vector_store = None
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.services.vector_store.CHROMA_AVAILABLE', False)
    def test_chroma_not_available(self):
        """Test error when Chroma is not available."""
        with pytest.raises(ImportError):
            ChromaVectorStore(self.temp_dir)
    
    @patch('src.services.vector_store.CHROMA_AVAILABLE', True)
    @patch('src.services.vector_store.chromadb')
    def test_vector_store_creation(self, mock_chromadb):
        """Test vector store creation."""
        mock_client = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        
        store = ChromaVectorStore(self.temp_dir)
        assert store.data_dir == Path(self.temp_dir)
        assert store.collection_prefix == "construction_rag"
    
    @patch('src.services.vector_store.CHROMA_AVAILABLE', True)
    @patch('src.services.vector_store.chromadb')
    def test_upsert_chunks(self, mock_chromadb):
        """Test upserting chunks."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_collection.side_effect = Exception("Not found")
        mock_client.create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client
        
        store = ChromaVectorStore(self.temp_dir)
        
        chunks = [
            create_test_chunk("chunk1", "Test text 1"),
            create_test_chunk("chunk2", "Test text 2")
        ]
        
        store.upsert_chunks(chunks)
        
        # Verify collection was created and upsert was called
        mock_client.create_collection.assert_called_once()
        mock_collection.upsert.assert_called_once()
    
    @patch('src.services.vector_store.CHROMA_AVAILABLE', True)
    @patch('src.services.vector_store.chromadb')
    def test_query_vector_store(self, mock_chromadb):
        """Test querying vector store."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client
        
        # Mock query results
        mock_collection.query.return_value = {
            "ids": [["chunk1", "chunk2"]],
            "documents": [["Test text 1", "Test text 2"]],
            "distances": [[0.1, 0.2]],
            "metadatas": [[
                {
                    "project_id": "test_project",
                    "doc_id": "test_doc",
                    "doc_name": "test.pdf",
                    "file_type": "pdf",
                    "page_start": "1",
                    "page_end": "1",
                    "content_type": "SpecSection",
                    "division_code": "09",
                    "division_title": "Finishes",
                    "section_code": "",
                    "section_title": "",
                    "discipline": "",
                    "sheet_number": "",
                    "sheet_title": "",
                    "bbox_regions": "[[0, 0, 100, 100]]",
                    "low_conf": "false",
                    "chunk_id": "chunk1",
                    "text_hash": "hash_chunk1",
                    "token_count": "3",
                    "has_html": "false"
                },
                {
                    "project_id": "test_project",
                    "doc_id": "test_doc",
                    "doc_name": "test.pdf",
                    "file_type": "pdf",
                    "page_start": "1",
                    "page_end": "1",
                    "content_type": "SpecSection",
                    "division_code": "09",
                    "division_title": "Finishes",
                    "section_code": "",
                    "section_title": "",
                    "discipline": "",
                    "sheet_number": "",
                    "sheet_title": "",
                    "bbox_regions": "[[0, 0, 100, 100]]",
                    "low_conf": "false",
                    "chunk_id": "chunk2",
                    "text_hash": "hash_chunk2",
                    "token_count": "3",
                    "has_html": "false"
                }
            ]]
        }
        
        store = ChromaVectorStore(self.temp_dir)
        
        query_vector = [0.1] * 384
        hits = store.query(
            vector=query_vector,
            k=2,
            where={"project_id": "test_project"}
        )
        
        assert len(hits) == 2
        assert hits[0]["id"] == "chunk1"
        assert hits[0]["score"] == 0.1
        assert hits[0]["chunk"]["text"] == "Test text 1"


class TestChunkCache:
    """Test chunk cache functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_file = Path(self.temp_dir) / "cache.json"
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cache_operations(self):
        """Test basic cache operations."""
        cache = ChunkCache(self.cache_file)
        
        # Test empty cache
        assert not cache.contains("hash1")
        
        # Test adding hashes
        cache.add("hash1")
        assert cache.contains("hash1")
        
        # Test batch add
        cache.add_batch(["hash2", "hash3"])
        assert cache.contains("hash2")
        assert cache.contains("hash3")
        
        # Test remove
        cache.remove("hash1")
        assert not cache.contains("hash1")
        
        # Test clear
        cache.clear()
        assert not cache.contains("hash2")
        assert not cache.contains("hash3")
    
    def test_cache_persistence(self):
        """Test cache persistence to file."""
        # Create cache and add some hashes
        cache1 = ChunkCache(self.cache_file)
        cache1.add_batch(["hash1", "hash2", "hash3"])
        cache1.save()
        
        # Create new cache instance and verify data is loaded
        cache2 = ChunkCache(self.cache_file)
        assert cache2.contains("hash1")
        assert cache2.contains("hash2")
        assert cache2.contains("hash3")


class TestIndexingPipeline:
    """Test indexing pipeline functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_id = "test_project"
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.services.indexing.create_embedding_service')
    @patch('src.services.indexing.create_vector_store')
    def test_pipeline_creation(self, mock_vector_store, mock_embedding_service):
        """Test pipeline creation."""
        mock_embedding_service.return_value = MockEmbeddingService()
        mock_vector_store.return_value = Mock()
        
        pipeline = IndexingPipeline(
            project_id=self.project_id,
            data_dir=self.temp_dir
        )
        
        assert pipeline.project_id == self.project_id
        assert pipeline.data_dir == Path(self.temp_dir)
        assert pipeline.embedding_service is not None
        assert pipeline.vector_store is not None
    
    def test_progress_tracking(self):
        """Test progress tracking functionality."""
        progress_updates = []
        
        def progress_callback(progress):
            progress_updates.append(progress.current_operation)
        
        with patch('src.services.indexing.create_embedding_service'), \
             patch('src.services.indexing.create_vector_store'):
            
            pipeline = IndexingPipeline(
                project_id=self.project_id,
                data_dir=self.temp_dir,
                progress_callback=progress_callback
            )
            
            pipeline._update_progress(current_operation="Test operation")
            assert "Test operation" in progress_updates
    
    def test_cancellation(self):
        """Test pipeline cancellation."""
        with patch('src.services.indexing.create_embedding_service'), \
             patch('src.services.indexing.create_vector_store'):
            
            pipeline = IndexingPipeline(
                project_id=self.project_id,
                data_dir=self.temp_dir
            )
            
            assert not pipeline.is_cancelled()
            
            pipeline.cancel()
            assert pipeline.is_cancelled()
            assert pipeline.progress.is_cancelled


if __name__ == "__main__":
    pytest.main([__file__])