"""
Tests for BM25 keyword search functionality.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List

from src.services.bm25_search import BM25Index, ConstructionFilter, create_construction_analyzer, create_bm25_index
from src.models.types import Chunk, ChunkMetadata, Hit


@pytest.fixture
def temp_index_dir():
    """Create a temporary directory for BM25 index."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_chunks() -> List[Chunk]:
    """Create sample chunks for testing."""
    chunks = [
        {
            "id": "chunk_1",
            "text": "HVAC system specifications for commercial building. The heating, ventilation, and air conditioning system shall comply with ASHRAE standards.",
            "html": None,
            "metadata": {
                "project_id": "test_project",
                "doc_id": "doc_1",
                "doc_name": "HVAC_Specs.pdf",
                "file_type": "pdf",
                "page_start": 1,
                "page_end": 1,
                "content_type": "SpecSection",
                "division_code": "23",
                "division_title": "Heating, Ventilating and Air Conditioning (HVAC)",
                "section_code": "23 05 00",
                "section_title": "Common Work Results for HVAC",
                "discipline": None,
                "sheet_number": None,
                "sheet_title": None,
                "bbox_regions": [[0, 0, 100, 50]],
                "low_conf": False
            },
            "token_count": 25,
            "text_hash": "hvac_hash_1"
        },
        {
            "id": "chunk_2", 
            "text": "Electrical panel installation requirements. All electrical panels shall be installed according to NEC code requirements.",
            "html": None,
            "metadata": {
                "project_id": "test_project",
                "doc_id": "doc_2",
                "doc_name": "Electrical_Specs.pdf",
                "file_type": "pdf",
                "page_start": 5,
                "page_end": 5,
                "content_type": "SpecSection",
                "division_code": "26",
                "division_title": "Electrical",
                "section_code": "26 24 00",
                "section_title": "Switchboards and Panelboards",
                "discipline": None,
                "sheet_number": None,
                "sheet_title": None,
                "bbox_regions": [[0, 50, 100, 100]],
                "low_conf": False
            },
            "token_count": 18,
            "text_hash": "electrical_hash_1"
        },
        {
            "id": "chunk_3",
            "text": "Concrete foundation details for structural elements. Foundation shall be reinforced concrete with minimum 28-day strength of 4000 psi.",
            "html": None,
            "metadata": {
                "project_id": "test_project",
                "doc_id": "doc_3",
                "doc_name": "Structural_Drawings.pdf",
                "file_type": "pdf",
                "page_start": 10,
                "page_end": 10,
                "content_type": "Drawing",
                "division_code": "03",
                "division_title": "Concrete",
                "section_code": None,
                "section_title": None,
                "discipline": "S",
                "sheet_number": "S-101",
                "sheet_title": "Foundation Plan",
                "bbox_regions": [[0, 100, 100, 150]],
                "low_conf": False
            },
            "token_count": 22,
            "text_hash": "concrete_hash_1"
        },
        {
            "id": "chunk_4",
            "text": "Plumbing fixture schedule and installation requirements. All fixtures shall meet ADA compliance standards.",
            "html": "<table><tr><th>Fixture</th><th>Model</th></tr><tr><td>Sink</td><td>ABC-123</td></tr></table>",
            "metadata": {
                "project_id": "test_project",
                "doc_id": "doc_4",
                "doc_name": "Plumbing_Schedule.pdf",
                "file_type": "pdf",
                "page_start": 3,
                "page_end": 3,
                "content_type": "Table",
                "division_code": "22",
                "division_title": "Plumbing",
                "section_code": "22 40 00",
                "section_title": "Plumbing Fixtures",
                "discipline": None,
                "sheet_number": None,
                "sheet_title": None,
                "bbox_regions": [[0, 150, 100, 200]],
                "low_conf": False
            },
            "token_count": 16,
            "text_hash": "plumbing_hash_1"
        }
    ]
    return chunks


class TestConstructionAnalyzer:
    """Test the construction-specific text analyzer."""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = create_construction_analyzer()
        assert analyzer is not None
    
    def test_construction_filter_initialization(self):
        """Test construction filter initialization."""
        filter_obj = ConstructionFilter()
        assert filter_obj is not None
        assert "hvac" in filter_obj.construction_terms
        assert "dwg" in filter_obj.abbreviations
    
    def test_abbreviation_expansion(self):
        """Test that construction abbreviations are expanded."""
        from whoosh.analysis import Token
        
        filter_obj = ConstructionFilter()
        
        # Create mock tokens
        tokens = [
            Token("see"),
            Token("dwg"),
            Token("for"),
            Token("spec"),
            Token("details")
        ]
        
        # Process tokens through filter
        processed_tokens = list(filter_obj(tokens))
        token_texts = [token.text for token in processed_tokens]
        
        # Should expand "dwg" to "drawing" and "spec" to "specification"
        assert "drawing" in token_texts
        assert "specification" in token_texts
    
    def test_analyzer_integration(self):
        """Test full analyzer with construction filter."""
        analyzer = create_construction_analyzer()
        
        # Test with construction terms and abbreviations
        text = "See dwg for HVAC spec details"
        tokens = list(analyzer(text))
        token_texts = [token.text for token in tokens]
        
        # Should have processed the text
        assert len(token_texts) > 0
        # Should contain some form of the words (may be stemmed)
        text_joined = " ".join(token_texts)
        assert any(word in text_joined for word in ["drawing", "specification", "hvac"])


class TestBM25Index:
    """Test the BM25 index functionality."""
    
    def test_index_initialization(self, temp_index_dir):
        """Test BM25 index initialization."""
        index = BM25Index(temp_index_dir)
        assert index is not None
        assert index.index_dir == Path(temp_index_dir)
        assert index._index is not None
    
    def test_index_chunks(self, temp_index_dir, sample_chunks):
        """Test indexing chunks."""
        index = BM25Index(temp_index_dir)
        try:
            # Index the sample chunks
            index.index_chunks(sample_chunks)
            
            # Verify index stats
            stats = index.get_stats()
            assert stats["document_count"] == len(sample_chunks)
            assert "chunk_id" in stats["field_names"]
            assert "searchable_text" in stats["field_names"]
        finally:
            index.close()
    
    def test_basic_search(self, temp_index_dir, sample_chunks):
        """Test basic keyword search."""
        index = BM25Index(temp_index_dir)
        try:
            index.index_chunks(sample_chunks)
            
            # Search for HVAC-related content
            result = index.search("HVAC heating ventilation", "test_project", k=5)
            
            assert len(result.hits) > 0
            assert result.total_found > 0
            assert result.search_time > 0
            
            # First result should be the HVAC chunk
            first_hit = result.hits[0]
            assert first_hit["id"] == "chunk_1"
            assert first_hit["score"] > 0
            assert first_hit["chunk"]["metadata"]["division_code"] == "23"
        finally:
            index.close()
    
    def test_search_with_filters(self, temp_index_dir, sample_chunks):
        """Test search with metadata filters."""
        index = BM25Index(temp_index_dir)
        index.index_chunks(sample_chunks)
        
        # Search with content type filter
        filters = {"content_types": ["SpecSection"]}
        result = index.search("system requirements", "test_project", k=5, filters=filters)
        
        # Should only return SpecSection chunks
        for hit in result.hits:
            assert hit["chunk"]["metadata"]["content_type"] == "SpecSection"
    
    def test_search_with_division_filter(self, temp_index_dir, sample_chunks):
        """Test search with division code filter."""
        index = BM25Index(temp_index_dir)
        index.index_chunks(sample_chunks)
        
        # Search with division filter
        filters = {"division_codes": ["26"]}
        result = index.search("electrical", "test_project", k=5, filters=filters)
        
        # Should only return electrical division chunks
        for hit in result.hits:
            assert hit["chunk"]["metadata"]["division_code"] == "26"
    
    def test_search_with_discipline_filter(self, temp_index_dir, sample_chunks):
        """Test search with discipline filter."""
        index = BM25Index(temp_index_dir)
        index.index_chunks(sample_chunks)
        
        # Search with discipline filter
        filters = {"disciplines": ["S"]}
        result = index.search("foundation concrete", "test_project", k=5, filters=filters)
        
        # Should only return structural discipline chunks
        for hit in result.hits:
            assert hit["chunk"]["metadata"]["discipline"] == "S"
    
    def test_html_content_search(self, temp_index_dir, sample_chunks):
        """Test that HTML content is searchable."""
        index = BM25Index(temp_index_dir)
        index.index_chunks(sample_chunks)
        
        # Search for content that appears in HTML table
        result = index.search("fixture model", "test_project", k=5)
        
        # Should find the plumbing chunk with HTML table
        found_plumbing = False
        for hit in result.hits:
            if hit["id"] == "chunk_4":
                found_plumbing = True
                break
        
        assert found_plumbing, "Should find chunk with HTML table content"
    
    def test_metadata_text_search(self, temp_index_dir, sample_chunks):
        """Test that metadata text fields are searchable."""
        index = BM25Index(temp_index_dir)
        index.index_chunks(sample_chunks)
        
        # Search for division title
        result = index.search("Heating Ventilating Air Conditioning", "test_project", k=5)
        
        # Should find HVAC chunk based on division title
        found_hvac = False
        for hit in result.hits:
            if hit["chunk"]["metadata"]["division_code"] == "23":
                found_hvac = True
                break
        
        assert found_hvac, "Should find chunk based on division title"
    
    def test_construction_terminology_search(self, temp_index_dir, sample_chunks):
        """Test search with construction-specific terminology."""
        index = BM25Index(temp_index_dir)
        index.index_chunks(sample_chunks)
        
        # Search using abbreviations that should be expanded
        result = index.search("spec requirements", "test_project", k=5)
        
        # Should find chunks with "specifications" or "specification"
        assert len(result.hits) > 0
        
        # Verify query terms were processed
        assert len(result.query_terms) > 0
    
    def test_empty_search(self, temp_index_dir, sample_chunks):
        """Test search with no results."""
        index = BM25Index(temp_index_dir)
        index.index_chunks(sample_chunks)
        
        # Search for something that doesn't exist
        result = index.search("nonexistent term xyz", "test_project", k=5)
        
        assert len(result.hits) == 0
        assert result.total_found == 0
    
    def test_project_isolation(self, temp_index_dir, sample_chunks):
        """Test that projects are isolated in search."""
        index = BM25Index(temp_index_dir)
        
        # Index chunks for test project
        index.index_chunks(sample_chunks)
        
        # Search in different project should return no results
        result = index.search("HVAC", "different_project", k=5)
        assert len(result.hits) == 0
    
    def test_delete_project(self, temp_index_dir, sample_chunks):
        """Test project deletion."""
        index = BM25Index(temp_index_dir)
        index.index_chunks(sample_chunks)
        
        # Verify chunks are indexed
        result = index.search("HVAC", "test_project", k=5)
        assert len(result.hits) > 0
        
        # Delete project
        index.delete_project("test_project")
        
        # Verify chunks are gone
        result = index.search("HVAC", "test_project", k=5)
        assert len(result.hits) == 0
    
    def test_get_stats(self, temp_index_dir, sample_chunks):
        """Test index statistics."""
        index = BM25Index(temp_index_dir)
        index.index_chunks(sample_chunks)
        
        stats = index.get_stats()
        
        assert "document_count" in stats
        assert "field_names" in stats
        assert "index_dir" in stats
        assert stats["document_count"] == len(sample_chunks)
        assert isinstance(stats["field_names"], list)
        assert len(stats["field_names"]) > 0


class TestBM25Factory:
    """Test BM25 index factory function."""
    
    def test_create_bm25_index(self):
        """Test factory function for creating BM25 index."""
        # This will use the default config data directory
        index = create_bm25_index("test_project")
        
        assert index is not None
        assert isinstance(index, BM25Index)
        assert "test_project" in str(index.index_dir)
        assert "bm25_index" in str(index.index_dir)


if __name__ == "__main__":
    pytest.main([__file__])