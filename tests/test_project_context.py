"""
Tests for project context generation and management.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List

from src.services.project_context import (
    ProjectContextGenerator, 
    ProjectContextManager, 
    ProjectContextCache,
    QueryEnhancer
)
from src.models.types import Chunk, ChunkMetadata, ProjectContext


class TestProjectContextGenerator:
    """Test project context generation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ProjectContextGenerator()
    
    def create_test_chunk(self, text: str, content_type: str = "SpecSection", 
                         division_code: str = None, discipline: str = None) -> Chunk:
        """Create a test chunk with specified properties."""
        metadata = ChunkMetadata(
            project_id="test_project",
            doc_id="test_doc",
            doc_name="test.pdf",
            file_type="pdf",
            page_start=1,
            page_end=1,
            content_type=content_type,
            division_code=division_code,
            division_title=None,
            section_code=None,
            section_title=None,
            discipline=discipline,
            sheet_number=None,
            sheet_title=None,
            bbox_regions=[[0, 0, 100, 100]],
            low_conf=False
        )
        
        return Chunk(
            id="test_chunk",
            text=text,
            html=None,
            metadata=metadata,
            token_count=len(text.split()),
            text_hash="test_hash"
        )
    
    def test_detect_project_type_commercial(self):
        """Test commercial office building detection."""
        chunks = [
            self.create_test_chunk("This commercial office building project includes workspace areas"),
            self.create_test_chunk("Office building specifications for corporate headquarters")
        ]
        
        context = self.generator.generate_context(chunks, "Test Project")
        assert context["project_type"] == "Commercial Office Building"
    
    def test_detect_project_type_residential(self):
        """Test residential complex detection."""
        chunks = [
            self.create_test_chunk("Residential complex with apartment building units"),
            self.create_test_chunk("Multi-family housing development specifications")
        ]
        
        context = self.generator.generate_context(chunks, "Test Project")
        assert context["project_type"] == "Residential Complex"
    
    def test_identify_key_systems_from_text(self):
        """Test key systems identification from text content."""
        chunks = [
            self.create_test_chunk("HVAC system specifications for heating and cooling"),
            self.create_test_chunk("Electrical power distribution and lighting systems"),
            self.create_test_chunk("Plumbing and water supply requirements")
        ]
        
        context = self.generator.generate_context(chunks, "Test Project")
        
        assert "HVAC" in context["key_systems"]
        assert "Electrical" in context["key_systems"]
        assert "Plumbing" in context["key_systems"]
    
    def test_identify_key_systems_from_divisions(self):
        """Test key systems identification from division codes."""
        chunks = [
            self.create_test_chunk("Fire suppression system", division_code="21"),
            self.create_test_chunk("Plumbing fixtures", division_code="22"),
            self.create_test_chunk("HVAC equipment", division_code="23"),
            self.create_test_chunk("Electrical panels", division_code="26")
        ]
        
        context = self.generator.generate_context(chunks, "Test Project")
        
        assert "Fire Protection" in context["key_systems"]
        assert "Plumbing" in context["key_systems"]
        assert "HVAC" in context["key_systems"]
        assert "Electrical" in context["key_systems"]
    
    def test_detect_disciplines_from_metadata(self):
        """Test discipline detection from chunk metadata."""
        chunks = [
            self.create_test_chunk("Architectural plans", discipline="A"),
            self.create_test_chunk("Structural drawings", discipline="S"),
            self.create_test_chunk("Mechanical systems", discipline="M"),
            self.create_test_chunk("Electrical plans", discipline="E")
        ]
        
        context = self.generator.generate_context(chunks, "Test Project")
        
        assert "Architectural" in context["disciplines_involved"]
        assert "Structural" in context["disciplines_involved"]
        assert "Mechanical" in context["disciplines_involved"]
        assert "Electrical" in context["disciplines_involved"]
    
    def test_detect_disciplines_from_divisions(self):
        """Test discipline detection from division codes."""
        chunks = [
            self.create_test_chunk("Concrete work", division_code="03"),
            self.create_test_chunk("Fire protection", division_code="21"),
            self.create_test_chunk("Plumbing", division_code="22"),
            self.create_test_chunk("HVAC", division_code="23")
        ]
        
        context = self.generator.generate_context(chunks, "Test Project")
        
        assert "Structural" in context["disciplines_involved"]
        assert "Fire Protection" in context["disciplines_involved"]
        assert "Plumbing" in context["disciplines_involved"]
        assert "Mechanical" in context["disciplines_involved"]
    
    def test_extract_location(self):
        """Test location extraction from text."""
        chunks = [
            self.create_test_chunk("Project location: 123 Main Street, Anytown, CA"),
            self.create_test_chunk("Building specifications for downtown office")
        ]
        
        context = self.generator.generate_context(chunks, "Test Project")
        assert context["location"] == "123 Main Street, Anytown, CA"
    
    def test_generate_summary(self):
        """Test project summary generation."""
        chunks = [
            self.create_test_chunk("Commercial office building project", content_type="SpecSection"),
            self.create_test_chunk("HVAC system specifications", content_type="SpecSection", division_code="23"),
            self.create_test_chunk("Electrical plans", content_type="Drawing", discipline="E")
        ]
        
        context = self.generator.generate_context(chunks, "Test Project")
        
        summary = context["summary"]
        assert "commercial office building" in summary.lower()
        assert "specification sections" in summary.lower()
        assert "drawing" in summary.lower()


class TestProjectContextManager:
    """Test project context persistence and management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ProjectContextManager(self.temp_dir)
        
        self.test_context = ProjectContext(
            project_name="Test Project",
            description="A test construction project",
            project_type="Commercial Office Building",
            location="123 Test Street",
            key_systems=["HVAC", "Electrical", "Plumbing"],
            disciplines_involved=["Architectural", "Mechanical", "Electrical"],
            summary="This is a test project summary."
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_context(self):
        """Test saving and loading project context."""
        project_id = "test_project"
        
        # Save context
        self.manager.save_context(project_id, self.test_context)
        
        # Verify file exists
        assert self.manager.context_exists(project_id)
        
        # Load context
        loaded_context = self.manager.load_context(project_id)
        
        assert loaded_context is not None
        assert loaded_context["project_name"] == self.test_context["project_name"]
        assert loaded_context["project_type"] == self.test_context["project_type"]
        assert loaded_context["key_systems"] == self.test_context["key_systems"]
        assert loaded_context["disciplines_involved"] == self.test_context["disciplines_involved"]
    
    def test_update_context(self):
        """Test updating specific context fields."""
        project_id = "test_project"
        
        # Save initial context
        self.manager.save_context(project_id, self.test_context)
        
        # Update context
        updates = {
            "project_type": "Industrial Facility",
            "location": "456 Updated Street"
        }
        
        success = self.manager.update_context(project_id, updates)
        assert success
        
        # Load and verify updates
        updated_context = self.manager.load_context(project_id)
        assert updated_context["project_type"] == "Industrial Facility"
        assert updated_context["location"] == "456 Updated Street"
        assert updated_context["project_name"] == self.test_context["project_name"]  # Unchanged
    
    def test_delete_context(self):
        """Test deleting project context."""
        project_id = "test_project"
        
        # Save context
        self.manager.save_context(project_id, self.test_context)
        assert self.manager.context_exists(project_id)
        
        # Delete context
        success = self.manager.delete_context(project_id)
        assert success
        assert not self.manager.context_exists(project_id)
    
    def test_load_nonexistent_context(self):
        """Test loading non-existent context."""
        context = self.manager.load_context("nonexistent_project")
        assert context is None


class TestProjectContextCache:
    """Test project context caching functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = ProjectContextCache(max_size=3)
        
        self.test_context = ProjectContext(
            project_name="Test Project",
            description="A test construction project",
            project_type="Commercial Office Building",
            location=None,
            key_systems=["HVAC", "Electrical"],
            disciplines_involved=["Architectural", "Mechanical"],
            summary="Test summary."
        )
    
    def test_cache_put_and_get(self):
        """Test putting and getting from cache."""
        project_id = "test_project"
        
        # Put in cache
        self.cache.put(project_id, self.test_context)
        
        # Get from cache
        cached_context = self.cache.get(project_id)
        assert cached_context is not None
        assert cached_context["project_name"] == self.test_context["project_name"]
    
    def test_cache_eviction(self):
        """Test cache eviction when max size exceeded."""
        # Fill cache to max size
        for i in range(3):
            self.cache.put(f"project_{i}", self.test_context)
        
        # Add one more (should evict oldest)
        self.cache.put("project_3", self.test_context)
        
        # First project should be evicted
        assert self.cache.get("project_0") is None
        assert self.cache.get("project_3") is not None
    
    def test_cache_lru_behavior(self):
        """Test LRU (Least Recently Used) behavior."""
        # Fill cache
        for i in range(3):
            self.cache.put(f"project_{i}", self.test_context)
        
        # Access project_0 (makes it most recently used)
        self.cache.get("project_0")
        
        # Add new project (should evict project_1, not project_0)
        self.cache.put("project_3", self.test_context)
        
        assert self.cache.get("project_0") is not None  # Should still be there
        assert self.cache.get("project_1") is None      # Should be evicted
        assert self.cache.get("project_3") is not None  # Should be there


class TestQueryEnhancer:
    """Test query enhancement functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.enhancer = QueryEnhancer()
        
        self.test_context = ProjectContext(
            project_name="Test Project",
            description="A commercial office building project",
            project_type="Commercial Office Building",
            location=None,
            key_systems=["HVAC", "Electrical", "Plumbing"],
            disciplines_involved=["Architectural", "Mechanical", "Electrical"],
            summary="Test summary."
        )
    
    def test_enhance_query_basic(self):
        """Test basic query enhancement."""
        query = "What are the HVAC requirements?"
        enhanced = self.enhancer.enhance_query(query, self.test_context)
        
        assert "Commercial Office Building" in enhanced
        assert "HVAC" in enhanced
        assert "Mechanical" in enhanced
    
    def test_disambiguate_terms(self):
        """Test technical term disambiguation."""
        query = "Where is the electrical panel located?"
        disambiguated = self.enhancer.disambiguate_terms(query, self.test_context)
        
        # Should remain as "electrical panel" for commercial building
        assert "electrical panel" in disambiguated.lower()
    
    def test_add_domain_knowledge(self):
        """Test adding construction domain knowledge."""
        query = "Show me the HVAC specifications"
        enhanced = self.enhancer.add_domain_knowledge(query, self.test_context)
        
        assert "heating ventilation air conditioning" in enhanced.lower()
        assert "specification" in enhanced.lower()
    
    def test_find_relevant_systems(self):
        """Test finding relevant systems from query."""
        query = "heating and cooling requirements"
        relevant_systems = self.enhancer._find_relevant_systems(
            query.lower(), 
            self.test_context["key_systems"]
        )
        
        assert "HVAC" in relevant_systems
    
    def test_find_relevant_disciplines(self):
        """Test finding relevant disciplines from query."""
        query = "electrical wiring and power distribution"
        relevant_disciplines = self.enhancer._find_relevant_disciplines(
            query.lower(),
            self.test_context["disciplines_involved"]
        )
        
        assert "Electrical" in relevant_disciplines
    
    def test_expand_technical_terms(self):
        """Test technical term expansion."""
        query = "hvac system requirements"
        expansions = self.enhancer._expand_technical_terms(
            query.lower(),
            self.test_context["project_type"]
        )
        
        assert len(expansions) > 0
        assert any("heating" in exp for exp in expansions)


if __name__ == "__main__":
    pytest.main([__file__])