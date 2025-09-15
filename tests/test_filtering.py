"""
Tests for the metadata filtering system.
"""
import pytest
from typing import List, Dict, Any

from src.services.filtering import (
    MetadataFilter, FilterCriteria, ContentType, Discipline,
    MASTERFORMAT_DIVISIONS, create_metadata_filter
)
from src.models.types import ChunkMetadata


class TestMetadataFilter:
    """Test cases for MetadataFilter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.filter = create_metadata_filter()
        
        # Sample chunk metadata for testing
        self.sample_metadata: ChunkMetadata = {
            "project_id": "test_project",
            "doc_id": "doc_001",
            "doc_name": "Test Specification.pdf",
            "file_type": "pdf",
            "page_start": 5,
            "page_end": 5,
            "content_type": "SpecSection",
            "division_code": "03",
            "division_title": "Concrete",
            "section_code": "03 30 00",
            "section_title": "Cast-in-Place Concrete",
            "discipline": None,
            "sheet_number": None,
            "sheet_title": None,
            "bbox_regions": [[100, 100, 200, 200]],
            "low_conf": False
        }
        
        self.drawing_metadata: ChunkMetadata = {
            "project_id": "test_project",
            "doc_id": "doc_002",
            "doc_name": "A-101 Floor Plan.pdf",
            "file_type": "pdf",
            "page_start": 1,
            "page_end": 1,
            "content_type": "Drawing",
            "division_code": None,
            "division_title": None,
            "section_code": None,
            "section_title": None,
            "discipline": "A",
            "sheet_number": "A-101",
            "sheet_title": "First Floor Plan",
            "bbox_regions": [[0, 0, 1000, 800]],
            "low_conf": False
        }
    
    def test_validate_criteria_valid(self):
        """Test validation of valid filter criteria."""
        criteria = FilterCriteria(
            content_types=["SpecSection", "Drawing"],
            division_codes=["03", "05"],
            disciplines=["A", "S"],
            file_types=["pdf", "docx"]
        )
        
        errors = self.filter.validate_criteria(criteria)
        assert errors == []
    
    def test_validate_criteria_invalid_content_types(self):
        """Test validation with invalid content types."""
        criteria = FilterCriteria(
            content_types=["InvalidType", "SpecSection"]
        )
        
        errors = self.filter.validate_criteria(criteria)
        assert len(errors) == 1
        assert "Invalid content types" in errors[0]
        assert "InvalidType" in errors[0]
    
    def test_validate_criteria_invalid_disciplines(self):
        """Test validation with invalid disciplines."""
        criteria = FilterCriteria(
            disciplines=["X", "Y", "A"]
        )
        
        errors = self.filter.validate_criteria(criteria)
        assert len(errors) == 1
        assert "Invalid disciplines" in errors[0]
        assert "X" in errors[0] and "Y" in errors[0]
    
    def test_validate_criteria_invalid_division_codes(self):
        """Test validation with invalid division codes."""
        criteria = FilterCriteria(
            division_codes=["99", "100", "03"]
        )
        
        errors = self.filter.validate_criteria(criteria)
        assert len(errors) == 1
        assert "Invalid division codes" in errors[0]
    
    def test_validate_criteria_page_range_errors(self):
        """Test validation with invalid page ranges."""
        criteria = FilterCriteria(
            page_start_min=0,  # Invalid: must be >= 1
            page_start_max=5,
            page_end_min=10,
            page_end_max=5  # Invalid: min > max
        )
        
        errors = self.filter.validate_criteria(criteria)
        assert len(errors) == 2
        assert any("page_start_min must be >= 1" in error for error in errors)
        assert any("page_end_min cannot be greater than page_end_max" in error for error in errors)
    
    def test_validate_criteria_conflicting_confidence(self):
        """Test validation with conflicting confidence filters."""
        criteria = FilterCriteria(
            low_conf_only=True,
            high_conf_only=True
        )
        
        errors = self.filter.validate_criteria(criteria)
        assert len(errors) == 1
        assert "Cannot filter for both low_conf_only and high_conf_only" in errors[0]
    
    def test_apply_filters_content_type_match(self):
        """Test applying content type filters."""
        criteria = FilterCriteria(
            project_id="test_project",
            content_types=["SpecSection"]
        )
        
        assert self.filter.apply_filters(self.sample_metadata, criteria) == True
        assert self.filter.apply_filters(self.drawing_metadata, criteria) == False
    
    def test_apply_filters_division_code_match(self):
        """Test applying division code filters."""
        criteria = FilterCriteria(
            project_id="test_project",
            division_codes=["03", "05"]
        )
        
        assert self.filter.apply_filters(self.sample_metadata, criteria) == True
        assert self.filter.apply_filters(self.drawing_metadata, criteria) == False
    
    def test_apply_filters_discipline_match(self):
        """Test applying discipline filters."""
        criteria = FilterCriteria(
            project_id="test_project",
            disciplines=["A", "S"]
        )
        
        assert self.filter.apply_filters(self.sample_metadata, criteria) == False
        assert self.filter.apply_filters(self.drawing_metadata, criteria) == True
    
    def test_apply_filters_page_range(self):
        """Test applying page range filters."""
        criteria = FilterCriteria(
            project_id="test_project",
            page_start_min=3,
            page_start_max=10
        )
        
        assert self.filter.apply_filters(self.sample_metadata, criteria) == True
        
        criteria.page_start_min = 10
        assert self.filter.apply_filters(self.sample_metadata, criteria) == False
    
    def test_apply_filters_confidence(self):
        """Test applying confidence filters."""
        criteria = FilterCriteria(
            project_id="test_project",
            low_conf_only=False
        )
        
        assert self.filter.apply_filters(self.sample_metadata, criteria) == True
        
        criteria.low_conf_only = True
        assert self.filter.apply_filters(self.sample_metadata, criteria) == False
    
    def test_apply_filters_doc_name_partial_match(self):
        """Test applying document name filters with partial matching."""
        criteria = FilterCriteria(
            project_id="test_project",
            doc_names=["Specification", "Drawing"]
        )
        
        assert self.filter.apply_filters(self.sample_metadata, criteria) == True
        assert self.filter.apply_filters(self.drawing_metadata, criteria) == False
        
        criteria.doc_names = ["Floor Plan"]
        assert self.filter.apply_filters(self.drawing_metadata, criteria) == True
    
    def test_apply_filters_wrong_project(self):
        """Test that wrong project ID fails all filters."""
        criteria = FilterCriteria(
            project_id="wrong_project",
            content_types=["SpecSection"]
        )
        
        assert self.filter.apply_filters(self.sample_metadata, criteria) == False
    
    def test_build_vector_store_filter(self):
        """Test building vector store filter dictionary."""
        criteria = FilterCriteria(
            project_id="test_project",
            content_types=["SpecSection", "Drawing"],
            division_codes=["03"],
            disciplines=["A"],
            low_conf_only=False
        )
        
        where = self.filter.build_vector_store_filter(criteria)
        
        expected = {
            "project_id": "test_project",
            "content_type": {"$in": ["SpecSection", "Drawing"]},
            "division_code": "03",
            "discipline": "A",
            "low_conf": "false"
        }
        
        assert where == expected
    
    def test_build_vector_store_filter_single_values(self):
        """Test building vector store filter with single values."""
        criteria = FilterCriteria(
            project_id="test_project",
            content_types=["SpecSection"],
            division_codes=["03"]
        )
        
        where = self.filter.build_vector_store_filter(criteria)
        
        # Single values should not use $in operator
        assert where["content_type"] == "SpecSection"
        assert where["division_code"] == "03"
    
    def test_build_bm25_filter(self):
        """Test building BM25 filter dictionary."""
        criteria = FilterCriteria(
            content_types=["SpecSection", "Drawing"],
            division_codes=["03"],
            disciplines=["A"],
            file_types=["pdf"],
            low_conf_only=False
        )
        
        filters = self.filter.build_bm25_filter(criteria)
        
        expected = {
            "content_types": ["SpecSection", "Drawing"],
            "division_codes": ["03"],
            "disciplines": ["A"],
            "file_types": ["pdf"],
            "low_conf_only": False
        }
        
        assert filters == expected
    
    def test_get_division_title(self):
        """Test getting division title by code."""
        assert self.filter.get_division_title("03") == "Concrete"
        assert self.filter.get_division_title("26") == "Electrical"
        assert self.filter.get_division_title("99") is None
    
    def test_get_all_divisions(self):
        """Test getting all divisions."""
        divisions = self.filter.get_all_divisions()
        assert isinstance(divisions, dict)
        assert len(divisions) == len(MASTERFORMAT_DIVISIONS)
        assert divisions["03"] == "Concrete"
        assert divisions["26"] == "Electrical"
    
    def test_suggest_filters_content_types(self):
        """Test filter suggestions based on query content types."""
        query = "show me the structural drawings and specifications"
        criteria = self.filter.suggest_filters(query)
        
        assert "Drawing" in criteria.content_types
        assert "SpecSection" in criteria.content_types
        assert "S" in criteria.disciplines
    
    def test_suggest_filters_divisions(self):
        """Test filter suggestions based on query divisions."""
        query = "concrete and electrical requirements"
        criteria = self.filter.suggest_filters(query)
        
        assert "03" in criteria.division_codes  # Concrete
        assert "26" in criteria.division_codes  # Electrical
    
    def test_suggest_filters_disciplines(self):
        """Test filter suggestions based on query disciplines."""
        query = "HVAC and plumbing systems"
        criteria = self.filter.suggest_filters(query)
        
        assert "M" in criteria.disciplines  # Mechanical (HVAC)
        assert "P" in criteria.disciplines  # Plumbing
    
    def test_suggest_filters_no_matches(self):
        """Test filter suggestions with no keyword matches."""
        query = "random query with no construction terms"
        criteria = self.filter.suggest_filters(query)
        
        assert criteria.content_types is None
        assert criteria.disciplines is None
        assert criteria.division_codes is None
    
    def test_get_available_values(self):
        """Test getting available values for filter fields."""
        # Test known fields
        content_types = self.filter.get_available_values("test_project", "content_type")
        assert "SpecSection" in content_types
        assert "Drawing" in content_types
        
        disciplines = self.filter.get_available_values("test_project", "discipline")
        assert "A" in disciplines
        assert "S" in disciplines
        
        # Test unknown field
        unknown = self.filter.get_available_values("test_project", "unknown_field")
        assert unknown == []


if __name__ == "__main__":
    pytest.main([__file__])