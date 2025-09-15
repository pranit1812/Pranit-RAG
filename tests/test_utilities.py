"""
Tests for utility functions and base classes.
"""
import pytest
from src.utils.hashing import generate_text_hash, generate_content_hash, verify_text_hash
from src.utils.bbox import (
    normalize_bbox, denormalize_bbox, bbox_area, bbox_intersection, 
    bbox_union, bbox_iou, bbox_contains_point, bbox_center, validate_bbox
)
from src.utils.io_utils import safe_filename, ensure_directory, get_file_extension
from src.models.types import (
    Span, Block, PageParse, ChunkMetadata, Chunk,
    validate_span, validate_block, validate_page_parse, validate_chunk
)
from src.models.validation import DataValidator, ValidationError
from src.extractors.base import BaseExtractor, ExtractorError
import tempfile
import shutil
from pathlib import Path


class TestHashing:
    """Test hashing utilities."""
    
    def test_generate_text_hash(self):
        """Test text hash generation."""
        text = "Hello, world!"
        hash1 = generate_text_hash(text)
        hash2 = generate_text_hash(text)
        
        assert hash1 == hash2  # Same text should produce same hash
        assert len(hash1) == 64  # SHA-256 produces 64-character hex string
        
        # Different text should produce different hash
        different_hash = generate_text_hash("Different text")
        assert hash1 != different_hash
    
    def test_verify_text_hash(self):
        """Test text hash verification."""
        text = "Test content"
        correct_hash = generate_text_hash(text)
        
        assert verify_text_hash(text, correct_hash) is True
        assert verify_text_hash(text, "wrong_hash") is False


class TestBbox:
    """Test bounding box utilities."""
    
    def test_validate_bbox(self):
        """Test bbox validation."""
        # Valid bbox
        assert validate_bbox([0, 0, 100, 100]) is True
        assert validate_bbox([10.5, 20.5, 30.5, 40.5]) is True
        
        # Invalid bboxes
        assert validate_bbox([100, 100, 0, 0]) is False  # x1 < x0
        assert validate_bbox([0, 100, 100, 0]) is False  # y1 < y0
        assert validate_bbox([0, 0, 100]) is False  # Wrong length
        assert validate_bbox(["a", "b", "c", "d"]) is False  # Non-numeric
    
    def test_bbox_area(self):
        """Test bbox area calculation."""
        assert bbox_area([0, 0, 10, 10]) == 100
        assert bbox_area([5, 5, 15, 25]) == 200
        assert bbox_area([0, 0, 0, 0]) == 0
    
    def test_bbox_intersection(self):
        """Test bbox intersection."""
        bbox1 = [0, 0, 10, 10]
        bbox2 = [5, 5, 15, 15]
        
        intersection = bbox_intersection(bbox1, bbox2)
        assert intersection == [5, 5, 10, 10]
        
        # No intersection
        bbox3 = [20, 20, 30, 30]
        assert bbox_intersection(bbox1, bbox3) is None
    
    def test_bbox_center(self):
        """Test bbox center calculation."""
        center = bbox_center([0, 0, 10, 10])
        assert center == (5.0, 5.0)
        
        center = bbox_center([10, 20, 30, 40])
        assert center == (20.0, 30.0)


class TestIOUtils:
    """Test I/O utilities."""
    
    def test_safe_filename(self):
        """Test safe filename generation."""
        assert safe_filename("normal_file.txt") == "normal_file.txt"
        assert safe_filename("file<>with|bad*chars.pdf") == "file__with_bad_chars.pdf"
        assert safe_filename("") == "unnamed_file"
        assert safe_filename("   ") == "unnamed_file"
    
    def test_get_file_extension(self):
        """Test file extension extraction."""
        assert get_file_extension("file.pdf") == "pdf"
        assert get_file_extension("file.PDF") == "pdf"
        assert get_file_extension("path/to/file.docx") == "docx"
        assert get_file_extension("no_extension") == ""
    
    def test_ensure_directory(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "new" / "nested" / "directory"
            
            result = ensure_directory(test_path)
            assert result.exists()
            assert result.is_dir()


class TestTypeValidation:
    """Test type validation functions."""
    
    def test_validate_span(self):
        """Test Span validation."""
        valid_span: Span = {
            "text": "Sample text",
            "bbox": [0, 0, 100, 20],
            "rot": 0.0,
            "conf": 0.95
        }
        assert validate_span(valid_span) is True
        
        # Invalid span - missing text
        invalid_span = {
            "bbox": [0, 0, 100, 20],
            "rot": 0.0,
            "conf": 0.95
        }
        assert validate_span(invalid_span) is False
    
    def test_validate_block(self):
        """Test Block validation."""
        valid_block: Block = {
            "type": "paragraph",
            "text": "Sample paragraph text",
            "html": None,
            "bbox": [0, 0, 200, 50],
            "spans": [{
                "text": "Sample paragraph text",
                "bbox": [0, 0, 200, 50],
                "rot": 0.0,
                "conf": 1.0
            }],
            "meta": {}
        }
        assert validate_block(valid_block) is True
        
        # Invalid block - wrong type
        invalid_block = valid_block.copy()
        invalid_block["type"] = "invalid_type"
        assert validate_block(invalid_block) is False


class MockExtractor(BaseExtractor):
    """Mock extractor for testing."""
    
    def supports(self, file_path) -> bool:
        return str(file_path).endswith('.txt')
    
    def parse_page(self, file_path, page_no: int):
        if page_no != 0:
            raise ExtractorError("Only page 0 supported", self.get_extractor_name())
        
        return {
            "page_no": 0,
            "width": 612,
            "height": 792,
            "blocks": [],
            "artifacts_removed": []
        }


class TestBaseExtractor:
    """Test base extractor functionality."""
    
    def test_extractor_interface(self):
        """Test extractor interface methods."""
        extractor = MockExtractor()
        
        assert extractor.supports("test.txt") is True
        assert extractor.supports("test.pdf") is False
        assert extractor.get_page_count("test.txt") == 1
        assert extractor.get_extractor_name() == "MockExtractor"
    
    def test_parse_page(self):
        """Test page parsing."""
        extractor = MockExtractor()
        
        page = extractor.parse_page("test.txt", 0)
        assert page["page_no"] == 0
        assert page["width"] == 612
        assert page["height"] == 792
        
        # Test error handling
        with pytest.raises(ExtractorError):
            extractor.parse_page("test.txt", 1)


class TestDataValidator:
    """Test data validator functionality."""
    
    def test_validation_error(self):
        """Test ValidationError creation."""
        error = ValidationError("Test error", "test_field", "TestType")
        assert "test_field" in str(error)
        assert "TestType" in str(error)
    
    def test_strict_validation(self):
        """Test strict validation methods."""
        valid_span: Span = {
            "text": "Test",
            "bbox": [0, 0, 100, 20],
            "rot": 0.0,
            "conf": 1.0
        }
        
        # Should not raise exception
        DataValidator.validate_span_strict(valid_span)
        
        # Should raise exception
        invalid_span = {"text": "Test"}  # Missing required fields
        with pytest.raises(ValidationError):
            DataValidator.validate_span_strict(invalid_span)


if __name__ == "__main__":
    pytest.main([__file__])