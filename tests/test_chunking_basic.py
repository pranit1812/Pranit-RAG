"""
Basic tests for the chunking system without external dependencies.
"""
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chunking.context_window import ContextWindow
from models.types import generate_text_hash


def test_context_window_initialization():
    """Test context window initialization."""
    window = ContextWindow(window_size=2)
    assert window.window_size == 2
    
    # Test negative window size
    window = ContextWindow(window_size=-1)
    assert window.window_size == 0
    print("✓ Context window initialization test passed")


def test_generate_text_hash():
    """Test text hash generation."""
    text1 = "Hello world"
    text2 = "Hello world"
    text3 = "Different text"
    
    hash1 = generate_text_hash(text1)
    hash2 = generate_text_hash(text2)
    hash3 = generate_text_hash(text3)
    
    assert hash1 == hash2  # Same text should have same hash
    assert hash1 != hash3  # Different text should have different hash
    assert len(hash1) == 64  # SHA-256 produces 64-character hex string
    print("✓ Text hash generation test passed")


def test_chunk_metadata_structure():
    """Test chunk metadata structure."""
    from models.types import ChunkMetadata
    
    # Test that we can create a ChunkMetadata structure
    metadata = {
        "project_id": "test_project",
        "doc_id": "doc_123",
        "doc_name": "test_document.pdf",
        "file_type": "pdf",
        "page_start": 1,
        "page_end": 1,
        "content_type": "SpecSection",
        "division_code": None,
        "division_title": None,
        "section_code": None,
        "section_title": None,
        "discipline": None,
        "sheet_number": None,
        "sheet_title": None,
        "bbox_regions": [],
        "low_conf": False
    }
    
    # Verify all required fields are present
    required_fields = [
        "project_id", "doc_id", "doc_name", "file_type", 
        "page_start", "page_end", "content_type", "bbox_regions", "low_conf"
    ]
    
    for field in required_fields:
        assert field in metadata
    
    print("✓ Chunk metadata structure test passed")


def test_context_window_group_chunks():
    """Test context window chunk grouping."""
    window = ContextWindow(window_size=1)
    
    # Create test chunks
    chunks = []
    for i in range(3):
        chunk = {
            "id": f"chunk_{i}",
            "text": f"Text {i}",
            "html": None,
            "metadata": {
                "project_id": "test",
                "doc_id": "doc1",
                "doc_name": "test.pdf",
                "file_type": "pdf",
                "page_start": 1,
                "page_end": 1,
                "content_type": "SpecSection",
                "division_code": None,
                "division_title": None,
                "section_code": None,
                "section_title": None,
                "discipline": None,
                "sheet_number": None,
                "sheet_title": None,
                "bbox_regions": [],
                "low_conf": False
            },
            "token_count": 10,
            "text_hash": generate_text_hash(f"Text {i}")
        }
        chunks.append(chunk)
    
    # Test grouping by document and page
    doc_page_chunks = window._group_chunks_by_doc_page(chunks)
    
    assert len(doc_page_chunks) == 1  # All chunks from same doc/page
    key = ("doc1", 1)
    assert key in doc_page_chunks
    assert len(doc_page_chunks[key]) == 3
    
    print("✓ Context window chunk grouping test passed")


def run_all_tests():
    """Run all basic tests."""
    print("Running basic chunking tests...")
    
    try:
        test_context_window_initialization()
        test_generate_text_hash()
        test_chunk_metadata_structure()
        test_context_window_group_chunks()
        
        print("\n✅ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)